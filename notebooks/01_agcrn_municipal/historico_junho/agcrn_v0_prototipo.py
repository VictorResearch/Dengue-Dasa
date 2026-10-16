import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import shap
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. SETUP E DICIONÁRIO DE VARIÁVEIS
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"?? A executar no hardware: {device.type.upper()}")

VARS = {
    'municipio':          'geocode',
    'semana_epi':         'epiweek',
    'casos_dengue':       'cases',
    'casos_chik':         'chikungunya',
    'populacao':          'population',
    'temp_med':           'temp_med',
    'precip_tot':         'precip_tot',
    'amplitude_termica':  'thermal_range',
    'pressao_atm':        'pressure_med',
    'dias_chuva':         'rainy_days',
    'umidade':            'rel_humid_med',
    'enso':               'enso',
    'iod':                'iod',
    'pdo':                'pdo',
    'afya_acessos':       'access_count',
    'bioma':              'biome',
    'koppen':             'koppen',
    'data':               'date',
}

# ==============================================================================
# 1. PROCESSOS GAUSSIANOS ESPARSOS (FÍSICA DA EPIDEMIA)
# ==============================================================================
class SparseGaussianProcess(nn.Module):
    def __init__(self, num_inducing: int = 30):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.randn(num_inducing, 1))
        self.inducing_points = nn.Parameter(
            torch.linspace(0, 1, num_inducing).view(-1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x, self.inducing_points)
        K = torch.exp(-self.gamma * dist ** 2)
        return K @ self.alpha


def aplicar_sgp_python(df_group: pd.DataFrame, epochs: int = 100) -> pd.DataFrame:
    """
    Ajusta um Processo Gaussiano Esparso sobre a incidência de um município,
    extraindo casos suavizados, velocidade (1ª derivada) e aceleração (2ª derivada).
    """
    df_group = df_group.sort_values(VARS['semana_epi']).copy()
    casos = df_group['incidencia'].fillna(0).values
    tempos = np.arange(1, len(df_group) + 1)
    df_group['time_index'] = tempos

    t_min, t_max = tempos.min(), tempos.max()
    dt_scale = t_max - t_min + 1e-5

    # x precisa de requires_grad=True para calcular as derivadas
    x = torch.tensor(
        (tempos - t_min) / dt_scale, dtype=torch.float32
    ).view(-1, 1).requires_grad_(True)
    y = torch.tensor(casos, dtype=torch.float32).view(-1, 1)

    n_inducing = min(30, len(tempos))
    model = SparseGaussianProcess(num_inducing=n_inducing)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y) + 0.01 * model.alpha.norm()
        loss.backward()
        optimizer.step()

    # --- Derivadas via autograd ---
    # grad() retorna uma tupla; [0] extrai o tensor de gradiente
    pred_final = model(x)

    # 1ª derivada (velocidade)
    d1_tuple = torch.autograd.grad(
        outputs=pred_final.sum(), inputs=x, create_graph=True
    )
    d1 = d1_tuple[0] / dt_scale

    # 2ª derivada (aceleração)
    d2_tuple = torch.autograd.grad(
        outputs=d1.sum(), inputs=x, create_graph=False
    )
    d2 = d2_tuple[0] / (dt_scale ** 2)

    df_group['casos_suavizados'] = pred_final.detach().cpu().numpy().flatten()
    df_group['velocidade']       = d1.detach().cpu().numpy().flatten()
    df_group['aceleracao']       = d2.detach().cpu().numpy().flatten()
    return df_group


def extrair_fisica_vetorizada(
    df: pd.DataFrame, janela: int = 7, ordem: int = 3
) -> pd.DataFrame:
    """
    Versão vetorizada do SGP para o loop de forecasting da Seção 5.
    Aplica Savitzky-Golay para estimar velocidade e aceleração de forma causal.
    """
    df = df.sort_values([VARS['municipio'], VARS['semana_epi']]).copy()

    # Garante time_index por município
    df['time_index'] = df.groupby(VARS['municipio']).cumcount() + 1

    def _fisica_grupo(grp: pd.DataFrame) -> pd.DataFrame:
        grp = grp.copy()
        y = grp['incidencia'].fillna(0).values
        n = len(y)
        win = min(janela, n if n % 2 == 1 else n - 1)
        win = max(win, ordem + 2 if (ordem + 2) % 2 == 1 else ordem + 3)

        if n < win:
            grp['casos_suavizados'] = y
            grp['velocidade']       = 0.0
            grp['aceleracao']       = 0.0
        else:
            grp['casos_suavizados'] = savgol_filter(y, win, ordem, deriv=0)
            grp['velocidade']       = savgol_filter(y, win, ordem, deriv=1)
            grp['aceleracao']       = savgol_filter(y, win, ordem, deriv=2)
        return grp

    return df.groupby(VARS['municipio'], group_keys=False).apply(_fisica_grupo)


# ==============================================================================
# 2. FEATURE ENGINEERING & SHAP
def engenharia_de_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera features biológicas, climáticas, sazonais e infodemiológicas.
    """
    df = df.sort_values(by=[VARS['municipio'], VARS['semana_epi']]).copy()

    # Incidência e Imunidade de Rebanho
    df['incidencia'] = (df[VARS['casos_dengue']] / df[VARS['populacao']]) * 100_000
    df['imunidade_cumulativa'] = df.groupby(VARS['municipio'])['incidencia'].cumsum()

    # Codificação Cíclica Sazonal
    semana_ano = pd.to_datetime(df[VARS['data']]).dt.isocalendar().week
    df['seno_semana']    = np.sin(2 * np.pi * semana_ano / 52).astype(float)
    df['cosseno_semana'] = np.cos(2 * np.pi * semana_ano / 52).astype(float)

    # Lags Biológicos (3 a 5 semanas) e Acúmulos de Chuva
    vars_clima = [
        VARS['temp_med'], VARS['precip_tot'], VARS['amplitude_termica'],
        VARS['pressao_atm'], VARS['dias_chuva'], VARS['umidade'],
    ]
    for var in vars_clima:
        if var in df.columns:
            for lag in range(3, 6):   # CORRIGIDO: range(3,6) = [3, 4, 5]
                df[f'{var}_lag{lag}'] = df.groupby(VARS['municipio'])[var].shift(lag)

    if VARS['precip_tot'] in df.columns:
        df['chuva_acumulada_4w'] = df.groupby(VARS['municipio'])[VARS['precip_tot']].transform(
            lambda x: x.rolling(4, min_periods=1).sum()
        )

    # Lags Macroclimáticos (6 a 9 semanas / ~2 meses)
    for var in [VARS['enso'], VARS['iod'], VARS['pdo']]:
        if var in df.columns:
            for lag in range(6, 10):  # CORRIGIDO: range(6,10) = [6, 7, 8, 9]
                df[f'{var}_lag{lag}'] = df.groupby(VARS['municipio'])[var].shift(lag)

    # Infodemiologia Afya
    if VARS['afya_acessos'] in df.columns:
        df['afya_variacao_pct'] = (
            df.groupby(VARS['municipio'])[VARS['afya_acessos']]
            .pct_change()
            .fillna(0)
        )
        df.replace([np.inf, -np.inf], 0, inplace=True)

    # Target Encoding (Biomas e Köppen)
    for col in [VARS['bioma'], VARS['koppen']]:
        if col in df.columns:
            df[f'{col}_encoded'] = df.groupby(col)['incidencia'].transform('mean')

    return df.dropna().reset_index(drop=True)


def selecao_shap(
    df: pd.DataFrame,
    col_alvo: str = 'incidencia',
    num_features: int = 15,
) -> list:
    """
    Usa XGBoost + SHAP para selecionar as features mais importantes.
    """
    print("?? A executar filtragem de ruído com SHAP...")
    cols_ignorar = [
        VARS['municipio'], VARS['semana_epi'], VARS['data'],
        VARS['casos_dengue'], VARS['casos_chik'],
        col_alvo, VARS['bioma'], VARS['koppen'],
        'node_id', 'time_index',
    ]

    X = df.drop(columns=[c for c in cols_ignorar if c in df.columns]).select_dtypes(include=[np.number])
    y = df[col_alvo]

    xgb = XGBRegressor(n_estimators=100, max_depth=5, tree_method='hist', n_jobs=-1)
    xgb.fit(X, y)

    shap_values = shap.TreeExplainer(xgb).shap_values(X)
    importancias = np.abs(shap_values).mean(axis=0)
    ranking = (
        pd.DataFrame({'Feature': X.columns, 'SHAP_Importancia': importancias})
        .sort_values('SHAP_Importancia', ascending=False)
    )

    features_selecionadas = ranking.head(num_features)['Feature'].tolist()
    print(f"? Top {num_features} features retidas: {features_selecionadas}")
    return features_selecionadas  # CORRIGIDO: código morto duplicado removido


# ==============================================================================
# 3. ARQUITETURA AGCRN
# ==============================================================================
class PreditorPandemicoAGCRN(nn.Module):
    def __init__(self, num_nodes: int, num_features: int, embedding_dim: int = 10):
        super().__init__()
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embedding_dim), requires_grad=True
        )
        self.recurrent = AGCRN(
            number_of_nodes=num_nodes,
            in_channels=num_features,
            out_channels=32,
        )
        self.linear = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, h=None):
        h = self.recurrent(x, self.node_embeddings, h)
        return self.linear(F.relu(h)), h


# ==============================================================================
# 3. EXECUÇÃO DA ENGENHARIA DE FEATURES, SHAP E SGP
# ==============================================================================
print("A aplicar Engenharia de Features Biológicas e Climáticas...")
train1 = pd.read_parquet("train1_dengue.parquet")
test1  = pd.read_parquet("test1_dengue.parquet")

df_treino_engenharia = engenharia_de_features(train1)

# Filtragem de ruído por SHAP
features_selecionadas = selecao_shap(df_treino_engenharia, col_alvo='incidencia', num_features=15)

# Extração da Física da Epidemia (SGP) por município
print("A extrair a Física da Epidemia (SGP)...")
tqdm.pandas(desc="A extrair SGP por município")
features_treino = df_treino_engenharia.groupby(
    VARS['municipio'], group_keys=False
).progress_apply(aplicar_sgp_python)

# ==============================================================================
# 3.5. PREPARAÇÃO DAS VARIÁVEIS AVANÇADAS (PÓS-SHAP) E MAPA
# ==============================================================================
# CORRIGIDO: usa features_treino (com SGP) em vez de df_treino_engenharia
dados_treino_fisica = features_treino.copy()

# Mapa de nós
mapa_simples = (
    dados_treino_fisica[[VARS['municipio']]]
    .drop_duplicates()
    .sort_values(VARS['municipio'])
    .reset_index(drop=True)
)
mapa_simples['node_id'] = mapa_simples.index

# Covariáveis finais: SHAP + variáveis vitais garantidas
covariaveis_alvo = list(set(features_selecionadas + [
    VARS['casos_chik'],
    'afya_variacao_pct',
    'imunidade_cumulativa',
    'seno_semana',
    'cosseno_semana',
]))

# ==============================================================================
# 4. TREINAMENTO DA AGCRN (COM MULTI-FEATURES)
# ==============================================================================
print("Preparando tensores de alta dimensionalidade...")

df_treino = (
    dados_treino_fisica
    .merge(mapa_simples, on=VARS['municipio'])
    .sort_values(by=['time_index', 'node_id'])
    .copy()
)
tempos      = df_treino['time_index'].unique()
num_nodes   = len(mapa_simples)

# Features: física SGP + covariáveis de alta dimensão
cols         = ['casos_suavizados', 'velocidade', 'aceleracao'] + covariaveis_alvo
num_features = len(cols)

# Normalização
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

df_treino[cols]          = scaler_X.fit_transform(df_treino[cols])
df_treino['target_scaled'] = scaler_Y.fit_transform(df_treino[['incidencia']])

# Tensores 3-D [Tempo, Nós, Features]
X_tensor = torch.tensor(
    df_treino[cols].values.reshape(len(tempos), num_nodes, num_features),
    dtype=torch.float32,
).to(device)
Y_tensor = torch.tensor(
    df_treino['target_scaled'].values.reshape(len(tempos), num_nodes),
    dtype=torch.float32,
).to(device)

modelo_agcrn = PreditorPandemicoAGCRN(num_nodes=num_nodes, num_features=num_features).to(device)
otimizador   = torch.optim.Adam(modelo_agcrn.parameters(), lr=0.01)

modelo_agcrn.train()
epochs_agcrn = 100
barra_treino = tqdm(range(epochs_agcrn), desc='Treinando AGCRN', unit='epoch', file=sys.stdout)

for epoch in barra_treino:
    h          = None
    epoch_loss = 0.0

    for t in range(len(tempos) - 1):
        otimizador.zero_grad()
        previsao, h = modelo_agcrn(X_tensor[t], h)
        h           = h.detach()
        loss        = F.mse_loss(previsao, Y_tensor[t + 1].view(-1, 1))
        loss.backward()
        otimizador.step()
        epoch_loss += loss.item()

    loss_media = epoch_loss / (len(tempos) - 1)
    barra_treino.set_postfix({'Loss': f'{loss_media:.4f}'})

modelo_agcrn.eval()
print(f"Treinamento finalizado com {num_features} features de alta performance!")

# ==============================================================================
# 5. FORECASTING PROCEDURAL COM JANELA DESLIZANTE (Simulação em Tempo Real)
# ==============================================================================
semanas_futuras_teste = sorted(test1[VARS['semana_epi']].unique())

dados_loop   = train1.copy()
h_teste      = None
tamanho_memoria = 15  # semanas de buffer para cálculo da cinemática

print("\nIniciando Forecasting Espaço-Temporal Contínuo...")
pbar = tqdm(semanas_futuras_teste, desc="Forecasting Procedural", unit="semana", file=sys.stdout)

for semana_futura in pbar:
    # 1. JANELA DESLIZANTE
    semana_maxima_historico = dados_loop[VARS['semana_epi']].max()
    semana_corte = semana_maxima_historico - tamanho_memoria
    dados_recentes = dados_loop[dados_loop[VARS['semana_epi']] > semana_corte].copy()

    # 2. FÍSICA CAUSAL (usa a função vetorizada definida na Seção 1)
    dados_recentes_eng   = engenharia_de_features(dados_recentes)
    dados_fisica_atual   = extrair_fisica_vetorizada(dados_recentes_eng, janela=7, ordem=3)
    dados_fisica_atual   = dados_fisica_atual.merge(mapa_simples, on=VARS['municipio'], how='left')

    # 3. ESTADO ATUAL (última semana registrada = Semana t)
    estado_atual = (
        dados_fisica_atual
        .sort_values(['time_index', 'node_id'])
        .groupby(VARS['municipio'])
        .tail(1)
        .sort_values('node_id')
    )

    # Garante que todas as colunas existam; preenche ausentes com 0
    for c in cols:
        if c not in estado_atual.columns:
            estado_atual[c] = 0.0

    features_matriz_real   = estado_atual[cols].values

    # 4. NORMALIZAÇÃO E FORWARD PASS
    features_matriz_scaled = scaler_X.transform(features_matriz_real)
    x_tensor = torch.tensor(features_matriz_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        previsao_tensor_scaled, h_teste = modelo_agcrn(x_tensor, h_teste)
        previsao_numpy_scaled = previsao_tensor_scaled.cpu().numpy().flatten()

    previsao_numpy_real = scaler_Y.inverse_transform(
        previsao_numpy_scaled.reshape(-1, 1)
    ).flatten()
    previsao_numpy_real = np.maximum(previsao_numpy_real, 0)

    # 5. ATUALIZAÇÃO DA LINHA DO TEMPO
    covariaveis_teste = test1[test1[VARS['semana_epi']] == semana_futura][
        [VARS['municipio']] + covariaveis_alvo
    ]

    novos_dados = pd.DataFrame({
        VARS['municipio']:    estado_atual[VARS['municipio']].values,
        VARS['semana_epi']:   semana_futura,
        VARS['casos_dengue']: previsao_numpy_real,  # CORRIGIDO: nome correto da coluna
    })
    novos_dados = (
        novos_dados
        .merge(mapa_simples, on=VARS['municipio'])
        .merge(covariaveis_teste, on=VARS['municipio'], how='left')
    )

    dados_loop = pd.concat([dados_loop, novos_dados], ignore_index=True)

print("\nProcedimento de Forecasting Espaço-Temporal finalizado.")

# ==============================================================================
# 6. GERAÇÃO DE PERCENTIS E VALIDAÇÃO DE REGRAS OPERACIONAIS
# ==============================================================================
print("\nFormatando dados e aplicando filtros de validação estritos...")

# 6.1 Isolamento e nomeação
df_export = dados_loop[dados_loop[VARS['semana_epi']].isin(semanas_futuras_teste)].copy()
df_export = df_export.rename(columns={VARS['casos_dengue']: 'pred'})  # CORRIGIDO: nome real

# 6.2 Conversão para domingo ISO
df_export['date_parsed'] = pd.to_datetime(
    df_export[VARS['semana_epi']].astype(str) + '-0', format='%G%V-%w'
)
df_export['date'] = df_export['date_parsed'].dt.strftime('%Y-%m-%d')

# 6.3 Grade perfeita geocode × data (sem lacunas)
todas_datas    = pd.date_range(
    start=df_export['date_parsed'].min(),
    end=df_export['date_parsed'].max(),
    freq='W-SUN',
)
todos_geocodes = df_export[VARS['municipio']].unique()

grade_perfeita = (
    pd.MultiIndex.from_product(
        [todos_geocodes, todas_datas],
        names=[VARS['municipio'], 'date_parsed'],
    )
    .to_frame()
    .reset_index(drop=True)
)
grade_perfeita['date'] = grade_perfeita['date_parsed'].dt.strftime('%Y-%m-%d')

df_export = pd.merge(
    grade_perfeita, df_export,
    on=[VARS['municipio'], 'date', 'date_parsed'],
    how='left',
)

df_export['pred'] = (
    df_export.groupby(VARS['municipio'])['pred']
    .transform(lambda x: x.interpolate(method='linear').bfill().ffill())
)
df_export['pred'] = np.maximum(df_export['pred'], 0).round(2)

# 6.4 Intervalos de predição aninhados
desvio_padrao = df_export['pred'] * 0.25 + 1.5
z_scores = {'50': 0.674, '80': 1.282, '90': 1.645, '95': 1.960}

for perc, z in z_scores.items():
    df_export[f'lower_{perc}'] = np.maximum(df_export['pred'] - z * desvio_padrao, 0).round(2)
    df_export[f'upper_{perc}'] = (df_export['pred'] + z * desvio_padrao).round(2)

# 6.5 Seleção final de colunas
colunas_finais = [
    VARS['municipio'], 'date', 'pred',
    'lower_50', 'upper_50',
    'lower_80', 'upper_80',
    'lower_90', 'upper_90',
    'lower_95', 'upper_95',
]
df_submissao = df_export[colunas_finais].sort_values(by=[VARS['municipio'], 'date']).copy()

# ==============================================================================
# PIPELINE INTERNO DE ASSERÇÃO (CHECKLIST DE REJEIÇÃO DA PLATAFORMA)
# ==============================================================================
print("\n[VERIFICAÇÃO DE REGRAS] Executando testes de validação...")

datas_sub = pd.to_datetime(df_submissao['date'])
assert (datas_sub.dt.dayofweek == 6).all(), \
    "Erro: Existem datas que não caem em um domingo."

assert (df_submissao['pred'] >= 0).all(), \
    "Erro: Existem predições centrais negativas."

for p in ['50', '80', '90', '95']:
    assert (df_submissao[f'lower_{p}'] >= 0).all(), \
        f"Erro: Valores negativos encontrados em lower_{p}."

condicao_aninhamento = (
    (df_submissao['lower_95'] <= df_submissao['lower_90']) &
    (df_submissao['lower_90'] <= df_submissao['lower_80']) &
    (df_submissao['lower_80'] <= df_submissao['lower_50']) &
    (df_submissao['lower_50'] <= df_submissao['pred'])     &
    (df_submissao['pred']     <= df_submissao['upper_50']) &
    (df_submissao['upper_50'] <= df_submissao['upper_80']) &
    (df_submissao['upper_80'] <= df_submissao['upper_90']) &
    (df_submissao['upper_90'] <= df_submissao['upper_95'])
).all()

assert condicao_aninhamento, \
    "Erro: Inversão detectada nos intervalos de confiança."

print("? Todos os testes internos passaram com sucesso. Dados em conformidade com o edital.")

# 6.6 Gravação final
nome_arquivo = "submission_dengue_agcrn.csv"
df_submissao.to_csv(nome_arquivo, index=False)
print(f"Arquivo salvo com segurança: {nome_arquivo}")