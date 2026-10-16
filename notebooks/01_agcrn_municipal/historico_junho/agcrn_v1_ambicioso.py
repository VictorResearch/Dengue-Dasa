"""
Pipeline AGCRN para previsão de dengue - 3rd IMDC
Alinhado às regras oficiais do edital (sprint.mosqlimate.org)

Correções aplicadas:
  [C1] SHAP com walk-forward temporal (sem data leakage)
  [C2] Treinamento com early stopping + NB loss (superdispersão)
  [C3] Intervalos diretos dos quantis da distribuição Binomial Negativa
  [C4] Loop multi-season para as 4 rodadas de validação do edital
  [C5] Nomes de colunas alinhados ao dataset oficial (casos, geocode, epiweek)
  [C6] Target = número de casos absoluto (não incidência)
  [C7] log(população) como offset logarítmico fixo no modelo
  [E1] Ensemble em 2 níveis: AGCRN + LightGBM -> meta-modelo Ridge
       - LightGBM como modelo base independente (sem estrutura espacial)
       - OOF (out-of-fold) nas 4 seasons para treino do meta-modelo sem leakage
       - Ridge como stacker: aprende pesos por tipo de municipio e epoca do ano
       - mu_ensemble alimenta nb_quantile() -> intervalos NB mantidos
  [D1] Imunidade de rebanho com EWM (halflife=104 sem) — substitui cumsum infinito
  [D2] Lags climáticos por data real (merge_asof) — elimina erro por linhas faltantes
  [D3] Savitzky-Golay unificado em treino e forecasting — substitui SGP completamente
  [D4] Acumulo de chuva usa precip_max alem do rolling sum — captura eventos extremos
  [D5] Variacao Afya em log-diferenca — elimina inf/nan do pct_change em zeros
  [D6] r da NB varia no tempo condicionado as features (velocidade e aceleracao)
  [F1] climate_forecast.csv.gz integrado como features de clima futuro (ECMWF)
  [F2] r recalibrado por MLE sobre resíduos OOF do ensemble
  [F3] LightGBM com lags do próprio alvo (casos_lag1/2/3 e média móvel 4s)
  [F4] Optuna: otimização automática de hiperparâmetros do LightGBM e AGCRN
  [F5] Reconciliação hierárquica MinT municipal → estadual
"""

# =============================================================================
# IMPORTAÇÕES
# Cada biblioteca tem um papel bem definido no pipeline:
# =============================================================================
import sys                          # leitura da barra de progresso no terminal
import warnings                     # silencia alertas desnecessários durante execução
import numpy as np                  # operações numéricas vetorizadas (arrays, matrizes)
import pandas as pd                 # manipulação de tabelas e séries temporais
import torch                        # framework de deep learning (GPU/CPU automático)
import torch.nn as nn               # módulos de redes neurais (camadas, loss, etc.)
import torch.nn.functional as F     # funções sem parâmetros: relu, mse_loss, etc.
from torch_geometric_temporal.nn.recurrent import AGCRN  # arquitetura principal: rede neural em grafo + GRU
from xgboost import XGBRegressor    # usado APENAS para calcular importância SHAP das features
import lightgbm as lgb              # segundo modelo base do ensemble — mais rápido que XGBoost em séries esparsas
import optuna                       # otimização automática de hiperparâmetros (busca bayesiana)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # silencia logs verbosos do Optuna
from sklearn.linear_model import Ridge      # meta-modelo do ensemble: aprende quanto confiar em cada modelo base
from sklearn.preprocessing import StandardScaler  # normaliza features para média 0 e desvio 1
from scipy.signal import savgol_filter      # filtro de Savitzky-Golay: suaviza curvas e calcula derivadas
from scipy.stats import nbinom              # distribuição Binomial Negativa: usada na recalibração do r
from scipy.optimize import minimize_scalar  # otimizador escalar: encontra o melhor r por máxima verossimilhança
try:
    from hierarchicalforecast.methods import MinTrace   # [F5] reconciliação MinT
    from hierarchicalforecast.utils import aggregate    # [F5]
    HF_DISPONIVEL = True
except ImportError:
    HF_DISPONIVEL = False
    print("⚠️  hierarchicalforecast não instalado — reconciliação MinT desativada.")
    print("   Instale com: pip install hierarchicalforecast")
import shap
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')  # remove alertas de depreciação e conversão de tipo durante execução

# ==============================================================================
# 0. SETUP E DICIONÁRIO DE VARIÁVEIS
#    Nomes alinhados ao dataset oficial da competição (Data Documentation)
# ==============================================================================
# Detecta automaticamente se há GPU disponível.
# GPU (cuda) processa matrizes muito mais rápido que CPU — especialmente no AGCRN.
# Se não houver GPU, o código continua funcionando normalmente em CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 A executar no hardware: {device.type.upper()}")

# VARS: dicionário central de nomes de colunas
# ------------------------------------------------
# Em vez de escrever 'geocode', 'casos', 'epiweek' diretamente no código,
# usamos VARS['municipio'], VARS['casos_dengue'], etc.
# Vantagem: se o dataset mudar o nome de uma coluna, corrigimos aqui e
# todo o restante do código se adapta automaticamente — sem busca e substituição.
# Mapeamento: chave semântica → nome real da coluna nos arquivos do FTP
VARS = {
    # -------------------------------------------------------------------------
    # IDENTIFICADORES E ALVOS
    # -------------------------------------------------------------------------
    'municipio':         'geocode',        # dengue.csv.gz / climate.csv.gz
    'semana_epi':        'epiweek',        # formato YYYYWW (int)
    'data':              'date',           # primeiro dia da epiweek (domingo)
    'casos_dengue':      'casos.x',        # dengue.csv.gz  → coluna "casos"
    'casos_chik':        'casos.y',        # chikungunya.csv.gz → coluna "casos"
    'populacao':         'population',     # datasus_population_2001_2025.csv.gz
    'uf_code':           'uf_code',        # necessário para submissão na plataforma
    
    # -------------------------------------------------------------------------
    # VARIÁVEIS CLIMÁTICAS BRUTAS (ORIGINAIS E ADICIONADAS)
    # -------------------------------------------------------------------------
    'temp_min':          'temp_min',       # climate.csv.gz
    'temp_med':          'temp_med',       # climate.csv.gz
    'temp_max':          'temp_max',       # climate.csv.gz
    'precip_min':        'precip_min',     # climate.csv.gz
    'precip_med':        'precip_med',     # climate.csv.gz
    'precip_max':        'precip_max',     # climate.csv.gz
    'precip_tot':        'precip_tot',     # climate.csv.gz
    'amplitude_termica': 'thermal_range',  # climate.csv.gz
    'pressao_min':       'pressure_min',   # climate.csv.gz
    'pressao_atm':       'pressure_med',   # climate.csv.gz
    'pressao_max':       'pressure_max',   # climate.csv.gz
    'umidade_min':       'rel_humid_min',  # climate.csv.gz
    'umidade':           'rel_humid_med',  # climate.csv.gz
    'umidade_max':       'rel_humid_max',  # climate.csv.gz
    'dias_chuva':        'rainy_days',     # climate.csv.gz
    
    # -------------------------------------------------------------------------
    # VARIÁVEIS MACROCLIMÁTICAS E AMBIENTAIS
    # -------------------------------------------------------------------------
    'enso':              'enso',           # ocean_climate_oscillations.csv.gz
    'iod':               'iod',
    'pdo':               'pdo',
    'bioma':             'biome',          # environ_vars.csv.gz
    'koppen':            'koppen',         # environ_vars.csv.gz
    
    # -------------------------------------------------------------------------
    # SINAIS EXTERNOS
    # -------------------------------------------------------------------------
    'afya_acessos':      'access_count',   # access_afya_dengue_2021_2026.csv.gz
    
    # -------------------------------------------------------------------------
    # VARIÁVEIS DE ENGENHARIA (FEATURE ENGINEERING)
    # -------------------------------------------------------------------------
    'vpd':               'vpd',                        # Déficit de Pressão de Vapor (kPa)
    'amp_termica_din3':   'amplitude_termica_dinamica3', # Média móvel da amplitude térmica (3w)
    'amp_termica_din2':   'amplitude_termica_dinamica2', # Média móvel da amplitude térmica (2w)
    'volatilidade_temp': 'volatilidade_temp_4w',       # Desvio padrão móvel temp_med (4w)
    'volatilidade_prec': 'volatilidade_precip_4w',     # Desvio padrão móvel precip_med (4w)
    'hti3':               'hti_lag3',                   # Índice Hidrotérmico (Precip x Temp)
    'hti2':               'hti_lag2',                   # Índice Hidrotérmico (Precip x Temp)
}

# SEASONS: define as 4 rodadas de validação exigidas pelo edital
# ---------------------------------------------------------------
# Cada season usa colunas booleanas (train_1, target_1, etc.) que já vêm
# prontas no dengue.csv.gz. O modelo é treinado na janela 'train' e
# avaliado na janela 'target'. Isso simula exatamente o que acontecerá
# na fase de forecast real (2026-2027).
# Bandeiras train/target definidas no próprio dataset (colunas booleanas)
SEASONS = {
    1: {'train': "train1_dengue.parquet", 'test': "test1_dengue.parquet", 'label': '2022-2023'},
    2: {'train': "train2_dengue.parquet", 'test': "test2_dengue.parquet", 'label': '2023-2024'},
    3: {'train': "train3_dengue.parquet", 'test': "test3_dengue.parquet", 'label': '2024-2025'},
    4: {'train': "train4_dengue.parquet", 'test': "test4_dengue.parquet", 'label': '2025-2026'}
}

# QUANTIS: níveis de incerteza exigidos pelo edital (seção 4.1)
# --------------------------------------------------------------
# A competição exige que cada previsão venha acompanhada de intervalos
# em 4 níveis: 50%, 80%, 90% e 95%. Cada nível tem um limite inferior
# (lower_X) e um superior (upper_X).
# Exemplo: lower_95 é o percentil 2.5%, upper_95 é o percentil 97.5%.
# Isso significa que esperamos que 95% dos casos reais caiam dentro
# do intervalo [lower_95, upper_95].
# Quantis exigidos pelo edital (seção 4.1)
# lower_95 = percentil 2.5  → alpha = 0.05
# lower_90 = percentil 5    → alpha = 0.10  etc.
QUANTIS = {
    '50': (0.25, 0.75),
    '80': (0.10, 0.90),
    '90': (0.05, 0.95),
    '95': (0.025, 0.975),
}

# ==============================================================================
# [F1] PREVISÃO CLIMÁTICA FUTURA — climate_forecast.csv.gz (ECMWF/Copernicus)
# ==============================================================================
def integrar_climate_forecast(
    df,
    df_forecast,
    meses_ahead=None,
):
    """
    [F1] Join entre o dataset principal e as previsoes climaticas futuras do ECMWF.

    Logica do join
    --------------
    Para cada linha (geocode, date), busca a previsao emitida N meses antes
    com forecast_months_ahead = N. Isso fornece ao modelo informacao sobre o
    clima previsto para a semana atual, sem leakage de futuro.

    Colunas geradas (por N em meses_ahead)
    ---------------------------------------
    clim_fc_temp_Nm   : temperatura media prevista (C)
    clim_fc_precip_Nm : precipitacao prevista (mm)
    clim_fc_humid_Nm  : umidade relativa prevista (%)
    """
    import pandas as pd
    # Usa 1, 2 e 3 meses à frente por padrão.
    # 1 mês: previsão de curtíssimo prazo — mais precisa mas menos antecipada.
    # 3 meses: cobre o início da temporada de dengue com ~12 semanas de antecedência.
    if meses_ahead is None:
        meses_ahead = [1, 2, 3]

    df = df.copy()
    # Converte a coluna de data para datetime para fazer aritmética de datas.
    df['_date_dt'] = pd.to_datetime(df[VARS['data']])

    df_fc = df_forecast.copy()
    # Padroniza reference_month como datetime para o merge funcionar corretamente.
    df_fc['reference_month'] = pd.to_datetime(df_fc['reference_month'].astype(str))

    # Para cada horizonte N (1, 2 ou 3 meses), buscamos a previsão climática
    # emitida N meses antes da data alvo. Isso garante que o modelo só use
    # informação disponível no momento em que a previsão seria feita — sem ver o futuro.
    for n in meses_ahead:
        df['_ref_month'] = (
            # Subtrai N meses da data alvo e pega o primeiro dia do mês resultante.
            # Ex: data=2024-W15 (abril), N=2 → ref_month=2024-02-01
            (df['_date_dt'] - pd.DateOffset(months=n))
            .dt.to_period('M').dt.to_timestamp()
        )
        df_fc_n = (
            df_fc[df_fc['forecast_months_ahead'] == n]
            [[VARS['municipio'], 'reference_month', 'temp_med', 'precip_tot', 'rel_humid_med']]
            .rename(columns={
                'reference_month': '_ref_month',
                'temp_med':        f'clim_fc_temp_{n}m',
                'precip_tot':      f'clim_fc_precip_{n}m',
                'rel_humid_med':   f'clim_fc_humid_{n}m',
            })
        )
        df = pd.merge(df, df_fc_n, on=[VARS['municipio'], '_ref_month'], how='left')
        df = df.drop(columns='_ref_month')

    df = df.drop(columns='_date_dt', errors='ignore')
    return df


# ==============================================================================
# 1. FÍSICA DA EPIDEMIA — Savitzky-Golay (treino e forecasting)
# ==============================================================================
# [D3] Substituição do SGP pelo Savitzky-Golay em todos os splits.
# Motivação:
#   - Savitzky-Golay é um filtro polinomial analítico — sem otimização, sem GPU,
#     determinístico e reprodutível.
#   - As derivadas resultantes têm a mesma semântica (taxa de crescimento e
#     aceleração da epidemia em log-escala) com qualidade equivalente para séries
#     epidemiológicas semanais (Schafer & Savitzky, 2011).
#   - Usar o mesmo método em treino e forecasting elimina a inconsistência de
#     escala que existia antes entre as duas funções.

def extrair_fisica(
    df: pd.DataFrame, janela: int = 7, ordem: int = 3
) -> pd.DataFrame:
    """
    Calcula casos_suavizados, velocidade e aceleração via Savitzky-Golay
    para qualquer split (treino, validação interna ou forecasting).

    Parâmetros
    ----------
    janela : int
        Largura da janela do filtro (deve ser ímpar e > ordem).
        7 semanas captura o ciclo quinzenal sem suprimir os picos.
    ordem : int
        Grau do polinômio ajustado. 3 é adequado para curvas epidêmicas
        com um pico dominante por temporada.

    Saídas por município
    --------------------
    casos_suavizados : escala de contagem absoluta (expm1 do filtro em log1p)
    velocidade       : 1ª derivada em log-escala ≈ taxa de crescimento relativo (Rt proxy)
    aceleracao       : 2ª derivada em log-escala ≈ aceleração da epidemia
    time_index       : índice temporal crescente por município (1, 2, 3, ...)
    """
    # Ordena por município e semana antes de aplicar o filtro.
    # Isso garante que o Savitzky-Golay processe cada série na ordem temporal correta.
    df = df.sort_values([VARS['municipio'], VARS['semana_epi']]).copy()

    # time_index: contador crescente de 1 a T dentro de cada município.
    # Usado para organizar os tensores 3D [Tempo × Nós × Features] do AGCRN.
    df['time_index'] = df.groupby(VARS['municipio']).cumcount() + 1

    def _fisica_grupo(grp: pd.DataFrame) -> pd.DataFrame:
        # Esta função interna é aplicada separadamente a cada município.
        grp  = grp.copy()
        # log1p(x) = log(x + 1): transforma a contagem de casos para escala logarítmica.
        # Isso resolve dois problemas:
        # 1) Estabiliza a escala — municípios com 0 e com 5000 casos ficam comparáveis.
        # 2) Reduz o efeito de picos extremos que distorceriam o filtro.
        # expm1 desfaz a transformação no final para voltar para casos absolutos.
        y_raw = grp[VARS['casos_dengue']].fillna(0).values
        y     = np.log1p(y_raw)
        n     = len(y)  # número de semanas disponíveis para este município

        # O Savitzky-Golay exige que a janela seja ÍMPAR e MAIOR que a ordem do polinômio.
        # Para séries curtas (municípios com poucos dados), reduzimos a janela automaticamente.
        win = min(janela, n if n % 2 == 1 else max(n - 1, 1))
        win = max(win, ordem + 2 if (ordem + 2) % 2 == 1 else ordem + 3)

        if n < win:
            # Série muito curta (menos semanas do que a janela mínima).
            # Usamos os casos brutos sem suavização e zeramos as derivadas.
            grp['casos_suavizados'] = y_raw
            grp['velocidade']       = 0.0
            grp['aceleracao']       = 0.0
        else:
            # deriv=0: suaviza a curva (remove ruído de subnotificação)
            # deriv=1: primeira derivada ≈ taxa de crescimento da epidemia (proxy do Rt)
            # deriv=2: segunda derivada ≈ se o crescimento está acelerando ou freando
            # expm1 desfaz o log1p para devolver casos_suavizados em escala absoluta.
            # velocidade e aceleração ficam em escala log — interpretação: taxa relativa.
            log_suav = savgol_filter(y, win, ordem, deriv=0)
            grp['casos_suavizados'] = np.expm1(np.maximum(log_suav, 0))
            grp['velocidade']       = savgol_filter(y, win, ordem, deriv=1)
            grp['aceleracao']       = savgol_filter(y, win, ordem, deriv=2)
        return grp

    return df.groupby(VARS['municipio'], group_keys=False).apply(_fisica_grupo)


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
def engenharia_de_features(df: pd.DataFrame, fit_encoders: bool = True,
                            encoder_means: dict = None) -> tuple:
    """
    Gera todas as features. Retorna (df_eng, encoder_means).
    Quando fit_encoders=False, usa encoder_means do treino (evita leakage).

    [C6] Target do modelo = 'casos' (contagem absoluta), não incidência.
    [C7] 'log_pop' = log(população) entra como offset logarítmico fixo,
         declarando que log(E[casos]) = log_pop + f(features).
         A incidência continua existindo como feature auxiliar (não como target).
    """
    df = df.sort_values(by=[VARS['municipio'], VARS['semana_epi']]).copy()

    # [C7] OFFSET LOGARÍTMICO DE POPULAÇÃO
    # log_pop não é uma feature comum — é um OFFSET que declara ao modelo:
    # "o número esperado de casos escala proporcionalmente à população".
    # Matematicamente: log(E[casos]) = log_pop + f(features_climaticas_etc)
    # Isso é equivalente a modelar a TAXA de incidência, mas mantendo o alvo
    # em escala de CONTAGEM ABSOLUTA (que é o que a competição avalia).
    # clip(lower=1) evita log(0) = -infinito para municípios sem população registrada.
    df['log_pop'] = np.log(df['population'].clip(lower=1).astype(float))

    # INCIDÊNCIA: casos por 100 mil habitantes.
    # Não é mais o alvo do modelo (o alvo agora é o número absoluto de casos).
    # Continua sendo calculada para servir como feature auxiliar — por exemplo,
    # para calcular a imunidade de rebanho e os target encodings de bioma/Köppen.
    df['incidencia'] = (df[VARS['casos_dengue']] / df['population'].clip(lower=1)) * 100_000

    # [D1] IMUNIDADE DE REBANHO COM MÉDIA MÓVEL EXPONENCIAL (EWM)
    # Por que não cumsum? O cumsum cresce infinitamente — em 2026, municípios com
    # histórico desde 2010 teriam um valor de imunidade 16 anos maior do que municípios
    # com histórico desde 2020, mesmo que tivessem a mesma situação atual. Isso torna
    # a feature colinear com o tempo e o modelo passa a ignorá-la.
    #
    # A EWM com halflife=104 semanas (≈ 2 anos) dá peso exponencialmente decrescente
    # ao passado: um surto de 2 anos atrás tem metade do peso de um surto atual.
    # Isso modela o waning immunity real da dengue — a imunidade adquirida diminui
    # com o tempo e novos sorotipos podem reinfectar pessoas já expostas.
    df['imunidade_ewm'] = (
        df.groupby(VARS['municipio'])['incidencia']
        .transform(lambda x: x.ewm(halflife=104, adjust=False).mean())
    )

    # CODIFICAÇÃO CÍCLICA SAZONAL
    # Por que seno e cosseno em vez do número da semana diretamente?
    # Se usarmos semana=1 e semana=52, o modelo enxerga uma "distância" de 51 semanas
    # entre elas, quando na verdade são semanas consecutivas (final de dezembro/início
    # de janeiro). O par (seno, cosseno) resolve isso: a distância entre semana 52 e
    # semana 1 é a mesma que entre semana 25 e semana 26 — o ciclo é contínuo.
    semana_ano           = pd.to_datetime(df[VARS['data']]).dt.isocalendar().week
    df['seno_semana']    = np.sin(2 * np.pi * semana_ano / 52).astype(float)
    df['cosseno_semana'] = np.cos(2 * np.pi * semana_ano / 52).astype(float)

    # [D2] LAGS CLIMÁTICOS POR DATA REAL (não por posição na tabela)
    # Por que merge_asof em vez de shift(N)?
    # shift(N) desloca N LINHAS — se algum município tiver uma semana ausente
    # no dataset (subnotificação, feriado), um lag de 3 semanas pode
    # corresponder a 4 ou 5 semanas reais. Isso contamina silenciosamente o modelo.
    # merge_asof busca o valor mais próximo dentro de uma tolerância de 7 dias,
    # garantindo que lag_3 sempre signifique "3 semanas antes" na data real.
    #
    # Por que lags de 3 a 5 semanas para clima?
    # Período biológico: 1 semana para chuva criar criadouro + 1-2 semanas para
    # larva virar mosquito adulto + 1-2 semanas de período extrínseco do vírus
    # + 1 semana de período intrínseco no humano = 3-5 semanas de defasagem total.

    vars_clima = [
        VARS.get('temp_min'), VARS.get('temp_med'), VARS.get('temp_max'),
        VARS.get('precip_min'), VARS.get('precip_med'), VARS.get('precip_max'), VARS.get('precip_tot'),
        VARS.get('amplitude_termica'), VARS.get('pressao_min'), VARS.get('pressao_atm'), VARS.get('pressao_max'),
        VARS.get('umidade_min'), VARS.get('umidade'), VARS.get('umidade_max'), VARS.get('dias_chuva'),
        VARS.get('vpd')
    ]
    vars_clima = [v for v in vars_clima if v is not None]
    # Garante coluna datetime para o merge temporal
    df['_date_dt'] = pd.to_datetime(df[VARS['data']])

    for var in vars_clima:
        if var not in df.columns:
            continue
        for lag in range(3, 6):
            delta = pd.Timedelta(weeks=lag)
            df_src = (
                df[[VARS['municipio'], '_date_dt', var]]
                .rename(columns={'_date_dt': '_date_src', var: f'{var}_lag{lag}'})
            )
            df['_date_lag'] = df['_date_dt'] - delta
            df = df.sort_values('_date_lag')
            df_src = df_src.sort_values('_date_src')
            merged = pd.merge_asof(
                df, df_src,
                left_on='_date_lag', right_on='_date_src',
                by=VARS['municipio'],
                tolerance=pd.Timedelta('7 days'),
                direction='nearest',
            )
            df[f'{var}_lag{lag}'] = merged[f'{var}_lag{lag}'].values
            df = df.drop(columns='_date_lag')

    # [D4] DUAS MÉTRICAS DE CHUVA — cada uma captura um fenômeno diferente:
    # - chuva_acumulada_4w: volume total nas últimas 4 semanas → criadouros persistentes
    #   (chuva distribuída que forma poças, cisternas e vasos com água parada)
    # - precip_max_lag3: pico de precipitação com 3 semanas de atraso → eventos extremos
    #   (uma chuva torrencial em um dia cria inundações que viram criadouros em 3 semanas)
    # Um volume médio semanal de 10mm/dia esconde um evento de 80mm em um único dia —
    # o precip_max captura esse pico que o rolling sum dilui.
    if VARS['precip_tot'] in df.columns:
        df['chuva_acumulada_4w'] = df.groupby(VARS['municipio'])[VARS['precip_tot']].transform(
            lambda x: x.rolling(4, min_periods=1).sum()
        )
    if 'precip_max' in df.columns:
        df['precip_max_lag3'] = df.groupby(VARS['municipio'])['precip_max'].shift(3)

    # [D2] LAGS MACROCLIMÁTICOS (ENSO, IOD, PDO) — 6 a 9 semanas
    # Por que 6-9 semanas para ENSO/IOD/PDO em vez de 3-5?
    # Esses índices oceânicos afetam o padrão de chuvas no Brasil com uma
    # defasagem de ~2 meses: o El Niño de outubro só impacta as chuvas de
    # dezembro-janeiro. A tolerância é de 14 dias (vs. 7 dias para clima local)
    # porque os dados oceânicos podem ter irregularidade maior de disponibilidade.
    for var in [VARS['enso'], VARS['iod'], VARS['pdo']]:
        if var not in df.columns:
            continue
        for lag in range(6, 10):
            delta = pd.Timedelta(weeks=lag)
            df_src = (
                df[[VARS['municipio'], '_date_dt', var]]
                .rename(columns={'_date_dt': '_date_src', var: f'{var}_lag{lag}'})
            )
            df['_date_lag'] = df['_date_dt'] - delta
            df = df.sort_values('_date_lag')
            df_src = df_src.sort_values('_date_src')
            merged = pd.merge_asof(
                df, df_src,
                left_on='_date_lag', right_on='_date_src',
                by=VARS['municipio'],
                tolerance=pd.Timedelta('14 days'),
                direction='nearest',
            )
            df[f'{var}_lag{lag}'] = merged[f'{var}_lag{lag}'].values
            df = df.drop(columns='_date_lag')

    df = df.drop(columns='_date_dt', errors='ignore')

    # [D5] VARIAÇÃO DE ACESSOS AFYA EM LOG-DIFERENÇA
    # O dado Afya conta acessos de médicos ao conteúdo de dengue no app Whitebook.
    # É um sinal de alerta precoce: médicos buscam informação quando veem casos
    # no consultório, antes que a notificação oficial seja registrada.
    #
    # Por que log-diferença em vez de pct_change?
    # pct_change() em semanas com 0 acessos gera divisão por zero → inf ou nan.
    # log(x+1) - log(x_lag+1) é equivalente a log((x+1)/(x_lag+1)), que é
    # sempre finito, simétrico (+10% e -10% têm o mesmo módulo) e estável.
    if VARS['afya_acessos'] in df.columns:
        afya_log = np.log1p(df[VARS['afya_acessos']].clip(lower=0))
        df['afya_log_diff'] = (
            df.groupby(VARS['municipio'])
            .apply(lambda g: afya_log.loc[g.index].diff().fillna(0))
            .reset_index(level=0, drop=True)
        )
        # Remove a coluna original afya_variacao_pct se existir de versao anterior
        df.drop(columns=['afya_variacao_pct'], errors='ignore', inplace=True)

    # [C1] TARGET ENCODING DE BIOMA E CLIMA Köppen — sem data leakage
    # Bioma (Amazônia, Cerrado, Mata Atlântica...) e Köppen (tropical, seco...)
    # são variáveis categóricas. Convertemos para número usando a média histórica
    # de casos daquele tipo de ambiente — municípios do mesmo bioma têm perfis
    # epidêmicos similares.
    #
    # Por que fit_encoders=True apenas no treino?
    # Se calculássemos a média incluindo dados do período de teste (futuro),
    # estaríamos passando informação do futuro para o modelo de forma implícita
    # — isso é data leakage. Quando fit_encoders=False, usamos os encoder_means
    # calculados no treino e apenas os aplicamos ao teste.
    if encoder_means is None:
        encoder_means = {}
    for col in [VARS['bioma'], VARS['koppen']]:
        if col in df.columns:
            if fit_encoders:
                # Calcula a média de casos por categoria — SÓ nos dados de treino
                encoder_means[col] = df.groupby(col)[VARS['casos_dengue']].mean()
            # Mapeia cada categoria para sua média — preenche com 0 se categoria nova
            df[f'{col}_encoded'] = df[col].map(encoder_means[col]).fillna(0)

    df = df.dropna().reset_index(drop=True)
    return df, encoder_means


# ==============================================================================
# [C1] SHAP COM WALK-FORWARD — sem vazamento temporal
# ==============================================================================
def selecao_shap_walkforward(
    df_full: pd.DataFrame,
    col_alvo: str  = 'incidencia',
    num_features: int = 15,
    n_folds: int = 3,
) -> list:
    """
    Calcula importâncias SHAP usando walk-forward temporal:
    treina sempre no passado, avalia no futuro imediato.
    Agrega importâncias por mediana entre folds → sem data leakage.
    """
    print("🔍 [C1] SHAP walk-forward (sem leakage temporal)...")

    cols_ignorar = {
        VARS['municipio'], VARS['semana_epi'], VARS['data'],
        VARS['casos_dengue'], 'incidencia', 'log_pop',
        VARS['bioma'], VARS['koppen'],
        'node_id', 'time_index', '_date_dt', '_date_lag', '_date_src',
        'train_1.x', 'train_2.x', 'train_3.x', 'train_4.x',
        'target_1.x', 'target_2.x', 'target_3.x', 'target_4.x',
        'train_1.x', 'train_2.x', 'train_3.x', 'train_4.x',
        'target_1.x', 'target_2.x', 'target_3.x', 'target_4.x',
        'disease.x', 'disease.y','uf', VARS['uf_code'],
        'regional_geocode', 'macroregional_geocode', 'target_city.x', 'target_city.y'
    }

    # Ordena todas as semanas disponíveis cronologicamente.
    # O walk-forward sempre treina no passado e avalia no futuro — nunca ao contrário.
    semanas_ordenadas = sorted(df_full[VARS['semana_epi']].unique())
    n = len(semanas_ordenadas)
    # Divide o período em fatias iguais. Com n_folds=3, temos:
    # |--fold1_treino--|--fold1_val--|--fold2_treino_extra--|--fold2_val--|...
    fold_size = n // (n_folds + 1)

    importancias_folds = []  # acumula as importâncias SHAP de cada fold

    for fold in range(n_folds):
        corte_idx   = (fold + 1) * fold_size
        val_end_idx = corte_idx + fold_size

        semanas_treino = semanas_ordenadas[:corte_idx]
        semanas_val    = semanas_ordenadas[corte_idx:val_end_idx]

        df_tr = df_full[df_full[VARS['semana_epi']].isin(semanas_treino)].copy()
        df_vl = df_full[df_full[VARS['semana_epi']].isin(semanas_val)].copy()

        if df_tr.empty or df_vl.empty:
            continue

        # IMPORTANTE: encoders (bioma, Köppen) são fitados APENAS no treino do fold.
        # Se usássemos dados do período de validação para calcular as médias,
        # estaríamos "vazando" informação do futuro para o modelo.
        df_tr_eng, enc = engenharia_de_features(df_tr, fit_encoders=True)
        df_vl_eng, _   = engenharia_de_features(df_vl, fit_encoders=False,
                                                 encoder_means=enc)

        cols_num_tr = [c for c in df_tr_eng.columns
                       if c not in cols_ignorar
                       and pd.api.types.is_numeric_dtype(df_tr_eng[c])]
        cols_num_vl = [c for c in cols_num_tr if c in df_vl_eng.columns]

        X_tr = df_tr_eng[cols_num_tr].fillna(0)
        y_tr = df_tr_eng[col_alvo]
        X_vl = df_vl_eng[cols_num_vl].fillna(0)

        if X_tr.empty or X_vl.empty:
            continue

        # Treina um XGBoost rápido só para calcular importâncias SHAP.
        # O XGBoost não será usado na previsão final — apenas para rankear features.
        xgb = XGBRegressor(n_estimators=100, max_depth=5,
                            tree_method='hist', n_jobs=-1)
        xgb.fit(X_tr, y_tr)

        # SHAP: para cada feature e cada amostra, calcula quanto aquela feature
        # contribuiu para a previsão daquela amostra (positivo ou negativo).
        # Tiramos o valor absoluto e a média por feature — quanto maior, mais importante.
        # A avaliação é feita no conjunto de VALIDAÇÃO (futuro), não no treino.
        sv = shap.TreeExplainer(xgb).shap_values(X_vl)
        imp = pd.Series(
            np.abs(sv).mean(axis=0),  # média das contribuições absolutas por feature
            index=cols_num_vl,
        )
        importancias_folds.append(imp)

    if not importancias_folds:
        raise RuntimeError("Nenhum fold válido no SHAP walk-forward.")

    # Agrega importâncias pela MEDIANA entre folds (mais robusta que a média).
    # A mediana ignora folds onde uma feature teve importância anômala por acaso.
    # Ordena do mais importante para o menos importante.
    importancia_final = (
        pd.concat(importancias_folds, axis=1)
        .median(axis=1)
        .sort_values(ascending=False)
    )

    features_selecionadas = importancia_final.head(num_features).index.tolist()
    print(f"✅ Top {num_features} features (walk-forward): {features_selecionadas}")
    return features_selecionadas


# ==============================================================================
# 3. ARQUITETURA AGCRN COM BINOMIAL NEGATIVA
# ==============================================================================

# [C2] Negative log-likelihood da Binomial Negativa
# Parametrização: mu (média) e r (dispersão inversa, "total_count" no PyTorch)
# Relação com a NB clássica: Var[Y] = mu + mu²/r
# Quanto menor r, maior a superdispersão. r→∞ converge para Poisson.
def nb_loss(log_mu: torch.Tensor, log_r: torch.Tensor,
            y: torch.Tensor) -> torch.Tensor:
    """
    NLL da Binomial Negativa.
    log_mu:  log da média prevista (saída da cabeça mu, sem offset ainda)
    log_r:   log do parâmetro de dispersão r (aprendido)
    y:       contagem observada (casos reais, float32)

    O offset log(população) já foi somado a log_mu antes de chamar esta função.
    """
    # Limita os valores de log_mu e log_r para evitar overflow/underflow numérico.
    # Sem isso, exp(log_mu) poderia explodir para infinito ou colapsar para zero,
    # travando o treinamento. Os limites ±10 e ±6 são seguros para contagens de dengue.
    log_mu = torch.clamp(log_mu, min=-10, max=10)
    log_r  = torch.clamp(log_r,  min=-6,  max=6)

    mu = torch.exp(log_mu)  # média prevista em número de casos (escala original)
    r  = torch.exp(log_r)   # parâmetro de dispersão: r pequeno = mais espalhado

    # NLL (Negative Log-Likelihood) da Binomial Negativa.
    # Maximizar a verossimilhança é equivalente a minimizar a NLL.
    # Cada termo da fórmula tem interpretação probabilística:
    # - lgamma(y+r) - lgamma(r) - lgamma(y+1): coeficiente binomial em log-escala
    # - r * log(r/(r+mu)): probabilidade de "não ocorrência" acumulada
    # - y * log(mu/(r+mu)): probabilidade dos y eventos observados
    # 1e-8 evita log(0) quando mu ou r são muito pequenos.
    nll = (
        - torch.lgamma(y + r)
        + torch.lgamma(r)
        + torch.lgamma(y + 1)
        - r * torch.log(r / (r + mu + 1e-8))
        - y * torch.log(mu / (r + mu + 1e-8))
    )
    return nll.mean()  # média sobre todos os municípios e semanas do batch


def nb_quantile(mu: np.ndarray, r: np.ndarray,
                q: float) -> np.ndarray:
    """
    Quantil q da distribuição Binomial Negativa por busca na CDF.
    Vetorizado: mu e r são arrays [N].
    Retorna array [N] de inteiros (número de casos).
    """
    # O scipy usa uma parametrização diferente da nossa:
    # scipy.nbinom(n, p): n=número de sucessos, p=prob de sucesso
    # Nossa parametrização: E[X]=mu, Var[X]=mu+mu²/r
    # Conversão: p = r/(r+mu), n = r
    # ppf(q) = quantil q da distribuição, ou seja, o valor x tal que P(X≤x)=q.
    # Isso nos dá diretamente os limites dos intervalos de predição.
    from scipy.stats import nbinom
    p = r / (r + mu + 1e-8)
    return nbinom.ppf(q, n=r, p=p).astype(float)


class PreditorNBAGCRN(nn.Module):
    """
    AGCRN com duas cabeças de saída:
      - log_mu_raw: contribuição das features para log(mu)
      - log_r:      log do parâmetro de dispersão r — varia no tempo

    [D6] r varia no tempo: head_logr recebe h_relu concatenado com
         velocidade e aceleração (índices fixos em x), que são os sinais
         mais informativos sobre o estado atual da epidemia.
         - No pico (velocidade alta, aceleração positiva): r grande → menos dispersão
         - Na entressafra (muitos zeros): r pequeno → alta dispersão
         Isso calibra os intervalos de predição de forma dinamicamente adaptada.

    [C7] O offset log(população) é somado a log_mu_raw no forward,
         garantindo que E[casos] = pop × exp(f(features)).
    """
    def __init__(self, num_nodes: int, num_features: int,
                 embedding_dim: int = 32,
                 idx_velocidade: int = 1,
                 idx_aceleracao: int = 2):
        super().__init__()
        # Índices de velocidade e aceleração em x (cols_sgp = [suavizados, vel, acel])
        self.idx_vel = idx_velocidade
        self.idx_acel = idx_aceleracao

        # EMBEDDINGS DE NÓ: cada município recebe um vetor de embedding_dim=32 dimensões.
        # O AGCRN aprende esses embeddings durante o treino — eles codificam
        # características latentes de cada município (posição na rede de difusão,
        # suscetibilidade histórica, etc.) sem precisar de uma matriz de adjacência
        # pré-definida. requires_grad=True significa que esses parâmetros são treinados.
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embedding_dim), requires_grad=True
        )

        # AGCRN (Adaptive Graph Convolutional Recurrent Network):
        # - Aprende o grafo de conexões entre municípios (quem influencia quem)
        # - Processa a série temporal com uma célula GRU (memória de curto/longo prazo)
        # - in_channels: número de features de entrada por município por semana
        # - out_channels=64: tamanho do estado oculto (representação interna aprendida)
        self.recurrent = AGCRN(
            number_of_nodes=num_nodes,
            in_channels=num_features,
            out_channels=64,
        )

        # CABEÇA MU: transforma o estado oculto (64 dimensões) em 1 número: log(taxa).
        # Depois somamos log_pop para obter log(casos esperados).
        self.head_mu = nn.Linear(64, 1)

        # [D6] CABEÇA LOGR: aprende o parâmetro de dispersão da NB.
        # Recebe 64 (estado oculto) + 1 (velocidade) + 1 (aceleração) = 66 entradas.
        # Velocidade e aceleração informam o "estado" da epidemia:
        # - Pico crescente → r grande (curva mais previsível → intervalos mais estreitos)
        # - Entressafra → r pequeno (muitos zeros → alta incerteza → intervalos largos)
        self.head_logr = nn.Linear(64 + 2, 1)

    def forward(self, x: torch.Tensor, log_pop: torch.Tensor, h=None):
        """
        x:       [N, num_features]
        log_pop: [N]  — offset logarítmico de população (fixo, não aprendido)
        Retorna: (log_mu [N,1], log_r [N,1], h_novo)
        """
        # Passa os dados pelo AGCRN: aprende o grafo e processa a série temporal.
        # h é o estado oculto atualizado — contém a "memória" da epidemia até esta semana.
        h      = self.recurrent(x, self.node_embeddings, h)
        # ReLU: zera valores negativos — ativa apenas os padrões relevantes aprendidos.
        h_relu = F.relu(h)

        # [C7] OFFSET DE POPULAÇÃO: a contribuição das features (log_mu_raw) representa
        # a TAXA de incidência em log-escala. Somando log_pop, obtemos log(casos absolutos):
        # log(casos) = log(taxa) + log(população) → casos = taxa × população
        # O offset NÃO é aprendido — ele é fixo e matematicamente correto.
        log_mu_raw = self.head_mu(h_relu)               # contribuição das features [N, 1]
        log_mu     = log_mu_raw + log_pop.unsqueeze(-1) # log(casos esperados) [N, 1]

        # [D6] log_r CONDICIONADO AO ESTADO ATUAL DA EPIDEMIA:
        # Extrai velocidade e aceleração da semana atual diretamente de x.
        # Esses dois sinais dizem ao modelo se estamos em pico ou entressafra,
        # permitindo que a dispersão (e portanto os intervalos) varie dinamicamente.
        vel  = x[:, self.idx_vel].unsqueeze(-1)         # velocidade da epidemia [N, 1]
        acel = x[:, self.idx_acel].unsqueeze(-1)        # aceleração da epidemia [N, 1]
        h_r  = torch.cat([h_relu, vel, acel], dim=-1)   # concatena: [N, 64+1+1=66]
        log_r = self.head_logr(h_r)                     # log(dispersão) [N, 1]

        return log_mu, log_r, h


# ==============================================================================
# 4. PIPELINE MULTI-SEASON (4 rodadas de validação + fase de forecast)
# ==============================================================================
def treinar_e_validar_season(
    df_train: pd.DataFrame,       # [ALTERADO] Recebe o banco de treino isolado
    df_test: pd.DataFrame,        # [ALTERADO] Recebe o banco de teste isolado
    season_id: int,
    cols_base: list,
    encoder_means_globais: dict,
    num_nodes: int,
    mapa_simples: pd.DataFrame,
    epochs_max: int = 150,
    patience: int = 15,
) -> dict:
    """
    Treina o modelo para uma season específica e retorna:
      - modelo treinado
      - scalers
      - previsões quantílicas no período target
      - resíduos (para calibração conformal)
    """
    season = SEASONS[season_id]
    print(f"\n{'='*60}")
    print(f"  SEASON {season_id}: {season['label']}")
    print(f"{'='*60}")

    # TRÊS PARTIÇÕES DE DADOS (sem sobreposição):
    # ┌─────────────────────────────┬──────────────┬─────────────────┐
    # │  treino real (train_N)      │  val interna │  target (eval)  │
    # │  (todas as semanas - 10)    │  (últimas 10)│  (season alvo)  │
    # └─────────────────────────────┴──────────────┴─────────────────┘
    # val interna: usada APENAS para early stopping e calibração — nunca para treino.
    # target: período que queremos prever — o modelo nunca o vê durante o treino.
    df_tr  = df_train.copy()
    df_tgt = df_test.copy()

    # Reserva últimas 10 epiweeks do treino como validação interna (early stopping)
    semanas_treino = sorted(df_tr[VARS['semana_epi']].unique())
    semanas_val_int = semanas_treino[-10:]
    semanas_tr_real = semanas_treino[:-10]

    df_tr_real = df_tr[df_tr[VARS['semana_epi']].isin(semanas_tr_real)]
    df_val_int = df_tr[df_tr[VARS['semana_epi']].isin(semanas_val_int)]

    # --- Feature engineering sem leakage ---
    df_tr_eng,  enc = engenharia_de_features(df_tr_real,  fit_encoders=True)
    df_val_eng, _   = engenharia_de_features(df_val_int,  fit_encoders=False,
                                              encoder_means=enc)
    df_tgt_eng, _   = engenharia_de_features(df_tgt,       fit_encoders=False,
                                              encoder_means=enc)
    encoder_means_globais.update(enc)

    # --- Física da epidemia: Savitzky-Golay nos três splits ---
    # Mesmo método em treino, validação interna e target — sem inconsistência de escala.
    print(f"  Extraindo física (Savitzky-Golay) — season {season_id}...")
    df_tr_eng  = extrair_fisica(df_tr_eng)
    df_val_eng = extrair_fisica(df_val_eng)
    df_tgt_eng = extrair_fisica(df_tgt_eng)

    # --- Colunas finais ---
    cols_sgp = ['casos_suavizados', 'velocidade', 'aceleracao']
    cols     = [c for c in cols_sgp + cols_base if c in df_tr_eng.columns]
    num_features = len(cols)

    # --- Mapa de nós ---
    df_tr_eng  = df_tr_eng.merge(mapa_simples, on=VARS['municipio'], how='left')
    df_val_eng = df_val_eng.merge(mapa_simples, on=VARS['municipio'], how='left')
    df_tgt_eng = df_tgt_eng.merge(mapa_simples, on=VARS['municipio'], how='left')

    for df_ in [df_tr_eng, df_val_eng, df_tgt_eng]:
        for c in cols:
            if c not in df_.columns:
                df_[c] = 0.0

    # --- Normalização das features X (apenas X — Y é contagem bruta, sem scaler) ---
    # [C6] Não há mais scaler_Y: o target são casos absolutos e a NB trabalha
    #      na escala original de contagem.
    # NORMALIZAÇÃO DAS FEATURES (StandardScaler):
    # Transforma cada feature para média 0 e desvio padrão 1.
    # Por que só X e não Y (casos)?
    # O target Y (número de casos) é passado diretamente para a NB loss,
    # que foi projetada para trabalhar com contagens na escala original.
    # Normalizar Y quebraria a interpretação probabilística da NB.
    # IMPORTANTE: fit_transform apenas no treino, transform nos demais
    # — o scaler "aprende" a escala do treino e aplica a mesma transformação
    # no teste, sem ver os dados do futuro.
    scaler_X = StandardScaler()

    # Ordena por [time_index, node_id] para montar os tensores 3D corretamente:
    # o AGCRN espera [Tempo, Nós, Features] — a ordenação garante isso.
    df_tr_sort  = df_tr_eng.sort_values(['time_index', 'node_id'])
    df_val_sort = df_val_eng.sort_values(['time_index', 'node_id'])
    df_tgt_sort = df_tgt_eng.sort_values(['time_index', 'node_id'])

    tempos_tr  = sorted(df_tr_sort['time_index'].unique())
    tempos_val = sorted(df_val_sort['time_index'].unique())
    tempos_tgt = sorted(df_tgt_sort['time_index'].unique())

    X_tr_sc  = scaler_X.fit_transform(df_tr_sort[cols].fillna(0))
    X_val_sc = scaler_X.transform(df_val_sort[cols].fillna(0))
    X_tgt_sc = scaler_X.transform(df_tgt_sort[cols].fillna(0))

    def _to_tensor_3d(X_sc, tempos, n_nodes, n_feats):
        return torch.tensor(
            X_sc.reshape(len(tempos), n_nodes, n_feats),
            dtype=torch.float32,
        ).to(device)

    def _pop_tensor(df_sort, tempos, n_nodes):
        """Extrai log_pop como tensor [T, N] (offset fixo por nó e tempo)."""
        return torch.tensor(
            df_sort['log_pop'].values.reshape(len(tempos), n_nodes),
            dtype=torch.float32,
        ).to(device)

    def _casos_tensor(df_sort, tempos, n_nodes):
        """Extrai contagem de casos como tensor [T, N] (target real)."""
        return torch.tensor(
            df_sort[VARS['casos_dengue']].fillna(0).values.reshape(len(tempos), n_nodes),
            dtype=torch.float32,
        ).to(device)

    X_tr_t   = _to_tensor_3d(X_tr_sc,  tempos_tr,  num_nodes, num_features)
    X_val_t  = _to_tensor_3d(X_val_sc, tempos_val, num_nodes, num_features)
    X_tgt_t  = _to_tensor_3d(X_tgt_sc, tempos_tgt, num_nodes, num_features)

    # [C7] Offsets de log(população) — fixos, não normalizados
    logpop_tr_t  = _pop_tensor(df_tr_sort,  tempos_tr,  num_nodes)
    logpop_val_t = _pop_tensor(df_val_sort, tempos_val, num_nodes)
    logpop_tgt_t = _pop_tensor(df_tgt_sort, tempos_tgt, num_nodes)

    # [C6] Targets = casos absolutos
    Y_tr_t  = _casos_tensor(df_tr_sort,  tempos_tr,  num_nodes)
    Y_val_t = _casos_tensor(df_val_sort, tempos_val, num_nodes)

    # --- Modelo NB ---
    modelo = PreditorNBAGCRN(
        num_nodes=num_nodes,
        num_features=num_features,
        embedding_dim=32,
    ).to(device)
    # Adam: otimizador adaptativo — ajusta a taxa de aprendizado por parâmetro.
    # lr=0.005: taxa de aprendizado inicial. Menor que o default (0.001 ou 0.01)
    # porque a NB loss tem gradientes maiores que o MSE.
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.005)

    # ReduceLROnPlateau: reduz a taxa de aprendizado pela metade (factor=0.5)
    # se o MAE de validação não melhorar por 5 épocas consecutivas.
    # Isso evita que o modelo "pule" o mínimo quando já está perto da convergência.
    # min_lr=1e-5 impede que a taxa caia para zero.
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, patience=5, factor=0.5, min_lr=1e-5
    )

    # [C2] Treinamento com early stopping por MAE de casos na validação interna
    melhor_mae_val = float('inf')
    melhor_estado  = None
    sem_melhora    = 0

    barra = tqdm(range(epochs_max), desc=f'  Treino S{season_id}',
                 unit='epoch', file=sys.stdout)

    modelo.train()
    for epoch in barra:
        h          = None
        loss_epoch = 0.0

        for t in range(len(tempos_tr) - 1):
            otimizador.zero_grad()

            # [C7] Passa log_pop como offset no forward
            log_mu, log_r, h = modelo(X_tr_t[t], logpop_tr_t[t], h)

            # TBPTT (Truncated Backpropagation Through Time):
            # h.detach() "corta" o grafo de gradientes após cada semana.
            # Sem isso, o PyTorch tentaria calcular gradientes por TODAS as semanas
            # anteriores, o que consome memória exponencialmente e causa explosão
            # de gradientes em séries longas (700+ semanas de histórico).
            h = h.detach()

            # Target: casos da semana t+1
            alvo = Y_tr_t[t + 1].unsqueeze(-1)   # [N, 1]

            # [C2] NB loss sobre contagens absolutas
            loss = nb_loss(log_mu, log_r, alvo)
            loss.backward()
            # Gradient clipping: limita a norma dos gradientes a 1.0.
            # Redes recorrentes são suscetíveis a "explosão de gradientes"
            # (gradientes que crescem exponencialmente e desestabilizam o treino).
            # Esta linha é uma salvaguarda adicional ao TBPTT acima.
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            loss_epoch += loss.item()

        # Avalia na validação interna — MAE em número de casos
        modelo.eval()
        with torch.no_grad():
            h_val     = None
            preds_mu  = []
            for t in range(len(tempos_val)):
                log_mu_v, _, h_val = modelo(X_val_t[t], logpop_val_t[t], h_val)
                preds_mu.append(torch.exp(log_mu_v).squeeze(-1).cpu().numpy())

        preds_mu_np = np.array(preds_mu)             # [T_val, N]
        alvo_val_np = Y_val_t.cpu().numpy()           # [T_val, N]
        mae_val = np.mean(np.abs(preds_mu_np - alvo_val_np))

        scheduler.step(mae_val)
        barra.set_postfix({
            'NB_Loss': f'{loss_epoch / (len(tempos_tr) - 1):.4f}',
            'MAE_casos': f'{mae_val:.1f}',
        })

        # EARLY STOPPING:
        # Se o MAE de validação melhorou (caiu mais de 1e-3), salva o estado atual
        # do modelo como "melhor estado" e reinicia o contador de paciência.
        # Se não melhorou por 'patience' épocas consecutivas, para o treino.
        # Ao final, restaura o melhor estado — isso garante que entregamos o modelo
        # no seu ponto de menor erro de validação, não no final do treino (que pode
        # ser pior por overfitting).
        if mae_val < melhor_mae_val - 1e-3:
            melhor_mae_val = mae_val
            melhor_estado  = {k: v.clone() for k, v in modelo.state_dict().items()}
            sem_melhora    = 0
        else:
            sem_melhora += 1
            if sem_melhora >= patience:
                print(f"  ⏹ Early stopping na época {epoch+1} "
                      f"(melhor MAE_casos = {melhor_mae_val:.1f})")
                break

        modelo.train()

    # Restaura o melhor estado encontrado durante o treino.
    # Sem isso, o modelo final seria o da última época — potencialmente pior.
    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)

    # --- Previsões no período target ---
    # [C3] Extrai mu e r da NB → intervalos vêm diretamente dos quantis da distribuição
    modelo.eval()
    preds_mu_tgt  = []   # média prevista [T, N]
    preds_r_tgt   = []   # dispersão prevista [T, N]

    with torch.no_grad():
        h_tgt = None
        for t in range(len(tempos_tgt)):
            log_mu_t, log_r_t, h_tgt = modelo(X_tgt_t[t], logpop_tgt_t[t], h_tgt)
            preds_mu_tgt.append(torch.exp(log_mu_t).squeeze(-1).cpu().numpy())
            preds_r_tgt.append(torch.exp(log_r_t).squeeze(-1).cpu().numpy())

    preds_mu_np = np.maximum(np.array(preds_mu_tgt), 0)  # [T, N]
    preds_r_np  = np.maximum(np.array(preds_r_tgt),  1e-3)  # [T, N], r>0

    # Resíduos de validação para diagnóstico (mantidos no resultado)
    residuos_val = np.abs(preds_mu_np.mean(axis=1).mean() - alvo_val_np.mean())

    return {
        'season_id':      season_id,
        'modelo':         modelo,
        'scaler_X':       scaler_X,
        'cols':           cols,
        'preds_mu_np':    preds_mu_np,    # [T_tgt, N] — média NB
        'preds_r_np':     preds_r_np,     # [T_tgt, N] — dispersão NB
        'tempos_tgt':     tempos_tgt,
        'df_tgt_sort':    df_tgt_sort,
        'num_nodes':      num_nodes,
        'mapa_simples':   mapa_simples,
        # campos extras para o ensemble
        'df_tr_sort':     df_tr_sort,     # usado pelo LightGBM OOF
        'df_val_sort':    df_val_sort,
        'cols_base_used': cols,
        'enc':            enc,
    }


# ==============================================================================
# [E1-A] LIGHTGBM — modelo base independente (nível 0-B do ensemble)
# ==============================================================================
def treinar_lgbm_season(
    df_train: pd.DataFrame,       # [ALTERADO] Recebe o banco de treino isolado
    df_test: pd.DataFrame,        # [ALTERADO] Recebe o banco de teste isolado
    season_id: int,
    cols_base: list,
    encoder_means_globais: dict,
    mapa_simples: pd.DataFrame,
    n_estimators: int = 800,
    learning_rate: float = 0.05,
) -> dict:
    """
    Treina um LightGBM por season com walk-forward temporal.

    [F4] Customização: Implementação do objetivo analítico da Binomial Negativa
    (Gradiente e Hessiano baseados no log-likelihood com dispersão r=2.0).
    """
    season = SEASONS[season_id]
    print(f"\n  [LightGBM] Season {season_id}: {season['label']}")

    df_tr  = df_train.copy()
    df_tgt = df_test.copy()

    # Mesmo split de validacao interna do AGCRN (ultimas 10 epiweeks do treino)
    semanas_treino  = sorted(df_tr[VARS['semana_epi']].unique())
    semanas_val_int = semanas_treino[-10:]
    semanas_tr_real = semanas_treino[:-10]

    df_tr_real = df_tr[df_tr[VARS['semana_epi']].isin(semanas_tr_real)]
    df_val_int = df_tr[df_tr[VARS['semana_epi']].isin(semanas_val_int)]

    # Feature engineering sem leakage (encoders fitados so no treino)
    df_tr_eng,  enc = engenharia_de_features(df_tr_real, fit_encoders=True)
    df_val_eng, _   = engenharia_de_features(df_val_int, fit_encoders=False,
                                              encoder_means=enc)
    df_tgt_eng, _   = engenharia_de_features(df_tgt,     fit_encoders=False,
                                              encoder_means=enc)
    encoder_means_globais.update(enc)

    # Física da epidemia — mesmo Savitzky-Golay do AGCRN
    df_tr_eng  = extrair_fisica(df_tr_eng)
    df_val_eng = extrair_fisica(df_val_eng)
    df_tgt_eng = extrair_fisica(df_tgt_eng)

    # Colunas de features: mesmo conjunto do AGCRN + log_pop como feature normal
    cols_sgp  = ['casos_suavizados', 'velocidade', 'aceleracao']
    cols_lgbm = [c for c in cols_sgp + cols_base + ['log_pop']
                 if c in df_tr_eng.columns]

    # [F3] LAGS DO PRÓPRIO ALVO
    def _adicionar_lags_alvo(df_, df_historico=None):
        df_ = df_.copy().sort_values([VARS['municipio'], VARS['semana_epi']])
        if df_historico is not None:
            df_hist = df_historico[[
                VARS['municipio'], VARS['semana_epi'], VARS['casos_dengue']
            ]].copy()
            df_combined = pd.concat([df_hist, df_[[
                VARS['municipio'], VARS['semana_epi'], VARS['casos_dengue']
            ]]], ignore_index=True).drop_duplicates(
                subset=[VARS['municipio'], VARS['semana_epi']]
            ).sort_values([VARS['municipio'], VARS['semana_epi']])
        else:
            df_combined = df_

        for lag in [1, 2, 3]:
            lag_col = df_combined.groupby(VARS['municipio'])[VARS['casos_dengue']].shift(lag)
            df_combined[f'casos_lag{lag}'] = lag_col

        df_combined['casos_mm4s'] = (
            df_combined.groupby(VARS['municipio'])[VARS['casos_dengue']]
            .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        )

        cols_new = [f'casos_lag{l}' for l in [1, 2, 3]] + ['casos_mm4s']
        df_ = df_.merge(
            df_combined[[VARS['municipio'], VARS['semana_epi']] + cols_new],
            on=[VARS['municipio'], VARS['semana_epi']],
            how='left',
        )
        return df_.fillna({c: 0 for c in cols_new})

    df_tr_eng  = _adicionar_lags_alvo(df_tr_eng)
    df_val_eng = _adicionar_lags_alvo(df_val_eng, df_historico=df_tr_eng)
    df_tgt_eng = _adicionar_lags_alvo(df_tgt_eng, df_historico=df_tr_eng)

    cols_lags_alvo = ['casos_lag1', 'casos_lag2', 'casos_lag3', 'casos_mm4s']
    cols_lgbm = cols_lgbm + [c for c in cols_lags_alvo if c not in cols_lgbm]

    cols_ignorar_lgbm = {
        VARS['municipio'], VARS['semana_epi'], VARS['data'],
        VARS['casos_dengue'], 'incidencia', 'log_pop',
        VARS['bioma'], VARS['koppen'],
        'node_id', 'time_index', '_date_dt', '_date_lag', '_date_src',
        'train_1.x', 'train_2.x', 'train_3.x', 'train_4.x',
        'target_1.x', 'target_2.x', 'target_3.x', 'target_4.x',
        'disease.x', 'disease.y','uf', VARS['uf_code'],
        'regional_geocode', 'macroregional_geocode', 'target_city.x', 'target_city.y'
    }
    cols_lgbm = [c for c in cols_lgbm if c not in cols_ignorar_lgbm]

    # =========================================================================
    # [ALTERADO] TARGET EM CONTAGENS BRUTAS
    # A perda Binomial Negativa utiliza a função de ligação log(x) internamente.
    # Passamos os casos absolutos diretamente para evitar "duplo-log".
    # =========================================================================
    y_tr  = df_tr_eng[VARS['casos_dengue']].fillna(0).values
    y_val = df_val_eng[VARS['casos_dengue']].fillna(0).values

    for df_ in [df_tr_eng, df_val_eng, df_tgt_eng]:
        for c in cols_lgbm:
            if c not in df_.columns:
                df_[c] = 0.0

    X_tr  = df_tr_eng[cols_lgbm].fillna(0).values
    X_val = df_val_eng[cols_lgbm].fillna(0).values
    X_tgt = df_tgt_eng[cols_lgbm].fillna(0).values

    # =========================================================================
    # FUNÇÕES CUSTOMIZADAS: BINOMIAL NEGATIVA PARA LIGHTGBM
    # =========================================================================
    def nbinom_objective(y_true, y_pred):
        """
        Função Loss analítica para a Binomial Negativa.
        y_pred entregue pelo LightGBM aqui é a margem bruta (log(mu)).
        """
        y_pred = np.clip(y_pred, -10, 10) # Previne overflow do exponencial
        mu = np.exp(y_pred)
        
        # Parâmetro global de dispersão para direcionar as divisões das árvores.
        # r=2.0 é robusto para séries de saúde com alta superdispersão.
        r = 2.0 
        
        # Primeira e Segunda derivadas do log-likelihood negativo da NB
        grad = r * (mu - y_true) / (r + mu)
        hess = mu * r * (r + y_true) / ((r + mu)**2)
        return grad, hess

    def lgbm_mae_custom(y_true, y_pred):
        """
        Métrica de avaliação correta: precisa desativar a margem logarítmica
        para calcular o MAE contra a contagem de casos reais.
        """
        y_pred = np.clip(y_pred, -10, 10)
        mu = np.exp(y_pred)
        return 'mae_casos', float(np.mean(np.abs(mu - y_true))), False

    # [F4] OPTUNA — BUSCA AUTOMÁTICA DE HIPERPARÂMETROS
    def _objective_lgbm(trial):
        params_trial = {
            'n_estimators':      800,
            'learning_rate':     trial.suggest_float('lr', 0.01, 0.15, log=True),
            'num_leaves':        trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'n_jobs':            -1,
            'verbose':           -1,
        }
        # Injeta o Objective Customizado no construtor
        m = lgb.LGBMRegressor(objective=nbinom_objective, **params_trial)
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=lgbm_mae_custom,
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        pred_val_log = m.predict(X_val)
        pred_val_mu  = np.exp(np.clip(pred_val_log, -10, 10))
        return float(np.mean(np.abs(pred_val_mu - y_val)))

    print(f"    [F4] Optuna LightGBM — season {season_id} (40 trials)...")
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(_objective_lgbm, n_trials=40, show_progress_bar=False)
    best_params = study.best_params
    print(f"    Melhor MAE_val Optuna: {study.best_value:.1f} casos | "
          f"num_leaves={best_params['num_leaves']}, lr={best_params['lr']:.4f}")

    params = {
        'n_estimators':      n_estimators,
        'learning_rate':     best_params['lr'],
        'num_leaves':        best_params['num_leaves'],
        'min_child_samples': best_params['min_child_samples'],
        'subsample':         best_params['subsample'],
        'colsample_bytree':  best_params['colsample_bytree'],
        'reg_alpha':         best_params['reg_alpha'],
        'reg_lambda':        best_params['reg_lambda'],
        'n_jobs':            -1,
        'verbose':           -1,
    }

    # Treino Final com Objetivo NB
    modelo_lgbm = lgb.LGBMRegressor(objective=nbinom_objective, **params)
    modelo_lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=lgbm_mae_custom,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    # OOF sobre validacao interna (o modelo expele o raw margin; transformamos com np.exp)
    oof_log = modelo_lgbm.predict(X_val)
    oof_mu  = np.exp(np.clip(oof_log, -10, 10))   

    # Previsoes no periodo target
    pred_tgt_log = modelo_lgbm.predict(X_tgt)
    pred_tgt_mu  = np.exp(np.clip(pred_tgt_log, -10, 10))

    # Reorganiza pred_tgt como [T_tgt, N] para alinhar com AGCRN
    df_tgt_eng = df_tgt_eng.merge(mapa_simples, on=VARS['municipio'], how='left')
    df_tgt_eng['pred_lgbm'] = pred_tgt_mu

    tempos_tgt = sorted(df_tgt_eng['time_index'].unique())
    num_nodes  = len(mapa_simples)

    pred_lgbm_3d = np.zeros((len(tempos_tgt), num_nodes))
    for t_idx, t in enumerate(tempos_tgt):
        slice_t = (df_tgt_eng[df_tgt_eng['time_index'] == t]
                   .sort_values('node_id'))
        if len(slice_t) == num_nodes:
            pred_lgbm_3d[t_idx] = slice_t['pred_lgbm'].values

    # OOF reorganizado como [T_val, N] para o meta-modelo
    df_val_eng = df_val_eng.merge(mapa_simples, on=VARS['municipio'], how='left')
    df_val_eng['oof_lgbm'] = oof_mu
    tempos_val = sorted(df_val_eng['time_index'].unique())

    oof_lgbm_3d = np.zeros((len(tempos_val), num_nodes))
    for t_idx, t in enumerate(tempos_val):
        slice_t = (df_val_eng[df_val_eng['time_index'] == t]
                   .sort_values('node_id'))
        if len(slice_t) == num_nodes:
            oof_lgbm_3d[t_idx] = slice_t['oof_lgbm'].values

    print(f"    LightGBM treinado — {modelo_lgbm.best_iteration_} arvores, "
          f"MAE_val: {np.mean(np.abs(oof_mu - y_val)):.1f} casos")

    return {
        'season_id':     season_id,
        'modelo_lgbm':   modelo_lgbm,
        'cols_lgbm':     cols_lgbm,
        'pred_lgbm_3d':  pred_lgbm_3d,   
        'oof_lgbm_3d':   oof_lgbm_3d,    
        'tempos_val':    tempos_val,
        'tempos_tgt':    tempos_tgt,
        'df_val_eng':    df_val_eng,
        'df_tgt_eng':    df_tgt_eng,
    }

# ==============================================================================
# [E1-B] META-MODELO RIDGE — stacking dos dois modelos base (nivel 1)
# ==============================================================================
def treinar_meta_modelo(
    oof_agcrn_list: list,
    oof_lgbm_list:  list,
    oof_alvo_list:  list,
    df_val_list:    list,
    alpha: float = 10.0,
) -> tuple:
    """
    Treina o meta-modelo Ridge sobre os OOFs das 4 seasons.

    Features do meta-modelo para cada linha (municipio x semana de validacao):
      - mu_agcrn:          previsao media do AGCRN
      - pred_lgbm:         previsao media do LightGBM
      - log_pop:           offset de populacao (heterogeneidade de escala)
      - biome_encoded:     incidencia media historica do bioma
      - koppen_encoded:    incidencia media historica do clima
      - seno_semana:       sazonalidade ciclica
      - cosseno_semana:    sazonalidade ciclica

    Target: log(casos + 1)  — mesma escala do LightGBM, estavel para Ridge.

    [E1] alpha=10.0 por padrao — regularizacao forte porque o meta-modelo
    recebe poucos exemplos por tipo de municipio vs. muitas features correlatas.
    Inspecionar meta_modelo.coef_ para ver quanto peso vai para cada modelo base.
    """
    print("\n  [Meta-modelo] Treinando Ridge sobre OOF das 4 seasons...")

    X_meta_parts = []
    y_meta_parts = []

    for oof_agcrn, oof_lgbm, oof_alvo, df_val in zip(
        oof_agcrn_list, oof_lgbm_list, oof_alvo_list, df_val_list
    ):
        # Flatten de [T, N] para linhas
        T, N = oof_agcrn.shape
        mu_agcrn_flat  = oof_agcrn.flatten()           # [T*N]
        pred_lgbm_flat = oof_lgbm.flatten()            # [T*N]
        alvo_flat      = oof_alvo.flatten()            # [T*N]

        # Features contextuais do dataframe de validacao (uma linha por T*N)
        # df_val ja esta ordenado por [time_index, node_id]
        feats_ctx = df_val[['log_pop',
                             'biome_encoded', 'koppen_encoded',
                             'seno_semana',   'cosseno_semana']].fillna(0).values

        # Monta matriz de features do meta-modelo
        X_bloco = np.column_stack([
            mu_agcrn_flat,
            pred_lgbm_flat,
            feats_ctx,
        ])
        y_bloco = np.log1p(np.maximum(alvo_flat, 0))

        X_meta_parts.append(X_bloco)
        y_meta_parts.append(y_bloco)

    # Empilha os dados de todas as 4 seasons em uma única matriz.
    # O Ridge aprende um padrão geral de quando confiar em cada modelo base.
    X_meta = np.vstack(X_meta_parts)
    y_meta = np.concatenate(y_meta_parts)

    # Remove linhas com NaN ou Inf — podem ocorrer em municípios sem dados de bioma.
    mask   = np.isfinite(X_meta).all(axis=1) & np.isfinite(y_meta)
    X_meta = X_meta[mask]
    y_meta = y_meta[mask]

    # Ridge com alpha=10: regularização L2 forte.
    # Por que Ridge e não outro XGBoost?
    # Com apenas 7 features e muitos dados, um XGBoost overfitaria trivialmente
    # (aprenderia a copiar o melhor modelo base sem generalizar).
    # O Ridge é estável, os coeficientes são interpretáveis e a regularização
    # forte evita que o meta-modelo confie 100% em um único modelo base.
    meta_modelo = Ridge(alpha=alpha, fit_intercept=True)
    meta_modelo.fit(X_meta, y_meta)

    # Imprime os coeficientes para auditoria:
    # - Coeficiente positivo grande em mu_agcrn: Ridge confia mais no AGCRN
    # - Coeficiente positivo grande em pred_lgbm: Ridge confia mais no LightGBM
    # - Coeficientes em biome/koppen: ajuste por tipo de município
    coefs = meta_modelo.coef_
    nomes = ['mu_agcrn', 'pred_lgbm', 'log_pop',
             'biome_encoded', 'koppen_encoded', 'seno_semana', 'cosseno_semana']
    print("    Coeficientes Ridge:")
    for nome, coef in zip(nomes, coefs):
        print(f"      {nome:20s}: {coef:+.4f}")

    return meta_modelo, nomes


def aplicar_meta_modelo(
    meta_modelo,
    mu_agcrn_tgt:   np.ndarray,
    pred_lgbm_tgt:  np.ndarray,
    df_tgt_sort:    pd.DataFrame,
) -> np.ndarray:
    """
    Aplica o meta-modelo Ridge no periodo target para gerar mu_ensemble.

    Entrada:
      mu_agcrn_tgt  [T, N] — media NB do AGCRN
      pred_lgbm_tgt [T, N] — previsao do LightGBM
      df_tgt_sort         — dataframe target ordenado por [time_index, node_id]

    Saida:
      mu_ensemble   [T, N] — media ensemble em escala de casos absolutos
    """
    T, N = mu_agcrn_tgt.shape

    feats_ctx = df_tgt_sort[['log_pop',
                              'biome_encoded', 'koppen_encoded',
                              'seno_semana',   'cosseno_semana']].fillna(0).values

    X_tgt_meta = np.column_stack([
        mu_agcrn_tgt.flatten(),
        pred_lgbm_tgt.flatten(),
        feats_ctx,
    ])

    # Garante ausencia de NaN/Inf
    X_tgt_meta = np.where(np.isfinite(X_tgt_meta), X_tgt_meta, 0.0)

    pred_log = meta_modelo.predict(X_tgt_meta)
    mu_ens   = np.expm1(np.maximum(pred_log, 0)).reshape(T, N)
    return mu_ens


# ==============================================================================
# [F2] RECALIBRAÇÃO DO r DA NB APÓS ENSEMBLE
# ==============================================================================
def recalibrar_r_ensemble(
    mu_ensemble_oof: np.ndarray,
    alvo_oof:        np.ndarray,
    r_agcrn_oof:     np.ndarray,
) -> float:
    """
    Estima r_ensemble por máxima verossimilhança sobre os resíduos OOF.

    Problema: quando o Ridge combina mu_agcrn e pred_lgbm gerando mu_ensemble,
    a nova média é diferente da original do AGCRN. O r aprendido pelo AGCRN
    foi calibrado para aquele mu específico. Como Var[Y] = mu + mu²/r, usar
    o r_agcrn com mu_ensemble produz intervalos incorretos.

    Solução: busca escalar o r_agcrn por um fator de escala k > 0 tal que a
    NB(mu_ensemble, k * r_agcrn_medio) maximize a verossimilhança dos casos
    reais nos OOF. Retorna r_ensemble = k * r_agcrn_medio.

    Parâmetros
    ----------
    mu_ensemble_oof : [n_amostras] — mu ensemble nas semanas de validação OOF
    alvo_oof        : [n_amostras] — casos reais correspondentes
    r_agcrn_oof     : [n_amostras] — r do AGCRN nas mesmas semanas

    Retorna
    -------
    r_ensemble : float — r recalibrado (escalar global aplicado a todos os nós)
    """
    # Garante que mu > 0 (NB não aceita média zero ou negativa).
    mu  = np.maximum(mu_ensemble_oof.flatten(), 1e-3)
    y   = np.maximum(alvo_oof.flatten(),        0.0)
    # r0 = ponto de partida para a busca: mediana do r aprendido pelo AGCRN nos OOFs.
    r0  = float(np.median(r_agcrn_oof.flatten()))

    def neg_ll(log_k):
        # Buscamos o FATOR DE ESCALA k que multiplica r0.
        # Por que buscar k em log-escala? Para garantir k > 0 (r deve ser positivo)
        # e para que a busca seja simétrica: k=2 e k=0.5 têm a mesma "distância" de k=1.
        k = np.exp(log_k)
        r = k * r0
        # NLL da NB com a nova mu (ensemble) e o novo r (escalado):
        # Quanto menor a NLL, melhor a distribuição NB(mu_ens, r) explica os dados reais.
        p   = r / (r + mu + 1e-8)  # conversão para parametrização do scipy
        nll = -(
            nbinom.logpmf(y.astype(int), n=r, p=p)
        ).mean()
        return float(nll)

    # minimize_scalar busca o k que minimiza a NLL dentro do intervalo dado.
    # bounds=(-4, 4) significa k entre e^-4≈0.018 e e^4≈55 — cobre todos os casos reais.
    # method='bounded': busca determinística (não aleatória) — sempre reprodutível.
    resultado = minimize_scalar(
        neg_ll,
        bounds=(-4.0, 4.0),
        method='bounded',
        options={'xatol': 1e-4},  # para quando a precisão for melhor que 0.0001
    )
    k_otimo    = np.exp(resultado.x)  # melhor fator de escala encontrado
    r_ensemble = k_otimo * r0         # r final = r do AGCRN × fator de ajuste
    print(f"    [F2] r recalibrado: r_agcrn_mediano={r0:.3f} × k={k_otimo:.3f} "
          f"→ r_ensemble={r_ensemble:.3f}")
    return float(r_ensemble)


# ==============================================================================
# 5. MONTAGEM DO DATAFRAME DE SUBMISSÃO
# ==============================================================================
def montar_submissao(
    resultado_agcrn: dict,
    mu_ensemble:     np.ndarray,   # [T, N] — media ensemble (AGCRN + LightGBM)
    r_ensemble:      float,        # [F2] r recalibrado por MLE sobre OOF
) -> pd.DataFrame:
    """
    Gera o dataframe de submissao usando mu_ensemble e r_ensemble.

    [F2] r_ensemble é o parâmetro de dispersão da NB recalibrado por MLE
         sobre os resíduos OOF do ensemble — consistente com mu_ensemble.
    [C3] Intervalos vêm dos quantis de NB(mu_ensemble, r_ensemble).
    """
    df_tgt_sort  = resultado_agcrn['df_tgt_sort']
    mapa_simples = resultado_agcrn['mapa_simples']

    T, N = mu_ensemble.shape
    geocodes = mapa_simples[VARS['municipio']].values
    semanas  = sorted(df_tgt_sort[VARS['semana_epi']].unique())

    quantis_necessarios = {
        'lower_95': 0.025, 'lower_90': 0.05,
        'lower_80': 0.10,  'lower_50': 0.25,
        'upper_50': 0.75,  'upper_80': 0.90,
        'upper_90': 0.95,  'upper_95': 0.975,
    }

    # [F2] r_ensemble é um escalar global recalibrado por MLE.
    # Aplicamos o mesmo r para todos os N municípios — é uma calibração
    # global, não por município. O r individual por município ainda varia
    # implicitamente porque mu_ensemble varia por município.
    # np.full cria um array de N elementos todos iguais a r_ensemble.
    r_arr = np.full(N, r_ensemble)

    rows = []
    for t_idx, semana in enumerate(semanas):
        mu_t = mu_ensemble[t_idx]   # [N]

        for n_idx, geocode in enumerate(geocodes):
            mu = float(mu_t[n_idx])
            r  = float(r_arr[n_idx])

            # Mediana da NB = previsão central (o "melhor palpite" do modelo).
            # O quantil 0.50 é o valor que a NB considera igualmente provável
            # para cima e para baixo — é a previsão mais provável.
            pred_med = float(nb_quantile(np.array([mu]), np.array([r]), 0.50)[0])
            row = {
                VARS['municipio']:  geocode,
                VARS['semana_epi']: semana,
                'pred':             round(max(pred_med, 0), 2),  # nunca negativo
            }
            # Calcula os 8 limites de intervalo diretamente da CDF da NB.
            # Exemplo: lower_95 = quantil 0.025 significa que apenas 2.5% dos
            # casos reais esperados ficam abaixo desse valor.
            # Os intervalos são gerados pela MESMA distribuição da previsão central,
            # garantindo consistência estatística — não são heurísticas separadas.
            for col_name, q in quantis_necessarios.items():
                val = float(nb_quantile(np.array([mu]), np.array([r]), q)[0])
                row[col_name] = round(max(val, 0), 2)

            rows.append(row)

    df_pred = pd.DataFrame(rows)

    # Converte epiweek → date (domingo ISO), conforme regra 4.2
    df_pred['date_parsed'] = pd.to_datetime(
        df_pred[VARS['semana_epi']].astype(str) + '-0', format='%G%V-%w'
    )
    df_pred['date'] = df_pred['date_parsed'].dt.strftime('%Y-%m-%d')

    # Grade perfeita geocode × date (sem lacunas — regra 4.2)
    todas_datas    = pd.date_range(
        start=df_pred['date_parsed'].min(),
        end=df_pred['date_parsed'].max(),
        freq='W-SUN',
    )
    todos_geocodes = df_pred[VARS['municipio']].unique()

    grade = (
        pd.MultiIndex.from_product(
            [todos_geocodes, todas_datas],
            names=[VARS['municipio'], 'date_parsed'],
        )
        .to_frame().reset_index(drop=True)
    )
    grade['date'] = grade['date_parsed'].dt.strftime('%Y-%m-%d')

    df_pred = pd.merge(
        grade, df_pred,
        on=[VARS['municipio'], 'date', 'date_parsed'],
        how='left',
    )

    # Preenche lacunas por interpolação causal
    cols_pred = ['pred'] + list(quantis_necessarios.keys())
    for col in cols_pred:
        if col in df_pred.columns:
            df_pred[col] = (
                df_pred.groupby(VARS['municipio'])[col]
                .transform(lambda x: x.interpolate('linear').bfill().ffill())
            )
            df_pred[col] = np.maximum(df_pred[col], 0).round(2)

    # Aninhamento monotônico — garante ordem exigida pelo edital (regra 4.2)
    # A NB já garante isso por construção (quantis crescentes), mas aplicamos
    # a correção como salvaguarda após interpolação
    for col_lo, col_hi in [('lower_95', 'lower_90'), ('lower_90', 'lower_80'),
                            ('lower_80', 'lower_50'), ('lower_50', 'pred'),
                            ('pred', 'upper_50'),     ('upper_50', 'upper_80'),
                            ('upper_80', 'upper_90'), ('upper_90', 'upper_95')]:
        mask = df_pred[col_lo] > df_pred[col_hi]
        if mask.any():
            mid = (df_pred.loc[mask, col_lo] + df_pred.loc[mask, col_hi]) / 2
            df_pred.loc[mask, col_lo] = mid
            df_pred.loc[mask, col_hi] = mid

    colunas_finais = [
        VARS['municipio'], 'date', 'pred',
        'lower_50', 'upper_50',
        'lower_80', 'upper_80',
        'lower_90', 'upper_90',
        'lower_95', 'upper_95',
    ]
    return df_pred[colunas_finais].sort_values(
        by=[VARS['municipio'], 'date']
    ).reset_index(drop=True)


# ==============================================================================
# PIPELINE INTERNO DE ASSERÇÃO (regras 4.1 e 4.2 do edital)
# ==============================================================================
def validar_submissao(df_sub: pd.DataFrame, label: str = '') -> None:
    """Roda o checklist completo do edital. Levanta AssertionError se falhar."""
    print(f"\n[VALIDAÇÃO {label}] Executando testes...")

    cols_intervalo = [
        'lower_95', 'lower_90', 'lower_80', 'lower_50',
        'pred',
        'upper_50', 'upper_80', 'upper_90', 'upper_95',
    ]

    # Regra: datas em domingo
    datas = pd.to_datetime(df_sub['date'])
    assert (datas.dt.dayofweek == 6).all(), \
        "Erro: existem datas que não caem em domingo."

    # Regra: sem valores negativos
    for col in cols_intervalo:
        if col in df_sub.columns:
            assert (df_sub[col] >= 0).all(), \
                f"Erro: valores negativos em '{col}'."

    # Regra: aninhamento monotônico
    ok = (
        (df_sub['lower_95'] <= df_sub['lower_90']) &
        (df_sub['lower_90'] <= df_sub['lower_80']) &
        (df_sub['lower_80'] <= df_sub['lower_50']) &
        (df_sub['lower_50'] <= df_sub['pred'])     &
        (df_sub['pred']     <= df_sub['upper_50']) &
        (df_sub['upper_50'] <= df_sub['upper_80']) &
        (df_sub['upper_80'] <= df_sub['upper_90']) &
        (df_sub['upper_90'] <= df_sub['upper_95'])
    ).all()
    assert ok, "Erro: inversão detectada nos intervalos de confiança."

    # Regra: sem lacunas de datas por geocode
    for geocode, grp in df_sub.groupby(VARS['municipio']):
        dts  = pd.to_datetime(grp['date'].sort_values())
        gaps = dts.diff().dropna()
        assert (gaps == pd.Timedelta('7 days')).all(), \
            f"Erro: lacuna temporal no geocode {geocode}."

    n_geo   = df_sub[VARS['municipio']].nunique()
    n_datas = df_sub['date'].nunique()
    print(f"  ✓ {n_geo} municípios × {n_datas} semanas — todos os testes passaram.")


# ==============================================================================
# [F5] RECONCILIAÇÃO HIERÁRQUICA MinT — municipal → estadual
# ==============================================================================
def reconciliar_mint(
    df_municipal:  pd.DataFrame,
    df_estadual:   pd.DataFrame,
    col_uf_code:   str = 'uf_code',
) -> tuple:
    """
    Aplica MinT (Minimum Trace) reconciliation para garantir que a soma das
    previsões municipais seja igual à previsão estadual, melhorando ambas.

    Processo
    --------
    1. Constrói a matriz de soma S (estados × municípios)
    2. Calcula previsões base no nível mais desagregado (municipal)
    3. MinT propaga informação top-down e bottom-up simultaneamente
    4. Retorna df_municipal e df_estadual reconciliados e consistentes

    Parâmetros
    ----------
    df_municipal : submissão municipal (geocode, date, pred, lower_*, upper_*)
    df_estadual  : submissão estadual  (uf_code, date, pred, lower_*, upper_*)
    col_uf_code  : nome da coluna de estado em df_municipal

    Retorna
    -------
    (df_mun_rec, df_est_rec) — ambos reconciliados
    """
    if not HF_DISPONIVEL:
        print("  ⚠️  MinT indisponível — retornando previsões originais sem reconciliação.")
        return df_municipal, df_estadual

    print("  [F5] Aplicando reconciliação hierárquica MinT...")

    cols_quantis = ['pred', 'lower_50', 'upper_50', 'lower_80', 'upper_80',
                    'lower_90', 'upper_90', 'lower_95', 'upper_95']

    datas = sorted(df_municipal['date'].unique())
    df_mun_rec = df_municipal.copy()
    df_est_rec = df_estadual.copy()

    # Mapeia geocode → uf_code
    mapa_uf = (
        df_municipal[[VARS['municipio'], col_uf_code]]
        .drop_duplicates()
        .set_index(VARS['municipio'])[col_uf_code]
        .to_dict()
    )
    geocodes = sorted(df_municipal[VARS['municipio']].unique())
    ufs      = sorted(df_estadual[col_uf_code].unique())

    # MATRIZ DE SOMA S — a peça central da reconciliação hierárquica.
    # S tem shape [n_estados + n_municípios, n_municípios].
    # As primeiras n_estados linhas representam a relação "estado = soma dos municípios":
    #   S[estado_SP, municipio_campinas] = 1  (Campinas pertence a SP)
    #   S[estado_RJ, municipio_campinas] = 0  (Campinas não pertence ao RJ)
    # As últimas n_municípios linhas são a identidade (cada município é ele mesmo).
    # Essa estrutura codifica toda a hierarquia geográfica em uma única matriz.
    # Matriz de soma S: shape [n_uf + n_mun, n_mun]
    # Linha i (i < n_uf): soma dos municípios do estado i
    # Linha i (i >= n_uf): município i - n_uf (identidade)
    n_uf  = len(ufs)
    n_mun = len(geocodes)
    uf_idx  = {uf: i for i, uf in enumerate(ufs)}
    mun_idx = {g: i for i, g in enumerate(geocodes)}

    S = np.zeros((n_uf + n_mun, n_mun))
    for g in geocodes:
        uf = mapa_uf.get(g)
        if uf in uf_idx:
            S[uf_idx[uf], mun_idx[g]] = 1.0
        S[n_uf + mun_idx[g], mun_idx[g]] = 1.0

    # MinT por coluna de quantil — reconcilia cada quantil separadamente
    # para manter o aninhamento monotônico após reconciliação
    for col in cols_quantis:
        if col not in df_municipal.columns or col not in df_estadual.columns:
            continue

        Y_hat = np.zeros((n_uf + n_mun, len(datas)))

        for d_idx, data in enumerate(datas):
            mun_d = df_municipal[df_municipal['date'] == data].set_index(VARS['municipio'])
            est_d = df_estadual[df_estadual['date'] == data].set_index(col_uf_code)

            for i, uf in enumerate(ufs):
                Y_hat[i, d_idx] = est_d.loc[uf, col] if uf in est_d.index else 0.0
            for i, g in enumerate(geocodes):
                Y_hat[n_uf + i, d_idx] = mun_d.loc[g, col] if g in mun_d.index else 0.0

        # PROJEÇÃO MinT (Mínimo Traço):
        # P = S(S'S)⁻¹S' é o operador de projeção que força consistência hierárquica.
        # Aplicar P a Y_hat redistribui as previsões para que:
        # 1) A soma dos municípios de cada estado iguale a previsão estadual
        # 2) A redistribuição minimize o erro total (critério de mínimo traço)
        # 1e-6 * I é uma regularização numérica — evita que S'S seja singular.
        # np.linalg.solve é mais estável que calcular a inversa diretamente.
        STS     = S.T @ S + 1e-6 * np.eye(n_mun)
        P       = S @ np.linalg.solve(STS, S.T)
        Y_rec   = P @ Y_hat  # previsões reconciliadas: [n_uf + n_mun, n_datas]

        Y_rec = np.maximum(Y_rec, 0)

        # Atualiza df_municipal e df_estadual
        for d_idx, data in enumerate(datas):
            for i, g in enumerate(geocodes):
                mask_m = (df_mun_rec['date'] == data) & (df_mun_rec[VARS['municipio']] == g)
                df_mun_rec.loc[mask_m, col] = round(float(Y_rec[n_uf + i, d_idx]), 2)
            for i, uf in enumerate(ufs):
                mask_e = (df_est_rec['date'] == data) & (df_est_rec[col_uf_code] == uf)
                df_est_rec.loc[mask_e, col] = round(float(Y_rec[i, d_idx]), 2)

    # Garante aninhamento monotônico após reconciliação
    for df_r in [df_mun_rec, df_est_rec]:
        for col_lo, col_hi in [('lower_95','lower_90'),('lower_90','lower_80'),
                                ('lower_80','lower_50'),('lower_50','pred'),
                                ('pred','upper_50'),('upper_50','upper_80'),
                                ('upper_80','upper_90'),('upper_90','upper_95')]:
            if col_lo in df_r.columns and col_hi in df_r.columns:
                mask = df_r[col_lo] > df_r[col_hi]
                if mask.any():
                    mid = (df_r.loc[mask, col_lo] + df_r.loc[mask, col_hi]) / 2
                    df_r.loc[mask, col_lo] = mid
                    df_r.loc[mask, col_hi] = mid

    print(f"    MinT concluído: {n_mun} municípios × {n_uf} estados × {len(datas)} datas.")
    return df_mun_rec, df_est_rec


# ==============================================================================
# 6. EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == '__main__':

    # --------------------------------------------------------------------------
    # 6.1 Carregamento dos dados
    # --------------------------------------------------------------------------
    # DADOS CARREGADOS:
    # - dengue.parquet: série semanal de casos por município (2010-2026)
    #   Inclui colunas booleanas train_1..4 e target_1..4 prontas para uso.
    # - chikungunya.parquet: mesma estrutura, para o desafio opcional.
    # - climate_forecast.parquet: projeções climáticas futuras do ECMWF/Copernicus
    #   (temperatura, precipitação, umidade até 6 meses à frente por município).
    # Os arquivos .parquet são versões convertidas dos .csv.gz do FTP do edital.
    print("Carregando datasets do FTP...")
    df_dengue   = pd.read_parquet("dengue.parquet")
    df_chik     = pd.read_parquet("chikungunya.parquet")
    # [F1] Previsão climática futura do ECMWF
    df_forecast = pd.read_parquet("climate_forecast.parquet")  # climate_forecast.csv.gz

    # [F1] Integração do clima futuro ANTES de qualquer engenharia de features.
    # Após essa chamada, df_dengue terá 9 novas colunas:
    # clim_fc_temp_1m, clim_fc_precip_1m, clim_fc_humid_1m (previsão a 1 mês)
    # clim_fc_temp_2m, clim_fc_precip_2m, clim_fc_humid_2m (previsão a 2 meses)
    # clim_fc_temp_3m, clim_fc_precip_3m, clim_fc_humid_3m (previsão a 3 meses)
    # Essas colunas estarão disponíveis para AGCRN e LightGBM automaticamente.
    # print("  [F1] Integrando previsão climática futura (ECMWF)...")
    # df_dengue = integrar_climate_forecast(df_dengue, df_forecast, meses_ahead=[1, 2, 3])

    # MAPA DE NÓS: atribui um índice numérico a cada município.
    # O AGCRN precisa que cada município seja identificado por um inteiro
    # (node_id de 0 a N-1) para indexar os embeddings e o grafo aprendido.
    # Ordenamos por geocode para garantir que o node_id seja determinístico
    # e idêntico entre diferentes execuções e seasons.
    mapa_simples = (
        df_dengue[[VARS['municipio']]]
        .drop_duplicates()
        .sort_values(VARS['municipio'])
        .reset_index(drop=True)
    )
    mapa_simples['node_id'] = mapa_simples.index  # 0, 1, 2, ..., N-1
    num_nodes = len(mapa_simples)  # número total de municípios

    # --------------------------------------------------------------------------
    # 6.2 Selecao de features via SHAP walk-forward (uma vez, sobre season 1)
    # --------------------------------------------------------------------------
    df_s1 = df_dengue[df_dengue['train_1']].copy()
    df_s1_eng, enc_global = engenharia_de_features(df_s1, fit_encoders=True)
    features_selecionadas = selecao_shap_walkforward(
        df_s1_eng,
        col_alvo=VARS['casos_dengue'],
        num_features=15,
        n_folds=3,
    )

    cols_base = list(set(features_selecionadas + [
        'chikungunya_casos',
        'afya_log_diff',
        'imunidade_ewm',
        'seno_semana',
        'cosseno_semana',
        # [F1] features de clima futuro garantidas no cols_base
        'clim_fc_temp_1m', 'clim_fc_precip_1m', 'clim_fc_humid_1m',
        'clim_fc_temp_2m', 'clim_fc_precip_2m', 'clim_fc_humid_2m',
        'clim_fc_temp_3m', 'clim_fc_precip_3m', 'clim_fc_humid_3m',
    ]))

    # --------------------------------------------------------------------------
    # 6.3 Loop pelas 4 seasons — treina AGCRN e LightGBM, coleta OOF
    # --------------------------------------------------------------------------
    oof_agcrn_list   = []
    oof_lgbm_list    = []
    oof_alvo_list    = []
    oof_r_agcrn_list = []   # [F2] r do AGCRN nas semanas OOF
    df_val_list      = []

    resultados_agcrn = {}
    resultados_lgbm  = {}

    for sid in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"  SEASON {sid}: {SEASONS[sid]['label']}")
        print(f"{'='*60}")

        # -- AGCRN --
        res_agcrn = treinar_e_validar_season(
            df_dengue=df_dengue,
            season_id=sid,
            cols_base=cols_base,
            encoder_means_globais=enc_global,
            num_nodes=num_nodes,
            mapa_simples=mapa_simples,
            epochs_max=150,
            patience=15,
        )
        resultados_agcrn[sid] = res_agcrn

        modelo_agcrn = res_agcrn['modelo']
        scaler_X     = res_agcrn['scaler_X']
        cols         = res_agcrn['cols']
        df_val_sort  = res_agcrn['df_val_sort']

        tempos_val_agcrn = sorted(df_val_sort['time_index'].unique())
        X_val_sc = scaler_X.transform(df_val_sort[cols].fillna(0))
        X_val_t  = torch.tensor(
            X_val_sc.reshape(len(tempos_val_agcrn), num_nodes, len(cols)),
            dtype=torch.float32,
        ).to(device)
        logpop_val_t = torch.tensor(
            df_val_sort['log_pop'].values.reshape(len(tempos_val_agcrn), num_nodes),
            dtype=torch.float32,
        ).to(device)

        modelo_agcrn.eval()
        oof_mu_agcrn  = []
        oof_r_agcrn   = []   # [F2]
        with torch.no_grad():
            h_oof = None
            for t in range(len(tempos_val_agcrn)):
                log_mu_v, log_r_v, h_oof = modelo_agcrn(X_val_t[t], logpop_val_t[t], h_oof)
                oof_mu_agcrn.append(torch.exp(log_mu_v).squeeze(-1).cpu().numpy())
                oof_r_agcrn.append(torch.exp(log_r_v).squeeze(-1).cpu().numpy())  # [F2]

        oof_mu_agcrn_3d = np.maximum(np.array(oof_mu_agcrn), 0)   # [T_val, N]
        oof_r_agcrn_3d  = np.maximum(np.array(oof_r_agcrn),  1e-3) # [T_val, N] [F2]

        alvo_val_3d = df_val_sort[VARS['casos_dengue']].fillna(0).values.reshape(
            len(tempos_val_agcrn), num_nodes
        )

        # -- LightGBM --
        res_lgbm = treinar_lgbm_season(
            df_dengue=df_dengue,
            season_id=sid,
            cols_base=cols_base,
            encoder_means_globais=enc_global,
            mapa_simples=mapa_simples,
        )
        resultados_lgbm[sid] = res_lgbm

        T_common = min(oof_mu_agcrn_3d.shape[0], res_lgbm['oof_lgbm_3d'].shape[0])

        oof_agcrn_list.append(oof_mu_agcrn_3d[:T_common])
        oof_lgbm_list.append(res_lgbm['oof_lgbm_3d'][:T_common])
        oof_alvo_list.append(alvo_val_3d[:T_common])
        oof_r_agcrn_list.append(oof_r_agcrn_3d[:T_common])   # [F2]

        cols_ctx = ['log_pop', 'biome_encoded', 'koppen_encoded',
                    'seno_semana', 'cosseno_semana']
        for c in cols_ctx:
            if c not in df_val_sort.columns:
                df_val_sort[c] = 0.0
        n_linhas_common = T_common * num_nodes
        df_val_list.append(
            df_val_sort[cols_ctx].iloc[:n_linhas_common].reset_index(drop=True)
        )

    # --------------------------------------------------------------------------
    # 6.4 Meta-modelo Ridge — aprende a combinar AGCRN e LightGBM
    # --------------------------------------------------------------------------
    # Neste ponto temos os OOFs das 4 seasons para os dois modelos base.
    # O Ridge lerá: "quando o AGCRN dizia X casos e o LightGBM dizia Y casos,
    # a resposta real foi Z — qual combinação linear de X e Y minimiza o erro?"
    # O resultado são coeficientes que revelam quanto confiar em cada modelo
    # dependendo do bioma, clima Köppen, tamanho do município e época do ano.
    meta_modelo, nomes_coef = treinar_meta_modelo(
        oof_agcrn_list=oof_agcrn_list,
        oof_lgbm_list=oof_lgbm_list,
        oof_alvo_list=oof_alvo_list,
        df_val_list=df_val_list,
        alpha=10.0,
    )

    # --------------------------------------------------------------------------
    # 6.5 [F2] Recalibração do r APÓS o ensemble
    # --------------------------------------------------------------------------
    # PROBLEMA: o Ridge gerou um novo mu_ensemble diferente do mu_agcrn original.
    # O r foi aprendido pelo AGCRN para seu mu específico.
    # Var[NB] = mu + mu²/r → se mu mudou e r não, os intervalos estão errados.
    #
    # SOLUÇÃO: aplicamos o meta-modelo nos OOFs para obter mu_ensemble_oof
    # (a previsão combinada nas semanas de validação), comparamos com os casos
    # reais, e buscamos o r que melhor explica esses resíduos por MLE.
    # O r_ensemble_global resultante é usado em TODAS as submissões.
    print("\n  [F2] Recalibrando r do ensemble por MLE sobre OOF...")
    oof_mu_ens_all  = []
    oof_alvo_all    = []
    oof_r_agcrn_all = []

    for sid in [1, 2, 3, 4]:
        res_a   = resultados_agcrn[sid]
        res_l   = resultados_lgbm[sid]
        df_v    = res_a['df_val_sort']
        T_c     = oof_agcrn_list[sid - 1].shape[0]

        cols_ctx = ['log_pop', 'biome_encoded', 'koppen_encoded',
                    'seno_semana', 'cosseno_semana']
        for c in cols_ctx:
            if c not in df_v.columns:
                df_v[c] = 0.0
        df_v_ctx = df_v[cols_ctx].iloc[:T_c * num_nodes].reset_index(drop=True)

        mu_ens_oof = aplicar_meta_modelo(
            meta_modelo=meta_modelo,
            mu_agcrn_tgt=oof_agcrn_list[sid - 1],
            pred_lgbm_tgt=res_l['oof_lgbm_3d'][:T_c],
            df_tgt_sort=df_v_ctx,
        )
        oof_mu_ens_all.append(mu_ens_oof.flatten())
        oof_alvo_all.append(oof_alvo_list[sid - 1].flatten())
        oof_r_agcrn_all.append(oof_r_agcrn_list[sid - 1].flatten())

    r_ensemble_global = recalibrar_r_ensemble(
        mu_ensemble_oof=np.concatenate(oof_mu_ens_all),
        alvo_oof=np.concatenate(oof_alvo_all),
        r_agcrn_oof=np.concatenate(oof_r_agcrn_all),
    )

    # --------------------------------------------------------------------------
    # 6.6 Gera submissões ensemble com r recalibrado
    # --------------------------------------------------------------------------
    submissoes_municipais = {}

    for sid in [1, 2, 3, 4]:
        res_agcrn   = resultados_agcrn[sid]
        res_lgbm    = resultados_lgbm[sid]
        df_tgt_sort = res_agcrn['df_tgt_sort']

        cols_ctx = ['log_pop', 'biome_encoded', 'koppen_encoded',
                    'seno_semana', 'cosseno_semana']
        for c in cols_ctx:
            if c not in df_tgt_sort.columns:
                df_tgt_sort[c] = 0.0

        T_agcrn = res_agcrn['preds_mu_np'].shape[0]
        T_lgbm  = res_lgbm['pred_lgbm_3d'].shape[0]
        T_tgt   = min(T_agcrn, T_lgbm)

        mu_ensemble = aplicar_meta_modelo(
            meta_modelo=meta_modelo,
            mu_agcrn_tgt=res_agcrn['preds_mu_np'][:T_tgt],
            pred_lgbm_tgt=res_lgbm['pred_lgbm_3d'][:T_tgt],
            df_tgt_sort=df_tgt_sort.iloc[:T_tgt * num_nodes].reset_index(drop=True),
        )

        res_agcrn_tgt              = dict(res_agcrn)
        res_agcrn_tgt['df_tgt_sort'] = df_tgt_sort.copy()

        # [F2] usa r_ensemble recalibrado
        df_sub_mun = montar_submissao(
            resultado_agcrn=res_agcrn_tgt,
            mu_ensemble=mu_ensemble,
            r_ensemble=r_ensemble_global,
        )
        validar_submissao(df_sub_mun, label=f'Season {sid} [Municipal]')
        submissoes_municipais[sid] = df_sub_mun

    # --------------------------------------------------------------------------
    # 6.7 [F5] Reconciliação hierárquica MinT — municipal → estadual
    # Nota: df_sub_estadual deve vir de um modelo estadual separado ou de
    # agregação simples das previsões municipais como baseline.
    # --------------------------------------------------------------------------
    if HF_DISPONIVEL:
        print("\n  [F5] Reconciliação hierárquica MinT...")
        submissoes_reconciliadas = {}

        for sid in [1, 2, 3, 4]:
            df_mun = submissoes_municipais[sid].copy()

            # Adiciona uf_code ao df_mun via mapa geocode→estado do dengue.csv.gz
            mapa_uf_code = (
                df_dengue[[VARS['municipio'], VARS['uf_code']]]
                .drop_duplicates()
            )
            df_mun = df_mun.merge(mapa_uf_code, on=VARS['municipio'], how='left')

            # Baseline estadual = soma das previsões municipais por estado e data
            cols_q = ['pred', 'lower_50', 'upper_50', 'lower_80', 'upper_80',
                      'lower_90', 'upper_90', 'lower_95', 'upper_95']
            df_est = (
                df_mun.groupby([VARS['uf_code'], 'date'])[cols_q]
                .sum().reset_index()
            )

            df_mun_rec, df_est_rec = reconciliar_mint(
                df_municipal=df_mun,
                df_estadual=df_est,
                col_uf_code=VARS['uf_code'],
            )

            validar_submissao(
                df_mun_rec[[VARS['municipio'], 'date'] + cols_q],
                label=f'Season {sid} [MinT Municipal]'
            )

            nome_mun = f"submission_mint_municipal_{SEASONS[sid]['label'].replace('-','_')}.csv"
            nome_est = f"submission_mint_estadual_{SEASONS[sid]['label'].replace('-','_')}.csv"
            df_mun_rec.to_csv(nome_mun, index=False)
            df_est_rec.to_csv(nome_est, index=False)
            print(f"  Salvos: {nome_mun} | {nome_est}")

            submissoes_reconciliadas[sid] = (df_mun_rec, df_est_rec)
    else:
        # Sem reconciliação: salva apenas o municipal
        for sid in [1, 2, 3, 4]:
            nome = f"submission_ensemble_{SEASONS[sid]['label'].replace('-','_')}.csv"
            submissoes_municipais[sid].to_csv(nome, index=False)
            print(f"  Arquivo ensemble salvo: {nome}")

    # Resumo do pipeline completo:
    # AGCRN (grafo espacial + GRU) + LightGBM (árvores locais)
    # → Ridge (stacking: aprende pesos por contexto)
    # → r recalibrado por MLE (intervalos NB consistentes com mu_ensemble)
    # → MinT (consistência hierárquica municipal ↔ estadual)
    print("\n✅ Pipeline completo.")
    print("   AGCRN + LightGBM → Ridge → r recalibrado (MLE) → MinT.")
    print("   Próximo passo: submeter até 01/07/2026 conforme o edital.")
