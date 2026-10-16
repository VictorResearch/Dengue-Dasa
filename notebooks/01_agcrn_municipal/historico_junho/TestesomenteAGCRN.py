# =============================================================================
# IMPORTAÇÕES
# =============================================================================
import sys
import warnings
import sqlite3                      # Banco de dados para exportação final
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN
from xgboost import XGBRegressor    # Usado APENAS para SHAP
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.stats import nbinom

try:
    from hierarchicalforecast.methods import MinTrace
    from hierarchicalforecast.utils import aggregate
    HF_DISPONIVEL = True
except ImportError:
    HF_DISPONIVEL = False
    print("⚠️ hierarchicalforecast não instalado — reconciliação MinT desativada.")

# import shap # shap não vai rodar com numpy <2, mas o AGCRN não roda com numpy >2
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. SETUP E DICIONÁRIO DE VARIÁVEIS
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 A executar no hardware: {device.type.upper()}")

VARS = {
    'municipio':         'geocode',
    'semana_epi':        'epiweek',
    'data':              'date',
    'casos_dengue':      'casos.x',
    'casos_chik':        'casos.y',
    'populacao':         'population',
    'uf_code':           'uf_code',
    'temp_min':          'temp_min',
    'temp_med':          'temp_med',
    'temp_max':          'temp_max',
    'precip_min':        'precip_min',
    'precip_med':        'precip_med',
    'precip_max':        'precip_max',
    'precip_tot':        'precip_tot',
    'amplitude_termica': 'thermal_range',
    'pressao_min':       'pressure_min',
    'pressao_atm':       'pressure_med',
    'pressao_max':       'pressure_max',
    'umidade_min':       'rel_humid_min',
    'umidade':           'rel_humid_med',
    'umidade_max':       'rel_humid_max',
    'dias_chuva':        'rainy_days',
    'enso':              'enso',
    'iod':               'iod',
    'pdo':               'pdo',
    'bioma':             'biome',
    'koppen':            'koppen',
    'afya_acessos':      'access_count',
    'vpd':               'vpd',
    'amp_termica_din3':  'amplitude_termica_dinamica3',
    'amp_termica_din2':  'amplitude_termica_dinamica2',
    'volatilidade_temp': 'volatilidade_temp_4w',
    'volatilidade_prec': 'volatilidade_precip_4w',
    'hti3':              'hti_lag3',
    'hti2':              'hti_lag2',
}

SEASONS = {
    1: {'train': "bancosDEF/train1_dengue_eng_COMPLETO_spatial.parquet", 'test': "bancosDEF/test1_dengue_eng_COMPLETO_spatial.parquet", 'label': '2022-2023'},
    2: {'train': "bancosDEF/train2_dengue_eng_COMPLETO_spatial.parquet", 'test': "bancosDEF/test2_dengue_eng_COMPLETO_spatial.parquet", 'label': '2023-2024'},
    3: {'train': "bancosDEF/train3_dengue_eng_COMPLETO_spatial.parquet", 'test': "bancosDEF/test3_dengue_eng_COMPLETO_spatial.parquet", 'label': '2024-2025'},
    4: {'train': "bancosDEF/train4_dengue_eng_COMPLETO_spatial.parquet", 'test': "bancosDEF/test4_dengue_eng_COMPLETO_spatial.parquet", 'label': '2025-2026'}
}

QUANTIS = {
    '50': (0.25, 0.75),
    '80': (0.10, 0.90),
    '90': (0.05, 0.95),
    '95': (0.025, 0.975),
}

# FÍSICA DA EPIDEMIA (Otimizado via pre-alocação NumPy)

def extrair_fisica(df: pd.DataFrame, janela: int = 7, ordem: int = 3) -> pd.DataFrame:
    df = df.sort_values([VARS['municipio'], VARS['semana_epi']]).copy()
    df['time_index'] = df.groupby(VARS['municipio']).cumcount() + 1
 
    # Quantas semanas o filtro "olha para frente" — é exatamente isso que
    # precisamos desfazer com o shift causal abaixo.
    deslocamento = janela // 2
 
    # 1. Pré-aloca arrays 1D no tamanho exato do dataframe (muito leve na memória)
    n_rows = len(df)
    casos_suavizados = np.zeros(n_rows, dtype=np.float64)
    velocidade       = np.zeros(n_rows, dtype=np.float64)
    aceleracao       = np.zeros(n_rows, dtype=np.float64)
 
    # 2. Extrai os valores brutos uma única vez para evitar acessos repetidos na memória
    y_raw_all = df[VARS['casos_dengue']].fillna(0).values
 
    # 3. Itera sobre os índices dos grupos em vez de fatiar o dataframe
    for _, idx in df.groupby(VARS['municipio']).indices.items():
        y_raw = y_raw_all[idx]
        y = np.log1p(y_raw)
        n = len(y)
 
        win = min(janela, n if n % 2 == 1 else max(n - 1, 1))
        win = max(win, ordem + 2 if (ordem + 2) % 2 == 1 else ordem + 3)
 
        if n < win:
            casos_suavizados[idx] = y_raw
            # velocidade e aceleracao já foram pré-alocadas como 0.0
        else:
            log_suav_centrado = savgol_filter(y, win, ordem, deriv=0)
            vel_centrada       = savgol_filter(y, win, ordem, deriv=1)
            acel_centrada      = savgol_filter(y, win, ordem, deriv=2)
 
            # [LEAKAGE FIX] Desloca o resultado 'deslocamento' posições para a
            # direita dentro do próprio município: a posição i passa a receber
            # o valor que o filtro centrado calculou na posição (i - deslocamento).
            # As primeiras 'deslocamento' posições da série não têm passado
            # suficiente para um valor causal válido — preenchidas com o
            # primeiro valor disponível (mantém a feature definida, sem
            # introduzir NaN que quebraria o treino).
            log_suav_causal = np.empty_like(log_suav_centrado)
            vel_causal       = np.empty_like(vel_centrada)
            acel_causal       = np.empty_like(acel_centrada)
 
            if deslocamento > 0 and n > deslocamento:
                log_suav_causal[:deslocamento] = log_suav_centrado[0]
                vel_causal[:deslocamento]      = vel_centrada[0]
                acel_causal[:deslocamento]     = acel_centrada[0]
 
                log_suav_causal[deslocamento:] = log_suav_centrado[:-deslocamento]
                vel_causal[deslocamento:]      = vel_centrada[:-deslocamento]
                acel_causal[deslocamento:]     = acel_centrada[:-deslocamento]
            else:
                log_suav_causal = log_suav_centrado
                vel_causal      = vel_centrada
                acel_causal      = acel_centrada
 
            casos_suavizados[idx] = np.expm1(np.maximum(log_suav_causal, 0))
            velocidade[idx]       = vel_causal
            aceleracao[idx]       = acel_causal
 
    # 4. Atribui os arrays calculados de volta como colunas do dataframe original
    df['casos_suavizados'] = casos_suavizados
    df['velocidade']       = velocidade
    df['aceleracao']       = aceleracao
 
    return df

# ENGENHARIA DE FEATURES
def engenharia_de_features(df: pd.DataFrame, fit_encoders: bool = True, encoder_means: dict = None) -> tuple:
    df = df.sort_values(by=[VARS['municipio'], VARS['semana_epi']]).copy()
    df['log_pop'] = np.log(df['population'].clip(lower=1).astype(float))
    df['incidencia'] = (df[VARS['casos_dengue']] / df['population'].clip(lower=1)) * 100_000

    df['imunidade_ewm'] = (
        df.groupby(VARS['municipio'])['incidencia']
        .transform(lambda x: x.ewm(halflife=104, adjust=False).mean())
    )

    semana_ano = pd.to_datetime(df[VARS['data']]).dt.isocalendar().week
    df['seno_semana']    = np.sin(2 * np.pi * semana_ano / 52).astype(float)
    df['cosseno_semana'] = np.cos(2 * np.pi * semana_ano / 52).astype(float)

    vars_clima = [
        VARS.get('temp_min'), VARS.get('temp_med'), VARS.get('temp_max'),
        VARS.get('precip_min'), VARS.get('precip_med'), VARS.get('precip_max'), VARS.get('precip_tot'),
        VARS.get('amplitude_termica'), VARS.get('pressao_min'), VARS.get('pressao_atm'), VARS.get('pressao_max'),
        VARS.get('umidade_min'), VARS.get('umidade'), VARS.get('umidade_max'), VARS.get('dias_chuva'),
        VARS.get('vpd')
    ]
    vars_clima = [v for v in vars_clima if v is not None]
    df['_date_dt'] = pd.to_datetime(df[VARS['data']])

    for var in tqdm(vars_clima, desc="  Lags Climáticos", leave=False, file=sys.stdout):
        if var not in df.columns: continue
        for lag in range(2, 5):
            delta = pd.Timedelta(weeks=lag)
            df_src = df[[VARS['municipio'], '_date_dt', var]].rename(columns={'_date_dt': '_date_src', var: f'{var}_lag{lag}'})
            df['_date_lag'] = df['_date_dt'] - delta
            df = df.sort_values('_date_lag')
            df_src = df_src.sort_values('_date_src')
            merged = pd.merge_asof(
                df, df_src, left_on='_date_lag', right_on='_date_src',
                by=VARS['municipio'], tolerance=pd.Timedelta('7 days'), direction='nearest'
            )
            df[f'{var}_lag{lag}'] = merged[f'{var}_lag{lag}'].values
            df = df.drop(columns='_date_lag')

    if VARS['precip_tot'] in df.columns:
        df['chuva_acumulada_4w'] = df.groupby(VARS['municipio'])[VARS['precip_tot']].transform(lambda x: x.rolling(4, min_periods=1).sum())
    if 'precip_max' in df.columns:
        df['precip_max_lag3'] = df.groupby(VARS['municipio'])['precip_max'].shift(3)

    for var in tqdm([VARS['enso'], VARS['iod'], VARS['pdo']], desc="  Lags Macroclimáticos", leave=False, file=sys.stdout):
        if var not in df.columns: continue
        for lag in range(6, 10):
            delta = pd.Timedelta(weeks=lag)
            df_src = df[[VARS['municipio'], '_date_dt', var]].rename(columns={'_date_dt': '_date_src', var: f'{var}_lag{lag}'})
            df['_date_lag'] = df['_date_dt'] - delta
            df = df.sort_values('_date_lag')
            df_src = df_src.sort_values('_date_src')
            merged = pd.merge_asof(
                df, df_src, left_on='_date_lag', right_on='_date_src',
                by=VARS['municipio'], tolerance=pd.Timedelta('14 days'), direction='nearest'
            )
            df[f'{var}_lag{lag}'] = merged[f'{var}_lag{lag}'].values
            df = df.drop(columns='_date_lag')

    df = df.drop(columns='_date_dt', errors='ignore')

    if VARS['afya_acessos'] in df.columns:
        df['afya_log_diff'] = (
            np.log1p(df[VARS['afya_acessos']].clip(lower=0))
            .groupby(df[VARS['municipio']])
            .diff()
            .fillna(0)
        )

    if encoder_means is None:
        encoder_means = {}
    for col in [VARS['bioma'], VARS['koppen']]:
        if col in df.columns:
            if fit_encoders:
                encoder_means[col] = df.groupby(col)[VARS['casos_dengue']].mean()
            df[f'{col}_encoded'] = df[col].map(encoder_means[col]).fillna(0)

    # CORREÇÃO: Preenchimento seguro de NaNs separando números e textos
    df = df.sort_values(by=[VARS['municipio'], VARS['semana_epi']])
    
    # Isola chaves que não devem ser tocadas
    cols_chaves = [VARS['municipio'], VARS['semana_epi'], VARS['data']]
    
    # Separa colunas numéricas e de texto
    cols_num = [c for c in df.columns if c not in cols_chaves and pd.api.types.is_numeric_dtype(df[c])]
    cols_texto = [c for c in df.columns if c not in cols_chaves and not pd.api.types.is_numeric_dtype(df[c])]
    
    # Preenche numéricos projetando dados temporalmente e joga 0 onde não há histórico
    if cols_num:
        df[cols_num] = (
            df.groupby(VARS['municipio'])[cols_num]
            .transform(lambda x: x.ffill().bfill())
            .fillna(0)
        )
        
    # Preenche textos projetando dados e converte tudo para string para evitar erro no PyArrow
    if cols_texto:
        df[cols_texto] = (
            df.groupby(VARS['municipio'])[cols_texto]
            .transform(lambda x: x.ffill().bfill())
            .fillna("desconhecido")
        )
        for col in cols_texto:
            df[col] = df[col].astype(str)

    df = df.reset_index(drop=True)
    return df, encoder_means


# SELEÇÃO SHAP WALK-FORWARD

def selecao_shap_walkforward(df_eng: pd.DataFrame, col_alvo: str = 'incidencia', num_features: int = 15, n_folds: int = 3) -> list:
    print("🔍 Rankeando importância com SHAP walk-forward...")
    cols_ignorar = {
        VARS['municipio'], VARS['semana_epi'], VARS['data'], VARS['casos_dengue'], 'incidencia', 'log_pop',
        VARS['bioma'], VARS['koppen'], 'node_id', 'time_index', '_date_dt', '_date_lag', '_date_src',
        'train_1.x', 'train_2.x', 'train_3.x', 'train_4.x', 'target_1.x', 'target_2.x', 'target_3.x', 'target_4.x',
        'disease.x', 'disease.y','uf', VARS['uf_code'], 'regional_geocode', 'macroregional_geocode', 'target_city.x', 'target_city.y', "population",
        "year", "casos_suavizados"
    }

    semanas_ordenadas = sorted(df_eng[VARS['semana_epi']].unique())
    n = len(semanas_ordenadas)
    fold_size = n // (n_folds + 1)
    importancias_folds = []

    cols_num = [c for c in df_eng.columns if c not in cols_ignorar and pd.api.types.is_numeric_dtype(df_eng[c])]

    for fold in tqdm(range(n_folds), desc="  Folds SHAP", file=sys.stdout):
        corte_idx = (fold + 1) * fold_size
        val_end_idx = corte_idx + fold_size
        
        df_tr = df_eng[df_eng[VARS['semana_epi']].isin(semanas_ordenadas[:corte_idx])].copy()
        df_vl = df_eng[df_eng[VARS['semana_epi']].isin(semanas_ordenadas[corte_idx:val_end_idx])].copy()

        if df_tr.empty or df_vl.empty: continue

        X_tr, y_tr = df_tr[cols_num].fillna(0), df_tr[col_alvo]
        X_vl = df_vl[cols_num].fillna(0)

        if X_tr.empty or X_vl.empty: continue
        
        xgb = XGBRegressor(n_estimators=100, max_depth=5, tree_method='hist', n_jobs=-1)
        xgb.fit(X_tr, y_tr)

        sv = shap.TreeExplainer(xgb).shap_values(X_vl)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=cols_num)
        importancias_folds.append(imp)

    if not importancias_folds:
        print("  ⚠️ Aviso: O fatiamento temporal ficou muito curto. Usando heurística base para selecionar features.")
        fallback = [c for c in cols_num if 'lag' in c or 'suavizado' in c or 'vel' in c]
        return fallback[:num_features]

    importancia_final = pd.concat(importancias_folds, axis=1).median(axis=1).sort_values(ascending=False)
    return importancia_final.head(num_features).index.tolist()

# ==============================================================================
# EXECUÇÃO DO PROCESSAMENTO (ETAPA 1)
# ==============================================================================
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # CONTROLE MANUAL DE EXECUÇÃO
    # Defina aqui qual Season você deseja pré-processar (1, 2, 3 ou 4)
    # --------------------------------------------------------------------------
    SEASON_ALVO = 4

    arquivo_treino = SEASONS[SEASON_ALVO]['train']
    arquivo_teste  = SEASONS[SEASON_ALVO]['test']

    print(f"\n{'='*60}")
    print(f" PREPARAÇÃO DE DADOS: SEASON {SEASON_ALVO} ({SEASONS[SEASON_ALVO]['label']})")
    print(f"{'='*60}")

    print(f"Carregando {arquivo_treino} e {arquivo_teste}...")
    df_train = pd.read_parquet(arquivo_treino)
    df_test  = pd.read_parquet(arquivo_teste)

    print("\n[1/3] Aplicando Engenharia de Features...")
    # Treino: fita os encoders e calcula as médias para bioma/koppen
    df_train_eng, enc_global = engenharia_de_features(df_train, fit_encoders=True)
    
    # Teste: aplica as transformações utilizando os encoders aprendidos no treino (sem leakage)
    df_test_eng, _ = engenharia_de_features(df_test, fit_encoders=False, encoder_means=enc_global)

    print("\n[2/3] Extraindo Física da Epidemia (Savitzky-Golay)...")
    df_train_eng = extrair_fisica(df_train_eng)
    df_test_eng  = extrair_fisica(df_test_eng)

    print("\n[3/3] Seleção de Features Otimizada...")
    # SHAP roda estritamente sobre os dados de treino
    features_selecionadas = selecao_shap_walkforward(df_train_eng, col_alvo=VARS['casos_dengue'], num_features=15, n_folds=3)
    cols_base = list(set(features_selecionadas))
    # + ['chikungunya_casos', 'afya_log_diff', 'imunidade_ewm', 'seno_semana', 'cosseno_semana']
    print(f"\n✅ Features primárias rankeadas (guarde para o pipeline do AGCRN):")
    for f in cols_base:
        print(f"   - {f}")

    # ==========================================================================
    # SALVAMENTO
    # ==========================================================================
    out_train = arquivo_treino.replace('.parquet', '_eng.parquet')
    out_test  = arquivo_teste.replace('.parquet', '_eng.parquet')

    print(f"\nSalvando datasets pré-processados em disco...")
    df_train_eng.to_parquet(out_train, index=False)
    df_test_eng.to_parquet(out_test, index=False)

    print(f"✅ Arquivos salvos com sucesso:")
    print(f"   Treino : {out_train} (Linhas: {len(df_train_eng)}, Colunas: {len(df_train_eng.columns)})")
    print(f"   Teste  : {out_test} (Linhas: {len(df_test_eng)}, Colunas: {len(df_test_eng.columns)})")
    print(f"\nOs datasets processados já contém todos os lags, física e numéricos alinhados.")

    out_train = arquivo_treino.replace('.parquet', '_eng.parquet')
    out_test  = arquivo_teste.replace('.parquet', '_eng.parquet')
print(df_train_eng.columns.tolist())

import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor
from tqdm import tqdm
import sys

def selecao_shap_walkforward(df_eng: pd.DataFrame, col_alvo: str = 'incidencia', num_features: int = 25, n_folds: int = 3) -> list:
    print(f"🔍 Rankeando importância com SHAP walk-forward (Top {num_features} features)...")
    
    # Lista expandida para blindar vazamento de dados do tempo atual (t) e metadados irrelevantes do novo merge
    cols_ignorar = {
        # Metadados de Tempo e Espaço
        VARS['municipio'], VARS['semana_epi'], VARS['data'], 'node_id', 'time_index', 
        '_date_dt', '_date_lag', '_date_src', 'year', 'date', 'epiweek', 'geocode',
        
        # Metadados Geográficos e Strings
        VARS['bioma'], VARS['koppen'], 'uf', VARS['uf_code'], 'regional_geocode', 
        'macroregional_geocode', 'macroregion_code', 'macroregion_name', 'uf_name', 
        'macroregional_name', 'regional_name', 'geocode_name', 'biome',
        
        # Variáveis de Alvo e População Base
        col_alvo, VARS['casos_dengue'], 'log_pop', 'population', 'incidencia',
        
        # Variáveis de Tempo Atual (t) que causariam Data Leakage (A rede só pode ver _t1 e w_)
        'casos_suavizados', 
        #'velocidade', 'aceleracao', 
        #'casos_chikungunya',
        'casos.x', 'casos.y', 'disease.x', 'disease.y',
        
        # Lixo residual de Múltiplos Merges (Padrões .x e .y)
        'target_city.x', 'target_city.y',
        'train_1.x', 'train_2.x', 'train_3.x', 'train_4.x', 
        'target_1.x', 'target_2.x', 'target_3.x', 'target_4.x',
        'train_1.y', 'train_2.y', 'train_3.y', 'train_4.y', 
        'target_1.y', 'target_2.y', 'target_3.y', 'target_4.y'
    }

    semanas_ordenadas = sorted(df_eng[VARS['semana_epi']].unique())
    n = len(semanas_ordenadas)
    fold_size = n // (n_folds + 1)
    importancias_folds = []

    # O filtro is_numeric_dtype garante que strings (se tiver sobrado alguma) não quebrem o XGBoost
    cols_num = [c for c in df_eng.columns if c not in cols_ignorar and pd.api.types.is_numeric_dtype(df_eng[c])]

    for fold in tqdm(range(n_folds), desc="   Folds SHAP", file=sys.stdout):
        corte_idx = (fold + 1) * fold_size
        val_end_idx = corte_idx + fold_size
        
        df_tr = df_eng[df_eng[VARS['semana_epi']].isin(semanas_ordenadas[:corte_idx])].copy()
        df_vl = df_eng[df_eng[VARS['semana_epi']].isin(semanas_ordenadas[corte_idx:val_end_idx])].copy()

        if df_tr.empty or df_vl.empty: continue

        X_tr, y_tr = df_tr[cols_num].fillna(0), df_tr[col_alvo]
        X_vl = df_vl[cols_num].fillna(0)

        if X_tr.empty or X_vl.empty: continue
        
        # XGBoost altamente otimizado para lidar com a nova largura do dataframe (~100 colunas)
        xgb = XGBRegressor(n_estimators=100, max_depth=5, tree_method='hist', n_jobs=-1)
        xgb.fit(X_tr, y_tr)

        # O TreeExplainer extrai o valor exato de contribuição marginal de cada variável nova (w1 e w2)
        sv = shap.TreeExplainer(xgb).shap_values(X_vl)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=cols_num)
        importancias_folds.append(imp)

    if not importancias_folds:
        print("   ⚠️ Aviso: O fatiamento temporal ficou muito curto. Usando heurística base para selecionar features.")
        fallback = [c for c in cols_num if 'lag' in c or 'suavizado' in c or 'vel' in c or 'w1_' in c]
        return fallback[:num_features]

    # Agrega a importância atravessando todos os folds temporais e pega as Top N
    importancia_final = pd.concat(importancias_folds, axis=1).median(axis=1).sort_values(ascending=False)
    
    # Print rápido no console para você auditar quem ganhou a disputa (Opcional, mas muito útil)
    #print("\n🏆 Top 25 Variáveis SHAP Globais:")
    #print(importancia_final.head(25).to_string())
    
    return importancia_final.head(num_features).index.tolist()

from collections import Counter

if __name__ == '__main__':
    # Dicionário para guardar as 25 features selecionadas de cada safra
    features_por_safra = {}

    # Loop principal para rodar as 4 Seasons
    for SEASON_ALVO in range(1, 5):
        arquivo_treino = SEASONS[SEASON_ALVO]['train']

        print(f"\n{'='*60}")
        print(f" PREPARAÇÃO DE DADOS E SHAP: SEASON {SEASON_ALVO} ({SEASONS[SEASON_ALVO]['label']})")
        print(f"{'='*60}")

        print(f"Carregando {arquivo_treino}...")
        df_train = pd.read_parquet(arquivo_treino)

        # ==========================================================================
        # ⚠️ INSIRA AQUI SUA LÓGICA DE ENGENHARIA DE FEATURES
        # (O código que gera df_train_eng e df_test_eng a partir do df_train/test)
        # ==========================================================================
        # df_train_eng = ...
        # df_test_eng = ...

        print(f"\n[3/3] Seleção de Features Otimizada (Top 25) - Safra {SEASON_ALVO}...")
        
        # SHAP roda estritamente sobre os dados de treino com 25 features
        features_selecionadas = selecao_shap_walkforward(
            df_train, 
            col_alvo=VARS['casos_dengue'], 
            num_features=25, 
            n_folds=3
        )
        
        cols_base = list(set(features_selecionadas))
        features_por_safra[SEASON_ALVO] = cols_base

        print(f"\n✅ Top 25 Features primárias rankeadas na Season {SEASON_ALVO}:")
        for f in cols_base:
            print(f"   - {f}")


    # ==========================================================================
    # CONSOLIDAÇÃO FINAL (PÓS-LOOP)
    # ==========================================================================
    print(f"\n{'='*60}")
    print(" 🏆 RESUMO GERAL DAS FEATURES SELECIONADAS (SAFRAS 1 A 4)")
    print(f"{'='*60}")
    
    todas_features = []
    for s, feats in features_por_safra.items():
        todas_features.extend(feats)
    
    # Conta quantas vezes cada variável apareceu no top 25 das 4 safras
    contagem = Counter(todas_features)
    
    print("Frequência de aparição das variáveis nos Top 25 (Máximo = 4x):")
    for feat, count in contagem.most_common():
        print(f"   - [{count}/4 safras] {feat}")




import pandas as pd
import numpy as np
import shap
from xgboost import XGBRegressor
from tqdm import tqdm
import sys
from collections import Counter

def selecao_shap_walkforward_regime(df_eng: pd.DataFrame, regime: str, col_alvo: str = 'incidencia', num_features: int = 25, n_folds: int = 3) -> list:
    print(f"🔍 Rankeando importância com SHAP walk-forward | Regime: [{regime.upper()}]...")
    # 1. Filtro Epidemiológico Rígido (Corrigido para YYYYWW)
    # O operador % 100 extrai apenas os dois últimos dígitos (a semana de 1 a 53)
    semana_pura = df_eng['epiweek'] % 100

    if regime == 'seca':
        # Seca: Semanas 23 a 48
        df_regime = df_eng[(semana_pura >= 23) & (semana_pura <= 48)].copy()
    elif regime == 'pico':
        # Pico: Semanas 49 até 53/52, e semanas 1 a 22 do ano seguinte
        df_regime = df_eng[(semana_pura >= 49) | (semana_pura <= 22)].copy()
    else:
        raise ValueError("O parâmetro 'regime' deve ser 'seca' ou 'pico'.")
    if df_regime.empty:
        print(f"⚠️ Erro: Nenhum dado encontrado para o regime {regime}.")
        return []

    # 2. Lista de Blindagem contra Vazamento (Data Leakage)
    cols_ignorar = {
        # Metadados de Tempo e Espaço
        VARS['municipio'], VARS['semana_epi'], VARS['data'], 'node_id', 'time_index', 
        '_date_dt', '_date_lag', '_date_src', 'year', 'date', 'epiweek', 'geocode',
        
        # Metadados Geográficos e Strings
        VARS['bioma'], VARS['koppen'], 'uf', VARS['uf_code'], 'regional_geocode', 
        'macroregional_geocode', 'macroregion_code', 'macroregion_name', 'uf_name', 
        'macroregional_name', 'regional_name', 'geocode_name', 'biome',
        
        # Variáveis de Alvo e População Base
        col_alvo, VARS['casos_dengue'], 'log_pop', 'population', 'incidencia',
        
        # Variáveis de Tempo Atual (t) - O modelo só pode ver _t1 e lags
        'casos_suavizados', 
        #'velocidade', 'aceleracao', 'casos_chikungunya',
        'casos.x', 'casos.y', 'disease.x', 'disease.y',
        
        # Lixo residual de Múltiplos Merges
        'target_city.x', 'target_city.y',
        'train_1.x', 'train_2.x', 'train_3.x', 'train_4.x', 
        'target_1.x', 'target_2.x', 'target_3.x', 'target_4.x',
        'train_1.y', 'train_2.y', 'train_3.y', 'train_4.y', 
        'target_1.y', 'target_2.y', 'target_3.y', 'target_4.y'
    }

    # As semanas ordenadas agora refletem APENAS as semanas do regime escolhido
    semanas_ordenadas = sorted(df_regime[VARS['semana_epi']].unique())
    n = len(semanas_ordenadas)
    fold_size = n // (n_folds + 1)
    importancias_folds = []

    cols_num = [c for c in df_regime.columns if c not in cols_ignorar and pd.api.types.is_numeric_dtype(df_regime[c])]

    for fold in tqdm(range(n_folds), desc=f"   Folds {regime.upper()}", file=sys.stdout):
        corte_idx = (fold + 1) * fold_size
        val_end_idx = corte_idx + fold_size
        
        df_tr = df_regime[df_regime[VARS['semana_epi']].isin(semanas_ordenadas[:corte_idx])].copy()
        df_vl = df_regime[df_regime[VARS['semana_epi']].isin(semanas_ordenadas[corte_idx:val_end_idx])].copy()

        if df_tr.empty or df_vl.empty: continue

        X_tr, y_tr = df_tr[cols_num].fillna(0), df_tr[col_alvo]
        X_vl = df_vl[cols_num].fillna(0)

        if X_tr.empty or X_vl.empty: continue
        
        xgb = XGBRegressor(n_estimators=100, max_depth=5, tree_method='hist', n_jobs=-1)
        xgb.fit(X_tr, y_tr)

        sv = shap.TreeExplainer(xgb).shap_values(X_vl)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=cols_num)
        importancias_folds.append(imp)

    if not importancias_folds:
        print("   ⚠️ Aviso: Fatiamento curto. Retornando heurística.")
        fallback = [c for c in cols_num if 'lag' in c or 'suavizado' in c or 'vel' in c or 'w1_' in c]
        return fallback[:num_features]

    importancia_final = pd.concat(importancia s_folds, axis=1).median(axis=1).sort_values(ascending=False)
    
    print(f"\n🏆 Top 5 Variáveis SHAP ({regime.upper()}):")
    print(importancia_final.head(5).to_string())
    
    return importancia_final.head(num_features).index.tolist()

if __name__ == '__main__':
    # Dicionários para guardar o histórico das seleções das 4 safras
    features_historico_seca = {}
    features_historico_pico = {}

    for SEASON_ALVO in range(1, 5):
        # Ajuste o nome do arquivo para casar com a saída gerada no seu script R
        arquivo_treino = f"bancosDEF/train{SEASON_ALVO}_dengue_eng_COMPLETO_spatial.parquet"

        print(f"\n{'='*70}")
        print(f" PREPARAÇÃO E SHAP BIFURCADO: SEASON {SEASON_ALVO}")
        print(f"{'='*70}")

        print(f"📥 Carregando {arquivo_treino}...")
        df_train_eng = pd.read_parquet(arquivo_treino)

        # ----------------------------------------------------------------------
        # 1. SHAP PARA O REGIME DE SECA (Calmaria)
        # ----------------------------------------------------------------------
        print(f"\n[Fase 1/2] Seleção Otimizada para o Regime de SECA (Top 25)...")
        features_seca = selecao_shap_walkforward_regime(
            df_train_eng, 
            regime='seca', 
            col_alvo=VARS['casos_dengue'], # Substitua pela sua coluna alvo real (ex: 'incidencia' se já calculada)
            num_features=25, 
            n_folds=3
        )
        cols_seca = list(set(features_seca))
        features_historico_seca[SEASON_ALVO] = cols_seca

        # ----------------------------------------------------------------------
        # 2. SHAP PARA O REGIME DE PICO (Surto)
        # ----------------------------------------------------------------------
        print(f"\n[Fase 2/2] Seleção Otimizada para o Regime de PICO (Top 25)...")
        features_pico = selecao_shap_walkforward_regime(
            df_train_eng, 
            regime='pico', 
            col_alvo=VARS['casos_dengue'], 
            num_features=25, 
            n_folds=3
        )
        cols_pico = list(set(features_pico))
        features_historico_pico[SEASON_ALVO] = cols_pico

        print(f"✅ Season {SEASON_ALVO} concluída! Memória RAM será limpa para a próxima safra.")
        import gc; gc.collect()

    # ==========================================================================
    # CONSOLIDAÇÃO FINAL E RELATÓRIO ECOLÓGICO (PÓS-LOOP)
    # ==========================================================================
    print(f"\n\n{'='*70}")
    print(" 🏆 RESUMO GERAL DAS FEATURES SELECIONADAS (SAFRAS 1 A 4)")
    print(f"{'='*70}")
    
    # Consolidação Seca
    todas_features_seca = [f for feats in features_historico_seca.values() for f in feats]
    contagem_seca = Counter(todas_features_seca)
    
    # Consolidação Pico
    todas_features_pico = [f for feats in features_historico_pico.values() for f in feats]
    contagem_pico = Counter(todas_features_pico)
    
    print("\n🌵 TOP FEATURES ESTÁVEIS NA SECA (Apareceram em 3 ou 4 safras):")
    for feat, count in contagem_seca.most_common():
        if count >= 2:
            print(f"   - [{count}/4] {feat}")

    print("\n🌊 TOP FEATURES ESTÁVEIS NO PICO (Apareceram em 3 ou 4 safras):")
    for feat, count in contagem_pico.most_common():
        if count >= 2:
            print(f"   - [{count}/4] {feat}")
            
    print("\n💡 Dica de Arquitetura: Anote essas duas listas resultantes acima. Elas serão o 'in_channels' do seu AGCRN_Seca e do seu AGCRN_Pico.")


print(df_train_eng.columns.to_list())





import pandas as pd
import gc
import os

if __name__ == '__main__':
    # ==========================================================================
    # 1. DEFINIÇÃO DAS LISTAS DE VARIÁVEIS A SEREM MANTIDAS
    # ==========================================================================
    
    # Variáveis de Identificação e Controle
    vars_identificacao = [
        'geocode', 'date', 'epiweek', 'uf_code', 'macroregional_geocode', 
        'regional_geocode', 'macroregion_code', 'node_id', 'time_index', 
        'year', 'log_pop', 'incidencia', 'forecast_months_ahead',
        'casos.x' # <-- MANTIDO COMO ALVO PARA A FUNÇÃO DE PERDA (NB_LOSS)
    ]

    # Variáveis Geográficas/Climáticas forçadas
    vars_forçadas = ['koppen', 'biome']

    # União de todas as features SHAP (Geral + Seca + Pico) retiradas do seu resumo
    vars_shap = [
        'access_count', 'seno_semana', 'w2_velocidade_t1', 'w1_population_t1',
        'access_count_t1', 'w2_dengue_t1', 'w1_dengue_t1', 'aceleracao', 
        'imunidade_ewm', 'imunidade_ewm_t1', 'casos_suavizados_t1', 
        'casos_chikungunya', 'population_t1', 'casos_chikungunya_t1', 
        'cosseno_semana', 'velocidade', 'w1_velocidade_t1', 'w2_umid_forecast_t1', 
        'w2_precip_forecast_t1', 'w2_temp_forecast_t1', 'umid_med_forecast', 
        'temp_med_forecast', 'w1_imunidade_t1', 'w1_umid_forecast_t1', 
        'w1_temp_forecast_t1', 'koppen_encoded', 'pressure_max_lag3', 
        'temp_med_forecast_t1', 'precip_tot_forecast', 'temp_min', 
        'pressure_min_t1', 'w1_aceleracao_t1', 'w2_population_t1', 
        'temp_min_lag4', 'volatilidade_temp_4w', 'rel_humid_max', 
        'temp_med', 'w2_aceleracao_t1'
    ]

    # Consolida tudo num único Set (conjunto) para busca rápida (O(1)) e sem duplicatas
    todas_variaveis_permitidas = set(vars_identificacao + vars_forçadas + vars_shap)

    # ==========================================================================
    # 2. LOOP DE PROCESSAMENTO (SAFRAS 1 A 4)
    # ==========================================================================
    for SEASON in range(1, 5):
        # Nomes dos arquivos de entrada
        arquivo_treino = f"bancosDEF/train{SEASON}_dengue_eng_COMPLETO_spatial.parquet"
        arquivo_teste  = f"bancosDEF/test{SEASON}_dengue_eng_COMPLETO_spatial.parquet"
        
        # Nomes dos arquivos de saída (salvaremos como '_limpo' para diferenciar)
        out_treino = f"bancosDEF/train{SEASON}_dengue_eng_LIMPO_spatial.parquet"
        out_teste  = f"bancosDEF/test{SEASON}_dengue_eng_LIMPO_spatial.parquet"

        # Verifica se os arquivos existem no diretório atual antes de tentar ler
        if not os.path.exists(arquivo_treino):
            print(f"⚠️ Arquivo não encontrado: {arquivo_treino}. Pulando...")
            continue

        print(f"\n{'='*70}")
        print(f" 🧹 LIMPANDO BASE DE DADOS: SEASON {SEASON}")
        print(f"{'='*70}")

        print(f"📥 Carregando arquivos...")
        df_train = pd.read_parquet(arquivo_treino)
        df_test  = pd.read_parquet(arquivo_teste)
        
        colunas_originais = len(df_train.columns)

        # ----------------------------------------------------------------------
        # 3. LÓGICA DE FILTRAGEM DE COLUNAS
        # ----------------------------------------------------------------------
        def filtrar_colunas(df):
            colunas_finais = []
            for col in df.columns:
                # REGRA 1: Expurgar imediatamente qualquer coluna que comece com 'target' ou 'train'
                if col.startswith('target') or col.startswith('train'):
                    continue
                
                # REGRA 2: Se passou na regra 1, verificar se está na lista mestre de permissões
                if col in todas_variaveis_permitidas:
                    colunas_finais.append(col)
                    
            return df[colunas_finais]

        print("⚙️ Aplicando filtros de colunas (Remoção de lixo e variáveis não selecionadas)...")
        df_train_limpo = filtrar_colunas(df_train)
        df_test_limpo  = filtrar_colunas(df_test)

        colunas_finais_qnt = len(df_train_limpo.columns)

        # ----------------------------------------------------------------------
        # 4. SALVAMENTO E LIMPEZA DE MEMÓRIA
        # ----------------------------------------------------------------------
        print("💾 Salvando arquivos limpos em formato Parquet...")
        df_train_limpo.to_parquet(out_treino, index=False)
        df_test_limpo.to_parquet(out_teste, index=False)

        print(f"✅ Season {SEASON} concluída!")
        print(f"   -> Redução de Colunas: De {colunas_originais} para {colunas_finais_qnt} colunas super-otimizadas.")
        
        # Limpeza agressiva da RAM
        del df_train, df_test, df_train_limpo, df_test_limpo
        gc.collect()

    print("\n🚀 Todos os bancos de dados foram purificados e estão prontos para os modelos AGCRN!")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import gc

def transformar_distribuicoes_para_rede_neural(df: pd.DataFrame) -> pd.DataFrame:
    df_transf = df.copy()
    
    # ==========================================================================
    # 1. TRATAMENTO POPULACIONAL (Logaritmo Natural)
    # ==========================================================================
    # Base populacional (usada como offset na nb_loss)
    if 'population' in df_transf.columns:
        df_transf['log_pop'] = np.log(df_transf['population'].clip(lower=1))
        
    if 'population_t1' in df_transf.columns:
        df_transf['log_pop_t1'] = np.log(df_transf['population_t1'].clip(lower=1))
        
    if 'w1_population_t1' in df_transf.columns:
        df_transf['w1_log_pop_t1'] = np.log(df_transf['w1_population_t1'].clip(lower=1))
        
    if 'w2_population_t1' in df_transf.columns:
        df_transf['w2_log_pop_t1'] = np.log(df_transf['w2_population_t1'].clip(lower=1))

    # ==========================================================================
    # 2. TRATAMENTO DE CASOS LOCAIS E CO-INFECÇÕES (Incidência + Log1p)
    # ==========================================================================
    # Dengue T-1
    if 'casos_suavizados_t1' in df_transf.columns and 'population_t1' in df_transf.columns:
        incidencia_local = (df_transf['casos_suavizados_t1'] / df_transf['population_t1'].clip(lower=1)) * 100000
        df_transf['log_inc_dengue_t1'] = np.log1p(incidencia_local)

    # Chikungunya T-1
    if 'casos_chikungunya_t1' in df_transf.columns and 'population_t1' in df_transf.columns:
        incidencia_chiku_t1 = (df_transf['casos_chikungunya_t1'] / df_transf['population_t1'].clip(lower=1)) * 100000
        df_transf['log_inc_chiku_t1'] = np.log1p(incidencia_chiku_t1)

    # 🔥 CORREÇÃO: Chikungunya T (Presente na sua vars_shap)
    if 'casos_chikungunya' in df_transf.columns and 'population' in df_transf.columns:
        incidencia_chiku_t0 = (df_transf['casos_chikungunya'] / df_transf['population'].clip(lower=1)) * 100000
        df_transf['log_inc_chiku'] = np.log1p(incidencia_chiku_t0)

    # ==========================================================================
    # 3. TRATAMENTO DAS BORDAS ESPACIAIS W1 e W2 (Incidência Espacial + Log1p)
    # ==========================================================================
    # Anel 1 (W1) - Dengue nas bordas
    if 'w1_dengue_t1' in df_transf.columns and 'w1_population_t1' in df_transf.columns:
        w1_incidencia = (df_transf['w1_dengue_t1'] / df_transf['w1_population_t1'].clip(lower=1)) * 100000
        df_transf['w1_log_inc_dengue_t1'] = np.log1p(w1_incidencia)

    # Anel 2 (W2) - Dengue nas bordas distantes
    if 'w2_dengue_t1' in df_transf.columns and 'w2_population_t1' in df_transf.columns:
        w2_incidencia = (df_transf['w2_dengue_t1'] / df_transf['w2_population_t1'].clip(lower=1)) * 100000
        df_transf['w2_log_inc_dengue_t1'] = np.log1p(w2_incidencia)

    # ==========================================================================
    # 4. LIMPEZA DA MATRIZ X (A purga das contagens brutas)
    # ==========================================================================
    # IMPORTANTE: Destruímos as variáveis brutas que foram transformadas para que
    # o StandardScaler não as veja, garantindo que apenas a versão 'log_inc' passe para o modelo.
    cols_para_dropar = [
        'casos_suavizados_t1', 
        'casos_chikungunya_t1', 
        'casos_chikungunya', # <-- Destruído após virar 'log_inc_chiku'
        'w1_dengue_t1', 
        'w2_dengue_t1',
        'population_t1', 
        'w1_population_t1', 
        'w2_population_t1'
    ]
    df_transf = df_transf.drop(columns=[c for c in cols_para_dropar if c in df_transf.columns])
    
    return df_transf


if __name__ == '__main__':
    # ==========================================================================
    # CONFIGURAÇÃO DE PROTEÇÃO DO SCALER
    # ==========================================================================
    # Estas colunas JAMAIS devem ser padronizadas (Z-score) pois são chaves,
    # variáveis alvo ou offsets matemáticos obrigatórios para a rede neural.
    cols_nao_escalonar = {
        'geocode', 'date', 'epiweek', 'uf_code', 'macroregional_geocode', 
        'regional_geocode', 'macroregion_code', 'node_id', 'time_index', 
        'year', 'forecast_months_ahead', 'casos.x', 'population', 'koppen', 
        'biome', 'incidencia', 'log_pop' 
        # Nota: 'log_pop' base fica protegido. As versões lagadas (log_pop_t1) 
        # podem ser escalonadas pois entram como features na rede.
    }

    # ==========================================================================
    # LOOP DE PROCESSAMENTO E ESCALONAMENTO (SAFRAS 1 A 4)
    # ==========================================================================
    for SEASON in range(1, 5):
        # Arquivos de entrada vindo da etapa de limpeza anterior
        arquivo_treino = f"bancosDEF/train{SEASON}_dengue_eng_LIMPO_spatial.parquet"
        arquivo_teste  = f"bancosDEF/test{SEASON}_dengue_eng_LIMPO_spatial.parquet"
        
        # Arquivos de saída finais (Prontos para o PyTorch)
        out_treino = f"bancosDEF/train{SEASON}_FINAL_NN.parquet"
        out_teste  = f"bancosDEF/test{SEASON}_FINAL_NN.parquet"

        if not os.path.exists(arquivo_treino):
            print(f"⚠️ Arquivo não encontrado: {arquivo_treino}. Pulando...")
            continue

        print(f"\n{'='*70}")
        print(f" 🧬 TRANSFORMAÇÃO E ESCALONAMENTO: SEASON {SEASON}")
        print(f"{'='*70}")

        print(f"📥 Carregando arquivos limpos...")
        df_train = pd.read_parquet(arquivo_treino)
        df_test  = pd.read_parquet(arquivo_teste)

        # ----------------------------------------------------------------------
        # PASSO A: Transformações de Distribuição (Log1p e Incidência)
        # ----------------------------------------------------------------------
        print("🔧 Amassando caudas exponenciais (Log1p) e extraindo incidências...")
        df_train = transformar_distribuicoes_para_rede_neural(df_train)
        df_test  = transformar_distribuicoes_para_rede_neural(df_test)

        # ----------------------------------------------------------------------
        # PASSO B: Identificação das Features para o StandardScaler
        # ----------------------------------------------------------------------
        # Seleciona apenas colunas numéricas que não estão na lista de proteção
        features_para_escalonar = [
            col for col in df_train.columns 
            if col not in cols_nao_escalonar and pd.api.types.is_numeric_dtype(df_train[col])
        ]
        
        print(f"📏 Aplicando StandardScaler global em {len(features_para_escalonar)} features...")

        # ----------------------------------------------------------------------
        # PASSO C: Ajuste (Fit) e Transformação (Transform)
        # ----------------------------------------------------------------------
        scaler = StandardScaler()
        
        # O Fit DEVE ser feito EXCLUSIVAMENTE nos dados de Treino (evita Data Leakage)
        # Preenchemos NaN temporariamente com 0 apenas para o scaler não quebrar
        df_train[features_para_escalonar] = scaler.fit_transform(df_train[features_para_escalonar].fillna(0))
        
        # O Teste apenas recebe a Transformação baseada nas médias do Treino
        df_test[features_para_escalonar]  = scaler.transform(df_test[features_para_escalonar].fillna(0))

        # ----------------------------------------------------------------------
        # PASSO D: Salvamento
        # ----------------------------------------------------------------------
        print(f"💾 Salvando matrizes finais para a AGCRN...")
        df_train.to_parquet(out_treino, index=False)
        df_test.to_parquet(out_teste, index=False)

        print(f"✅ Season {SEASON} concluída com sucesso! (Treino: {df_train.shape} | Teste: {df_test.shape})")
        
        # Limpeza de RAM
        del df_train, df_test
        gc.collect()

    print("\n🚀 Todos os datasets estão padronizados e prontos para o Treinamento no PyTorch!")





# ==============================================================================
# FILTRAGEM PÓS-SHAP — reduz os 8 datasets às variáveis selecionadas
# ==============================================================================
# Mantém: variáveis de indexação (chaves, datas, flags de season) +
#         alvo de dengue + as 12 features finais selecionadas pelo SHAP.
# Tudo o que não está nessas três listas é descartado.

# --- 1. Variáveis de indexação — nunca filtradas, mesmo que não usadas como feature ---
FISICA_CONTEXTO_SEMANAS = 12
cols_indexacao = [
    VARS['municipio'],        # geocode
    VARS['data'],              # date
    VARS['semana_epi'],        # epiweek
    VARS['uf_code'],           # uf_code — exigido na submissão da plataforma
    VARS['populacao'], 
    'uf',
    'macroregional_geocode',
    'regional_geocode'
]

# --- 2. Alvo — casos de dengue (renomeado de 'casos.x' para o nome padrão do VARS) ---
cols_alvo = [VARS['casos_dengue']]   # 'casos'

# --- 3. As 12 features finais selecionadas pelo SHAP walk-forward (estáveis em 3-4 seasons) ---
cols_shap_final = [
    'seno_semana',
    'umid_med_forecast',
    #'aceleracao',
    'imunidade_ewm',
    'precip_tot_forecast',
    'access_count',
    'temp_med_forecast',
    #'velocidade',
    'pressure_min',
    'casos_chikungunya',   # renomeada de 'casos.y'
    #'casos_suavizados',
    'koppen_encoded',
]
cols_fisica_recalculada = ['casos_suavizados', 'velocidade', 'aceleracao']

cols_manter_do_original = cols_indexacao + cols_alvo + cols_shap_final

COLS_MINIMAS_FISICA = [VARS['municipio'], VARS['semana_epi'], VARS['casos_dengue']]

def extrair_fisica(df: pd.DataFrame, janela: int = 7, ordem: int = 3) -> pd.DataFrame:
    df = df.sort_values([VARS['municipio'], VARS['semana_epi']]).copy()
    df['time_index'] = df.groupby(VARS['municipio']).cumcount() + 1

    deslocamento = janela // 2

    n_rows = len(df)
    casos_suavizados = np.zeros(n_rows, dtype=np.float64)
    velocidade       = np.zeros(n_rows, dtype=np.float64)
    aceleracao       = np.zeros(n_rows, dtype=np.float64)

    y_raw_all = df[VARS['casos_dengue']].fillna(0).values

    for _, idx in df.groupby(VARS['municipio']).indices.items():
        y_raw = y_raw_all[idx]
        y = np.log1p(y_raw)
        n = len(y)

        win = min(janela, n if n % 2 == 1 else max(n - 1, 1))
        win = max(win, ordem + 2 if (ordem + 2) % 2 == 1 else ordem + 3)

        if n < win:
            casos_suavizados[idx] = y_raw
        else:
            log_suav_centrado = savgol_filter(y, win, ordem, deriv=0)
            vel_centrada       = savgol_filter(y, win, ordem, deriv=1)
            acel_centrada      = savgol_filter(y, win, ordem, deriv=2)

            log_suav_causal = np.empty_like(log_suav_centrado)
            vel_causal       = np.empty_like(vel_centrada)
            acel_causal      = np.empty_like(acel_centrada)

            if deslocamento > 0 and n > deslocamento:
                log_suav_causal[:deslocamento] = log_suav_centrado[0]
                vel_causal[:deslocamento]      = vel_centrada[0]
                acel_causal[:deslocamento]     = acel_centrada[0]

                log_suav_causal[deslocamento:] = log_suav_centrado[:-deslocamento]
                vel_causal[deslocamento:]      = vel_centrada[:-deslocamento]
                acel_causal[deslocamento:]     = acel_centrada[:-deslocamento]
            else:
                log_suav_causal = log_suav_centrado
                vel_causal      = vel_centrada
                acel_causal     = acel_centrada

            casos_suavizados[idx] = np.expm1(np.maximum(log_suav_causal, 0))
            velocidade[idx]       = vel_causal
            aceleracao[idx]       = acel_causal

    df['casos_suavizados'] = casos_suavizados
    df['velocidade']       = velocidade
    df['aceleracao']       = aceleracao

    return df


def _montar_df_com_contexto(
    df_teste_filtrado: pd.DataFrame,
    df_treino_filtrado: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Concatena as últimas FISICA_CONTEXTO_SEMANAS semanas do treino no início
    do teste, marcando-as com uma flag '_contexto_temp' = True para remoção
    segura após o cálculo da física — independente de reordenação interna.
    """
    FLAG = '_contexto_temp'

    cauda_treino = (
        df_treino_filtrado[COLS_MINIMAS_FISICA]
        .sort_values([VARS['municipio'], VARS['semana_epi']])
        .groupby(VARS['municipio'], sort=False)
        .tail(FISICA_CONTEXTO_SEMANAS)
        .copy()
    )
    cauda_treino[FLAG] = True

    df_teste_marcado = df_teste_filtrado.copy()
    df_teste_marcado[FLAG] = False

    df_com_contexto = pd.concat(
        [cauda_treino, df_teste_marcado],
        ignore_index=True,
    )

    return df_com_contexto, FLAG


def filtrar_e_recalcular_fisica(
    df: pd.DataFrame,
    nome_dataset: str,
    df_treino_contexto: pd.DataFrame | None = None,
) -> pd.DataFrame:

    df = df.rename(columns={
        'casos.x': VARS['casos_dengue'],
        'casos.y': 'casos_chikungunya',
    })

    cols_antigas_presentes = [c for c in cols_fisica_recalculada if c in df.columns]
    if cols_antigas_presentes:
        print(f"  🗑  [{nome_dataset}] descartando física antiga: {cols_antigas_presentes}")

    cols_presentes = [c for c in cols_manter_do_original if c in df.columns]
    cols_ausentes  = [c for c in cols_manter_do_original if c not in df.columns]
    if cols_ausentes:
        print(f"  ⚠️  [{nome_dataset}] colunas ausentes (ignoradas): {cols_ausentes}")

    df_filtrado = df[cols_presentes].copy()

    if df_treino_contexto is not None:
        df_para_fisica, FLAG = _montar_df_com_contexto(df_filtrado, df_treino_contexto)
        n_contexto = df_para_fisica[FLAG].sum()
        print(f"  📎 [{nome_dataset}] {n_contexto} linhas de contexto adicionadas "
              f"temporariamente (removidas após física)")
    else:
        df_para_fisica = df_filtrado
        FLAG = None

    # Roda o Savitzky-Golay causal sobre o conjunto com contexto
    df_com_fisica = extrair_fisica(df_para_fisica, janela=7, ordem=3)

    # Remove as linhas de contexto pela flag — robusto a qualquer reordenação
    # interna que extrair_fisica faça (ela reordena por municipio+semana_epi)
    if FLAG is not None:
        df_com_fisica = (
            df_com_fisica[~df_com_fisica[FLAG]]
            .drop(columns=[FLAG])
            .reset_index(drop=True)
        )

    print(f"  ✓ [{nome_dataset}] {df.shape[1]} → {df_com_fisica.shape[1]} colunas "
          f"| {len(df_com_fisica)} linhas | física recalculada sem leakage")

    return df_com_fisica

# --- 5. Aplica a filtragem + recálculo nos 8 datasets e salva ---
pares_datasets = [
    ('train1_dengue_eng', 'test1_dengue_eng'),
    ('train2_dengue_eng', 'test2_dengue_eng'),
    ('train3_dengue_eng', 'test3_dengue_eng'),
    ('train4_dengue_eng', 'test4_dengue_eng'),
]

print("Filtrando datasets e recalculando física (Savitzky-Golay causal)...")

for nome_train, nome_test in pares_datasets:

    # Treino — sem contexto externo
    df_train_original = pd.read_parquet(f"{nome_train}.parquet")
    df_train_reduzido = filtrar_e_recalcular_fisica(df_train_original, nome_train)
    df_train_reduzido.to_parquet(f"{nome_train}_shap_final.parquet", index=False)

    # Teste — recebe a cauda do treino já filtrado como contexto para a física
    df_test_original = pd.read_parquet(f"{nome_test}.parquet")
    df_test_reduzido = filtrar_e_recalcular_fisica(
        df_test_original,
        nome_test,
        df_treino_contexto=df_train_reduzido,   # ← cauda do treino como âncora
    )
    df_test_reduzido.to_parquet(f"{nome_test}_shap_final.parquet", index=False)

print("\n✅ 8 datasets filtrados, física recalculada sem leakage, e salvos "
      "com sufixo '_shap_final.parquet'")
# ==============================================================================
# AGCRN & BINOMIAL NEGATIVA
# ==============================================================================
def nb_loss(log_mu: torch.Tensor, log_r: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    log_mu = torch.clamp(log_mu, min=-10, max=10)
    log_r  = torch.clamp(log_r,  min=-6,  max=6)
    mu, r = torch.exp(log_mu), torch.exp(log_r)
    nll = (
        - torch.lgamma(y + r) + torch.lgamma(r) + torch.lgamma(y + 1)
        - r * torch.log(r / (r + mu + 1e-8)) - y * torch.log(mu / (r + mu + 1e-8))
    )
    return nll.mean()

def nb_quantile(mu: np.ndarray, r: np.ndarray, q: float) -> np.ndarray:
    p = r / (r + mu + 1e-8)
    return nbinom.ppf(q, n=r, p=p).astype(float)

COLS_FISICA  = ['casos_suavizados', 'velocidade', 'aceleracao']
COLS_SHAP    = [
    'seno_semana',
    'umid_med_forecast',
    'imunidade_ewm',
    'precip_tot_forecast',
    'access_count',
    'temp_med_forecast',
    'pressure_min',
    'casos_chikungunya',
    'koppen_encoded',
]
COLS_MODELO = COLS_FISICA + COLS_SHAP   # 12 features no total
class PreditorNBAGCRN(nn.Module):
    def __init__(self, num_nodes: int, num_features: int, embedding_dim: int = 32, idx_velocidade: int = 1, idx_aceleracao: int = 2):
        super().__init__()
        self.idx_vel = idx_velocidade
        self.idx_acel = idx_aceleracao
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim), requires_grad=True)
        self.recurrent = AGCRN(
            number_of_nodes=num_nodes, 
            in_channels=num_features, 
            out_channels=64,
            K=2,                                   # Ordem de vizinhança espacial
            embedding_dimensions=embedding_dim     # Passando a mesma dimensão (32)
        )
        self.head_mu = nn.Linear(64, 1)
        self.head_logr = nn.Linear(64 + 2, 1)

    def forward(self, x: torch.Tensor, log_pop: torch.Tensor, h=None):
        # 1. Adiciona a dimensão de Batch exigida pela nova versão do AGCRN
        # x passa de [num_nodes, features] para [1, num_nodes, features]
        x_batched = x.unsqueeze(0)
        
        # 2. Passa pela rede recorrente
        h = self.recurrent(x_batched, self.node_embeddings, h)

        h_out = h.squeeze(0)
        
        h_relu = F.relu(h_out)
        log_mu_raw = self.head_mu(h_relu)
        log_mu = log_mu_raw + log_pop.unsqueeze(-1)

        vel = x[:, self.idx_vel].unsqueeze(-1)
        acel = x[:, self.idx_acel].unsqueeze(-1)
        h_r = torch.cat([h_relu, vel, acel], dim=-1)
        log_r = self.head_logr(h_r)

        # Retorna o log_mu e log_r processados, e o h original (com batch) para a próxima iteração
        return log_mu, log_r, h

def treinar_e_validar_season(df_train: pd.DataFrame, df_test: pd.DataFrame, 
                             season_id: int, 
                             # cols_base: list, 
                             encoder_means_globais: dict,
                             num_nodes: int, mapa_simples: pd.DataFrame, epochs_max: int = 150, patience: int = 15) -> dict:
    df_tr, df_tgt = df_train.copy(), df_test.copy()
    if 'log_pop' not in df_tr.columns:
        df_tr['log_pop'] = np.log(df_tr[VARS['populacao']].clip(lower=1).astype(float))
    if 'log_pop' not in df_tgt.columns:
        df_tgt['log_pop'] = np.log(df_tgt[VARS['populacao']].clip(lower=1).astype(float))
    semanas_treino = sorted(df_tr[VARS['semana_epi']].unique())
    df_tr_real = df_tr[df_tr[VARS['semana_epi']].isin(semanas_treino[:-10])]
    df_val_int = df_tr[df_tr[VARS['semana_epi']].isin(semanas_treino[-10:])]
        # --- Merge com mapa de nós --------------------------------------------------
    df_tr_sort  = df_tr_real.merge(mapa_simples, on=VARS['municipio'], how='left') \
                             .sort_values(['time_index', 'node_id'])
    df_val_sort = df_val_int.merge(mapa_simples, on=VARS['municipio'], how='left') \
                             .sort_values(['time_index', 'node_id'])
    df_tgt_sort = df_tgt.merge(mapa_simples, on=VARS['municipio'], how='left') \
                         .sort_values(['time_index', 'node_id'])

    # --- Verifica quais das 12 features estão presentes no dataset -------------
    # (proteção caso alguma coluna SHAP ainda não exista num parquet específico)
    cols = [c for c in COLS_MODELO if c in df_tr_sort.columns]
    cols_ausentes = [c for c in COLS_MODELO if c not in df_tr_sort.columns]
    if cols_ausentes:
        print(f"  ⚠️  [S{season_id}] features ausentes no dataset: {cols_ausentes}")

    num_features = len(cols)

    # ← ALTERADO: idx_velocidade e idx_aceleracao derivados da lista cols,
    #   não hardcoded — robusto se cols_ausentes remover alguma física
    idx_vel  = cols.index('velocidade')  if 'velocidade'  in cols else 1
    idx_acel = cols.index('aceleracao')  if 'aceleracao'  in cols else 2

    # --- Normalização X (apenas scaler, sem recalcular features) ---------------
    tempos_tr  = sorted(df_tr_sort['time_index'].unique())
    tempos_val = sorted(df_val_sort['time_index'].unique())
    tempos_tgt = sorted(df_tgt_sort['time_index'].unique())

    scaler_X = StandardScaler()
    X_tr_sc  = scaler_X.fit_transform(df_tr_sort[cols].fillna(0))
    X_val_sc = scaler_X.transform(df_val_sort[cols].fillna(0))
    X_tgt_sc = scaler_X.transform(df_tgt_sort[cols].fillna(0))

    def _to_tensor(X_sc, tempos): return torch.tensor(X_sc.reshape(len(tempos), num_nodes, num_features), dtype=torch.float32).to(device)
    def _pop_tensor(df_sort, tempos): return torch.tensor(df_sort['log_pop'].values.reshape(len(tempos), num_nodes), dtype=torch.float32).to(device)
    def _casos_tensor(df_sort, tempos): return torch.tensor(df_sort[VARS['casos_dengue']].fillna(0).values.reshape(len(tempos), num_nodes), dtype=torch.float32).to(device)

    X_tr_t, X_val_t, X_tgt_t = _to_tensor(X_tr_sc, tempos_tr), _to_tensor(X_val_sc, tempos_val), _to_tensor(X_tgt_sc, tempos_tgt)
    logpop_tr_t, logpop_val_t, logpop_tgt_t = _pop_tensor(df_tr_sort, tempos_tr), _pop_tensor(df_val_sort, tempos_val), _pop_tensor(df_tgt_sort, tempos_tgt)
    Y_tr_t, Y_val_t = _casos_tensor(df_tr_sort, tempos_tr), _casos_tensor(df_val_sort, tempos_val)

    # modelo = PreditorNBAGCRN(num_nodes=num_nodes, num_features=num_features, embedding_dim=32).to(device)
    modelo = PreditorNBAGCRN(
        num_nodes=num_nodes,
        num_features=num_features,
        embedding_dim=32,
        idx_velocidade=idx_vel,
        idx_aceleracao=idx_acel,
    ).to(device)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, patience=5, factor=0.5, min_lr=1e-5)

    melhor_mae_val = float('inf')
    melhor_estado = None
    sem_melhora = 0

    barra = tqdm(range(epochs_max), desc=f'  Treino S{season_id}', unit='epoch', file=sys.stdout)
    modelo.train()
    for epoch in barra:
        h, loss_epoch = None, 0.0
        for t in range(len(tempos_tr) - 1):
            otimizador.zero_grad()
            log_mu, log_r, h = modelo(X_tr_t[t], logpop_tr_t[t], h)
            h = h.detach()
            alvo = Y_tr_t[t + 1].unsqueeze(-1)
            loss = nb_loss(log_mu, log_r, alvo)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            loss_epoch += loss.item()

        modelo.eval()
        with torch.no_grad():
            h_val, preds_mu = None, []
            for t in range(len(tempos_val)):
                log_mu_v, _, h_val = modelo(X_val_t[t], logpop_val_t[t], h_val)
                preds_mu.append(torch.exp(log_mu_v).squeeze(-1).cpu().numpy())

        preds_mu_np = np.array(preds_mu)
        mae_val = np.mean(np.abs(preds_mu_np - Y_val_t.cpu().numpy()))
        scheduler.step(mae_val)
        barra.set_postfix({'Loss': f'{loss_epoch / (len(tempos_tr) - 1):.4f}', 'MAE': f'{mae_val:.1f}'})

        if mae_val < melhor_mae_val - 1e-3:
            melhor_mae_val = mae_val
            melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= patience: break
        modelo.train()

    if melhor_estado: modelo.load_state_dict(melhor_estado)

    modelo.eval()
    preds_mu_tgt, preds_r_tgt = [], []
    with torch.no_grad():
        h_tgt = None
        for t in range(len(tempos_tgt)):
            log_mu_t, log_r_t, h_tgt = modelo(X_tgt_t[t], logpop_tgt_t[t], h_tgt)
            preds_mu_tgt.append(torch.exp(log_mu_t).squeeze(-1).cpu().numpy())
            preds_r_tgt.append(torch.exp(log_r_t).squeeze(-1).cpu().numpy())

    return {
        'season_id': season_id,
        'preds_mu_np': np.maximum(np.array(preds_mu_tgt), 0),
        'preds_r_np': np.maximum(np.array(preds_r_tgt), 1e-3),
        'df_tgt_sort': df_tgt_sort,
        'mapa_simples': mapa_simples,
    }

# ==============================================================================
# GERAÇÃO DA SUBMISSÃO MUNICIPAL (COM CASOS REAIS)
# ==============================================================================
def montar_submissao_agcrn(resultado_agcrn: dict) -> pd.DataFrame:
    df_tgt_sort  = resultado_agcrn['df_tgt_sort']
    mapa_simples = resultado_agcrn['mapa_simples']
    mu_agcrn     = resultado_agcrn['preds_mu_np']
    r_agcrn      = resultado_agcrn['preds_r_np']

    geocodes = mapa_simples[VARS['municipio']].values
    semanas  = sorted(df_tgt_sort[VARS['semana_epi']].unique())

    quantis_necessarios = {
        'lower_95': 0.025, 'lower_90': 0.05, 'lower_80': 0.10, 'lower_50': 0.25,
        'upper_50': 0.75,  'upper_80': 0.90, 'upper_90': 0.95, 'upper_95': 0.975,
    }

    rows = []
    for t_idx, semana in enumerate(semanas):
        mu_t = mu_agcrn[t_idx]
        r_t  = r_agcrn[t_idx]
        for n_idx, geocode in enumerate(geocodes):
            mu = float(mu_t[n_idx])
            r  = float(r_t[n_idx])

            pred_med = float(nb_quantile(np.array([mu]), np.array([r]), 0.50)[0])
            row = {
                VARS['municipio']:  geocode,
                VARS['semana_epi']: semana,
                'pred':             round(max(pred_med, 0), 2),
            }
            for col_name, q in quantis_necessarios.items():
                row[col_name] = round(max(float(nb_quantile(np.array([mu]), np.array([r]), q)[0]), 0), 2)
            rows.append(row)

    df_pred = pd.DataFrame(rows)

    df_pred['date_parsed'] = pd.to_datetime(df_pred[VARS['semana_epi']].astype(str) + '-0', format='%G%V-%w')
    df_pred['date'] = df_pred['date_parsed'].dt.strftime('%Y-%m-%d')

    todas_datas = pd.date_range(start=df_pred['date_parsed'].min(), end=df_pred['date_parsed'].max(), freq='W-SUN')
    grade = pd.MultiIndex.from_product([geocodes, todas_datas], names=[VARS['municipio'], 'date_parsed']).to_frame().reset_index(drop=True)
    grade['date'] = grade['date_parsed'].dt.strftime('%Y-%m-%d')

    df_pred = pd.merge(grade, df_pred, on=[VARS['municipio'], 'date', 'date_parsed'], how='left')

    # Merge target cases
    df_reais = df_tgt_sort[[VARS['municipio'], VARS['semana_epi'], VARS['casos_dengue']]].rename(columns={VARS['casos_dengue']: 'casos_reais'})
    df_reais['date_parsed'] = pd.to_datetime(df_reais[VARS['semana_epi']].astype(str) + '-0', format='%G%V-%w')
    
    df_pred = pd.merge(df_pred, df_reais[[VARS['municipio'], 'date_parsed', 'casos_reais']], on=[VARS['municipio'], 'date_parsed'], how='left')

    cols_pred = ['pred'] + list(quantis_necessarios.keys())
    for col in cols_pred:
        df_pred[col] = df_pred.groupby(VARS['municipio'])[col].transform(lambda x: x.interpolate('linear').bfill().ffill())
        df_pred[col] = np.maximum(df_pred[col], 0).round(2)

    colunas_finais = [
        VARS['municipio'], 'date', 'casos_reais', 'pred',
        'lower_50', 'upper_50', 'lower_80', 'upper_80',
        'lower_90', 'upper_90', 'lower_95', 'upper_95',
    ]
    return df_pred[colunas_finais].sort_values(by=[VARS['municipio'], 'date']).reset_index(drop=True)

# ==============================================================================
# RECONCILIAÇÃO HIERÁRQUICA MINT
# ==============================================================================
def reconciliar_mint(df_municipal: pd.DataFrame, df_estadual: pd.DataFrame, col_uf_code: str = 'uf_code') -> tuple:
    if not HF_DISPONIVEL: return df_municipal, df_estadual
    print("  [MinT] Reconciliação municipal → estadual...")

    # Exclui casos_reais da reconciliação (estes são observações ground truth, não predições)
    cols_quantis = ['pred', 'lower_50', 'upper_50', 'lower_80', 'upper_80', 'lower_90', 'upper_90', 'lower_95', 'upper_95']
    datas = sorted(df_municipal['date'].unique())
    
    df_mun_rec = df_municipal.copy()
    df_est_rec = df_estadual.copy()

    mapa_uf = df_municipal[[VARS['municipio'], col_uf_code]].drop_duplicates().set_index(VARS['municipio'])[col_uf_code].to_dict()
    geocodes, ufs = sorted(df_municipal[VARS['municipio']].unique()), sorted(df_estadual[col_uf_code].unique())

    n_uf, n_mun = len(ufs), len(geocodes)
    uf_idx, mun_idx = {uf: i for i, uf in enumerate(ufs)}, {g: i for i, g in enumerate(geocodes)}

    S = np.zeros((n_uf + n_mun, n_mun))
    for g in geocodes:
        if mapa_uf.get(g) in uf_idx: S[uf_idx[mapa_uf.get(g)], mun_idx[g]] = 1.0
        S[n_uf + mun_idx[g], mun_idx[g]] = 1.0

    STS = S.T @ S + 1e-6 * np.eye(n_mun)
    P = S @ np.linalg.solve(STS, S.T)

    for col in cols_quantis:
        Y_hat = np.zeros((n_uf + n_mun, len(datas)))
        for d_idx, data in enumerate(datas):
            mun_d = df_municipal[df_municipal['date'] == data].set_index(VARS['municipio'])
            est_d = df_estadual[df_estadual['date'] == data].set_index(col_uf_code)

            for i, uf in enumerate(ufs): Y_hat[i, d_idx] = est_d.loc[uf, col] if uf in est_d.index else 0.0
            for i, g in enumerate(geocodes): Y_hat[n_uf + i, d_idx] = mun_d.loc[g, col] if g in mun_d.index else 0.0

        Y_rec = np.maximum(P @ Y_hat, 0)

        for d_idx, data in enumerate(datas):
            for i, g in enumerate(geocodes):
                df_mun_rec.loc[(df_mun_rec['date'] == data) & (df_mun_rec[VARS['municipio']] == g), col] = round(float(Y_rec[n_uf + i, d_idx]), 2)
            for i, uf in enumerate(ufs):
                df_est_rec.loc[(df_est_rec['date'] == data) & (df_est_rec[col_uf_code] == uf), col] = round(float(Y_rec[i, d_idx]), 2)

    return df_mun_rec, df_est_rec

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # CONTROLE MANUAL DE EXECUÇÃO
    # Defina aqui qual Season deve ser executada (1, 2, 3 ou 4)
    # --------------------------------------------------------------------------
    SEASON_ALVO = 1

    # Carrega os arquivos JÁ PRÉ-PROCESSADOS do disco
    arquivo_treino = SEASONS[SEASON_ALVO]['train'].replace('.parquet', '_eng_shap_final.parquet')
    arquivo_teste  = SEASONS[SEASON_ALVO]['test'].replace('.parquet', '_eng_shap_final.parquet')

    print(f"Carregando datasets pré-processados:\n  - {arquivo_treino}\n  - {arquivo_teste}")
    df_train = pd.read_parquet(arquivo_treino)
    df_test  = pd.read_parquet(arquivo_teste)
    df_train = df_train[df_train[VARS['uf_code']] != 32].reset_index(drop=True)
    df_test  = df_test[df_test[VARS['uf_code']] != 32].reset_index(drop=True)
    
    print(f"Espírito Santo removido. Linhas restantes -> Treino: {len(df_train)} | Teste: {len(df_test)}")
    # O mapa de nós é extraído do dataframe
    mapa_simples = df_train[[VARS['municipio']]].drop_duplicates().sort_values(VARS['municipio']).reset_index(drop=True)
    mapa_simples['node_id'] = mapa_simples.index
    num_nodes = len(mapa_simples)

    # --------------------------------------------------------------------------
    # FEATURES DO MODELO (Física da epidemia + Top SHAP rankeadas anteriormente)
    # --------------------------------------------------------------------------

    db_conn = sqlite3.connect('resultados_agcrn.db')

    print(f"\n{'='*60}")
    print(f"  EXECUÇÃO ISOLADA: SEASON {SEASON_ALVO} ({SEASONS[SEASON_ALVO]['label']})")
    print(f"{'='*60}")
    
    # 1. Treinamento
    res_agcrn = treinar_e_validar_season(
        df_train=df_train, 
        df_test=df_test, 
        season_id=SEASON_ALVO, 
        encoder_means_globais={}, # Parâmetro residual da assinatura original (mantido vazio)
        num_nodes=num_nodes, 
        mapa_simples=mapa_simples, 
        epochs_max=150, 
        patience=15
    )
    
    # 2. DataFrame Municipal (com Casos Reais)
    df_mun = montar_submissao_agcrn(res_agcrn)
    
    # Mapeamento do uf_code (agora diretamente do df_train pré-processado)
    mapa_uf_code = df_train[[VARS['municipio'], VARS['uf_code']]].drop_duplicates()
    df_mun = df_mun.merge(mapa_uf_code, on=VARS['municipio'], how='left')

    # 3. Aggregation para DataFrame Estadual Baseline (com Casos Reais)
    cols_agregacao = ['pred', 'lower_50', 'upper_50', 'lower_80', 'upper_80', 'lower_90', 'upper_90', 'lower_95', 'upper_95', 'casos_reais']
    df_est = df_mun.groupby([VARS['uf_code'], 'date'])[cols_agregacao].sum().reset_index()

    # 4. Reconciliação MinT
    df_mun_rec, df_est_rec = reconciliar_mint(df_mun, df_est, col_uf_code=VARS['uf_code'])

    # 5. Salvar no Banco de Dados
    df_mun_rec['season'] = SEASON_ALVO
    df_est_rec['season'] = SEASON_ALVO
    
    df_mun_rec.to_sql('municipal_results', db_conn, if_exists='append', index=False)
    df_est_rec.to_sql('estadual_results', db_conn, if_exists='append', index=False)
    
    print(f"  Resultados S{SEASON_ALVO} salvos com sucesso no banco de dados SQLite.")

    db_conn.close()
    print("\n✅ Execução finalizada.")