"""
shap_walkforward_corrigido.py
=============================
Refaz a seleção de variáveis do Modelo 2 (SHAP walk-forward por regime SECA/PICO,
TestesomenteAGCRN.py, "SHAP BIFURCADO") sobre os bancos CORRIGIDOS S{N}_corrigido/,
nas 4 seasons, e gera as listas COLS_SECA / COLS_PICO para retestar no Modelo 2.

Mantido do método original
--------------------------
 - XGBoost (100 árvores, profundidade 5, hist) + TreeSHAP, só nas linhas de TREINO da season.
 - Walk-forward de 3 folds em janela expansiva sobre as semanas do regime; importância =
   mediana dos folds; Top 25 por regime e season; estabilidade = nº de seasons em que a
   variável entra no Top 25 (o original usava >= 3 de 4).

Corrigido em relação ao original
--------------------------------
 1. ALVO ALINHADO AO MODELO: o AGCRN usa a linha da semana t para prever os casos de t+1
    (X[t] -> Y[t+1] no treino). O SHAP antigo explicava casos.x da PRÓPRIA semana t, o que
    inflava variáveis calculadas com y(t) (casos_suavizados_t1, velocidade, aceleracao).
    Aqui: features da linha t, alvo = casos de t+1.
 2. ESCALA: o antigo regredia casos ABSOLUTOS com erro quadrático (as capitais dominam e
    ganham as variáveis de "tamanho da cidade"). Aqui: objetivo Poisson com offset log_pop,
    a mesma estrutura da NB do modelo (log mu = log_pop + f(x)).
 3. REGIMES iguais aos do modelo: PICO = semana-alvo >= 47 ou <= 25 (o antigo usava 49–22).
 4. REPRESENTAÇÃO do FINAL_NN: o SHAP avalia as variáveis como a rede as recebe
    (log-incidência em vez de casos, log da população, log1p dos acessos Afya), então os
    nomes selecionados já são colunas do FINAL_NN / FINAL_NN_v2.
 5. UTILIDADE FORA DA AMOSTRA: |SHAP| mede o quanto o modelo USA a variável, não se ela
    ajuda. Cada variável também passa por permutação na validação (aumento da deviance de
    Poisson ao embaralhá-la). Só entra na lista quem piora a validação ao ser embaralhada
    em >= 2 dos 3 folds.
 6. REDUNDÂNCIA: percorrendo o ranking, uma variável com |correlação| > 0,95 com outra já
    escolhida é descartada (ex.: imunidade_ewm x imunidade_ewm_t1, que são quase idênticas).
 7. CANDIDATAS NOVAS: incidência observada (não suavizada) em t, t-1, t-2 e nos vizinhos
    W1/W2; chikungunya dos vizinhos em incidência; todos os lags climáticos 2–5; ENSO/IOD/PDO
    com datas corrigidas; Afya como soma semanal (NaN antes de 2021, que o XGBoost trata
    como ausente, equivalente à flag afya_disponivel).

Uso:    python shap_walkforward_corrigido.py                 (4 seasons, ~1 h em 6 núcleos)
        python shap_walkforward_corrigido.py --seasons 1 --max-nos 600 --arvores 20   (teste rápido)
Saídas nesta pasta: relatorio_shap_corrigido.md, listas_features_modelo2.py,
        colunas_extras_nn.json, importancia_*.csv
Dependências: numpy, pandas, pyarrow, scipy, xgboost (TreeSHAP nativo: pred_contribs=True,
        o mesmo algoritmo do shap.TreeExplainer).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from scipy import sparse

DIR = Path(__file__).resolve().parent
BASE = DIR.parent
SEED = 42
N_FOLDS = 3
TOP_N = 25
MIN_SEASONS = 3                 # estável = Top 25 em >= 3 das 4 seasons (critério original)
CORR_MAX = 0.95
AMOSTRA_VAL = 200_000           # linhas da validação usadas no SHAP e na permutação
AMOSTRA_CORR = 300_000
LIMITE_PICO = (47, 25)          # igual ao Modelo 2: pico = semana >= 47 ou <= 25

# ----------------------------------------------------------------------------- candidatas
VARS_CLIMA = ['temp_min', 'temp_med', 'temp_max', 'precip_min', 'precip_med', 'precip_max',
              'pressure_min', 'pressure_med', 'pressure_max', 'rel_humid_min', 'rel_humid_med',
              'rel_humid_max', 'thermal_range', 'rainy_days']
LAGS_CLIMA = [f'{v}_lag{k}' for v in VARS_CLIMA + ['vpd'] for k in (2, 3, 4, 5)]
OCEANO = [f'{v}{s}' for v in ('enso', 'iod', 'pdo') for s in ('', '_lag6', '_lag7', '_lag8', '_lag9')]
DIRETAS = (VARS_CLIMA + ['vpd', 'amplitude_termica_dinamica3', 'amplitude_termica_dinamica2',
                         'volatilidade_temp_4w', 'volatilidade_precip_4w', 'hti_lag3', 'hti_lag2']
           + LAGS_CLIMA + OCEANO
           + ['temp_med_forecast', 'umid_med_forecast', 'precip_tot_forecast',
              'temp_med_forecast_t1', 'umid_med_forecast_t1', 'precip_tot_forecast_t1', 'pressure_min_t1']
           + ['imunidade_ewm', 'imunidade_ewm_t1', 'seno_semana', 'cosseno_semana', 'velocidade', 'aceleracao',
              'koppen_encoded', 'biome_encoded', 'afya_log_diff']
           + ['w1_imunidade_t1', 'w1_velocidade_t1', 'w1_aceleracao_t1', 'w2_velocidade_t1', 'w2_aceleracao_t1',
              'w1_temp_forecast_t1', 'w1_umid_forecast_t1', 'w1_precip_forecast_t1', 'w2_temp_forecast_t1',
              'w2_umid_forecast_t1', 'w2_precip_forecast_t1', 'w1_pressure_min_t1'])
# transformadas exatamente como no FINAL_NN (refazer_season.py, dicionário `transform`)
TRANSFORMADAS = ['log_pop_t1', 'w1_log_pop_t1', 'w2_log_pop_t1', 'log_inc_dengue_t1', 'log_inc_chiku',
                 'log_inc_chiku_t1', 'w1_log_inc_dengue_t1', 'w2_log_inc_dengue_t1', 'access_count',
                 'access_count_t1', 'w1_acessos_afya_t1', 'w1_log_inc_chiku_t1', 'w2_log_inc_chiku_t1',
                 'log_inc_dengue_obs', 'log_inc_dengue_obs_lag1', 'log_inc_dengue_obs_lag2',
                 'w1_log_inc_dengue_obs', 'w2_log_inc_dengue_obs']
BRUTAS_P_TRANSF = ['population_t1', 'w1_population_t1', 'w2_population_t1', 'casos_suavizados_t1',
                   'casos_chikungunya', 'casos_chikungunya_t1', 'w1_dengue_t1', 'w2_dengue_t1',
                   'access_count', 'access_count_t1', 'w1_acessos_afya_t1', 'w1_chikungunya_t1',
                   'w2_chikungunya_t1']
CANDIDATAS = DIRETAS + TRANSFORMADAS
# sondas de ruído (ideia do Boruta): uma variável só é "útil" se, embaralhada, piorar a validação
# MAIS do que qualquer sonda. Elas nunca entram nas listas.
SONDAS = ['sonda_normal', 'sonda_uniforme', 'sonda_embaralhada']
COLUNAS = CANDIDATAS + SONDAS

FAMILIA = {c: 'sonda (ruído)' for c in SONDAS}
for _c in CANDIDATAS:
    if 'dengue' in _c or _c in ('velocidade', 'aceleracao') or 'velocidade' in _c or 'aceleracao' in _c:
        FAMILIA[_c] = 'dengue (histórico)'
    elif 'imunidade' in _c:
        FAMILIA[_c] = 'imunidade'
    elif 'chik' in _c:
        FAMILIA[_c] = 'chikungunya'
    elif 'afya' in _c or 'access' in _c:
        FAMILIA[_c] = 'Afya'
    elif 'pop' in _c:
        FAMILIA[_c] = 'população'
    elif 'forecast' in _c:
        FAMILIA[_c] = 'clima previsto'
    elif _c.startswith(('enso', 'iod', 'pdo')):
        FAMILIA[_c] = 'oceano'
    elif 'semana' in _c:
        FAMILIA[_c] = 'sazonalidade'
    elif 'encoded' in _c:
        FAMILIA[_c] = 'estático'
    else:
        FAMILIA[_c] = 'clima observado'

# listas atuais do notebook do Modelo 2 (para comparação)
COLS_SECA_ATUAL = ['velocidade', 'aceleracao', 'access_count', 'seno_semana', 'temp_min', 'w2_velocidade_t1',
                   'w1_log_pop_t1', 'w1_imunidade_t1', 'w2_log_inc_dengue_t1', 'w1_log_inc_dengue_t1',
                   'imunidade_ewm', 'imunidade_ewm_t1', 'log_inc_dengue_t1', 'log_inc_chiku', 'log_pop_t1',
                   'rel_humid_max', 'log_inc_chiku_t1', 'cosseno_semana', 'w1_velocidade_t1', 'temp_med',
                   'koppen_encoded']
COLS_PICO_ATUAL = ['velocidade', 'aceleracao', 'access_count', 'seno_semana', 'w2_velocidade_t1', 'w1_log_pop_t1',
                   'w2_aceleracao_t1', 'access_count_t1', 'w1_log_inc_dengue_t1', 'imunidade_ewm',
                   'imunidade_ewm_t1', 'log_inc_dengue_t1', 'log_inc_chiku', 'log_pop_t1', 'cosseno_semana',
                   'w1_velocidade_t1', 'w1_aceleracao_t1', 'w2_log_inc_dengue_t1', 'w2_temp_forecast_t1',
                   'w1_temp_forecast_t1', 'log_inc_chiku_t1', 'umid_med_forecast', 'temp_min', 'koppen_encoded',
                   'w2_umid_forecast_t1', 'temp_med_forecast', 'w2_precip_forecast_t1']

T0 = time.time()


def log(msg: str) -> None:
    print(f'[{time.time() - T0:6.0f}s] {msg}', flush=True)


def log_inc(c, p):
    return np.log1p(np.asarray(c, np.float64) / np.clip(np.asarray(p, np.float64), 1, None) * 1e5).astype(np.float32)


# ----------------------------------------------------------------------------- dados
def ler_season(s: int, max_nos: int | None) -> dict:
    pasta = BASE / f'S{s}_corrigido'
    arq = pasta / f'train{s}_dengue_eng_COMPLETO_spatial.parquet'
    cols = ['node_id', 'epiweek', 'casos.x', 'population', 'log_pop'] + \
        sorted(set(DIRETAS) | set(BRUTAS_P_TRANSF))
    filtros = [('node_id', '<', max_nos)] if max_nos else None
    log(f'S{s}: lendo {arq.name} ({len(cols)} colunas{", " + str(max_nos) + " municípios" if max_nos else ""}) ...')
    tab = pq.read_table(arq, columns=cols, filters=filtros)
    ew = tab['epiweek'].to_numpy()
    nos = tab['node_id'].to_numpy()
    semanas = np.unique(ew)
    T, N = len(semanas), len(np.unique(nos))
    assert len(ew) == T * N, 'grade desbalanceada'
    assert (np.diff(ew.reshape(T, N)[:, 0]) > 0).all() and (nos.reshape(T, N) == nos[:N]).all(), 'ordem (date, node_id)'
    D = {c: tab[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in cols if c not in ('node_id', 'epiweek')}
    del tab

    # --- W1/W2 e incidência observada dos vizinhos (grade completa, depois recorta)
    W1 = sparse.load_npz(pasta / f'W1_queen_season{s}.npz')
    W2 = sparse.load_npz(pasta / f'W2_season{s}.npz')
    if max_nos:
        full = pq.read_table(arq, columns=['casos.x', 'population'])
        Nf = W1.shape[0]
        cas_f = full['casos.x'].to_numpy().astype(np.float64).reshape(-1, Nf)
        pop_f = full['population'].to_numpy().astype(np.float64).reshape(-1, Nf)
        del full
    else:
        cas_f = D['casos.x'].astype(np.float64).reshape(T, N)
        pop_f = D['population'].astype(np.float64).reshape(T, N)
    X = {}
    for nome, W in (('w1_log_inc_dengue_obs', W1), ('w2_log_inc_dengue_obs', W2)):
        c = (W @ cas_f.T).T[:, :N]
        p = (W @ pop_f.T).T[:, :N]
        X[nome] = log_inc(c, p).reshape(-1)
    del cas_f, pop_f

    cas = D['casos.x'].reshape(T, N).astype(np.float64)
    pop = D['population'].reshape(T, N).astype(np.float64)

    def lag(m, k):
        out = np.full_like(m, np.nan)
        out[k:] = m[:-k]
        return out
    X['log_inc_dengue_obs'] = log_inc(cas, pop).reshape(-1)
    X['log_inc_dengue_obs_lag1'] = log_inc(lag(cas, 1), lag(pop, 1)).reshape(-1)
    X['log_inc_dengue_obs_lag2'] = log_inc(lag(cas, 2), lag(pop, 2)).reshape(-1)
    X['log_pop_t1'] = np.log(np.clip(D['population_t1'], 1, None))
    X['w1_log_pop_t1'] = np.log(np.clip(D['w1_population_t1'], 1, None))
    X['w2_log_pop_t1'] = np.log(np.clip(D['w2_population_t1'], 1, None))
    X['log_inc_dengue_t1'] = log_inc(D['casos_suavizados_t1'], D['population_t1'])
    X['log_inc_chiku'] = log_inc(D['casos_chikungunya'], D['population'])
    X['log_inc_chiku_t1'] = log_inc(D['casos_chikungunya_t1'], D['population_t1'])
    X['w1_log_inc_dengue_t1'] = log_inc(D['w1_dengue_t1'], D['w1_population_t1'])
    X['w2_log_inc_dengue_t1'] = log_inc(D['w2_dengue_t1'], D['w2_population_t1'])
    X['w1_log_inc_chiku_t1'] = log_inc(D['w1_chikungunya_t1'], D['w1_population_t1'])
    X['w2_log_inc_chiku_t1'] = log_inc(D['w2_chikungunya_t1'], D['w2_population_t1'])
    for c in ('access_count', 'access_count_t1', 'w1_acessos_afya_t1'):
        X[c] = np.log1p(D[c])                                  # NaN (antes de 2021) continua NaN
    for c in DIRETAS:
        X[c] = D[c]

    # --- alvo: casos de t+1 (a última semana de treino não tem alvo dentro do treino)
    y_next = cas[1:].reshape(-1)
    n_lin = (T - 1) * N
    M = np.empty((n_lin, len(COLUNAS)), np.float32)
    for j, c in enumerate(CANDIDATAS):
        M[:, j] = X[c][:n_lin]
    rs = np.random.default_rng(SEED + 100 * s)
    k = len(CANDIDATAS)
    M[:, k] = rs.standard_normal(n_lin, dtype=np.float32)
    M[:, k + 1] = rs.random(n_lin, dtype=np.float32)
    M[:, k + 2] = X['log_inc_dengue_t1'][:n_lin][rs.permutation(n_lin)]   # mesma distribuição, sem relação
    del X, D
    gc.collect()
    sem_alvo = semanas[1:]
    return {
        'X': M, 'y': y_next.astype(np.float32),
        'offset': np.log(np.clip(pop[:-1].reshape(-1), 1, None)).astype(np.float32),   # log_pop da linha t (como no modelo)
        'sem_linha': np.repeat(np.arange(T - 1), N),        # índice da semana-ALVO (t+1) de cada linha
        'ew_alvo': sem_alvo, 'N': N, 'T': T,
    }


def regime_da_semana(ew: np.ndarray) -> np.ndarray:
    w = ew % 100
    return np.where((w >= LIMITE_PICO[0]) | (w <= LIMITE_PICO[1]), 'pico', 'seca')


def deviance_poisson(y, mu) -> float:
    y = y.astype(np.float64)
    mu = np.clip(mu.astype(np.float64), 1e-12, None)
    termo = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(np.mean(2 * (termo - (y - mu))))


# ----------------------------------------------------------------------------- SHAP por fold
def rodar_regime(d: dict, regime: str, s: int, n_arvores: int, nthread: int) -> list[dict]:
    rng = np.random.default_rng(SEED + s)
    reg = regime_da_semana(d['ew_alvo'])
    sem_reg = np.flatnonzero(reg == regime)                 # semanas-alvo do regime, em ordem
    n = len(sem_reg)
    fs = n // (N_FOLDS + 1)
    linhas = []
    for fold in range(N_FOLDS):
        corte = (fold + 1) * fs
        s_tr, s_vl = sem_reg[:corte], sem_reg[corte:corte + fs]
        m_tr = np.isin(d['sem_linha'], s_tr)
        m_vl = np.flatnonzero(np.isin(d['sem_linha'], s_vl))
        if len(m_vl) > AMOSTRA_VAL:
            m_vl = np.sort(rng.choice(m_vl, AMOSTRA_VAL, replace=False))
        y_tr = d['y'][m_tr]
        taxa0 = np.log(y_tr.sum() / np.exp(d['offset'][m_tr].astype(np.float64)).sum())
        marg_tr = d['offset'][m_tr] + taxa0
        dtr = xgb.QuantileDMatrix(d['X'][m_tr], label=y_tr, base_margin=marg_tr, max_bin=256)
        del m_tr
        params = dict(objective='count:poisson', max_depth=5, eta=0.3, tree_method='hist',
                      nthread=nthread, seed=SEED, verbosity=0)
        t = time.time()
        bst = xgb.train(params, dtr, num_boost_round=n_arvores)
        n_tr = dtr.num_row()
        del dtr
        gc.collect()
        Xv, yv = d['X'][m_vl], d['y'][m_vl]
        mv = d['offset'][m_vl] + taxa0
        mu = bst.inplace_predict(Xv, base_margin=mv)
        dev_mod, dev_base = deviance_poisson(yv, mu), deviance_poisson(yv, np.exp(mv))
        contrib = bst.predict(xgb.DMatrix(Xv, base_margin=mv), pred_contribs=True)
        shap_abs = np.abs(contrib[:, :-1]).mean(axis=0)
        del contrib
        perm = np.zeros(len(COLUNAS))
        for j in range(len(COLUNAS)):
            col = Xv[:, j].copy()
            Xv[:, j] = col[rng.permutation(len(col))]
            perm[j] = deviance_poisson(yv, bst.inplace_predict(Xv, base_margin=mv)) - dev_mod
            Xv[:, j] = col
        log(f'  S{s} {regime.upper()} fold {fold + 1}/{N_FOLDS}: treino {n_tr:,} linhas '
            f'(EW alvo {d["ew_alvo"][s_tr[0]]}–{d["ew_alvo"][s_tr[-1]]}), validação EW '
            f'{d["ew_alvo"][s_vl[0]]}–{d["ew_alvo"][s_vl[-1]]} · deviance {dev_mod:.3f} vs só-offset {dev_base:.3f} '
            f'(ganho {1 - dev_mod / dev_base:.1%}) · {time.time() - t:.0f}s')
        for j, c in enumerate(COLUNAS):
            linhas.append({'season': s, 'regime': regime, 'fold': fold + 1, 'feature': c,
                           'shap_abs': float(shap_abs[j]), 'shap_share': float(shap_abs[j] / shap_abs.sum()),
                           'perm_delta_dev': float(perm[j]), 'perm_delta_rel': float(perm[j] / dev_mod),
                           'dev_modelo': dev_mod, 'dev_so_offset': dev_base,
                           'val_ini': int(d['ew_alvo'][s_vl[0]]), 'val_fim': int(d['ew_alvo'][s_vl[-1]])})
        del Xv, yv, mv, bst
        gc.collect()
    return linhas


def correlacao(d: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(d['X']), min(AMOSTRA_CORR, len(d['X'])), replace=False)
    A = pd.DataFrame(d['X'][idx], columns=COLUNAS)
    A = A.fillna(A.mean())
    return A.corr().abs().fillna(0)


def selecionar(det: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    # limiares de ruído por fold: o maior valor obtido por uma sonda (permutação limitada a >= 0)
    son = det[det.feature.isin(SONDAS)].groupby('fold')
    lim_perm = son['perm_delta_dev'].max().clip(lower=0)
    lim_shap = son['shap_share'].max()
    det = det.assign(acima_ruido=det['perm_delta_dev'] > det['fold'].map(lim_perm))
    g = det.groupby('feature')
    t = pd.DataFrame({
        'shap_med': g['shap_share'].median(), 'shap_ultimo_fold': det[det.fold == N_FOLDS].set_index('feature')['shap_share'],
        'perm_med': g['perm_delta_rel'].median(), 'folds_acima_ruido': g['acima_ruido'].sum().astype(int),
    }).sort_values('shap_med', ascending=False)
    t['util'] = (t['folds_acima_ruido'] >= 2) & (t['shap_med'] > lim_shap.median()) & ~t.index.isin(SONDAS)
    t['redundante_de'] = ''
    t['selecionada'] = False
    escolhidas = []
    for f in t.index:
        if not t.at[f, 'util']:
            continue
        par = [e for e in escolhidas if corr.at[f, e] > CORR_MAX]
        if par:
            t.at[f, 'redundante_de'] = par[0]
            continue
        if len(escolhidas) < TOP_N:
            escolhidas.append(f)
            t.at[f, 'selecionada'] = True
    t['rank'] = np.nan
    t.loc[escolhidas, 'rank'] = np.arange(1, len(escolhidas) + 1)
    return t


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', type=int, nargs='+', default=[1, 2, 3, 4])
    ap.add_argument('--max-nos', type=int, default=None, help='só p/ teste rápido: primeiros K municípios')
    ap.add_argument('--arvores', type=int, default=100)
    ap.add_argument('--nthread', type=int, default=6)
    ap.add_argument('--saida', default=str(DIR))
    a = ap.parse_args()
    out = Path(a.saida)
    out.mkdir(exist_ok=True)

    det_all, sel_all, corrs = [], [], {}
    for s in a.seasons:
        d = ler_season(s, a.max_nos)
        log(f'S{s}: matriz {d["X"].shape[0]:,} linhas x {d["X"].shape[1]} candidatas '
            f'({d["N"]} municípios x {d["T"] - 1} semanas-alvo) · {d["X"].nbytes / 1e9:.1f} GB')
        corr = correlacao(d)
        corrs[s] = corr
        for regime in ('seca', 'pico'):
            det = pd.DataFrame(rodar_regime(d, regime, s, a.arvores, a.nthread))
            det_all.append(det)
            sel = selecionar(det, corr)
            sel.insert(0, 'regime', regime)
            sel.insert(0, 'season', s)
            sel_all.append(sel.reset_index().rename(columns={'index': 'feature'}))
            pd.concat(det_all).to_csv(out / 'importancia_detalhada_folds.csv', index=False)
            pd.concat(sel_all).to_csv(out / 'importancia_por_season.csv', index=False)
        del d
        gc.collect()

    det = pd.concat(det_all)
    sel = pd.concat(sel_all)
    corr_media = sum(corrs.values()) / len(corrs)
    ns = len(a.seasons)
    min_s = MIN_SEASONS if ns == 4 else max(1, int(np.ceil(ns * 0.75)))

    # ---- estabilidade entre seasons
    sel = sel[~sel.feature.isin(SONDAS)]
    est = sel.groupby(['regime', 'feature']).agg(
        n_seasons=('selecionada', 'sum'), shap_medio=('shap_med', 'mean'), perm_medio=('perm_med', 'mean'),
        rank_medio=('rank', 'mean'), util_seasons=('util', 'sum')).reset_index()
    est['familia'] = est['feature'].map(FAMILIA)
    est = est.sort_values(['regime', 'n_seasons', 'shap_medio'], ascending=[True, False, False])
    est.to_csv(out / 'estabilidade_entre_seasons.csv', index=False)

    listas = {}
    for regime in ('seca', 'pico'):
        e = est[est.regime == regime]
        estaveis = e[e.n_seasons >= min_s].sort_values('shap_medio', ascending=False)['feature'].tolist()
        # redundância também entre seasons (usa a correlação média)
        final = []
        for f in estaveis:
            if not any(corr_media.at[f, g] > CORR_MAX for g in final):
                final.append(f)
        listas[regime] = final

    COLS_FISICA = ['velocidade', 'aceleracao']       # o modelo precisa delas (índices da cabeça r)
    seca = COLS_FISICA + [f for f in listas['seca'] if f not in COLS_FISICA]
    pico = COLS_FISICA + [f for f in listas['pico'] if f not in COLS_FISICA]

    # colunas que não existem no FINAL_NN atual -> FINAL_NN_v2
    s_ref = a.seasons[0]
    nn_cols = pq.ParquetFile(BASE / f'S{s_ref}_corrigido' / f'train{s_ref}_FINAL_NN.parquet').schema_arrow.names
    extras = [c for c in dict.fromkeys(seca + pico) if c not in nn_cols]
    (out / 'colunas_extras_nn.json').write_text(json.dumps(extras, indent=1), encoding='utf-8')

    def fmt(nome, lst, coment):
        linhas = [f'# {coment}', f'{nome} = COLS_FISICA + [']
        corpo = [f for f in lst if f not in COLS_FISICA]
        for i in range(0, len(corpo), 4):
            linhas.append('    ' + ', '.join(f"'{c}'" for c in corpo[i:i + 4]) + ',')
        linhas.append(']')
        return '\n'.join(linhas)
    py = ['# Gerado por SHAP_corrigido/shap_walkforward_corrigido.py em '
          f'{pd.Timestamp.now():%Y-%m-%d %H:%M} (seasons {a.seasons})',
          f'# Critério: Top {TOP_N} por season e regime, acima das sondas de ruído na permutação (>= 2 de 3 folds), sem redundância '
          f'(|r| > {CORR_MAX}), presente em >= {min_s} de {ns} seasons. Ordem: importância SHAP média.',
          f'# Requer train{{N}}_FINAL_NN_v2.parquet se usar alguma destas colunas novas: {extras}', '',
          "COLS_FISICA = ['velocidade', 'aceleracao']", '',
          fmt('COLS_SECA', seca, f'🌵 SECA ({len(seca)} features)'), '',
          fmt('COLS_PICO', pico, f'🌊 PICO ({len(pico)} features)'), '']
    (out / 'listas_features_modelo2.py').write_text('\n'.join(py), encoding='utf-8')

    relatorio(out, det, sel, est, seca, pico, extras, ns, min_s, a)
    log('Fim.')
    return 0


def relatorio(out, det, sel, est, seca, pico, extras, ns, min_s, a) -> None:
    L = ['# SHAP walk-forward corrigido — seleção de features do Modelo 2', '',
         f'Gerado por `shap_walkforward_corrigido.py` em {pd.Timestamp.now():%Y-%m-%d %H:%M} · seasons {a.seasons} · '
         f'{len(CANDIDATAS)} candidatas + {len(SONDAS)} sondas de ruído · {a.arvores} árvores · tempo {time.time() - T0:.0f}s'
         + (f' · **TESTE com {a.max_nos} municípios**' if a.max_nos else ''), '',
         '## Qualidade dos modelos auxiliares (validação de cada fold)', '',
         'Deviance de Poisson na validação: XGBoost vs. modelo só com offset (taxa média × população).', '',
         '| Season | Regime | Fold | Validação (EW alvo) | Deviance modelo | Só offset | Ganho |', '|---|---|---|---|---|---|---|']
    q = det.drop_duplicates(['season', 'regime', 'fold'])
    for _, r in q.iterrows():
        L.append(f'| S{r.season} | {r.regime} | {r.fold} | {r.val_ini}–{r.val_fim} | {r.dev_modelo:.3f} | '
                 f'{r.dev_so_offset:.3f} | {1 - r.dev_modelo / r.dev_so_offset:.1%} |')
    for regime, lst, atual in (('seca', seca, COLS_SECA_ATUAL), ('pico', pico, COLS_PICO_ATUAL)):
        e = est[est.regime == regime].set_index('feature')
        L += ['', f'## Regime {regime.upper()} — lista recomendada ({len(lst)} features)', '',
              f'| # | Feature | Família | Seasons no Top {TOP_N} | SHAP médio (% do total) | Permutação (Δ deviance rel.) | Na lista atual? | Coluna nova? |',
              '|---|---|---|---|---|---|---|---|']
        for i, f in enumerate(lst, 1):
            L.append(f'| {i} | `{f}` | {FAMILIA[f]} | {int(e.at[f, "n_seasons"])}/{ns} | {e.at[f, "shap_medio"]:.1%} | '
                     f'{e.at[f, "perm_medio"]:+.2%} | {"sim" if f in atual else "**não**"} | {"**sim**" if f in extras else ""} |')
        saem = [f for f in atual if f not in lst]
        L += ['', f'**Saem da lista atual ({len(saem)}):**', '',
              '| Feature | Seasons no Top | SHAP médio | Permutação | Por quê |', '|---|---|---|---|---|']
        for f in saem:
            if f not in e.index:
                L.append(f'| `{f}` | – | – | – | não é candidata |')
                continue
            r = e.loc[f]
            rs = sel[(sel.regime == regime) & (sel.feature == f)]
            red = [x for x in rs['redundante_de'] if x]
            motivo = []
            if r.util_seasons < ns / 2:
                motivo.append('embaralhá-la não piora a validação mais do que embaralhar ruído')
            if red:
                motivo.append(f'redundante com `{red[0]}` (|r| > {CORR_MAX})')
            if not motivo:
                motivo.append(f'fora do Top {TOP_N} na maioria das seasons')
            L.append(f'| `{f}` | {int(r.n_seasons)}/{ns} | {r.shap_medio:.1%} | {r.perm_medio:+.2%} | {"; ".join(motivo)} |')
    # sondas
    son = det[det.feature.isin(SONDAS)].groupby(['regime', 'feature'])[['shap_share', 'perm_delta_rel']].max()
    L += ['', '## Sondas de ruído (limiar)', '',
          'Colunas aleatórias incluídas nas candidatas. Uma variável só é considerada útil se, embaralhada, '
          'piorar a validação mais do que a pior sonda em pelo menos 2 dos 3 folds, e se tiver |SHAP| mediano '
          'acima do das sondas. Máximos observados:', '', '| Regime | Sonda | SHAP máx. | Permutação máx. |', '|---|---|---|---|']
    for (rg, f), r in son.iterrows():
        L.append(f'| {rg} | `{f}` | {r.shap_share:.2%} | {r.perm_delta_rel:+.3%} |')
    # famílias
    L += ['', '## Importância por família (SHAP somado, média das seasons e folds)', '',
          '| Família | SECA | PICO |', '|---|---|---|']
    dfam = det.assign(familia=det.feature.map(FAMILIA)).groupby(['regime', 'familia', 'season', 'fold'])['shap_share'].sum() \
        .groupby(['regime', 'familia']).mean().unstack(0).fillna(0).sort_values('pico', ascending=False)
    for fam, r in dfam.iterrows():
        L.append(f'| {fam} | {r.get("seca", 0):.1%} | {r.get("pico", 0):.1%} |')
    # top 40 por season
    L += ['', '## Ranking completo por season (Top 25 selecionadas)', '']
    for (s, regime), g in sel.groupby(['season', 'regime']):
        g = g[g.selecionada].sort_values('rank')
        L.append(f'- **S{s} {regime}**: ' + ', '.join(f'`{f}`' for f in g.feature))
    L += ['', 'Arquivos: `importancia_detalhada_folds.csv` (SHAP e permutação por fold), `importancia_por_season.csv`, '
          '`estabilidade_entre_seasons.csv`, `listas_features_modelo2.py`, `colunas_extras_nn.json`.']
    (out / 'relatorio_shap_corrigido.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
