"""Replica o piloto do Felipe (github felipe-matsuoka123/Dengue) nos dados por UF do Victor
e mede SMAPE em vários protocolos. Leve: ~1-3 min de CPU."""
import glob, warnings
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore')

BASE = r'F:/AmbientePython/Projetos/data_imdc_2026'
D = pd.read_parquet(f'{BASE}/UF/dados/tabela_uf_semanal.parquet', columns=['uf_code', 'uf', 'date', 'casos', 'pop'])
D['date'] = pd.to_datetime(D['date'])
Y = D.pivot(index='date', columns='uf', values='casos').sort_index()
P = D.pivot(index='date', columns='uf', values='pop').sort_index()
COD = D.drop_duplicates('uf').set_index('uf_code')['uf'].to_dict()
UFS = sorted(Y.columns)
LAGS = [1, 2, 4, 8, 52]


def painel(h):
    """Features idênticas às do piloto, para alvo na semana t e origem t-(h-1)."""
    s = h - 1
    f = {f'lag_{k}': Y.shift(s + k) for k in LAGS}
    f['media_4'] = sum(Y.shift(s + k) for k in range(1, 5)) / 4
    f['media_8'] = sum(Y.shift(s + k) for k in range(1, 9)) / 8
    longo = pd.concat({k: v.stack(dropna=False) for k, v in f.items()}, axis=1)
    longo['populacao'] = P.stack(dropna=False)
    longo['y'] = Y.stack(dropna=False)
    longo = longo.reset_index().rename(columns={'level_0': 'date', 'level_1': 'uf'})
    origem = longo['date'] - pd.to_timedelta(7 * s, 'D')
    ang = 2 * np.pi * (origem.dt.dayofyear - 1) / 365.2425
    longo['sen'], longo['cos'] = np.sin(ang), np.cos(ang)
    for u in UFS:
        longo[f'uf_{u}'] = (longo['uf'] == u).astype(float)
    # Exclusões do piloto: ES a partir de 2020 (alerta de cobertura) e sem lag 1.
    longo = longo[~((longo.uf == 'ES') & (longo.date + pd.Timedelta(days=6) >= '2020-01-01'))]
    return longo.dropna(subset=['y', 'lag_1'])


X_COLS = [f'lag_{k}' for k in LAGS] + ['media_4', 'media_8', 'sen', 'cos', 'populacao'] + [f'uf_{u}' for u in UFS]


def modelos():
    return dict(
        xgboost=XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, objective='reg:squarederror',
                             tree_method='hist', n_jobs=2, random_state=42),
        lightgbm=LGBMRegressor(n_estimators=300, max_depth=5, num_leaves=31, learning_rate=0.05, n_jobs=2,
                               random_state=42, deterministic=True, force_col_wise=True, verbosity=-1))


def smape_vec(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    return np.where(den > 0, 2 * np.abs(p - y) / np.where(den > 0, den, 1), 0.0)


def metricas(df, col='p'):
    y, p = df['y'].values, df[col].values
    s = smape_vec(y, p)
    nac = df.groupby('date')[['y', col]].sum()
    return dict(n=len(df), MAE=np.abs(p - y).mean(), WAPE=100 * np.abs(p - y).sum() / y.sum(),
                sMAPE=100 * s.mean(),                      # convenção 0–200%
                sMAPE_0a100=50 * s.mean(),                 # convenção |e|/(|y|+|p|), 0–100%
                sMAPE_mediaUF=100 * df.assign(s=s).groupby('uf')['s'].mean().mean(),
                sMAPE_mediana=100 * np.median(s),
                sMAPE_Brasil=100 * smape_vec(nac['y'], nac[col]).mean())


def ajustar_prever(tr, te, nomes=('xgboost', 'lightgbm')):
    out = {'persistencia': te['lag_1'].values}
    ms = modelos()
    for n in nomes:
        ms[n].fit(tr[X_COLS], tr['y'])
        out[n] = np.maximum(0, ms[n].predict(te[X_COLS]))
    return out


def dentro(df, ini, fim):
    return df[(df.date >= ini) & (df.date + pd.Timedelta(days=6) <= fim)]


linhas = []
# ---------- A) Protocolo do Felipe: h=1, ano civil, teste 2023–2024 ----------
L1 = painel(1)
for ano in [2019, 2020, 2021, 2022, 2023]:
    fim_av = '2024-12-31' if ano == 2023 else f'{ano}-12-31'
    tr = dentro(L1, '2015-01-01', f'{ano - 1}-12-31')
    te = dentro(L1, f'{ano}-01-01', fim_av).copy()
    for n, p in ajustar_prever(tr, te).items():
        linhas.append(dict(protocolo='A_felipe_h1', avaliacao='teste_2023-24' if ano == 2023 else f'valid_{ano}',
                           modelo=n, **metricas(te.assign(p=p))))

# ---------- B) Mesmo modelo na temporada 1 do IMDC (SE41/2022–SE40/2023) ----------
S1_INI, S1_FIM, CORTE = pd.Timestamp('2022-10-09'), pd.Timestamp('2023-10-07'), pd.Timestamp('2022-06-26')  # corte: dados até SE25/2022
for h in [1, 4, 8, 13, 26]:
    Lh = painel(h)
    tr = Lh[(Lh.date >= '2015-01-01') & (Lh.date + pd.Timedelta(days=6) < CORTE)]
    te = dentro(Lh, S1_INI, S1_FIM).copy()
    for n, p in ajustar_prever(tr, te, ('xgboost',)).items():
        linhas.append(dict(protocolo=f'B_temporada1_h{h}', avaliacao='S1', modelo=n, **metricas(te.assign(p=p))))

# ---------- C) Regra do IMDC: só dados até SE25/2022; prever SE41→SE40 (h = 16…68) ----------
prev = []
for h in sorted({int((d - CORTE).days // 7) + 1 for d in pd.date_range(S1_INI, S1_FIM - pd.Timedelta(days=6), freq='7D')}):
    Lh = painel(h)
    alvo = CORTE + pd.Timedelta(weeks=h - 1)
    tr = Lh[(Lh.date >= '2015-01-01') & (Lh.date + pd.Timedelta(days=6) < CORTE)]
    te = Lh[Lh.date == alvo].copy()
    m = modelos()['xgboost'].fit(tr[X_COLS], tr['y'])
    te['p'] = np.maximum(0, m.predict(te[X_COLS]))
    te['persist'] = Y.loc[CORTE - pd.Timedelta(weeks=1), te.uf].values  # último valor observado antes do corte
    prev.append(te)
C = pd.concat(prev)
linhas.append(dict(protocolo='C_regra_IMDC', avaliacao='S1', modelo='xgboost_felipe_direto', **metricas(C)))
linhas.append(dict(protocolo='C_regra_IMDC', avaliacao='S1', modelo='persistencia_SE25', **metricas(C, 'persist')))

# ---------- D) Equipes do 3º IMDC (mesma regra), contra os seus dados ----------
verdade = D[['uf_code', 'uf', 'date', 'casos']].rename(columns={'casos': 'y'})
for arq in sorted(glob.glob(f'{BASE}/V4/referencias_imdc3/*.csv.gz')):
    e = pd.read_csv(arq, usecols=['date', 'pred', 'adm_1', 'validation'])
    e['date'] = pd.to_datetime(e['date'])
    e = e.merge(verdade, left_on=['adm_1', 'date'], right_on=['uf_code', 'date'])
    e = e[(e.validation == 1) & (e.uf != 'ES')].dropna(subset=['y', 'pred'])
    if e.uf.nunique() < 20:
        continue
    nome = arq.split('3rd_imdc_')[-1].split('dengue_state_')[-1].replace('.csv.gz', '')
    linhas.append(dict(protocolo='D_equipes_IMDC', avaliacao='S1', modelo=nome, **metricas(e, 'pred')))

R = pd.DataFrame(linhas)
pd.set_option('display.width', 250, 'display.max_rows', 200)
R.to_csv(r'C:/Users/Victor/AppData/Local/Temp/claude/F--AmbientePython-Projetos/26b2b86b-fbda-4ad6-a9ea-40efc3ab2fa6/scratchpad/replica_felipe_resultados.csv', index=False)
print(R.round(1).to_string(index=False))
