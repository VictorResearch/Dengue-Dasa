"""O modelo do Felipe nas MINHAS regras: dados até a SE25, prever SE41→SE40 (h = 16…68), temporadas 2022–2025,
9 quantis, WIS relativo ao BB (média geométrica por UF, igual ao `avaliar` do v4_lib).
Mesmas features e hiperparâmetros do piloto dele; um modelo direto por horizonte h.
  Felipe_XGB_ponto     o XGBoost dele (erro quadrático); os 9 quantis = a previsão pontual
  Felipe_XGB_quantis   mesmas features, mas XGBoost quantílico (9 quantis): o mínimo para entrar no WIS
  persistencia_SE25    repete a última semana observada
CPU: ~10 min (n_jobs=2). Saída: felipe_no_meu_jogo_{previsoes.parquet, resumo.csv}."""
import glob, os, time, warnings
import numpy as np, pandas as pd
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(AQUI, '..')
Q = np.array([.025, .05, .10, .25, .5, .75, .90, .95, .975])
ALFAS = np.array([.50, .20, .10, .05])
QCOLS = ['lower_95', 'lower_90', 'lower_80', 'lower_50', 'pred', 'upper_50', 'upper_80', 'upper_90', 'upper_95']

D = pd.read_parquet(f'{BASE}/UF/dados/tabela_uf_semanal.parquet', columns=['uf_code', 'uf', 'date', 'casos', 'pop'])
D['date'] = pd.to_datetime(D['date'])
Y = D.pivot(index='date', columns='uf_code', values='casos').sort_index()
P = D.pivot(index='date', columns='uf_code', values='pop').sort_index()
UFS = list(Y.columns)
LAGS = [1, 2, 4, 8, 52]
X_COLS = [f'lag_{k}' for k in LAGS] + ['media_4', 'media_8', 'sen', 'cos', 'populacao'] + [f'uf_{u}' for u in UFS]


def inicio_ano_epi(ano):
    j4 = pd.Timestamp(ano, 1, 4)
    return j4 - pd.Timedelta(days=(j4.weekday() + 1) % 7)


def datas_temporada(s):
    return pd.date_range(inicio_ano_epi(s) + pd.Timedelta(weeks=40), inicio_ano_epi(s + 1) + pd.Timedelta(weeks=39), freq='7D')


def painel(h):
    """features do piloto do Felipe para alvo t com a última observação em t−h (origem t−(h−1))."""
    s = h - 1
    f = {f'lag_{k}': Y.shift(s + k) for k in LAGS}
    f['media_4'] = sum(Y.shift(s + k) for k in range(1, 5)) / 4
    f['media_8'] = sum(Y.shift(s + k) for k in range(1, 9)) / 8
    lg = pd.concat({k: v.stack(dropna=False) for k, v in f.items()}, axis=1)
    lg['populacao'] = P.stack(dropna=False)
    lg['y'] = Y.stack(dropna=False)
    lg = lg.reset_index().rename(columns={'level_0': 'date', 'level_1': 'uf_code'})
    ang = 2 * np.pi * ((lg['date'] - pd.to_timedelta(7 * s, 'D')).dt.dayofyear - 1) / 365.2425
    lg['sen'], lg['cos'] = np.sin(ang), np.cos(ang)
    for u in UFS:
        lg[f'uf_{u}'] = (lg['uf_code'] == u).astype(float)
    return lg.dropna(subset=['lag_1'])


def wis(q, y):
    s = 0.5 * np.abs(y - q[..., 4])
    for i, a in enumerate(ALFAS):
        lo, up = q[..., 3 - i], q[..., 5 + i]
        s = s + a / 2 * ((up - lo) + 2 / a * (lo - y) * (y < lo) + 2 / a * (y - up) * (y > up))
    return s / (len(ALFAS) + 0.5)


def xgb(**kw):
    return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, tree_method='hist', n_jobs=2,
                        random_state=42, **kw)


if __name__ == '__main__':
    t0, saidas, cache = time.time(), [], {}
    for s in [2022, 2023, 2024, 2025]:
        ult = inicio_ano_epi(s) + pd.Timedelta(weeks=24)                 # semana SE25 (última observada)
        alvos = [d for d in datas_temporada(s) if d <= Y.index.max()]
        for t in alvos:
            h = int((t - ult).days // 7)
            lg = cache.setdefault(h, painel(h))
            tr = lg[(lg.date <= ult) & (lg.date >= '2015-01-01')].dropna(subset=['y'])
            te = lg[lg.date == t].copy()
            m = xgb(objective='reg:squarederror').fit(tr[X_COLS], tr['y'])
            ponto = np.maximum(0, m.predict(te[X_COLS]))
            mq = xgb(objective='reg:quantileerror', quantile_alpha=Q).fit(tr[X_COLS], tr['y'])
            qq = np.sort(np.maximum(0, mq.predict(te[X_COLS])), 1)
            base = te[['uf_code', 'date']].assign(season=s, h=h)
            per = Y.loc[ult, te.uf_code].values
            saidas += [base.assign(modelo='Felipe_XGB_ponto', **{c: ponto for c in QCOLS}),
                       base.assign(modelo='Felipe_XGB_quantis', **{c: qq[:, i] for i, c in enumerate(QCOLS)}),
                       base.assign(modelo='persistencia_SE25', **{c: per for c in QCOLS})]
        print(f'temporada {s}: {len(alvos)} semanas · {time.time() - t0:.0f}s', flush=True)
    prev = pd.concat(saidas, ignore_index=True)

    # referências do 3º IMDC (BB e algumas equipes) no mesmo formato
    for arq in glob.glob(f'{BASE}/V4/referencias_imdc3/*.csv.gz'):
        nome = os.path.basename(arq).replace('.csv.gz', '').split('3rd_imdc_')[-1].replace('dengue_state_', '')
        e = pd.read_csv(arq, usecols=['date', 'adm_1', 'validation'] + QCOLS)
        e['date'] = pd.to_datetime(e['date'])
        e = e.rename(columns={'adm_1': 'uf_code'}).assign(season=e.validation + 2021, modelo='IMDC:' + nome, h=np.nan)
        prev = pd.concat([prev, e[e.uf_code.isin(UFS)].drop(columns='validation')], ignore_index=True)
    prev = prev.merge(D[['uf_code', 'date', 'casos']].rename(columns={'casos': 'y'}), on=['uf_code', 'date'])
    prev = prev[prev.date.isin(set(np.concatenate([datas_temporada(s) for s in range(2022, 2026)])))].dropna(subset=['y'])
    prev.to_parquet(os.path.join(AQUI, 'felipe_no_meu_jogo_previsoes.parquet'), index=False)

    # avaliação: WIS por UF × temporada → razão geométrica contra o BB (como o `avaliar`), + métricas pontuais
    prev['WIS'] = wis(prev[QCOLS].values, prev['y'].values)
    prev['ae'] = (prev['pred'] - prev['y']).abs()
    den = prev['pred'].abs() + prev['y'].abs()
    prev['smape'] = np.where(den > 0, 2 * prev['ae'] / den.where(den > 0, 1), 0)
    por = prev.groupby(['modelo', 'season', 'uf_code']).agg(WIS=('WIS', 'mean'), n=('WIS', 'size')).reset_index()
    bb = por[por.modelo == 'IMDC:procc_bb_model'][['season', 'uf_code', 'WIS', 'n']].rename(columns={'WIS': 'WIS_BB', 'n': 'n_BB'})
    por = por.merge(bb, on=['season', 'uf_code'])
    por = por[por.n == por.n_BB]                                        # mesmas semanas que o BB
    por['lr'] = np.log(por.WIS / por.WIS_BB)
    linhas = []
    for m, g in por.groupby('modelo'):
        geo = g.groupby('uf_code').lr.mean()
        p = prev[prev.modelo == m]
        linhas.append(dict(modelo=m, temporadas=g.season.nunique(), R_vs_BB=np.exp(geo.mean()),
                           UFs_melhor_que_BB=int((geo < 0).sum()),
                           **{f'R_S{s - 2021}': np.exp(g[g.season == s].lr.mean()) for s in range(2022, 2026)},
                           MAE=p.ae.mean(), WAPE=100 * p.ae.sum() / p.y.sum(), sMAPE=100 * p.smape.mean()))
    R = pd.DataFrame(linhas).sort_values('R_vs_BB')
    R.to_csv(os.path.join(AQUI, 'felipe_no_meu_jogo_resumo.csv'), index=False)
    pd.set_option('display.width', 250)
    print(R.round(3).to_string(index=False))
    print(f'{time.time() - t0:.0f}s')
