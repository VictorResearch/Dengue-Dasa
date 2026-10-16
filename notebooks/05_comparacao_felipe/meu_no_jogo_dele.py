"""As MINHAS famílias de modelo nas regras do Felipe: previsão h semanas à frente (h = 1 é a dele), modelo fixo por
rodada (treino até 31/12 do ano anterior), histórico observado atualizado a cada semana; validação 2019–2022,
teste 2023–2024 (ano civil) e, à parte, a temporada 1 do IMDC (treino até a SE25/2022).
Estimativa local barata (CPU, ~3 min). As redes de grafo nesse jogo ficam no Colab (CompFelipe_GPU).
  persistencia          repete a última semana (o melhor modelo dele no teste)
  Felipe_XGB            o XGBoost do piloto dele (contagens, erro quadrático)
  GBM_log_cresc         estilo dos meus modelos: LightGBM (mediana, L1) no CRESCIMENTO log1p(inc_t) − log1p(inc_origem),
                        com âncora sazonal (mesma semana em anos anteriores), vizinhos no grafo de fronteiras, UF, pop.
                        Extrapola recordes (as árvores do Felipe, em contagens, não).
  ARIMA_Fourier         minha família ARIMA: SARIMAX(2,0,1) + Fourier K=3 em log1p(inc), parâmetros fixos na rodada,
                        previsão um passo à frente com o filtro de Kalman atualizado (só h=1)
  ENS_mediana           mediana (persistência, GBM_log_cresc, ARIMA_Fourier); ENS_GBM_ARIMA = média dos dois
Saída: meu_no_jogo_dele_{previsoes.parquet, resumo.csv}."""
import os, time, warnings
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore')
import felipe_no_meu_jogo as F                                       # mesmas tabelas e o painel do Felipe

AQUI = os.path.dirname(os.path.abspath(__file__))
Y, P, UFS = F.Y, F.P, F.UFS
L = np.log1p(Y / P * 1e5)                                            # log1p(incidência por 100 mil)
viz = np.load(os.path.join(AQUI, '..', 'UF', 'dados', 'vizinhanca_uf.npz'))
W = pd.DataFrame(viz['W1'], index=viz['ufs'], columns=viz['ufs']).loc[UFS, UFS].values
W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)


def painel_meu(h):
    a = L.shift(h)                                                   # última semana observada (origem − 1)
    f = {'nivel': a, 'd1': a - L.shift(h + 1), 'd4': a - L.shift(h + 4), 'd8': a - L.shift(h + 8),
         'viz_nivel': pd.DataFrame(a.values @ W.T, index=L.index, columns=UFS),
         'viz_d1': pd.DataFrame((a - L.shift(h + 1)).values @ W.T, index=L.index, columns=UFS)}
    anos = [L.shift(52 * k) for k in (1, 2, 3)]                     # mesma semana em anos anteriores (h < 52)
    f['ancora'] = sum(anos) / 3 - a
    f['ano_passado'] = anos[0] - a
    f['ano_passado_d'] = L.shift(52) - L.shift(52 + 1)              # como a curva andava há um ano nesta semana
    lg = pd.concat({k: v.stack(dropna=False) for k, v in f.items()}, axis=1)
    lg['z'] = (L - a).stack(dropna=False)
    lg['y'] = Y.stack(dropna=False)
    lg['lpop'] = np.log(P).stack(dropna=False)
    lg = lg.reset_index().rename(columns={'level_0': 'date', 'level_1': 'uf_code'})
    ang = 2 * np.pi * (lg['date'].dt.dayofyear - 1) / 365.2425
    lg['sen'], lg['cos'] = np.sin(ang), np.cos(ang)
    lg['uf'] = lg['uf_code'].astype('category')
    return lg.dropna(subset=['nivel', 'ancora'])


COLS_MEU = ['nivel', 'd1', 'd4', 'd8', 'viz_nivel', 'viz_d1', 'ancora', 'ano_passado', 'ano_passado_d', 'lpop',
            'sen', 'cos', 'uf']


def arima_um_passo(fim_treino, ini_av, fim_av, anos=8):
    """SARIMAX(2,0,1)+Fourier K=3 por UF; ajusta até fim_treino e filtra (sem reajustar) até fim_av."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    idx = L.index[(L.index > pd.Timestamp(fim_treino) - pd.Timedelta(weeks=52 * anos)) & (L.index <= fim_av)]
    k = np.arange(len(idx))
    ex = np.column_stack([fn(2 * np.pi * j * k / 52.18) for j in (1, 2, 3) for fn in (np.sin, np.cos)])
    tr = idx <= pd.Timestamp(fim_treino)
    out = {}
    for u in UFS:
        y = L.loc[idx, u].values
        m = SARIMAX(y[tr], exog=ex[tr], order=(2, 0, 1), trend='c').fit(disp=False, maxiter=200)
        r = m.apply(y, exog=ex)                                       # mesmos parâmetros, dados até fim_av
        out[u] = pd.Series(r.predict(), index=idx)                    # previsão um passo à frente
    lp = pd.DataFrame(out).loc[ini_av:fim_av]
    return (np.expm1(lp).clip(lower=0) * P.loc[lp.index] / 1e5).stack().rename('ARIMA_Fourier')


def rodada(h, fim_treino, ini_av, fim_av, com_arima):
    pf, pm = F.painel(h), painel_meu(h)
    dentro = lambda d: d[(d.date >= ini_av) & (d.date + pd.Timedelta(days=6) <= pd.Timestamp(fim_av))]   # noqa: E731
    trf = pf[(pf.date >= '2015-01-01') & (pf.date + pd.Timedelta(days=6) <= pd.Timestamp(fim_treino))].dropna(subset=['y'])
    tef = dentro(pf).dropna(subset=['y']).set_index(['date', 'uf_code'])
    xf = F.xgb(objective='reg:squarederror').fit(trf[F.X_COLS], trf['y'])
    trm = pm[(pm.date >= '2012-01-01') & (pm.date + pd.Timedelta(days=6) <= pd.Timestamp(fim_treino))].dropna(subset=['z'])
    tem = dentro(pm).set_index(['date', 'uf_code'])
    gm = LGBMRegressor(objective='l1', n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=2,
                       verbosity=-1).fit(trm[COLS_MEU], trm['z'])
    r = pd.DataFrame({'y': tef['y'], 'persistencia': tef['lag_1'],
                      'Felipe_XGB': np.maximum(0, xf.predict(tef[F.X_COLS]))})
    gl = np.expm1(tem['nivel'] + gm.predict(tem[COLS_MEU])).clip(lower=0) * np.exp(tem['lpop']) / 1e5
    r['GBM_log_cresc'] = gl.reindex(r.index)
    if com_arima:
        r['ARIMA_Fourier'] = arima_um_passo(fim_treino, ini_av, fim_av).reindex(r.index)
        r['ENS_mediana'] = r[['persistencia', 'GBM_log_cresc', 'ARIMA_Fourier']].median(1)
        r['ENS_GBM_ARIMA'] = r[['GBM_log_cresc', 'ARIMA_Fourier']].mean(1)
    return r.dropna()


def metricas(r, col):
    y, p = r['y'].values, r[col].values
    den = np.abs(y) + np.abs(p)
    s = np.where(den > 0, 2 * np.abs(p - y) / np.where(den > 0, den, 1), 0)
    return dict(MAE=np.abs(p - y).mean(), WAPE=100 * np.abs(p - y).sum() / y.sum(), sMAPE=100 * s.mean(),
                sMAPE_0a100=50 * s.mean(), sMAPE_mediana=100 * np.median(s))


if __name__ == '__main__':
    t0, linhas, prevs = time.time(), [], []
    planos = [(f'valid_{a}', f'{a - 1}-12-31', f'{a}-01-01', f'{a}-12-31') for a in (2019, 2020, 2021, 2022)]
    planos += [('teste_2023-24', '2022-12-31', '2023-01-01', '2024-12-31'),
               ('S1_IMDC', '2022-06-25', '2022-10-09', '2023-10-07')]
    for h in (1, 4):
        for nome, ft, ia, fa in planos:
            if h == 4 and nome.startswith('valid'):
                continue
            r = rodada(h, ft, ia, fa, com_arima=(h == 1))
            prevs.append(r.reset_index().assign(avaliacao=nome, h=h))
            for col in [c for c in r.columns if c != 'y']:
                linhas.append(dict(h=h, avaliacao=nome, modelo=col, n=len(r), **metricas(r, col)))
            print(f'h={h} {nome}: {time.time() - t0:.0f}s', flush=True)
    pd.concat(prevs).to_parquet(os.path.join(AQUI, 'meu_no_jogo_dele_previsoes.parquet'), index=False)
    R = pd.DataFrame(linhas)
    R.to_csv(os.path.join(AQUI, 'meu_no_jogo_dele_resumo.csv'), index=False)
    pd.set_option('display.width', 250, 'display.max_rows', 200)
    print(R.round(1).to_string(index=False))
