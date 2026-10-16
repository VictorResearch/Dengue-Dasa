"""Aplica o reconciliar_mint_dinamico do notebook às previsões locais e compara com o CSV do Colab."""
import runpy, json, os, numpy as np, pandas as pd
from scipy.stats import pearsonr
ns = runpy.run_path('diagnostico_agregacao.py')           # refaz inferência (MU, QM, Yuf, ...)
G = ns['G']; cells = ns['cells']
exec(compile(cells[9], '<celula 9>', 'exec'), G)          # reconciliar_mint_dinamico
QM, Y, geos, ufs_no, datas, ufs = ns['QM'], ns['Y'], ns['geos'], ns['ufs_no'], ns['datas'], ns['ufs']
T, N = Y.shape
nome = {'l95': 'lower_95', 'l90': 'lower_90', 'l50': 'lower_50', 'med': 'pred', 'u50': 'upper_50', 'u90': 'upper_90', 'u95': 'upper_95'}
df_mun = pd.DataFrame({'geocode': np.tile(geos, T), 'date': np.repeat(datas, N), 'uf_code': np.tile(ufs_no, T),
                       'casos_reais': Y.ravel()})
for k, c in nome.items():
    df_mun[c] = QM[k].ravel()
df_mun['lower_80'] = df_mun['lower_90']; df_mun['upper_80'] = df_mun['upper_90']
cols = ['pred', 'lower_50', 'upper_50', 'lower_80', 'upper_80', 'lower_90', 'upper_90', 'lower_95', 'upper_95', 'casos_reais']
df_est = df_mun.groupby(['uf_code', 'date'])[cols].sum().reset_index()
m_rec, e_rec = G['reconciliar_mint_dinamico'](df_municipal=df_mun, df_estadual=df_est, semana_inicio_alta=47,
                                             semana_fim_alta=25, col_uf_code='uf_code')
pr = e_rec.pivot(index='date', columns='uf_code', values='pred').reindex(columns=ufs).to_numpy()
csv = pd.read_csv('previsoes_estaduais_s1.csv')
cp = csv.pivot(index='date', columns='uf_code', values='pred').reindex(columns=ufs).to_numpy()
Yuf = ns['Yuf']; antes = df_est.pivot(index='date', columns='uf_code', values='pred').reindex(columns=ufs).to_numpy()
print('\nsoma estadual ANTES do MinT / real :', round(antes.sum() / Yuf.sum(), 3), '| MAE', round(np.abs(antes - Yuf).mean(), 1))
print('soma estadual DEPOIS do MinT / real:', round(pr.sum() / Yuf.sum(), 3), '| MAE', round(np.abs(pr - Yuf).mean(), 1))
print('CSV do Colab / real                 :', round(cp.sum() / Yuf.sum(), 3), '| MAE', round(np.abs(cp - Yuf).mean(), 1))
print('MinT local x CSV: r', round(pearsonr(pr.ravel(), cp.ravel())[0], 4), '| MAE entre eles', round(np.abs(pr - cp).mean(), 1))
# por município: quanto o MinT encolhe, em função do tamanho
pm = m_rec.pivot(index='date', columns='geocode', values='pred').reindex(columns=geos).to_numpy()
fator = pm.sum(0) / np.maximum(QM['med'].sum(0), 1e-9)
tam = Y.sum(0)
for a, b in [(0, 10), (10, 100), (100, 1000), (1000, 10000), (10000, 1e9)]:
    m = (tam >= a) & (tam < b) & (QM['med'].sum(0) > 0)
    print(f'municípios com {a:>6}–{b:<10.0f} casos no teste: {m.sum():5d} | MinT multiplica a previsão por {np.median(fator[m]):.3f} (mediana)')
