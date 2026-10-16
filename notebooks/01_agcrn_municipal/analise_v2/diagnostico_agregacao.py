"""Diagnóstico do erro estadual do Modelo 2 v2 (S1).
Recarrega os pesos treinados no Colab (modelo_agcrn_{SECA,PICO}_s1.pt), refaz a inferência alinhada nos bancos
v2 e compara formas de agregar município -> estado. Usa as células do próprio Modelo2_v2.ipynb."""
import json, os, sys, types
import numpy as np, pandas as pd, torch
from scipy.stats import nbinom, pearsonr

AQUI = os.path.dirname(os.path.abspath(__file__))
S1 = os.path.dirname(AQUI)
nb = json.load(open(os.path.join(S1, 'Modelo2_v2.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells']]
G = {'__name__': 'nb'}
for i in (3, 4, 7, 14, 15):
    exec(compile(cells[i], f'<celula {i}>', 'exec'), G)
PreditorNBAGCRN, COLS_SECA, COLS_PICO = G['PreditorNBAGCRN'], G['COLS_SECA'], G['COLS_PICO']
torch.set_grad_enabled(False)

# ------------------------------------------------------------------ dados (ordem date, node_id)
te = pd.read_parquet(os.path.join(S1, 'test1_FINAL_NN_v2.parquet')).sort_values(['date', 'node_id'])
cx = pd.read_parquet(os.path.join(S1, 'ctx1_FINAL_NN_v2.parquet')).sort_values(['date', 'node_id'])
N = te['node_id'].nunique(); T = te['date'].nunique(); C = cx['date'].nunique()
geos = te['geocode'].to_numpy()[:N]; ufs_no = te['uf_code'].to_numpy()[:N]
ew = te.groupby('date')['epiweek'].first().to_numpy() % 100
datas = sorted(te['date'].unique())
def tens(df, cols, n): return torch.tensor(df[cols].to_numpy(np.float32).reshape(n, N, len(cols)))
Xs = torch.cat([tens(cx, COLS_SECA, C), tens(te, COLS_SECA, T)])
Xp = torch.cat([tens(cx, COLS_PICO, C), tens(te, COLS_PICO, T)])
lp = torch.cat([torch.tensor(cx['log_pop'].to_numpy(np.float32).reshape(C, N)),
                torch.tensor(te['log_pop'].to_numpy(np.float32).reshape(T, N))])
Y = te['casos.x'].to_numpy(np.float64).reshape(T, N)

def carregar(nome, cols):
    m = PreditorNBAGCRN(N, len(cols), 64, cols.index('velocidade'), cols.index('aceleracao'))
    sd = torch.load(os.path.join(AQUI, f'modelo_agcrn_{nome}_s1.pt'), map_location='cpu', weights_only=False)
    m.load_state_dict({k.replace('_orig_mod.', ''): v.float() for k, v in sd.items()})
    return m.eval()
ms, mp = carregar('SECA', COLS_SECA), carregar('PICO', COLS_PICO)

MU, R = np.zeros((T, N)), np.zeros((T, N))
hs = hp = None
for t in range(C + T - 1):
    lms, lrs, hs = ms(Xs[t], lp[t], hs)
    lmp, lrp, hp = mp(Xp[t], lp[t], hp)
    k = t + 1 - C
    if k < 0:
        continue
    pico = ew[k] >= 47 or ew[k] <= 25
    MU[k] = torch.exp(lmp if pico else lms).squeeze(-1).numpy()
    R[k] = torch.exp(lrp if pico else lrs).squeeze(-1).numpy()
MU = np.maximum(MU, 1e-8); R = np.maximum(R, 1e-3)
P = R / (R + MU)
Q = {'l95': .025, 'l90': .05, 'l50': .25, 'med': .5, 'u50': .75, 'u90': .95, 'u95': .975}
QM = {k: nbinom.ppf(q, R, P) for k, q in Q.items()}          # quantis municipais (como o notebook)

# ------------------------------------------------------------------ agregações estaduais
ufs = np.unique(ufs_no)
A = (ufs_no[None, :] == ufs[:, None]).astype(float)           # (UF, N)
Yuf = Y @ A.T                                                  # (T, UF)
agg = {}
# (A) como o notebook: soma das medianas e soma dos quantis
agg['A_soma_medianas'] = {k: v @ A.T for k, v in QM.items()}
agg['A_soma_medianas']['pred'] = agg['A_soma_medianas']['med']
# (B) média = soma das médias; incerteza = NB com mesma média e variância da soma (municípios independentes)
m_uf = MU @ A.T
v_uf = (MU + MU ** 2 / R) @ A.T
r_uf = np.where(v_uf > m_uf, m_uf ** 2 / np.maximum(v_uf - m_uf, 1e-9), 1e6)
p_uf = r_uf / (r_uf + m_uf)
agg['B_soma_medias_NB'] = {k: nbinom.ppf(q, r_uf, p_uf) for k, q in Q.items()}
agg['B_soma_medias_NB']['pred'] = m_uf
# persistência
pers_uf = np.vstack([cx.groupby(['date', 'uf_code'])['casos.x'].sum().unstack().to_numpy()[-1:], Yuf[:-1]])

def metr(y, p, lo=None, hi=None, lo9=None, hi9=None, lo95=None, hi95=None):
    y, p = y.ravel(), p.ravel()
    den = (np.abs(y) + np.abs(p)) / 2
    d = {'MAE': np.mean(np.abs(y - p)), 'RMSE': np.sqrt(np.mean((y - p) ** 2)),
         'sMAPE': np.mean(np.where(den == 0, 0, np.abs(y - p) / np.where(den == 0, 1, den))) * 100,
         'r': pearsonr(y, p)[0], 'viés médio': np.mean(p - y)}
    for nome, (a, b) in {'IC50': (lo, hi), 'IC90': (lo9, hi9), 'IC95': (lo95, hi95)}.items():
        if a is not None:
            d[nome] = np.mean((y >= a.ravel()) & (y <= b.ravel())) * 100
    return d

linhas = {}
for nome, g in agg.items():
    linhas[nome] = metr(Yuf, g['pred'], g['l50'], g['u50'], g['l90'], g['u90'], g['l95'], g['u95'])
linhas['persistencia'] = metr(Yuf, pers_uf)
res = pd.DataFrame(linhas).T
pd.set_option('display.width', 200)
print('\n=== NÍVEL ESTADUAL (27 UFs x 52 semanas) ===')
print(res.round(2).to_string())

# conferência com o CSV do Colab (pred = soma das medianas depois do MinT)
csv = pd.read_csv(os.path.join(AQUI, 'previsoes_estaduais_s1.csv')).sort_values(['uf_code', 'date'])
csv_pred = csv.pivot(index='date', columns='uf_code', values='pred').reindex(columns=ufs).to_numpy()
csv_real = csv.pivot(index='date', columns='uf_code', values='casos_reais').reindex(columns=ufs).to_numpy()
print('\nconferência: casos_reais do CSV = dados locais?', np.allclose(csv_real, Yuf))
print('conferência: pred do CSV x soma das medianas local: r =',
      round(pearsonr(csv_pred.ravel(), agg["A_soma_medianas"]["pred"].ravel())[0], 5),
      '| MAE entre os dois =', round(np.mean(np.abs(csv_pred - agg["A_soma_medianas"]["pred"])), 2),
      '| MAE do CSV contra o real =', round(np.mean(np.abs(csv_pred - Yuf)), 2))

# por UF
nome_uf = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN',
           25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC',
           43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'}
por_uf = pd.DataFrame({
    'UF': [nome_uf.get(u, u) for u in ufs], 'casos_total': Yuf.sum(0),
    'viés_%_A (soma medianas)': (agg['A_soma_medianas']['pred'].sum(0) / Yuf.sum(0) - 1) * 100,
    'viés_%_B (soma médias)': (m_uf.sum(0) / Yuf.sum(0) - 1) * 100,
    'MAE_A': np.abs(agg['A_soma_medianas']['pred'] - Yuf).mean(0), 'MAE_B': np.abs(m_uf - Yuf).mean(0),
    'MAE_persist': np.abs(pers_uf - Yuf).mean(0),
}).sort_values('casos_total', ascending=False)
print('\n=== POR UF (ordenado por casos) ===')
print(por_uf.round(1).to_string(index=False))
# municipal
print('\n=== NÍVEL MUNICIPAL ===')
print(pd.DataFrame({'pred = mediana (notebook)': metr(Y, QM['med'], QM['l50'], QM['u50'], QM['l90'], QM['u90'], QM['l95'], QM['u95']),
                    'pred = média μ': metr(Y, MU)}).T.round(3).to_string())
print('\nfração de município-semanas com mediana 0 e casos > 0:', round(np.mean((QM['med'] == 0) & (Y > 0)) * 100, 1), '%')
print('soma de todas as medianas / soma dos casos reais:', round(QM['med'].sum() / Y.sum(), 3),
      '| soma das médias μ / casos reais:', round(MU.sum() / Y.sum(), 3))
res.to_csv(os.path.join(AQUI, 'diagnostico_estadual.csv')); por_uf.to_csv(os.path.join(AQUI, 'diagnostico_por_uf.csv'), index=False)
