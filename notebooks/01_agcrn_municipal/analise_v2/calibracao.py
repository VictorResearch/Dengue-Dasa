"""Recalibração por regime usando SÓ a temporada de validação (EW202126–202225), sem tocar no teste."""
import runpy, numpy as np, pandas as pd, torch, os
from scipy.stats import nbinom, pearsonr
ns = runpy.run_path('diagnostico_agregacao.py')
ms, mp, COLS_SECA, COLS_PICO, N, metr, A = ns['ms'], ns['mp'], ns['COLS_SECA'], ns['COLS_PICO'], ns['N'], ns['metr'], ns['A']
S1 = ns['S1']
tr = pd.read_parquet(os.path.join(S1, 'train1_FINAL_NN_v2.parquet')).sort_values(['date', 'node_id'])
sem = sorted(tr['epiweek'].unique()); tr = tr[tr['epiweek'].isin(sem[-67:])]          # 15 aquecimento + 52 validação
Tv = tr['date'].nunique(); ewv = tr.groupby('date')['epiweek'].first().to_numpy() % 100
t_ = lambda cols: torch.tensor(tr[cols].to_numpy(np.float32).reshape(Tv, N, len(cols)))
Xs, Xp = t_(COLS_SECA), t_(COLS_PICO); lp = torch.tensor(tr['log_pop'].to_numpy(np.float32).reshape(Tv, N))
Yv = tr['casos.x'].to_numpy(np.float64).reshape(Tv, N)
hs = hp = None; mu_v, r_v, y_v, reg_v = [], [], [], []
for t in range(Tv - 1):
    lms, lrs, hs = ms(Xs[t], lp[t], hs); lmp, lrp, hp = mp(Xp[t], lp[t], hp)
    if t + 1 < 15: continue
    pico = ewv[t + 1] >= 47 or ewv[t + 1] <= 25
    mu_v.append(torch.exp(lmp if pico else lms).squeeze(-1).numpy()); r_v.append(torch.exp(lrp if pico else lrs).squeeze(-1).numpy())
    y_v.append(Yv[t + 1]); reg_v.append(pico)
mu_v, r_v, y_v, reg_v = map(np.array, (mu_v, r_v, y_v, reg_v))
med_v = nbinom.ppf(.5, r_v, r_v / (r_v + mu_v))
f = {}
for nome, m in (('pico', reg_v), ('seca', ~reg_v)):
    f[nome] = {'mu': y_v[m].sum() / mu_v[m].sum(), 'med': y_v[m].sum() / med_v[m].sum()}
    print(f'validação {nome}: soma real/soma μ = {f[nome]["mu"]:.3f} · soma real/soma medianas = {f[nome]["med"]:.3f}')
# aplica no teste
MU, R, ew, Y, Yuf = ns['MU'], ns['R'], ns['ew'], ns['Y'], ns['Yuf']
pico_t = (ew >= 47) | (ew <= 25)
fmu = np.where(pico_t, f['pico']['mu'], f['seca']['mu'])[:, None]
MUc = MU * fmu
Pc = R / (R + MUc)
Q = {'l95': .025, 'l90': .05, 'l50': .25, 'med': .5, 'u50': .75, 'u90': .95, 'u95': .975}
QMc = {k: nbinom.ppf(q, R, Pc) for k, q in Q.items()}
m_uf = MUc @ A.T; v_uf = (MUc + MUc ** 2 / R) @ A.T
r_uf = np.where(v_uf > m_uf, m_uf ** 2 / np.maximum(v_uf - m_uf, 1e-9), 1e6); p_uf = r_uf / (r_uf + m_uf)
Qe = {k: nbinom.ppf(q, r_uf, p_uf) for k, q in Q.items()}
res = pd.read_csv('diagnostico_estadual.csv', index_col=0)
res.loc['C_calibrado_soma_medianas'] = pd.Series(metr(Yuf, QMc['med'] @ A.T, QMc['l50'] @ A.T, QMc['u50'] @ A.T, QMc['l90'] @ A.T, QMc['u90'] @ A.T, QMc['l95'] @ A.T, QMc['u95'] @ A.T))
res.loc['D_calibrado_soma_medias_NB'] = pd.Series(metr(Yuf, m_uf, Qe['l50'], Qe['u50'], Qe['l90'], Qe['u90'], Qe['l95'], Qe['u95']))
res.loc['D2_calibrado_mediana_NB_agregada'] = pd.Series(metr(Yuf, Qe['med'], Qe['l50'], Qe['u50'], Qe['l90'], Qe['u90'], Qe['l95'], Qe['u95']))
print(res.round(2).to_string())
print('\nMunicipal calibrado (mediana):', {k: round(v, 3) for k, v in metr(Y, QMc['med'], QMc['l50'], QMc['u50'], QMc['l90'], QMc['u90'], QMc['l95'], QMc['u95']).items()})
