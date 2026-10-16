"""Diagnóstico da v3 (S1): reproduz a previsão com os pesos da v3, separa o que é do modelo e o que é do
pós-processamento, e mede viés por UF/região, início/fim da série e atraso dos picos."""
import json, os
import numpy as np, pandas as pd, torch
from scipy.stats import nbinom, pearsonr
import pyarrow.parquet as pq

AQUI = os.path.dirname(os.path.abspath(__file__)); S1 = os.path.dirname(AQUI)
nb = json.load(open(os.path.join(S1, 'Modelo2_v3.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells']]
G = {'__name__': 'nb'}
for i in (3, 4, 7, 9, 11, 12, 14, 15):
    exec(compile(cells[i], f'<celula {i}>', 'exec'), G)
PreditorNBAGCRN, CS, CP, VARS = G['PreditorNBAGCRN'], G['COLS_SECA'], G['COLS_PICO'], G['VARS']
torch.set_grad_enabled(False)
SEMVAL, AQ, ENC = 52, 15, 2000
GRADE_K = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
Q = {'lower_95': .025, 'lower_90': .05, 'lower_50': .25, 'pred': .5, 'upper_50': .75, 'upper_90': .95, 'upper_95': .975}

arq_tr = os.path.join(S1, 'train1_FINAL_NN_v2.parquet')
eps = np.sort(pq.read_table(arq_tr, columns=['epiweek'])['epiweek'].to_pandas().unique())
dv = pd.read_parquet(arq_tr, filters=[('epiweek', '>=', int(eps[-(SEMVAL + AQ)]))]).sort_values(['date', 'node_id'])
dc = pd.read_parquet(os.path.join(S1, 'ctx1_FINAL_NN_v2.parquet')).sort_values(['date', 'node_id'])
dt = pd.read_parquet(os.path.join(S1, 'test1_FINAL_NN_v2.parquet')).sort_values(['date', 'node_id'])
N = dt['node_id'].nunique(); uf_no = dt['uf_code'].to_numpy()[:N]; ufs = np.unique(uf_no)
A = (uf_no[None, :] == ufs[:, None]).astype(float)


def carregar(nome, cols):
    m = PreditorNBAGCRN(N, len(cols), 64, cols.index('velocidade'), cols.index('aceleracao'))
    sd = torch.load(os.path.join(AQUI, f'modelo_agcrn_{nome}_s1.pt'), map_location='cpu', weights_only=False)
    m.load_state_dict({k.replace('_orig_mod.', ''): v.float() for k, v in sd.items()})
    return m.eval()
ms, mp = carregar('SECA', CS), carregar('PICO', CP)


def prever(aq, alvo):
    df = pd.concat([aq, alvo]); n_aq, n = aq['date'].nunique(), alvo['date'].nunique()
    t_ = lambda c: torch.tensor(df[c].to_numpy(np.float32).reshape(-1, N, len(c)))
    Xs, Xp = t_(CS), t_(CP); lp = torch.tensor(df['log_pop'].to_numpy(np.float32).reshape(-1, N))
    se = alvo.groupby('date')['epiweek'].first().to_numpy() % 100; pico = (se >= 47) | (se <= 25)
    mu_s, mu_p, r_s, r_p = (np.zeros((n, N)) for _ in range(4)); hs = hp = None
    for t in range(n_aq + n - 1):
        a, b, hs = ms(Xs[t], lp[t], hs); c, d, hp = mp(Xp[t], lp[t], hp); k = t + 1 - n_aq
        if k < 0: continue
        mu_s[k], r_s[k] = a.exp().squeeze(-1).numpy(), b.exp().squeeze(-1).numpy()
        mu_p[k], r_p[k] = c.exp().squeeze(-1).numpy(), d.exp().squeeze(-1).numpy()
    return mu_s, r_s, mu_p, r_p, pico, se


sv = dv['epiweek'].unique()
av = dv[dv['epiweek'].isin(sv[AQ:])]
mus_v, rs_v, mup_v, rp_v, pv, sev = prever(dv[dv['epiweek'].isin(sv[:AQ])], av)
yv = av['casos.x'].to_numpy(float).reshape(-1, N)
mus_t, rs_t, mup_t, rp_t, pt, set_ = prever(dc, dt)
Y = dt['casos.x'].to_numpy(float).reshape(-1, N); Yuf = Y @ A.T
datas = sorted(dt['date'].unique())
sel = lambda pico, a, b: np.where(pico[:, None], b, a)
mu_v, r_v = sel(pv, mus_v, mup_v), sel(pv, rs_v, rp_v)
mu_t, r_t = sel(pt, mus_t, mup_t), sel(pt, rs_t, rp_t)


def nb_est(mu, r, k):
    m = mu @ A.T; v = (mu + mu ** 2 / r) @ A.T
    rr = np.where(v > m, m ** 2 / np.maximum(k * (v - m), 1e-9), 1e6)
    return m, {c: nbinom.ppf(q, rr, rr / (rr + m)) for c, q in Q.items()}


def calibrar(por_uf=True, k_fixo=None):
    f, K = {}, {}
    for nome, m in (('pico', pv), ('seca', ~pv)):
        g = yv[m].sum() / mu_v[m].sum()
        num, den = (yv[m] @ A.T).sum(0), (mu_v[m] @ A.T).sum(0)
        w = den / (den + ENC) if por_uf else np.zeros_like(den)
        f[nome] = (np.where(den > 0, w * num / np.maximum(den, 1e-9) + (1 - w) * g, g)) @ A
        if k_fixo is not None:
            K[nome] = k_fixo; continue
        errs = []
        yvu = yv[m] @ A.T
        for k in GRADE_K:
            _, q = nb_est(mu_v[m] * f[nome], r_v[m], k)
            c50 = np.mean((yvu >= q['lower_50']) & (yvu <= q['upper_50'])) * 100
            c90 = np.mean((yvu >= q['lower_90']) & (yvu <= q['upper_90'])) * 100
            errs.append(abs(c50 - 50) + abs(c90 - 90))
        K[nome] = GRADE_K[int(np.argmin(errs))]
    return f, K


def aplicar(f, K, ponto='mediana_k'):
    mu = mu_t * np.where(pt[:, None], f['pico'][None, :], f['seca'][None, :])
    out = {c: np.zeros_like(Yuf) for c in Q}; media = np.zeros_like(Yuf); med1 = np.zeros_like(Yuf)
    for nome, m in (('pico', pt), ('seca', ~pt)):
        mm, q = nb_est(mu[m], r_t[m], K[nome]); _, q1 = nb_est(mu[m], r_t[m], 1.0)
        for c in Q: out[c][m] = q[c]
        media[m] = mm; med1[m] = q1['pred']
    if ponto == 'media': out['pred'] = media
    if ponto == 'mediana_k1': out['pred'] = med1
    return out, mu


def metr(y, o):
    p = o['pred']; den = (np.abs(y) + np.abs(p)) / 2
    d = {'MAE': np.mean(np.abs(y - p)), 'RMSE': np.sqrt(np.mean((y - p) ** 2)),
         'sMAPE': np.mean(np.where(den == 0, 0, np.abs(y - p) / np.where(den == 0, 1, den))) * 100,
         'r': pearsonr(y.ravel(), p.ravel())[0]}
    for c in ('50', '90', '95'):
        d['IC' + c] = np.mean((y >= o['lower_' + c]) & (y <= o['upper_' + c])) * 100
    return d


f, K = calibrar()
print('fatores: PICO nacional/UF min-max', np.round([f['pico'].min(), f['pico'].max()], 3),
      '| SECA', np.round([f['seca'].min(), f['seca'].max()], 3), '| K =', K)
print('validação: soma real/soma μ bruto -> PICO', round(yv[pv].sum() / mu_v[pv].sum(), 3), 'SECA', round(yv[~pv].sum() / mu_v[~pv].sum(), 3))
o_nb, mu_cal = aplicar(f, K)
csv = pd.read_csv(os.path.join(AQUI, 'previsoes_estaduais_s1.csv'))
cp = csv.pivot(index='date', columns='uf_code', values='pred').reindex(columns=ufs).to_numpy()
print('reprodução do CSV do Colab: MAE entre local e CSV =', round(np.abs(cp - o_nb['pred']).mean(), 1), '| r =', round(pearsonr(cp.ravel(), o_nb['pred'].ravel())[0], 4))

var = {}
var['v3 como rodou (mediana da NB inflada)'] = metr(Yuf, o_nb)
o, _ = aplicar(f, K, 'mediana_k1'); var['ponto = mediana sem inflação, IC inflado'] = metr(Yuf, o)
o, _ = aplicar(f, K, 'media'); var['ponto = média, IC inflado'] = metr(Yuf, o)
f0, K0 = calibrar(por_uf=False); o, _ = aplicar(f0, K0, 'mediana_k1'); var['fator nacional + mediana sem inflação'] = metr(Yuf, o)
fk, Kk = calibrar(k_fixo=1.0); o, _ = aplicar(fk, Kk); var['sem inflação (k=1)'] = metr(Yuf, o)
res = pd.DataFrame(var).T
print('\n=== ESTADO: variantes do pós-processamento (mesmos pesos v3) ===\n' + res.round(2).to_string())

# ---- totais Brasil por semana
br = pd.DataFrame({'data': datas, 'SE': set_, 'real': Yuf.sum(1), 'mu_bruto': mu_t.sum(1), 'mu_calibrado': mu_cal.sum(1),
                   'pred_v3': o_nb['pred'].sum(1)})
br['regime'] = np.where(pt, 'pico', 'seca')
print('\n=== BRASIL: totais por regime ===')
print(br.groupby('regime')[['real', 'mu_bruto', 'mu_calibrado', 'pred_v3']].sum().round(0).to_string())
print('\nsemanas de SECA (início e fim):')
print(br[br.regime == 'seca'].round(0).to_string(index=False))

# ---- por UF: viés por regime, início/fim, pico
nome_uf = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB',
           26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC', 43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'}
regiao = {1: 'Norte', 2: 'Nordeste', 3: 'Sudeste', 4: 'Sul', 5: 'Centro-Oeste'}
P = o_nb['pred']; linhas = []
for j, u in enumerate(ufs):
    y, p = Yuf[:, j], P[:, j]
    lag = max(range(-4, 5), key=lambda L: np.corrcoef(y[max(0, L):len(y) + min(0, L)], p[max(0, -L):len(p) - max(0, L)])[0, 1])
    linhas.append({'UF': nome_uf[u], 'região': regiao[u // 10], 'casos': y.sum(),
                   'viés PICO %': (p[pt].sum() / y[pt].sum() - 1) * 100, 'viés SECA %': (p[~pt].sum() / y[~pt].sum() - 1) * 100,
                   'pico real/prev': round(p.max() / y.max(), 2),
                   'semana pico real': int(set_[y.argmax()]), 'semana pico prev': int(set_[p.argmax()]),
                   'atraso ótimo (sem.)': lag,
                   'MAE início (SE41-46)': np.abs(y - p)[:6].mean(), 'MAE fim (SE26-40)': np.abs(y - p)[-15:].mean()})
uf = pd.DataFrame(linhas).sort_values('casos', ascending=False)
print('\n=== POR UF (v3 como rodou) ===\n' + uf.round(1).to_string(index=False))
print('\n=== POR REGIÃO ===')
print(uf.groupby('região').apply(lambda d: pd.Series({'casos': d.casos.sum(), 'viés PICO % (mediana UFs)': d['viés PICO %'].median(),
                                                          'viés SECA % (mediana UFs)': d['viés SECA %'].median(),
                                                          'pico prev/real (mediana)': d['pico real/prev'].median()})).round(1).to_string())
res.to_csv(os.path.join(AQUI, 'variantes_estado.csv')); uf.to_csv(os.path.join(AQUI, 'por_uf.csv'), index=False); br.to_csv(os.path.join(AQUI, 'brasil_semanal.csv'), index=False)
