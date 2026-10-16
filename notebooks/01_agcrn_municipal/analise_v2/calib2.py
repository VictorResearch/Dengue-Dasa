"""Testa melhorias do agregador usando SÓ a validação: fator por UF e inflação de variância estadual."""
import runpy, numpy as np, pandas as pd
from scipy.stats import nbinom
ns = runpy.run_path('calibracao.py')
mu_v, r_v, y_v, reg_v, A, MU, R, ew, Yuf, metr = (ns[k] for k in ('mu_v', 'r_v', 'y_v', 'reg_v', 'A', 'MU', 'R', 'ew', 'Yuf', 'metr'))
pico_t = (ew >= 47) | (ew <= 25)
Q = {'l95': .025, 'l90': .05, 'l50': .25, 'med': .5, 'u50': .75, 'u90': .95, 'u95': .975}

def agrega(mu, r, k=1.0):
    m = mu @ A.T; v = (mu + mu ** 2 / r) @ A.T
    rr = np.where(v > m, m ** 2 / np.maximum(k * (v - m), 1e-9), 1e6); p = rr / (rr + m)
    return {q: nbinom.ppf(x, rr, p) for q, x in Q.items()}

def fatores(por_uf, enc=0):
    """fator = soma real / soma prevista na validação, por regime (e por UF). enc = encolhimento para o fator nacional."""
    f = {}
    for nome, m in (('pico', reg_v), ('seca', ~reg_v)):
        g = y_v[m].sum() / mu_v[m].sum()
        if por_uf:
            fu = (y_v[m] @ A.T).sum(0) / (mu_v[m] @ A.T).sum(0)
            n = (mu_v[m] @ A.T).sum(0)
            w = n / (n + enc) if enc else 1.0
            f[nome] = (w * fu + (1 - w) * g) @ A     # por município (via UF)
        else:
            f[nome] = np.full(A.shape[1], g)
    return f

def cobertura(y, q):
    return {c: np.mean((y >= q['l' + c]) & (y <= q['u' + c])) * 100 for c in ('50', '90', '95')}

linhas = {}
for rot, por_uf, enc in (('global', False, 0), ('por UF', True, 0), ('por UF encolhido', True, 2000)):
    f = fatores(por_uf, enc)
    fv = np.where(reg_v[:, None], f['pico'][None, :], f['seca'][None, :])
    ft = np.where(pico_t[:, None], f['pico'][None, :], f['seca'][None, :])
    muv_c, mut_c = mu_v * fv, MU * ft
    # inflação de variância escolhida na validação (por regime): cobertura IC50 e IC90 mais perto do nominal
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    yv_uf = y_v @ A.T
    kbest = {}
    for nome, m in (('pico', reg_v), ('seca', ~reg_v)):
        err = []
        for k in ks:
            q = agrega(muv_c[m], r_v[m], k); c = cobertura(yv_uf[m], q)
            err.append(abs(c['50'] - 50) + abs(c['90'] - 90))
        kbest[nome] = ks[int(np.argmin(err))]
    for infl in (False, True):
        qs = {}
        for nome, m in (('pico', pico_t), ('seca', ~pico_t)):
            qs[nome] = agrega(mut_c[m], R[m], kbest[nome] if infl else 1.0)
        qt = {q: np.zeros_like(Yuf) for q in Q}
        for nome, m in (('pico', pico_t), ('seca', ~pico_t)):
            for q in Q: qt[q][m] = qs[nome][q]
        linhas[f'fator {rot}' + (f' + inflação k={kbest}' if infl else '')] = metr(Yuf, qt['med'], qt['l50'], qt['u50'], qt['l90'], qt['u90'], qt['l95'], qt['u95'])
print(pd.DataFrame(linhas).T.round(2).to_string())
