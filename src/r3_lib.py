"""RODADA 3 · temporada com detector de tamanho + mistura normal/pico + revisões mensais.

Ideia (doc 15 do mapa): o modelo de temporada atual (ENS_grafos3_λ) acerta a época do pico, mas desenha uma temporada
típica e não vê os anos epidêmicos. Em junho já existem sinais do tamanho (casos fora de época relativos ao normal da UF,
espalhamento, vizinhos, ENSO). Então:
  ① DETECTOR: uma linha por UF × temporada × corte → quantis do log do tamanho da temporada RELATIVO ao normal da UF;
  ② NORMAL: ENS_grafos3_λ (rodada 2), sem mudança;
  ③ PICO: o mesmo normal reescalado (MIX_escala) ou a forma média dos anos grandes (MIX_forma);
  mistura por trajetórias: em cada trajetória, se o tamanho sorteado do detector passa do limiar, entra o cenário de pico.
  Revisões: o detector roda de novo em jul/ago/set/out com os dados novos; o normal continua o de junho.
Sem vazamento: o detector de cada temporada treina só com temporadas já terminadas no corte; a escolha das variáveis e dos
parâmetros da mistura usa só as temporadas de afinação (2018–2021). Tudo grava em arquivo e retoma de onde parou.
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import v4_lib as V
import comp_lib as L
import uf_lib as U
import semanal_lib as S
import r2_lib as R2

Q, QCOLS = V.Q, V.QCOLS
ZQ = norm.ppf(Q)
TUNING, TESTE = U.TUNING, U.TESTE
CORTES = {'jun': 25, 'jul': 30, 'ago': 34, 'set': 38, 'out': 40}     # SE do último dado usado; jun = regra oficial
LIM_GRANDE = np.log(2.0)                                                # "temporada grande" = 2× o normal da UF
salvar_tabela, json_grava, json_le, carimbo = R2.salvar_tabela, R2.json_grava, R2.json_le, R2.carimbo


def _grava_pq(df, arq):
    df.to_parquet(arq + '.tmp', index=False)
    os.replace(arq + '.tmp', arq)


# =====================================================================================================================
# ① DETECTOR · alvo e variáveis
# =====================================================================================================================
def indice_corte(D, s, nome):
    i = np.where(D.epiweek == s * 100 + CORTES[nome])[0]
    return int(i[0]) if len(i) else None


def _completa_ate(D, s):
    """índice da última semana da temporada s (SE40 de s+1) ou None se a temporada não terminou nos dados."""
    i = D.idx_temporada(s)
    if len(i) < 52 or i[-1] > D.ultimo_obs or D.epiweek[i[0]] % 100 != 41:
        return None
    return int(i[-1])


def tamanhos(D):
    """log1p(incidência da temporada inteira, por 100 mil) de cada temporada completa: DataFrame season × UF."""
    lin = {}
    for s in np.unique(D.season):
        fim = _completa_ate(D, s)
        if fim is None:
            continue
        i = D.idx_temporada(s)
        lin[int(s)] = np.log1p(np.nansum(D.Y[i], 0) / D.POP[i[len(i) // 2]] * 1e5)
    return pd.DataFrame(lin, index=D.unidades).T.sort_index()


def _rel_mesma_epoca(D, x, c, anos=10):
    """x (T, N) → x[c] menos a média de x na mesma SE dos anos anteriores (só dados ≤ c)."""
    se = D.epiweek[c] % 100
    ano = D.epiweek[c] // 100
    idx = [np.where(D.epiweek == (ano - k) * 100 + se)[0] for k in range(1, anos + 1)]
    idx = [i[0] for i in idx if len(i)]
    if not idx:
        return np.full(D.N, np.nan)
    return x[c] - np.nanmean(x[idx], 0)


def variaveis_corte(ctx, s, nome, TAM):
    """variáveis conhecidas no corte `nome` da temporada s (uma linha por UF)."""
    D = ctx.Du
    c = indice_corte(D, s, nome)
    if c is None or c > D.ultimo_obs:
        return None
    inc = np.nan_to_num(D.INC)
    cs = np.vstack([np.zeros(D.N), np.cumsum(inc, 0)])
    soma = lambda k: np.log1p(cs[np.arange(1, D.T + 1)] - cs[np.maximum(np.arange(1, D.T + 1) - k, 0)])   # noqa: E731
    f = {}
    for k in (4, 8, 13):
        f[f'rel_inc_{k}'] = _rel_mesma_epoca(D, soma(k), c)
        f[f'inc_{k}'] = soma(k)[c]
    linc = np.nan_to_num(D.LINC)
    k8 = np.arange(8) - 3.5
    tend = np.zeros((D.T, D.N))
    for t in range(7, D.T):
        tend[t] = (k8 @ linc[t - 7:t + 1]) / (k8 @ k8)
    f['tend8'] = tend[c]
    f['rel_tend8'] = _rel_mesma_epoca(D, tend, c)
    M = ctx.tab.M if getattr(ctx, 'tab', None) is not None else {}
    if 'pct_mun_com_casos_8' in M:
        f['rel_espalhamento'] = _rel_mesma_epoca(D, np.nan_to_num(M['pct_mun_com_casos_8']), c)
    A = D.ADJ / np.maximum(D.ADJ.sum(1, keepdims=True), 1)
    f['viz_rel_inc_8'] = A @ np.nan_to_num(f['rel_inc_8'])
    macro = D.unidades // 10
    f['macro_rel_inc_8'] = np.array([np.nanmean(f['rel_inc_8'][macro == m]) for m in macro])
    for o in ('enso', 'iod', 'pdo'):
        x = np.nan_to_num(D.OCEANO[o])
        f[o] = np.repeat(x[c], D.N)
        f[o + '_delta12'] = np.repeat(x[c] - x[max(c - 12, 0)], D.N)
    # imunidade: tamanho das temporadas anteriores relativo ao normal da UF (só as já terminadas no corte)
    ant = TAM.loc[[i for i in TAM.index if (_completa_ate(D, i) or 10 ** 9) <= c and i < s]]
    if len(ant) >= 3:
        base = ant.mean()
        f['anom_ultima'] = (ant.iloc[-1] - base).values
        f['anom_penultima'] = (ant.iloc[-2] - base).values
        grande = (ant - base) > LIM_GRANDE
        f['temporadas_desde_grande'] = np.array([len(ant) - 1 - np.where(grande[u].values)[0][-1] if grande[u].any()
                                                 else len(ant) for u in D.unidades], float)
    # temporada anterior parcial (até o corte), relativa ao normal
    i_ant = D.idx_temporada(s - 1, ate=c)
    if len(i_ant) and len(ant):
        f['anom_anterior_parcial'] = np.log1p(np.nansum(D.Y[i_ant], 0) / D.POP[c] * 1e5) - ant.mean().values
    out = pd.DataFrame(f)
    out.insert(0, 'uf_code', D.unidades)
    out.insert(1, 'season', s)
    out.insert(2, 'corte', nome)
    return out


def base_normal(TAM, D, s, c):
    """normal da UF: média do log do tamanho das temporadas terminadas até o corte (e anteriores a s)."""
    ant = TAM.loc[[i for i in TAM.index if (_completa_ate(D, i) or 10 ** 9) <= c and i < s]]
    return ant.mean()


def tabela_detector(ctx, temporadas, pasta_cache, externos=None):
    """variáveis + alvo de todas as UFs × temporadas × cortes (grava em cache; retoma)."""
    arq = os.path.join(pasta_cache, 'detector_variaveis.parquet')
    if os.path.exists(arq):
        X = pd.read_parquet(arq)
    else:
        D = ctx.Du
        TAM = tamanhos(D)
        partes = []
        for s in temporadas:
            for nome in CORTES:
                v = variaveis_corte(ctx, s, nome, TAM)
                if v is None:
                    continue
                c = indice_corte(D, s, nome)
                b = base_normal(TAM, D, s, c)
                v['base'] = b.reindex(D.unidades).values
                v['alvo'] = (TAM.loc[s].reindex(D.unidades).values - v['base']) if s in TAM.index else np.nan
                v['pop'] = D.POP[c]
                partes.append(v)
        X = pd.concat(partes, ignore_index=True)
        _grava_pq(X, arq)
    if externos is not None and len(externos):
        chaves = [k for k in ('uf_code', 'season', 'corte') if k in externos]
        X = X.merge(externos, on=chaves, how='left')
    return X


NAO_VARIAVEIS = {'uf_code', 'season', 'corte', 'base', 'alvo', 'pop'}


def selecionar_sinais(X, corte, ate_temporada=max(TUNING), k=10, rho_min=0.12, arq=None):
    """escolhe até k sinais pela correlação de Spearman com o alvo, SÓ com temporadas ≤ ate_temporada."""
    if arq and os.path.exists(arq):
        sel = json_le(arq, {})
        if corte in sel:
            return sel[corte]['sinais']
    g = X[(X.corte == corte) & (X.season <= ate_temporada)].dropna(subset=['alvo'])
    cols = [c for c in X.columns if c not in NAO_VARIAVEIS and X[c].dtype.kind in 'fi']
    r = []
    for c in cols:
        x = g[[c, 'alvo']].dropna()
        if len(x) > 40 and x[c].std() > 0:
            r.append((c, float(spearmanr(x[c], x.alvo)[0])))
    r = sorted([t for t in r if abs(t[1]) >= rho_min], key=lambda t: -abs(t[1]))[:k]
    if arq:
        sel = json_le(arq, {})
        sel[corte] = dict(sinais=[c for c, _ in r], rho={c: round(v, 3) for c, v in r}, ate_temporada=int(ate_temporada))
        json_grava(sel, arq)
    return [c for c, _ in r]


def _quantis_detector(metodo, Xtr, ytr, Xte, seed=7):
    """devolve (n_te, 9) quantis do alvo."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    if metodo == 'ridge':
        from sklearn.linear_model import RidgeCV
        m = RidgeCV(alphas=np.logspace(-2, 3, 25)).fit(Ztr, ytr)
        res = ytr - m.predict(Ztr)
        q = np.quantile(res, Q) * 1.15                                  # resíduo de treino é otimista: alarga 15%
        return m.predict(Zte)[:, None] + q[None]
    if metodo == 'qlin':
        from sklearn.linear_model import QuantileRegressor
        return np.stack([QuantileRegressor(quantile=float(a), alpha=0.02, solver='highs').fit(Ztr, ytr).predict(Zte)
                         for a in Q], -1)
    if metodo == 'qgbm':
        import lightgbm as lgb
        return np.stack([lgb.LGBMRegressor(objective='quantile', alpha=float(a), n_estimators=200, learning_rate=0.05,
                                           num_leaves=7, min_child_samples=15, subsample=0.8, subsample_freq=1,
                                           colsample_bytree=0.8, verbose=-1, random_state=seed)
                         .fit(Ztr, ytr).predict(Zte) for a in Q], -1)
    raise ValueError(metodo)


METODOS = ('ridge', 'qlin', 'qgbm')


def prever_detector(ctx, X, sinais, temporadas, pasta_cache, metodos=METODOS, min_treino=60):
    """janela expansiva: para a temporada s e o corte k, treina só com temporadas terminadas até esse corte.
    Grava um parquet por (método, corte) e retoma."""
    D = ctx.Du
    TAM_fim = {s: _completa_ate(D, s) for s in sorted(X.season.unique())}
    saidas = []
    for metodo in list(metodos) + ['media']:
        for corte in CORTES:
            arq = os.path.join(pasta_cache, f'detector_{metodo}_{corte}.parquet')
            if os.path.exists(arq):
                saidas.append(pd.read_parquet(arq))
                continue
            if metodo == 'media':
                ps = [pd.read_parquet(os.path.join(pasta_cache, f'detector_{m}_{corte}.parquet')) for m in metodos]
                out = ps[0].copy()
                out[QCOLS] = np.sort(np.mean([p[QCOLS].values for p in ps], 0), -1)
                out['metodo'] = 'media'
            else:
                g = X[X.corte == corte]
                cols = sinais[corte]
                linhas = []
                for s in temporadas:
                    c = indice_corte(D, s, corte)
                    te = g[g.season == s]
                    if c is None or te.empty:
                        continue
                    tr = g[g.season.map(lambda x: (TAM_fim.get(x) or 10 ** 9) <= c) & (g.season < s)].dropna(subset=['alvo'])
                    tr = tr.dropna(subset=cols)
                    if len(tr) < min_treino:
                        continue
                    Xte = te[cols].fillna(tr[cols].median()).values
                    q = np.sort(_quantis_detector(metodo, tr[cols].values, tr.alvo.values, Xte), -1)
                    o = te[['uf_code', 'season', 'corte', 'base', 'alvo', 'pop']].copy()
                    o[QCOLS] = q
                    o['metodo'] = metodo
                    o['n_treino'] = len(tr)
                    linhas.append(o)
                if not linhas:
                    continue
                out = pd.concat(linhas, ignore_index=True)
            out['p_grande'] = _prob_acima(out[QCOLS].values, LIM_GRANDE)
            _grava_pq(out, arq)
            saidas.append(out)
            print(f'detector {metodo:6s} {corte}: {len(out)} linhas', flush=True)
    return pd.concat(saidas, ignore_index=True)


def _prob_acima(q, lim):
    """P(alvo > lim) pela função quantil (interpolação linear contra o z normal dos níveis)."""
    p = np.empty(len(q))
    for i, qi in enumerate(q):
        z = np.interp(lim, qi, ZQ, left=ZQ[0] - 1, right=ZQ[-1] + 1)
        p[i] = 1 - norm.cdf(z)
    return p


def avaliar_detector(P):
    """pinball médio, correlação da mediana e alarmes (temporada grande), por método × corte × grupo."""
    P = P.dropna(subset=['alvo']).copy()
    P['grupo'] = np.where(P.season.isin(TESTE), 'teste', np.where(P.season.isin(TUNING), 'afinacao', 'anterior'))
    lin = []
    for (m, k, gr), g in P.groupby(['metodo', 'corte', 'grupo']):
        y, q = g.alvo.values, g[QCOLS].values
        pin = np.mean([np.mean(np.maximum(a * (y - q[:, j]), (a - 1) * (y - q[:, j]))) for j, a in enumerate(Q)])
        grande, alarme = y > LIM_GRANDE, g.p_grande.values > 0.5
        lin.append(dict(metodo=m, corte=k, grupo=gr, n=len(g), pinball=pin,
                        corr_mediana=np.corrcoef(y, q[:, 4])[0, 1] if len(g) > 2 else np.nan,
                        IC90=100 * np.mean((y >= q[:, 1]) & (y <= q[:, 7])),
                        n_grandes=int(grande.sum()), alarmes=int(alarme.sum()),
                        sensibilidade=100 * (grande & alarme).sum() / max(grande.sum(), 1),
                        precisao=100 * (grande & alarme).sum() / max(alarme.sum(), 1)))
    return pd.DataFrame(lin)


def escolher_detector(T):
    """por corte, o método com menor pinball na afinação."""
    a = T[T.grupo == 'afinacao']
    return {k: g.sort_values('pinball').metodo.iloc[0] for k, g in a.groupby('corte')}


# =====================================================================================================================
# ② NORMAL · ENS_grafos3_λ (o mesmo do R2_1/R2_7), para afinação e teste, em cache
# =====================================================================================================================
def normal_ens_grafos3(ctx, PASTA_DRIVE, PASTA_AV2, PASTA_COMP, PASTA_EXT, pasta_cache, temporadas):
    arq = os.path.join(pasta_cache, 'normal_ens_grafos3_bb_clim.parquet')
    if os.path.exists(arq):
        return pd.read_parquet(arq)
    PASTAS_UF = [p for p in sorted(glob.glob(os.path.join(PASTA_DRIVE, 'resultados_uf*'))) if os.path.isdir(p)
                 and not p.endswith('_r3')]
    FONTES = [('região→UF', PASTA_COMP)] + [(os.path.basename(p), p) for p in PASTAS_UF] + [('externo', PASTA_EXT)]
    PU = L.carregar_previsoes(FONTES, ctx.Du.ufs, 'uf', mostrar=False)
    PU = PU[~PU.modelo.str.startswith('ENS')][L.COLS_PREV]
    e0 = [os.path.join(p, 'ensembles_uf.parquet') for p in PASTAS_UF if os.path.exists(os.path.join(p, 'ensembles_uf.parquet'))]
    ENS0 = pd.read_parquet(e0[0])[L.COLS_PREV] if e0 else PU.iloc[:0]
    REF = ctx.ref[L.COLS_PREV] if ctx.ref is not None else PU.iloc[:0]
    LAM = json_le(os.path.join(PASTA_AV2, 'lambda_temporada.json'), {})
    CAL2 = [U.ajusta_largura(PU[PU.modelo == m], l).assign(modelo=f'{m}+λ') for m, l in LAM.items()
            if l != 1.0 and (PU.modelo == m).any()]
    TODOS = pd.concat([REF, PU, ENS0] + CAL2, ignore_index=True).drop_duplicates(['modelo', 'season', 'uf_code', 'date'], keep='last')
    TODOS = TODOS[TODOS.season.isin(temporadas)]
    arq_pt = sorted(glob.glob(os.path.join(PASTA_AV2, 'pesos_stack_temporada_*.json')))
    memb = list(next(iter(json_le(arq_pt[-1]).values())).keys()) if arq_pt else []
    BASE_G = ['MTGNN_UF+λ', 'A3TGCN_UF+λ', 'GCNGRU_UF+λ']
    fixos = ['MTGNN_pesopop_UF+λ', 'TFT_pool_UF', 'DoisEstagios_pool_UF'] + BASE_G
    novo = [m for m in memb if m not in fixos][:1]
    e3, _ = V.ensemble(TODOS, BASE_G + novo, 'ENS_grafos3_λ', metodo='mediana')
    out = pd.concat([e3[L.COLS_PREV], TODOS[TODOS.modelo.isin(['BB', 'climatologia'])][L.COLS_PREV]], ignore_index=True)
    out['date'] = pd.to_datetime(out.date)
    _grava_pq(out, arq)
    print('normal: ENS_grafos3_λ =', BASE_G + novo, '·', out.groupby('modelo').season.unique().to_dict())
    return out


# =====================================================================================================================
# ③ PICO e MISTURA por trajetórias
# =====================================================================================================================
def _interp(lq, z):
    """lq (W, 9) em log1p · z (M,) → (M, W) pela função quantil (linear contra ZQ, extrapolação linear nas caudas)."""
    i = np.clip(np.searchsorted(ZQ, z), 1, len(ZQ) - 1)
    t = np.clip((z - ZQ[i - 1]) / (ZQ[i] - ZQ[i - 1]), -1.5, 2.5)
    return lq[:, i - 1].T + t[:, None] * (lq[:, i].T - lq[:, i - 1].T)


def formas_grandes(D, TAM, s, c, j, W):
    """forma média (fração do total por semana da temporada) dos anos grandes terminados até o corte:
    da própria UF; se não houver, da macrorregião; se não, de todas. Suavizada (3 semanas)."""
    ant = TAM.loc[[i for i in TAM.index if (_completa_ate(D, i) or 10 ** 9) <= c and i < s]]
    if len(ant) < 3:
        return None
    an = ant - ant.mean()
    macro = D.unidades // 10
    for filtro in ([D.unidades[j]], list(D.unidades[macro == macro[j]]), list(D.unidades)):
        pares = [(i, u) for u in filtro for i in an.index if an.loc[i, u] > LIM_GRANDE]
        if pares:
            break
    if not pares:
        return None
    fs = []
    for i, u in pares:
        idx = D.idx_temporada(i)
        y = np.nan_to_num(D.Y[idx, list(D.unidades).index(u)])
        y = np.pad(y, (0, max(0, W - len(y))), mode='edge')[:W]
        fs.append(y / max(y.sum(), 1))
    f = pd.Series(np.mean(fs, 0)).rolling(3, center=True, min_periods=1).mean().values
    return f / f.sum()


def mistura_uf(qn, det_q, base, pop, forma, variante, par, M=1000, seed=0):
    """qn (W, 9) casos do NORMAL · det_q (9,) quantis do log-tamanho relativo · base: normal da UF (log1p inc).
    variante: 'escala' (par = limiar), 'forma' (par = limiar) ou 'continuo' (par = peso w). Devolve (W, 9)."""
    rng = np.random.default_rng(seed)
    lq = np.log1p(np.maximum(np.sort(qn, -1), 0))
    traj = np.expm1(_interp(lq, rng.standard_normal(M))).clip(0)          # (M, W) trajetórias coerentes no tempo
    T_n = traj.sum(1) + 1e-9
    a = _interp(np.sort(det_q)[None, :], rng.standard_normal(M))[:, 0]    # tamanho relativo sorteado (caudas extrapoladas)
    alvo = np.expm1(base + a) * pop / 1e5                                   # total da temporada em casos
    if variante == 'continuo':
        out = traj * ((alvo / T_n) ** par)[:, None]
    else:
        grande = a > par
        if variante == 'escala' or forma is None:
            pico = traj * (alvo / T_n)[:, None]
        else:
            ruido = np.exp(0.15 * rng.standard_normal(traj.shape))
            pico = forma[None] * alvo[:, None] * ruido
        out = np.where(grande[:, None], pico, traj)
    return np.quantile(out, Q, axis=0).T                                     # (W, 9)


GRADE = {'escala': [np.log(1.5), np.log(2), np.log(3)], 'forma': [np.log(1.5), np.log(2), np.log(3)],
         'continuo': [0.25, 0.5, 0.75, 1.0]}


def gerar_mistura(ctx, NORMAL, DET, variante, par, corte, temporadas, M=1000):
    """previsões da mistura (formato COLS_PREV) para as temporadas pedidas, num corte."""
    D = ctx.Du
    TAM = tamanhos(D)
    nm = NORMAL[NORMAL.modelo == 'ENS_grafos3_λ']
    lin = []
    for s in temporadas:
        g = nm[nm.season == s]
        d = DET[(DET.season == s) & (DET.corte == corte)].set_index('uf_code')
        if g.empty or d.empty:
            continue
        datas = np.sort(g.date.unique())
        qn = V.de_df(g, datas, D.unidades)                                  # (W, N, 9)
        c = indice_corte(D, s, corte)
        out = np.zeros_like(qn)
        for j, u in enumerate(D.unidades):
            if u not in d.index:
                out[:, j] = qn[:, j]
                continue
            r = d.loc[u]
            forma = formas_grandes(D, TAM, s, c, j, len(datas)) if variante == 'forma' else None
            out[:, j] = mistura_uf(qn[:, j], r[QCOLS].values.astype(float), float(r.base), float(r['pop']), forma,
                                   variante, par, M=M, seed=1000 * s + j)
        lin.append(V.para_df(V.arruma_quantis(out), datas, D.unidades, 'x', s))
    return pd.concat(lin, ignore_index=True) if lin else None


def nome_mistura(variante, corte):
    base = {'escala': 'MIX_escala_UF', 'forma': 'MIX_forma_UF', 'continuo': 'AJ_continuo_UF'}[variante]
    return base if corte == 'jun' else f'{base}@{corte}'


def escolher_parametros(ctx, NORMAL, DET, pasta_cache, M=500):
    """por variante, o parâmetro da grade com menor R × climatologia na AFINAÇÃO (corte de junho). Grava em json."""
    arq = os.path.join(pasta_cache, 'mistura_parametros.json')
    esc = json_le(arq, {})
    clim = NORMAL[(NORMAL.modelo == 'climatologia') & NORMAL.season.isin(TUNING)]
    for v, grade in GRADE.items():
        if v in esc:
            continue
        rs = []
        for p in grade:
            x = gerar_mistura(ctx, NORMAL, DET, v, p, 'jun', TUNING, M=M)
            if x is None:
                continue
            _, r = V.avaliar(pd.concat([x.assign(modelo='x'), clim])[L.COLS_PREV], ctx.ver_uf, ref=('climatologia',))
            rs.append((p, float(r.set_index('modelo').loc['x', 'R_vs_climatologia'])))
            print(f'  {v} {p:.3f}: R × clim (afinação) = {rs[-1][1]:.3f}', flush=True)
        e0 = NORMAL[(NORMAL.modelo == 'ENS_grafos3_λ') & NORMAL.season.isin(TUNING)]
        _, r0 = V.avaliar(pd.concat([e0, clim])[L.COLS_PREV], ctx.ver_uf, ref=('climatologia',))
        esc[v] = dict(par=min(rs, key=lambda t: t[1])[0], grade=rs,
                      normal=float(r0.set_index('modelo').loc['ENS_grafos3_λ', 'R_vs_climatologia']))
        json_grava(esc, arq)
    return esc


# =====================================================================================================================
# Gráficos
# =====================================================================================================================
def grafico_temporada_uf(obs, prev, u, arq, temporadas, det=None, titulo=''):
    """obs: Series date → casos (u = 0: Brasil, soma) · prev: {'rótulo': df}; o 1º recebe o leque.
    det: DataFrame (season, p_grande, mediana do multiplicador) da UF, anotado em cada temporada."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(13, 4.6))
    cores = ['#c0392b', '#2a6fdb', '#7f8c8d', '#16a085']
    estilos = ['-', '-', '--', ':']
    for i, (rot, df) in enumerate(prev.items()):
        for s in temporadas:
            p = R2._serie(df[df.season == s], u)
            if not len(p):
                continue
            if i == 0:
                ax.fill_between(p.index, p['lower_90'], p['upper_90'], color=cores[0], alpha=.12, lw=0,
                                label='IC 90%' if s == temporadas[0] else None)
                ax.fill_between(p.index, p['lower_50'], p['upper_50'], color=cores[0], alpha=.2, lw=0,
                                label='IC 50%' if s == temporadas[0] else None)
            ax.plot(p.index, p['pred'], color=cores[i % 4], ls=estilos[i % 4], lw=1.6 if i == 0 else 1.2,
                    label=rot if s == temporadas[0] else None)
    x0, x1 = pd.Timestamp(f'{temporadas[0]}-10-01'), pd.Timestamp(f'{temporadas[-1] + 1}-10-15')
    o = obs[(obs.index >= x0) & (obs.index <= x1)]
    ax.plot(o.index, o.values, color='#1a1a1a', lw=1.1, marker='o', ms=2, label='casos observados')
    ymax = ax.get_ylim()[1]
    for s in temporadas:
        ax.axvline(pd.Timestamp(f'{s}-10-08'), color='#bbbbbb', lw=.8, ls=':')
        if det is not None and s in det.index:
            r = det.loc[s]
            ax.text(pd.Timestamp(f'{s}-10-20'), ymax * .93, f"P(grande) {r.p_grande:.0%}\ntamanho ×{np.exp(r.pred):.1f}",
                    fontsize=8, color='#7a4b00', va='top')
    ax.set_title(f'{R2.SIGLAS.get(u, u)} · temporada prevista em junho (SE25) · detector + mistura{titulo}', loc='left', fontsize=11)
    ax.set_ylabel('casos por semana')
    ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'.replace(',', '.')))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#eeeeee')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(arq + '.tmp.png', dpi=110)
    os.replace(arq + '.tmp.png', arq)
    return fig
