"""
RODADA 2 · complementar à rodada semanal × temporada (não altera nada do que já existe; grava em pastas novas).

  A  métricas de pico e entressafra: WIS/MAE/viés por fase da semana-alvo, erro na semana e na altura do pico,
     cobertura de cada quantil
  B  transformações: B1 escala por UF (redes), B2 pré-treino nas 435 regiões + ajuste nas UFs, B3 variáveis de fase
     (Rt pela equação de renovação, semanas desde a subida, distância ao último pico, nível × limiares da UF)
  C  variáveis tsfresh de uma série única (janelas de 52 semanas na origem), seleção FRESH só na afinação
  D  D1 pinball com mais peso nos quantis altos / nas semanas de pico, D2 calibração por horizonte e fase (largura λ e
     deslocamento δ da mediana), D3 cabeça do total da temporada (pinball no log do total)
  E  E1 Optuna (GBM semanal, MTGNN de temporada), E2 mais sementes, E3 Chronos-2 ajustado, E4 pesos de ensemble por
     horizonte e fase

Retomada: tudo grava assim que termina (arquivos .tmp renomeados); treinos longos gravam checkpoints a cada
`ckpt_cada` épocas; o Optuna grava o estudo no Drive a cada tentativa; o tsfresh grava uma UF por vez.
"""
import glob
import inspect
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

import v4_lib as V
import comp_lib as L
import comp_modelos as CM
import uf_lib as U
import semanal_lib as S
from comp_lib import Modelo, Saida

Q, QCOLS, NIVEIS = V.Q, V.QCOLS, V.NIVEIS
TUNING, TESTE = U.TUNING, U.TESTE
H_SEM = S.H_SEM


# =====================================================================================================================
# utilidades: tabelas com carimbo (histórico preservado), json atômico
# =====================================================================================================================
def carimbo():
    return time.strftime('%Y%m%d-%H%M')


def salvar_tabela(df, pasta, nome):
    """grava <nome>_<data-hora>.csv (histórico) e <nome>_ultimo.csv (atalho para a versão mais recente)."""
    os.makedirs(pasta, exist_ok=True)
    arq = os.path.join(pasta, f'{nome}_{carimbo()}.csv')
    df.to_csv(arq + '.tmp', index=False)
    os.replace(arq + '.tmp', arq)
    shutil.copy(arq, os.path.join(pasta, f'{nome}_ultimo.csv'))
    return arq


def json_grava(obj, arq):
    with open(arq + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(arq + '.tmp', arq)


def json_le(arq, padrao=None):
    return json.load(open(arq)) if os.path.exists(arq) else padrao


# =====================================================================================================================
# A · fases, pico e cobertura
# =====================================================================================================================
def limiares(D, ate='2017-12-31'):
    """p25 e p75 da incidência semanal de cada UF com dados até `ate` (antes da 1ª temporada de afinação: causal)."""
    i = int(np.searchsorted(pd.to_datetime(D.datas), pd.Timestamp(ate), 'right'))
    inc = D.INC[:i]
    return pd.DataFrame({'uf_code': D.unidades, 'p25': np.nanpercentile(inc, 25, 0), 'p75': np.nanpercentile(inc, 75, 0)})


def _fase(inc, p25, p75):
    return np.where(inc >= p75, 'alta (pico)', np.where(inc < p25, 'entressafra', 'intermediaria'))


def com_fase(df, D, lim):
    """df com y (casos): acrescenta a fase da semana-alvo pela incidência OBSERVADA (só para avaliar)."""
    pop = pd.DataFrame(D.POP, index=pd.to_datetime(D.datas), columns=D.unidades).stack().rename('pop').reset_index() \
        .rename(columns={'level_0': 'date', 'level_1': 'uf_code'})
    x = df.merge(pop, on=['date', 'uf_code'], how='left').merge(lim, on='uf_code', how='left')
    x['fase'] = _fase(x.y / x['pop'] * 1e5, x.p25, x.p75)
    return x.drop(columns=['pop', 'p25', 'p75'])


def metricas_fase(X, por):
    """métricas (S.metricas) por grupo `por` + fase da semana-alvo."""
    return pd.DataFrame([dict(zip(por + ['fase'], k if isinstance(k, tuple) else (k,)), **S.metricas(g))
                         for k, g in X.groupby(por + ['fase'])])


def erros_pico(X, chave):
    """por modelo × chave (rodada ou season) × UF: semana do pico prevista (mediana) − observada, e altura relativa."""
    linhas = []
    for (m, r, u), g in X.groupby(['modelo', chave, 'uf_code']):
        g = g.sort_values('date')
        if g.y.max() <= 0 or len(g) < 8:
            continue
        iy, ip = int(np.argmax(g.y.values)), int(np.argmax(g.pred.values))
        linhas.append(dict(modelo=m, chave=r, uf_code=u, erro_semana=ip - iy,
                           erro_altura_pct=100 * (g.pred.values[ip] - g.y.values[iy]) / g.y.values[iy],
                           erro_total_pct=100 * (g.pred.sum() - g.y.sum()) / max(g.y.sum(), 1),
                           pico_no_IC90=bool(g.lower_90.values[iy] <= g.y.values[iy] <= g.upper_90.values[iy])))
    E = pd.DataFrame(linhas)
    if E.empty:
        return E
    return E.groupby('modelo').agg(n=('uf_code', 'size'), erro_semana_abs_medio=('erro_semana', lambda v: np.abs(v).mean()),
                                   erro_semana_mediano=('erro_semana', 'median'),
                                   erro_altura_mediano_pct=('erro_altura_pct', 'median'),
                                   altura_dentro_25pct=('erro_altura_pct', lambda v: 100 * np.mean(np.abs(v) <= 25)),
                                   erro_total_mediano_pct=('erro_total_pct', 'median'),
                                   pico_no_IC90_pct=('pico_no_IC90', lambda v: 100 * np.mean(v))).reset_index()


def cobertura_quantis(X, por=('modelo',)):
    """fração de y ≤ cada quantil (ideal = o nível do quantil): mostra se a cauda de cima está curta."""
    linhas = []
    for k, g in X.groupby(list(por)):
        k = k if isinstance(k, tuple) else (k,)
        linhas.append(dict(zip(por, k), **{f'≤q{q:g}': 100 * np.mean(g.y.values <= g[c].values) for q, c in zip(Q, QCOLS)}))
    return pd.DataFrame(linhas)


# =====================================================================================================================
# D2 · calibração por horizonte e fase (λ = largura, δ = deslocamento da mediana, em log1p)
# =====================================================================================================================
GRADE_LAM = (0.6, 0.8, 1.0, 1.2, 1.5, 1.8)
GRADE_DELTA = (-0.2, 0.0, 0.2, 0.4, 0.6, 0.9, 1.2)


def _aplica(q, lam, delta):
    lq = np.log1p(np.clip(q, 0, None))
    med = lq[:, [4]]
    return np.clip(np.expm1(np.sort(med + delta + lam * (lq - med), 1)), 0, None)


def fase_origem(df, D, lim):
    """fase na ORIGEM (conhecida na hora da previsão): nível da semana da origem × limiares da UF."""
    inc = pd.DataFrame(D.INC, index=pd.to_datetime(D.datas), columns=D.unidades).stack().rename('inc_o').reset_index() \
        .rename(columns={'level_0': 'origem', 'level_1': 'uf_code'})
    x = df.merge(inc, on=['origem', 'uf_code'], how='left').merge(lim, on='uf_code', how='left')
    return _fase(x.inc_o.values, x.p25.values, x.p75.values)


def bloco_horizonte(h):
    """temporada: blocos de semanas de antecedência a partir da SE25."""
    return pd.cut(h, [0, 29, 42, 55, 200], labels=['16-29', '30-42', '43-55', '56-68']).astype(str)


def calibrar(T, chave, grade_lam=GRADE_LAM, grade_delta=GRADE_DELTA, min_linhas=200):
    """T: previsões da AFINAÇÃO com y e a coluna `chave` (balde). Escolhe (λ, δ) por balde pelo menor WIS em log."""
    out = {}
    yl = None
    for b, g in T.groupby(chave):
        if len(g) < min_linhas:
            continue
        q, yl = g[QCOLS].values.astype(float), np.log1p(g.y.values.astype(float))
        melhor = None
        for l in grade_lam:
            for d in grade_delta:
                w = V.wis(np.log1p(_aplica(q, l, d)), yl).mean()
                if melhor is None or w < melhor[0]:
                    melhor = (w, l, d)
        w0 = V.wis(np.log1p(q), yl).mean()
        out[str(b)] = dict(lam=melhor[1], delta=melhor[2], wis_log_antes=float(w0), wis_log_depois=float(melhor[0]))
    return out


def aplicar_cal(X, chave, tab, sufixo='+cal'):
    X = X.copy()
    for b, p in tab.items():
        i = X[chave].astype(str) == b
        if i.any():
            X.loc[i, QCOLS] = _aplica(X.loc[i, QCOLS].values.astype(float), p['lam'], p['delta'])
    X['modelo'] = X['modelo'] + sufixo
    return X


# =====================================================================================================================
# E4 · pesos de ensemble (média dos quantis, "Vincentização") por balde, escolhidos na afinação
# =====================================================================================================================
def _alinha(X, membros, chaves):
    w = X[X.modelo.isin(membros)].pivot_table(index=chaves, columns='modelo', values=QCOLS)
    w = w.dropna()
    arr = np.stack([w[c].reindex(columns=membros).values for c in QCOLS], -1)       # (n, K, 9)
    return w.index.to_frame(index=False), arr


def pesos_stacking(T, membros, chaves, balde, passo=0.1, max_iter=60):
    """T (afinação, com y): pesos ≥ 0 que somam 1, por balde, minimizando o WIS em log da média ponderada dos quantis."""
    idx, arr = _alinha(T, membros, chaves)
    y = idx.merge(T.drop_duplicates(chaves)[chaves + ['y']], on=chaves, how='left').y.values
    out = {}
    for b in idx[balde].astype(str).unique():
        s = (idx[balde].astype(str) == b).values
        A, yl = arr[s], np.log1p(y[s])
        f = lambda w: V.wis(np.log1p(np.sort(np.tensordot(A, w, (1, 0)), 1)), yl).mean()   # noqa: E731
        w = np.full(len(membros), 1 / len(membros))
        atual = f(w)
        for _ in range(max_iter):
            melhorou = False
            for i in range(len(membros)):
                for j in range(len(membros)):
                    if i == j or w[j] < passo - 1e-9:
                        continue
                    w2 = w.copy()
                    w2[i] += passo
                    w2[j] -= passo
                    v = f(w2)
                    if v < atual - 1e-6:
                        w, atual, melhorou = w2, v, True
            if not melhorou:
                break
        out[b] = dict(zip(membros, np.round(w, 3).tolist()))
    return out


def aplicar_stacking(X, membros, chaves, balde, pesos, nome):
    idx, arr = _alinha(X, membros, chaves)
    q = np.zeros(arr.shape[0::2])
    for b, pw in pesos.items():
        s = (idx[balde].astype(str) == b).values
        q[s] = np.tensordot(arr[s], np.array([pw[m] for m in membros]), (1, 0))
    out = idx.copy()
    out[QCOLS] = np.sort(q, 1)
    out['modelo'] = nome
    return out


# =====================================================================================================================
# B3 · variáveis de fase  e  C · tsfresh  (para o GBM semanal)
# =====================================================================================================================
def vars_fase(D, lim_inc):
    """(T, N, k) causais: log Rt (renovação, intervalo serial semanal 1–4), variação do Rt, semanas desde a última
    entressafra, semanas desde o máximo das últimas 52, nível − log1p(p75), fração de semanas em alta (8)."""
    Y = np.nan_to_num(D.Y)
    sm = pd.DataFrame(Y).rolling(3, min_periods=1).mean().values
    w = np.array([0.3, 0.4, 0.2, 0.1])
    den = sum(w[k] * np.vstack([np.zeros((k + 1, D.N)), sm[:-(k + 1)]]) for k in range(4))
    lrt = np.log((sm + 1) / (den + 1))
    drt = lrt - np.vstack([np.zeros((2, D.N)), lrt[:-2]])
    inc = D.INC
    p25, p75 = lim_inc
    desde_baixa = np.zeros_like(inc)
    desde_max = np.zeros_like(inc)
    cont = np.zeros(D.N)
    for t in range(D.T):
        cont = np.where(np.nan_to_num(inc[t]) < p25, 0, cont + 1)
        desde_baixa[t] = np.minimum(cont, 104)
        ini = max(0, t - 51)
        desde_max[t] = t - (ini + np.argmax(np.nan_to_num(inc[ini:t + 1]), 0))
    rel75 = np.log1p(np.nan_to_num(inc)) - np.log1p(p75)[None]
    alta = pd.DataFrame((np.diff(np.nan_to_num(D.LINC), axis=0, prepend=0) > 0).astype(float)).rolling(8, min_periods=1).mean().values
    return np.stack([lrt, drt, desde_baixa, desde_max, rel75, alta], -1).astype(np.float32)


def tsfresh_cache(D, pasta, janela=52, parametros='efficient', n_jobs=None):
    """características tsfresh da janela de `janela` semanas de log1p(inc) terminando em cada origem, por UF.
    Grava uma UF por vez em <pasta>/uf_<codigo>.parquet (retoma de onde parou). Devolve (T, N, k) e a lista."""
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
    fc = EfficientFCParameters() if parametros == 'efficient' else MinimalFCParameters()
    os.makedirs(pasta, exist_ok=True)
    linc = np.nan_to_num(D.LINC)
    origens = np.arange(janela - 1, D.ultimo_obs + 1)
    for j, u in enumerate(D.unidades):
        arq = os.path.join(pasta, f'uf_{u}.parquet')
        if os.path.exists(arq):
            continue
        t0 = time.time()
        lg = pd.DataFrame({'id': np.repeat(origens, janela), 'tempo': np.tile(np.arange(janela), len(origens)),
                           'v': np.concatenate([linc[o - janela + 1:o + 1, j] for o in origens])})
        F = extract_features(lg, column_id='id', column_sort='tempo', default_fc_parameters=fc,
                             n_jobs=os.cpu_count() if n_jobs is None else n_jobs, disable_progressbar=True)
        F = F.replace([np.inf, -np.inf], np.nan).astype(np.float32)
        F.index.name = 'origem_idx'
        F.reset_index().to_parquet(arq + '.tmp', index=False)
        os.replace(arq + '.tmp', arq)
        print(f'tsfresh UF {u}: {F.shape[1]} variáveis · {time.time() - t0:.0f}s', flush=True)
    partes = [pd.read_parquet(os.path.join(pasta, f'uf_{u}.parquet')).set_index('origem_idx') for u in D.unidades]
    cols = sorted(set.intersection(*[set(p.columns) for p in partes]))
    M = np.full((D.T, D.N, len(cols)), np.nan, np.float32)
    for j, p in enumerate(partes):
        M[p.index.values, j] = p[cols].values
    return M, cols


def selecionar_tsfresh(D, M, cols, ate_idx, arq, k=40, fdr=0.05, n_jobs=None):
    """FRESH (teste de relevância com controle de falsas descobertas) contra o crescimento da semana seguinte,
    só com origens cujo alvo é anterior a `ate_idx` (afinação). Guarda as k mais relevantes em `arq`."""
    sel = json_le(arq)
    if sel is not None:
        return sel['colunas']
    from tsfresh.feature_selection.relevance import calculate_relevance_table
    linc = np.nan_to_num(D.LINC)
    o = np.arange(200, ate_idx - 1)
    X = pd.DataFrame(M[o].reshape(-1, M.shape[-1]), columns=cols)
    y = pd.Series((linc[o + 1] - linc[o]).ravel())
    ok = X.notna().mean() > 0.95
    X = X.loc[:, ok].fillna(X.loc[:, ok].median())
    X = X.loc[:, X.std() > 0]
    rel = calculate_relevance_table(X, y, ml_task='regression', fdr_level=fdr,
                                    n_jobs=os.cpu_count() if n_jobs is None else n_jobs)
    rel = rel[rel.relevant].sort_values('p_value')
    escolhidas = rel.feature.head(k).tolist()
    json_grava(dict(colunas=escolhidas, n_relevantes=int(len(rel)), n_testadas=int(X.shape[1])), arq)
    return escolhidas


def _quantis_peso(motor, X, y, Xp, seed, n_estimators, w=None, prm=None):
    """9 quantis com pesos por linha (D1: semanas de pico pesam mais)."""
    prm = prm or {}
    if motor == 'lgb':
        import lightgbm as lgb
        base = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.7,
                    subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, random_state=seed)
        base.update(prm)
        return np.stack([lgb.LGBMRegressor(objective='quantile', alpha=float(a), **base).fit(X, y, sample_weight=w).predict(Xp)
                         for a in Q], -1)
    import xgboost as xgb
    try:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        dev = 'cpu'
    base = dict(n_estimators=n_estimators, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    base.update(prm)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=Q, tree_method='hist', device=dev,
                         random_state=seed, **base)
    return m.fit(X, y, sample_weight=w).predict(Xp)


class GBMSemanalR2(S.ModeloSemanal):
    """GBM semanal da rodada 2: mesmas variáveis do GBM_*_todas + blocos opcionais:
    fase=True (B3), tsf=lista de colunas tsfresh (C), peso_pico>0 (D1: linhas cujo alvo está acima do p75 da UF
    pesam 1+peso_pico), mediana_L1=True (a mediana vem de um modelo só da mediana, perda L1, e os outros quantis são
    recentrados nela), prm (hiperparâmetros, ex. do Optuna)."""
    familia = 'aprendizado de máquina (rodada 2)'

    def __init__(self, nome, motor='xgb', grupos='todas', fase=True, tsf=None, tsf_M=None, tsf_cols=None, peso_pico=0.0,
                 mediana_L1=False, n_estimators=300, prm=None, lim=None):
        self.nome, self.motor, self.grupos, self.fase, self.tsf = nome, motor, grupos, fase, tsf
        self.tsf_M, self.tsf_cols, self.peso_pico, self.mediana_L1 = tsf_M, tsf_cols, peso_pico, mediana_L1
        self.n_estimators, self.prm, self.lim = n_estimators, prm, lim

    def _extra(self, ctx):
        D = ctx.Du
        partes = []
        if self.fase:
            partes.append(vars_fase(D, (self.lim.p25.values, self.lim.p75.values)))
        if self.tsf:
            idx = [self.tsf_cols.index(c) for c in self.tsf]
            partes.append(self.tsf_M[..., idx])
        return np.concatenate(partes, -1) if partes else None

    def dados(self, ctx, R):
        D, H, c = ctx.Du, H_SEM, R['c']
        P = V.PreparaA(D, c, L=104, H=H)
        cols, seas = S._cols_grupos(self.grupos)
        cov = [np.stack([ctx.tab.M[x] for x in cols], -1)] if cols else []
        ex = self._extra(ctx)
        if ex is not None:
            cov.append(ex)
        cov = np.concatenate(cov, -1) if cov else None
        Wn = S._vizinhos(D)
        linc = np.nan_to_num(D.LINC)
        o_tr = np.arange(160, c)
        X, lv, anc = S._feats(D, P, Wn, o_tr, H, cov, ctx.tab, seas)
        t = o_tr[:, None] + np.arange(1, H + 1)[None]
        tt = np.minimum(t, D.T - 1)
        y = linc[tt] - lv[:, None]
        m = np.broadcast_to((t <= c)[..., None], y.shape)
        w = None
        if self.peso_pico:
            alto = linc[tt] >= np.log1p(self.lim.p75.values)[None, None]
            w = (1 + self.peso_pico * alto)[m]
        o_ev = S.origens_de(R)
        Xp, lvp, _ = S._feats(D, P, Wn, o_ev, H, cov, ctx.tab, seas)
        return X[m], y[m], w, Xp, lvp, o_ev, c

    def prever_semanal(self, ctx, R):
        X, y, w, Xp, lvp, o, c = self.dados(ctx, R)
        B, H, N, F = Xp.shape
        Xpf = Xp.reshape(-1, F)
        pr = np.sort(np.asarray(_quantis_peso(self.motor, X, y, Xpf, 7, self.n_estimators, w, self.prm)), -1)
        if self.mediana_L1:
            import lightgbm as lgb
            med = lgb.LGBMRegressor(objective='l1', n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=7) \
                .fit(X, y, sample_weight=w).predict(Xpf)
            pr = pr + (med - pr[:, 4])[:, None]
        lq = lvp[:, None, :, None] + pr.reshape(B, H, N, 9)
        lq = np.minimum(lq, S._teto(ctx.Du, c)[None, None, :, None])
        return o, S._para_casos(lq, ctx.Du, o)


# =====================================================================================================================
# B1 / B2 / D1 / D3 · redes de grafo com escala por UF, pré-treino nas regiões, pesos nos quantis e cabeça do total
# =====================================================================================================================
PESO_Q_ALTO = np.array([0.6, 0.6, 0.7, 0.8, 1.0, 1.3, 1.6, 1.8, 2.0])      # D1: quantis altos pesam mais


def _amostras(D, c, L, H, alvo, escala, peso_pop=None, cov=None, o_ini=160):
    P = V.PreparaA(D, c, L=L, H=H)
    S_ = P.S if cov is None else np.concatenate([P.S, cov], -1).astype(np.float32)
    o = np.arange(max(o_ini, L), c)
    t = o[:, None] + np.arange(1, H + 1)[None]
    linc = np.nan_to_num(D.LINC)
    base = S._base(P, D, o, H, alvo)                                           # (B, H, N)
    esc = (np.nanstd(D.LINC[:c + 1], 0) + 0.1).astype(np.float32) if escala else np.ones(D.N, np.float32)
    Y = ((linc[np.minimum(t, D.T - 1)] - base) / esc[None, None]).astype(np.float32)
    Mk = np.repeat((t <= c)[..., None], D.N, -1).astype(np.float32)
    if peso_pop:
        pw = D.POP[c] ** peso_pop
        Mk = Mk * (pw / pw.mean()).astype(np.float32)[None, None]
    X = np.stack([P.X[x - L + 1:x + 1] for x in o]).astype(np.float32)
    A = D.ADJ + np.eye(D.N)
    dg = 1 / np.sqrt(A.sum(1))
    A = (A * dg[:, None] * dg[None]).astype(np.float32)
    return dict(P=P, S=S_, o=o, X=X, Y=Y, Mk=Mk, base=base.astype(np.float32), esc=esc, A=A, t=t,
                ytrue=linc[np.minimum(t, D.T - 1)].astype(np.float32))


def _perda(pred, y, m, qt, wq, base=None, esc=None, ytrue=None, tot=None):
    """pinball ponderado por quantil (D1) + λ_tot × pinball no log1p do total das semanas tot=(h_ini, h_fim) (D3).
    pred (B,N,H,9) em unidades escaladas; y,m (B,N,H)."""
    import torch
    e = y[..., None] - pred
    l = torch.maximum(qt * e, (qt - 1) * e) * wq
    perda = (l * m[..., None]).sum() / (m.sum() * len(qt) + 1e-9)
    if tot is not None and tot[2] > 0:
        a, b, lam = tot
        lq = base[:, :, a - 1:b, None] + esc[None, :, None, None] * pred[:, :, a - 1:b]      # log1p(inc) por quantil
        tq = torch.log1p(torch.clamp(torch.expm1(lq), min=0).sum(2))                     # (B, N, 9)
        ty = torch.log1p(torch.clamp(torch.expm1(ytrue[:, :, a - 1:b]), min=0).sum(2))   # (B, N)
        mt = (m[:, :, a - 1:b].min(2).values > 0).float()                               # total só se todas as semanas ≤ c
        et = ty[..., None] - tq
        lt = torch.maximum(qt * et, (qt - 1) * et) * mt[..., None]
        perda = perda + lam * lt.sum() / (mt.sum() * len(qt) + 1e-9)
    return perda


def treinar_ckpt(rede_cfg, net, dd, epocas, lr, seed, dev, arq, bs=128, wq=None, tot=None, ckpt_cada=5):
    """treino com checkpoint a cada `ckpt_cada` épocas em `arq`.part (modelo, otimizador, agenda, época);
    grava `arq` (só os pesos) no fim. Se `arq` existe, só carrega."""
    import torch
    if os.path.exists(arq):
        net.load_state_dict(torch.load(arq, map_location=dev))
        return net
    torch.manual_seed(seed)
    T_ = lambda a, tp=torch.float32: torch.tensor(a, dtype=tp, device=dev)   # noqa: E731
    Xt, St, Wt = T_(dd['X']), T_(dd['S'][dd['o']]), T_(rede_cfg.D.woyc[dd['o']], torch.long)
    Yt, Mt = T_(dd['Y'].transpose(0, 2, 1)), T_(dd['Mk'].transpose(0, 2, 1))
    Bt, Et, Ytr = T_(dd['base'].transpose(0, 2, 1)), T_(dd['esc']), T_(dd['ytrue'].transpose(0, 2, 1))
    qt = T_(Q)
    wqt = T_(wq if wq is not None else np.ones(9))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    ep0 = 0
    if os.path.exists(arq + '.part'):
        ck = torch.load(arq + '.part', map_location=dev)
        net.load_state_dict(ck['net'])
        opt.load_state_dict(ck['opt'])
        sch.load_state_dict(ck['sch'])
        ep0 = ck['ep']
        print(f'   retomando {os.path.basename(arq)} na época {ep0}', flush=True)
    n = len(Xt)
    for ep in range(ep0, epocas):
        net.train()
        ordem = np.random.default_rng(seed * 1000 + ep).permutation(n)
        for b in np.array_split(ordem, max(1, n // bs)):
            pred = net(Xt[b], St[b], Wt[b])
            loss = _perda(pred, Yt[b], Mt[b], qt, wqt, Bt[b], Et, Ytr[b], tot)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sch.step()
        if (ep + 1) % ckpt_cada == 0 and ep + 1 < epocas:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), ep=ep + 1), arq + '.part')
    torch.save(net.state_dict(), arq + '.tmp')
    os.replace(arq + '.tmp', arq)
    if os.path.exists(arq + '.part'):
        os.remove(arq + '.part')
    return net


class _Cfg:
    def __init__(self, D):
        self.D = D


def _transfere(net, estado):
    """copia do pré-treino só os parâmetros de mesmo formato (os específicos de cada nó ficam novos)."""
    meu = net.state_dict()
    ok = {k: v for k, v in estado.items() if k in meu and meu[k].shape == v.shape}
    meu.update(ok)
    net.load_state_dict(meu)
    return len(ok), len(meu)


def treinar_r2(ctx, cfg, c_data, H, pasta_ckpt, chave):
    """treina as sementes de uma configuração (dict) até a data de corte; devolve (redes, dados)."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    D = ctx.Du
    c = int(np.searchsorted(pd.to_datetime(D.datas), pd.Timestamp(c_data), 'right') - 1)
    rede = cfg['fabrica']()
    if cfg.get('cov_fn'):                                   # R2_6: variáveis de fase / tsfresh
        cov = cfg['cov_fn'](ctx, c)
    else:
        cov = S.cov_padronizada(ctx, c, cfg['cov'], cfg.get('arq_sel')) if cfg.get('cov') else None
    dd = _amostras(D, c, rede.L, H, cfg['alvo'], cfg['escala'], cfg.get('peso_pop'), cov)
    wq = cfg.get('peso_q')
    tot = cfg.get('tot')
    redes = []
    for seed in cfg.get('sementes', rede.sementes):
        torch.manual_seed(seed)
        net = rede.construir(nn, torch, D.N, dd['X'].shape[-1], dd['S'].shape[-1], H, dd['A'], rede.d).to(dev)
        if cfg.get('pre'):
            arq_pre = os.path.join(pasta_ckpt, f'pre_{cfg["pre_nome"]}_{chave}_s{seed}.pt')
            if not os.path.exists(arq_pre):
                Dr = ctx.Dreg
                cr = int(np.searchsorted(pd.to_datetime(Dr.datas), pd.Timestamp(c_data), 'right') - 1)
                ddr = _amostras(Dr, cr, rede.L, H, cfg['alvo'], cfg['escala'], None, None)
                netr = rede.construir(nn, torch, Dr.N, ddr['X'].shape[-1], ddr['S'].shape[-1], H, ddr['A'], rede.d).to(dev)
                treinar_ckpt(_Cfg(Dr), netr, ddr, cfg.get('ep_pre', 30), rede.lr, seed, dev, arq_pre,
                             bs=cfg.get('bs_pre', 16), wq=wq, tot=tot)
                del netr
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            k, n = _transfere(net, torch.load(arq_pre, map_location=dev))
        arq = os.path.join(pasta_ckpt, f'{cfg["nome"]}_{chave}_s{seed}.pt')
        treinar_ckpt(_Cfg(D), net, dd, cfg.get('ep_ajuste', rede.epocas) if cfg.get('pre') else rede.epocas,
                     rede.lr * (0.5 if cfg.get('pre') else 1.0), seed, dev, arq, bs=cfg.get('bs', 128), wq=wq, tot=tot)
        net.eval()
        redes.append(net)
    return redes, dd, c


def prever_r2(redes, dd, D, L, o, alvo, lote=32):
    r = S.prever_rede(redes, dd['P'], dd['S'], D, L, o, lote)                 # (B, H, N, 9) escalado
    return S._base(dd['P'], D, o, r.shape[1], alvo)[..., None] + dd['esc'][None, None, :, None] * r


class RedeSemanalR2(S.ModeloSemanal):
    familia, usa_gpu = 'rede de grafos (rodada 2)', True

    def __init__(self, cfg, pasta_ckpt):
        self.cfg, self.pasta_ckpt, self.nome = cfg, pasta_ckpt, cfg['nome']

    def prever_semanal(self, ctx, R):
        D = ctx.Du
        c_data = pd.Timestamp(D.datas[R['c']]) + pd.Timedelta(days=6)
        redes, dd, c = treinar_r2(ctx, self.cfg, c_data, self.cfg.get('H', H_SEM), self.pasta_ckpt, f"c{R['c']}")
        o = S.origens_de(R)
        lq = prever_r2(redes, dd, D, self.cfg['fabrica']().L, o, self.cfg['alvo'])[:, :H_SEM]
        lq = np.minimum(lq, S._teto(D, c)[None, None, :, None])
        return o, S._para_casos(lq, D, o)


class RedeTemporadaR2(Modelo):
    familia, origem, usa_gpu = 'rede de grafos (rodada 2)', 'rodada 2', True

    def __init__(self, cfg, pasta_ckpt):
        self.cfg, self.pasta_ckpt, self.nome = cfg, pasta_ckpt, cfg['nome']

    def prever(self, ctx, s):
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        c_data = pd.Timestamp(D.datas[c]) + pd.Timedelta(days=6)
        redes, dd, c2 = treinar_r2(ctx, self.cfg, c_data, 68, self.pasta_ckpt, f's{s}')
        assert c2 == c, (c2, c)
        lq = prever_r2(redes, dd, D, self.cfg['fabrica']().L, np.array([c]), self.cfg['alvo'])[0]   # (68, N, 9)
        lq = np.minimum(np.sort(lq, -1), S._teto(D, c)[None, :, None])[h - 1]
        return Saida(quant=CM._para_contagem(lq, CM._pop_c(D, s)[None, :, None]))


# =====================================================================================================================
# E1 · Optuna com estudo salvo no Drive (retoma) e objetivo da temporada sem depender de outras pastas
# =====================================================================================================================
def estudo_optuna(nome, pasta, direcao='minimize'):
    """estudo em SQLite LOCAL (/content), copiado para o Drive após cada tentativa e restaurado no início."""
    import optuna
    local = f'/content/{nome}.db' if os.path.isdir('/content') else os.path.abspath(f'{nome}.db')
    drive = os.path.join(pasta, f'{nome}.db')
    if os.path.exists(drive) and not os.path.exists(local):
        shutil.copy(drive, local)
    st = optuna.create_study(study_name=nome, storage=f'sqlite:///{local}', direction=direcao, load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=1))

    def salva(study, trial):
        shutil.copy(local, drive + '.tmp')
        os.replace(drive + '.tmp', drive)
    return st, salva


def R_temporada_vs_clim(ctx, modelo, temporadas):
    """WIS relativo à climatologia (média geométrica por UF), calculado direto, para o objetivo do Optuna."""
    D = ctx.Du
    lr = []
    for s in temporadas:
        out = modelo.prever(ctx, s)
        datas, qc = V.base_climatologia(D, s)
        y = ctx.ver_uf.y(datas)
        ok = ~np.isnan(y)
        wm = np.nanmean(np.where(ok, V.wis(out.quant, y), np.nan), 0)
        wc = np.nanmean(np.where(ok, V.wis(qc, y), np.nan), 0)
        lr.append(np.log(wm / wc))
    return float(np.exp(np.nanmean(np.nanmean(lr, 0))))


# =====================================================================================================================
# E3 · Chronos-2 ajustado (fine-tuning) aos dados de dengue, salvo no Drive
# =====================================================================================================================
def _sem_torchao_velho():
    """o Colab traz torchao 0.10 e o peft (LoRA) recusa versões < 0.16 com ImportError mesmo sem usar o torchao.
    Se a versão instalada for velha, faz o peft enxergar 'torchao indisponível' (vale também na sessão já aberta)."""
    import sys
    try:
        import importlib.metadata as md
        from packaging.version import Version
        if Version(md.version('torchao')) >= Version('0.16.0'):
            return
    except Exception:                                       # torchao não instalado: nada a fazer
        return
    import peft.import_utils as iu
    falso = lambda *a, **k: False                           # noqa: E731
    for nome, mod in list(sys.modules.items()):
        if nome.startswith('peft') and hasattr(mod, 'is_torchao_available'):
            setattr(mod, 'is_torchao_available', falso)
    iu.is_torchao_available = falso


def chronos_ajustado(pasta, chave, series, prediction_length, num_steps=1000, lr=1e-5, base='amazon/chronos-2'):
    """ajusta o Chronos-2 nas séries (lista de arrays) e salva em <pasta>/<chave>; se já existe, só carrega."""
    import torch
    from chronos import Chronos2Pipeline
    _sem_torchao_velho()
    dm = 'cuda' if torch.cuda.is_available() else 'cpu'
    destino = os.path.join(pasta, chave)
    if os.path.exists(os.path.join(destino, 'PRONTO')):
        return Chronos2Pipeline.from_pretrained(destino, device_map=dm)
    pipe = Chronos2Pipeline.from_pretrained(base, device_map=dm)
    if not hasattr(pipe, 'fit'):
        raise RuntimeError('esta versão do chronos-forecasting não tem Chronos2Pipeline.fit; atualize o pacote (pip install -U chronos-forecasting)')
    kw = dict(inputs=[{'target': np.asarray(s, np.float32)} for s in series], prediction_length=prediction_length,
              num_steps=num_steps, learning_rate=lr, batch_size=32, finetune_mode='lora')
    aceitos = set(inspect.signature(pipe.fit).parameters)
    if 'inputs' in aceitos:
        ft = pipe.fit(**{k: v for k, v in kw.items() if k in aceitos})
    else:                                                   # 1º argumento com outro nome
        ft = pipe.fit(kw['inputs'], **{k: v for k, v in kw.items() if k in aceitos and k != 'inputs'})
    ft = ft if ft is not None else pipe
    tmp = destino + '_tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    (ft if hasattr(ft, 'save_pretrained') else ft.model).save_pretrained(tmp)
    open(os.path.join(tmp, 'PRONTO'), 'w').write(time.ctime())
    shutil.rmtree(destino, ignore_errors=True)
    shutil.move(tmp, destino)
    return Chronos2Pipeline.from_pretrained(destino, device_map=dm)


class Chronos2FTSemanal(S.Chronos2Semanal):
    def __init__(self, pasta, num_steps=1000, contexto=520):
        super().__init__(contexto=contexto)
        self.nome, self.pasta, self.num_steps = 'Chronos2FT_sem', pasta, num_steps

    def prever_semanal(self, ctx, R):
        D = ctx.Du
        c = R['c']
        series = [np.nan_to_num(D.LINC[max(0, c + 1 - 520):c + 1, j]) for j in range(D.N)]
        self._pipe = chronos_ajustado(self.pasta, f'sem_c{c}', series, H_SEM, self.num_steps)
        return super().prever_semanal(ctx, R)


class Chronos2FTTemporada(U.Chronos2):
    def __init__(self, pasta, num_steps=1000):
        super().__init__(covariaveis=False, nome='Chronos2FT_UF')
        self.pasta, self.num_steps = pasta, num_steps

    def prever(self, ctx, s):
        D = ctx.Du
        c = D.corte(s)
        series = [np.nan_to_num(D.LINC[max(0, c + 1 - 520):c + 1, j]) for j in range(D.N)]
        self._pipe = chronos_ajustado(self.pasta, f'temp_s{s}', series, 68, self.num_steps)
        return super().prever(ctx, s)


# =====================================================================================================================
# R2_6 · variáveis da rodada 2 (fase/Rt e tsfresh) nos outros modelos: redes, ARIMA com regressores, GBM otimizado
# =====================================================================================================================
_CACHE_FASE = {}


def _fase_cache(D, lim):
    k = (id(D), tuple(np.round(lim.p25.values, 6)), tuple(np.round(lim.p75.values, 6)))
    if k not in _CACHE_FASE:
        _CACHE_FASE[k] = vars_fase(D, (lim.p25.values, lim.p75.values))
    return _CACHE_FASE[k]


def cov_r2(ctx, c, lim, fase=True, tsf=None, tsf_M=None, tsf_cols=None, corte=5.0):
    """(T, N, k) padronizadas só com dados até c (sem vazamento), NaN → 0, limitadas a ±corte desvios."""
    D = ctx.Du
    partes = []
    if fase:
        partes.append(_fase_cache(D, lim))
    if tsf:
        partes.append(tsf_M[..., [tsf_cols.index(x) for x in tsf]])
    M = np.concatenate(partes, -1).astype(np.float32)
    mu, sd = np.nanmean(M[:c + 1], (0, 1)), np.nanstd(M[:c + 1], (0, 1)) + 1e-6
    return np.clip(np.nan_to_num((M - mu) / sd), -corte, corte).astype(np.float32)


def fabrica_cov(lim, fase=True, tsf=None, tsf_M=None, tsf_cols=None):
    """função (ctx, c) → covariáveis, para cfg['cov_fn'] das redes."""
    return lambda ctx, c: cov_r2(ctx, c, lim, fase, tsf, tsf_M, tsf_cols)


class ARIMAXSemanalR2(S.ARIMASemanal):
    """ARIMA_Fourier_sem + regressores de fase (log Rt, variação do Rt, nível − p75) defasados `defasagem` semanas
    (≥ 4, assim o valor usado para prever h = 1…4 já é conhecido na origem). Padronização só com dados até c."""

    def __init__(self, lim, defasagem=4, vars_idx=(0, 1, 4), K=3, anos=8, nome='ARIMAX_fase_sem'):
        super().__init__(auto=False, K=K, anos=anos, nome=nome)
        assert defasagem >= self.H
        self.lim, self.defasagem, self.vars_idx = lim, defasagem, list(vars_idx)

    def prever_semanal(self, ctx, R):
        import warnings
        from scipy.stats import norm
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        warnings.filterwarnings('ignore')
        D, H, c = ctx.Du, self.H, R['c']
        F = _fase_cache(D, self.lim)[..., self.vars_idx]
        mu, sd = F[:c + 1].mean((0, 1)), F[:c + 1].std((0, 1)) + 1e-6
        F = np.clip((F - mu) / sd, -5, 5)
        ini, fim = max(self.defasagem, c - self.anos * 52), min(R['t_fim'] + H, D.T - 1)
        k = np.arange(ini, fim + 1)
        four = np.column_stack([f(2 * np.pi * j * k / 52.18) for j in range(1, self.K + 1) for f in (np.sin, np.cos)])
        ntr = c - ini + 1
        o = S.origens_de(R)
        z = norm.ppf(Q)
        lq = np.zeros((len(o), H, D.N, 9))
        for j in range(D.N):
            ex = np.column_stack([four, F[k - self.defasagem, j]])
            y = D.LINC[ini:fim + 1, j].astype(float)
            try:
                fit = SARIMAX(y[:ntr], exog=ex[:ntr], order=(2, 0, 1), trend='c').fit(disp=False, maxiter=200)
            except Exception:                                # se não convergir, cai no ARIMA_Fourier sem regressores
                ex = four
                fit = SARIMAX(y[:ntr], exog=ex[:ntr], order=(2, 0, 1), trend='c').fit(disp=False, maxiter=200)
            r = fit.apply(y, exog=ex)
            for i, oo in enumerate(o):
                s0 = oo + 1 - ini
                pr = r.get_prediction(start=s0, end=s0 + H - 1, dynamic=True)
                mu_, se = np.asarray(pr.predicted_mean), np.sqrt(np.maximum(np.asarray(pr.var_pred_mean), 1e-9))
                lq[i, :, j] = mu_[:, None] + se[:, None] * z[None]
        lq = np.minimum(lq, S._teto(D, c)[None, None, :, None])
        return o, S._para_casos(lq, D, o)


# =====================================================================================================================
# R2_7 · gráficos por UF (semanal e temporada) com os campeões de cada área
# =====================================================================================================================
SIGLAS = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN',
          25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC',
          43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF', 0: 'Brasil'}
COR = dict(obs='#1a1a1a', sem1='#2a6fdb', sem4='#d9822b', temp='#2a6fdb', pico='#c0392b', total='#16a085', bb='#7f8c8d')


def _serie(df, u):
    """previsões (date × QCOLS) de uma UF; u = 0 soma as UFs (aproximação: soma quantil a quantil)."""
    g = df if u == 0 else df[df.uf_code == u]
    return g.groupby('date')[QCOLS].sum().sort_index() if u == 0 else g.set_index('date')[QCOLS].sort_index()


def totais_temporada(obs, temp, u, temporadas):
    """total observado e previsto (mediana) por temporada, para a UF u."""
    linhas = []
    for s in temporadas:
        ini, fim = pd.Timestamp(f'{s}-10-01'), pd.Timestamp(f'{s + 1}-10-15')
        o = obs[(obs.index >= ini) & (obs.index <= fim)]
        lin = dict(uf=SIGLAS.get(u, u), temporada=f'{s}/{s + 1}')
        ok = None
        for rot, df in temp.items():
            p = _serie(df[df.season == s], u)
            if len(p):
                lin[rot] = float(p['pred'].sum())
                ok = p.index
        if ok is not None:
            lin['observado'] = float(o.reindex(ok).sum())
        linhas.append(lin)
    return pd.DataFrame(linhas)


def grafico_uf(obs, sem, temp, u, arq, temporadas, titulo_extra=''):
    """obs: Series (date → casos) da UF; sem: {'h=1 · nome': df, 'h=4 · nome': df} (date, uf_code, QCOLS);
    temp: {'rótulo': df} — o 1º recebe o leque (IC50/IC90), os outros só a mediana. Grava PNG em arq e devolve a figura."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.2))
    nome = SIGLAS.get(u, str(u))
    # --- semanal
    (r1, d1), (r4, d4) = list(sem.items())[:2]
    p1, p4 = _serie(d1, u), _serie(d4, u)
    if len(p1):
        x0, x1 = p1.index.min(), p1.index.max()
        o = obs[(obs.index >= x0) & (obs.index <= x1)]
        a1.fill_between(p1.index, p1['lower_90'], p1['upper_90'], color=COR['sem1'], alpha=.18, lw=0, label='IC 90% (1 semana)')
        a1.plot(p1.index, p1['pred'], color=COR['sem1'], lw=1.6, label=r1)
        if len(p4):
            a1.plot(p4.index, p4['pred'], color=COR['sem4'], lw=1.3, ls='--', label=r4)
        a1.plot(o.index, o.values, color=COR['obs'], lw=1.1, marker='o', ms=2, label='casos observados')
    a1.set_title(f'{nome} · previsão semanal (teste 2022–2025): 1 e 4 semanas à frente{titulo_extra}', loc='left', fontsize=11)
    a1.set_ylabel('casos por semana')
    a1.legend(loc='upper left', fontsize=8, frameon=False)
    # --- temporada
    rots = list(temp.keys())
    ultimo = None
    for i, rot in enumerate(rots):
        p = _serie(temp[rot][temp[rot].season.isin(temporadas)], u)
        if not len(p):
            continue
        if i == 0:
            for s in temporadas:
                ps = _serie(temp[rot][temp[rot].season == s], u)
                if len(ps):
                    a2.fill_between(ps.index, ps['lower_90'], ps['upper_90'], color=COR['temp'], alpha=.13, lw=0,
                                    label='IC 90% (temporada)' if s == temporadas[0] else None)
                    a2.fill_between(ps.index, ps['lower_50'], ps['upper_50'], color=COR['temp'], alpha=.22, lw=0,
                                    label='IC 50%' if s == temporadas[0] else None)
                    a2.plot(ps.index, ps['pred'], color=COR['temp'], lw=1.6, label=rot if s == temporadas[0] else None)
                    ultimo = ps.index
        else:
            cor = [COR['pico'], COR['total'], COR['bb']][min(i - 1, 2)]
            ls = ['-', ':', '--'][min(i - 1, 2)]
            for s in temporadas:
                ps = _serie(temp[rot][temp[rot].season == s], u)
                if len(ps):
                    a2.plot(ps.index, ps['pred'], color=cor, lw=1.2, ls=ls, label=rot if s == temporadas[0] else None)
    x0 = pd.Timestamp(f'{temporadas[0]}-10-01')
    x1 = ultimo.max() if ultimo is not None else pd.Timestamp(f'{temporadas[-1] + 1}-10-15')
    o = obs[(obs.index >= x0) & (obs.index <= x1)]
    a2.plot(o.index, o.values, color=COR['obs'], lw=1.1, marker='o', ms=2, label='casos observados')
    for s in temporadas:
        a2.axvline(pd.Timestamp(f'{s}-10-08'), color='#bbbbbb', lw=.8, ls=':')
    a2.set_title(f'{nome} · previsão da temporada inteira (feita em junho, SE25, para SE41→SE40)', loc='left', fontsize=11)
    a2.set_ylabel('casos por semana')
    a2.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)
    for a in (a1, a2):
        a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        a.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        a.spines[['top', 'right']].set_visible(False)
        a.grid(axis='y', color='#eeeeee')
        a.set_ylim(bottom=0)
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'.replace(',', '.')))
    fig.tight_layout()
    fig.savefig(arq + '.tmp.png', dpi=110)
    os.replace(arq + '.tmp.png', arq)
    return fig
