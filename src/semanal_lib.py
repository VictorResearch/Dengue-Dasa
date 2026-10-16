"""
Previsão SEMANAL (1 a 4 semanas à frente) dos modelos de temporada, nas mesmas 26 UFs e nas mesmas 8 temporadas
(afinação 2018–2021, teste 2022–2025), mais a rodada do Felipe (treino até 31/12/2022, avaliação 2023–2024).
Também traz os próximos passos da TEMPORADA (MTGNN v2: alvo em crescimento / misto / perda ponderada pela população)
e as funções de avaliação que põem os dois níveis lado a lado.

Protocolo semanal (igual para todos os modelos):
  rodada da temporada s: avalia as semanas-alvo SE41/s → SE40/s+1 (até o último dado); o modelo é treinado UMA vez,
  só com alvos até c = (primeira semana-alvo − 4) e depois aplicado a cada origem o = c … fim−1 com os dados
  observados até o (sem reajustar), prevendo o+1 … o+4. É o "modelo fixo por rodada" do piloto do Felipe.
  rodada felipe_2023-24: c = última semana até 31/12/2022; alvos de 01/01/2023 a 31/12/2024 (o teste dele).

Saída (uma por modelo × rodada, retomável): resultados_semanal/<modelo>__<rodada>.parquet com
  modelo, rodada, season, h, origem, date, uf_code e os 9 quantis (casos).
"""
import glob
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import norm

import v4_lib as V
import comp_lib as L
import comp_modelos as CM
import uf_lib as U
from comp_lib import Modelo, Saida

Q, QCOLS, NIVEIS = V.Q, V.QCOLS, V.NIVEIS
TUNING, TESTE = U.TUNING, U.TESTE
H_SEM = 4
COLS_SEM = ['modelo', 'rodada', 'season', 'h', 'origem', 'date', 'uf_code'] + QCOLS
REFS_SEM = ('Climatologia_sem', 'Persistencia_UF')


def grupo_rodada(r):
    if r.startswith('felipe'):
        return 'felipe'
    return 'afinacao' if int(r[1:]) in TUNING else 'teste'


# =====================================================================================================================
# rodadas, gravação e execução retomável
# =====================================================================================================================
def rodadas(D, H=H_SEM, temporadas=None, felipe=True):
    out = {}
    for s in (temporadas if temporadas is not None else TUNING + TESTE):
        datas = V.datas_temporada(s)
        t_ini = D.idx_data[pd.Timestamp(datas[0])]
        t_fim = min(D.idx_data[pd.Timestamp(datas[-1])], D.ultimo_obs)
        if t_fim >= t_ini:
            out[f's{s}'] = dict(c=int(t_ini - H), t_ini=int(t_ini), t_fim=int(t_fim), season=int(s))
    if felipe:
        d = pd.to_datetime(D.datas)
        c = int(np.searchsorted(d, pd.Timestamp('2022-12-25'), 'right') - 1)        # semana 25–31/12/2022
        t_ini = int(np.searchsorted(d, pd.Timestamp('2023-01-01'), 'left'))
        t_fim = int(np.searchsorted(d, pd.Timestamp('2024-12-25'), 'right') - 1)    # última semana inteira de 2024
        out['felipe_2023-24'] = dict(c=c, t_ini=t_ini, t_fim=min(t_fim, D.ultimo_obs), season=-1)
    return out


def origens_de(R, H=H_SEM):
    if 'origens' in R:                     # produção: só as origens pedidas (ex.: a última semana observada)
        return np.asarray(R['origens'], dtype=int)
    return np.arange(R['t_ini'] - H, R['t_fim'])


def salvar(df, arq):
    """grava num .tmp e renomeia: um arquivo pela metade (queda de energia/internet) nunca vira resultado."""
    tmp = arq + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, arq)


def _teto(D, c):
    return np.log1p(3 * np.nanmax(D.INC[:c + 1], 0))                           # (N,)


def _para_casos(lq, D, o):
    """lq (B, H, N, 9) em log1p(incidência por 100 mil) → casos, com a população da semana-alvo."""
    t = np.minimum(o[:, None] + np.arange(1, lq.shape[1] + 1)[None], D.T - 1)
    return np.clip(np.expm1(lq), 0, None) * D.POP[t][..., None] / 1e5


def montar(nome, rod, R, D, o, q):
    B, H, N, _ = q.shape
    oo, k = np.repeat(o, H), np.tile(np.arange(1, H + 1), B)
    t = oo + k
    ok = (t >= R['t_ini']) & (t <= R['t_fim'])
    n = int(ok.sum())
    df = pd.DataFrame(q.reshape(B * H, N, 9)[ok].reshape(-1, 9), columns=QCOLS)
    df['modelo'], df['rodada'], df['season'] = nome, rod, R['season']
    df['h'] = np.repeat(k[ok], N)
    df['origem'] = np.repeat(pd.to_datetime(D.datas[oo[ok]]), N)
    df['date'] = np.repeat(pd.to_datetime(D.datas[t[ok]]), N)
    df['uf_code'] = np.tile(D.unidades, n)
    return df[COLS_SEM]


class ModeloSemanal:
    nome, familia, usa_gpu = 'base', '', False
    H = H_SEM

    def prever_semanal(self, ctx, R):
        """→ (origens (B,), quantis em casos (B, H, N, 9))."""
        raise NotImplementedError


def rodar_semanal(ctx, modelo, rods, pasta, refazer=False):
    """roda um modelo em todas as rodadas que faltam; cada rodada vira um arquivo (retoma de onde parou).
    Devolve a lista de erros (um erro numa rodada não interrompe as outras)."""
    os.makedirs(os.path.join(pasta, 'log'), exist_ok=True)
    erros = []
    for rod, R in rods.items():
        arq = os.path.join(pasta, f'{modelo.nome}__{rod}.parquet')
        if os.path.exists(arq) and not refazer:
            continue
        t0 = time.time()
        try:
            o, q = modelo.prever_semanal(ctx, R)
            assert q.shape[1:] == (H_SEM, ctx.Du.N, 9), f'formato {q.shape}'
            salvar(montar(modelo.nome, rod, R, ctx.Du, o, V.arruma_quantis(q)), arq)
            dt = time.time() - t0
            with open(os.path.join(pasta, 'log', f'{modelo.nome}__{rod}.json'), 'w') as f:
                json.dump(dict(modelo=modelo.nome, rodada=rod, segundos=dt), f)
            print(f'✅ {modelo.nome:34s} {rod:15s} {dt / 60:5.1f} min', flush=True)
        except Exception as e:
            erros.append((modelo.nome, rod, repr(e)[:500]))
            print(f'❌ {modelo.nome:34s} {rod:15s} {repr(e)[:300]}', flush=True)
            traceback.print_exc(limit=2)
    return erros


# =====================================================================================================================
# referências semanais
# =====================================================================================================================
class PersistenciaQ(ModeloSemanal):
    """repete o último valor; quantis = quantis empíricos do crescimento em h semanas (log), no treino."""
    nome, familia = 'Persistencia_UF', 'referência'

    def prever_semanal(self, ctx, R):
        D, H, c = ctx.Du, self.H, R['c']
        linc, ini = D.LINC, max(1, c - 8 * 52)
        g = np.stack([np.nanquantile((linc[ini + h:c + 1] - linc[ini:c + 1 - h]).ravel(), Q) for h in range(1, H + 1)])
        g = g - g[:, [4]]
        o = origens_de(R)
        lq = np.nan_to_num(linc[o])[:, None, :, None] + g[None, :, None, :]
        return o, _para_casos(lq, D, o)


class ClimatologiaSem(ModeloSemanal):
    """quantis empíricos da incidência da mesma semana do ano (dados até c) × população: a mesma climatologia da
    temporada, só que avaliada semana a semana (é a referência comum aos dois níveis)."""
    nome, familia = 'Climatologia_sem', 'referência'

    def prever_semanal(self, ctx, R):
        D, H, c = ctx.Du, self.H, R['c']
        inc, w = D.INC[:c + 1], D.woyc[:c + 1]
        qw = np.stack([np.nanquantile(inc[w == k], Q, axis=0).T for k in range(1, 53)])     # (52, N, 9)
        o = origens_de(R)
        t = np.minimum(o[:, None] + np.arange(1, H + 1)[None], D.T - 1)
        return o, qw[D.woyc[t] - 1] * D.POP[t][..., None] / 1e5


# =====================================================================================================================
# GBM semanal (crescimento em log; com todas as variáveis ou com grupos selecionados na afinação)
# =====================================================================================================================
def _vizinhos(D):
    W = np.asarray(D.ADJ, float).copy()
    np.fill_diagonal(W, 0)
    return W / np.maximum(W.sum(1, keepdims=True), 1e-9)


def _feats(D, P, Wn, o, H, cov=None, tab=None, seas=False):
    """linhas (origem o, horizonte h, UF) → X (B, H, N, F). Tudo usa só dados até a origem o."""
    linc = np.nan_to_num(D.LINC)
    B, N = len(o), D.N
    hs = np.arange(1, H + 1)
    t = np.minimum(o[:, None] + hs[None], D.T - 1)
    lv = linc[o]
    d = [lv - linc[np.maximum(o - k, 0)] for k in (1, 2, 4, 8)]
    anc_t = P.CM[P.ult_woy[o[:, None], D.woyc[t]]]                           # (B, H, N)
    anc_o = P.CM[P.ult_woy[o, D.woyc[o]]]                                     # (B, N)
    ap = linc[np.maximum(t - 52, 0)]
    apd = ap - linc[np.maximum(t - 53, 0)]
    a = 2 * np.pi * D.woyc[t] / 52
    bh = lambda x: np.broadcast_to(x[:, None, :], (B, H, N))                 # noqa: E731
    hn = lambda x: np.broadcast_to(x[:, :, None], (B, H, N))                 # noqa: E731
    F = [bh(lv)] + [bh(x) for x in d] + [bh(lv @ Wn.T), bh(d[0] @ Wn.T), anc_t - bh(lv), bh(anc_o - lv), ap - bh(lv),
                                         apd, hn(np.sin(a)), hn(np.cos(a)), hn(np.broadcast_to(hs, (B, H)).astype(float)),
                                         np.broadcast_to(np.arange(N, dtype=float), (B, H, N)), bh(np.log(D.POP[o]))]
    partes = [np.stack(F, -1), np.broadcast_to(P.S[o][:, None], (B, H, N, P.S.shape[-1]))]
    if cov is not None:
        partes.append(np.broadcast_to(cov[o][:, None], (B, H, N, cov.shape[-1])))
    if seas:
        partes.append(np.stack([tab.seas_alvo(int(x), int(x) + hs) for x in o]))
    return np.concatenate(partes, -1).astype(np.float32), lv, anc_t


def _cols_grupos(grupos, arq_sel=None):
    """'casos' | 'todas' | 'sel' (lê os grupos escolhidos na afinação) | lista de grupos."""
    if grupos == 'sel':
        if not arq_sel or not os.path.exists(arq_sel):
            raise FileNotFoundError('rode antes a seleção de variáveis (selecionar_grupos)')
        grupos = json.load(open(arq_sel))['final']
    if grupos == 'casos':
        return [], False
    return U.colunas_grupos(grupos)


def _dados_gbm(ctx, R, grupos, alvo, arq_sel=None, o_ini=160):
    D, H, c = ctx.Du, H_SEM, R['c']
    P = V.PreparaA(D, c, L=104, H=H)
    cols, seas = _cols_grupos(grupos, arq_sel)
    cov = np.stack([ctx.tab.M[x] for x in cols], -1) if cols else None
    Wn = _vizinhos(D)
    linc = np.nan_to_num(D.LINC)
    o_tr = np.arange(o_ini, c)
    X, lv, anc = _feats(D, P, Wn, o_tr, H, cov, ctx.tab, seas)
    t = o_tr[:, None] + np.arange(1, H + 1)[None]
    base = anc if alvo == 'ancora' else np.broadcast_to(lv[:, None], anc.shape)
    y = linc[np.minimum(t, D.T - 1)] - base
    m = np.broadcast_to((t <= c)[..., None], y.shape)
    o_ev = origens_de(R)
    Xp, lvp, ancp = _feats(D, P, Wn, o_ev, H, cov, ctx.tab, seas)
    basep = ancp if alvo == 'ancora' else np.broadcast_to(lvp[:, None], ancp.shape)
    return X[m], y[m], Xp, basep, o_ev, c


class GBMSemanal(ModeloSemanal):
    """LightGBM / XGBoost / CatBoost quantílicos (9 quantis), um modelo para h = 1…4 (h é variável).
    alvo 'cresc': log1p(inc) da semana-alvo − log1p(inc) da origem (extrapola recordes); 'ancora': − âncora sazonal.
    grupos: 'casos' (só casos, vizinhos, âncora e variáveis de origem do V4), 'todas' (+ todos os grupos de
    covariáveis do uf_lib, inclusive SEAS5) ou 'sel' (grupos escolhidos por `selecionar_grupos` na afinação)."""
    familia = 'aprendizado de máquina'

    def __init__(self, motor='lgb', grupos='todas', alvo='cresc', n_estimators=300, arq_sel=None, nome=None):
        self.motor, self.grupos, self.alvo, self.n_estimators, self.arq_sel = motor, grupos, alvo, n_estimators, arq_sel
        self.nome = nome or f'GBM_{motor}_{grupos}' + ('' if alvo == 'cresc' else f'_{alvo}') + '_sem'

    def prever_semanal(self, ctx, R):
        X, y, Xp, basep, o, c = _dados_gbm(ctx, R, self.grupos, self.alvo, self.arq_sel)
        B, H, N, F = Xp.shape
        pr = U._ajusta_quantis(self.motor, X, y, Xp.reshape(-1, F), 7, self.n_estimators)
        lq = basep[..., None] + np.sort(np.asarray(pr).reshape(B, H, N, 9), -1)
        lq = np.minimum(lq, _teto(ctx.Du, c)[None, None, :, None])
        return o, _para_casos(lq, ctx.Du, o)


def selecionar_grupos(ctx, rods, arq, motor='lgb', tol=0.002, n_estimators=200, alvo='cresc'):
    """seleção para trás por GRUPO de variáveis, só com as rodadas de afinação (2018–2021).
    Critério: erro absoluto médio da mediana em log1p(inc), h = 1…4. Tira o grupo cuja retirada mais melhora;
    para quando nenhuma retirada melhora mais que `tol`. O estado é gravado a cada avaliação (retoma de onde parou)."""
    import lightgbm as lgb
    est = json.load(open(arq)) if os.path.exists(arq) else {'avaliados': {}, 'atual': list(U.GRUPOS), 'historico': []}
    if est.get('final') is not None:
        return est['final']
    rt = {k: v for k, v in rods.items() if not k.startswith('felipe') and grupo_rodada(k) == 'afinacao'}
    linc = np.nan_to_num(ctx.Du.LINC)

    def nota(gs):
        chave = ','.join(sorted(gs)) or '(só casos)'
        if chave not in est['avaliados']:
            erros = []
            for rod, R in rt.items():
                X, y, Xp, basep, o, c = _dados_gbm(ctx, R, list(gs) if gs else 'casos', alvo)
                m = lgb.LGBMRegressor(objective='quantile', alpha=0.5, n_estimators=n_estimators, learning_rate=0.05,
                                      num_leaves=31, min_child_samples=100, subsample=0.7, subsample_freq=1,
                                      colsample_bytree=0.8, verbose=-1, random_state=1).fit(X, y)
                B, H, N, F = Xp.shape
                pred = basep + m.predict(Xp.reshape(-1, F)).reshape(B, H, N)
                t = o[:, None] + np.arange(1, H + 1)[None]
                ok = (t >= R['t_ini']) & (t <= R['t_fim'])
                erros.append(np.abs(pred - linc[np.minimum(t, ctx.Du.T - 1)])[ok].mean())
            est['avaliados'][chave] = float(np.mean(erros))
            json.dump(est, open(arq, 'w'), indent=1)
            print(f'   {est["avaliados"][chave]:.4f}  {chave}', flush=True)
        return est['avaliados'][chave]

    atual = list(est['atual'])
    while len(atual) > 0:
        base = nota(atual)
        cand = sorted((nota([g for g in atual if g != x]), x) for x in atual)
        melhor, g = cand[0]
        if melhor < base * (1 - tol):
            atual.remove(g)
            est['atual'] = atual
            est['historico'].append(dict(tirou=g, erro=melhor, antes=base))
            json.dump(est, open(arq, 'w'), indent=1)
            print(f'→ tira {g}: {base:.4f} → {melhor:.4f}', flush=True)
        else:
            break
    est['final'] = atual
    json.dump(est, open(arq, 'w'), indent=1)
    return atual


# =====================================================================================================================
# modelos do Felipe (piloto dele: contagens, lags 1/2/4/8/52, médias, sen/cos, população, UF) — um modelo por h
# =====================================================================================================================
class FelipeSemanal(ModeloSemanal):
    familia, origem = 'Felipe (piloto)', 'github felipe-matsuoka123/Dengue'

    def __init__(self, motor='xgb', quantis=False, nome=None):
        self.motor, self.quantis = motor, quantis
        self.nome = nome or f'Felipe_{motor.upper()}{"_q" if quantis else ""}_sem'

    @staticmethod
    def _X(D, o):
        Yc = np.nan_to_num(D.Y)
        B, N = len(o), D.N
        lags = [Yc[np.maximum(o - k + 1, 0)] for k in (1, 2, 4, 8, 52)]
        m4 = np.mean([Yc[np.maximum(o - j, 0)] for j in range(4)], 0)
        m8 = np.mean([Yc[np.maximum(o - j, 0)] for j in range(8)], 0)
        dia = pd.to_datetime(D.datas[np.minimum(o + 1, D.T - 1)]).dayofyear.values
        a = 2 * np.pi * (dia - 1) / 365.2425
        cols = lags + [m4, m8, np.repeat(np.sin(a)[:, None], N, 1), np.repeat(np.cos(a)[:, None], N, 1),
                       D.POP[np.minimum(o + 1, D.T - 1)]]
        X = np.stack(cols, -1)
        oh = np.broadcast_to(np.eye(N), (B, N, N))
        return np.concatenate([X, oh], -1).reshape(B * N, -1)

    def _modelo(self):
        if self.quantis:
            from xgboost import XGBRegressor
            return XGBRegressor(objective='reg:quantileerror', quantile_alpha=Q, n_estimators=300, max_depth=5,
                                learning_rate=0.05, tree_method='hist', random_state=42)
        if self.motor == 'xgb':
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, objective='reg:squarederror',
                                tree_method='hist', random_state=42)
        if self.motor == 'lgbm':
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=300, max_depth=5, num_leaves=31, learning_rate=0.05, random_state=42,
                                 deterministic=True, force_col_wise=True, verbosity=-1)
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05, loss_function='RMSE', random_seed=42,
                                 verbose=False, allow_writing_files=False)

    def prever_semanal(self, ctx, R):
        D, H, c = ctx.Du, self.H, R['c']
        d = pd.to_datetime(D.datas)
        o_min = max(52, int(np.searchsorted(d, pd.Timestamp('2015-01-04'))))
        o_ev = origens_de(R)
        q = np.zeros((len(o_ev), H, D.N, 9))
        for h in range(1, H + 1):
            o_tr = np.arange(o_min - h, c - h + 1)
            y = np.nan_to_num(D.Y[o_tr + h]).ravel()
            m = self._modelo().fit(self._X(D, o_tr), y)
            p = np.maximum(0, m.predict(self._X(D, o_ev)))
            q[:, h - 1] = (np.sort(p, -1) if self.quantis else np.repeat(p[:, None], 9, 1)).reshape(len(o_ev), D.N, 9)
        return o_ev, q


# =====================================================================================================================
# estatísticos: ARIMA com Fourier (parâmetros fixos na rodada, filtro de Kalman atualizado) e VAR por macrorregião
# =====================================================================================================================
class ARIMASemanal(ModeloSemanal):
    """SARIMAX em log1p(inc) + Fourier K=3; auto=True escolhe (p, q) pelo AIC em {1,2,3}×{0,1,2} (d = 0, como o
    AutoARIMA_d0). Ajusta até c e prevê h = 1…4 de cada origem com o filtro (sem reajustar)."""
    familia = 'estatístico'

    def __init__(self, auto=False, K=3, anos=8, nome=None):
        self.auto, self.K, self.anos = auto, K, anos
        self.nome = nome or ('AutoARIMA_d0_sem' if auto else 'ARIMA_Fourier_sem')

    def prever_semanal(self, ctx, R):
        import warnings
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        warnings.filterwarnings('ignore')
        D, H, c = ctx.Du, self.H, R['c']
        ini, fim = max(0, c - self.anos * 52), R['t_fim'] + H
        k = np.arange(ini, fim + 1)
        ex = np.column_stack([f(2 * np.pi * j * k / 52.18) for j in range(1, self.K + 1) for f in (np.sin, np.cos)])
        ntr = c - ini + 1
        o = origens_de(R)
        z = norm.ppf(Q)
        lq = np.zeros((len(o), H, D.N, 9))
        ordens = [(p, 0, q_) for p in (1, 2, 3) for q_ in (0, 1, 2)] if self.auto else [(2, 0, 1)]
        for j in range(D.N):
            y = D.LINC[ini:fim + 1, j].astype(float)                     # NaN depois do último dado: o filtro ignora
            fits = []
            for od in ordens:
                try:
                    fits.append(SARIMAX(y[:ntr], exog=ex[:ntr], order=od, trend='c').fit(disp=False, maxiter=200))
                except Exception:
                    pass
            r = min(fits, key=lambda f: f.aic).apply(y, exog=ex)
            for i, oo in enumerate(o):
                s0 = oo + 1 - ini
                pr = r.get_prediction(start=s0, end=s0 + H - 1, dynamic=True)
                mu, se = np.asarray(pr.predicted_mean), np.sqrt(np.maximum(np.asarray(pr.var_pred_mean), 1e-9))
                lq[i, :, j] = mu[:, None] + se[:, None] * z[None]
        lq = np.minimum(lq, _teto(D, c)[None, None, :, None])
        return o, _para_casos(lq, D, o)


class VARSemanal(ModeloSemanal):
    """VAR por macrorregião nas anomalias de log1p(inc) (− climatologia da semana, dados até c), p ≤ 4 pelo BIC;
    previsão iterada h = 1…4 de cada origem; incerteza pela soma Ψ Σ Ψ'."""
    familia, nome = 'estatístico', 'VAR_macro_sem'

    def __init__(self, pmax=4, anos=8):
        self.pmax, self.anos = pmax, anos

    def prever_semanal(self, ctx, R):
        D, H, c = ctx.Du, self.H, R['c']
        linc = np.nan_to_num(D.LINC)
        w = D.woyc
        clim = np.stack([linc[:c + 1][w[:c + 1] == k].mean(0) for k in range(1, 53)])         # (52, N)
        A = linc - clim[w - 1]
        ini = max(self.pmax, c - self.anos * 52)
        o = origens_de(R)
        z = norm.ppf(Q)
        lq = np.zeros((len(o), H, D.N, 9))
        macro = D.unidades // 10
        for g in np.unique(macro):
            cols = np.where(macro == g)[0]
            k = len(cols)
            melhor = None
            for p in range(1, self.pmax + 1):
                ts = np.arange(ini, c + 1)
                Xd = np.column_stack([np.ones(len(ts))] + [A[ts - i][:, cols] for i in range(1, p + 1)])
                Yd = A[ts][:, cols]
                B_ = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
                E = Yd - Xd @ B_
                Sig = E.T @ E / max(len(ts) - k * p - 1, 1)
                bic = np.linalg.slogdet(Sig)[1] + np.log(len(ts)) * (k * k * p + k) / len(ts)
                if melhor is None or bic < melhor[0]:
                    melhor = (bic, p, B_, Sig)
            _, p, B_, Sig = melhor
            c0, Ai = B_[0], [B_[1 + i * k:1 + (i + 1) * k].T for i in range(p)]
            Psi = [np.eye(k)]
            for j in range(1, H):
                Psi.append(sum(Ai[i - 1] @ Psi[j - i] for i in range(1, min(j, p) + 1)))
            var = np.stack([np.diag(sum(Psi[j] @ Sig @ Psi[j].T for j in range(h))) for h in range(1, H + 1)])
            for b, oo in enumerate(o):
                hist = [A[oo - i, cols] for i in range(p)]
                for hh in range(H):
                    yh = c0 + sum(Ai[i] @ hist[i] for i in range(p))
                    hist.insert(0, yh)
                    t = min(oo + hh + 1, D.T - 1)
                    lq[b, hh, cols] = (clim[w[t] - 1, cols] + yh)[:, None] + np.sqrt(var[hh])[:, None] * z[None]
        lq = np.minimum(lq, _teto(D, c)[None, None, :, None])
        return o, _para_casos(lq, D, o)


# =====================================================================================================================
# redes de grafo (as do uf_lib) com alvo / horizonte / covariáveis configuráveis — servem à semana E à temporada
# =====================================================================================================================
def _base(P, D, o, H, alvo):
    """nível de referência em log para os alvos o+1…o+H: 'ancora' (média sazonal), 'cresc' (último nível),
    'misto' (âncora + anomalia atual persistente)."""
    linc = np.nan_to_num(D.LINC)
    t = np.minimum(o[:, None] + np.arange(1, H + 1)[None], D.T - 1)
    anc_t = P.CM[P.ult_woy[o[:, None], D.woyc[t]]]
    if alvo == 'ancora':
        return anc_t
    lv = linc[o]
    if alvo == 'cresc':
        return np.broadcast_to(lv[:, None], anc_t.shape).copy()
    return anc_t + (lv - P.CM[P.ult_woy[o, D.woyc[o]]])[:, None]


def cov_padronizada(ctx, c, grupos='todas', arq_sel=None):
    cols, _ = _cols_grupos(grupos, arq_sel)
    if not cols:
        return None
    M = np.stack([ctx.tab.M[x] for x in cols], -1)
    mu, sd = np.nanmean(M[:c + 1], (0, 1)), np.nanstd(M[:c + 1], (0, 1)) + 1e-6
    return np.nan_to_num((M - mu) / sd).astype(np.float32)


def treinar_rede(rede, D, c, H, alvo='ancora', cov=None, peso=None, o_ini=160):
    """treina as sementes de uma rede do uf_lib (MTGNN, GCNGRU, STGCN) com alvos até c.
    peso (N,): multiplica a perda pinball de cada UF (perda ponderada pela população)."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    P = V.PreparaA(D, c, L=rede.L, H=H)
    S = P.S if cov is None else np.concatenate([P.S, cov], -1).astype(np.float32)
    o = np.arange(max(o_ini, rede.L), c)
    t = o[:, None] + np.arange(1, H + 1)[None]
    Y = (np.nan_to_num(D.LINC)[np.minimum(t, D.T - 1)] - _base(P, D, o, H, alvo)).astype(np.float32)   # (B, H, N)
    Mk = np.repeat((t <= c)[..., None], D.N, -1).astype(np.float32)
    if peso is not None:
        Mk = Mk * np.asarray(peso, np.float32)[None, None, :]
    X = np.stack([P.X[x - rede.L + 1:x + 1] for x in o]).astype(np.float32)
    A = D.ADJ + np.eye(D.N)
    dg = 1 / np.sqrt(A.sum(1))
    A = (A * dg[:, None] * dg[None]).astype(np.float32)
    T_ = lambda a, tp=torch.float32: torch.tensor(a, dtype=tp, device=dev)    # noqa: E731
    Xt, St, Wt = T_(X), T_(S[o]), T_(D.woyc[o], torch.long)
    Yt, Mt = T_(Y.transpose(0, 2, 1)), T_(Mk.transpose(0, 2, 1))
    redes = []
    for seed in rede.sementes:
        torch.manual_seed(seed)
        m = rede.construir(nn, torch, D.N, X.shape[-1], S.shape[-1], H, A, rede.d).to(dev)
        U._treina(m, lambda b: ((Xt[b], St[b], Wt[b]), Yt[b], Mt[b]), len(Xt), rede.epocas, rede.lr, seed, dev)
        m.eval()
        redes.append(m)
    return redes, P, S


def prever_rede(redes, P, S, D, L, o, lote=64):
    """→ (B, H, N, 9) em log, relativo à base (mediana das sementes)."""
    import torch
    dev = next(redes[0].parameters()).device
    T_ = lambda a, tp=torch.float32: torch.tensor(a, dtype=tp, device=dev)    # noqa: E731
    Xo = np.stack([P.X[x - L + 1:x + 1] for x in o]).astype(np.float32)
    res = []
    with torch.no_grad():
        for m in redes:
            res.append(np.concatenate([m(T_(Xo[i:i + lote]), T_(S[o[i:i + lote]]),
                                         T_(D.woyc[o[i:i + lote]], torch.long)).cpu().numpy()
                                       for i in range(0, len(o), lote)]))       # (B, N, H, 9)
    return np.sort(np.median(np.stack(res), 0), -1).transpose(0, 2, 1, 3)


class RedeSemanal(ModeloSemanal):
    """rede de grafo do uf_lib prevendo 1–4 semanas. H_treino=4: especializada no curto prazo; H_treino=68: a MESMA
    configuração da rede de temporada (a 'extrapolação' do modelo de temporada para a semana seguinte).
    cov: None (entradas do V4) | 'todas' | 'sel' (grupos de covariáveis somados às variáveis de origem)."""
    familia, usa_gpu = 'rede de grafos', True

    def __init__(self, rede, H_treino=4, alvo='ancora', cov=None, arq_sel=None, nome=None):
        self.rede, self.H_treino, self.alvo, self.cov, self.arq_sel = rede, H_treino, alvo, cov, arq_sel
        base = rede.nome.replace('_UF', '')
        self.nome = nome or (f'{base}' + (f'_{alvo}' if alvo != 'ancora' else '') + (f'_{cov}' if cov else '')
                             + (f'_H{H_treino}' if H_treino != 4 else '') + '_sem')

    def prever_semanal(self, ctx, R):
        D, c = ctx.Du, R['c']
        cov = cov_padronizada(ctx, c, self.cov, self.arq_sel) if self.cov else None
        redes, P, S = treinar_rede(self.rede, D, c, self.H_treino, self.alvo, cov)
        o = origens_de(R)
        r = prever_rede(redes, P, S, D, self.rede.L, o)[:, :H_SEM]
        lq = _base(P, D, o, H_SEM, self.alvo)[..., None] + r
        lq = np.minimum(lq, _teto(D, c)[None, None, :, None])
        return o, _para_casos(lq, D, o)


class MTGNNv2(Modelo):
    """PRÓXIMO PASSO DA TEMPORADA (roda pelo L.rodar, como os CompUF): rede de grafo de temporada (H = 68) com
    alvo 'cresc' (relativo ao nível da SE25), 'misto' (âncora + anomalia da SE25) ou 'ancora' (o original), e
    perda pinball opcionalmente ponderada pela população^gama (os estados grandes pesam mais)."""
    familia, origem, usa_gpu = 'rede de grafos', 'MTGNN v2 (próximos passos)', True

    def __init__(self, alvo='cresc', peso_pop=None, rede=None, nome=None):
        self.rede = rede or U.MTGNN()
        self.alvo, self.peso_pop = alvo, peso_pop
        base = self.rede.nome.replace('_UF', '')
        self.nome = nome or f'{base}' + (f'_{alvo}' if alvo != 'ancora' else '') + ('_pesopop' if peso_pop else '') + '_UF'

    def prever(self, ctx, s):
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        H = 68
        peso = None
        if self.peso_pop:
            peso = D.POP[c] ** self.peso_pop
            peso = peso / peso.mean()
        redes, P, S = treinar_rede(self.rede, D, c, H, self.alvo, peso=peso)
        o = np.array([c])
        lq = (_base(P, D, o, H, self.alvo)[..., None] + prever_rede(redes, P, S, D, self.rede.L, o))[0]   # (H, N, 9)
        lq = np.minimum(lq, _teto(D, c)[None, :, None])[h - 1]
        return Saida(quant=CM._para_contagem(lq, CM._pop_c(D, s)[None, :, None]))


# =====================================================================================================================
# neuralforecast (validação cruzada com refit=False: treina uma vez antes da 1ª origem) e Chronos-2 (zero-shot)
# =====================================================================================================================
class NFSemanal(ModeloSemanal):
    familia, usa_gpu = 'deep learning', True

    def __init__(self, arq='NHITS', cov=None, arq_sel=None, input_size=104, max_steps=1000, anos=12,
                 sementes=(1, 2, 3), nome=None):
        self.arq, self.cov, self.arq_sel = arq, cov, arq_sel
        self.input_size, self.max_steps, self.anos, self.sementes = input_size, max_steps, anos, tuple(sementes)
        self.nome = nome or f'{arq}_{cov or "semcov"}_sem'

    def prever_semanal(self, ctx, R):
        import logging
        import neuralforecast.models as nfm
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        for lg in ('pytorch_lightning', 'lightning.pytorch', 'lightning_fabric'):
            logging.getLogger(lg).setLevel(logging.ERROR)
        D, H, c = ctx.Du, self.H, R['c']
        o0, E = R['t_ini'] - H, R['t_fim'] + H
        ini = max(0, o0 - self.anos * 52)
        y = pd.DataFrame(D.LINC[ini:E + 1]).ffill().fillna(0).values      # depois do último dado: só preenche (não treina)
        ds = pd.to_datetime(D.datas[ini:E + 1])
        ids = D.unidades.astype(str)
        df = pd.DataFrame({'unique_id': np.repeat(ids, len(ds)), 'ds': np.tile(ds, D.N), 'y': y.T.ravel()})
        df = CM._add_fourier(df, 2)
        hist = []
        cv = cov_padronizada(ctx, c, self.cov, self.arq_sel) if self.cov else None
        if cv is not None:
            for j in range(cv.shape[-1]):
                df[f'x{j}'] = cv[ini:E + 1, :, j].T.ravel()
                hist.append(f'x{j}')
        n_win = (E - H) - o0 + 1
        res = []
        for seed in self.sementes:
            kw = dict(h=H, input_size=self.input_size, max_steps=self.max_steps, random_seed=seed, scaler_type='robust',
                      loss=MQLoss(level=NIVEIS), futr_exog_list=U.FOURIER_NF, val_check_steps=50,
                      early_stop_patience_steps=5)
            if hist:
                kw['hist_exog_list'] = hist
            if self.arq == 'TFT':
                kw['hidden_size'] = 64
            m = getattr(nfm, self.arq)(**kw)
            nf = NeuralForecast(models=[m], freq='W-SUN')
            cvr = nf.cross_validation(df=df, n_windows=n_win, step_size=1, refit=False, val_size=52)
            cvr = cvr.reset_index() if 'unique_id' not in cvr.columns else cvr
            nm = type(m).__name__
            if f'{nm}-median' in cvr.columns:
                cvr = cvr.drop(columns=[nm], errors='ignore').rename(columns={f'{nm}-median': nm})
            cols = [f'{nm}-lo-{n}' for n in NIVEIS[::-1]] + [nm] + [f'{nm}-hi-{n}' for n in NIVEIS]
            cvr['o'] = cvr['cutoff'].map(lambda d: D.idx_data[pd.Timestamp(d)])
            cvr['k'] = ((pd.to_datetime(cvr['ds']) - pd.to_datetime(cvr['cutoff'])).dt.days // 7).astype(int)
            cvr['j'] = cvr['unique_id'].map({u: i for i, u in enumerate(ids)})
            arr = np.full((n_win, H, D.N, 9), np.nan)
            arr[(cvr.o - o0).values, (cvr.k - 1).values, cvr.j.values] = cvr[cols].values
            res.append(arr)
        o = np.arange(o0, o0 + n_win)
        lq = np.sort(np.nanmedian(np.stack(res), 0), -1)
        keep = o < R['t_fim']
        lq = np.minimum(lq[keep], _teto(D, c)[None, None, :, None])
        return o[keep], _para_casos(np.nan_to_num(lq), D, o[keep])


class Chronos2Semanal(ModeloSemanal):
    """Chronos-2 zero-shot: em cada origem, contexto = log1p(inc) das 26 UFs até a origem; prevê 1–4 semanas."""
    familia, usa_gpu, nome = 'fundação (zero-shot)', True, 'Chronos2_sem'

    def __init__(self, contexto=520, modelo_hf='amazon/chronos-2'):
        self.contexto, self.modelo_hf, self._pipe = contexto, modelo_hf, None

    def prever_semanal(self, ctx, R):
        import torch
        from chronos import Chronos2Pipeline
        if self._pipe is None:
            self._pipe = Chronos2Pipeline.from_pretrained(self.modelo_hf,
                                                          device_map='cuda' if torch.cuda.is_available() else 'cpu')
        D, H, c = ctx.Du, self.H, R['c']
        ids = D.unidades.astype(str)
        o = origens_de(R)
        lq = np.zeros((len(o), H, D.N, 9))
        niveis = [float(x) for x in np.round(Q, 3)]
        for b, oo in enumerate(o):
            ini = max(0, oo + 1 - self.contexto)
            ds = pd.to_datetime(D.datas[ini:oo + 1])
            cx = pd.DataFrame({'id': np.repeat(ids, len(ds)), 'timestamp': np.tile(ds, D.N),
                               'target': np.nan_to_num(D.LINC[ini:oo + 1]).T.ravel()})
            try:
                pr = self._pipe.predict_df(cx, prediction_length=H, quantile_levels=niveis, id_column='id',
                                           timestamp_column='timestamp', target='target')
                nv = niveis
            except Exception:
                nv = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
                pr = self._pipe.predict_df(cx, prediction_length=H, quantile_levels=nv, id_column='id',
                                           timestamp_column='timestamp', target='target')
            pr = pr.reset_index() if 'id' not in pr.columns else pr
            qc = {}
            for col in pr.columns:
                try:
                    qc[round(float(col), 3)] = col
                except (TypeError, ValueError):
                    pass
            pr['_u'] = pr['id'].astype(str).map({k: i for i, k in enumerate(ids)})
            pr = pr.sort_values(['_u', 'timestamp'])
            arr = np.stack([pr[qc[round(n, 3)]].values.reshape(D.N, H) for n in nv], -1)        # (N, H, k)
            arr = U._quantis_de_niveis(nv, arr) if len(nv) != 9 else np.sort(arr, -1)
            lq[b] = np.moveaxis(arr, 0, 1)
        lq = np.minimum(lq, _teto(D, c)[None, None, :, None])
        return o, _para_casos(lq, D, o)


# =====================================================================================================================
# avaliação (as mesmas métricas nos dois níveis)
# =====================================================================================================================
def ler_semanal(pastas):
    fs = [f for p in pastas for f in glob.glob(os.path.join(p, '*.parquet'))]
    if not fs:
        return pd.DataFrame(columns=COLS_SEM)
    P = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    return P.drop_duplicates(['modelo', 'rodada', 'h', 'date', 'uf_code'], keep='last')


def com_verdade(P, D):
    """acrescenta y (casos observados); tira semanas sem dado."""
    y = pd.DataFrame(D.Y[:D.ultimo_obs + 1], index=pd.to_datetime(D.datas[:D.ultimo_obs + 1]), columns=D.unidades)
    y = y.stack().rename('y').reset_index().rename(columns={'level_0': 'date', 'level_1': 'uf_code'})
    P = P.copy()
    P['date'] = pd.to_datetime(P['date'])
    return P.merge(y, on=['date', 'uf_code']).dropna(subset=['y'])


def metricas(df):
    q, y = df[QCOLS].values.astype(float), df['y'].values.astype(float)
    p = q[:, 4]
    e = p - y
    den = np.abs(y) + np.abs(p)
    s = np.where(den > 0, 2 * np.abs(e) / np.where(den > 0, den, 1), 0)
    out = dict(n=len(df), WIS=float(V.wis(q, y).mean()), WIS_log=float(V.wis(np.log1p(q), np.log1p(y)).mean()),
               MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())),
               WAPE=float(100 * np.abs(e).sum() / max(y.sum(), 1)), sMAPE=float(100 * s.mean()),
               sMAPE_0a100=float(50 * s.mean()), sMAPE_mediana=float(100 * np.median(s)),
               r=float(np.corrcoef(p, y)[0, 1]) if len(y) > 2 and p.std() > 0 else np.nan,
               vies_pct=float(100 * e.sum() / max(y.sum(), 1)))
    for i, n in enumerate(NIVEIS):
        out[f'IC{n}'] = float(100 * np.mean((y >= q[:, 3 - i]) & (y <= q[:, 5 + i])))
    return out


def razao(P, ref, chave_rodada='rodada', extra=('h',)):
    """WIS relativo à referência, como no 3º IMDC: média geométrica nas rodadas (por UF) e depois nas UFs."""
    extra = list(extra)
    w = P.assign(WIS=V.wis(P[QCOLS].values.astype(float), P['y'].values.astype(float)))
    w = w.groupby(['modelo', chave_rodada, 'uf_code'] + extra).WIS.mean().reset_index()
    b = w[w.modelo == ref].drop(columns='modelo').rename(columns={'WIS': 'WIS_ref'})
    m = w.merge(b, on=[chave_rodada, 'uf_code'] + extra)
    m = m[(m.WIS_ref > 0) & (m.WIS > 0)]
    m['lr'] = np.log(m.WIS / m.WIS_ref)
    geo = m.groupby(['modelo', 'uf_code'] + extra).lr.mean()
    g = geo.groupby(['modelo'] + extra)
    return pd.DataFrame({f'R_vs_{ref}': np.exp(g.mean()), f'UFs_melhor_que_{ref}': (geo < 0).groupby(['modelo'] + extra).sum()})


def resumo_semanal(P, grupo, refs=REFS_SEM):
    """P já com y. Uma linha por (modelo, h) no grupo de rodadas ('afinacao', 'teste' ou 'felipe')."""
    X = P[P.rodada.map(grupo_rodada) == grupo]
    if X.empty:
        return pd.DataFrame()
    R = pd.DataFrame([dict(modelo=m, h=h, rodadas=g.rodada.nunique(), **metricas(g)) for (m, h), g in X.groupby(['modelo', 'h'])])
    for ref in refs:
        if (X.modelo == ref).any():
            R = R.merge(razao(X, ref).reset_index(), on=['modelo', 'h'], how='left')
    col = f'R_vs_{refs[0]}' if f'R_vs_{refs[0]}' in R.columns else 'WIS'
    return R.sort_values(['h', col]).reset_index(drop=True)


def ensemble_semanal(P, membros, nome, metodo='mediana'):
    """mediana (ou média) dos quantis, só nas linhas em que TODOS os membros existem."""
    chave = ['rodada', 'season', 'h', 'origem', 'date', 'uf_code']
    sub = P[P.modelo.isin(membros)]
    if sub.modelo.nunique() < len(membros):
        return P.iloc[:0]
    g = sub.groupby(chave)[QCOLS]
    a = (g.median() if metodo == 'mediana' else g.mean())[g.size() == len(membros)]
    a = pd.DataFrame(np.sort(a.values, 1), index=a.index, columns=QCOLS).reset_index()
    return a.assign(modelo=nome)


def calibrar_lambda(P, modelos, ref='Climatologia_sem', grade=(0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5)):
    """λ (largura dos intervalos) escolhido SÓ na afinação, pela razão de WIS contra a referência (média em h)."""
    A = P[P.rodada.map(grupo_rodada) == 'afinacao']
    rf = A[A.modelo == ref]
    lam = {}
    for m in modelos:
        g = A[A.modelo == m]
        if g.empty or rf.empty:
            continue
        rs = []
        for l in grade:
            x = U.ajusta_largura(g, l).assign(modelo='_x')
            rs.append(float(razao(pd.concat([x, rf]), ref).loc['_x'].iloc[:, 0].mean()))
        lam[m] = grade[int(np.argmin(rs))]
    return lam


def ensemble_por_porte(TODOS, candidatos, ver, D, temporadas_escolha, n_grupos=3, nome='ENS_porte'):
    """temporada: para cada grupo de UFs por tamanho (população), escolhe na AFINAÇÃO o candidato com menor WIS
    relativo à climatologia dentro do grupo; o ensemble usa, em cada UF, o candidato do seu grupo."""
    pop = D.POP[D.ultimo_obs]
    corte = np.quantile(pop, np.linspace(0, 1, n_grupos + 1)[1:-1])
    porte = dict(zip(D.unidades, np.digitize(pop, corte)))                     # 0 = pequenos … n−1 = grandes
    sub = TODOS[TODOS.modelo.isin(list(candidatos) + ['climatologia']) & TODOS.season.isin(temporadas_escolha)]
    por, _ = V.avaliar(sub, ver, ref=('climatologia',))
    b = por[por.modelo == 'climatologia'][['season', 'uf_code', 'WIS']].rename(columns={'WIS': 'W0'})
    m = por.merge(b, on=['season', 'uf_code'])
    m['lr'] = np.log(m.WIS / m.W0)
    m['porte'] = m.uf_code.map(porte)
    tab = m[m.modelo.isin(candidatos)].groupby(['porte', 'modelo']).lr.mean().unstack()
    escolha = tab.idxmin(1).to_dict()
    partes = [TODOS[(TODOS.modelo == escolha[k]) & TODOS.uf_code.map(porte).eq(k)] for k in escolha]
    out = pd.concat(partes, ignore_index=True).assign(modelo=nome)
    # só temporadas em que todas as UFs ficaram cobertas
    ok = out.groupby('season').uf_code.nunique()
    out = out[out.season.isin(ok[ok == D.N].index)]
    return out, np.exp(tab).round(3), escolha
