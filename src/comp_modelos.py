"""
Competição de modelos · os competidores.

Cada classe é uma REIMPLEMENTAÇÃO da ideia de uma equipe do IMDC ou de um artigo, rodando nos nossos dados por
região de saúde (435 regiões). Todas seguem a regra da SE25 e devolvem previsões por região (ver comp_lib.Saida).

  #  nome                 família              inspiração
  01 climatologia         referência           PURDUE (componente) / régua universal
  02 canal_endemico       referência           Cornell (endemic channel)
  03 BB_INLA              bayesiano            PROCC — modelo-base oficial (NB + semana e ano por macrorregião, R-INLA)
  04 AR_longo             estatístico          LNCC ARP26 (AR(92) em log2)
  05 SURGE                estatístico          LNCC SURGE (surto típico alinhado × ganho log-normal)
  06 ARIMA_Fourier        estatístico          EMAP SARIMAX (ARIMA + termos de Fourier)
  07 Prophet              estatístico          EMAP Prophet
  08 ETS_MSTL             estatístico          PUC-Rio (ETS) com decomposição sazonal MSTL
  09 GHR_INLA             bayesiano            BSC/GHR (Lowe): NB + BYM2 + RW cíclico + ano + clima (R-INLA)
  10 LightGBM_direto      aprendizado de máq.  PURDUE (LightGBM + resíduos conformais por horizonte, origens sintéticas)
  11 XGBoost_quantis      aprendizado de máq.  EMAP XGB / RKI (XGBoost multi-quantil)
  12 DeepAR               deep learning        Dengue Oracle / GRU do PURDUE (RNN probabilística)
  13 NHITS, TFT, TSMixer  deep learning        Imperial (TFT), RKI (TSMixer), N-HiTS
  14 AGCRN                rede de grafos       literatura de GNN (nossa versão A)
  15 Renovacao_S          mecanístico          NUS / IFGW (renovação com depleção de suscetíveis)
  16 Tamanho_x_forma      híbrido              IFGW outbreak features (nossa versão B)
  17 Chronos              fundação (zero-shot) RKI / NUS (Chronos, TimesFM)
"""
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

import v4_lib as V
from comp_lib import Modelo, Saida, ZQ

Q = V.Q
NIVEIS_INT = [50, 80, 90, 95]


def _pop_c(D, s):
    return D.POP[D.corte(s)]


def _para_contagem(lq, pop):
    """log1p(incidência por 100 mil) → casos."""
    return np.clip(np.expm1(lq), 0, None) * pop / 1e5


def _quantis_de_intervalos(med, lo, hi):
    """med (W,N); lo/hi: dict nível→(W,N) → (W,N,9) na ordem de QCOLS."""
    return np.stack([lo[95], lo[90], lo[80], lo[50], med, hi[50], hi[80], hi[90], hi[95]], -1)


def _horizontes(ctx, s):
    D = ctx.Dr
    c = D.corte(s)
    datas, idx = ctx.alvo(s)
    return c, datas, idx, idx - c                     # h = 1 .. H


# =====================================================================================================================
# 01–02 referências
# =====================================================================================================================
class Climatologia(Modelo):
    nome, familia, origem = 'climatologia', 'referência', 'PURDUE (componente)'

    def prever(self, ctx, s):
        return Saida(quant=V.base_climatologia(ctx.Dr, s)[1])


class CanalEndemico(Modelo):
    nome, familia, origem = 'canal_endemico', 'referência', 'Cornell'

    def prever(self, ctx, s):
        return Saida(quant=V.base_canal_endemico(ctx.Dr, s)[1])


# =====================================================================================================================
# 04–05 LNCC
# =====================================================================================================================
class ARLongo(Modelo):
    nome, familia, origem = 'AR_longo', 'estatístico', 'LNCC ARP26'

    def __init__(self, p=92, M=500, ridge=1.0):
        self.p, self.M, self.ridge = p, M, ridge

    def prever(self, ctx, s):
        D = ctx.Dr
        rng = np.random.default_rng(s)
        c, datas, idx, h = _horizontes(ctx, s)
        fim = idx[-1]
        p = self.p
        traj = np.zeros((self.M, len(idx), D.N), dtype=np.float32)
        for u in range(D.N):
            z = np.log2(np.nan_to_num(D.Y[:c + 1, u]) + 1)
            mz = z.mean()
            z = z - mz
            X = np.column_stack([z[p - k - 1:len(z) - k - 1] for k in range(p)])
            yv = z[p:]
            coef = np.linalg.solve(X.T @ X + self.ridge * np.eye(p), X.T @ yv)
            sd = np.std(yv - X @ coef)
            tr = np.zeros((self.M, fim + 1))
            tr[:, :c + 1] = z
            teto0 = np.log2(3 * np.nanmax(D.Y[:c + 1, u]) + 1) - mz
            for t in range(c + 1, fim + 1):
                tr[:, t] = np.minimum(tr[:, t - p:t][:, ::-1] @ coef + rng.normal(0, sd, self.M), teto0)
            teto = np.log2(3 * np.nanmax(D.Y[:c + 1, u]) + 1) - mz        # saneamento: ≤ 3× o maior valor semanal já visto
            traj[..., u] = np.clip(2 ** (np.minimum(tr[:, idx], teto) + mz) - 1, 0, None)
        return Saida(traj=traj)


class Surge(Modelo):
    """curva típica: temporadas passadas suavizadas, alinhadas pelo pico na semana mediana de pico da região;
    ganho de cada temporada = projeção na curva típica; ganho futuro ~ log-normal ajustada aos ganhos."""
    nome, familia, origem = 'SURGE', 'estatístico', 'LNCC SURGE'

    def __init__(self, M=500, janela=5):
        self.M, self.janela = M, janela

    def prever(self, ctx, s):
        D = ctx.Dr
        rng = np.random.default_rng(s)
        c, datas, idx, h = _horizontes(ctx, s)
        W = len(idx)
        temps = [j for j in range(2010, s) if D.temporada_completa(j, c) and D.idx_temporada(j)[0] > 0]
        k = np.ones(self.janela) / self.janela
        C = []
        for j in temps:
            y = D.Y[D.idx_temporada(j)][:52]
            C.append(np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 0, y))
        C = np.stack(C)                                   # (J, 52, N)
        picos = C.argmax(1)                               # (J, N)
        alvo_pico = np.median(picos, 0).round().astype(int)
        al = np.stack([np.stack([np.roll(C[j, :, u], alvo_pico[u] - picos[j, u]) for u in range(D.N)], 1)
                       for j in range(len(temps))])       # (J, 52, N)
        tip = al.mean(0)
        tip = tip / np.maximum(tip.sum(0), 1e-9)          # forma com soma 1
        g = (al * tip[None]).sum(1) / np.maximum((tip ** 2).sum(0), 1e-12)   # ganho = total equivalente
        lg = np.log(np.maximum(g, 1.0))
        mu, sd = lg.mean(0), np.maximum(lg.std(0), 0.2)
        gan = np.exp(mu[None] + sd[None] * rng.standard_normal((self.M, D.N)))
        forma = np.vstack([tip, np.repeat(tip[-1:], max(0, W - 52), 0)])[:W]
        lam = gan[:, None, :] * forma[None]
        return Saida(traj=rng.poisson(lam).astype(np.float32))


# =====================================================================================================================
# 06–08 estatísticos clássicos por série (statsforecast / Prophet) em log1p(incidência)
# =====================================================================================================================
def _df_series(D, c, anos=8, fourier=0):
    ini = max(0, c - anos * 52)
    ds = pd.to_datetime(D.datas[ini:c + 1])
    y = D.LINC[ini:c + 1]
    df = pd.DataFrame({'unique_id': np.repeat(D.unidades.astype(str), len(ds)),
                       'ds': np.tile(ds, D.N), 'y': y.T.ravel()})
    if fourier:
        df = _add_fourier(df, fourier)
    return df


def _add_fourier(df, K):
    w = np.array([V.data_para_epiweek(d) % 100 for d in df.ds.dt.date.unique()])
    mapa = dict(zip(df.ds.unique(), np.minimum(w, 52)))
    a = 2 * np.pi * df.ds.map(mapa).values / 52
    for k in range(1, K + 1):
        df[f's{k}'], df[f'c{k}'] = np.sin(k * a), np.cos(k * a)
    return df


def _futuro(D, c, H, K):
    ds = pd.date_range(pd.Timestamp(D.datas[c]) + pd.Timedelta(weeks=1), periods=H, freq='7D')
    f = pd.DataFrame({'unique_id': np.repeat(D.unidades.astype(str), H), 'ds': np.tile(ds, D.N)})
    return _add_fourier(f, K) if K else f


def _sf_para_quantis(fc, nome, D, idx_h, pop):
    """saída do statsforecast (colunas nome, nome-lo-95...) → (W, N, 9) em casos."""
    fc = fc.reset_index() if 'unique_id' not in fc.columns else fc
    fc['unique_id'] = fc['unique_id'].astype(str)
    ordem = {u: i for i, u in enumerate(D.unidades.astype(str))}
    fc['_u'] = fc.unique_id.map(ordem)
    fc = fc.sort_values(['_u', 'ds'])
    H = fc.groupby('_u').size().iloc[0]

    def m(col):
        return fc[col].values.reshape(D.N, H).T[idx_h - 1]
    med = m(nome)
    lo = {n: m(f'{nome}-lo-{n}') for n in NIVEIS_INT}
    hi = {n: m(f'{nome}-hi-{n}') for n in NIVEIS_INT}
    return _para_contagem(_quantis_de_intervalos(med, lo, hi), pop[None, :, None])


class ARIMAFourier(Modelo):
    nome, familia, origem = 'ARIMA_Fourier', 'estatístico', 'EMAP SARIMAX'

    def __init__(self, K=3, anos=8):
        self.K, self.anos = K, anos

    def prever(self, ctx, s):
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA
        D = ctx.Dr
        c, datas, idx, h = _horizontes(ctx, s)
        H = int(h.max())
        df = _df_series(D, c, self.anos, self.K)
        sf = StatsForecast(models=[AutoARIMA(season_length=1, d=0, allowdrift=False, max_p=3, max_q=3)],   # sem tendência: a
                           freq='W-SUN', n_jobs=-1)                                                # sazonalidade vem do Fourier
        fc = sf.forecast(df=df, h=H, X_df=_futuro(D, c, H, self.K), level=NIVEIS_INT)
        return Saida(quant=_sf_para_quantis(fc, 'AutoARIMA', D, h, _pop_c(D, s)))


class ETSMSTL(Modelo):
    nome, familia, origem = 'ETS_MSTL', 'estatístico', 'PUC-Rio (ETS)'

    def __init__(self, anos=8):
        self.anos = anos

    def prever(self, ctx, s):
        from statsforecast import StatsForecast
        from statsforecast.models import MSTL, AutoETS
        D = ctx.Dr
        c, datas, idx, h = _horizontes(ctx, s)
        H = int(h.max())
        df = _df_series(D, c, self.anos)
        sf = StatsForecast(models=[MSTL(season_length=52, trend_forecaster=AutoETS(model='ZNN'))],
                           freq='W-SUN', n_jobs=-1)
        fc = sf.forecast(df=df, h=H, level=NIVEIS_INT)
        return Saida(quant=_sf_para_quantis(fc, 'MSTL', D, h, _pop_c(D, s)))


def _prophet_uma(y, ds, H, amostras, seed):
    import logging
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    logging.getLogger('prophet').setLevel(logging.ERROR)
    from prophet import Prophet
    np.random.seed(seed)
    m = Prophet(yearly_seasonality=8, weekly_seasonality=False, daily_seasonality=False,
                changepoint_prior_scale=0.05, uncertainty_samples=amostras)
    m.fit(pd.DataFrame({'ds': ds, 'y': y}))
    fut = pd.DataFrame({'ds': pd.date_range(ds[-1] + pd.Timedelta(weeks=1), periods=H, freq='7D')})
    return m.predictive_samples(fut)['yhat']              # (H, amostras)


class ProphetModelo(Modelo):
    nome, familia, origem = 'Prophet', 'estatístico', 'EMAP Prophet'

    def __init__(self, anos=8, amostras=300, n_jobs=-1):
        self.anos, self.amostras, self.n_jobs = anos, amostras, n_jobs

    def prever(self, ctx, s):
        from joblib import Parallel, delayed
        D = ctx.Dr
        c, datas, idx, h = _horizontes(ctx, s)
        H = int(h.max())
        ini = max(0, c - self.anos * 52)
        ds = pd.to_datetime(D.datas[ini:c + 1])
        res = Parallel(n_jobs=self.n_jobs)(delayed(_prophet_uma)(D.LINC[ini:c + 1, u], ds, H, self.amostras, u)
                                           for u in range(D.N))
        a = np.stack(res, -1)[h - 1]                      # (W, amostras, N)
        lq = np.moveaxis(np.quantile(a, Q, axis=1), 0, -1)
        return Saida(quant=_para_contagem(lq, _pop_c(D, s)[None, :, None]))


# =====================================================================================================================
# 03 e 09 · R-INLA (BB/PROCC e GHR/BSC)
# =====================================================================================================================
def _covariaveis_ghr(D, c, idx_alvo):
    """anomalia padronizada de temperatura e de chuva, média das semanas t-12..t-4 (defasagem 1–3 meses).
    Depois do corte: anomalia prevista pela SEAS5 de junho enquanto a defasagem cair dentro de 6 meses; depois 0."""
    an = V._anomalias_clima(D, c)
    seas = V._seas5_anomalias(D, pd.Timestamp(D.datas[c]).year, pd.Timestamp(D.datas[c]).year)
    out = {}
    for v, sv in (('temp_med', 'seas_temp_med_anom'), ('precip_med', 'seas_precip_tot_anom')):
        a = np.nan_to_num(an[v]).copy()
        a[c + 1:] = np.nan
        fut = np.nan_to_num(seas[sv])
        T = D.T
        for t in range(c + 1, T):
            a[t] = fut if (t - 8) <= c + 26 else 0.0
        m = pd.DataFrame(a).rolling(9, min_periods=1).mean().shift(4).values
        out[v] = np.nan_to_num(m)
    return out['temp_med'], out['precip_med']


class ModeloINLA(Modelo):
    familia = 'bayesiano (R-INLA)'
    dependencia = 'propria'

    def __init__(self, tipo, anos=6, M=300, threads=8, rscript=None, r_libs=None, pasta_script=None):
        self.tipo, self.anos, self.M, self.threads = tipo, anos, M, threads
        self.nome = {'bb': 'BB_INLA', 'ghr': 'GHR_INLA', 'ghr_nl': 'GHR_NL_INLA'}[tipo]
        self.origem = {'bb': 'PROCC (modelo-base oficial)', 'ghr': 'BSC/GHR (Lowe)',
                       'ghr_nl': 'BSC/GHR com efeitos não lineares do clima (RW1 em decis) + ENSO'}[tipo]
        self.rscript = rscript or os.environ.get('RSCRIPT', 'Rscript')
        self.r_libs = r_libs
        self.pasta_script = pasta_script or os.path.dirname(os.path.abspath(__file__))

    def prever(self, ctx, s):
        D = ctx.Dr
        c, datas, idx, h = _horizontes(ctx, s)
        ini = max(D.idx_temporada(s - self.anos)[0], 0)
        t_tr = np.arange(ini, c + 1)
        t_all = np.concatenate([t_tr, idx])
        n_tr = len(t_tr)
        season = D.season[t_all]
        reg = np.arange(D.N) + 1
        uf_id = np.searchsorted(D.ufs, D.uf_of) + 1
        macro_br = D.uf_of // 10                          # 5 grandes regiões (N, NE, SE, S, CO), como no BB original
        macros = np.unique(macro_br)
        macro_id = np.searchsorted(macros, macro_br) + 1
        s_id = season - season.min()
        df = pd.DataFrame({
            't': np.repeat(t_all, D.N),
            'reg_id': np.tile(reg, len(t_all)),
            'uf_id': np.tile(uf_id, len(t_all)),
            'macro_id': np.tile(macro_id, len(t_all)),
            'woy': np.repeat(D.woyc[t_all], D.N),
            'y': np.concatenate([D.Y[t_tr].ravel(), np.full(len(idx) * D.N, np.nan)]),
            'log_E': np.log(D.POP[np.minimum(t_all, D.T - 1)] / 1e5).ravel(),
        })
        df['ano_macro'] = np.repeat(s_id, D.N) * len(macros) + df.macro_id
        df['ano_uf'] = np.repeat(s_id, D.N) * len(D.ufs) + df.uf_id
        if self.tipo in ('ghr', 'ghr_nl'):
            ta, pa = _covariaveis_ghr(D, c, idx)
            df['tanom'] = ta[t_all].ravel()
            df['panom'] = pa[t_all].ravel()
        if self.tipo == 'ghr_nl':
            # efeito não linear: decis das anomalias (cortes só com o treino) como níveis de um RW1
            for v, nb in (('tanom', 'tbin'), ('panom', 'pbin')):
                cortes = np.quantile(df[v].values[:n_tr * D.N], np.linspace(0.1, 0.9, 9))
                df[nb] = np.searchsorted(cortes, df[v].values) + 1                     # 1..10
            en = np.nan_to_num(pd.Series(D.OCEANO['enso']).ffill().values)
            en[c + 1:] = en[c]                                                          # persistência após a SE25
            df['enso'] = np.repeat(en[t_all], D.N)
        tmp = tempfile.mkdtemp(prefix=f'inla_{self.tipo}_{s}_')
        try:
            df.to_csv(os.path.join(tmp, 'dados.csv'), index=False, na_rep='NA')
            with open(os.path.join(tmp, 'grafo.adj'), 'w') as f:
                f.write(f'{D.N}\n')
                for i in range(D.N):
                    viz = np.where(D.ADJ[i] > 0)[0] + 1
                    f.write(f'{i + 1} {len(viz)} ' + ' '.join(map(str, viz)) + '\n')
            with open(os.path.join(tmp, 'params.txt'), 'w') as f:
                f.write(f'modelo={self.tipo}\nn_amostras={self.M}\nthreads={self.threads}\n')
            env = dict(os.environ)
            if self.r_libs:
                env['R_LIBS_USER'] = self.r_libs
            t0 = time.time()
            p = subprocess.run([self.rscript, os.path.join(self.pasta_script, 'comp_inla.R'), tmp],
                               capture_output=True, text=True, env=env)
            tempo = open(os.path.join(tmp, 'tempo.txt')).read() if os.path.exists(os.path.join(tmp, 'tempo.txt')) else ''
            if p.returncode != 0 or not os.path.exists(os.path.join(tmp, 'amostras.bin')):
                raise RuntimeError(f'INLA falhou (pasta {tmp}):\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}')
            with open(os.path.join(tmp, 'amostras.bin'), 'rb') as f:
                n, M = np.fromfile(f, dtype=np.int32, count=2)
                a = np.fromfile(f, dtype=np.float32).reshape(M, n)
            hip = pd.read_csv(os.path.join(tmp, 'hiper.csv'))
        finally:
            if not os.environ.get('INLA_MANTER_TMP'):
                shutil.rmtree(tmp, ignore_errors=True)
        traj = a.reshape(M, len(idx), D.N)                 # linhas do R na ordem (t, região)
        return Saida(traj=traj, info={'segundos_inla': time.time() - t0, 'linhas_treino': n_tr * D.N, 'tempo_R': tempo,
                                      'hiper': hip.iloc[:, :3].to_dict('records')})


# =====================================================================================================================
# 10–11 · árvores diretas multi-horizonte com origens sintéticas
# =====================================================================================================================
def painel_direto(ctx, s, passo_o=3, passo_h=2, H=68):
    """linhas (origem o, horizonte h, região): variáveis da origem + do alvo; alvo = log1p(inc) − âncora sazonal."""
    D = ctx.Dr
    c = D.corte(s)
    chave = ('painel', s, passo_o, passo_h)
    if chave in getattr(ctx, 'cache', {}):
        return ctx.cache[chave]
    P = V.PreparaA(D, c, L=104, H=H)
    linc = np.nan_to_num(D.LINC)

    def bloco(o, hs, lim):
        F, _ = P.futuro(o)
        t = o + hs
        ok = t <= lim
        hs, t = hs[ok], t[ok]
        anc = P.CM[P.ult_woy[o, D.woyc[t]]]                            # (len(hs), N)
        X = np.concatenate([np.repeat(P.S[o][None], len(hs), 0), F[hs - 1]], -1)   # (len(hs), N, 15)
        y = linc[np.minimum(t, D.T - 1)] - anc
        hh = np.repeat(hs[:, None], D.N, 1)
        return X.reshape(-1, X.shape[-1]), y.ravel(), hh.ravel(), anc

    hs_all = np.arange(1, H + 1)
    blocos = [bloco(o, hs_all[::passo_h], c) for o in range(160, c, passo_o)]
    X = np.concatenate([b[0] for b in blocos]).astype(np.float32)
    y = np.concatenate([b[1] for b in blocos]).astype(np.float32)
    hh = np.concatenate([b[2] for b in blocos])
    oo = np.concatenate([np.full(len(b[1]), o) for b, o in zip(blocos, range(160, c, passo_o))])
    F, _ = P.futuro(c)
    Xp = np.concatenate([np.repeat(P.S[c][None], H, 0), F], -1)          # (H, N, 15)
    ancp = np.stack([P.CM[P.ult_woy[c, D.woyc[c + k]]] for k in range(1, H + 1)])
    out = dict(X=X, y=y, h=hh, o=oo, Xp=Xp, ancp=ancp, c=c, c_val=D.corte(s - 1))
    if not hasattr(ctx, 'cache'):
        ctx.cache = {}
    ctx.cache = {chave: out}                                            # guarda só a última (memória)
    return out


BINS_H = [(1, 15), (16, 30), (31, 45), (46, 68)]


class LightGBMDireto(Modelo):
    """Regressão quantílica direta (um LightGBM por quantil) sobre o painel de origens sintéticas (PURDUE)."""
    nome, familia, origem = 'LightGBM_direto', 'aprendizado de máquina', 'PURDUE (LightGBM quantílico)'

    def __init__(self, n_estimators=300, max_linhas=600_000):
        self.params = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, min_child_samples=300,
                           subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1)
        self.max_linhas = max_linhas

    def prever(self, ctx, s):
        import lightgbm as lgb
        D = ctx.Dr
        c, datas, idx, h = _horizontes(ctx, s)
        P = painel_direto(ctx, s)
        rng = np.random.default_rng(s)
        sel = rng.choice(len(P['y']), min(len(P['y']), self.max_linhas), replace=False)
        Xp = P['Xp'].reshape(-1, P['Xp'].shape[-1])
        pr = np.stack([lgb.LGBMRegressor(objective='quantile', alpha=float(a), **self.params)
                       .fit(P['X'][sel], P['y'][sel]).predict(Xp) for a in Q], -1)
        pr = np.sort(pr.reshape(*P['Xp'].shape[:2], len(Q)), -1)[h - 1]          # (W, N, 9)
        lq = P['ancp'][h - 1][..., None] + pr
        return Saida(quant=_para_contagem(lq, _pop_c(D, s)[None, :, None]))


class XGBoostQuantis(Modelo):
    nome, familia, origem = 'XGBoost_quantis', 'aprendizado de máquina', 'EMAP XGB / RKI'

    def __init__(self, n_estimators=250, max_linhas=600_000):
        self.n_estimators, self.max_linhas = n_estimators, max_linhas

    def prever(self, ctx, s):
        import xgboost as xgb
        import torch
        D = ctx.Dr
        c, datas, idx, h = _horizontes(ctx, s)
        P = painel_direto(ctx, s)
        rng = np.random.default_rng(s)
        sel = rng.choice(len(P['y']), min(len(P['y']), self.max_linhas), replace=False)
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=Q, tree_method='hist',
                             n_estimators=self.n_estimators, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, device='cuda' if torch.cuda.is_available() else 'cpu')
        m.fit(P['X'][sel], P['y'][sel])
        pr = m.predict(P['Xp'].reshape(-1, P['Xp'].shape[-1])).reshape(*P['Xp'].shape[:2], len(Q))[h - 1]
        lq = P['ancp'][h - 1][..., None] + np.sort(pr, -1)
        return Saida(quant=_para_contagem(lq, _pop_c(D, s)[None, :, None]))


# =====================================================================================================================
# 12–13 · deep learning (neuralforecast), alvo log1p(incidência), saída em quantis
# =====================================================================================================================
class NeuralForecastModelo(Modelo):
    familia = 'deep learning'
    usa_gpu = True

    def __init__(self, arquitetura='DeepAR', max_steps=1000, input_size=104, anos=12, seed=1):
        self.arq, self.max_steps, self.input_size, self.anos, self.seed = arquitetura, max_steps, input_size, anos, seed
        self.nome = arquitetura
        self.origem = {'DeepAR': 'Dengue Oracle / PURDUE GRU', 'NHITS': 'N-HiTS (literatura)',
                       'TFT': 'Imperial TFT', 'TSMixer': 'RKI TSMixer'}[arquitetura]

    def _modelo(self, H, N):
        from neuralforecast.models import DeepAR, NHITS, TFT, TSMixerx
        from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
        comum = dict(h=H, input_size=self.input_size, max_steps=self.max_steps, random_seed=self.seed,
                     futr_exog_list=['s1', 'c1'], val_check_steps=50, early_stop_patience_steps=6)
        if self.arq == 'DeepAR':
            return DeepAR(loss=DistributionLoss('Normal', level=NIVEIS_INT), valid_loss=MQLoss(level=NIVEIS_INT),
                          lstm_n_layers=2, lstm_hidden_size=64, scaler_type='standard', **comum)
        if self.arq == 'NHITS':
            return NHITS(loss=MQLoss(level=NIVEIS_INT), scaler_type='robust', **comum)
        if self.arq == 'TFT':
            return TFT(loss=MQLoss(level=NIVEIS_INT), hidden_size=64, scaler_type='robust', **comum)
        if self.arq == 'TSMixer':
            return TSMixerx(n_series=N, loss=MQLoss(level=NIVEIS_INT), scaler_type='robust', **comum)
        raise ValueError(self.arq)

    def prever(self, ctx, s):
        from neuralforecast import NeuralForecast
        import logging
        logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
        logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
        D = ctx.Dr
        c, datas, idx, h = _horizontes(ctx, s)
        H = 68
        df = _df_series(D, c, self.anos, fourier=1)[['unique_id', 'ds', 'y', 's1', 'c1']]
        fut = _futuro(D, c, H, 1)[['unique_id', 'ds', 's1', 'c1']]
        nf = NeuralForecast(models=[self._modelo(H, D.N)], freq='W-SUN')
        nf.fit(df=df, val_size=H)                          # o neuralforecast exige val_size >= horizonte
        fc = nf.predict(futr_df=fut)
        fc = fc.reset_index() if 'unique_id' not in fc.columns else fc
        nome = {'TSMixer': 'TSMixerx'}.get(self.arq, self.arq)
        if f'{nome}-median' in fc.columns:                 # com MQLoss/DistributionLoss a mediana vem como '-median'
            fc = fc.drop(columns=[nome], errors='ignore').rename(columns={f'{nome}-median': nome})
        cols = ['unique_id', 'ds', nome] + [f'{nome}-{x}-{n}' for x in ('lo', 'hi') for n in NIVEIS_INT]
        return Saida(quant=_sf_para_quantis(fc[cols].copy(), nome, D, h, _pop_c(D, s)))


# =====================================================================================================================
# 14 e 16 · nossas versões A e B (v4_lib)
# =====================================================================================================================
class AGCRN(Modelo):
    nome, familia, origem = 'AGCRN', 'rede de grafos', 'GNN espaço-temporal (versão A do V4)'
    usa_gpu = True
    dependencia = 'propria'

    def __init__(self, **cfg):
        self.cfg = dict(n_sementes=2, epocas_max=40, paciencia=6, frac_origens=0.5, lote=8, retreino_final=True,
                        n_traj=500)
        self.cfg.update(cfg)

    def prever(self, ctx, s):
        m = V.ModeloA(**self.cfg)
        m.prever(ctx.Dr, s)
        return Saida(traj=m.traj_unid_, info={'historico': pd.concat(m.historicos_).to_dict('records')})


class TamanhoForma(Modelo):
    nome, familia, origem = 'Tamanho_x_forma', 'híbrido', 'IFGW (versão B do V4)'
    dependencia = 'propria'

    def __init__(self, **cfg):
        self.cfg = dict(n_traj=500, forma='media', desloc_max=3)
        self.cfg.update(cfg)

    def prever(self, ctx, s):
        m = V.ModeloB(**self.cfg)
        m.prever(ctx.Dr, s)
        return Saida(traj=m.traj_unid_)


# =====================================================================================================================
# 15 · renovação com depleção de suscetíveis
# =====================================================================================================================
class RenovacaoS(Modelo):
    """I_t ~ NB(R(semana) · (S_t / S̄) · Σ w_k I_{t-k} + importação).
    R(semana): perfil sazonal de R_t estimado pela equação de renovação nos anos passados (mediana por semana do ano,
    encolhido para o perfil da UF). S_t: 1 − ρ × incidência acumulada ponderada das últimas temporadas, e cai com
    os casos simulados. S̄: a média histórica do mesmo índice (o perfil de R já embute a imunidade média).
    Incerteza: fator de R por trajetória (UF + região) e ruído NB."""
    nome, familia, origem = 'Renovacao_S', 'mecanístico', 'NUS / IFGW (renovação + suscetíveis)'
    dependencia = 'propria'

    def __init__(self, M=500, rho=4.0, sig_uf=0.07, sig_reg=0.05, disp=10.0, w=(0.35, 0.40, 0.18, 0.07)):
        self.M, self.rho, self.sig_uf, self.sig_reg, self.disp = M, rho, sig_uf, sig_reg, disp
        self.w = np.array(w)

    def _suscetivel(self, D, cj):
        ini1 = D.idx_temporada(D.season[cj])[0]
        pesos = [(ini1, cj, 1.0)]
        s_ = D.season[cj]
        for k, p in ((1, 0.5), (2, 0.25)):
            ii = D.idx_temporada(s_ - k)
            if len(ii) and ii[0] > 0:
                pesos.append((ii[0], ii[-1], p))
        f = sum(p * D.soma(a, b) / D.POP[cj] for a, b, p in pesos)
        return np.clip(1 - self.rho * f, 0.15, 1.0)

    def prever(self, ctx, s):
        D = ctx.Dr
        rng = np.random.default_rng(s)
        c, datas, idx, h = _horizontes(ctx, s)
        I = np.nan_to_num(D.Y[:c + 1])
        K = len(self.w)
        lam = sum(self.w[k] * np.vstack([np.zeros((k + 1, D.N)), I[:-(k + 1)]]) for k in range(K))
        logR = np.log((I + 1) / (lam + 1))
        inf = lam >= 5
        prof = np.zeros((53, D.N))
        uf_i = np.searchsorted(D.ufs, D.uf_of)
        for w in range(1, 53):
            sel = D.woyc[:c + 1] == w
            L = np.where(inf[sel], logR[sel], np.nan)
            med = np.nanmedian(L, 0)
            n_inf = np.sum(~np.isnan(L), 0)
            med_uf = np.array([np.nanmedian(L[:, uf_i == j]) for j in range(len(D.ufs))])[uf_i]
            a = n_inf / (n_inf + 5)
            prof[w] = np.nan_to_num(a * np.nan_to_num(med) + (1 - a) * np.nan_to_num(med_uf))
        k5 = np.ones(5) / 5
        pr = prof[1:53]
        pr = np.vstack([pr[-2:], pr, pr[:2]])
        prof[1:53] = np.apply_along_axis(lambda v: np.convolve(v, k5, mode='valid'), 0, pr)
        prof[53 - 1:] = prof[52]
        # suscetíveis: agora e média histórica (nas SE25 passadas)
        S0 = self._suscetivel(D, c)
        Sh = np.mean([self._suscetivel(D, D.corte(j)) for j in range(2013, s) if D.corte(j) < c], 0)
        clim = np.array([np.nanmean(D.Y[:c + 1][D.woyc[:c + 1] == w], 0) for w in range(1, 53)])
        fim = idx[-1]
        eps = (rng.normal(0, self.sig_uf, (self.M, len(D.ufs)))[:, uf_i] + rng.normal(0, self.sig_reg, (self.M, D.N)))
        tr = np.zeros((self.M, fim + 1, D.N), dtype=np.float32)
        tr[:, :c + 1] = I[None]
        S = np.repeat(S0[None], self.M, 0)
        pop = D.POP[c]
        for t in range(c + 1, fim + 1):
            lam_t = sum(self.w[k] * tr[:, t - k - 1] for k in range(K))
            R = np.exp(np.clip(prof[D.woyc[t]], -3, 1.1)[None] + eps) * (S / Sh[None])
            mu = np.minimum(R * lam_t + 0.05 * clim[D.woyc[t] - 1][None], 0.05 * pop[None])
            p = self.disp / (self.disp + mu)
            tr[:, t] = rng.negative_binomial(self.disp, np.clip(p, 1e-9, 1))
            S = np.clip(S - self.rho * tr[:, t] / pop[None], 0.05, 1)
        return Saida(traj=tr[:, idx], info={'S0_mediana': float(np.median(S0)), 'Shist_mediana': float(np.median(Sh))})


# =====================================================================================================================
# 17 · Chronos-Bolt (zero-shot)
# =====================================================================================================================
class Chronos(Modelo):
    nome, familia, origem = 'Chronos', 'fundação (zero-shot)', 'RKI / NUS (Chronos, TimesFM)'
    usa_gpu = True

    def __init__(self, modelo_hf='amazon/chronos-bolt-base', contexto=512):
        self.modelo_hf, self.contexto = modelo_hf, contexto
        self._pipe = None

    def prever(self, ctx, s):
        import torch
        from chronos import BaseChronosPipeline
        if self._pipe is None:
            self._pipe = BaseChronosPipeline.from_pretrained(
                self.modelo_hf, device_map='cuda' if torch.cuda.is_available() else 'cpu', torch_dtype=torch.float32)
        D = ctx.Dr
        c, datas, idx, h = _horizontes(ctx, s)
        H = int(h.max())
        ini = max(0, c + 1 - self.contexto)
        ctxs = [torch.tensor(D.LINC[ini:c + 1, u], dtype=torch.float32) for u in range(D.N)]
        niveis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        qs, _ = self._pipe.predict_quantiles(ctxs, prediction_length=H, quantile_levels=niveis)
        qs = qs.cpu().numpy()                                          # (N, H, 9) em log1p(inc)
        med, q10, q90 = qs[..., 4], qs[..., 0], qs[..., 8]
        q25 = qs[..., 1] + 0.5 * (qs[..., 2] - qs[..., 1])            # 0,25 entre 0,2 e 0,3
        q75 = qs[..., 6] + 0.5 * (qs[..., 7] - qs[..., 6])
        slo = (med - q10) / norm.ppf(0.9)
        shi = (q90 - med) / norm.ppf(0.9)
        lq = np.stack([med - 1.96 * slo, med - 1.645 * slo, q10, q25, med, q75, q90, med + 1.645 * shi,
                       med + 1.96 * shi], -1)                          # (N, H, 9)
        lq = np.moveaxis(lq, 0, 1)[h - 1]                              # (W, N, 9)
        return Saida(quant=_para_contagem(lq, _pop_c(D, s)[None, :, None]))


# =====================================================================================================================
# catálogo
# =====================================================================================================================
def catalogo(cfg_agcrn=None, inla_kw=None):
    inla_kw = inla_kw or {}
    return [
        Climatologia(), CanalEndemico(), ModeloINLA('bb', **inla_kw), ARLongo(), Surge(),
        ARIMAFourier(), ProphetModelo(), ETSMSTL(), ModeloINLA('ghr', **inla_kw),
        LightGBMDireto(), XGBoostQuantis(),
        NeuralForecastModelo('DeepAR'), NeuralForecastModelo('NHITS'), NeuralForecastModelo('TFT'),
        NeuralForecastModelo('TSMixer'),
        AGCRN(**(cfg_agcrn or {})), RenovacaoS(), TamanhoForma(), Chronos(),
    ]
