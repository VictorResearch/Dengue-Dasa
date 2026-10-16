"""
V4 · Biblioteca do Modelo de previsão de temporada (3rd IMDC).

Organização (mesma numeração do 04_ESTADO_DA_ARTE.md):
  ETAPA 1  avaliador oficial (WIS, WIS relativo, cobertura) + modelos-base (climatologia, canal endêmico NB,
           AR recursivo com trajetórias) + leitura das previsões publicadas do 3º IMDC (BB e concorrentes)
  ETAPA 5  variáveis de temporada (taxa de ataque 1–3 temporadas, incidência acumulada, tendência de 8 semanas,
           anomalias climáticas prolongadas, SEAS5, ENSO, parâmetros de Richards estilo Episcanner, vizinhos)
  ETAPA 2  Versão A (AGCRN direta multi-horizonte, origens sintéticas)  e  Versão B (tamanho × forma)
  ETAPA 3  soma por trajetórias correlacionadas (dentro de A e B) + calibração conformal em temporadas passadas
  ETAPA 4  ensemble por quantis (média, mediana ou pesos encolhidos)

Regra de ouro: para a temporada s (SE41 de s → SE40 de s+1), nada depois da SE25 de s é usado — nem nos modelos,
nem na calibração, nem nos pesos do ensemble.
"""
import os
import time
import math
import datetime as dt
import warnings

import numpy as np
import pandas as pd
from scipy.stats import nbinom

warnings.filterwarnings('ignore')

Q = np.array([.025, .05, .10, .25, .5, .75, .90, .95, .975])
QCOLS = ['lower_95', 'lower_90', 'lower_80', 'lower_50', 'pred', 'upper_50', 'upper_80', 'upper_90', 'upper_95']
NIVEIS = [50, 80, 90, 95]
ALFAS = np.array([.50, .20, .10, .05])           # intervalo central de nível (1-α): 50, 80, 90, 95
CLIMA_COLS = ['temp_min', 'temp_med', 'temp_max', 'precip_med', 'rainy_days', 'rel_humid_med', 'thermal_range']
TEMPORADAS_OFICIAIS = [2022, 2023, 2024, 2025]   # S1..S4
TEMPORADA_DE_VALIDACAO = {2022: 1, 2023: 2, 2024: 3, 2025: 4}


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# =====================================================================================================================
# Calendário epidemiológico (igual ao MMWR: a SE1 é a semana, de domingo a sábado, que contém 4 de janeiro)
# =====================================================================================================================
def inicio_ano_epi(ano):
    jan4 = dt.date(ano, 1, 4)
    return jan4 - dt.timedelta(days=(jan4.weekday() + 1) % 7)


def datas_temporada(s):
    """Domingos da SE41 do ano s até a SE40 do ano s+1 (52 ou 53 semanas)."""
    ini = inicio_ano_epi(s) + dt.timedelta(weeks=40)
    fim = inicio_ano_epi(s + 1) + dt.timedelta(weeks=39)
    return pd.date_range(ini, fim, freq='7D')


def data_para_epiweek(d):
    d = pd.Timestamp(d).date()
    a = d.year + 1
    while inicio_ano_epi(a) > d:
        a -= 1
    return a * 100 + (d - inicio_ano_epi(a)).days // 7 + 1


# =====================================================================================================================
# Dados
# =====================================================================================================================
class Dados:
    """Painel semanal de um nível ('regiao' ou 'uf') em matrizes tempo × unidade. O calendário é estendido 80 semanas
    além do último dado para que as datas futuras (horizontes) existam."""

    def __init__(self, pasta, nivel='regiao', semanas_futuras=80):
        self.pasta, self.nivel = pasta, nivel
        cal = pd.read_parquet(os.path.join(pasta, 'calendario.parquet')).sort_values('date').reset_index(drop=True)
        self.ultimo_obs = len(cal) - 1
        fut = pd.date_range(cal.date.iloc[-1] + pd.Timedelta(weeks=1), periods=semanas_futuras, freq='7D')
        ep = np.array([data_para_epiweek(d) for d in fut])
        extra = pd.DataFrame({'date': fut, 'epiweek': ep, 'woy': ep % 100})
        extra['season'] = np.where(extra.woy >= 41, ep // 100, ep // 100 - 1)
        self.cal = pd.concat([cal, extra], ignore_index=True)
        self.datas = pd.to_datetime(self.cal.date).values
        self.T = len(self.cal)
        self.epiweek = self.cal.epiweek.values.astype(int)
        self.woy = self.cal.woy.values.astype(int)
        self.woyc = np.minimum(self.woy, 52)                       # SE53 usa a climatologia da SE52
        self.season = self.cal.season.values.astype(int)
        self.idx_data = {pd.Timestamp(d): i for i, d in enumerate(self.datas)}

        p = pd.read_parquet(os.path.join(pasta, f'painel_{nivel}.parquet'))
        p['date'] = pd.to_datetime(p['date'])
        self.unidades = np.sort(p.unidade.unique())
        self.N = len(self.unidades)
        datas_obs = self.cal.date.iloc[:self.ultimo_obs + 1]

        def piv(col):
            m = p.pivot(index='date', columns='unidade', values=col).reindex(index=datas_obs, columns=self.unidades)
            out = np.full((self.T, self.N), np.nan)
            out[:self.ultimo_obs + 1] = m.values
            return out

        self.Y = piv('casos')
        self.Y[:self.ultimo_obs + 1] = np.nan_to_num(self.Y[:self.ultimo_obs + 1])
        pop = pd.DataFrame(piv('pop')).ffill().bfill().values
        self.POP = np.maximum(pop, 1.0)
        self.CLIMA = {c: piv(c) for c in CLIMA_COLS if c in p.columns}
        info = p.drop_duplicates('unidade').set_index('unidade').loc[self.unidades]
        self.uf_of = info.uf_code.values.astype(int)
        self.macro_of = info.macroregional_geocode.values.astype(int)
        self.ufs = np.sort(np.unique(self.uf_of))
        self.A_uf = (self.uf_of[None, :] == self.ufs[:, None]).astype(float)   # (n_uf, N): soma unidades -> UF

        oc = pd.read_parquet(os.path.join(pasta, 'oceano.parquet'))
        oc['date'] = pd.to_datetime(oc['date'])
        oc = oc.set_index('date').reindex(self.cal.date).ffill()
        self.OCEANO = {c: oc[c].values for c in ('enso', 'iod', 'pdo')}

        self.seas5 = pd.read_parquet(os.path.join(pasta, f'seas5_{nivel}.parquet'))
        self.seas5['reference_month'] = pd.to_datetime(self.seas5['reference_month'])
        adj = np.load(os.path.join(pasta, f'adj_{nivel}.npz'), allow_pickle=True)
        pos = {u: i for i, u in enumerate(adj['unidades'])}
        ordem = [pos[u] for u in self.unidades]
        self.ADJ = adj['adj'][np.ix_(ordem, ordem)].astype(float)
        self.rich = pd.read_parquet(os.path.join(pasta, f'richards_{nivel}.parquet')).set_index(['unidade', 'season'])

        self.INC = self.Y / self.POP * 1e5
        self.LINC = np.log1p(self.INC)
        self.cs = np.vstack([np.zeros(self.N), np.nancumsum(np.nan_to_num(self.Y), 0)])   # cs[b+1]-cs[a] = soma a..b
        log(f'Dados[{nivel}]: {self.N} unidades, {self.ultimo_obs + 1} semanas observadas '
            f'({pd.Timestamp(self.datas[0]).date()} → {pd.Timestamp(self.datas[self.ultimo_obs]).date()})')

    # ---------------------------------------------------------------- índices de temporada
    def corte(self, s):
        """índice da SE25 do ano s (último dado permitido para prever a temporada s)."""
        i = np.where(self.epiweek == s * 100 + 25)[0]
        assert len(i) == 1, f'SE25 de {s} não encontrada'
        return int(i[0])

    def idx_temporada(self, s, ate=None):
        i = np.where(self.season == s)[0]
        if ate is not None:
            i = i[i <= ate]
        return i

    def temporada_completa(self, s, ate=None):
        lim = self.ultimo_obs if ate is None else min(ate, self.ultimo_obs)
        i = np.where(self.epiweek == (s + 1) * 100 + 40)[0]
        return len(i) == 1 and i[0] <= lim

    def soma(self, a, b):
        """soma de casos nas semanas a..b (inclusive), por unidade."""
        return self.cs[b + 1] - self.cs[a]

    def inc_periodo(self, a, b):
        pop = self.POP[b]
        return self.soma(a, b) / pop * 1e5


# =====================================================================================================================
# ETAPA 1 · métricas
# =====================================================================================================================
def wis(q, y):
    """WIS do 3º IMDC (Bracher et al. 2021). q: (..., 9) na ordem QCOLS; y: (...). Ignora y NaN."""
    q = np.asarray(q, float)
    y = np.asarray(y, float)
    s = 0.5 * np.abs(y - q[..., 4])
    for i, a in enumerate(ALFAS):
        lo, up = q[..., 3 - i], q[..., 5 + i]
        s = s + a / 2 * ((up - lo) + 2 / a * (lo - y) * (y < lo) + 2 / a * (y - up) * (y > up))
    return s / (len(ALFAS) + 0.5)


def cobertura(q, y, nivel):
    i = NIVEIS.index(nivel)
    lo, up = q[..., 3 - i], q[..., 5 + i]
    ok = ~np.isnan(y)
    return np.mean(((y >= lo) & (y <= up))[ok]) * 100


def arruma_quantis(q):
    """ordena, garante ≥ 0 e intervalos aninhados."""
    q = np.sort(np.maximum(np.nan_to_num(q, nan=0.0), 0), axis=-1)
    return q


def quantis_de_amostras(amostras, eixo=0):
    return np.moveaxis(np.quantile(amostras, Q, axis=eixo), 0, -1)


# ---------------------------------------------------------------- formato longo (uma linha por UF × semana)
def para_df(qarr, datas, ufs, modelo, s):
    W, U = qarr.shape[:2]
    df = pd.DataFrame({'modelo': modelo, 'season': s,
                       'uf_code': np.tile(ufs, W), 'date': np.repeat(pd.to_datetime(datas), U)})
    for j, c in enumerate(QCOLS):
        df[c] = qarr[..., j].ravel()
    return df


def de_df(df, datas, ufs):
    piv = df.set_index(['date', 'uf_code'])[QCOLS]
    idx = pd.MultiIndex.from_product([pd.to_datetime(datas), ufs])
    return piv.reindex(idx).values.reshape(len(datas), len(ufs), 9)


class Verdade:
    """casos observados por UF × semana (painel_uf)."""

    def __init__(self, D_uf):
        self.D = D_uf
        self.ufs = D_uf.ufs

    def y(self, datas):
        idx = [self.D.idx_data.get(pd.Timestamp(d)) for d in datas]
        out = np.full((len(datas), self.D.N), np.nan)
        for k, i in enumerate(idx):
            if i is not None and i <= self.D.ultimo_obs:
                out[k] = self.D.Y[i]
        return out


def avaliar(prev, verdade, ref=('BB', 'climatologia')):
    """prev: DataFrame longo com várias colunas 'modelo' e 'season'. Devolve (por_uf, resumo).
    por_uf: modelo × season × UF com WIS, WIS em log, cobertura e MAE.
    resumo: por modelo, média geométrica do WIS relativo (como no 3º IMDC) contra cada referência."""
    linhas, incompletos = [], []
    for (m, s), g in prev.groupby(['modelo', 'season']):
        g = g.drop_duplicates(['uf_code', 'date'], keep='last')
        datas = np.sort(g.date.unique())
        ufs = np.sort(g.uf_code.unique())
        q = de_df(g, datas, ufs)
        y = verdade.y(datas)[:, np.searchsorted(verdade.ufs, ufs)]
        ok = ~np.isnan(y)
        if np.isnan(q[ok]).any():                         # previsão com buracos: fora (senão o WIS vira NaN)
            incompletos.append(f'{m} {s}')
            continue
        w = wis(q, y)
        wl = wis(np.log1p(q), np.log1p(y))
        for j, u in enumerate(ufs):
            o = ok[:, j]
            if o.sum() == 0:
                continue
            linhas.append(dict(modelo=m, season=s, uf_code=u, n_sem=int(o.sum()),
                               WIS=np.mean(w[o, j]), WIS_log=np.mean(wl[o, j]),
                               MAE=np.mean(np.abs(q[o, j, 4] - y[o, j])), casos=np.sum(y[o, j]),
                               **{f'IC{n}': cobertura(q[o, j][None], y[o, j][None], n) for n in NIVEIS}))
    if incompletos:
        print(f'avaliar: {len(incompletos)} previsões com semanas/UFs faltando ficaram de fora:', ', '.join(incompletos[:15]))
    por_uf = pd.DataFrame(linhas)
    if por_uf.empty:
        return por_uf, pd.DataFrame(columns=['modelo'])
    resumo = []
    for m, g in por_uf.groupby('modelo'):
        r = dict(modelo=m, temporadas=g.season.nunique(), WIS_medio=g.WIS.mean(), WIS_log_medio=g.WIS_log.mean(),
                 WIS_norm=g.WIS.sum() / g.casos.sum() * g.n_sem.mean(),
                 **{f'IC{n}': g[f'IC{n}'].mean() for n in NIVEIS})
        for rf in ref:
            b = por_uf[por_uf.modelo == rf][['season', 'uf_code', 'WIS']].rename(columns={'WIS': 'WIS_ref'})
            mm = g.merge(b, on=['season', 'uf_code'])
            if len(mm) == 0 or rf == m:
                continue
            lr = np.log(mm.WIS / mm.WIS_ref)
            por_uf_geo = mm.assign(lr=lr).groupby('uf_code').lr.mean()          # média geométrica nas temporadas
            r[f'R_vs_{rf}'] = float(np.exp(por_uf_geo.mean()))                 # média geométrica nas UFs
            r[f'dp_logR_vs_{rf}'] = float(lr.std())
            r[f'UFs_melhor_que_{rf}'] = int((por_uf_geo < 0).sum())
        resumo.append(r)
    resumo = pd.DataFrame(resumo)
    col = [c for c in resumo.columns if c.startswith('R_vs_')]
    if col:
        resumo = resumo.sort_values(col[0])
    return por_uf, resumo


# ---------------------------------------------------------------- previsões publicadas do 3º IMDC
def ler_referencias_imdc3(pasta_ref, ufs):
    """lê todas as previsões estaduais publicadas (BB = modelo-base oficial). season = 2021 + validation."""
    out = []
    for f in sorted(os.listdir(pasta_ref)):
        if not f.startswith('dengue_state') or not f.endswith('.csv.gz'):
            continue
        nome = f.replace('dengue_state_', '').replace('.csv.gz', '').replace('3rd_imdc_', '')
        d = pd.read_csv(os.path.join(pasta_ref, f))
        if 'adm_1' not in d.columns or 'validation' not in d.columns:
            continue
        d = d[d.adm_1.isin(ufs)]
        d = d.assign(modelo='BB' if 'procc_bb' in nome else f'IMDC:{nome}', season=2021 + d.validation,
                     uf_code=d.adm_1, date=pd.to_datetime(d.date))
        out.append(d[['modelo', 'season', 'uf_code', 'date'] + QCOLS])
    return pd.concat(out, ignore_index=True)


# =====================================================================================================================
# ETAPA 1 · modelos-base (nível UF; qualquer painel serve)
# =====================================================================================================================
def _pop_alvo(D, s):
    return D.POP[D.corte(s)]


def base_climatologia(D, s):
    """quantis empíricos da incidência histórica na mesma semana epidemiológica, × população atual."""
    c, datas = D.corte(s), datas_temporada(s)
    w_alvo = np.minimum([data_para_epiweek(d) % 100 for d in datas], 52)
    pop = _pop_alvo(D, s)
    q = np.zeros((len(datas), D.N, 9))
    for i, w in enumerate(w_alvo):
        hist = D.INC[:c + 1][D.woyc[:c + 1] == w]
        q[i] = np.quantile(hist, Q, axis=0).T * pop[:, None] / 1e5
    return datas, arruma_quantis(q)


def base_canal_endemico(D, s):
    """Canal endêmico (Cornell): Binomial Negativa ajustada por momentos aos valores da mesma semana nos anos anteriores."""
    c, datas = D.corte(s), datas_temporada(s)
    w_alvo = np.minimum([data_para_epiweek(d) % 100 for d in datas], 52)
    pop = _pop_alvo(D, s)
    q = np.zeros((len(datas), D.N, 9))
    for i, w in enumerate(w_alvo):
        x = D.INC[:c + 1][D.woyc[:c + 1] == w] * pop / 1e5
        m, v = x.mean(0), x.var(0, ddof=1)
        r = np.where(v > m * 1.001, m ** 2 / np.maximum(v - m, 1e-9), 1e6)
        p = r / (r + np.maximum(m, 1e-9))
        q[i] = nbinom.ppf(Q[:, None], r[None], p[None]).T
    return datas, arruma_quantis(q)


def base_ar_recursivo(D, s, n_traj=1000, seed=0):
    """AR em log(incidência) com defasagens 1–4 e 52, harmônicos e intercepto por unidade, desenrolado a partir da SE25
    com resíduos reamostrados (trajetórias). É o 'incremental' — mantido como membro de ensemble."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(seed)
    c, datas = D.corte(s), datas_temporada(s)
    y = D.LINC
    lags = [1, 2, 3, 4, 52]

    def harm(t):
        a = 2 * np.pi * D.woyc[t] / 52
        return np.array([np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)])

    X, Y_ = [], []
    for t in range(53, c + 1):
        X.append(np.c_[np.column_stack([y[t - l] for l in lags]), np.tile(harm(t), (D.N, 1)), np.eye(D.N)])
        Y_.append(y[t])
    X, Y_ = np.vstack(X), np.concatenate(Y_)
    m = Ridge(alpha=1.0).fit(X, Y_)
    res = Y_ - m.predict(X)
    cl, ch, cu = m.coef_[:5], m.coef_[5:9], m.coef_[9:] + m.intercept_
    fim = D.idx_data[pd.Timestamp(datas[-1])]
    traj = np.repeat(y[None, :c + 1], n_traj, 0)
    traj = np.concatenate([traj, np.zeros((n_traj, fim - c, D.N))], 1)
    for t in range(c + 1, fim + 1):
        p = sum(cl[i] * traj[:, t - l] for i, l in enumerate(lags)) + harm(t) @ ch + cu
        traj[:, t] = np.clip(p + rng.choice(res, size=p.shape), 0, None)
    idx = [D.idx_data[pd.Timestamp(d)] for d in datas]
    casos = np.expm1(traj[:, idx]) * _pop_alvo(D, s) / 1e5
    return datas, arruma_quantis(quantis_de_amostras(casos))


def base_ar_longo(D, s, p=92, n_traj=1000, seed=0, ridge=1.0):
    """AR(p) longo por unidade em log2(casos+1) centrado (estilo LNCC_ARP26 do 3º IMDC), ajustado por mínimos quadrados
    com penalização leve e simulado da SE25 até o fim da temporada com inovações gaussianas (trajetórias)."""
    rng = np.random.default_rng(seed + s)
    c, datas = D.corte(s), datas_temporada(s)
    fim = D.idx_data[pd.Timestamp(datas[-1])]
    idx = [D.idx_data[pd.Timestamp(d)] for d in datas]
    out = np.zeros((n_traj, len(datas), D.N))
    for u in range(D.N):
        z = np.log2(np.nan_to_num(D.Y[:c + 1, u]) + 1)
        mz = z.mean()
        z = z - mz
        X = np.column_stack([z[p - k - 1:len(z) - k - 1] for k in range(p)])      # defasagens 1..p
        yv = z[p:]
        coef = np.linalg.solve(X.T @ X + ridge * np.eye(p), X.T @ yv)
        sd = np.std(yv - X @ coef)
        tr = np.zeros((n_traj, fim + 1))
        tr[:, :c + 1] = z
        for t in range(c + 1, fim + 1):
            tr[:, t] = tr[:, t - p:t][:, ::-1] @ coef + rng.normal(0, sd, n_traj)
        out[..., u] = np.clip(2 ** (tr[:, idx] + mz) - 1, 0, None)
    return datas, arruma_quantis(quantis_de_amostras(out))


# =====================================================================================================================
# ETAPA 5 · variáveis de temporada (tudo calculado "como visto na SE25 do ano j")
# =====================================================================================================================
def _anomalias_clima(D, c_global):
    """anomalia padronizada semanal de cada variável climática (climatologia por semana do ano usando só t ≤ c_global)."""
    out = {}
    for v in ('temp_med', 'temp_min', 'precip_med', 'rel_humid_med'):
        x = D.CLIMA[v]
        clim = np.full((53, D.N), np.nan)
        for w in range(1, 53):
            sel = (D.woyc[:c_global + 1] == w)
            clim[w] = np.nanmean(x[:c_global + 1][sel], 0)
        an = x - clim[D.woyc]
        sd = np.nanstd(an[:c_global + 1], 0)
        out[v] = an / np.where(sd > 0, sd, 1)
    return out


def _seas5_anomalias(D, ano, ano_max):
    """anomalia padronizada da previsão SEAS5 emitida em junho do ano (meses 1–6 à frente: jul–dez),
    relativa às previsões de junho dos anos ≤ ano_max."""
    f = D.seas5[D.seas5.reference_month.dt.month == 6].copy()
    f['ano'] = f.reference_month.dt.year
    f = f[f.ano <= ano_max]
    out = {}
    for v in ('temp_med', 'precip_tot', 'umid_med'):
        g = f.groupby(['unidade', 'ano'])[v].mean().unstack()          # média dos 6 meses
        z = (g.sub(g.mean(1), axis=0)).div(g.std(1).replace(0, np.nan), axis=0)
        out[f'seas_{v}_anom'] = z[ano].reindex(D.unidades).values if ano in z.columns else np.full(D.N, np.nan)
    return out


def _vizinhos(D, v):
    grau = D.ADJ.sum(1)
    m = D.ADJ @ np.nan_to_num(v) / np.maximum(grau, 1)
    return np.where(grau > 0, m, v)


def features_temporada(D, j, c_global, anom=None):
    """DataFrame (uma linha por unidade) com as variáveis da temporada j, calculadas com dados até a SE25 do ano j.
    c_global = índice do corte da temporada que está sendo prevista (limita climatologias)."""
    cj = D.corte(j)
    assert cj <= c_global
    if anom is None:
        anom = _anomalias_clima(D, c_global)
    L = np.log1p
    f = {}
    ini1 = D.idx_temporada(j - 1)[0]
    f['ar1_parcial'] = L(D.inc_periodo(ini1, cj))                         # temporada j-1 até a SE25 (inclui o pico)
    for k, nome in ((2, 'ar2'), (3, 'ar3'), (4, 'ar4')):
        s_ = j - k
        if D.temporada_completa(s_, cj) and D.idx_temporada(s_)[0] > 0:
            ii = D.idx_temporada(s_)
            f[nome] = L(D.inc_periodo(ii[0], ii[-1]))
        else:
            f[nome] = np.full(D.N, np.nan)
    ex = lambda a: np.expm1(np.nan_to_num(a))
    f['inc_acum_4temp'] = L(ex(f['ar1_parcial']) + ex(f['ar2']) + ex(f['ar3']) + ex(f['ar4']))
    f['inc_acum_decaimento'] = L(ex(f['ar1_parcial']) + .5 * ex(f['ar2']) + .25 * ex(f['ar3']) + .125 * ex(f['ar4']))
    f['inc_ano_se1_25'] = L(D.inc_periodo(cj - 24, cj))
    f['inc_ult8'] = L(D.inc_periodo(cj - 7, cj))
    f['inc_ult4'] = L(D.inc_periodo(cj - 3, cj))
    k8 = np.arange(8) - 3.5
    f['tend8'] = (k8 @ D.LINC[cj - 7:cj + 1]) / (k8 @ k8)                # inclinação de log(incidência) nas últimas 8 sem.
    ii = D.idx_temporada(j - 1, ate=cj)
    f['pico_parcial_sem'] = np.argmax(D.Y[ii], 0).astype(float)          # semana (desde a SE41) do pico parcial
    f['razao_ult8_pico'] = f['inc_ult8'] - L(np.max(D.INC[ii], 0))
    # clima prolongado (anomalias padronizadas)
    for v, a in anom.items():
        f[f'anom_{v}_12s'] = np.nanmean(a[cj - 11:cj + 1], 0)
        f[f'anom_{v}_26s'] = np.nanmean(a[cj - 25:cj + 1], 0)
    f['semanas_quentes_26'] = np.nansum(anom['temp_med'][cj - 25:cj + 1] > 1, 0).astype(float)
    f['semanas_secas_26'] = np.nansum(anom['precip_med'][cj - 25:cj + 1] < -1, 0).astype(float)
    f.update(_seas5_anomalias(D, j, pd.Timestamp(D.datas[c_global]).year))
    for v in ('enso', 'iod', 'pdo'):
        f[f'{v}_12s'] = np.full(D.N, np.nanmean(D.OCEANO[v][cj - 11:cj + 1]))
    f['enso_delta'] = np.full(D.N, np.nanmean(D.OCEANO['enso'][cj - 3:cj + 1]) - np.nanmean(D.OCEANO['enso'][cj - 15:cj - 11]))
    # Richards (estilo Episcanner): temporada j-1 truncada na SE25 e temporada j-2 completa
    pop = D.POP[cj]
    for s_, pref, cols in ((j - 1, 'richT', ('richT_logL', 'richT_r', 'richT_pico', 'richT_ok')),
                           (j - 2, 'rich', ('rich_logL', 'rich_r', 'rich_pico', 'rich_ok'))):
        idx = pd.MultiIndex.from_arrays([D.unidades, np.full(D.N, s_)])
        r = D.rich.reindex(idx)[list(cols)]
        f[f'{pref}{"_1" if pref == "richT" else "_2"}_logInc'] = L(np.expm1(r[cols[0]].values) / pop * 1e5)
        f[f'{pref}{"_1" if pref == "richT" else "_2"}_r'] = np.where(r[cols[3]].values == 1, r[cols[1]].values, np.nan)
        pico = r[cols[2]].values
        f[f'{pref}{"_1" if pref == "richT" else "_2"}_pico'] = np.where((r[cols[3]].values == 1) & (pico >= 0) & (pico <= 52),
                                                                      pico, np.nan)   # ajuste ruim → faltante
    # vizinhos
    for v in ('ar1_parcial', 'ar2', 'inc_ult8', 'tend8'):
        f[f'viz_{v}'] = _vizinhos(D, f[v])
    # contexto da UF (só faz sentido no nível região)
    if D.nivel == 'regiao':
        pop_uf = D.A_uf @ pop
        for v, (a, b) in (('uf_ar1_parcial', (ini1, cj)), ('uf_inc_ult8', (cj - 7, cj))):
            f[v] = (L(D.A_uf @ D.soma(a, b) / pop_uf * 1e5))[np.searchsorted(D.ufs, D.uf_of)]
    f['log_pop'] = np.log(pop)
    f['uf_code'] = D.uf_of
    f['macro'] = D.macro_of
    df = pd.DataFrame(f, index=D.unidades)
    df.index.name = 'unidade'
    return df


# grupos de variáveis da ETAPA 5 (para a ablação: tirar um grupo por vez e medir o efeito)
GRUPOS_VARS = {
    'taxa_de_ataque_1_2_3': ['ar1_parcial', 'ar2', 'ar3', 'ar4'],
    'incidencia_acumulada': ['inc_acum_4temp', 'inc_acum_decaimento', 'inc_ano_se1_25'],
    'tendencia_8_semanas': ['inc_ult8', 'inc_ult4', 'tend8', 'pico_parcial_sem', 'razao_ult8_pico'],
    'clima_prolongado': [f'anom_{v}_{j}s' for v in ('temp_med', 'temp_min', 'precip_med', 'rel_humid_med') for j in (12, 26)]
                        + ['semanas_quentes_26', 'semanas_secas_26'],
    'seas5_previsao': ['seas_temp_med_anom', 'seas_precip_tot_anom', 'seas_umid_med_anom'],
    'enso': ['enso_12s'],
    'richards_episcanner': ['richT_1_logInc', 'richT_1_r', 'richT_1_pico', 'rich_2_logInc', 'rich_2_r', 'rich_2_pico'],
    'vizinhos_e_uf': ['viz_ar1_parcial', 'viz_ar2', 'viz_inc_ult8', 'viz_tend8', 'uf_ar1_parcial', 'uf_inc_ult8'],
}


def alvo_temporada(D, j):
    ii = D.idx_temporada(j)
    return np.log1p(D.inc_periodo(ii[0], ii[-1]))


# =====================================================================================================================
# Trajetórias → UF
# =====================================================================================================================
def soma_para_uf(D, traj):
    """traj: (M, W, N) casos por unidade → (M, W, n_uf)."""
    return traj @ D.A_uf.T


# =====================================================================================================================
# ETAPA 2 · Versão B: tamanho da temporada (LightGBM) × forma da curva (temporadas passadas)
# =====================================================================================================================
def formas_historicas(D, temporadas, kappa=100.0, suaviza=3):
    """forma semanal normalizada (soma 1) por temporada × unidade, encolhida para a forma da UF quando há poucos casos."""
    formas = {}
    for j in temporadas:
        ii = D.idx_temporada(j)
        y = D.Y[ii]                                    # (W, N)
        yu = y @ D.A_uf.T                              # (W, n_uf)
        pu = yu / np.maximum(yu.sum(0), 1)
        pu_n = pu[:, np.searchsorted(D.ufs, D.uf_of)]  # forma da UF de cada unidade
        tot = y.sum(0)
        p = (y + kappa * pu_n) / (tot + kappa)
        if suaviza > 1:
            k = np.ones(suaviza) / suaviza
            p = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 0, p)
        p = np.nan_to_num(np.maximum(p, 0))
        soma = p.sum(0, keepdims=True)
        formas[j] = np.where(soma > 0, p / np.where(soma > 0, soma, 1), 1.0 / len(p))   # sem casos → forma uniforme
    return formas


class ModeloB:
    """Tamanho × forma.
    1) LightGBM prevê log(taxa de ataque da temporada) por unidade com as variáveis da ETAPA 5.
    2) Resíduos fora da amostra (deixando uma temporada de fora) são guardados por temporada inteira.
    3) Cada trajetória sorteia UMA temporada passada j*: usa o vetor de resíduos de j* (correlação espacial real) e
       a forma semanal de j* (momento do pico); adiciona ruído de observação Gama-Poisson.
    4) Soma as unidades → UF."""

    def __init__(self, n_traj=1000, ano_ini=2012, kappa=100.0, jitter=0.15, disp_obs=30.0, seed=0,
                 params=None, usar_vars=None, excluir_vars=('macro', 'iod_12s', 'pdo_12s', 'enso_delta'),
                 forma='media', desloc_max=3, peso_recente=0.0):
        self.n_traj, self.ano_ini, self.kappa = n_traj, ano_ini, kappa
        self.jitter, self.disp_obs, self.seed = jitter, disp_obs, seed
        self.params = params or dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=25,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                     verbose=-1)
        self.usar_vars, self.excluir_vars = usar_vars, tuple(excluir_vars or ())
        self.forma, self.desloc_max, self.peso_recente = forma, desloc_max, peso_recente

    def _xy(self, D, temporadas, c, anom):
        X, y, g = [], [], []
        for j in temporadas:
            X.append(features_temporada(D, j, c, anom))
            y.append(alvo_temporada(D, j))
            g.append(np.full(D.N, j))
        return pd.concat(X), np.concatenate(y), np.concatenate(g)

    def prever(self, D, s, devolver_traj=False):
        import lightgbm as lgb
        rng = np.random.default_rng(self.seed + s)
        c = D.corte(s)
        anom = _anomalias_clima(D, c)
        treino = [j for j in range(self.ano_ini, s) if D.temporada_completa(j, c)]
        X, y, g = self._xy(D, treino, c, anom)
        Xs = features_temporada(D, s, c, anom)
        cols = self.usar_vars or [v for v in X.columns if v not in self.excluir_vars]
        cats = [cc for cc in ('uf_code', 'macro') if cc in cols]
        for cc in cats:
            X[cc] = X[cc].astype('category')
            Xs[cc] = pd.Categorical(Xs[cc], categories=X[cc].cat.categories)
        Xs = Xs[cols]
        X = X[cols]
        # resíduos fora da amostra, uma temporada de fora por vez
        E = np.zeros((len(treino), D.N))
        for k, j in enumerate(treino):
            m = lgb.LGBMRegressor(**self.params).fit(X[g != j], y[g != j])
            E[k] = y[g == j] - m.predict(X[g == j])
        modelo = lgb.LGBMRegressor(**self.params).fit(X, y)
        self.modelo_, self.cols_, self.treino_ = modelo, cols, treino
        pred = modelo.predict(Xs)
        # formas das temporadas de treino
        formas = formas_historicas(D, treino, self.kappa)
        datas = datas_temporada(s)
        W = len(datas)
        pop = D.POP[c]
        # temporada sorteada por trajetória (resíduo conjunto de todas as unidades = correlação espacial real)
        pesos = np.exp(self.peso_recente * (np.arange(len(treino)) - len(treino) + 1))
        js = rng.choice(len(treino), self.n_traj, p=pesos / pesos.sum())
        # forma: 'amostrada' = forma da temporada sorteada; 'media' = forma média das temporadas de treino,
        # deslocada ±desloc_max semanas (incerteza no momento do pico)
        def ajusta_W(f):
            return np.vstack([f, np.repeat(f[-1:], W - len(f), 0)])[:W] if len(f) < W else f[:W]
        f_media = np.mean([ajusta_W(formas[j]) for j in treino], 0)
        traj = np.zeros((self.n_traj, W, D.N), dtype=np.float32)
        for m_ in range(self.n_traj):
            k = js[m_]
            e = E[k] + rng.normal(0, self.jitter, D.N)
            tot = np.maximum(np.expm1(pred + e), 0) * pop / 1e5
            if self.forma == 'amostrada':
                f = ajusta_W(formas[treino[k]])
            else:
                d_ = rng.integers(-self.desloc_max, self.desloc_max + 1) if self.desloc_max else 0
                f = np.roll(f_media, d_, axis=0)
            traj[m_] = tot[None] * f
        # ruído de observação Gama-Poisson
        gam = rng.gamma(self.disp_obs, 1 / self.disp_obs, traj.shape).astype(np.float32)
        traj = rng.poisson(traj * gam).astype(np.float32)
        self.traj_unid_ = traj                       # (M, W, N) por unidade — usado pela competição
        tu = soma_para_uf(D, traj)
        self.importancia_ = pd.Series(modelo.feature_importances_, index=cols).sort_values(ascending=False)
        self.residuos_ = pd.DataFrame(E, index=treino, columns=D.unidades)
        q = arruma_quantis(quantis_de_amostras(tu))
        return (datas, q, tu) if devolver_traj else (datas, q)


# =====================================================================================================================
# ETAPA 2 · Versão A: AGCRN direta multi-horizonte (origens sintéticas)
# =====================================================================================================================
def _torch():
    import torch
    return torch


class PreparaA:
    """Tensores para a Versão A. Tudo é função do corte c (nada depois de c entra nas padronizações)."""

    def __init__(self, D, c, L=104, H=68):
        self.D, self.c, self.L, self.H = D, c, L, H
        T, N = D.T, D.N
        anom = _anomalias_clima(D, c)
        linc = np.nan_to_num(D.LINC)
        mu, sd = linc[:c + 1].mean(), linc[:c + 1].std()
        a = 2 * np.pi * D.woyc / 52
        enso = np.nan_to_num(D.OCEANO['enso'])
        dyn = [(linc - mu) / sd] + [np.nan_to_num(anom[v]) for v in ('temp_med', 'temp_min', 'precip_med', 'rel_humid_med')]
        dyn += [np.repeat(np.sin(a)[:, None], N, 1), np.repeat(np.cos(a)[:, None], N, 1), np.repeat(enso[:, None], N, 1)]
        self.X = np.stack(dyn, -1).astype(np.float32)                        # (T, N, d_in)
        # variáveis da origem (versão "rolante" das variáveis de temporada)
        cs = np.vstack([np.zeros(N), np.cumsum(np.nan_to_num(D.Y), 0)])

        def inc_win(t, a_, b_):   # incidência nas semanas t-b_ .. t-a_
            i0 = np.clip(t - b_, 0, None)
            return np.log1p((cs[t - a_ + 1] - cs[i0]) / D.POP[t] * 1e5)
        t_all = np.arange(T)
        t_ok = np.clip(t_all, 156, T - 1)
        r52 = np.stack([inc_win(t, 0, 51) for t in t_ok])
        r104 = np.stack([inc_win(t, 52, 103) for t in t_ok])
        r156 = np.stack([inc_win(t, 104, 155) for t in t_ok])
        u8 = np.stack([inc_win(t, 0, 7) for t in t_ok])
        u4 = np.stack([inc_win(t, 0, 3) for t in t_ok])
        k8 = np.arange(8) - 3.5
        tend = np.zeros((T, N))
        for t in range(7, T):
            tend[t] = (k8 @ linc[t - 7:t + 1]) / (k8 @ k8)
        viz = np.stack([_vizinhos(D, r52[t]) for t in range(T)])
        at12 = pd.DataFrame(np.nan_to_num(anom['temp_med'])).rolling(12, min_periods=1).mean().values
        ap12 = pd.DataFrame(np.nan_to_num(anom['precip_med'])).rolling(12, min_periods=1).mean().values
        e12 = pd.Series(enso).rolling(12, min_periods=1).mean().values
        lpop = np.log(D.POP)
        S = np.stack([r52, r104, r156, u8, u4, tend, viz, at12, ap12, np.repeat(e12[:, None], N, 1), lpop], -1)
        m_, s_ = S[:c + 1].reshape(-1, S.shape[-1]).mean(0), S[:c + 1].reshape(-1, S.shape[-1]).std(0) + 1e-6
        self.S = ((S - m_) / s_).astype(np.float32)                          # (T, N, d_static)
        # âncora sazonal: média de log(incidência) na mesma semana do ano, usando só semanas ≤ origem
        cm = np.zeros((T, N))
        self.ult_woy = np.zeros((T, 53), dtype=int)
        for w in range(1, 53):
            idx = np.where(D.woyc == w)[0]
            idx_obs = idx[idx <= D.ultimo_obs]
            v = linc[idx_obs]
            cm[idx_obs] = np.cumsum(v, 0) / np.arange(1, len(idx_obs) + 1)[:, None]
            pos = np.searchsorted(idx_obs, t_all, side='right') - 1
            self.ult_woy[:, w] = np.where(pos >= 0, idx_obs[np.clip(pos, 0, None)], -1)
        self.CM = cm
        self.pop_log = np.log(D.POP / 1e5).astype(np.float32)
        self.Yc = np.nan_to_num(D.Y).astype(np.float32)
        self.d_in, self.d_static, self.d_fut = self.X.shape[-1], self.S.shape[-1], 4

    def futuro(self, o):
        """variáveis conhecidas na origem o para os horizontes 1..H: (H, N, d_fut) e o offset log(μ0) (H, N)."""
        D, H = self.D, self.H
        t = o + np.arange(1, H + 1)
        w = D.woyc[t]
        anc = self.CM[self.ult_woy[o, w]]                                   # (H, N)
        a = 2 * np.pi * w / 52
        F = np.stack([np.repeat(np.sin(a)[:, None], D.N, 1), np.repeat(np.cos(a)[:, None], D.N, 1),
                      (anc - 1.5) / 1.5, np.repeat((np.arange(1, H + 1) / H)[:, None], D.N, 1)], -1)
        off = np.log(np.expm1(anc) + 0.05) + self.pop_log[np.minimum(t, D.T - 1)]
        return F.astype(np.float32), off.astype(np.float32)

    def lote(self, origens, lim_alvo):
        X = np.stack([self.X[o - self.L + 1:o + 1] for o in origens])       # (B, L, N, d)
        S = np.stack([self.S[o] for o in origens])
        F, OFF = zip(*[self.futuro(o) for o in origens])
        t = np.array(origens)[:, None] + np.arange(1, self.H + 1)[None]
        Y = self.Yc[np.minimum(t, self.D.T - 1)]
        M = (t <= lim_alvo)[..., None] & np.ones((1, 1, self.D.N), bool)
        return X, S, np.stack(F), np.stack(OFF), Y, M


def construir_modelo_A(N, d_in, d_static, d_fut, H, hid=64, emb=16, K=2, drop=0.1):
    torch = _torch()
    nn = torch.nn

    class AVWGCN(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.w = nn.Parameter(torch.empty(emb, K, cin, cout))
            self.b = nn.Parameter(torch.zeros(emb, cout))
            nn.init.xavier_uniform_(self.w.view(emb * K, cin, cout))

        def forward(self, x, E, sup):                                        # x (B,N,cin); sup (K,N,N)
            W = torch.einsum('nd,dkio->nkio', E, self.w)
            xg = torch.einsum('knm,bmc->bknc', sup, x).permute(0, 2, 1, 3)   # (B,N,K,cin)
            return torch.einsum('bnki,nkio->bno', xg, W) + E @ self.b

    class Celula(nn.Module):
        def __init__(self, cin):
            super().__init__()
            self.gate, self.upd = AVWGCN(cin + hid, 2 * hid), AVWGCN(cin + hid, hid)

        def forward(self, x, h, E, sup):
            z, r = torch.split(torch.sigmoid(self.gate(torch.cat([x, h], -1), E, sup)), hid, -1)
            hc = torch.tanh(self.upd(torch.cat([x, z * h], -1), E, sup))
            return r * h + (1 - r) * hc

    class ModeloA(nn.Module):
        def __init__(self):
            super().__init__()
            self.E = nn.Parameter(torch.randn(N, emb) * 0.1)
            self.cel = Celula(d_in)
            self.proj_s = nn.Linear(d_static, 32)
            self.h_emb = nn.Embedding(H + 1, 16)
            self.dec = nn.Sequential(nn.Linear(hid + 32 + 16 + d_fut, 128), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))
            nn.init.zeros_(self.dec[-1].weight)
            nn.init.zeros_(self.dec[-1].bias)

        def forward(self, X, S, F):
            B, L_ = X.shape[:2]
            A_ = torch.softmax(torch.relu(self.E @ self.E.T), 1)
            sup = torch.stack([torch.eye(N, device=X.device, dtype=A_.dtype), A_])
            h = torch.zeros(B, N, hid, device=X.device, dtype=X.dtype)
            for l in range(L_):
                h = self.cel(X[:, l], h, self.E, sup)
            ctx = torch.cat([h, torch.relu(self.proj_s(S))], -1)             # (B,N,hid+32)
            ctx = ctx[:, None].expand(B, H, N, ctx.shape[-1])
            he = self.h_emb(torch.arange(1, H + 1, device=X.device))[None, :, None].expand(B, H, N, 16)
            out = self.dec(torch.cat([ctx, he, F], -1))
            return out[..., 0], out[..., 1]                                  # delta (log μ − âncora), log r

    return ModeloA()


def nb_nll_torch(log_mu, log_r, y):
    torch = _torch()
    log_mu, log_r, y = log_mu.float().clamp(-10, 14), log_r.float().clamp(-4, 8), y.float()
    r = torch.exp(log_r)
    lrm = torch.logaddexp(log_r, log_mu)
    return (torch.lgamma(r) + torch.lgamma(y + 1) - torch.lgamma(y + r) - r * (log_r - lrm) - y * (log_mu - lrm))


class ModeloA:
    """Versão A. Treina com origens sintéticas (toda semana vira origem; alvos 1..H semanas à frente, só ≤ corte).
    Parada antecipada numa 'temporada de validação' (origens perto da SE25 do ano s-1, alvos até o corte) e,
    se retreino_final, retreina com todos os dados pelo mesmo número de épocas."""

    def __init__(self, L=104, H=68, hid=64, emb=16, lote=8, lr=2e-3, epocas_max=60, paciencia=8, frac_origens=1.0,
                 peso_h_curto=0.3, n_sementes=2, retreino_final=True, n_traj=1000, sig_nac=0.30, sig_uf=0.30,
                 sig_unid=0.20, seed=0, dispositivo=None, pasta_ckpt=None, max_origens_epoca=None):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def _treina(self, P, lim_alvo, origens_val, lim_val, epocas_fixas=None, semente=0):
        torch = _torch()
        dev = self.dispositivo
        torch.manual_seed(semente)
        np.random.seed(semente)
        m = construir_modelo_A(P.D.N, P.d_in, P.d_static, P.d_fut, self.H, self.hid, self.emb).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        wh = torch.tensor(np.where(np.arange(1, self.H + 1) >= 16, 1.0, self.peso_h_curto), device=dev)[None, :, None]
        origens = np.arange(max(self.L, 156), lim_alvo)
        usa_amp = dev.startswith('cuda') and torch.cuda.is_bf16_supported()   # L4/A100 sim; T4 roda em float32

        def passo(orig, treino=True, lim=lim_alvo):
            X, S, F, OFF, Y, M = P.lote(orig, lim)
            X, S, F = (torch.tensor(a, device=dev) for a in (X, S, F))
            OFF, Y, M = torch.tensor(OFF, device=dev), torch.tensor(Y, device=dev), torch.tensor(M, device=dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=usa_amp):
                d, lr_ = m(X, S, F)
            nll = nb_nll_torch(OFF + d.float(), lr_.float(), Y)
            w = wh * M.float()
            return (nll * w).sum() / w.sum().clamp(min=1)

        melhor, melhor_ep, sem, hist = np.inf, 0, 0, []
        n_ep = epocas_fixas or self.epocas_max
        for ep in range(1, n_ep + 1):
            m.train()
            rng = np.random.default_rng(semente * 1000 + ep)
            ords = rng.permutation(origens)
            if self.frac_origens < 1:
                ords = ords[:int(len(ords) * self.frac_origens)]
            if self.max_origens_epoca:
                ords = ords[:self.max_origens_epoca]
            perdas = []
            for i in range(0, len(ords), self.lote):
                loss = passo(ords[i:i + self.lote])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                perdas.append(loss.item())
            if epocas_fixas:
                hist.append((ep, np.mean(perdas), np.nan))
                continue
            m.eval()
            with torch.no_grad():
                lv = np.mean([passo(origens_val[i:i + self.lote], lim=lim_val).item()
                              for i in range(0, len(origens_val), self.lote)])
            sched.step(lv)
            hist.append((ep, np.mean(perdas), lv))
            if lv < melhor - 1e-4:
                melhor, melhor_ep, sem = lv, ep, 0
                estado = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                sem += 1
            log(f'      ép {ep:3d} · treino {np.mean(perdas):.4f} · val {lv:.4f} · melhor {melhor:.4f} (ép {melhor_ep})')
            if sem >= self.paciencia:
                break
        if not epocas_fixas:
            m.load_state_dict(estado)
        return m, melhor_ep, pd.DataFrame(hist, columns=['epoca', 'loss_treino', 'loss_val'])

    def prever(self, D, s):
        torch = _torch()
        if self.dispositivo is None:
            self.dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
        rng = np.random.default_rng(self.seed + s)
        c = D.corte(s)
        P = PreparaA(D, c, self.L, self.H)
        c_val = D.corte(s - 1)
        origens_val = np.arange(c_val - 4, c_val + 5)
        datas = datas_temporada(s)
        h_alvo = np.array([D.idx_data[pd.Timestamp(d)] - c for d in datas])
        assert h_alvo.max() <= self.H, 'aumente H'
        mus, rs, self.historicos_ = [], [], []
        for sem in range(self.n_sementes):
            t0 = time.time()
            log(f'   [A] temporada {s} · semente {sem}: treino até {pd.Timestamp(D.datas[c_val]).date()}, validação até o corte')
            m, ep, h = self._treina(P, c_val, origens_val, c, semente=self.seed + sem)
            self.historicos_.append(h.assign(fase='validacao', semente=sem))
            if self.retreino_final:
                log(f'   [A] retreino com todos os dados até {pd.Timestamp(D.datas[c]).date()} por {ep} épocas')
                m, _, h2 = self._treina(P, c, origens_val, c, epocas_fixas=max(ep, 3), semente=self.seed + sem)
                self.historicos_.append(h2.assign(fase='final', semente=sem))
            m.eval()
            with torch.no_grad():
                X, S, F, OFF, _, _ = P.lote([c], c)
                d, lr_ = m(*(torch.tensor(a, device=self.dispositivo) for a in (X, S, F)))
                mu = np.exp(OFF[0] + d[0].float().cpu().numpy())
                r = np.exp(np.clip(lr_[0].float().cpu().numpy(), -4, 8))
            mus.append(mu[h_alvo - 1])
            rs.append(r[h_alvo - 1])
            log(f'   [A] semente {sem} pronta em {(time.time() - t0) / 60:.1f} min')
            if self.pasta_ckpt:
                torch.save(m.state_dict(), os.path.join(self.pasta_ckpt, f'modeloA_s{s}_sem{sem}.pt'))
        self.mu_, self.r_ = mus, rs
        # trajetórias com fatores multiplicativos compartilhados (nacional, por UF e por unidade) → soma por UF
        M_ = self.n_traj
        uf_i = np.searchsorted(D.ufs, D.uf_of)
        k = rng.integers(0, len(mus), M_)
        e = (rng.normal(0, self.sig_nac, (M_, 1)) + rng.normal(0, self.sig_uf, (M_, len(D.ufs)))[:, uf_i]
             + rng.normal(0, self.sig_unid, (M_, D.N)))
        mu = np.stack(mus)[k] * np.exp(e)[:, None, :]
        r = np.stack(rs)[k]
        traj = rng.negative_binomial(np.maximum(r, 1e-3), r / (r + mu)).astype(np.float32)
        self.traj_unid_ = traj                       # (M, W, N) por unidade — usado pela competição
        tu = soma_para_uf(D, traj)
        return datas, arruma_quantis(quantis_de_amostras(tu))


# =====================================================================================================================
# ETAPA 3 · calibração conformal (em escala log, com temporadas passadas)
# =====================================================================================================================
def pares_calibracao(prev_df, verdade, modelo, s, D_uf):
    """quantis e observados das temporadas anteriores a s, usando só semanas observáveis até o corte de s."""
    c_data = pd.Timestamp(D_uf.datas[D_uf.corte(s)])
    g = prev_df[(prev_df.modelo == modelo) & (prev_df.season < s) & (prev_df.date <= c_data)]
    if len(g) == 0:
        return None, None
    qs, ys = [], []
    for s_, gg in g.groupby('season'):
        datas = np.sort(gg.date.unique())
        ufs = np.sort(gg.uf_code.unique())
        q = de_df(gg, datas, ufs)
        y = verdade.y(datas)[:, np.searchsorted(verdade.ufs, ufs)]
        ok = ~np.isnan(y)
        qs.append(q[ok])
        ys.append(y[ok])
    return np.concatenate(qs), np.concatenate(ys)


def _interval_score(lo, up, y, a):
    return (up - lo) + 2 / a * (lo - y) * (y < lo) + 2 / a * (y - up) * (y > up)


GRADE_F = np.round(np.arange(0.4, 2.01, 0.05), 2)
GRADE_B = np.round(np.arange(-0.6, 0.61, 0.05), 2)


def calibrar_conformal(q_alvo, q_cal, y_cal, metodo='score', desloca_mediana=False, eps=0.05, **_):
    """Recalibração dos quantis usando temporadas passadas (nunca a temporada prevista).
    metodo='score' (padrão): em escala log1p, escolhe (i) um deslocamento b da mediana que minimiza o erro absoluto e
      (ii) para cada nível (50/80/90/95) o fator f das meias-larguras que minimiza o INTERVAL SCORE em contagens —
      o mesmo critério do WIS. É uma recalibração "ótima pelo score", mais segura que forçar a cobertura nominal.
    metodo='cobertura': conformal clássico — f = quantil do escore de não conformidade (cobertura nominal).
    Devolve (q_ajustado, fatores, deslocamento)."""
    lq_c, ly = np.log1p(q_cal), np.log1p(y_cal)
    if not desloca_mediana:
        b = 0.0
    elif metodo == 'score':
        b = float(GRADE_B[np.argmin([np.mean(np.abs(np.expm1(lq_c[:, 4] + b_) - y_cal)) for b_ in GRADE_B])])
    else:
        b = float(np.median(ly - lq_c[:, 4])) * 0.5
    lq_c, lq_a = lq_c + b, np.log1p(q_alvo) + b
    out, fatores = lq_a.copy(), {}
    n = len(ly)
    for i, (nivel, a) in enumerate(zip(NIVEIS, ALFAS)):
        m = lq_c[:, 4]
        dl, du = np.maximum(m - lq_c[:, 3 - i], eps), np.maximum(lq_c[:, 5 + i] - m, eps)
        if metodo == 'score':
            sc = [np.mean(_interval_score(np.expm1(m - f * dl), np.expm1(m + f * du), y_cal, a)) for f in GRADE_F]
            f = float(GRADE_F[int(np.argmin(sc))])
        else:
            nc = np.maximum((m - ly) / dl, (ly - m) / du)
            f = float(np.quantile(nc, min(1.0, (1 - a) * (1 + 1 / n))))
        fatores[nivel] = f
        ma = lq_a[..., 4]
        out[..., 3 - i] = ma - f * np.maximum(ma - lq_a[..., 3 - i], eps)
        out[..., 5 + i] = ma + f * np.maximum(lq_a[..., 5 + i] - ma, eps)
    for i in range(1, 4):   # aninhamento
        out[..., 3 - i] = np.minimum(out[..., 3 - i], out[..., 4 - i])
        out[..., 5 + i] = np.maximum(out[..., 5 + i], out[..., 4 + i])
    return arruma_quantis(np.expm1(out)), fatores, b


def aplicar_conformal(prev_df, verdade, modelos, D_uf, sufixo='+conf', min_temporadas=2, **kw):
    """cria o modelo '<m>+conf' para cada temporada que tenha ≥ min_temporadas anteriores para calibrar."""
    novos, rel = [], []
    for m in modelos:
        for s in sorted(prev_df[prev_df.modelo == m].season.unique()):
            q_cal, y_cal = pares_calibracao(prev_df, verdade, m, s, D_uf)
            n_temp = prev_df[(prev_df.modelo == m) & (prev_df.season < s)].season.nunique()
            if q_cal is None or n_temp < min_temporadas:
                continue
            g = prev_df[(prev_df.modelo == m) & (prev_df.season == s)]
            datas, ufs = np.sort(g.date.unique()), np.sort(g.uf_code.unique())
            qa, fat, vies = calibrar_conformal(de_df(g, datas, ufs), q_cal, y_cal, **kw)
            novos.append(para_df(qa, datas, ufs, m + sufixo, s))
            rel.append(dict(modelo=m, season=s, n_cal=len(y_cal), desloc_log=vies, **{f'f{k}': v for k, v in fat.items()}))
    return (pd.concat(novos, ignore_index=True) if novos else pd.DataFrame()), pd.DataFrame(rel)


# =====================================================================================================================
# ETAPA 4 · ensemble
# =====================================================================================================================
def _alinha(prev_df, membros, s):
    g = prev_df[(prev_df.season == s) & (prev_df.modelo.isin(membros))]
    g = g.drop_duplicates(['modelo', 'uf_code', 'date'], keep='last')
    datas, ufs = np.sort(g.date.unique()), np.sort(g.uf_code.unique())
    arr = np.stack([de_df(g[g.modelo == m], datas, ufs) for m in membros])
    return datas, ufs, arr


def pesos_ensemble(prev_df, verdade, membros, s, D_uf, encolhe=0.5):
    """pesos que minimizam o WIS relativo médio (log) nas temporadas anteriores a s, encolhidos para pesos iguais."""
    from scipy.optimize import minimize
    c_data = pd.Timestamp(D_uf.datas[D_uf.corte(s)])
    temps = [t for t in sorted(prev_df.season.unique()) if t < s]
    blocos = []
    for t in temps:
        try:
            datas, ufs, arr = _alinha(prev_df[prev_df.date <= c_data], membros, t)
        except Exception:
            continue
        if np.isnan(arr).any():
            continue
        y = verdade.y(datas)[:, np.searchsorted(verdade.ufs, ufs)]
        blocos.append((arr, y))
    M = len(membros)
    if not blocos:
        return np.ones(M) / M

    def perda(z):
        w = np.exp(z) / np.exp(z).sum()
        tot = []
        for arr, y in blocos:
            q = np.tensordot(w, arr, 1)
            ws = np.nanmean(wis(q, y), 0)
            wr = np.nanmean(np.stack([wis(a, y) for a in arr]).mean(0), 0)   # referência: média dos membros
            tot.append(np.nanmean(np.log(ws / wr)))
        return np.mean(tot)

    z = minimize(perda, np.zeros(M), method='Nelder-Mead', options=dict(maxiter=400)).x
    w = np.exp(z) / np.exp(z).sum()
    return encolhe * w + (1 - encolhe) / M


def ensemble(prev_df, membros, nome, metodo='media', verdade=None, D_uf=None, encolhe=0.5):
    out, pesos = [], []
    for s in sorted(prev_df.season.unique()):
        ok = [m for m in membros if ((prev_df.modelo == m) & (prev_df.season == s)).any()]
        if len(ok) < len(membros):
            continue
        datas, ufs, arr = _alinha(prev_df, membros, s)
        if np.isnan(arr).any():                           # algum membro sem todas as semanas/UFs: não inventa zeros
            print(f'ensemble {nome}: temporada {s} pulada (membro com semanas ou UFs faltando)')
            continue
        if metodo == 'media':
            q = arr.mean(0)
        elif metodo == 'mediana':
            q = np.median(arr, 0)
        else:
            w = pesos_ensemble(prev_df, verdade, membros, s, D_uf, encolhe)
            pesos.append(dict(season=s, **dict(zip(membros, np.round(w, 3)))))
            q = np.tensordot(w, arr, 1)
        out.append(para_df(arruma_quantis(q), datas, ufs, nome, s))
    if not out:
        return pd.DataFrame(columns=['modelo', 'season', 'uf_code', 'date'] + QCOLS), pd.DataFrame(pesos)
    return pd.concat(out, ignore_index=True), pd.DataFrame(pesos)


# =====================================================================================================================
# utilidades de saída
# =====================================================================================================================
def exportar_submissao(prev_df, modelo, pasta):
    """um CSV por temporada no formato da plataforma (date, pred, lower_x, upper_x) + coluna adm_1 (código da UF)."""
    os.makedirs(pasta, exist_ok=True)
    arqs = []
    for s, g in prev_df[prev_df.modelo == modelo].groupby('season'):
        g = g.sort_values(['uf_code', 'date'])
        out = g[['date'] + QCOLS].copy()
        out.insert(1, 'adm_1', g.uf_code.values)
        out['date'] = pd.to_datetime(out.date).dt.strftime('%Y-%m-%d')
        nome = f'submissao_{modelo.replace("+", "_").replace(":", "_")}_temporada{s}.csv'
        out.to_csv(os.path.join(pasta, nome), index=False)
        arqs.append(nome)
    return arqs


def grafico_leque(prev_df, verdade, modelos, s, ufs=None, arquivo=None):
    import matplotlib.pyplot as plt
    ufs = ufs if ufs is not None else verdade.ufs
    n = len(ufs)
    cols = 6
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3.2 * cols, 2.4 * np.ceil(n / cols)), squeeze=False)
    cores = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k, u in enumerate(ufs):
        ax = axs.flat[k]
        for i, m in enumerate(modelos):
            g = prev_df[(prev_df.modelo == m) & (prev_df.season == s) & (prev_df.uf_code == u)].sort_values('date')
            if len(g) == 0:
                continue
            ax.fill_between(g.date, g.lower_90, g.upper_90, color=cores[i], alpha=.15)
            ax.plot(g.date, g.pred, color=cores[i], lw=1.2, label=m)
            datas = g.date.values
        y = verdade.y(datas)[:, list(verdade.ufs).index(u)]
        ax.plot(datas, y, 'k.', ms=3)
        ax.set_title(f'UF {u}', fontsize=9)
        ax.tick_params(labelsize=6)
    for ax in list(axs.flat)[n:]:
        ax.axis('off')
    axs.flat[0].legend(fontsize=6)
    fig.suptitle(f'Temporada {s}/{s + 1}: mediana e IC90 (pontos = observado)')
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=110)
    return fig
