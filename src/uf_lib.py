"""
Modelos DIRETO POR UF (plano em _MAPA_DO_PROJETO/06_NIVEL_ESTADUAL_PESQUISA_E_PLANO.md).

Usa a mesma régua e a mesma execução da competição (comp_lib.rodar): cada modelo recebe um contexto e a temporada s,
só usa dados até a SE25 do ano s e devolve quantis ou trajetórias por unidade. Aqui a unidade é a UF (26).

  Fase 0 · ContextoUF (o "nível de modelagem" passa a ser a UF) e Porta (roda qualquer modelo da competição por UF)
  Fase 2 · TabelaUF: covariáveis da tabela unificada (UF/dados), alinhadas ao calendário do V4, sem vazamento
  Fase 3 · GBMDireto (LightGBM / XGBoost / CatBoost quantílicos, com covariáveis, pooling regiões+UFs, ablação,
           aumento de dados), DoisEstagios (tamanho e pico da temporada + forma das temporadas passadas),
           AutoARIMA (auto.arima completo em Python ou R, Fourier com K pelo AICc; ARIMAX com EXOG_FUTURAS;
           opção d = 0 sem drift) e VARModelo (VAR por macrorregião, VAR espacial e BVAR de Minnesota; VARX com ENSO)
  Fase 4 · NFPooled (TFT, NHITS, TiDE, TSMixerx, DeepAR, PatchTST treinados em regiões + UFs, com e sem as exógenas
           futuras), STID (MLP com identidade espacial e temporal) e redes de grafo nas 26 UFs: GCNGRU (T-GCN),
           A3T-GCN (+ atenção temporal), STGCN (convolução temporal com GLU) e MTGNN (grafo aprendido + mix-hop)
  Fase 5 · Chronos2 (com e sem covariáveis) e TimesFM (opcional, experimental)
  Fase 6 · aumento de dados (no GBMDireto) e calibração da largura (ajusta_largura, usada no CompUF4)
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

import v4_lib as V
import comp_lib as L
import comp_modelos as CM
from comp_lib import Modelo, Saida

Q = V.Q
QCOLS = V.QCOLS
TUNING = [2018, 2019, 2020, 2021]          # escolhas (hiperparâmetros, variáveis, membros do ensemble)
TESTE = [2022, 2023, 2024, 2025]           # olhado uma vez, no fim (S1–S4)
TODAS_UF = TUNING + TESTE


# =====================================================================================================================
# Fase 0 · contexto e porta
# =====================================================================================================================
class ContextoUF(L.Contexto):
    """Igual ao Contexto da competição, mas o nível de modelagem (ctx.Dr) é a UF: a soma até a UF vira identidade."""

    def __init__(self, pasta_dados, pasta_ref=None, pasta_uf=None):
        self.Du = V.Dados(pasta_dados, 'uf')
        self.Dr = self.Du
        self.ver_uf = V.Verdade(self.Du)
        self.ver_reg = self.ver_uf
        self.ref = V.ler_referencias_imdc3(pasta_ref, self.Du.ufs) if pasta_ref else None
        self.pasta_dados, self.pasta_uf = pasta_dados, pasta_uf
        self._Dreg, self._tab = None, None
        assert np.allclose(self.Du.A_uf, np.eye(self.Du.N)), 'no nível UF a matriz de soma deve ser a identidade'

    @property
    def Dreg(self):                     # regiões de saúde, só para os modelos com pooling
        if self._Dreg is None:
            self._Dreg = V.Dados(self.pasta_dados, 'regiao')
        return self._Dreg

    @property
    def tab(self):
        if self._tab is None:
            self._tab = TabelaUF(self.pasta_uf, self.Du)
        return self._tab


class Porta(Modelo):
    """Roda um modelo da competição direto por UF (nome + '_UF')."""

    def __init__(self, base, sufixo='_UF', nome=None):
        self.base = base
        self.nome = nome or base.nome + sufixo
        self.familia, self.origem = base.familia, base.origem
        self.dependencia = base.dependencia
        self.usa_gpu = getattr(base, 'usa_gpu', False)

    def prever(self, ctx, s):
        return self.base.prever(ctx, s)


def portados_cpu(inla_kw=None):
    """os modelos da competição que rodam em CPU (INLA só se inla_kw for dado)."""
    ms = [CM.Climatologia(), CM.CanalEndemico(), CM.ARLongo(), CM.Surge(), CM.ARIMAFourier(), CM.ETSMSTL(),
          CM.ProphetModelo(), CM.LightGBMDireto(), CM.XGBoostQuantis(), CM.RenovacaoS(), CM.TamanhoForma()]
    if inla_kw is not None:
        ms += [CM.ModeloINLA('bb', **inla_kw), CM.ModeloINLA('ghr', **inla_kw)]
    return [Porta(m) for m in ms]


def portados_gpu(cfg_agcrn=None):
    return [Porta(CM.AGCRN(**(cfg_agcrn or {}))), Porta(CM.Chronos())]


# =====================================================================================================================
# Fase 2 · covariáveis da tabela unificada (sem vazamento: tudo é função do passado de cada semana)
# =====================================================================================================================
def _anom_passado(X, woy, min_anos=3):
    """anomalia padronizada contra as ocorrências ANTERIORES da mesma semana do ano (expansiva, sem olhar o futuro)."""
    out = np.full_like(X, np.nan, dtype=np.float64)
    for w in range(1, 53):
        idx = np.where(woy == w)[0]
        if len(idx) < 2:
            continue
        v = X[idx].astype(np.float64)
        ok = ~np.isnan(v)
        vz = np.where(ok, v, 0)
        cs, cs2, cn = np.cumsum(vz, 0), np.cumsum(vz ** 2, 0), np.cumsum(ok, 0)
        z = np.zeros((1, v.shape[1]))
        n_ant = np.vstack([z, cn[:-1]])
        m = np.vstack([z, cs[:-1]]) / np.maximum(n_ant, 1)
        var = np.vstack([z, cs2[:-1]]) / np.maximum(n_ant, 1) - m ** 2
        out[idx] = np.where(n_ant >= min_anos, (v - m) / np.sqrt(np.maximum(var, 1e-6)), np.nan)
    return out


def _rol(X, k):
    return pd.DataFrame(X).rolling(k, min_periods=1).mean().values


def _incl8(X):
    k8 = np.arange(8) - 3.5
    out = np.full_like(X, np.nan, dtype=np.float64)
    for t in range(7, len(X)):
        out[t] = (k8 @ np.nan_to_num(X[t - 7:t + 1])) / (k8 @ k8)
    return out


class TabelaUF:
    """Matrizes (T, 26) alinhadas a D.datas (o calendário do V4, que se estende 80 semanas além do último dado)."""

    def __init__(self, pasta_uf, D):
        self.D = D
        t = pd.read_parquet(os.path.join(pasta_uf, 'tabela_uf_semanal.parquet'))
        t['date'] = pd.to_datetime(t['date'])
        nao = {'uf_code', 'uf', 'date', 'epiweek', 'woy', 'season', 'macro', 'bioma_predominante', 'koppen_predominante'}
        cols = [c for c in t.columns if c not in nao and pd.api.types.is_numeric_dtype(t[c])]
        p = t.set_index(['date', 'uf_code'])[cols]
        idx = pd.MultiIndex.from_product([pd.to_datetime(D.datas), D.unidades])
        A = p.reindex(idx).values.reshape(len(D.datas), D.N, len(cols))
        self.M = {c: A[..., j].astype(np.float64) for j, c in enumerate(cols)}
        assert np.allclose(np.nan_to_num(self.M['casos'][:D.ultimo_obs + 1]), np.nan_to_num(D.Y[:D.ultimo_obs + 1])), \
            'tabela UF desalinhada dos casos do V4'
        w = D.woyc
        M = self.M
        # derivadas (todas causais)
        for v in ('temp_med', 'precip_med', 'rel_humid_med', 'vpd', 'temp_min'):
            a = _anom_passado(M[v], w)
            M[f'an_{v}_12'] = _rol(a, 12)
            M[f'an_{v}_26'] = _rol(a, 26)
        M['enso_12'] = _rol(M['enso'], 12)
        e = np.nan_to_num(M['enso'])
        M['enso_delta'] = _rol(e, 4) - np.vstack([np.full((12, D.N), np.nan), _rol(e, 4)[:-12]])
        M['w1_tend8'] = _incl8(M['w1_log_inc_dengue'])
        for v in ('pct_mun_com_casos', 'pct_pop_mun_com_casos', 'log_inc_chiku'):
            M[f'{v}_8'] = _rol(M[v], 8)
        M['afya_4'] = pd.DataFrame(M['afya_log_acessos_100k']).rolling(4, min_periods=1).mean().values
        self.temporada = pd.read_parquet(os.path.join(pasta_uf, 'tabela_uf_temporada.parquet'))
        self.seas = pd.read_parquet(os.path.join(pasta_uf, 'seas5_uf.parquet'))
        self.seas['mes_emissao'] = pd.to_datetime(self.seas['mes_emissao'])
        self._cubo_seas()

    def _cubo_seas(self):
        s = self.seas
        self.meses = pd.period_range(s.mes_emissao.min(), s.mes_emissao.max(), freq='M')
        pos_u = {u: i for i, u in enumerate(self.D.unidades)}
        mi = ((s.mes_emissao.dt.year - self.meses[0].year) * 12 + s.mes_emissao.dt.month - self.meses[0].month).values
        ui = s.uf_code.map(pos_u).values
        ok = ~np.isnan(ui.astype(float))
        self.cubo = {}
        for v in ('temp_med', 'precip_tot'):
            c = np.full((self.D.N, len(self.meses), 6), np.nan)
            c[ui[ok].astype(int), mi[ok], s.meses_a_frente.values[ok] - 1] = s[v].values[ok]
            self.cubo[v] = c

    def seas_alvo(self, o, t):
        """SEAS5 para a semana-alvo t, usando só a última emissão disponível na origem o (NaN se o alvo passa de 6 meses).
        Emissão disponível: o mês de o se o dia ≥ 7 (o ECMWF publica no início do mês), senão o mês anterior."""
        do = pd.Timestamp(self.D.datas[o])
        emis = do.to_period('M') - (0 if do.day >= 7 else 1)
        e = (emis - self.meses[0]).n
        alvo = (pd.to_datetime(self.D.datas[np.minimum(t, self.D.T - 1)]) + pd.Timedelta(days=3)).to_period('M')
        lead = np.array([(a - emis).n + 1 for a in alvo])                    # 1 = mês de emissão
        out = np.full((len(t), self.D.N, 2), np.nan)
        if 0 <= e < len(self.meses):
            ok = (lead >= 1) & (lead <= 6)
            for j, v in enumerate(('temp_med', 'precip_tot')):
                out[ok, :, j] = self.cubo[v][:, e, lead[ok] - 1].T
        return out


ESTATICAS = ['pct_pop_bioma_Amazônia', 'pct_pop_bioma_Caatinga', 'pct_pop_bioma_Cerrado', 'pct_pop_bioma_Mata_Atlântica',
             'pct_pop_bioma_Pampa', 'pct_pop_bioma_Pantanal', 'pct_pop_koppen_grupo_A', 'pct_pop_koppen_grupo_B',
             'pct_pop_koppen_grupo_C', 'pct_pop_mun_100k', 'grau_vizinhanca']
GRUPOS = {
    'clima': ['an_temp_med_12', 'an_precip_med_12', 'an_rel_humid_med_12', 'an_vpd_12', 'an_temp_min_12',
              'an_temp_med_26', 'an_precip_med_26'],
    'oceano': ['enso', 'iod', 'pdo', 'enso_12', 'enso_delta'],
    'vizinhos': ['w1_log_inc_dengue', 'w1c_log_inc_dengue', 'w2_log_inc_dengue', 'w1_velocidade',
                 'w1_imunidade_ewm', 'w1_tend8'],
    'imunidade_fisica': ['imunidade_ewm', 'velocidade', 'aceleracao'],
    'espalhamento': ['pct_mun_com_casos_8', 'pct_pop_mun_com_casos_8'],
    'chik': ['log_inc_chiku_8'],
    'afya': ['afya_4'],
    'estaticas': ESTATICAS,
    'dispersao': ['temp_med_dp_mun', 'precip_med_dp_mun'],
    'seas5': [],                                                             # tratado à parte (depende do alvo)
}


# Conjunto "selecionadas" (definido em avaliar_variaveis.py SÓ com as temporadas de afinação 2015–2021): fora a
# previsão sazonal SEAS5 como variável de origem (piorou sozinha), chikungunya, Afya e dispersão (sem evidência).
# O nível/tendência recente e as taxas de ataque já entram em todas as variantes pelas variáveis de origem do V4.
SELECIONADAS = ['clima', 'oceano', 'vizinhos', 'imunidade_fisica', 'espalhamento', 'estaticas']


def colunas_grupos(grupos):
    if grupos in (None, 'nenhum'):
        return [], False
    gs = list(GRUPOS) if grupos == 'todas' else (SELECIONADAS if grupos == 'selecionadas' else list(grupos))
    cols = [c for g in gs for c in GRUPOS[g]]
    return cols, 'seas5' in gs


# =====================================================================================================================
# Fase 3 · árvores quantílicas diretas (origens sintéticas), com covariáveis e pooling
# =====================================================================================================================
def _painel(D, c, H, passo_o, passo_h, cov=None, tab=None, usa_seas=False, o_ini=160):
    """linhas (origem o, horizonte h, unidade): [variáveis da origem (V4) | do alvo (V4) | covariáveis | SEAS5 do alvo].
    alvo = log1p(incidência) − âncora sazonal (média da mesma semana do ano, só com semanas ≤ origem)."""
    P = V.PreparaA(D, c, L=104, H=H)
    linc = np.nan_to_num(D.LINC)
    hs_all = np.arange(1, H + 1)

    def feats(o, hs):
        F, _ = P.futuro(o)
        partes = [np.repeat(P.S[o][None], len(hs), 0), F[hs - 1]]
        if cov is not None:
            partes.append(np.repeat(cov[o][None], len(hs), 0))
        if usa_seas:
            partes.append(tab.seas_alvo(o, o + hs))
        return np.concatenate(partes, -1)

    X, y, hh = [], [], []
    for o in range(o_ini, c, passo_o):
        hs = hs_all[::passo_h]
        hs = hs[o + hs <= c]
        if len(hs) == 0:
            continue
        t = o + hs
        anc = P.CM[P.ult_woy[o, D.woyc[t]]]
        Xo = feats(o, hs)
        X.append(Xo.reshape(-1, Xo.shape[-1]))
        y.append((linc[t] - anc).ravel())
        hh.append(np.repeat(hs, D.N))
    Xp = feats(c, hs_all)                                                    # (H, N, k)
    ancp = np.stack([P.CM[P.ult_woy[c, D.woyc[c + k]]] for k in hs_all])     # (H, N)
    return (np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.float32), np.concatenate(hh),
            Xp.astype(np.float32), ancp)


def _ajusta_quantis(motor, X, y, Xp, seed, n_estimators):
    if motor == 'lgb':
        import lightgbm as lgb
        prm = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.7,
                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, random_state=seed)
        return np.stack([lgb.LGBMRegressor(objective='quantile', alpha=float(a), **prm).fit(X, y).predict(Xp)
                         for a in Q], -1)
    if motor == 'xgb':
        import xgboost as xgb
        try:
            import torch
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            dev = 'cpu'
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=Q, tree_method='hist', device=dev,
                             n_estimators=n_estimators, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed)
        return m.fit(X, y).predict(Xp)
    if motor == 'cat':
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(loss_function='MultiQuantile:alpha=' + ','.join(str(float(a)) for a in Q),
                              iterations=n_estimators * 2, depth=6, learning_rate=0.05, random_seed=seed, verbose=0,
                              thread_count=-1)
        return m.fit(X, y).predict(Xp)
    raise ValueError(motor)


class GBMDireto(Modelo):
    """Regressão quantílica direta multi-horizonte por UF (estilo PURDUE/Flusion).
    motor: 'lgb' | 'xgb' | 'cat'. grupos: 'todas' | lista de GRUPOS | None (só casos).
    sem: um grupo a tirar (ablação). pool: treina também com as 435 regiões (só as variáveis comuns aos dois níveis).
    aumento: fração de linhas duplicadas com o alvo deslocado em log(g), g ~ U(1,5; 4) (temporadas maiores que o
    histórico, Fase 6)."""
    familia, origem = 'aprendizado de máquina', 'PURDUE / Flusion (GBM quantílico global)'

    def __init__(self, motor='lgb', grupos='todas', sem=None, pool=False, aumento=0.0, n_estimators=300,
                 passo_o=2, passo_h=2, max_linhas=800_000, nome=None):
        self.motor, self.grupos, self.sem, self.pool, self.aumento = motor, grupos, sem, pool, aumento
        self.n_estimators, self.passo_o, self.passo_h, self.max_linhas = n_estimators, passo_o, passo_h, max_linhas
        rot = 'casos' if grupos in (None, 'nenhum') else (grupos if isinstance(grupos, str) else '+'.join(grupos))
        self.nome = nome or (f'GBM_{motor}_{rot}' + (f'_sem_{sem}' if sem else '') + ('_pool' if pool else '')
                             + ('_aum' if aumento else '') + '_UF')

    def prever(self, ctx, s):
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        H = 68
        rng = np.random.default_rng(s)
        if self.pool:                       # só variáveis comuns (V4) + indicador de nível
            Xu, yu, _, Xp, ancp = _painel(D, c, H, self.passo_o, self.passo_h)
            Dr = ctx.Dreg
            Xr, yr, _, _, _ = _painel(Dr, Dr.corte(s), H, self.passo_o * 3, self.passo_h * 2)
            X = np.vstack([np.c_[Xu, np.ones(len(Xu))], np.c_[Xr, np.zeros(len(Xr))]])
            y = np.concatenate([yu, yr])
            hh = np.concatenate([np.zeros(len(yu)), np.zeros(len(yr))])
            Xp = np.concatenate([Xp, np.ones((*Xp.shape[:2], 1), np.float32)], -1)
        else:
            cols, usa_seas = colunas_grupos(self.grupos)
            if self.sem:
                cols = [c_ for c_ in cols if c_ not in GRUPOS[self.sem]]
                usa_seas = usa_seas and self.sem != 'seas5'
            cov = np.stack([ctx.tab.M[c_] for c_ in cols], -1) if cols else None
            X, y, hh, Xp, ancp = _painel(D, c, H, self.passo_o, self.passo_h, cov, ctx.tab, usa_seas)
        if self.aumento:
            sel = rng.random(len(y)) < self.aumento
            X = np.vstack([X, X[sel]])
            y = np.concatenate([y, y[sel] + np.log(rng.uniform(1.5, 4.0, sel.sum())).astype(np.float32)])
        if len(y) > self.max_linhas:
            k = rng.choice(len(y), self.max_linhas, replace=False)
            X, y = X[k], y[k]
        pr = _ajusta_quantis(self.motor, X, y, Xp.reshape(-1, Xp.shape[-1]), s, self.n_estimators)
        pr = np.sort(pr.reshape(*Xp.shape[:2], len(Q)), -1)[h - 1]
        lq = ancp[h - 1][..., None] + pr
        return Saida(quant=CM._para_contagem(lq, CM._pop_c(D, s)[None, :, None]),
                     info={'linhas_treino': int(len(y)), 'variaveis': int(X.shape[1])})


def ablacao(motor='lgb'):
    """um modelo sem cada grupo (rodar só nas temporadas de afinação)."""
    return [GBMDireto(motor, 'todas', sem=g) for g in GRUPOS]


# =====================================================================================================================
# Fase 3 · dois estágios: tamanho e pico da temporada (GBM com variáveis da SE25) + forma das temporadas passadas
# =====================================================================================================================
EXTRAS_TEMP = ['vizp_w1_ar1_parcial', 'vizp_w1_inc_ult8', 'vizp_w1_tend8', 'vizp_w1c_ar1_parcial',
               'pct_mun_com_casos_ult8', 'pct_pop_mun_com_casos_ult8'] + ESTATICAS


class DoisEstagios(Modelo):
    """Estágio 1: LightGBM quantílico prevê log(incidência da temporada) e a semana do pico a partir das variáveis
    conhecidas na SE25 (features_temporada do V4 + extras da tabela UF). Treina só com temporadas COMPLETAS antes do
    corte (a temporada s−1 termina na SE40 do ano s, depois da SE25: não entra).
    Estágio 2: cada trajetória sorteia um tamanho (dos quantis), uma semana de pico e a FORMA de uma temporada passada
    da própria UF (normalizada e deslocada para o pico sorteado); ruído binomial negativo."""
    familia, origem = 'híbrido', 'IFGW (características da temporada) + LNCC SURGE (forma)'
    dependencia = 'propria'

    # selecao=True: tira o que piorou na avaliação de temporada (afinação 2015–2021): SEAS5, taxas de ataque das
    # temporadas 2–4 e acumuladas, e os vizinhos por média simples (substituídos pela razão agregada)
    FORA_SELECAO = ['seas_temp_med_anom', 'seas_precip_tot_anom', 'seas_umid_med_anom', 'ar2', 'ar3', 'ar4',
                    'inc_acum_4temp', 'inc_acum_decaimento', 'viz_ar1_parcial', 'viz_ar2', 'viz_inc_ult8', 'viz_tend8']

    def __init__(self, pool=True, selecao=False, M=500, disp=20.0, n_estimators=300, nome=None):
        self.pool, self.selecao, self.M, self.disp, self.n_estimators = pool, selecao, M, disp, n_estimators
        self.nome = nome or ('DoisEstagios' + ('_pool' if pool else '') + ('_sel' if selecao else '') + '_UF')

    @staticmethod
    def _linhas(D, js, c, anom, extras=None):
        F, Y = [], []
        for j in js:
            f = V.features_temporada(D, j, c, anom)
            if extras is not None:
                e = extras[extras.season == j].set_index('uf_code').reindex(D.unidades)
                for col in EXTRAS_TEMP:
                    f[col] = e[col].values
            ii = D.idx_temporada(j)
            y = np.nan_to_num(D.Y[ii])
            F.append(f)
            Y.append(np.c_[np.log1p(y.sum(0) / D.POP[ii[0]] * 1e5), np.argmax(y[:52], 0)])
        return pd.concat(F, ignore_index=True), np.vstack(Y)

    def prever(self, ctx, s):
        import lightgbm as lgb
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        rng = np.random.default_rng(s)
        js = [j for j in range(2011, s) if D.temporada_completa(j, c) and D.idx_temporada(j)[0] > 0]
        anom = V._anomalias_clima(D, c)
        ext = ctx.tab.temporada
        Fu, Yu = self._linhas(D, js, c, anom, ext)
        Fp, _ = self._linhas(D, [s], c, anom, ext)
        if self.pool:
            Dr = ctx.Dreg
            Fr, Yr = self._linhas(Dr, js, c, V._anomalias_clima(Dr, c))
            for col in EXTRAS_TEMP:
                Fr[col] = np.nan
            F, Yt = pd.concat([Fu.assign(nivel_uf=1), Fr.assign(nivel_uf=0)], ignore_index=True), np.vstack([Yu, Yr])
            Fp = Fp.assign(nivel_uf=1)
        else:
            F, Yt = Fu, Yu
        fora = ('uf_code', 'macro') + (tuple(self.FORA_SELECAO) if self.selecao else ())
        cols = [c_ for c_ in F.columns if c_ not in fora and c_ in Fp.columns
                and pd.api.types.is_numeric_dtype(F[c_])]                       # só as comuns aos dois níveis
        X, Xp = F[cols].values.astype(np.float32), Fp[cols].values.astype(np.float32)
        prm = dict(n_estimators=self.n_estimators, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, random_state=s)
        tam = np.sort(np.stack([lgb.LGBMRegressor(objective='quantile', alpha=float(a), **prm).fit(X, Yt[:, 0])
                                .predict(Xp) for a in Q], -1), -1)                       # (N, 9) log1p(inc)
        pk = np.stack([lgb.LGBMRegressor(objective='quantile', alpha=a, **prm).fit(X, Yt[:, 1]).predict(Xp)
                       for a in (0.1, 0.5, 0.9)], -1)                                    # (N, 3) semanas desde a SE41
        # estágio 2: trajetórias
        pop = D.POP[c]
        tot = L.amostrar_quantis((np.expm1(tam) * pop[:, None] / 1e5)[None], self.M)[:, 0, :]   # (M, N) casos
        sd_pk = np.maximum((pk[:, 2] - pk[:, 0]) / (2 * 1.2816), 1.0)
        pico = np.clip(np.round(pk[:, 1][None] + sd_pk[None] * rng.standard_normal((self.M, D.N))), 0, 51).astype(int)
        W = len(idx)
        formas = []
        for j in js:
            y = np.nan_to_num(D.Y[D.idx_temporada(j)][:52])
            ys = np.apply_along_axis(lambda v: np.convolve(v, np.ones(3) / 3, mode='same'), 0, y)
            formas.append((y / np.maximum(y.sum(0), 1), ys.argmax(0)))
        lam = np.zeros((self.M, W, D.N))
        for u in range(D.N):
            jj = rng.integers(0, len(formas), self.M)
            for m in range(self.M):
                f, pk_j = formas[jj[m]]
                d = pico[m, u] - pk_j[u]
                g = np.zeros(52)
                if d >= 0:
                    g[d:] = f[:52 - d, u]
                else:
                    g[:52 + d] = f[-d:, u]
                g = g / max(g.sum(), 1e-9)
                g = np.r_[g, np.repeat(g[-1:], max(0, W - 52))][:W]
                lam[m, :, u] = tot[m, u] * g
        p = self.disp / (self.disp + lam)
        traj = rng.negative_binomial(self.disp, np.clip(p, 1e-9, 1)).astype(np.float32)
        return Saida(traj=traj, info={'temporadas_treino': js, 'linhas_treino': int(len(X))})


# =====================================================================================================================
# Fase 3 · auto.arima (Python e R) e VAR (vários estados ao mesmo tempo)
# =====================================================================================================================
EXOG_FUTURAS = ('enso', 'tanom', 'panom')     # o MESMO conjunto para todo modelo que usa exógenas futuras


def exogenas_futuras(D, c, idx=None):
    """matrizes (T, N). Passado: o observado. Depois da SE25: ENSO persistido (valor da SE25) e anomalias de
    temperatura e chuva da SEAS5 emitida em junho enquanto a defasagem cair em até 6 meses, depois 0 (regra do GHR)."""
    ta, pa = CM._covariaveis_ghr(D, c, idx)
    en = np.nan_to_num(pd.Series(D.OCEANO['enso']).ffill().values)
    en = np.repeat(en[:, None], D.N, 1)
    en[c + 1:] = en[c]
    return {'enso': en, 'tanom': ta, 'panom': pa}


def _teto_casos(D, c):
    """saneamento comum: nenhuma previsão passa de 3× a maior incidência semanal já vista na UF."""
    return 3 * np.nanmax(D.INC[:c + 1], 0) * D.POP[c] / 1e5


class AutoARIMA(Modelo):
    """auto.arima COMPLETO (Hyndman & Khandakar 2008): d (teste KPSS), p, q e drift escolhidos pelo AICc; a
    sazonalidade anual entra por Fourier e o número de termos K = 1..Kmax também é escolhido pelo AICc
    (regressão harmônica dinâmica, a forma recomendada no FPP3 para sazonalidade longa como a semanal).
    motor='py': statsforecast (porte do forecast::auto.arima); motor='r': o próprio forecast::auto.arima no R.
    exog=EXOG_FUTURAS → ARIMAX (regressão com erros ARIMA)."""
    familia, origem = 'estatístico', 'auto.arima (forecast / statsforecast) + Fourier; ARIMAX com ENSO e clima'

    def __init__(self, motor='py', exog=(), estacionario=False, Kmax=6, anos=8, n_jobs=-1, rscript=None, r_libs=None,
                 pasta_script=None, nome=None):
        self.motor, self.exog, self.est, self.Kmax, self.anos = motor, tuple(exog), estacionario, Kmax, anos
        self.n_jobs = n_jobs
        self.rscript = rscript or os.environ.get('RSCRIPT', 'Rscript')
        self.r_libs = r_libs
        self.pasta_script = pasta_script or os.path.dirname(os.path.abspath(__file__))
        self.nome = nome or f'AutoARIMA{"X" if self.exog else ""}{"_d0" if estacionario else ""}_{motor}_UF'

    def _dados(self, D, c, idx, H):
        ini = max(0, c - self.anos * 52)
        ex = exogenas_futuras(D, c, idx) if self.exog else {}
        ids = D.unidades.astype(str)
        df = pd.DataFrame({'unique_id': np.repeat(ids, c + 1 - ini),
                           'ds': np.tile(pd.to_datetime(D.datas[ini:c + 1]), D.N),
                           'y': np.nan_to_num(D.LINC[ini:c + 1]).T.ravel()})
        dsf = pd.date_range(pd.Timestamp(D.datas[c]) + pd.Timedelta(weeks=1), periods=H, freq='7D')
        fut = pd.DataFrame({'unique_id': np.repeat(ids, H), 'ds': np.tile(dsf, D.N)})
        for v in self.exog:
            df[v] = ex[v][ini:c + 1].T.ravel()
            fut[v] = ex[v][c + 1:c + H + 1].T.ravel()
        return df, fut

    def _py(self, df, fut, H):
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA as AA
        melhor = {}
        for K in range(1, self.Kmax + 1):
            dk, fk = CM._add_fourier(df.copy(), K), CM._add_fourier(fut.copy(), K)
            mod = AA(season_length=1, d=0, allowdrift=False) if self.est else AA(season_length=1)
            sf = StatsForecast(models=[mod], freq='W-SUN', n_jobs=self.n_jobs)
            sf.fit(df=dk)
            uids = getattr(sf, 'uids', None)
            uids = np.sort(dk.unique_id.unique()) if uids is None else np.asarray(uids)
            info = {}
            for i, u in enumerate(uids):
                md = sf.fitted_[i, 0].model_
                arma = md.get('arma', [np.nan] * 7)                     # (p, q, P, Q, s, d, D)
                info[str(u)] = (float(md.get('aicc', np.inf)), f'{arma[0]},{arma[5]},{arma[1]}')
            fc = sf.predict(h=H, X_df=fk, level=CM.NIVEIS_INT)
            fc = fc.reset_index() if 'unique_id' not in fc.columns else fc
            fc['unique_id'] = fc['unique_id'].astype(str)
            for u, (a, ordem) in info.items():
                if np.isfinite(a) and (u not in melhor or a < melhor[u][0]):
                    melhor[u] = (a, fc[fc.unique_id == u], f'K={K} ARIMA({ordem})')
        return pd.concat([v[1] for v in melhor.values()]), {u: v[2] for u, v in melhor.items()}

    def _r(self, df, fut, H):
        import shutil
        import subprocess
        import tempfile
        tmp = tempfile.mkdtemp(prefix='autoarima_')
        try:
            df.drop(columns='ds').rename(columns={'unique_id': 'uid'}).to_csv(os.path.join(tmp, 'dados.csv'), index=False)
            fut.drop(columns='ds').rename(columns={'unique_id': 'uid'}).to_csv(os.path.join(tmp, 'futuro.csv'), index=False)
            with open(os.path.join(tmp, 'params.txt'), 'w') as f:
                f.write(f'H={H}\nKmax={self.Kmax}\nthreads={os.cpu_count()}\nestacionario={int(self.est)}\n')
            env = dict(os.environ)
            if self.r_libs:
                env['R_LIBS_USER'] = self.r_libs
            p = subprocess.run([self.rscript, os.path.join(self.pasta_script, 'uf_autoarima.R'), tmp],
                               capture_output=True, text=True, env=env)
            if p.returncode != 0 or not os.path.exists(os.path.join(tmp, 'previsao.csv')):
                raise RuntimeError(f'auto.arima (R) falhou:\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}')
            out = pd.read_csv(os.path.join(tmp, 'previsao.csv'))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        out['unique_id'] = out.uid.astype(str)
        dsf = pd.date_range(pd.Timestamp(fut.ds.min()), periods=H, freq='7D')
        out['ds'] = dsf[out.h.values - 1]
        fc = out.rename(columns={'media': 'AutoARIMA'})
        for n in CM.NIVEIS_INT:
            fc[f'AutoARIMA-lo-{n}'], fc[f'AutoARIMA-hi-{n}'] = out[f'lo{n}'], out[f'hi{n}']
        u1 = out.drop_duplicates('unique_id').set_index('unique_id')
        Ks = {u: f'K={int(r.K)} ARIMA({r.ordem})' for u, r in u1.iterrows()}
        return fc, Ks

    def prever(self, ctx, s):
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        H = int(h.max())
        df, fut = self._dados(D, c, idx, H)
        fc, Ks = self._py(df, fut, H) if self.motor == 'py' else self._r(df, fut, H)
        q = CM._sf_para_quantis(fc, 'AutoARIMA', D, h, CM._pop_c(D, s))
        q = np.minimum(q, _teto_casos(D, c)[None, :, None])
        return Saida(quant=q, info={'modelo_escolhido': {str(k): str(v) for k, v in Ks.items()}})


def _clim_semana(D, c, anos):
    """média de log1p(incidência) por semana do ano, nas temporadas até o corte → (53, N)."""
    ini = max(0, c - anos * 52)
    Lm, w = np.nan_to_num(D.LINC[ini:c + 1]), D.woyc[ini:c + 1]
    cl = np.zeros((53, D.N))
    for k in range(1, 53):
        cl[k] = Lm[w == k].mean(0)
    return cl


def _desenho(Y, p, X=None):
    """regressores de um VAR(p): [1, exógenas, Y_{t-1}, ..., Y_{t-p}] (linhas t = p..T-1)."""
    T = len(Y)
    partes = [np.ones((T - p, 1))] + ([X[p:]] if X is not None else []) + [Y[p - j - 1:T - j - 1] for j in range(p)]
    return np.column_stack(partes)


def _desempacota(coef, N, p, kx):
    c0 = coef[0]
    B = coef[1:1 + kx].T if kx else None
    A = [coef[1 + kx + j * N:1 + kx + (j + 1) * N].T for j in range(p)]
    return A, c0, B


def _var_ols(Y, p, X=None):
    Dz = _desenho(Y, p, X)
    coef = np.linalg.lstsq(Dz, Y[p:], rcond=None)[0]
    res = Y[p:] - Dz @ coef
    kx = 0 if X is None else X.shape[1]
    return (*_desempacota(coef, Y.shape[1], p, kx), res)


def _var_bic(Y, pmax, X=None):
    """ordem do VAR pelo BIC (Schwarz), com a mesma amostra efetiva para todo p (como o vars::VARselect)."""
    melhor = None
    N = Y.shape[1]
    for p in range(1, pmax + 1):
        Yp = Y[pmax - p:]
        Xp = None if X is None else X[pmax - p:]
        *_, res = _var_ols(Yp, p, Xp)
        n = len(res)
        S = res.T @ res / n
        k = N * (1 + (0 if X is None else X.shape[1]) + N * p)
        bic = np.linalg.slogdet(S + 1e-10 * np.eye(N))[1] + np.log(n) / n * k
        if melhor is None or bic < melhor[0]:
            melhor = (bic, p)
    return melhor[1]


def _simula_var(A, c0, B, hist, Xf, Sig, H, M, rng):
    """trajetórias de um VAR(p): hist (p, N) com o último valor no fim; Xf (H, k) exógenas futuras."""
    p, N = len(A), len(c0)
    Lc = np.linalg.cholesky(Sig + 1e-8 * np.eye(N))
    buf = np.repeat(hist[None], M, 0)
    out = np.zeros((M, H, N))
    for t in range(H):
        m = c0[None] + sum(buf[:, -j - 1] @ A[j].T for j in range(p))
        if B is not None:
            m = m + Xf[t] @ B.T
        y = m + rng.standard_normal((M, N)) @ Lc.T
        out[:, t] = y
        buf = np.concatenate([buf[:, 1:], y[:, None]], 1)
    return out


class VARModelo(Modelo):
    """VAR nas anomalias de log1p(incidência) (série menos a média da semana do ano), todas as UFs juntas.
    tipo='macro'   : um VAR por grande região (N, NE, SE, S, CO), ordem pelo BIC (≤ pmax).
    tipo='espacial': cada UF depende das próprias defasagens e da média das UFs vizinhas (VAR restrito/espacial:
                     2p coeficientes por equação em vez de 26p — Lütkepohl, VAR com restrições).
    tipo='bvar'    : VAR com as 26 UFs e prior de Minnesota (Litterman): coeficientes encolhidos para 0, com a
                     própria defasagem 1 perto de `delta`; os cruzados encolhem mais (theta).
    exog=True      : VARX com o ENSO (persistido depois da SE25). Incerteza: trajetórias com a covariância dos
                     resíduos (correlação entre UFs e no tempo saem do próprio modelo)."""
    familia, origem = 'estatístico multivariado', 'VAR / VAR espacial / BVAR Minnesota (Lütkepohl 2005)'
    dependencia = 'propria'

    def __init__(self, tipo='macro', exog=False, pmax=4, p_bvar=3, anos=8, M=500, lam=0.2, theta=0.5, delta=0.8,
                 nome=None):
        self.tipo, self.exog, self.pmax, self.p_bvar, self.anos, self.M = tipo, exog, pmax, p_bvar, anos, M
        self.lam, self.theta, self.delta = lam, theta, delta
        base = {'macro': 'VAR_macro', 'espacial': 'VAR_espacial', 'bvar': 'BVAR_Minnesota'}[tipo]
        self.nome = nome or base + ('X' if exog else '') + '_UF'

    def _bvar(self, Y, p, X):
        N = Y.shape[1]
        kx = 0 if X is None else X.shape[1]
        Dz = _desenho(Y, p, X)
        s = np.array([np.std(_var_ols(Y[:, [i]], p)[-1]) + 1e-6 for i in range(N)])
        coef = np.zeros((Dz.shape[1], N))
        for i in range(N):
            b0 = np.zeros(Dz.shape[1])
            v = np.full(Dz.shape[1], 100.0)                                   # constante e exógenas: vagas
            for j in range(p):
                for k in range(N):
                    pos = 1 + kx + j * N + k
                    v[pos] = (self.lam / (j + 1)) ** 2 * (1.0 if k == i else self.theta ** 2 * s[i] ** 2 / s[k] ** 2)
                    if k == i and j == 0:
                        b0[pos] = self.delta
            Vi = 1 / v
            Prec = Dz.T @ Dz / s[i] ** 2 + np.diag(Vi)
            coef[:, i] = np.linalg.solve(Prec, Dz.T @ Y[p:, i] / s[i] ** 2 + Vi * b0)
        res = Y[p:] - Dz @ coef
        return (*_desempacota(coef, N, p, kx), res)

    def _espacial(self, Y, p, X, Wn):
        N = Y.shape[1]
        kx = 0 if X is None else X.shape[1]
        Yv = Y @ Wn.T
        T = len(Y)
        A = [np.zeros((N, N)) for _ in range(p)]
        c0, B = np.zeros(N), (np.zeros((N, kx)) if kx else None)
        res = np.zeros((T - p, N))
        for i in range(N):
            cols = [np.ones(T - p)] + ([X[p:, k] for k in range(kx)] if kx else []) \
                + [Y[p - j - 1:T - j - 1, i] for j in range(p)] + [Yv[p - j - 1:T - j - 1, i] for j in range(p)]
            Dz = np.column_stack(cols)
            b = np.linalg.lstsq(Dz, Y[p:, i], rcond=None)[0]
            res[:, i] = Y[p:, i] - Dz @ b
            c0[i] = b[0]
            if kx:
                B[i] = b[1:1 + kx]
            for j in range(p):
                A[j][i, i] += b[1 + kx + j]
                A[j][i] += b[1 + kx + p + j] * Wn[i]
        return A, c0, B, res

    def prever(self, ctx, s):
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        H = int(h.max())
        rng = np.random.default_rng(s)
        cl = _clim_semana(D, c, self.anos)
        ini = max(0, c - self.anos * 52)
        Y = np.nan_to_num(D.LINC[ini:c + 1]) - cl[D.woyc[ini:c + 1]]
        X = Xf = None
        if self.exog:
            en = exogenas_futuras(D, c, idx)['enso'][:, :1]
            X, Xf = en[ini:c + 1], en[c + 1:c + H + 1]
        sim = np.zeros((self.M, H, D.N))
        ordens = {}
        if self.tipo == 'macro':
            macro = D.uf_of // 10
            for g in np.unique(macro):
                cols = np.flatnonzero(macro == g)
                p = _var_bic(Y[:, cols], self.pmax, X)
                A, c0, B, res = _var_ols(Y[:, cols], p, X)
                Sig = res.T @ res / max(len(res) - res.shape[1] * p - 1, 1)
                sim[..., cols] = _simula_var(A, c0, B, Y[-p:, cols], Xf, Sig, H, self.M, rng)
                ordens[int(g)] = p
        else:
            if self.tipo == 'espacial':
                Wn = D.ADJ / np.maximum(D.ADJ.sum(1, keepdims=True), 1)
                p = self.p_bvar
                A, c0, B, res = self._espacial(Y, p, X, Wn)
            else:
                p = self.p_bvar
                A, c0, B, res = self._bvar(Y, p, X)
            Sig = res.T @ res / len(res)
            sim = _simula_var(A, c0, B, Y[-p:], Xf, Sig, H, self.M, rng)
            ordens['todas'] = p
        lt = sim + cl[D.woyc[c + 1:c + H + 1]][None]
        casos = np.expm1(np.clip(lt, 0, None)) * D.POP[c][None, None] / 1e5
        casos = np.minimum(casos, _teto_casos(D, c)[None, None])
        return Saida(traj=casos[:, h - 1].astype(np.float32), info={'ordem_p': ordens})


# =====================================================================================================================
# Fase 4 · redes do neuralforecast treinadas com regiões + UFs (pooling), prevendo as UFs
# =====================================================================================================================
SUPORTE_EXOG = {'TFT': (True, True), 'NHITS': (True, True), 'TiDE': (True, True), 'TSMixerx': (True, True),
                'DeepAR': (True, True), 'PatchTST': (False, False)}            # (exógenas futuras, estáticas)
FOURIER_NF = ['s1', 'c1', 's2', 'c2']


class NFPooled(Modelo):
    """Redes do neuralforecast treinadas com regiões + UFs, prevendo as UFs. Exógenas futuras: Fourier e, com
    covariaveis=True, o MESMO conjunto do ARIMAX/VARX/INLA/Chronos-2 (ENSO persistido + anomalias de clima com SEAS5)."""
    familia, origem = 'deep learning', 'neuralforecast (TFT, N-HiTS, TiDE, TSMixer, DeepAR, PatchTST)'
    usa_gpu = True

    def __init__(self, arq='TFT', pool=True, covariaveis=True, max_steps=1500, input_size=104, anos=12,
                 sementes=(1, 2, 3), nome=None):
        self.arq, self.pool, self.cov, self.max_steps = arq, pool, covariaveis, max_steps
        self.input_size, self.anos, self.sementes = input_size, anos, tuple(sementes)
        self.futr = FOURIER_NF + (list(EXOG_FUTURAS) if covariaveis else [])
        self.nome = nome or f'{arq}{"_pool" if pool else ""}{"" if covariaveis else "_semcov"}_UF'

    def _dados(self, D, c, pref, H):
        ini = max(0, c - self.anos * 52)
        ds = pd.to_datetime(D.datas[ini:c + 1])
        ids = np.array([f'{pref}{u}' for u in D.unidades])
        df = pd.DataFrame({'unique_id': np.repeat(ids, len(ds)), 'ds': np.tile(ds, D.N),
                           'y': np.nan_to_num(D.LINC[ini:c + 1]).T.ravel()})
        df = CM._add_fourier(df, 2)
        dsf = pd.date_range(pd.Timestamp(D.datas[c]) + pd.Timedelta(weeks=1), periods=H, freq='7D')
        fut = CM._add_fourier(pd.DataFrame({'unique_id': np.repeat(ids, H), 'ds': np.tile(dsf, D.N)}), 2)
        if self.cov:
            ex = exogenas_futuras(D, c)
            for v in EXOG_FUTURAS:
                df[v] = ex[v][ini:c + 1].T.ravel()
                fut[v] = ex[v][c + 1:c + H + 1].T.ravel()
        st = pd.DataFrame({'unique_id': ids, 'nivel_uf': float(pref == 'u'), 'log_pop': np.log(D.POP[c])})
        return df, fut, st

    def _modelo(self, H, n_series, seed):
        import neuralforecast.models as nfm
        from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
        fu, st = SUPORTE_EXOG[self.arq]
        kw = dict(h=H, input_size=self.input_size, max_steps=self.max_steps, random_seed=seed, val_check_steps=50,
                  early_stop_patience_steps=6)
        if fu:
            kw['futr_exog_list'] = self.futr
        if st:
            kw['stat_exog_list'] = ['nivel_uf', 'log_pop']
        mq = MQLoss(level=CM.NIVEIS_INT)
        if self.arq == 'DeepAR':
            return nfm.DeepAR(loss=DistributionLoss('Normal', level=CM.NIVEIS_INT), valid_loss=mq, lstm_n_layers=2,
                              lstm_hidden_size=64, scaler_type='standard', **kw)
        if self.arq == 'TSMixerx':
            return nfm.TSMixerx(n_series=n_series, loss=mq, scaler_type='robust', **kw)
        if self.arq == 'TFT':
            return nfm.TFT(loss=mq, hidden_size=64, scaler_type='robust', **kw)
        return getattr(nfm, self.arq)(loss=mq, scaler_type='robust', **kw)

    def prever(self, ctx, s):
        from neuralforecast import NeuralForecast
        import logging
        for lg in ('pytorch_lightning', 'lightning.pytorch', 'lightning_fabric'):
            logging.getLogger(lg).setLevel(logging.ERROR)
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        H = 68
        partes = [self._dados(D, c, 'u', H)]
        if self.pool:
            Dr = ctx.Dreg
            partes.append(self._dados(Dr, Dr.corte(s), 'r', H))
        df = pd.concat([p[0] for p in partes], ignore_index=True)
        fut = pd.concat([p[1] for p in partes], ignore_index=True)
        st = pd.concat([p[2] for p in partes], ignore_index=True)
        n_series = df.unique_id.nunique()
        ids_uf = [f'u{u}' for u in D.unidades]
        fu, sx = SUPORTE_EXOG[self.arq]
        res = []
        for seed in self.sementes:
            m = self._modelo(H, n_series, seed)
            nf = NeuralForecast(models=[m], freq='W-SUN')
            cols = ['unique_id', 'ds', 'y'] + (self.futr if fu else [])
            nf.fit(df=df[cols], static_df=st if sx else None, val_size=H)
            fc = nf.predict(futr_df=fut[['unique_id', 'ds'] + self.futr]) if fu else nf.predict()
            fc = fc.reset_index() if 'unique_id' not in fc.columns else fc
            nome = type(m).__name__
            if f'{nome}-median' in fc.columns:
                fc = fc.drop(columns=[nome], errors='ignore').rename(columns={f'{nome}-median': nome})
            fc = fc[fc.unique_id.isin(ids_uf)].copy()
            fc['_u'] = fc.unique_id.map({k: i for i, k in enumerate(ids_uf)})
            fc = fc.sort_values(['_u', 'ds'])

            def col(x):
                return fc[x].values.reshape(D.N, H).T
            lo = {n: col(f'{nome}-lo-{n}') for n in CM.NIVEIS_INT}
            hi_ = {n: col(f'{nome}-hi-{n}') for n in CM.NIVEIS_INT}
            res.append(CM._quantis_de_intervalos(col(nome), lo, hi_))           # (H, N, 9) em log1p(inc)
        lq = np.sort(np.median(np.stack(res), 0), -1)
        teto = np.log1p(3 * np.nanmax(D.INC[:c + 1], 0))
        lq = np.minimum(lq, teto[None, :, None])[h - 1]
        return Saida(quant=CM._para_contagem(lq, CM._pop_c(D, s)[None, :, None]), info={'series_treino': n_series})


# =====================================================================================================================
# Fase 4 · STID e GCN-GRU (PyTorch, treinados aqui mesmo com perda quantílica)
# =====================================================================================================================
def _amostras_janela(D, c, L=104, H=68, passo=1, o_ini=160, canais=None):
    """origens o (≤ c−1): entrada = canais dinâmicos do V4 nas L semanas (0 = log1p(inc) padronizado; depois
    anomalias de clima, calendário e ENSO); variáveis de origem (V4);
    alvo = log1p(inc) − âncora nas H semanas seguintes (máscara onde o alvo passa do corte)."""
    P = V.PreparaA(D, c, L=L, H=H)
    linc = np.nan_to_num(D.LINC)
    origens = np.arange(max(o_ini, L), c, passo)
    t = origens[:, None] + np.arange(1, H + 1)[None]
    w = D.woyc[np.minimum(t, D.T - 1)]
    anc = np.stack([P.CM[P.ult_woy[o, w[i]]] for i, o in enumerate(origens)])    # (B, H, N)
    Y = (linc[np.minimum(t, D.T - 1)] - anc).astype(np.float32)
    Mk = np.repeat((t <= c)[..., None], D.N, -1)
    Xd = P.X if canais is None else P.X[..., canais]
    X = np.stack([Xd[o - L + 1:o + 1] for o in origens]).astype(np.float32)      # (B, L, N, d)
    S = np.stack([P.S[o] for o in origens]).astype(np.float32)
    wo = D.woyc[origens]
    tp = c + np.arange(1, H + 1)
    ancp = np.stack([P.CM[P.ult_woy[c, D.woyc[k]]] for k in tp])                 # (H, N)
    pred = (Xd[c - L + 1:c + 1][None].astype(np.float32), P.S[c][None].astype(np.float32), np.array([D.woyc[c]]))
    return X, S, wo, Y, Mk, pred, ancp


def _pinball(pred, y, m, q):
    e = y[..., None] - pred
    l = torch_max(q * e, (q - 1) * e)
    return (l * m[..., None]).sum() / (m.sum() * len(q) + 1e-9)


def torch_max(a, b):
    import torch
    return torch.maximum(a, b)


def _treina(modelo, lotes_fn, n, epocas, lr, seed, dev):
    import torch
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    qt = torch.tensor(Q, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)
    for _ in range(epocas):
        modelo.train()
        for b in np.array_split(rng.permutation(n), max(1, n // 256)):
            entradas, y, m = lotes_fn(b)
            loss = _pinball(modelo(*entradas), y, m, qt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
        sch.step()
    return modelo


class STID(Modelo):
    """MLP com identidade espacial (embedding da unidade) e temporal (embedding da semana do ano) — Shao et al. 2022.
    Com pool=True treina com as 26 UFs + 435 regiões (cada uma é uma 'unidade' com embedding próprio)."""
    familia, origem = 'deep learning', 'STID (identidade espaço-temporal; empata com STGNNs)'
    usa_gpu = True

    def __init__(self, pool=True, d=64, epocas=30, lr=2e-3, sementes=(1, 2, 3), L=104, nome=None):
        self.pool, self.d, self.epocas, self.lr, self.sementes, self.L = pool, d, epocas, lr, tuple(sementes), L
        self.nome = nome or ('STID_pool_UF' if pool else 'STID_UF')

    def prever(self, ctx, s):
        import torch
        import torch.nn as nn
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        H = 68
        bases = [(D, D.corte(s), 1)] + ([(ctx.Dreg, ctx.Dreg.corte(s), 2)] if self.pool else [])
        Xs, Ss, Ws, Ys, Ms, Ns = [], [], [], [], [], []
        off = 0
        for Dx, cx, passo in bases:
            X, S, wo, Y, Mk, pred, ancp = _amostras_janela(Dx, cx, self.L, H, passo, canais=[0])
            B, _, N, _ = X.shape
            Xs.append(X[..., 0].transpose(0, 2, 1).reshape(B * N, self.L))          # só log-incidência
            Ss.append(S.reshape(B * N, -1))
            Ws.append(np.repeat(wo, N))
            Ys.append(Y.transpose(0, 2, 1).reshape(B * N, H))
            Ms.append(Mk.transpose(0, 2, 1).reshape(B * N, H).astype(np.float32))
            Ns.append(np.tile(np.arange(N) + off, B))
            if Dx is D:
                pred_uf, anc_uf = pred, ancp
            off += N
        X, S, Wo, Y, Mk, No = (np.concatenate(a) for a in (Xs, Ss, Ws, Ys, Ms, Ns))
        n_nos, d_s = off, S.shape[1]
        d = self.d

        class Rede(nn.Module):
            def __init__(self):
                super().__init__()
                self.x, self.s = nn.Linear(X.shape[1], d), nn.Linear(d_s, d)
                self.no, self.w = nn.Embedding(n_nos, d), nn.Embedding(54, d)
                self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(4 * d, 4 * d), nn.ReLU(), nn.Dropout(0.1),
                                                        nn.Linear(4 * d, 4 * d)) for _ in range(3)])
                self.out = nn.Linear(4 * d, H * len(Q))

            def forward(self, x, s_, no, w):
                z = torch.cat([self.x(x), self.s(s_), self.no(no), self.w(w)], -1)
                for b in self.mlp:
                    z = z + b(z)
                return self.out(z).view(-1, H, len(Q))

        T_ = lambda a, tp=torch.float32: torch.tensor(a, dtype=tp, device=dev)   # noqa: E731
        Xt, St, Nt, Wt, Yt, Mt = T_(X), T_(S), T_(No, torch.long), T_(Wo, torch.long), T_(Y), T_(Mk)
        xp = T_(pred_uf[0][0, :, :, 0].T)                                           # (N_uf, L)
        sp, nop = T_(pred_uf[1][0]), T_(np.arange(D.N), torch.long)
        wp = T_(np.repeat(pred_uf[2], D.N), torch.long)
        res = []
        for seed in self.sementes:
            rede = Rede().to(dev)
            _treina(rede, lambda b: ((Xt[b], St[b], Nt[b], Wt[b]), Yt[b], Mt[b]), len(Xt), self.epocas, self.lr, seed, dev)
            rede.eval()
            with torch.no_grad():
                res.append(rede(xp, sp, nop, wp).cpu().numpy())                    # (N, H, 9)
        r = np.sort(np.median(np.stack(res), 0), -1).transpose(1, 0, 2)              # (H, N, 9)
        lq = np.minimum(anc_uf[..., None] + r, np.log1p(3 * np.nanmax(D.INC[:c + 1], 0))[None, :, None])[h - 1]
        return Saida(quant=CM._para_contagem(lq, CM._pop_c(D, s)[None, :, None]), info={'amostras_treino': int(len(Xt))})


class _RedeGrafoUF(Modelo):
    """Base comum das redes de grafo nas 26 UFs: MESMAS entradas (janela de 104 semanas com log-incidência,
    anomalias de clima, calendário e ENSO; variáveis de origem do V4), MESMA perda (quantílica, 9 quantis),
    MESMA saída direta (68 semanas, sem propagar erro passo a passo) e MESMO grafo a priori (fronteira entre UFs).
    Cada subclasse só define a rede em `construir`."""
    familia = 'rede de grafos'
    usa_gpu = True

    def __init__(self, d=48, epocas=60, lr=2e-3, sementes=(1, 2, 3), L=104, nome=None):
        self.d, self.epocas, self.lr, self.sementes, self.L = d, epocas, lr, tuple(sementes), L
        if nome:
            self.nome = nome

    def construir(self, nn, torch, N, d_in, d_s, H, A, d):
        raise NotImplementedError

    def prever(self, ctx, s):
        import torch
        import torch.nn as nn
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        H = 68
        X, S, wo, Y, Mk, pred, ancp = _amostras_janela(D, c, self.L, H, 1)
        A = D.ADJ + np.eye(D.N)
        dg = 1 / np.sqrt(A.sum(1))
        A = (A * dg[:, None] * dg[None]).astype(np.float32)
        T_ = lambda a, tp=torch.float32: torch.tensor(a, dtype=tp, device=dev)   # noqa: E731
        Xt, St, Wt = T_(X), T_(S), T_(wo, torch.long)
        Yt, Mt = T_(Y.transpose(0, 2, 1)), T_(Mk.transpose(0, 2, 1).astype(np.float32))   # (B, N, H)
        res = []
        for seed in self.sementes:
            torch.manual_seed(seed)
            rede = self.construir(nn, torch, D.N, X.shape[-1], S.shape[-1], H, A, self.d).to(dev)
            _treina(rede, lambda b: ((Xt[b], St[b], Wt[b]), Yt[b], Mt[b]), len(Xt), self.epocas, self.lr, seed, dev)
            rede.eval()
            with torch.no_grad():
                res.append(rede(T_(pred[0]), T_(pred[1]), T_(pred[2], torch.long))[0].cpu().numpy())   # (N, H, 9)
        r = np.sort(np.median(np.stack(res), 0), -1).transpose(1, 0, 2)
        lq = np.minimum(ancp[..., None] + r, np.log1p(3 * np.nanmax(D.INC[:c + 1], 0))[None, :, None])[h - 1]
        return Saida(quant=CM._para_contagem(lq, CM._pop_c(D, s)[None, :, None]))


def _cabeca(nn, d, d_s, N, H):
    """embeddings (UF, semana do ano) + variáveis de origem + saída H × 9."""
    class Cab(nn.Module):
        def __init__(self):
            super().__init__()
            self.s, self.no, self.w = nn.Linear(d_s, d), nn.Embedding(N, d), nn.Embedding(54, d)
            self.out = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, H * len(Q)))

        def forward(self, z, s_, w):                       # z (B, N, d)
            z = z + self.s(s_) + self.no.weight[None] + self.w(w)[:, None]
            return self.out(z).view(z.shape[0], N, H, len(Q))
    return Cab()


class GCNGRU(_RedeGrafoUF):
    """T-GCN / A3T-GCN (Zhao et al. 2019; Zhu et al. 2020): GRU por UF na janela + convolução no grafo de fronteiras.
    atencao=True (A3T-GCN): atenção sobre todos os estados do GRU (pesa semanas distantes, como o pico do ano
    anterior), em vez de usar só o último estado."""
    origem = 'T-GCN / A3T-GCN (artigos do Drive)'

    def __init__(self, atencao=False, **kw):
        super().__init__(**kw)
        self.atencao = atencao
        if 'nome' not in kw or kw['nome'] is None:
            self.nome = 'A3TGCN_UF' if atencao else 'GCNGRU_UF'

    def construir(self, nn, torch, N, d_in, d_s, H, A, d):
        att = self.atencao

        class Rede(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(d_in, d, batch_first=True)
                self.g1, self.g2 = nn.Linear(d, d), nn.Linear(d, d)
                self.A = nn.Parameter(torch.tensor(A), requires_grad=False)
                self.a = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
                self.cab = _cabeca(nn, d, d_s, N, H)

            def forward(self, x, s_, w):
                B = x.shape[0]
                seq, z = self.gru(x.permute(0, 2, 1, 3).reshape(B * N, x.shape[1], d_in))
                if att:
                    al = torch.softmax(self.a(seq), 1)                  # (B*N, L, 1)
                    z = (al * seq).sum(1)
                else:
                    z = z[-1]
                z = z.view(B, N, d)
                z = z + torch.relu(self.A @ self.g1(z))
                z = z + torch.relu(self.A @ self.g2(z))
                return self.cab(z, s_, w)
        return Rede()


class STGCN(_RedeGrafoUF):
    """STGCN (Yu, Yin & Zhu, IJCAI 2018): blocos 'sanduíche' convolução temporal com porta (GLU) → convolução no grafo
    → convolução temporal; tudo convolucional (rápido, poucos parâmetros). Adaptado para saída direta quantílica."""
    origem = 'STGCN (artigo do Drive)'
    nome = 'STGCN_UF'

    def construir(self, nn, torch, N, d_in, d_s, H, A, d):
        class TempGLU(nn.Module):
            def __init__(self, ci, co, k=3, dil=1):
                super().__init__()
                self.conv = nn.Conv2d(ci, 2 * co, (k, 1), dilation=(dil, 1))
                self.res = nn.Conv2d(ci, co, 1)
                self.k, self.dil = k, dil

            def forward(self, x):                                  # x (B, C, T, N)
                y = self.conv(x)
                p, q = y.chunk(2, 1)
                corte = (self.k - 1) * self.dil
                return (p + self.res(x)[:, :, corte:]) * torch.sigmoid(q)

        class Bloco(nn.Module):
            def __init__(self, ci, co, dil):
                super().__init__()
                self.t1, self.t2 = TempGLU(ci, co, 3, dil), TempGLU(co, co, 3, dil)
                self.g = nn.Linear(co, co)
                self.A = nn.Parameter(torch.tensor(A), requires_grad=False)
                self.bn = nn.BatchNorm2d(co)

            def forward(self, x):
                x = self.t1(x)                                     # (B, C, T, N)
                x = torch.relu(self.g(torch.einsum('nm,bctm->bctn', self.A, x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
                return self.bn(self.t2(x))

        class Rede(nn.Module):
            def __init__(self):
                super().__init__()
                self.b1, self.b2, self.b3 = Bloco(d_in, d, 1), Bloco(d, d, 4), Bloco(d, d, 8)
                self.cab = _cabeca(nn, d, d_s, N, H)

            def forward(self, x, s_, w):
                z = self.b3(self.b2(self.b1(x.permute(0, 3, 1, 2))))        # (B, d, T', N)
                return self.cab(z.mean(2).permute(0, 2, 1), s_, w)
        return Rede()


class MTGNN(_RedeGrafoUF):
    """MTGNN (Wu et al., KDD 2020 — 'Connecting the Dots'), versão enxuta: grafo APRENDIDO e direcionado
    (embeddings de nó, top-k vizinhos) somado ao grafo de fronteiras como conhecimento prévio; propagação mix-hop
    (mantém parte do sinal original a cada salto) e convoluções temporais dilatadas com porta."""
    origem = 'MTGNN (artigo do Drive)'
    nome = 'MTGNN_UF'

    def __init__(self, k_top=4, beta=0.05, **kw):
        super().__init__(**kw)
        self.k_top, self.beta = k_top, beta

    def construir(self, nn, torch, N, d_in, d_s, H, A, d):
        k_top, beta = self.k_top, self.beta

        class Rede(nn.Module):
            def __init__(self):
                super().__init__()
                self.e1, self.e2 = nn.Embedding(N, 16), nn.Embedding(N, 16)
                self.l1, self.l2 = nn.Linear(16, 16), nn.Linear(16, 16)
                self.A0 = nn.Parameter(torch.tensor(A), requires_grad=False)
                self.ent = nn.Conv2d(d_in, d, 1)
                self.dil = [1, 2, 4, 8, 16]
                self.f = nn.ModuleList([nn.Conv2d(d, d, (2, 1), dilation=(k, 1)) for k in self.dil])
                self.g = nn.ModuleList([nn.Conv2d(d, d, (2, 1), dilation=(k, 1)) for k in self.dil])
                self.mix = nn.ModuleList([nn.Linear(3 * d, d) for _ in self.dil])
                self.cab = _cabeca(nn, d, d_s, N, H)

            def grafo(self):
                m1, m2 = torch.tanh(3 * self.l1(self.e1.weight)), torch.tanh(3 * self.l2(self.e2.weight))
                a = torch.relu(torch.tanh(3 * (m1 @ m2.T - m2 @ m1.T)))
                topo = torch.topk(a, k_top, 1).indices
                mask = torch.zeros_like(a).scatter_(1, topo, 1.0)
                a = a * mask + self.A0
                return a / a.sum(1, keepdim=True)

            def forward(self, x, s_, w):
                Ag = self.grafo()
                z = self.ent(x.permute(0, 3, 1, 2))                       # (B, d, T, N)
                for k, f, g, mx in zip(self.dil, self.f, self.g, self.mix):
                    zp = nn.functional.pad(z, (0, 0, k, 0))                 # convolução causal
                    t = torch.tanh(f(zp)) * torch.sigmoid(g(zp))
                    h1 = beta * t + (1 - beta) * torch.einsum('nm,bctm->bctn', Ag, t)
                    h2 = beta * t + (1 - beta) * torch.einsum('nm,bctm->bctn', Ag, h1)
                    z = z + mx(torch.cat([t, h1, h2], 1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                return self.cab(z[:, :, -1].permute(0, 2, 1), s_, w)
        return Rede()


# =====================================================================================================================
# Fase 5 · modelos de fundação
# =====================================================================================================================
def _quantis_de_niveis(niveis, arr):
    """arr (..., len(niveis)) em log → (..., 9) na ordem de Q: interpola dentro e extrapola as caudas como normal."""
    niveis = np.asarray(niveis, float)
    z_n, z_q = norm.ppf(niveis), norm.ppf(Q)
    out = np.empty(arr.shape[:-1] + (len(Q),))
    for k, zq in enumerate(z_q):
        if zq < z_n[0]:
            i, j = 0, 1
        elif zq > z_n[-1]:
            i, j = len(z_n) - 2, len(z_n) - 1
        else:
            j = int(np.searchsorted(z_n, zq))
            j = min(max(j, 1), len(z_n) - 1)
            i = j - 1
        t = (zq - z_n[i]) / (z_n[j] - z_n[i])
        out[..., k] = arr[..., i] + t * (arr[..., j] - arr[..., i])
    return np.sort(out, -1)


class Chronos2(Modelo):
    """Chronos-2 (Amazon, 2025) zero-shot em log1p(incidência), com atenção de grupo entre as 26 UFs.
    covariaveis=True: passado = anomalias de temperatura e chuva + ENSO; futuro = SEAS5 (6 meses) e ENSO da SE25."""
    familia, origem = 'fundação (zero-shot)', 'Chronos-2 (covariáveis + multivariado)'
    usa_gpu = True

    def __init__(self, covariaveis=False, contexto=520, modelo_hf='amazon/chronos-2', nome=None):
        self.cov, self.contexto, self.modelo_hf = covariaveis, contexto, modelo_hf
        self.nome = nome or ('Chronos2_cov_UF' if covariaveis else 'Chronos2_UF')
        self._pipe = None

    def prever(self, ctx, s):
        import torch
        from chronos import Chronos2Pipeline
        if self._pipe is None:
            self._pipe = Chronos2Pipeline.from_pretrained(self.modelo_hf,
                                                          device_map='cuda' if torch.cuda.is_available() else 'cpu')
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        H = int(h.max())
        ini = max(0, c + 1 - self.contexto)
        ds = pd.to_datetime(D.datas[ini:c + 1])
        dsf = pd.date_range(pd.Timestamp(D.datas[c]) + pd.Timedelta(weeks=1), periods=H, freq='7D')
        ids = D.unidades.astype(str)
        cx = pd.DataFrame({'id': np.repeat(ids, len(ds)), 'timestamp': np.tile(ds, D.N),
                           'target': np.nan_to_num(D.LINC[ini:c + 1]).T.ravel()})
        fut = None
        if self.cov:
            ta, pa = CM._covariaveis_ghr(D, c, idx)
            en = np.nan_to_num(pd.Series(D.OCEANO['enso']).ffill().values)
            en = np.repeat(en[:, None], D.N, 1)
            en[c + 1:] = en[c]
            fut = pd.DataFrame({'id': np.repeat(ids, H), 'timestamp': np.tile(dsf, D.N)})
            for nome, M_ in (('tanom', ta), ('panom', pa), ('enso', en)):
                cx[nome] = M_[ini:c + 1].T.ravel()
                fut[nome] = M_[c + 1:c + H + 1].T.ravel()
        niveis = list(np.round(Q, 3))
        try:
            pr = self._pipe.predict_df(cx, future_df=fut, prediction_length=H, quantile_levels=niveis,
                                       id_column='id', timestamp_column='timestamp', target='target')
        except Exception:                       # níveis fora do suporte do modelo: pede 0,05–0,95 e extrapola
            niveis = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            pr = self._pipe.predict_df(cx, future_df=fut, prediction_length=H, quantile_levels=niveis,
                                       id_column='id', timestamp_column='timestamp', target='target')
        pr = pr.reset_index() if 'id' not in pr.columns else pr
        qcols = {}
        for col in pr.columns:
            try:
                qcols[round(float(col), 3)] = col
            except (TypeError, ValueError):
                pass
        pr['_u'] = pr['id'].astype(str).map({k: i for i, k in enumerate(ids)})
        pr = pr.sort_values(['_u', 'timestamp'])
        arr = np.stack([pr[qcols[round(n, 3)]].values.reshape(D.N, H) for n in niveis], -1)   # (N, H, k)
        lq = _quantis_de_niveis(niveis, arr) if len(niveis) != len(Q) else np.sort(arr, -1)
        lq = np.minimum(lq, np.log1p(3 * np.nanmax(D.INC[:c + 1], 0))[:, None, None])
        lq = np.moveaxis(lq, 0, 1)[h - 1]
        return Saida(quant=CM._para_contagem(lq, CM._pop_c(D, s)[None, :, None]))


class TimesFM(Modelo):
    """OPCIONAL / EXPERIMENTAL: a API do TimesFM mudou na versão 3 (set/2026) e não foi testada aqui.
    Se falhar, o erro vai para erros_*.csv e os outros modelos seguem."""
    familia, origem = 'fundação (zero-shot)', 'TimesFM 3 (Google)'
    usa_gpu = True
    nome = 'TimesFM_UF'

    def prever(self, ctx, s):
        import torch
        from timesfm3 import TimesFM3Evaluator, ModelConfig
        D = ctx.Du
        c, datas, idx, h = CM._horizontes(ctx, s)
        H = int(h.max())
        cfg = ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch', per_core_batch_size=32,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
        fc = TimesFM3Evaluator(cfg)
        ctxs = [np.nan_to_num(D.LINC[max(0, c - 519):c + 1, u]).astype(np.float32) for u in range(D.N)]
        out = list(fc.predict_batch(contexts=ctxs, horizon=H, return_quantiles=True))
        qs = np.stack([np.asarray(o['quantiles'] if isinstance(o, dict) else o[1]).reshape(H, -1) for o in out])
        lq = _quantis_de_niveis(np.arange(1, 10) / 10, qs)                                      # decis → 9 quantis
        lq = np.minimum(lq, np.log1p(3 * np.nanmax(D.INC[:c + 1], 0))[:, None, None])
        lq = np.moveaxis(lq, 0, 1)[h - 1]
        return Saida(quant=CM._para_contagem(lq, CM._pop_c(D, s)[None, :, None]))


# =====================================================================================================================
# Fase 6–7 · calibração da largura e ensemble (usados no CompUF4)
# =====================================================================================================================
def ajusta_largura(prev, lam):
    """multiplica a distância de cada quantil à mediana por lam, em log1p (a mediana não muda)."""
    p = prev.copy()
    lq = np.log1p(np.clip(p[QCOLS].values, 0, None))
    med = lq[:, [4]]
    p[QCOLS] = np.clip(np.expm1(np.sort(med + lam * (lq - med), 1)), 0, None)
    return p


def carregar_uf(pastas, rotulos=None, ids=None, so_completas=True):
    """junta as previsões por UF de várias fontes (pastas de resultados, arquivos de modelos externos, DataFrames)
    com L.carregar_previsoes: padroniza o formato, remove duplicatas e deixa fora previsões incompletas.
    O relatório do que foi descartado fica em L.ULTIMO_RELATORIO."""
    if ids is None:
        raise ValueError('passe ids=ctx.Du.ufs')
    fontes = list(zip(rotulos, pastas)) if rotulos else list(pastas)
    return L.carregar_previsoes(fontes, ids, 'uf', so_completas=so_completas)
