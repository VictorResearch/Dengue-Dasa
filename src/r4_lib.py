"""RODADA 4 · temporada com detector ampliado, forma que depende do tamanho, MTGNN condicionado e ensemble por regime.

Por que (rodada 3, teste 2022–2025, corte de junho):
  - MIX_escala_UF acertou a ALTURA dos picos (erro mediano da altura −4% contra −42% do normal; WIS absoluto 776 contra
    1.343; 91% dos picos dentro do IC90), mas perdeu na régua R × BB (1,19) porque reescalava a curva INTEIRA: a
    entressafra ficou +156% acima do real;
  - a régua R (média geométrica por UF) premia previsões tímidas; por isso a recomendada foi a AJ_continuo (R 0,83),
    que continua 25% abaixo no total e 28% abaixo na altura do pico;
  - o detector acertou a direção (corr. 0,8), mas encolhe os extremos (SC 2023 ×45 contra ×206; RS ×53 contra ×160).

O que esta rodada faz (doc 17 do mapa, literatura):
  ① DETECTOR AMPLIADO: + estabilidade entre surtos e sinais de alerta precoce (Rypdal 2019; Brett 2020), imunidade da
    população reconstruída (McGough 2021), seca (índice tipo SPEI; Lowe 2021), clima observado, previsão SEAS5 já
    emitida no corte, temperatura do mar (IOBW, Atlântico tropical e sudoeste, Niño 3.4 do ERSST v5; Chen 2024,
    Beal 2025) e, se baixado, sorotipo do SINAN (Finch 2025). Métodos: os 3 da R3 + DUAS ETAPAS (P(ano grande) e
    tamanho dado grande/não grande; Francisco 2024) + versões com CALIBRAÇÃO CONFORME por quantil (Mills 2025).
  ② MIX_elastico_UF: o normal (ENS_grafos3_λ) vai até o tamanho do detector, mas cada semana cresce conforme a sua
    ELASTICIDADE histórica (o pico cresce muito nos anos grandes; a entressafra quase nada).
  ③ DoisEstagios_det_UF: tamanho do detector × forma de temporadas passadas PARECIDAS EM TAMANHO (análogos).
  ④ MTGNN_cond_UF (GPU): MTGNN_pesopop retreinada com o tamanho relativo da temporada como entrada, alvo das amostras
    aumentado com temporadas grandes sintéticas (Al Bannoud 2026) e cópias sem condição; na previsão, o tamanho vem
    do detector (cenários). MTGNN_aug_UF = a mesma rede sem condição.
  ⑤ ENS_regime_UF(+cal): pesos dos membros por faixa de P(ano grande) e calibração da largura/posição por faixa,
    escolhidos na afinação com um CRITÉRIO MISTO = ½ R × climatologia + ½ WIS absoluto relativo (o 1º cuida das UFs
    pequenas e da entressafra; o 2º, da altura dos picos).
Sem vazamento: janela expansiva; sinais, métodos, pesos e parâmetros escolhidos só com temporadas ≤ 2021; o teste é
olhado uma vez. Tudo grava em arquivo e retoma de onde parou.
"""
import glob
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, kendalltau

import v4_lib as V
import comp_lib as L
import comp_modelos as CM
import uf_lib as U
import semanal_lib as S
import r2_lib as R2
import r3_lib as R3

Q, QCOLS, ZQ = R3.Q, R3.QCOLS, R3.ZQ
TUNING, TESTE = U.TUNING, U.TESTE
CORTES, LIM_GRANDE = R3.CORTES, R3.LIM_GRANDE
salvar_tabela, json_grava, json_le, carimbo, _grava_pq = R2.salvar_tabela, R2.json_grava, R2.json_le, R2.carimbo, R3._grava_pq


# =====================================================================================================================
# 0 · séries causais (cada semana só usa o próprio passado)
# =====================================================================================================================
def _clim_causal(M, anos=10, min_anos=3):
    """M (T, N) → (média, dp) de M nas mesmas semanas dos `anos` anos anteriores (defasagens de 52 semanas)."""
    T = M.shape[0]
    pilha = np.full((anos,) + M.shape, np.nan)
    for k in range(1, anos + 1):
        pilha[k - 1, 52 * k:] = M[:T - 52 * k]
    n = np.sum(~np.isnan(pilha), 0)
    mu, sd = np.nanmean(pilha, 0), np.nanstd(pilha, 0)
    mu[n < min_anos], sd[n < min_anos] = np.nan, np.nan
    return mu, sd


def _rolsoma(M, k):
    c = np.vstack([np.zeros((1,) + M.shape[1:]), np.nancumsum(M, 0)])
    out = np.full(M.shape, np.nan)
    out[k - 1:] = c[k:] - c[:-k]
    return out


def _z(M, anos=10):
    mu, sd = _clim_causal(M, anos)
    return (M - mu) / (sd + 1e-6)


_CACHE = {}


def series_causais(D):
    """anomalias causais usadas pelo detector (calculadas uma vez por base de dados)."""
    if id(D) in _CACHE:
        return _CACHE[id(D)]
    C = D.CLIMA
    out = {}
    linc = np.nan_to_num(D.LINC)
    mu, _ = _clim_causal(linc)
    out['x'] = linc - np.nan_to_num(mu)                                     # log-incidência dessazonalizada
    tmed, tmin, tmax = (np.asarray(C[k], float) for k in ('temp_med', 'temp_min', 'temp_max'))
    prec = np.asarray(C['precip_med'], float)
    pet = (tmed + 17.8) * np.sqrt(np.clip(tmax - tmin, 0.1, None))        # Hargreaves sem a radiação (cancela no z por semana)
    for k in (13, 26):
        zp, ze = _z(_rolsoma(prec, k)), _z(_rolsoma(pet, k))
        out[f'seca_{k}'] = -(zp - ze) / np.sqrt(2)                          # + = mais seco que o normal (tipo SPEI com sinal trocado)
        out[f'clim_temp_{k}'] = _z(_rolsoma(tmed, k))
        out[f'clim_precip_{k}'] = zp
        if 'rel_humid_med' in C:
            out[f'clim_umid_{k}'] = _z(_rolsoma(np.asarray(C['rel_humid_med'], float), k))
    s13 = pd.DataFrame(out['seca_13'])
    out['seca_max_52'] = s13.rolling(52, min_periods=26).max().values       # seca extrema no último ano (Lowe 2021: 3–5 meses antes)
    _CACHE[id(D)] = out
    return out


# =====================================================================================================================
# ① DETECTOR AMPLIADO · variáveis novas
# =====================================================================================================================
def _ews(x, y, c, W=26, w=13):
    """sinais de alerta precoce na janela [c−W+1, c] de uma UF: x = log-incidência dessazonalizada, y = casos."""
    xs = x[c - W + 1:c + 1]
    if np.isnan(xs).any() or len(xs) < W:
        return {}
    t = np.arange(W)
    r = xs - np.polyval(np.polyfit(t, xs, 1), t)
    ar1 = np.corrcoef(r[:-1], r[1:])[0, 1] if r.std() > 0 else 0.0
    vs, acs = [], []
    for i in range(W - w + 1):
        z = r[i:i + w]
        vs.append(z.var())
        acs.append(np.corrcoef(z[:-1], z[1:])[0, 1] if z.std() > 0 else 0.0)
    ys = y[c - w + 1:c + 1]
    m = max(ys.mean(), 1e-6)
    return dict(ews_ar1=ar1, ews_var=float(np.log(r.var() + 1e-6)),
                ews_tau_var=float(kendalltau(np.arange(len(vs)), vs)[0]) if np.std(vs) > 0 else 0.0,
                ews_tau_ar1=float(kendalltau(np.arange(len(acs)), np.nan_to_num(acs))[0]) if np.std(acs) > 0 else 0.0,
                ews_disp=float(np.log(ys.var() / m + 1e-6)), ews_cv=float(ys.std() / m),
                piso_rel=float(pd.Series(xs).rolling(4).mean().min()))


def _imunidade(D, TAM, s, c):
    """tamanho relativo das temporadas anteriores (s−1 parcial até o corte; s−2… completas) e médias com esquecimento."""
    ant = TAM.loc[[i for i in TAM.index if (R3._completa_ate(D, i) or 10 ** 9) <= c and i < s]]
    if len(ant) < 2:
        return {}
    base = ant.mean()
    anoms = [(ant.loc[i] - base).values for i in sorted(ant.index, reverse=True)][:6]     # s−2, s−3, …
    # temporada s−1 até o corte, contra as anteriores até a mesma semana da temporada
    i1 = D.idx_temporada(s - 1, ate=c)
    k = len(i1)
    parc = lambda j: np.log1p(np.nansum(D.Y[D.idx_temporada(j)[:k]], 0) / D.POP[D.idx_temporada(j)[0]] * 1e5)   # noqa: E731
    p_ant = np.mean([parc(j) for j in ant.index], 0) if k else 0.0
    a1 = parc(s - 1) - p_ant if k else np.zeros(D.N)
    seq = [a1] + anoms
    f = {}
    for tau in (1.0, 3.0):
        w = np.exp(-np.arange(len(seq)) / tau)
        f[f'imun_tau{tau:g}'] = np.log(np.tensordot(w, np.exp(np.vstack(seq)), (0, 0)) / w.sum())
    abs3 = [np.nansum(D.Y[D.idx_temporada(s - 1, ate=c)], 0)] + \
           [np.nansum(D.Y[D.idx_temporada(j)], 0) for j in (s - 2, s - 3) if j in TAM.index]
    f['ataque_abs_3'] = np.log1p(np.sum(abs3, 0) / D.POP[c] * 1e5)
    f['anom_s1_parcial'] = a1
    return f


def _seas5(ctx, D, c):
    """anomalia padronizada da SEAS5 emitida até o corte (meses 1–3 e 4–6 à frente), contra emissões do mesmo mês
    em anos anteriores (só passado)."""
    sz = getattr(getattr(ctx, 'tab', None), 'seas', None)
    if sz is None or not len(sz):
        return {}
    sz = sz.copy()
    sz['mes_emissao'] = pd.to_datetime(sz.mes_emissao)
    dc = pd.Timestamp(D.datas[c]) + pd.Timedelta(days=6)
    em = sz.mes_emissao[sz.mes_emissao <= dc - pd.Timedelta(days=5)].max()
    if pd.isna(em):
        return {}
    f = {}
    for nome, (a, b) in (('1a3', (1, 3)), ('4a6', (4, 6))):
        g = sz[(sz.mes_emissao.dt.month == em.month) & sz.meses_a_frente.between(a, b) & (sz.mes_emissao <= em)]
        g = g.groupby(['uf_code', 'mes_emissao'])[['temp_med', 'precip_tot', 'umid_med']].mean().reset_index()
        for v in ('temp_med', 'precip_tot', 'umid_med'):
            p = g.pivot(index='mes_emissao', columns='uf_code', values=v).reindex(columns=D.unidades)
            if em not in p.index or len(p) < 4:
                continue
            hist = p[p.index < em]
            f[f'seas_{v}_{nome}'] = ((p.loc[em] - hist.mean()) / (hist.std() + 1e-6)).values
    return f


# ----------------------------------------------------------------------------- temperatura do mar (ERSST v5, NOAA)
REGIOES_SST = {                                   # (lat_min, lat_max, lon_min, lon_max) em graus leste 0–360
    'nino34': (-5, 5, 190, 240),
    'iobw': (-20, 20, 40, 110),                   # Oceano Índico inteiro (Chen et al. 2024, Science)
    'tna': (5.5, 23.5, 302.5, 345),               # Atlântico tropical norte
    'tsa': (-20, 0, 330, 370),                    # Atlântico tropical sul (passa do meridiano 0)
    'swa': (-35, -15, 310, 330),                  # Atlântico sudoeste, costa do Sul/Sudeste do Brasil
}
URL_ERSST = 'https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc'
URL_ERSST_MES = 'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ersst.v5.{a}{m:02d}.nc'


def _baixar(url, arq, tentativas=3):
    for t in range(tentativas):
        try:
            urllib.request.urlretrieve(url, arq + '.tmp')
            os.replace(arq + '.tmp', arq)
            return True
        except Exception as e:                    # noqa: BLE001
            print(f'   falhou ({t + 1}/{tentativas}) {url}: {e}', flush=True)
            time.sleep(5 * (t + 1))
    return False


def indices_oceano(pasta_cache, pasta_tmp='/content/ersst', ano_ini=1975):
    """índices mensais de temperatura do mar (°C de anomalia contra os 30 anos ANTERIORES do mesmo mês).
    Grava `oceano_ersst.csv`; se a NOAA estiver fora do ar, devolve None e o detector segue sem eles."""
    arq = os.path.join(pasta_cache, 'oceano_ersst.csv')
    if os.path.exists(arq):
        return pd.read_csv(arq, parse_dates=['mes'])
    import xarray as xr
    os.makedirs(pasta_tmp, exist_ok=True)
    nc = os.path.join(pasta_tmp, 'sst.mnmean.nc')
    if os.path.exists(nc) or _baixar(URL_ERSST, nc):
        ds = xr.open_dataset(nc)
        sst = ds['sst'].sel(time=slice(f'{ano_ini}-01-01', None))
    else:                                         # plano B: um arquivo por mês no NCEI (~130 kB cada)
        arqs = []
        hoje = pd.Timestamp.today()
        for a in range(ano_ini, hoje.year + 1):
            for m in range(1, 13):
                if pd.Timestamp(a, m, 1) > hoje - pd.Timedelta(days=35):
                    break
                f = os.path.join(pasta_tmp, f'ersst.v5.{a}{m:02d}.nc')
                if os.path.exists(f) or _baixar(URL_ERSST_MES.format(a=a, m=m), f, 2):
                    arqs.append(f)
        if not arqs:
            print('⚠️ ERSST indisponível: o detector segue sem a temperatura do mar.')
            return None
        sst = xr.open_mfdataset(arqs, combine='by_coords')['sst'].squeeze(drop=True)
    lat, lon = sst['lat'].values, sst['lon'].values % 360
    out = {}
    for nome, (la0, la1, lo0, lo1) in REGIOES_SST.items():
        mlat = (lat >= la0) & (lat <= la1)
        mlon = ((lon >= lo0) & (lon <= lo1)) if lo1 <= 360 else ((lon >= lo0) | (lon <= lo1 - 360))
        sub = sst.isel(lat=np.where(mlat)[0], lon=np.where(mlon)[0])
        w = np.cos(np.deg2rad(sub['lat'])).broadcast_like(sub)
        out[nome] = (sub * w).sum(('lat', 'lon')).values / w.where(~np.isnan(sub)).sum(('lat', 'lon')).values
    df = pd.DataFrame(out, index=pd.to_datetime(sst['time'].values).to_period('M').to_timestamp())
    anom = df.copy()
    for m in range(1, 13):                        # anomalia contra os 30 anos anteriores do mesmo mês (sem olhar o futuro)
        g = df[df.index.month == m]
        anom.loc[g.index] = g - g.shift(1).rolling(30, min_periods=15).mean()
    anom = anom.dropna(how='all').rename_axis('mes').reset_index()
    anom.to_csv(arq, index=False)
    print('ERSST:', anom.mes.min().date(), '→', anom.mes.max().date(), list(REGIOES_SST))
    return anom


def _oceano(OC, D, c):
    """valor do último mês publicado no corte (o mês anterior ao da data do corte), média de 3 meses e variação em 6."""
    if OC is None:
        return {}
    dc = pd.Timestamp(D.datas[c]) + pd.Timedelta(days=6)
    ult = (dc - pd.Timedelta(days=7)).to_period('M').to_timestamp() - pd.offsets.MonthBegin(1)
    g = OC[OC.mes <= ult].set_index('mes')
    if len(g) < 7:
        return {}
    f = {}
    for r in REGIOES_SST:
        f[f'{r}_m3'] = np.repeat(g[r].iloc[-3:].mean(), D.N)
        f[f'{r}_d6'] = np.repeat(g[r].iloc[-1] - g[r].iloc[-7], D.N)
    return f


# ----------------------------------------------------------------------------- sorotipo (SINAN, OpenDataSUS)
URL_SINAN = 'https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINAN/Dengue/csv/DENGBR{aa:02d}.csv.zip'
COLS_SINAN = ['SG_UF', 'SG_UF_NOT', 'SEM_PRI', 'SOROTIPO', 'CLASSI_FIN']


def contar_sorotipo(df):
    """chunk do SINAN → contagem semanal de casos com sorotipo identificado (1–4) por UF de residência."""
    d = df.copy()
    d.columns = [c.upper() for c in d.columns]
    col = lambda n: pd.to_numeric(d[n], errors='coerce') if n in d else pd.Series(np.nan, index=d.index)   # noqa: E731
    uf = col('SG_UF').fillna(col('SG_UF_NOT'))
    sor, sem, cf = col('SOROTIPO'), col('SEM_PRI'), col('CLASSI_FIN')
    ok = sor.isin([1, 2, 3, 4]) & uf.notna() & sem.between(200001, 203053) & (cf != 5)      # 5 = descartado
    x = pd.DataFrame({'uf_code': uf[ok].astype(int), 'sem_pri': sem[ok].astype(int), 'sorotipo': sor[ok].astype(int)})
    if x.empty:
        return pd.DataFrame(columns=['uf_code', 'sem_pri', 'DENV1', 'DENV2', 'DENV3', 'DENV4'])
    t = x.groupby(['uf_code', 'sem_pri', 'sorotipo']).size().unstack(fill_value=0)
    t = t.reindex(columns=[1, 2, 3, 4], fill_value=0)
    t.columns = ['DENV1', 'DENV2', 'DENV3', 'DENV4']
    return t.reset_index()


def baixar_sorotipo(anos, pasta_cache, pasta_tmp='/content/sinan', chunksize=500_000):
    """um parquet por ano em cache_r4/sorotipo/ (retoma); devolve a tabela UF × semana × sorotipo somada."""
    import zipfile
    pasta = os.path.join(pasta_cache, 'sorotipo')
    os.makedirs(pasta, exist_ok=True)
    os.makedirs(pasta_tmp, exist_ok=True)
    for a in anos:
        arq = os.path.join(pasta, f'sorotipo_{a}.parquet')
        if os.path.exists(arq):
            continue
        t0 = time.time()
        zp = os.path.join(pasta_tmp, f'DENGBR{a % 100:02d}.csv.zip')
        if not (os.path.exists(zp) or _baixar(URL_SINAN.format(aa=a % 100), zp)):
            print(f'⚠️ {a}: não baixou; fica sem esse ano')
            continue
        partes = []
        with zipfile.ZipFile(zp) as z:
            nome = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
            with z.open(nome) as f:
                linha = f.readline().decode('latin1').strip()
            sep = ';' if linha.count(';') > linha.count(',') else ','
            usa = [c.strip('"') for c in linha.split(sep) if c.strip('"').upper() in COLS_SINAN]
            with z.open(nome) as f:
                for ch in pd.read_csv(f, usecols=usa, dtype=str, chunksize=chunksize, encoding='latin1', sep=sep,
                                      on_bad_lines='skip', low_memory=True):
                    partes.append(contar_sorotipo(ch))
        t = pd.concat(partes).groupby(['uf_code', 'sem_pri'], as_index=False).sum()
        _grava_pq(t, arq)
        os.remove(zp)
        print(f'✅ sorotipo {a}: {int(t[["DENV1", "DENV2", "DENV3", "DENV4"]].values.sum()):,} casos tipados · '
              f'{time.time() - t0:.0f}s', flush=True)
    fs = sorted(glob.glob(os.path.join(pasta, 'sorotipo_*.parquet')))
    if not fs:
        return None
    return pd.concat([pd.read_parquet(f) for f in fs]).groupby(['uf_code', 'sem_pri'], as_index=False).sum()


def ler_sorotipo(pasta_cache):
    fs = sorted(glob.glob(os.path.join(pasta_cache, 'sorotipo', 'sorotipo_*.parquet')))
    if not fs:
        return None
    return pd.concat([pd.read_parquet(f) for f in fs]).groupby(['uf_code', 'sem_pri'], as_index=False).sum()


def _matriz_sorotipo(SOR, D):
    if id(SOR) in _CACHE:
        return _CACHE[id(SOR)]
    A = np.zeros((D.T, D.N, 4))
    pos = {int(e): i for i, e in enumerate(D.epiweek)}
    uj = {int(u): j for j, u in enumerate(D.unidades)}
    for r in SOR.itertuples(index=False):
        i, j = pos.get(int(r.sem_pri)), uj.get(int(r.uf_code))
        if i is not None and j is not None:
            A[i, j] = (r.DENV1, r.DENV2, r.DENV3, r.DENV4)
    _CACHE[id(SOR)] = A
    return A


def _sorotipo(SOR, D, c, atraso=8):
    """frações de sorotipo nas 26 semanas até c−atraso (encolhidas para a fração nacional) contra os 3 anos anteriores.
    Avisos: (1) usa os dados FINAIS do SINAN, que hoje estão mais completos do que estavam na época; (2) o atraso de
    8 semanas imita o tempo até o resultado do exame chegar ao sistema."""
    if SOR is None:
        return {}
    A = _matriz_sorotipo(SOR, D)
    b = c - atraso
    if b - 26 - 156 < 0:
        return {}
    a = A[b - 25:b + 1].sum(0)                          # (N, 4) recente
    h = A[b - 26 - 155:b - 25].sum(0)                   # (N, 4) três anos antes
    na, nh = a.sum(1, keepdims=True), h.sum(1, keepdims=True)
    fa_br, fh_br = a.sum(0) / max(a.sum(), 1), h.sum(0) / max(h.sum(), 1)
    fa = (a + 10 * fa_br) / (na + 10)
    fh = (h + 10 * fh_br) / (nh + 10)
    f = {f'sor_frac_DENV{k + 1}': fa[:, k] for k in range(4)}
    f['sor_emergente'] = (fa - fh).max(1)                                  # quanto o sorotipo que mais cresceu ganhou espaço
    f['sor_pouco_circulante'] = (fa * (fh < 0.15)).sum(1)                  # fração de sorotipos que eram raros
    f['sor_emergente_br'] = np.repeat((fa_br - fh_br).max(), D.N)
    f['sor_tipados_log'] = np.log1p(na[:, 0])
    return f


def variaveis_novas(ctx, s, nome, TAM, OC=None, SOR=None):
    D = ctx.Du
    c = R3.indice_corte(D, s, nome)
    if c is None or c > D.ultimo_obs:
        return None
    SC = series_causais(D)
    f = {}
    ys = np.nan_to_num(D.Y)
    ew = [_ews(SC['x'][:, j], ys[:, j], c) for j in range(D.N)]
    for k in next((e for e in ew if e), {}):
        f[k] = np.array([e.get(k, np.nan) for e in ew])
    f.update(_imunidade(D, TAM, s, c))
    for k in ('seca_13', 'seca_26', 'seca_max_52', 'clim_temp_13', 'clim_temp_26', 'clim_precip_13', 'clim_precip_26',
              'clim_umid_13', 'clim_umid_26'):
        if k in SC:
            f[k] = SC[k][c]
    f.update(_seas5(ctx, D, c))
    f.update(_oceano(OC, D, c))
    f.update(_sorotipo(SOR, D, c))
    out = pd.DataFrame(f)
    out.insert(0, 'uf_code', D.unidades)
    out.insert(1, 'season', s)
    out.insert(2, 'corte', nome)
    return out


def tabela_detector4(ctx, temporadas, pasta_cache, OC=None, SOR=None, externos=None):
    """variáveis da R3 + novas, alvo e base (cache `detector4_variaveis.parquet`)."""
    arq = os.path.join(pasta_cache, 'detector4_variaveis.parquet')
    if os.path.exists(arq):
        X = pd.read_parquet(arq)
    else:
        X3 = R3.tabela_detector(ctx, temporadas, pasta_cache)               # grava detector_variaveis.parquet (R3)
        TAM = R3.tamanhos(ctx.Du)
        novas = [v for s in temporadas for k in CORTES for v in [variaveis_novas(ctx, s, k, TAM, OC, SOR)] if v is not None]
        X = X3.merge(pd.concat(novas, ignore_index=True), on=['uf_code', 'season', 'corte'], how='left')
        _grava_pq(X, arq)
    if externos is not None and len(externos):
        chaves = [k for k in ('uf_code', 'season', 'corte') if k in externos]
        X = X.merge(externos, on=chaves, how='left')
    return X


def _nacional(X, col):
    """variável igual em todas as UFs na mesma temporada (ENSO, mar, sorotipo nacional): tem só ~10 valores distintos."""
    chave = ['season', 'corte'] if 'corte' in X else ['season']      # (corrigido em 14/10: sem o corte, nada parecia nacional)
    g = X[chave + [col]].dropna()
    return len(g) > 0 and float(g.groupby(chave)[col].std().fillna(0).max()) < 1e-9


def selecionar_sinais4(X, corte, ate_temporada=max(TUNING), k=12, rho_min=0.12, rho_min_nacional=0.30, arq=None):
    """Spearman com o alvo SÓ em temporadas ≤ ate_temporada. Variáveis nacionais precisam de |ρ| maior, porque têm
    só uma observação independente por temporada."""
    if arq and os.path.exists(arq):
        sel = json_le(arq, {})
        if corte in sel:
            return sel[corte]['sinais']
    g = X[(X.corte == corte) & (X.season <= ate_temporada)].dropna(subset=['alvo'])
    cols = [c for c in X.columns if c not in R3.NAO_VARIAVEIS and X[c].dtype.kind in 'fi']
    r = []
    for c in cols:
        x = g[[c, 'alvo']].dropna()
        if len(x) > 40 and x[c].std() > 0:
            rho = float(spearmanr(x[c], x.alvo)[0])
            lim = rho_min_nacional if _nacional(g, c) else rho_min
            if abs(rho) >= lim:
                r.append((c, rho))
    r = sorted(r, key=lambda t: -abs(t[1]))[:k]
    if arq:
        sel = json_le(arq, {})
        sel[corte] = dict(sinais=[c for c, _ in r], rho={c: round(v, 3) for c, v in r}, ate_temporada=int(ate_temporada))
        json_grava(sel, arq)
    return [c for c, _ in r]


# ----------------------------------------------------------------------------- métodos
def _duas_etapas(Xtr, ytr, Xte, seed=7, M=4000):
    """1ª etapa: P(temporada grande) por regressão logística; 2ª: tamanho por ridge separado em grandes e não grandes
    (faixa pelos resíduos). Mistura das duas pelas probabilidades."""
    from sklearn.linear_model import LogisticRegression, RidgeCV
    g = ytr > LIM_GRANDE
    if g.sum() < 12 or (~g).sum() < 12:
        return R3._quantis_detector('ridge', Xtr, ytr, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    p = LogisticRegression(C=0.5, max_iter=5000).fit(Ztr, g).predict_proba(Zte)[:, 1]
    rng = np.random.default_rng(seed)
    out = []
    mods = []
    for sel in (g, ~g):
        m = RidgeCV(alphas=np.logspace(-1, 3, 20)).fit(Ztr[sel], ytr[sel])
        mods.append((m.predict(Zte), (ytr[sel] - m.predict(Ztr[sel])) * 1.15))
    for i in range(len(Zte)):
        eg = rng.random(M) < p[i]
        amo = np.where(eg, mods[0][0][i] + rng.choice(mods[0][1], M), mods[1][0][i] + rng.choice(mods[1][1], M))
        out.append(np.quantile(amo, Q))
    return np.array(out)


def _quantis4(metodo, Xtr, ytr, Xte, seed=7):
    if metodo == 'duas_etapas':
        return _duas_etapas(Xtr, ytr, Xte, seed)
    return R3._quantis_detector(metodo, Xtr, ytr, Xte, seed)


METODOS4 = ('ridge', 'qlin', 'qgbm', 'duas_etapas')


def conformalizar(P, D, min_linhas=50):
    """calibração conforme por quantil, com as previsões FORA DA AMOSTRA de temporadas já terminadas no corte:
    q_τ ← q_τ + quantil_τ(y − q_τ). Corrige a cauda de cima curta (anos grandes)."""
    out = []
    fim = {s: R3._completa_ate(D, s) for s in P.season.unique()}
    for corte, g in P.groupby('corte'):
        g = g.sort_values('season')
        for s, gs in g.groupby('season'):
            c = R3.indice_corte(D, s, corte)
            ant = g[(g.season < s) & g.alvo.notna() & g.season.map(lambda x: (fim.get(x) or 10 ** 9) <= c)]
            q = gs[QCOLS].values.astype(float)
            if len(ant) >= min_linhas:
                d = ant.alvo.values[:, None] - ant[QCOLS].values
                q = q + np.array([np.quantile(d[:, j], a) for j, a in enumerate(Q)])[None]
            o = gs.copy()
            o[QCOLS] = np.sort(q, 1)
            out.append(o)
    out = pd.concat(out, ignore_index=True)
    out['metodo'] = out.metodo + '+conf'
    out['p_grande'] = R3._prob_acima(out[QCOLS].values, LIM_GRANDE)
    return out


def prever_detector4(ctx, X, sinais, temporadas, pasta_cache, metodos=METODOS4, min_treino=60):
    """janela expansiva (como a R3) + média + versões conformes. Um parquet por (método, corte) em cache_r4."""
    D = ctx.Du
    fim = {s: R3._completa_ate(D, s) for s in sorted(X.season.unique())}
    saidas = []
    for metodo in list(metodos) + ['media']:
        for corte in CORTES:
            arq = os.path.join(pasta_cache, f'detector4_{metodo}_{corte}.parquet')
            if os.path.exists(arq):
                saidas.append(pd.read_parquet(arq))
                continue
            if metodo == 'media':
                ps = [pd.read_parquet(os.path.join(pasta_cache, f'detector4_{m}_{corte}.parquet')) for m in metodos]
                out = ps[0].copy()
                out[QCOLS] = np.sort(np.mean([p[QCOLS].values for p in ps], 0), -1)
                out['metodo'] = 'media'
            else:
                g = X[X.corte == corte]
                cols = sinais[corte]
                linhas = []
                for s in temporadas:
                    c = R3.indice_corte(D, s, corte)
                    te = g[g.season == s]
                    if c is None or te.empty:
                        continue
                    tr = g[g.season.map(lambda x: (fim.get(x) or 10 ** 9) <= c) & (g.season < s)].dropna(subset=['alvo'])
                    med = tr[cols].median()
                    trX = tr[cols].fillna(med)
                    if len(tr) < min_treino:
                        continue
                    q = np.sort(_quantis4(metodo, trX.values, tr.alvo.values, te[cols].fillna(med).values), -1)
                    o = te[['uf_code', 'season', 'corte', 'base', 'alvo', 'pop']].copy()
                    o[QCOLS] = q
                    o['metodo'] = metodo
                    o['n_treino'] = len(tr)
                    linhas.append(o)
                if not linhas:
                    continue
                out = pd.concat(linhas, ignore_index=True)
            out['p_grande'] = R3._prob_acima(out[QCOLS].values, LIM_GRANDE)
            _grava_pq(out, arq)
            saidas.append(out)
            print(f'detector4 {metodo:12s} {corte}: {len(out)} linhas', flush=True)
    P = pd.concat(saidas, ignore_index=True)
    arq_c = os.path.join(pasta_cache, 'detector4_conformes.parquet')
    if os.path.exists(arq_c):
        PC = pd.read_parquet(arq_c)
    else:
        PC = pd.concat([conformalizar(g, D) for _, g in P.groupby('metodo')], ignore_index=True)
        _grava_pq(PC, arq_c)
    return pd.concat([P, PC], ignore_index=True)


def avaliar_detector4(P):
    """como a R3 + erro nas temporadas grandes (mediana prevista − real, em ×) e cobertura da cauda de cima."""
    T = R3.avaliar_detector(P)
    P = P.dropna(subset=['alvo']).copy()
    P['grupo'] = np.where(P.season.isin(TESTE), 'teste', np.where(P.season.isin(TUNING), 'afinacao', 'anterior'))
    ex = []
    for (m, k, gr), g in P.groupby(['metodo', 'corte', 'grupo']):
        gg = g[g.alvo > LIM_GRANDE]
        ex.append(dict(metodo=m, corte=k, grupo=gr,
                       vies_grandes_x=float(np.exp(np.median(gg.pred - gg.alvo))) if len(gg) else np.nan,
                       acima_q95_pct=float(100 * np.mean(g.alvo > g.upper_90))))
    return T.merge(pd.DataFrame(ex), on=['metodo', 'corte', 'grupo'])


def pinball_escolha(P, ate=max(TUNING), peso_grande=3.0):
    """pinball médio nas temporadas ≤ 2021 (anteriores + afinação), com as temporadas grandes valendo `peso_grande`
    vezes: o erro que mais custa aqui é prever ano normal quando vem ano grande."""
    P = P[(P.season <= ate)].dropna(subset=['alvo'])
    lin = []
    for (m, k), g in P.groupby(['metodo', 'corte']):
        y, q = g.alvo.values, g[QCOLS].values
        pb = np.mean([np.maximum(a * (y - q[:, j]), (a - 1) * (y - q[:, j])) for j, a in enumerate(Q)], 0)
        w = 1 + (peso_grande - 1) * (y > LIM_GRANDE)
        lin.append(dict(metodo=m, corte=k, n_escolha=len(g), pinball_escolha=float(np.sum(w * pb) / np.sum(w))))
    return pd.DataFrame(lin)


def escolher_detector4(PE):
    return {k: g.sort_values('pinball_escolha').metodo.iloc[0] for k, g in PE.groupby('corte')}


def detector_escolhido4(pasta_cache):
    esc = json_le(os.path.join(pasta_cache, 'detector4_escolhido.json'))
    assert esc, 'Rode antes o R4_1_Detector_ampliado.'
    P = pd.read_parquet(os.path.join(pasta_cache, 'detector4_todos.parquet'))
    return esc, pd.concat([P[(P.metodo == m) & (P.corte == k)] for k, m in esc.items()], ignore_index=True)


# =====================================================================================================================
# ② MIX_elastico · cada semana cresce conforme a sua elasticidade histórica ao tamanho da temporada
# =====================================================================================================================
NIVEIS_ELAST = (0, 0.5, 1, 1.5, 2, 3, 9)        # faixas do nível normal da semana, em log1p(incidência por 100 mil)


def elasticidades(D, TAM, s, c, W=53):
    """b (W, N): quanto o log1p(incidência) de cada semana × UF sobe por unidade de log-tamanho relativo da temporada.
    Depende do NÍVEL normal da semana (média das temporadas terminadas até o corte): no histórico (até 2021) é ~0,1
    nas semanas quase sem casos e ~1 no pico. Estimado por faixa de nível, juntando UFs e temporadas, e interpolado."""
    js = [j for j in TAM.index if (R3._completa_ate(D, j) or 10 ** 9) <= c and j < s]
    if len(js) < 4:
        return np.ones((W, D.N))
    A = (TAM.loc[js] - TAM.loc[js].mean()).values                          # (J, N)
    Ys = np.stack([np.log1p(np.nan_to_num(D.INC[D.idx_temporada(j)[:52]])) for j in js])   # (J, 52, N)
    mu = Ys.mean(0)                                                         # (52, N) nível normal
    dev = Ys - mu[None]
    Ab = np.broadcast_to(A[:, None, :], Ys.shape)
    centros, inc = [], []
    for lo, hi in zip(NIVEIS_ELAST[:-1], NIVEIS_ELAST[1:]):
        m = np.broadcast_to(((mu >= lo) & (mu < hi))[None], Ys.shape)
        if m.sum() < 40:
            continue
        x, y = Ab[m], dev[m]
        centros.append(float(np.median(np.broadcast_to(mu[None], Ys.shape)[m])))
        inc.append(float((x * y).sum() / max((x * x).sum(), 1e-9)))
    if len(centros) < 2:
        return np.ones((W, D.N))
    b = np.clip(np.interp(mu, centros, np.maximum.accumulate(inc)), 0.03, 1.5)
    return np.vstack([b, np.repeat(b[-1:], max(0, W - 52), 0)])[:W]


def _kappa(traj, b, alvo, it=40):
    """κ (M,) tal que Σ_w traj·exp(κ·b_w) = alvo (bisseção; a soma cresce com κ porque b > 0)."""
    lo, hi = np.full(len(traj), -12.0), np.full(len(traj), 12.0)
    la = np.log(np.maximum(alvo, 1e-6))
    for _ in range(it):
        k = (lo + hi) / 2
        f = np.log(np.sum(traj * np.exp(np.clip(k[:, None] * b[None], -50, 50)), 1) + 1e-9) - la
        hi = np.where(f > 0, k, hi)
        lo = np.where(f > 0, lo, k)
    return (lo + hi) / 2


def mistura_elastica_uf(qn, det_q, base, pop, b, w=1.0, M=1000, seed=0):
    rng = np.random.default_rng(seed)
    lq = np.log1p(np.maximum(np.sort(qn, -1), 0))
    traj = np.expm1(R3._interp(lq, rng.standard_normal(M))).clip(0) + 1e-3   # (M, W)
    T_n = traj.sum(1)
    a = R3._interp(np.sort(det_q)[None, :], rng.standard_normal(M))[:, 0]
    alvo = np.expm1(base + a) * pop / 1e5
    alvo = np.exp(np.log(T_n) + w * (np.log(np.maximum(alvo, 1)) - np.log(T_n)))
    k = _kappa(traj, b, alvo)
    out = traj * np.exp(np.clip(k[:, None] * b[None], -50, 50))
    return np.quantile(out, Q, axis=0).T


def gerar_mix_elastico(ctx, NORMAL, DET, w, corte, temporadas, M=1000, nome='MIX_elastico_UF'):
    D = ctx.Du
    TAM = R3.tamanhos(D)
    nm = NORMAL[NORMAL.modelo == 'ENS_grafos3_λ']
    lin = []
    for s in temporadas:
        g = nm[nm.season == s]
        d = DET[(DET.season == s) & (DET.corte == corte)].set_index('uf_code')
        if g.empty or d.empty:
            continue
        datas = np.sort(g.date.unique())
        qn = V.de_df(g, datas, D.unidades)
        c = R3.indice_corte(D, s, corte)
        b = elasticidades(D, TAM, s, c, len(datas))
        out = np.zeros_like(qn)
        for j, u in enumerate(D.unidades):
            if u not in d.index:
                out[:, j] = qn[:, j]
                continue
            r = d.loc[u]
            out[:, j] = mistura_elastica_uf(qn[:, j], r[QCOLS].values.astype(float), float(r.base), float(r['pop']), b[:, j], w,
                                            M=M, seed=1000 * s + j)
        lin.append(V.para_df(V.arruma_quantis(out), datas, D.unidades, nome, s))
    return pd.concat(lin, ignore_index=True) if lin else None


# =====================================================================================================================
# ③ DoisEstagios_det · tamanho do detector × forma de temporadas parecidas em tamanho
# =====================================================================================================================
def gerar_dois_estagios_det(ctx, DET, corte, temporadas, M=1000, banda=0.6, disp=20.0, nome='DoisEstagios_det_UF'):
    """para cada trajetória: sorteia o tamanho relativo a (detector) e uma temporada passada com peso
    exp(−(a − a_j)²/2·banda²) — da própria UF (peso 1) ou da macrorregião (peso 0,4) —, desloca ±2 semanas e
    aplica ruído binomial negativo. A forma dos anos grandes (começo mais cedo, pico estreito) vem junto."""
    D = ctx.Du
    TAM = R3.tamanhos(D)
    lin = []
    macro = D.unidades // 10
    for s in temporadas:
        d = DET[(DET.season == s) & (DET.corte == corte)].set_index('uf_code')
        if d.empty:
            continue
        datas, _ = ctx.alvo(s)
        W = len(datas)
        c = R3.indice_corte(D, s, corte)
        js = [j for j in TAM.index if (R3._completa_ate(D, j) or 10 ** 9) <= c and j < s]
        if len(js) < 3:
            continue
        an = TAM.loc[js] - TAM.loc[js].mean()
        formas = {}
        for j in js:
            y = np.nan_to_num(D.Y[D.idx_temporada(j)][:52])
            ys = np.apply_along_axis(lambda v: np.convolve(v, np.ones(3) / 3, mode='same'), 0, y)
            formas[j] = ys / np.maximum(ys.sum(0, keepdims=True), 1e-9)       # (52, N)
        out = np.zeros((W, D.N, 9))
        for jn, u in enumerate(D.unidades):
            rng = np.random.default_rng(1000 * s + jn)
            if u not in d.index:
                continue
            r = d.loc[u]
            cand = [(j, k, 1.0 if k == jn else 0.4) for j in js for k in range(D.N) if macro[k] == macro[jn]]
            aj = np.array([an.loc[j].values[k] for j, k, _ in cand])
            wsrc = np.array([w for _, _, w in cand])
            a = R3._interp(np.sort(r[QCOLS].values.astype(float))[None, :], rng.standard_normal(M))[:, 0]
            tot = np.expm1(float(r.base) + a) * float(r['pop']) / 1e5
            pesos = wsrc[None] * np.exp(-(a[:, None] - aj[None]) ** 2 / (2 * banda ** 2)) + 1e-12
            pesos /= pesos.sum(1, keepdims=True)
            esc = np.minimum((np.cumsum(pesos, 1) < rng.random(M)[:, None]).sum(1), len(cand) - 1)
            desl = rng.integers(-2, 3, M)
            lam = np.zeros((M, W))
            for m in range(M):
                j, k, _ = cand[esc[m]]
                f = np.roll(formas[j][:, k], desl[m])
                f = np.r_[f, np.repeat(f[-1:], max(0, W - 52))][:W]
                lam[m] = tot[m] * f / max(f.sum(), 1e-9)
            p = disp / (disp + lam)
            traj = rng.negative_binomial(disp, np.clip(p, 1e-9, 1))
            out[:, jn] = np.quantile(traj, Q, axis=0).T
        lin.append(V.para_df(V.arruma_quantis(out), datas, D.unidades, nome, s))
    return pd.concat(lin, ignore_index=True) if lin else None


# =====================================================================================================================
# critério misto (afinação): ½ R × referência + ½ WIS absoluto relativo
# =====================================================================================================================
PESO_ABS = 0.5                 # peso do WIS absoluto (altura dos picos) no critério misto; 0 = só a régua R do IMDC


def criterio(ctx, X, ref='climatologia', peso_abs=None):
    """X: previsões de vários modelos + a referência. Devolve uma tabela por modelo com R, WIS_rel e
    criterio = (1 − peso_abs)·R + peso_abs·WIS_rel. peso_abs maior = prioriza acertar a altura dos picos."""
    peso_abs = PESO_ABS if peso_abs is None else peso_abs
    _, r = V.avaliar(X[L.COLS_PREV], ctx.ver_uf, ref=(ref,))
    T = S.com_verdade(X, ctx.Du)
    T['w'] = V.wis(T[QCOLS].values.astype(float), T.y.values.astype(float))
    wr = T[T.modelo == ref][['season', 'uf_code', 'date', 'w']].rename(columns={'w': 'w_ref'})
    lin = []
    for m, g in T[T.modelo != ref].groupby('modelo'):
        mm = g.merge(wr, on=['season', 'uf_code', 'date'])
        lin.append(dict(modelo=m, WIS_rel=float(mm.w.sum() / max(mm.w_ref.sum(), 1e-9))))
    out = r[['modelo', f'R_vs_{ref}']].merge(pd.DataFrame(lin), on='modelo')
    out['criterio'] = (1 - peso_abs) * out[f'R_vs_{ref}'] + peso_abs * out.WIS_rel
    return out.sort_values('criterio')


def escolher_w_elastico(ctx, NORMAL, DET, pasta_cache, grade=(0.5, 0.75, 1.0), M=500):
    arq = os.path.join(pasta_cache, 'mix_elastico_w.json')
    esc = json_le(arq, {})
    if 'w' in esc:
        return esc
    clim = NORMAL[(NORMAL.modelo == 'climatologia') & NORMAL.season.isin(TUNING)]
    partes = [clim, NORMAL[(NORMAL.modelo == 'ENS_grafos3_λ') & NORMAL.season.isin(TUNING)]]
    for w in grade:
        x = gerar_mix_elastico(ctx, NORMAL, DET, w, 'jun', TUNING, M=M, nome=f'w={w}')
        if x is not None:
            partes.append(x)
    C = criterio(ctx, pd.concat(partes, ignore_index=True))
    print(C.round(3).to_string(index=False))
    gw = C[C.modelo.str.startswith('w=')]
    esc = dict(w=float(gw.modelo.iloc[0][2:]), tabela=C.round(4).to_dict('records'))
    json_grava(esc, arq)
    return esc


# =====================================================================================================================
# ④ MTGNN condicionado ao tamanho da temporada (GPU) + aumento com temporadas grandes sintéticas
# =====================================================================================================================
DESLOC_PRINCIPAL = 35          # a "temporada principal" de uma origem o é a que contém a semana o+35 (meio da janela de 68)


def cov_condicao(D, c, TAM):
    """(T, N, 2): [tamanho relativo da temporada principal, 1 = conhecido]. Só temporadas terminadas até o corte; a
    base é a média de todas elas (a mesma régua do detector na previsão)."""
    js = [j for j in TAM.index if (R3._completa_ate(D, j) or 10 ** 9) <= c]
    base = TAM.loc[js].mean()
    cov = np.zeros((D.T, D.N, 2), np.float32)
    sp = D.season[np.minimum(np.arange(D.T) + DESLOC_PRINCIPAL, D.T - 1)]
    for t in range(D.T):
        if sp[t] in js:
            cov[t, :, 0] = (TAM.loc[sp[t]] - base).reindex(D.unidades).values
            cov[t, :, 1] = 1.0
    return cov


def amostras_cond(ctx, rede, c, H, peso_pop, TAM, frac_sem=0.2, frac_aum=0.6, k_min=2.0, k_max=8.0, seed=0):
    """amostras da rodada 2 (âncora, peso pela população) + condição; cópias sem condição (a rede aprende os dois
    usos) e cópias AUMENTADAS: a temporada principal multiplicada por k ∈ [2, 8] (entrada igual, futuro maior,
    condição + log k) — ensina a rede a desenhar anos maiores que os já vistos."""
    D = ctx.Du
    cov = cov_condicao(D, c, TAM)
    dd = R2._amostras(D, c, rede.L, H, 'ancora', False, peso_pop, cov)
    o = dd['o']
    Srows = dd['S'][o]                                                      # (B, N, d)
    B = len(o)
    rng = np.random.default_rng(seed)
    i1 = rng.choice(B, int(frac_sem * B), replace=False)
    S1 = Srows[i1].copy()
    S1[..., -2:] = 0
    flag = Srows[..., -1]
    eleg = np.where(flag.max(1) > 0)[0]
    i2 = rng.choice(eleg, int(frac_aum * len(eleg)), replace=True) if len(eleg) else np.array([], int)
    k = np.exp(rng.uniform(np.log(k_min), np.log(k_max), (len(i2), D.N))) * flag[i2] + (1 - flag[i2])
    sp = D.season[np.minimum(o[i2] + DESLOC_PRINCIPAL, D.T - 1)]
    st = D.season[np.minimum(dd['t'][i2], D.T - 1)]                        # (n2, H)
    msk = (st == sp[:, None])[..., None]                                    # (n2, H, 1)
    yt2 = np.where(msk, np.log1p(k[:, None, :] * np.expm1(dd['ytrue'][i2])), dd['ytrue'][i2]).astype(np.float32)
    Y2 = ((yt2 - dd['base'][i2]) / dd['esc'][None, None]).astype(np.float32)
    S2 = Srows[i2].copy()
    S2[..., -2] += np.log(k) * flag[i2]
    junta = lambda a, b1, b2: np.concatenate([a, b1, b2]).astype(np.float32)   # noqa: E731
    out = dict(X=junta(dd['X'], dd['X'][i1], dd['X'][i2]), S=junta(Srows, S1, S2),
               Y=junta(dd['Y'], dd['Y'][i1], Y2), Mk=junta(dd['Mk'], dd['Mk'][i1], dd['Mk'][i2]),
               base=junta(dd['base'], dd['base'][i1], dd['base'][i2]), esc=dd['esc'],
               ytrue=junta(dd['ytrue'], dd['ytrue'][i1], yt2))
    out['wo'] = np.concatenate([D.woyc[o], D.woyc[o[i1]], D.woyc[o[i2]]])
    out['o'] = np.arange(len(out['X']))
    return out, dd


class _Semanas:
    """o treino do r2 lê D.woyc[o]; aqui as amostras repetidas/aumentadas têm a semana do ano já em `wo`."""
    def __init__(self, wo):
        self.woyc = wo


def treinar_cond(ctx, rede, c, H, pasta_ckpt, chave, peso_pop=0.5, nome='MTGNN_cond_UF'):
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    D = ctx.Du
    TAM = R3.tamanhos(D)
    dd2, dd = amostras_cond(ctx, rede, c, H, peso_pop, TAM)
    redes = []
    for seed in rede.sementes:
        torch.manual_seed(seed)
        net = rede.construir(nn, torch, D.N, dd2['X'].shape[-1], dd2['S'].shape[-1], H, dd['A'], rede.d).to(dev)
        arq = os.path.join(pasta_ckpt, f'{nome}_{chave}_s{seed}.pt')
        R2.treinar_ckpt(R2._Cfg(_Semanas(dd2['wo'])), net, dd2, rede.epocas, rede.lr, seed, dev, arq, bs=128)
        net.eval()
        redes.append(net)
    return redes, dd


def cenarios_tamanho(det_q, K=24, rho=0.6, seed=0):
    """K vetores de tamanho relativo (N,) a partir dos quantis do detector de cada UF, com correlação ρ entre UFs."""
    rng = np.random.default_rng(seed)
    G = norm.ppf((np.arange(K) + 0.5) / K)
    E = rng.standard_normal((K, det_q.shape[0]))
    Z = rho * G[:, None] + np.sqrt(1 - rho ** 2) * E
    return np.stack([R3._interp(np.sort(det_q[j])[None, :], Z[:, j])[:, 0] for j in range(det_q.shape[0])], 1)   # (K, N)


def prever_cond(redes, dd, D, L_, c, z=None, H=68):
    """→ (H, N, 9) log1p(incidência). z=None: sem condição (MTGNN_aug)."""
    Sm = dd['S'].copy()
    if z is None:
        Sm[c, :, -2:] = 0
    else:
        Sm[c, :, -2], Sm[c, :, -1] = z, 1.0
    o = np.array([c])
    r = S.prever_rede(redes, dd['P'], Sm, D, L_, o)                         # (1, H, N, 9)
    return (S._base(dd['P'], D, o, H, 'ancora')[..., None] + dd['esc'][None, None, :, None] * r)[0]


def mistura_cenarios(lqs, M_por=50, seed=0):
    """lqs (K, H, N, 9) → (H, N, 9): mistura de trajetórias coerentes no tempo de cada cenário."""
    rng = np.random.default_rng(seed)
    K, H, N, _ = lqs.shape
    out = np.zeros((H, N, 9))
    for j in range(N):
        tr = np.concatenate([R3._interp(np.sort(lqs[k, :, j], -1), rng.standard_normal(M_por)) for k in range(K)])
        out[:, j] = np.quantile(tr, Q, axis=0).T
    return out


def prever_temporada_cond(ctx, redes, dd, rede, s, DET, corte='jun', K=24, teto_x=10.0):
    """previsões MTGNN_cond (cenários do detector) e MTGNN_aug (sem condição) da temporada s em casos (W, N, 9)."""
    D = ctx.Du
    c, datas, idx, h = CM._horizontes(ctx, s)
    teto = np.log1p(teto_x * np.nanmax(D.INC[:c + 1], 0))[None, :, None]
    pop = CM._pop_c(D, s)[None, :, None]
    lq0 = np.minimum(np.sort(prever_cond(redes, dd, D, rede.L, c, None), -1), teto)
    saida = {'MTGNN_aug_UF': CM._para_contagem(lq0[h - 1], pop)}
    d = DET[(DET.season == s) & (DET.corte == corte)].set_index('uf_code').reindex(D.unidades)
    if d[QCOLS].notna().all().all():
        Zs = cenarios_tamanho(d[QCOLS].values.astype(float), K=K, seed=s)
        lqs = np.stack([np.minimum(np.sort(prever_cond(redes, dd, D, rede.L, c, z), -1), teto) for z in Zs])
        saida['MTGNN_cond_UF'] = CM._para_contagem(mistura_cenarios(lqs, seed=s)[h - 1], pop)
    return datas, saida


# =====================================================================================================================
# ⑤ ENSEMBLE POR REGIME (faixa de P(ano grande)) + calibração por faixa
# =====================================================================================================================
FAIXAS = (0.35, 0.65)


def regime(p):
    p = np.asarray(p, float)
    return np.where(p < FAIXAS[0], 'baixa', np.where(p < FAIXAS[1], 'media', 'alta'))


def com_regime(X, DET, corte='jun'):
    d = DET[DET.corte == corte][['season', 'uf_code', 'p_grande']]
    X = X.merge(d, on=['season', 'uf_code'], how='left')
    X['regime'] = regime(X.p_grande.fillna(0.5))
    return X


def pesos_regime(T, membros, ref='climatologia', passo=0.1, max_iter=60):
    """T (afinação, com y e regime): pesos ≥ 0 que somam 1 por regime, minimizando o critério misto
    ½ WIS_log/WIS_log_ref + ½ WIS/WIS_ref da média ponderada dos quantis."""
    chaves = ['season', 'uf_code', 'date', 'regime']
    idx, arr = R2._alinha(T, membros + [ref], chaves)
    y = idx.merge(T.drop_duplicates(chaves)[chaves + ['y']], on=chaves, how='left').y.values.astype(float)
    out = {}
    for b in idx.regime.astype(str).unique():
        s = (idx.regime.astype(str) == b).values
        A, R_, yy = arr[s, :-1], arr[s, -1], y[s]
        wl0, wa0 = V.wis(np.log1p(R_), np.log1p(yy)).mean(), V.wis(R_, yy).sum()

        def f(w):
            q = np.sort(np.tensordot(A, w, (1, 0)), 1)
            return (1 - PESO_ABS) * V.wis(np.log1p(q), np.log1p(yy)).mean() / wl0 + PESO_ABS * V.wis(q, yy).sum() / max(wa0, 1e-9)
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
        out[b]['_criterio'] = round(float(atual), 4)
    return out


GRADE_LAM4 = (0.8, 1.0, 1.2, 1.5)
GRADE_DELTA4 = (-0.2, 0.0, 0.2, 0.4)           # deslocamento máximo da mediana: ×1,5 (a grade da R2 ia até ×3,3)


def ajustar_regime(ctx, XR, membros):
    """XR: previsões (com regime) das temporadas de ajuste → (pesos por regime, calibração por regime)."""
    TR = S.com_verdade(XR, ctx.Du)
    pesos = pesos_regime(TR, membros)
    TE = S.com_verdade(aplicar_regime(XR, membros, pesos), ctx.Du)
    cal = R2.calibrar(TE, 'regime', GRADE_LAM4, GRADE_DELTA4, min_linhas=100)
    return pesos, cal


def prever_regime(XR, membros, pesos, cal, nome='ENS_regime_UF'):
    ens = aplicar_regime(XR, membros, pesos, nome)
    return ens, R2.aplicar_cal(ens, 'regime', cal)


def cv_regime(ctx, XR, membros, temporadas=TUNING):
    """validação na afinação deixando uma temporada de fora (pesos e calibração ajustados nas outras 3): mostra se o
    ensemble e a calibração ganham fora da amostra antes de olhar o teste."""
    out = []
    for s in temporadas:
        tr = XR[XR.season.isin(temporadas) & (XR.season != s)]
        pesos, cal = ajustar_regime(ctx, tr, membros)
        out += list(prever_regime(XR[XR.season == s], membros, pesos, cal))
    return pd.concat(out, ignore_index=True)


def aplicar_regime(X, membros, pesos, nome='ENS_regime_UF'):
    chaves = ['season', 'uf_code', 'date', 'regime']
    pw = {b: {m: v for m, v in p.items() if not m.startswith('_')} for b, p in pesos.items()}
    for b in ('baixa', 'media', 'alta'):                          # faixa sem exemplo na afinação: pesos iguais
        pw.setdefault(b, {m: 1 / len(membros) for m in membros})
    out = R2.aplicar_stacking(X, membros, chaves, 'regime', pw, nome)
    return out


# =====================================================================================================================
# leitura comum
# =====================================================================================================================
def ler_normal(ctx, PASTA_DRIVE, PASTA_AV2, PASTA_COMP, PASTA_EXT, PASTA_C3, temporadas):
    """o mesmo normal da R3 (ENS_grafos3_λ + BB + climatologia), lido do cache_r3."""
    return R3.normal_ens_grafos3(ctx, PASTA_DRIVE, PASTA_AV2, PASTA_COMP, PASTA_EXT, PASTA_C3, temporadas)


def ler_pastas(*pastas):
    fs = [f for p in pastas for f in glob.glob(os.path.join(p, 'uf', '*.parquet'))]
    if not fs:
        return pd.DataFrame(columns=L.COLS_PREV)
    X = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    X['date'] = pd.to_datetime(X.date)
    return X[L.COLS_PREV]


def grava_modelo(df, pasta, nome):
    for s, g in df.assign(modelo=nome).groupby('season'):
        arq = L._arq(pasta, 'uf', nome, s)
        g[L.COLS_PREV].to_parquet(arq + '.tmp', index=False)
        os.replace(arq + '.tmp', arq)
