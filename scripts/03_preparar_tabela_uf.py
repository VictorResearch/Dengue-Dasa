"""
Tabelas unificadas no nível ESTADUAL (UF), recalculadas a partir dos brutos de data_imdc_2026/variaveis/.

Regra geral: nenhuma variável é "somada" ou "tirada a média" sem pensar no que ela significa.
  • CONTAGENS (casos de dengue e chikungunya, acessos Afya, população): SOMA dos municípios.
  • TAXAS (incidência, acessos por 100 mil): recalculadas como soma(contagem) / soma(população). Nunca média de taxas.
  • CLIMA (grandezas intensivas: temperatura, umidade, pressão, chuva média diária, dias de chuva, amplitude):
    média PONDERADA PELA POPULAÇÃO (exposição de quem mora lá), com pesos fixos (população de 2015) para que a
    composição não mude de um ano para o outro. Município sem dado: sai da média e os pesos são renormalizados
    (nunca entra como 0). `cobertura_clima_pop` informa a fração da população com dado.
  • min/máx semanais (temp_min, temp_max, precip_max, ...): média ponderada dos mínimos/máximos MUNICIPAIS
    = "o mínimo típico que a população viveu". NÃO é o mínimo dos mínimos (que mediria só o município extremo).
  • VARIÁVEIS NÃO LINEARES do clima (VPD, HTI = chuva × temperatura, volatilidade de 4 semanas): calculadas em CADA
    MUNICÍPIO e só depois agregadas, porque média de função ≠ função da média (desigualdade de Jensen).
  • VARIÁVEIS LINEARES NO TEMPO (lags, médias móveis): calculadas sobre a série da UF (com pesos fixos, comutam com a
    agregação; o script confere isso).
  • CASOS → física (Savitzky-Golay), imunidade (EWM): recalculadas sobre a série da UF (log e derivada não comutam
    com a soma).
  • VIZINHOS: recalculados no GRAFO DAS UFs (fronteira Queen entre municípios de UFs diferentes), como razão
    agregada: soma(casos dos vizinhos) / soma(população dos vizinhos). W2 = UFs a exatamente 2 passos.
    Também há a versão ponderada pelo CONTATO (nº de pares de municípios vizinhos na fronteira).
    O ES (fora do desafio) não entra como vizinho (os dados dele não são usados no desafio).
  • CATEGÓRICAS ESTÁTICAS (bioma, Köppen): % da população em cada classe (não a moda, que perde a mistura).
    O target encoding municipal NÃO é refeito: com 26 UFs ele vazaria o alvo e não acrescenta nada às frações.
  • NACIONAIS (ENSO, IOD, PDO) e de calendário (seno/cosseno): iguais para todas as UFs (repetidas).
  • NOVAS, que só existem no nível agregado: % de municípios (e da população) com casos na semana, % da população
    em municípios com ≥ 100 mil habitantes, dispersão espacial do clima dentro da UF.

Saídas (UF/dados/):
  tabela_uf_semanal.parquet     UF × semana (2010-01-03 → última semana com casos)
  tabela_uf_estatica.parquet    UF: frações de bioma e Köppen, área de contato com vizinhos
  tabela_uf_temporada.parquet   UF × temporada: variáveis conhecidas na SE25 + alvos da temporada
  seas5_uf.parquet              previsão sazonal ECMWF por UF × mês de emissão × horizonte (chuva em mm/dia)
  vizinhanca_uf.npz             W1 (binária), W1 de contato, W2
  relatorio_validacao_uf.md     todas as checagens

Uso:  python scripts/03_preparar_tabela_uf.py  (~10–20 min; o climate.csv tem 1,7 GB)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.csv as pv
import pyarrow.compute as pc
from scipy import sparse
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # raiz do repositório
BRUTOS = os.path.join(RAIZ, 'dados', 'brutos')
DERIVADOS = os.path.join(RAIZ, 'dados', 'derivados')
V4_DADOS = os.path.join(RAIZ, 'dados', 'v4')                        # saída do 02_preparar_painel_v4.py
SRC = os.path.join(RAIZ, 'src')
SAIDA = os.path.join(RAIZ, 'dados', 'uf')
os.makedirs(SAIDA, exist_ok=True)

UF_EXCLUIDA = 32                        # Espírito Santo: fora do desafio
ANO_PESO = 2015                         # pesos fixos do clima (igual ao V4)
DATA_HIST_INI = pd.Timestamp('2009-10-04')   # histórico só para lags/janelas do clima
CLIMA = ['temp_min', 'temp_med', 'temp_max', 'precip_min', 'precip_med', 'precip_max', 'pressure_min',
         'pressure_med', 'pressure_max', 'rel_humid_min', 'rel_humid_med', 'rel_humid_max', 'thermal_range',
         'rainy_days']
LAGS_CLIMA = [1, 2, 3, 4, 5]
LAGS_OCEANO = [6, 7, 8, 9]
t0 = time.time()
CHECKS = []


def log(msg):
    print(f'[{time.time() - t0:6.0f}s] {msg}', flush=True)


def checar(nome, ok, detalhe=''):
    CHECKS.append((nome, bool(ok), detalhe))
    print(f'   {"OK " if ok else "FALHOU"} · {nome} {detalhe}', flush=True)


def ler(nome, cols, tipos=None):
    return pv.read_csv(os.path.join(BRUTOS, nome),
                       convert_options=pv.ConvertOptions(include_columns=cols, column_types=tipos or {}))


# =====================================================================================================================
# 1. Municípios, UFs, calendário, população
# =====================================================================================================================
mapa = pd.read_csv(os.path.join(BRUTOS, 'map_regional_health.csv'), usecols=['geocode', 'uf_code', 'uf'])
mapa = mapa[mapa.uf_code != UF_EXCLUIDA].drop_duplicates('geocode')
geos = np.sort(mapa.geocode.values)
N = len(geos)
pos_geo = pd.Series(np.arange(N), index=geos)
uf_of = mapa.set_index('geocode').loc[geos, 'uf_code'].values
ufs = np.sort(np.unique(uf_of))
U = len(ufs)
A = (uf_of[None, :] == ufs[:, None]).astype(np.float64)             # (U, N): município -> UF
sigla = mapa.drop_duplicates('uf_code').set_index('uf_code').uf.to_dict()
log(f'{N} municípios em {U} UFs (sem o ES)')

den = ler('dengue.csv', ['geocode', 'date', 'casos'], {'geocode': 'int64', 'casos': 'float64'}).to_pandas()
den['date'] = pd.to_datetime(den['date'])
den = den[den.geocode.isin(geos)]
checar('dengue: sem (município, semana) duplicado', not den.duplicated(['geocode', 'date']).any())
datas_tab = pd.DatetimeIndex(np.sort(den.date.unique()))
checar('dengue: todas as datas são domingo', bool((datas_tab.dayofweek == 6).all()))
checar('dengue: semanas contínuas', bool((np.diff(datas_tab.values).astype('timedelta64[D]').astype(int) == 7).all()))
datas = pd.date_range(DATA_HIST_INI, datas_tab[-1], freq='7D')       # grade com histórico do clima
T = len(datas)
i0 = int(np.searchsorted(datas, datas_tab[0]))                        # 1ª semana da tabela (2010-01-03)
pos_data = pd.Series(np.arange(T), index=datas)
log(f'tabela: {datas_tab[0].date()} → {datas_tab[-1].date()} ({len(datas_tab)} semanas); histórico do clima desde {datas[0].date()}')


def matriz(df, col, preencher=np.nan):
    m = np.full((N, T), preencher, dtype=np.float64)
    ii = pos_geo.reindex(df.geocode.values).values
    tt = pos_data.reindex(df.date.values).values
    ok = ~np.isnan(ii) & ~np.isnan(tt)
    m[ii[ok].astype(int), tt[ok].astype(int)] = df[col].values[ok]
    return m


CASOS = matriz(den, 'casos')
checar('dengue: grade município × semana completa desde 2010', not np.isnan(CASOS[:, i0:]).any())
del den

# população: igual ao V4 (ano faltante → extrapolação com crescimento limitado a ±3%/ano), para a incidência bater
pop = pd.read_csv(os.path.join(BRUTOS, 'datasus_population_2001_2025.csv'))
pop = pop[pop.geocode.isin(geos)]
ult = pop[pop.year == pop.year.max()].set_index('geocode').population
pen = pop[pop.year == pop.year.max() - 1].set_index('geocode').population
cresc = (ult / pen).clip(0.97, 1.03).fillna(1.0)
ano_max_real = int(pop.year.max())
extra = [pd.DataFrame({'geocode': ult.index, 'year': a, 'population': (ult * cresc ** i).round().values})
         for i, a in enumerate(range(ano_max_real + 1, 2028), start=1)]
pop = pd.concat([pop] + extra, ignore_index=True)
pv_pop = pop.pivot(index='geocode', columns='year', values='population').reindex(index=geos).ffill(axis=1).bfill(axis=1)
checar('população: todo município tem valor em todo ano', not pv_pop.isna().any().any())
POP = pv_pop[datas.year].values.astype(np.float64)                    # (N, T) população do ano da semana
w_clima = pv_pop[ANO_PESO].values.astype(np.float64)                  # pesos fixos do clima
pop_extrap = (datas.year > ano_max_real).astype(np.int8)

# =====================================================================================================================
# 2. Funções de agregação
# =====================================================================================================================


def soma_uf(M):
    return A @ np.nan_to_num(M)


def media_pond_uf(M, w=w_clima):
    """média ponderada por UF ignorando municípios sem dado (pesos renormalizados). Devolve (média, cobertura)."""
    ok = ~np.isnan(M)
    num = A @ (w[:, None] * np.where(ok, M, 0.0))
    den_ = A @ (w[:, None] * ok)
    tot = (A @ w)[:, None]
    return np.where(den_ > 0, num / np.maximum(den_, 1e-12), np.nan), den_ / tot


def dp_pond_uf(M, media, w=w_clima):
    ok = ~np.isnan(M)
    num = A @ (w[:, None] * np.where(ok, M, 0.0) ** 2)
    den_ = A @ (w[:, None] * ok)
    var = num / np.maximum(den_, 1e-12) - media ** 2
    return np.sqrt(np.clip(var, 0, None))


def desloca(M, k):
    out = np.full_like(M, np.nan)
    out[:, k:] = M[:, :-k]
    return out


def rolling(M, k, fn):
    df = pd.DataFrame(M.T)
    r = df.rolling(k, min_periods=k)
    return (r.mean() if fn == 'mean' else r.std()).to_numpy().T


def log_inc(casos, pop_):
    return np.log1p(np.clip(casos, 0, None) / np.maximum(pop_, 1) * 1e5)


COL = {}                       # nome -> matriz (U, T)

# =====================================================================================================================
# 3. Casos, população, incidência, física, imunidade, espalhamento dentro da UF
# =====================================================================================================================
COL['casos'] = soma_uf(CASOS)
COL['pop'] = soma_uf(POP)
COL['pop_extrapolada'] = np.repeat(pop_extrap[None, :], U, 0).astype(float)
COL['log_pop'] = np.log(COL['pop'])
COL['incidencia'] = COL['casos'] / COL['pop'] * 1e5
COL['log_inc_dengue'] = log_inc(COL['casos'], COL['pop'])
for k in (1, 2):
    COL[f'log_inc_dengue_lag{k}'] = desloca(COL['log_inc_dengue'], k)
# espalhamento: fração de municípios (e da população) com ≥1 caso na semana
tem = (np.nan_to_num(CASOS) > 0).astype(float)
COL['pct_mun_com_casos'] = (A @ tem) / A.sum(1)[:, None]
COL['pct_pop_mun_com_casos'] = (A @ (POP * tem)) / COL['pop']
# urbanização: fração da população em municípios com ≥ 100 mil habitantes
COL['pct_pop_mun_100k'] = (A @ (POP * (POP >= 1e5))) / COL['pop']

# física (Savitzky-Golay janela 7, ordem 3, deslocada 3 semanas = causal), na série da UF
y = np.log1p(COL['casos'][:, i0:])
for der, nome in ((0, 'casos_suavizados'), (1, 'velocidade'), (2, 'aceleracao')):
    c = savgol_filter(y, 7, 3, deriv=der, axis=1)
    causal = np.empty_like(c)
    causal[:, :3] = c[:, [0]]
    causal[:, 3:] = c[:, :-3]
    full = np.full((U, T), np.nan)
    full[:, i0:] = causal
    COL[nome] = np.expm1(np.maximum(full, 0)) if der == 0 else full

# imunidade: EWM (meia-vida 104 semanas) da incidência da UF (linear na incidência: refeita sobre a série da UF)
alpha = 1 - np.exp(np.log(0.5) / 104)
ewm = np.full((U, T), np.nan)
ewm[:, i0] = COL['incidencia'][:, i0]
for k in range(i0 + 1, T):
    ewm[:, k] = (1 - alpha) * ewm[:, k - 1] + alpha * COL['incidencia'][:, k]
COL['imunidade_ewm'] = ewm
log('casos, incidência, física e imunidade prontos')

# =====================================================================================================================
# 4. Chikungunya
# =====================================================================================================================
chk = ler('chikungunya.csv', ['geocode', 'date', 'casos'], {'geocode': 'int64', 'casos': 'float64'}).to_pandas()
chk['date'] = pd.to_datetime(chk['date'])
chk = chk[chk.geocode.isin(geos)]
checar('chikungunya: sem duplicata', not chk.duplicated(['geocode', 'date']).any())
ini_chk = chk.date.min()
CHK = matriz(chk, 'casos')
CHK[:, datas < ini_chk] = 0.0
checar('chikungunya: grade completa desde o início da série', not np.isnan(CHK[:, datas >= ini_chk]).any())
COL['casos_chikungunya'] = soma_uf(CHK)
COL['log_inc_chiku'] = log_inc(COL['casos_chikungunya'], COL['pop'])
COL['chik_disponivel'] = np.repeat((datas >= ini_chk).astype(float)[None, :], U, 0)
chk_soma_bruta = chk.merge(mapa[['geocode', 'uf_code']], on='geocode').groupby(['uf_code', 'date']).casos.sum()
del chk
log(f'chikungunya desde {ini_chk.date()}')

# =====================================================================================================================
# 5. Clima observado: intensivas (média ponderada) e não lineares (município → UF)
# =====================================================================================================================
log('lendo climate.csv (1,7 GB) ...')
tc = ler('climate.csv', ['date', 'geocode'] + CLIMA, {'geocode': 'int64', **{c: 'float64' for c in CLIMA}})
tc = tc.filter(pc.greater_equal(pc.cast(tc['date'], 'string'), str(DATA_HIST_INI.date())))
cli = tc.to_pandas()
del tc
cli['date'] = pd.to_datetime(cli['date'])
cli = cli[cli.geocode.isin(geos) & (cli.date <= datas[-1])]
checar('clima: sem (município, semana) duplicado', not cli.duplicated(['geocode', 'date']).any())
amostra_cli = cli.sample(20000, random_state=1)
CLI = {v: matriz(cli, v) for v in CLIMA}
sem_clima = np.flatnonzero(np.isnan(CLI['temp_med'][:, i0:]).all(1))
log(f'clima: {len(sem_clima)} municípios sem nenhum dado (saem da média; pesos renormalizados): '
    f'{[int(geos[i]) for i in sem_clima]}')
del cli

# não lineares, no município
t_, rh = CLI['temp_med'], CLI['rel_humid_med']
CLI['vpd'] = 0.611 * np.exp(17.27 * t_ / (t_ + 237.3)) * (1 - rh / 100)
CLI['hti'] = CLI['precip_med'] * CLI['temp_med']
CLI['volatilidade_temp_4w'] = rolling(CLI['temp_med'], 4, 'std')
CLI['volatilidade_precip_4w'] = rolling(CLI['precip_med'], 4, 'std')

for v in CLIMA + ['vpd', 'hti', 'volatilidade_temp_4w', 'volatilidade_precip_4w']:
    COL[v], cob = media_pond_uf(CLI[v])
    if v == 'temp_med':
        COL['cobertura_clima_pop'] = cob
COL['temp_med_dp_mun'] = dp_pond_uf(CLI['temp_med'], COL['temp_med'])
COL['precip_med_dp_mun'] = dp_pond_uf(CLI['precip_med'], COL['precip_med'])

# lineares no tempo: na série da UF
COL['amplitude_termica_dinamica2'] = rolling(COL['thermal_range'], 2, 'mean')
COL['amplitude_termica_dinamica3'] = rolling(COL['thermal_range'], 3, 'mean')
for v in CLIMA + ['vpd']:
    for k in LAGS_CLIMA:
        COL[f'{v}_lag{k}'] = desloca(COL[v], k)
for k in (2, 3):
    COL[f'hti_lag{k}'] = desloca(COL['hti'], k)

# checagem: média ponderada fica entre o mínimo e o máximo municipal (amostra de UF × semana)
rng = np.random.default_rng(0)
ok_int = True
for _ in range(300):
    u, t = rng.integers(U), rng.integers(i0, T)
    vals = CLI['temp_med'][uf_of == ufs[u], t]
    vals = vals[~np.isnan(vals)]
    if len(vals):
        ok_int &= vals.min() - 1e-9 <= COL['temp_med'][u, t] <= vals.max() + 1e-9
checar('clima: média ponderada dentro do intervalo [mín, máx] dos municípios (300 amostras)', ok_int)
# checagem: lag na série da UF = agregação dos lags municipais (pesos fixos ⇒ comutam)
lag_mun, _ = media_pond_uf(desloca(CLI['temp_med'], 3))
dif = np.nanmax(np.abs(lag_mun[:, i0:] - COL['temp_med_lag3'][:, i0:]))
checar('clima: lag da série da UF = UF dos lags municipais', dif < 1e-8, f'(máx. dif. {dif:.2e})')
# checagem: Jensen (quanto a VPD da média difere da média da VPD)
vpd_da_media = 0.611 * np.exp(17.27 * COL['temp_med'] / (COL['temp_med'] + 237.3)) * (1 - COL['rel_humid_med'] / 100)
jensen = np.nanmean(np.abs(vpd_da_media[:, i0:] - COL['vpd'][:, i0:]) / np.abs(COL['vpd'][:, i0:]))
checar('VPD calculada no município antes de agregar (diferença média para a "VPD da média")', True, f'= {jensen:.1%}')
# checagem pontual contra o CSV bruto (média ponderada refeita do zero)
ok_pt = True
for _, r in amostra_cli[amostra_cli.date >= datas_tab[0]].head(5).iterrows():
    u = int(np.searchsorted(ufs, uf_of[pos_geo[r.geocode]]))
    t = int(pos_data[r.date])
    m = (uf_of == ufs[u]) & ~np.isnan(CLI['temp_med'][:, t])
    esperado = np.sum(w_clima[m] * CLI['temp_med'][m, t]) / np.sum(w_clima[m])
    ok_pt &= abs(esperado - COL['temp_med'][u, t]) < 1e-9
ok_csv = np.allclose(CLI['temp_med'][pos_geo[amostra_cli.geocode].values, pos_data[amostra_cli.date].values],
                     amostra_cli.temp_med.values, equal_nan=True)
checar('clima: grade município × semana = CSV (20 mil amostras)', ok_csv)
checar('clima: média ponderada refeita à mão = tabela (5 amostras)', ok_pt)
log('clima observado pronto')
del CLI

# =====================================================================================================================
# 6. Previsão sazonal ECMWF (SEAS5): tabela longa por emissão e coluna semanal (emissão = mês-alvo − 1)
# =====================================================================================================================
fc = pd.read_csv(os.path.join(BRUTOS, 'forecasting_climate.csv'))
fc['reference_month'] = pd.to_datetime(fc['reference_month'].astype(str).str[:10])
fc = fc[fc.geocode.isin(geos)]
checar('SEAS5: sem (município, emissão, horizonte) duplicado',
       not fc.duplicated(['geocode', 'reference_month', 'forecast_months_ahead']).any())
fc['precip_tot'] = fc['precip_tot'] * 86400.0                        # kg m⁻² s⁻¹ → mm/dia (igual ao refazer_season)
fc = fc.merge(mapa[['geocode', 'uf_code']], on='geocode')
fc['w'] = pv_pop.loc[fc.geocode.values, ANO_PESO].values
FCV = ['temp_med', 'umid_med', 'precip_tot']
for v in FCV:
    fc[f'{v}_w'] = fc[v] * fc.w
    fc[f'{v}_ok'] = fc[v].notna() * fc.w
g = fc.groupby(['uf_code', 'reference_month', 'forecast_months_ahead'])
seas = pd.DataFrame({v: g[f'{v}_w'].sum() / g[f'{v}_ok'].sum() for v in FCV}).reset_index()
cob = g['temp_med_ok'].sum().reset_index(name='w_ok')
tot = pd.Series(A @ w_clima, index=ufs)
seas['cobertura_pop'] = cob.w_ok.values / tot.loc[cob.uf_code.values].values
seas = seas.rename(columns={'forecast_months_ahead': 'meses_a_frente', 'reference_month': 'mes_emissao'})
seas.to_parquet(os.path.join(SAIDA, 'seas5_uf.parquet'), index=False)
del fc
# coluna semanal: mês-alvo = mês da quarta-feira; emissão = mês-alvo − 1 (horizonte 2 na convenção do arquivo)
alvo = (datas + pd.Timedelta(days=3)).to_period('M')
emis = alvo - 1
sidx = seas.set_index(['uf_code', 'mes_emissao', 'meses_a_frente'])
for v in FCV:
    M = np.full((U, T), np.nan)
    for j, u in enumerate(ufs):
        chave_e = [(u, e.to_timestamp(), 2) for e in emis]
        chave_a = [(u, a.to_timestamp(), 1) for a in alvo]                 # fallback: sem emissão anterior
        val = sidx[v].reindex(chave_e).values
        val_a = sidx[v].reindex(chave_a).values
        M[j] = np.where(np.isnan(val), val_a, val)
    COL[f'{v}_forecast'] = M
COL['forecast_mes_emissao_anterior'] = np.repeat(
    (~np.isnan(sidx['temp_med'].reindex([(ufs[0], e.to_timestamp(), 2) for e in emis]).values)).astype(float)[None, :], U, 0)
for v in FCV:
    COL[f'{v}_forecast_lag1'] = desloca(COL[f'{v}_forecast'], 1)
log(f'SEAS5: {len(seas):,} linhas (UF × emissão × horizonte), {seas.mes_emissao.min().date()} → {seas.mes_emissao.max().date()}')

# =====================================================================================================================
# 7. Oceano (nacional) e calendário
# =====================================================================================================================
oc = pd.read_csv(os.path.join(BRUTOS, 'ocean_climate_oscillations.csv'))
oc['date'] = pd.to_datetime(oc['date'])
oc['date'] = oc.date - pd.to_timedelta((oc.date.dt.dayofweek + 1) % 7, unit='D')
oc = oc.groupby('date', as_index=False)[['enso', 'iod', 'pdo']].mean().sort_values('date')
oc = pd.merge_asof(pd.DataFrame({'date': datas}), oc, on='date', direction='backward')
for v in ('enso', 'iod', 'pdo'):
    COL[v] = np.repeat(oc[v].values[None, :], U, 0)
    for k in LAGS_OCEANO:
        COL[f'{v}_lag{k}'] = desloca(COL[v], k)
doy = (datas + pd.Timedelta(days=3)).dayofyear.values
COL['seno_semana'] = np.repeat(np.sin(2 * np.pi * doy / 365.25)[None, :], U, 0)
COL['cosseno_semana'] = np.repeat(np.cos(2 * np.pi * doy / 365.25)[None, :], U, 0)

# =====================================================================================================================
# 8. Afya (acessos a conteúdo de dengue): diário → semanal no município → soma na UF; taxa por 100 mil
# =====================================================================================================================
log('lendo access_afya ...')
af = ler('access_afya_dengue_2021_2026.csv', ['access_date', 'geocode', 'access_count'],
         {'geocode': 'int64', 'access_count': 'float64'}).to_pandas()
af['access_date'] = pd.to_datetime(af['access_date'])
af = af.groupby(['geocode', 'access_date'], as_index=False).access_count.sum()        # soma duplicatas do dia
d_ini, d_fim = af.access_date.min(), af.access_date.max()
af['date'] = af.access_date - pd.to_timedelta((af.access_date.dt.weekday + 1) % 7, unit='D')
sem_af = af.groupby(['geocode', 'date']).agg(acc=('access_count', 'sum'), dias=('access_count', 'size')).reset_index()
cobertas = datas[(datas >= d_ini) & (datas + pd.Timedelta(days=6) <= d_fim)]
sem_af = sem_af[sem_af.date.isin(cobertas) & sem_af.geocode.isin(geos)]
AF = matriz(sem_af, 'acc')
t_cob = np.isin(datas, cobertas)
AF[:, t_cob] = np.nan_to_num(AF[:, t_cob])                           # município ausente numa semana coberta = 0
AF[:, ~t_cob] = np.nan
afya = soma_uf(AF)
afya[:, ~t_cob] = np.nan
COL['afya_acessos'] = afya
COL['afya_log_acessos_100k'] = np.log1p(afya / COL['pop'] * 1e5)
d = np.full_like(afya, np.nan)
d[:, 1:] = np.log1p(afya[:, 1:]) - np.log1p(afya[:, :-1])
COL['afya_log_diff'] = d
COL['afya_disponivel'] = np.repeat(t_cob.astype(float)[None, :], U, 0)
# checagem: UF × semana = soma diária do bruto
af = af.merge(mapa[['geocode', 'uf_code']], on='geocode')
ok_af = True
for t in rng.choice(np.flatnonzero(t_cob), 5, replace=False):
    u = rng.integers(U)
    esp = af[(af.uf_code == ufs[u]) & (af.access_date >= datas[t]) & (af.access_date <= datas[t] + pd.Timedelta(days=6))].access_count.sum()
    ok_af &= abs(esp - afya[u, t]) < 1e-6
checar('Afya: UF × semana = soma dos acessos diários do bruto (5 amostras)', ok_af)
del af, sem_af, AF

# =====================================================================================================================
# 9. Vizinhança entre UFs e variáveis de vizinhos (recalculadas no grafo das UFs)
# =====================================================================================================================
Wm = sparse.load_npz(os.path.join(DERIVADOS, 'W1_queen_municipios.npz')).tocoo()
nos = pd.read_csv(os.path.join(DERIVADOS, 'nos_municipios.csv'), usecols=['geocode', 'node_id'])
g_of = nos.set_index('node_id').geocode
ua = pd.Series(uf_of, index=geos).reindex(g_of.loc[Wm.row].values).values
ub = pd.Series(uf_of, index=geos).reindex(g_of.loc[Wm.col].values).values
ok = ~np.isnan(ua) & ~np.isnan(ub) & (ua != ub)
CONTATO = np.zeros((U, U))
for a_, b_ in zip(np.searchsorted(ufs, ua[ok].astype(int)), np.searchsorted(ufs, ub[ok].astype(int))):
    CONTATO[a_, b_] += 1
CONTATO = (CONTATO + CONTATO.T) / 2                                   # nº de pares de municípios vizinhos na fronteira
W1 = (CONTATO > 0).astype(float)
W2 = ((W1 @ W1 > 0) & (W1 == 0)).astype(float)
np.fill_diagonal(W2, 0)
checar('vizinhança: W1 simétrica e sem diagonal', np.allclose(W1, W1.T) and np.trace(W1) == 0)
checar('vizinhança: W2 não contém W1 nem a própria UF', (W2 * W1).sum() == 0 and np.trace(W2) == 0)
checar('vizinhança: nenhuma UF isolada', (W1.sum(1) > 0).all(),
       f'(grau mín. {int(W1.sum(1).min())}, máx. {int(W1.sum(1).max())})')
np.savez(os.path.join(SAIDA, 'vizinhanca_uf.npz'), W1=W1, W1_contato=CONTATO, W2=W2, ufs=ufs)


def viz_razao(W, cont, base):
    """razão agregada: soma(contagem dos vizinhos) / soma(base dos vizinhos), com pesos W."""
    return (W @ np.nan_to_num(cont)) / np.maximum(W @ base, 1)


def viz_media_pond(W, M):
    """média de uma intensiva dos vizinhos, ponderada por W × população de cada vizinho."""
    pw = COL['pop']
    ok = ~np.isnan(M)
    return (W @ (pw * np.where(ok, M, 0))) / np.maximum(W @ (pw * ok), 1e-12)


for nomeW, W in (('w1', W1), ('w1c', CONTATO), ('w2', W2)):
    casos_v = W @ COL['casos']
    pop_v = W @ COL['pop']
    COL[f'{nomeW}_log_inc_dengue'] = np.log1p(casos_v / np.maximum(pop_v, 1) * 1e5)
    COL[f'{nomeW}_log_inc_chiku'] = np.log1p(viz_razao(W, COL['casos_chikungunya'], COL['pop']) * 1e5)
    COL[f'{nomeW}_log_pop'] = np.log(np.maximum(pop_v, 1))
    # física e imunidade dos vizinhos: sobre a série AGREGADA dos vizinhos (não média de derivadas)
    yv = np.log1p(casos_v[:, i0:])
    for der, nome in ((1, 'velocidade'), (2, 'aceleracao')):
        c = savgol_filter(yv, 7, 3, deriv=der, axis=1)
        causal = np.empty_like(c)
        causal[:, :3] = c[:, [0]]
        causal[:, 3:] = c[:, :-3]
        full = np.full((U, T), np.nan)
        full[:, i0:] = causal
        COL[f'{nomeW}_{nome}'] = full
    inc_v = casos_v / np.maximum(pop_v, 1) * 1e5
    e = np.full((U, T), np.nan)
    e[:, i0] = inc_v[:, i0]
    for k in range(i0 + 1, T):
        e[:, k] = (1 - alpha) * e[:, k - 1] + alpha * inc_v[:, k]
    COL[f'{nomeW}_imunidade_ewm'] = e
    for v in ('temp_med', 'precip_med', 'rel_humid_med', 'temp_med_forecast', 'umid_med_forecast', 'precip_tot_forecast'):
        COL[f'{nomeW}_{v}'] = viz_media_pond(W, COL[v])
    if nomeW != 'w2':
        COL[f'{nomeW}_log_inc_dengue_lag1'] = desloca(COL[f'{nomeW}_log_inc_dengue'], 1)
# checagem pontual do vizinho (razão agregada refeita à mão)
u, t = 5, T - 30
viz = np.flatnonzero(W1[u])
esp = np.log1p(COL['casos'][viz, t].sum() / COL['pop'][viz, t].sum() * 1e5)
checar('vizinhos: w1_log_inc_dengue refeita à mão (razão agregada, não média de taxas)',
       abs(esp - COL['w1_log_inc_dengue'][u, t]) < 1e-9, f'({sigla[ufs[u]]}, {datas[t].date()})')
log('vizinhança pronta')

# =====================================================================================================================
# 10. Estáticas: % da população por bioma e por Köppen (pesos de 2015)
# =====================================================================================================================
env = pd.read_csv(os.path.join(BRUTOS, 'environ_vars.csv'))
env = env[env.geocode.isin(geos)].set_index('geocode').reindex(geos)
checar('ambiente: bioma e Köppen para todos os municípios', not env[['biome', 'koppen']].isna().any().any())
est = pd.DataFrame({'uf_code': ufs, 'uf': [sigla[u] for u in ufs]})
for col, pref in (('biome', 'pct_pop_bioma_'), ('koppen', 'pct_pop_koppen_')):
    for cl in sorted(env[col].unique()):
        m = (env[col].values == cl).astype(float)
        est[pref + str(cl).replace(' ', '_')] = (A @ (w_clima * m)) / (A @ w_clima)
    grupo = env[col].astype(str).str[0] if col == 'koppen' else None
    if col == 'koppen':
        for cl in sorted(grupo.unique()):
            m = (grupo.values == cl).astype(float)
            est[f'pct_pop_koppen_grupo_{cl}'] = (A @ (w_clima * m)) / (A @ w_clima)
fb = [c for c in est if c.startswith('pct_pop_bioma_')]
fk = [c for c in est if c.startswith('pct_pop_koppen_') and 'grupo' not in c]
checar('estáticas: frações de bioma somam 1 em cada UF', np.allclose(est[fb].sum(1), 1))
checar('estáticas: frações de Köppen somam 1 em cada UF', np.allclose(est[fk].sum(1), 1))
est['bioma_predominante'] = est[fb].idxmax(1).str.replace('pct_pop_bioma_', '')
est['koppen_predominante'] = est[fk].idxmax(1).str.replace('pct_pop_koppen_', '')
est['grau_vizinhanca'] = W1.sum(1)
est['contato_fronteira_pares'] = CONTATO.sum(1)
est['n_municipios'] = A.sum(1)
est['macro'] = ufs // 10
est.to_parquet(os.path.join(SAIDA, 'tabela_uf_estatica.parquet'), index=False)

# =====================================================================================================================
# 11. Tabela semanal (formato longo: uma linha por UF × semana)
# =====================================================================================================================
cal = pd.read_parquet(os.path.join(V4_DADOS, 'calendario.parquet'))[['date', 'epiweek', 'woy', 'season']]
tt = np.arange(i0, T)
tab = pd.DataFrame({'uf_code': np.repeat(ufs, len(tt)), 'date': np.tile(datas[tt], U)})
tab['uf'] = tab.uf_code.map(sigla)
for nome, M in COL.items():
    tab[nome] = M[:, tt].ravel().astype(np.float32 if nome not in ('casos', 'pop', 'casos_chikungunya', 'afya_acessos') else np.float64)
tab = tab.merge(cal, on='date', how='left')
tab = tab.merge(est.drop(columns=['uf']), on='uf_code', how='left')
fixas = ['uf_code', 'uf', 'date', 'epiweek', 'woy', 'season', 'macro']
tab = tab[fixas + [c for c in tab.columns if c not in fixas]]
checar('tabela semanal: sem (UF, semana) duplicado', not tab.duplicated(['uf_code', 'date']).any())
checar('tabela semanal: UF × semana completa', len(tab) == U * len(datas_tab), f'({len(tab):,} linhas)')
checar('tabela semanal: toda semana tem temporada e epiweek', tab[['season', 'epiweek']].notna().all().all())

# contra o bruto e contra o V4
v4 = pd.read_parquet(os.path.join(V4_DADOS, 'painel_uf.parquet')).drop(columns=['uf_code']).rename(columns={'unidade': 'uf_code'}).set_index(['uf_code', 'date'])
m = tab.set_index(['uf_code', 'date']).join(v4, rsuffix='_v4', how='inner')
checar('casos por UF = painel_uf do V4 (exato)', np.array_equal(m.casos.values, m.casos_v4.values), f'({len(m):,} linhas)')
checar('população por UF = painel_uf do V4 (exato)', np.array_equal(m['pop'].values, m.pop_v4.values))
for v in ('temp_min', 'temp_med', 'temp_max', 'precip_med', 'rainy_days', 'rel_humid_med', 'thermal_range'):
    dif = np.nanmax(np.abs(m[v].values - m[f'{v}_v4'].values))
    checar(f'clima {v} = painel_uf do V4 (mesmos pesos)', dif < 1e-3, f'(máx. dif. {dif:.1e})')
chk_tab = tab.set_index(['uf_code', 'date']).casos_chikungunya
cmp_ = chk_soma_bruta.reindex(chk_tab.index).fillna(0)
checar('chikungunya por UF = soma do bruto', np.allclose(cmp_.values, chk_tab.values))

# física causal: o valor em t não muda se a série for cortada em t
ok_c = True
for u in range(3):
    for corte in (200, 400, 600):
        yc = np.log1p(COL['casos'][u, i0:i0 + corte + 1])
        c = savgol_filter(yc, 7, 3, deriv=1)
        ok_c &= abs(c[-4] - COL['velocidade'][u, i0 + corte]) < 1e-9
checar('física: velocidade em t só usa dados até t (série cortada em t dá o mesmo valor)', ok_c)
tab.to_parquet(os.path.join(SAIDA, 'tabela_uf_semanal.parquet'), index=False)
log(f'tabela_uf_semanal: {len(tab):,} linhas × {tab.shape[1]} colunas')

# =====================================================================================================================
# 12. Tabela por temporada (UF × temporada): variáveis conhecidas na SE25 + alvos
# =====================================================================================================================
sys.path.insert(0, SRC)
import v4_lib as V  # noqa: E402

D = V.Dados(V4_DADOS, 'uf')
assert np.array_equal(D.unidades, ufs)
anos = [s for s in range(2011, int(datas_tab[-1].year)) if D.corte(s) <= D.ultimo_obs]
linhas = []
for s in anos:
    c = D.corte(s)
    f = V.features_temporada(D, s, c).drop(columns=['uf_code']).reset_index().rename(columns={'unidade': 'uf_code'})
    f.insert(1, 'season', s)
    # vizinhos pela razão agregada (o v4_lib usa média simples dos logs dos vizinhos: mantida como viz_*)
    ini1 = D.idx_temporada(s - 1)[0]
    for nomeW, W in (('w1', W1), ('w1c', CONTATO)):
        for nome, (a, b) in (('ar1_parcial', (ini1, c)), ('inc_ult8', (c - 7, c))):
            casos_v = W @ D.soma(a, b)
            f[f'vizp_{nomeW}_{nome}'] = np.log1p(casos_v / np.maximum(W @ D.POP[c], 1) * 1e5)
        k8 = np.arange(8) - 3.5
        linc_v = np.log1p((D.Y[c - 7:c + 1] @ W.T) / np.maximum(D.POP[c - 7:c + 1] @ W.T, 1) * 1e5)
        f[f'vizp_{nomeW}_tend8'] = (k8 @ linc_v) / (k8 @ k8)
    # espalhamento e urbanização na SE25
    janela = tab[(tab.date > D.datas[c - 8]) & (tab.date <= D.datas[c])]
    for v in ('pct_mun_com_casos', 'pct_pop_mun_com_casos'):
        f[f'{v}_ult8'] = janela.groupby('uf_code')[v].mean().reindex(ufs).values
    f['pct_pop_mun_100k'] = tab[tab.date == D.datas[c]].set_index('uf_code').pct_pop_mun_100k.reindex(ufs).values
    # alvos da temporada s (SE41/s → SE40/s+1)
    ii = D.idx_temporada(s)
    completa = D.temporada_completa(s)
    yy = D.Y[ii]
    f['alvo_completo'] = completa
    f['alvo_casos_temporada'] = yy.sum(0) if completa else np.nan
    f['alvo_log_inc_temporada'] = np.log1p(yy.sum(0) / D.POP[ii[0]] * 1e5) if completa else np.nan
    f['alvo_pico_casos'] = yy.max(0) if completa else np.nan
    f['alvo_semana_pico'] = np.argmax(yy, 0).astype(float) if completa else np.nan
    linhas.append(f)
tem = pd.concat(linhas, ignore_index=True)
tem['uf'] = tem.uf_code.map(sigla)
tem = tem.merge(est.drop(columns=['uf', 'macro']), on='uf_code', how='left')
# alvo = soma da tabela semanal da mesma temporada
x = tab[tab.season.isin(tem.season.unique())].groupby(['uf_code', 'season']).casos.sum()
y_ = tem[tem.alvo_completo].set_index(['uf_code', 'season']).alvo_casos_temporada
checar('temporada: alvo = soma das semanas da tabela semanal', np.allclose(x.reindex(y_.index).values, y_.values))
checar('temporada: nenhuma variável usa dado depois da SE25 (corte = índice da SE25 de cada temporada)',
       all(V.data_para_epiweek(D.datas[D.corte(s)]) == s * 100 + 25 for s in anos))
tem.to_parquet(os.path.join(SAIDA, 'tabela_uf_temporada.parquet'), index=False)
log(f'tabela_uf_temporada: {len(tem):,} linhas ({anos[0]}–{anos[-1]}) × {tem.shape[1]} colunas')

# =====================================================================================================================
# 13. Relatório
# =====================================================================================================================
falhas = [c for c in CHECKS if not c[1]]
with open(os.path.join(SAIDA, 'relatorio_validacao_uf.md'), 'w', encoding='utf-8') as fh:
    fh.write(f'# Validação das tabelas por UF\n\nGerado em {time.strftime("%d/%m/%Y %H:%M")} por `preparar_tabela_uf.py` '
             f'em {time.time() - t0:.0f} s.\n\n**{len(CHECKS) - len(falhas)} de {len(CHECKS)} checagens OK.**\n\n')
    fh.write('| Checagem | Resultado | Detalhe |\n|---|---|---|\n')
    for n, ok_, det in CHECKS:
        fh.write(f'| {n} | {"✅" if ok_ else "❌"} | {det} |\n')
    fh.write(f'\n## Arquivos\n\n- `tabela_uf_semanal.parquet`: {len(tab):,} linhas × {tab.shape[1]} colunas\n'
             f'- `tabela_uf_temporada.parquet`: {len(tem):,} linhas × {tem.shape[1]} colunas\n'
             f'- `tabela_uf_estatica.parquet`: {len(est)} linhas × {est.shape[1]} colunas\n'
             f'- `seas5_uf.parquet`: {len(seas):,} linhas\n- `vizinhanca_uf.npz`: W1, W1_contato, W2\n')
    fh.write('\n## Colunas da tabela semanal\n\n' + ', '.join(f'`{c}`' for c in tab.columns) + '\n')
    fh.write('\n## Colunas da tabela por temporada\n\n' + ', '.join(f'`{c}`' for c in tem.columns) + '\n')
log(f'pronto: {len(CHECKS) - len(falhas)}/{len(CHECKS)} checagens OK' + (f' · FALHAS: {[f[0] for f in falhas]}' if falhas else ''))
sys.exit(1 if falhas else 0)
