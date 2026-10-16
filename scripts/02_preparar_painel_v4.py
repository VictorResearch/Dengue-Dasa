"""
V4 · Preparação dos dados (roda no PC, não no Colab).

Lê os brutos de data_imdc_2026/variaveis/ e grava em V4/dados/ arquivos pequenos, prontos para subir no Drive:
  calendario.parquet      date, epiweek, woy, season, train_1..4, target_1..4
  painel_regiao.parquet   unidade (regional_geocode), uf_code, macro, date, casos, pop, clima ponderado pela população
  painel_uf.parquet       o mesmo no nível UF (unidade = uf_code)
  oceano.parquet          enso, iod, pdo alinhados ao domingo da semana epidemiológica
  seas5_regiao.parquet    previsão sazonal ECMWF (SEAS5) por região: reference_month, meses_a_frente, temp_med, precip_tot, umid_med
  seas5_uf.parquet        o mesmo por UF
  adj_regiao.npz          vizinhança entre regiões de saúde (derivada da vizinhança Queen dos municípios)
  adj_uf.npz              vizinhança entre UFs
  richards_regiao.parquet parâmetros estilo Episcanner (curva de Richards) por região × temporada: completa e truncada na SE25
  richards_uf.parquet     o mesmo por UF

Uso:  python scripts/02_preparar_painel_v4.py          (~10–15 min; o climate.csv tem 1,7 GB)
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import sparse

warnings.filterwarnings('ignore')
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # raiz do repositório
BRUTOS = os.path.join(RAIZ, 'dados', 'brutos')
DERIVADOS = os.path.join(RAIZ, 'dados', 'derivados')
SAIDA = os.path.join(RAIZ, 'dados', 'v4')
os.makedirs(SAIDA, exist_ok=True)
UF_EXCLUIDA = 32  # Espírito Santo fica fora do desafio
CLIMA_COLS = ['temp_min', 'temp_med', 'temp_max', 'precip_med', 'rainy_days', 'rel_humid_med', 'thermal_range']
t0 = time.time()


def log(msg):
    print(f'[{time.time() - t0:6.0f}s] {msg}', flush=True)


# ------------------------------------------------------------------ mapa município -> região -> UF
mapa = pd.read_csv(os.path.join(BRUTOS, 'map_regional_health.csv'),
                   usecols=['geocode', 'regional_geocode', 'macroregional_geocode', 'uf_code'])
mapa = mapa[mapa.uf_code != UF_EXCLUIDA]

# ------------------------------------------------------------------ casos + calendário oficial
flags = [f'{p}_{k}' for p in ('train', 'target') for k in range(1, 5)]
den = pd.read_csv(os.path.join(BRUTOS, 'dengue.csv'), usecols=['date', 'epiweek', 'geocode', 'casos'] + flags)
den['date'] = pd.to_datetime(den['date'])
cal = den.groupby('date')[['epiweek'] + flags].first().reset_index().sort_values('date')
cal['woy'] = (cal.epiweek % 100).astype(int)
# temporada s = SE41 do ano s até SE40 do ano s+1 (rótulo = ano de início)
ano = cal.epiweek // 100
cal['season'] = np.where(cal.woy >= 41, ano, ano - 1).astype(int)
cal.to_parquet(os.path.join(SAIDA, 'calendario.parquet'), index=False)
log(f'calendário: {cal.date.min().date()} → {cal.date.max().date()} ({len(cal)} semanas)')

den = den.merge(mapa, on='geocode', how='inner')
assert den.duplicated(['geocode', 'date']).sum() == 0, 'município × semana duplicado'

# ------------------------------------------------------------------ população (com extrapolação até 2027)
pop = pd.read_csv(os.path.join(BRUTOS, 'datasus_population_2001_2025.csv'))
pop = pop[pop.geocode.isin(mapa.geocode)]
ult = pop[pop.year == pop.year.max()].set_index('geocode').population
pen = pop[pop.year == pop.year.max() - 1].set_index('geocode').population
cresc = (ult / pen).clip(0.97, 1.03).fillna(1.0)
extra = []
for i, a in enumerate(range(pop.year.max() + 1, 2028), start=1):
    extra.append(pd.DataFrame({'geocode': ult.index, 'year': a, 'population': (ult * cresc ** i).round().values}))
pop = pd.concat([pop] + extra, ignore_index=True)
pop = pop.merge(mapa, on='geocode')
log(f'população: {pop.year.min()}–{pop.year.max()} (extrapolada após 2025)')


def painel(nivel):
    chave = 'regional_geocode' if nivel == 'regiao' else 'uf_code'
    c = den.groupby([chave, 'date'], as_index=False).casos.sum()
    c['year'] = c.date.dt.year
    p = pop.groupby([chave, 'year'], as_index=False).population.sum().rename(columns={'population': 'pop'})
    c = c.merge(p, on=[chave, 'year'], how='left').drop(columns='year')
    info = mapa.drop_duplicates(chave)[[chave, 'uf_code', 'macroregional_geocode']] if nivel == 'regiao' \
        else mapa.drop_duplicates('uf_code')[['uf_code']].assign(macroregional_geocode=lambda d: d.uf_code)
    c = c.merge(info, on=chave, how='left') if nivel == 'regiao' else c.assign(macroregional_geocode=c.uf_code)
    c = c.rename(columns={chave: 'unidade'})
    if nivel == 'uf':
        c['uf_code'] = c['unidade']
    return c


pr, pu = painel('regiao'), painel('uf')
log(f'casos: {pr.unidade.nunique()} regiões, {pu.unidade.nunique()} UFs')

# ------------------------------------------------------------------ clima semanal ponderado pela população de 2015
peso = pop[pop.year == 2015][['geocode', 'population']]
acc_r, acc_u = [], []
for ch in pd.read_csv(os.path.join(BRUTOS, 'climate.csv'), usecols=['date', 'geocode'] + CLIMA_COLS, chunksize=2_000_000):
    ch['date'] = pd.to_datetime(ch['date'])
    ch = ch[ch.date >= '2009-01-01'].merge(mapa, on='geocode').merge(peso, on='geocode')
    for c in CLIMA_COLS:
        ch[c] = ch[c] * ch.population
    for chave, acc in (('regional_geocode', acc_r), ('uf_code', acc_u)):
        acc.append(ch.groupby([chave, 'date'])[CLIMA_COLS + ['population']].sum())
    log(f'   clima: pedaço até {ch.date.max().date() if len(ch) else "-"}')


def fecha_clima(acc, chave):
    g = pd.concat(acc).groupby(level=[0, 1]).sum()
    for c in CLIMA_COLS:
        g[c] = g[c] / g['population']
    return g.drop(columns='population').reset_index().rename(columns={chave: 'unidade'})


pr = pr.merge(fecha_clima(acc_r, 'regional_geocode'), on=['unidade', 'date'], how='left')
pu = pu.merge(fecha_clima(acc_u, 'uf_code'), on=['unidade', 'date'], how='left')
for p_, nome in ((pr, 'regiao'), (pu, 'uf')):
    p_.sort_values(['unidade', 'date']).to_parquet(os.path.join(SAIDA, f'painel_{nome}.parquet'), index=False)
    log(f'painel_{nome}: {len(p_):,} linhas · clima faltando em {p_.temp_med.isna().mean():.1%}')

# ------------------------------------------------------------------ ENSO/IOD/PDO alinhados ao domingo (os brutos caem em seg/ter)
oc = pd.read_csv(os.path.join(BRUTOS, 'ocean_climate_oscillations.csv'))
oc['date'] = pd.to_datetime(oc['date'])
oc['date'] = oc.date - pd.to_timedelta((oc.date.dt.dayofweek + 1) % 7, unit='D')   # domingo da semana
oc = oc.groupby('date', as_index=False)[['enso', 'iod', 'pdo']].mean()
oc = pd.merge_asof(cal[['date']], oc.sort_values('date'), on='date', direction='backward')
oc.to_parquet(os.path.join(SAIDA, 'oceano.parquet'), index=False)
log(f'oceano: {oc.enso.notna().mean():.1%} das semanas com ENSO')

# ------------------------------------------------------------------ SEAS5 (previsão sazonal) ponderada pela população
fc = pd.read_csv(os.path.join(BRUTOS, 'forecasting_climate.csv'))
fc['reference_month'] = pd.to_datetime(fc['reference_month'].astype(str).str[:10])
fc = fc.merge(mapa, on='geocode').merge(peso, on='geocode')
for c in ('temp_med', 'precip_tot', 'umid_med'):
    fc[c] = fc[c] * fc.population
for chave, nome in (('regional_geocode', 'regiao'), ('uf_code', 'uf')):
    g = fc.groupby([chave, 'reference_month', 'forecast_months_ahead'])[['temp_med', 'precip_tot', 'umid_med', 'population']].sum()
    for c in ('temp_med', 'precip_tot', 'umid_med'):
        g[c] = g[c] / g['population']
    g = g.drop(columns='population').reset_index().rename(columns={chave: 'unidade', 'forecast_months_ahead': 'meses_a_frente'})
    g.to_parquet(os.path.join(SAIDA, f'seas5_{nome}.parquet'), index=False)
    log(f'seas5_{nome}: {len(g):,} linhas, {g.reference_month.min().date()} → {g.reference_month.max().date()}')
del fc

# ------------------------------------------------------------------ vizinhança entre regiões e entre UFs
W = sparse.load_npz(os.path.join(DERIVADOS, 'W1_queen_municipios.npz')).tocoo()
nos = pd.read_csv(os.path.join(DERIVADOS, 'nos_municipios.csv'), usecols=['geocode', 'node_id'])
g_of = nos.set_index('node_id').geocode
m_idx = mapa.set_index('geocode')
for chave, nome, ordem in (('regional_geocode', 'regiao', np.sort(pr.unidade.unique())),
                           ('uf_code', 'uf', np.sort(pu.unidade.unique()))):
    a = m_idx.loc[g_of.loc[W.row].values, chave].values
    b = m_idx.loc[g_of.loc[W.col].values, chave].values
    pos = {u: i for i, u in enumerate(ordem)}
    M = np.zeros((len(ordem), len(ordem)), dtype=np.float32)
    for x, y in zip(a, b):
        if x != y:
            M[pos[x], pos[y]] = M[pos[y], pos[x]] = 1
    np.savez(os.path.join(SAIDA, f'adj_{nome}.npz'), adj=M, unidades=ordem)
    log(f'adj_{nome}: {len(ordem)} unidades, {int(M.sum() / 2)} pares vizinhos, {int((M.sum(1) == 0).sum())} ilhas')


# ------------------------------------------------------------------ parâmetros estilo Episcanner (Richards)
def richards_cum(t, L, r, tau, a):
    return L * (1 + a * np.exp(-r * (t - tau))) ** (-1 / a)


def ajusta(y):
    """y = casos semanais da temporada (ou do trecho). Devolve (log_total, r, semana_pico, ok)."""
    tot = y.sum()
    if tot < 50 or (y > 0).sum() < 3:
        return np.log1p(tot), np.nan, np.nan, 0
    t = np.arange(len(y), dtype=float)
    C = np.cumsum(y)
    try:
        p, _ = curve_fit(richards_cum, t, C, p0=[tot * 1.1, 0.3, float(np.argmax(y)), 1.0],
                         bounds=([tot * 0.8, 0.01, -10, 0.05], [tot * 20, 3.0, len(y) + 30, 20]), maxfev=4000)
        L, r, tau, a = p
        pico = tau + np.log(a) / r
        return np.log1p(L), r, pico, 1
    except Exception:
        return np.log1p(tot), np.nan, float(np.argmax(y)), 0


for p_, nome in ((pr, 'regiao'), (pu, 'uf')):
    d = p_[['unidade', 'date', 'casos']].merge(cal[['date', 'season', 'woy']], on='date')
    linhas = []
    for (u, s), g in d.groupby(['unidade', 'season']):
        if len(g) < 50:          # temporada incompleta (2009/10 ou a atual)
            full = (np.nan,) * 4
        else:
            full = ajusta(g.casos.values.astype(float))
        # trecho observável até a SE25 (SE41 → SE25: 36 ou 37 semanas)
        tr = g[~((g.woy > 25) & (g.woy < 41))]
        trunc = ajusta(tr.casos.values.astype(float)) if len(tr) >= 30 else (np.nan,) * 4
        linhas.append((u, s, *full, *trunc))
    rich = pd.DataFrame(linhas, columns=['unidade', 'season', 'rich_logL', 'rich_r', 'rich_pico', 'rich_ok',
                                         'richT_logL', 'richT_r', 'richT_pico', 'richT_ok'])
    rich.to_parquet(os.path.join(SAIDA, f'richards_{nome}.parquet'), index=False)
    log(f'richards_{nome}: {len(rich):,} ajustes · convergiram {rich.rich_ok.mean():.0%} (completa) / {rich.richT_ok.mean():.0%} (até SE25)')

log('pronto. Suba a pasta V4/dados/ (e V4/referencias_imdc3/) para MyDrive/dengue_V4/')
