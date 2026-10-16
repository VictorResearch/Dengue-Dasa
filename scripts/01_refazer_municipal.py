"""
refazer_season.py  (cópia generalizada de S1_corrigido/refazer_season1.py)
=================
Reconstrói, PARA QUALQUER SEASON (1 a 4), as tabelas usadas pelos modelos AGCRN:

    train{N}_dengue_eng_COMPLETO_spatial.parquet   test{N}_dengue_eng_COMPLETO_spatial.parquet
    train{N}_FINAL_NN.parquet                      test{N}_FINAL_NN.parquet
    train{N}_FINAL_NN_v2.parquet / test{N}_...     (só com --etapa nn_v2: FINAL_NN + colunas
                                                    extras escolhidas pelo SHAP corrigido)

Janelas (regras do 3rd IMDC, colunas train_k/target_k do dengue.csv):
    S1 treino 201001–202225 · teste 202241–202340      S2 treino 201001–202325 · teste 202341–202440
    S3 treino 201001–202425 · teste 202441–202540      S4 treino 201001–202525 · teste 202541–202610

partindo direto dos CSVs brutos em data_imdc_2026/variaveis/ e salvando tudo em
data_imdc_2026/S{N}_corrigido/. A lógica é idêntica à do refazer_season1.py (que continua
valendo para a S1); só mudam as janelas, os nomes dos arquivos e as colunas extras do v2. Nenhum arquivo antigo é lido para construir as tabelas;
os bancos antigos só são usados, se existirem, numa checagem de validação no final.

O que foi corrigido em relação ao pipeline antigo (R + notebook + TestesomenteAGCRN.py)
-------------------------------------------------------------------------------------
 1. DUPLICATAS  – o join antigo com o clima previsto (date == reference_month) repetia 6x
    toda semana que começava no dia 1 do mês (~15% das linhas). Aqui tudo é montado numa
    grade balanceada [município x semana] e cada fonte entra por posição, com checagem de
    unicidade: é impossível existir (geocode, date) repetido.
 2. CLIMA PREVISTO – `reference_month` é o mês de EMISSÃO e `forecast_months_ahead = 1` é a
    previsão para o próprio mês de emissão (verificado empiricamente: o erro contra o ERA5
    cresce com o horizonte só nessa convenção). Para cada semana usa-se a previsão EMITIDA NO
    MÊS ANTERIOR ao mês-alvo (horizonte 2) -> disponível em toda a semana, sem vazamento.
    precip_tot da previsão vem em mm/s (kg m-2 s-1) e é convertida para mm/dia (x 86400).
 3. OSCILAÇÕES OCEÂNICAS – as datas do arquivo caem em segunda/terça; eram 100% NA no join
    exato. Aqui cada data é levada ao domingo da própria semana e semanas sem observação
    recebem o último valor (LOCF), exatamente como descreve a documentação oficial.
 4. AFYA – a série é DIÁRIA. O join antigo pegava só o valor do domingo. Aqui: registros
    duplicados do mesmo dia são somados, a série é somada por semana epidemiológica
    (domingo–sábado, só semanas com os 7 dias cobertos), municípios ausentes do arquivo
    recebem 0 no período coberto e o período ANTES de 2021 fica NaN + flag `afya_disponivel`
    (antes era preenchido com bfill, criando uma constante falsa por município).
 5. ILHAS SEM CLIMA – Fernando de Noronha (2605459), Itaparica (2916104) e Madre de Deus
    (2919926) não existem no climate.csv (e Noronha não existe na previsão). Antes viravam 0.
    Aqui recebem a média ponderada pelo inverso da distância dos 3 municípios mais próximos
    (centroides da malha) + flags `clima_imputado` / `forecast_imputado`.
 6. POPULAÇÃO – join por (geocode, ano da SE). Se faltar algum ano (ex.: 2026 nas safras
    futuras), usa-se o ano disponível mais próximo + flag `pop_imputada` (na S1 não falta nenhum).
    Previsão climática com buracos (Noronha: 180 de 195 meses vazios) é imputada célula a célula.
 7. SEM ffill/bfill GENÉRICO – nada é preenchido "do futuro para o passado".
 8. CONTINUIDADE TREINO -> TESTE – todas as janelas, lags, EWM e a física (Savitzky-Golay)
    são calculados na série contínua 2009–2023 e só depois recortados. No pipeline antigo o
    teste era processado isolado (EWM de imunidade reiniciava do zero, lags do início do teste
    ficavam vazios, a física atravessava o buraco EW26–40 como se fosse contínuo).
    ATENÇÃO: isso mantém o esquema one-step-ahead dos modelos atuais (a linha da semana t do
    teste usa casos observados até t, inclusive das semanas EW26–40 como histórico). Não é o
    formato exigido pelo desafio (prever a safra inteira só com dados até EW25).
 9. hti_lag2 – no R era idêntico a hti_lag3 (usava lag 3); agora usa lag 2.
10. SAZONALIDADE – seno/cosseno eram calculados sobre a semana ISO do domingo (1 semana
    defasada da SE). Agora usam o dia do ano do meio da semana (quarta-feira).
11. ILHAS NA MALHA – municípios sem vizinho Queen (Noronha, Ilhabela) tinham lags espaciais = 0
    ("vizinhos com população zero"). Agora são ligados ao município mais próximo (simétrico)
    + flag `w1_vizinho_knn`.
12. Target encoding (koppen/bioma) e StandardScaler são ajustados SÓ nas linhas de treino.

Mantido de propósito (decisões de modelagem do projeto, não erros)
-------------------------------------------------------------------
 - Nomes de colunas do FINAL_NN original (o notebook roda sem mudar COLS_SECA / COLS_PICO).
 - Espírito Santo fora (regra do desafio). 5.492 municípios, node_id na ordem do geocode.
 - Física: Savitzky-Golay janela 7, ordem 3 em log1p(casos), deslocada 3 semanas (causal).
 - Convenção dos sufixos `_t1`: clima/Afya/população/chik usam shift(1); as físicas
   (casos_suavizados_t1, velocidade_t1, aceleracao_t1) usam shift(0), como no notebook.
 - Lags climáticos: agora TODOS os lags 2, 3, 4 e 5 (antes a S1 tinha só 3–5).

Dependências: numpy, pandas, scipy, pyarrow  (pip install numpy pandas scipy pyarrow)
Uso:          python scripts/01_refazer_municipal.py --season 2
              python scripts/01_refazer_municipal.py --season 2 --etapa nn_v2   (depois do SHAP corrigido)
Tempo:        ~5–10 min · RAM: ~6 GB de pico
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from scipy import sparse
from scipy.signal import savgol_filter

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================
SEASONS = {                      # regras 3rd IMDC (train_k / target_k do dengue.csv)
    1: ((201001, 202225), (202241, 202340)),
    2: ((201001, 202325), (202341, 202440)),
    3: ((201001, 202425), (202441, 202540)),
    4: ((201001, 202525), (202541, 202610)),   # dengue.csv termina em 202610
}
_ap = argparse.ArgumentParser(description='Reconstrói COMPLETO/FINAL_NN de uma season a partir dos CSVs brutos.')
_ap.add_argument('--season', type=int, choices=sorted(SEASONS), required=True)
_ap.add_argument('--etapa', choices=['tudo', 'nn_v2'], default='tudo',
                 help="'tudo' = COMPLETO + FINAL_NN (+ W, nós, validação); "
                      "'nn_v2' = só FINAL_NN_v2 (FINAL_NN + colunas de SHAP_corrigido/colunas_extras_nn.json)")
ARGS = _ap.parse_args()
SEASON = ARGS.season

RAIZ = Path(__file__).resolve().parent.parent       # raiz do repositório
DIR_BASE = RAIZ / 'dados' / 'municipal'             # bancos antigos (só para a checagem opcional do fim)
DIR_OUT = RAIZ / 'dados' / 'municipal' / f'S{SEASON}'
DIR_OUT.mkdir(parents=True, exist_ok=True)
DIR_VAR = RAIZ / 'dados' / 'brutos'
ARQ_EXTRAS_NN = RAIZ / 'notebooks' / '01_agcrn_municipal' / 'shap_corrigido' / 'colunas_extras_nn.json'

EW_TREINO, EW_TESTE = SEASONS[SEASON]
EW_HIST_INI = 200940             # 13 semanas antes de 201001: só histórico p/ lags/janelas
UF_EXCLUIDA = 32                 # Espírito Santo
BLOCO_SEMANAS = 26               # semanas por row group ao gravar

LAGS_CLIMA = (2, 3, 4, 5)
LAGS_OCEANO = (6, 7, 8, 9)
VARS_CLIMA = ['temp_min', 'temp_med', 'temp_max', 'precip_min', 'precip_med', 'precip_max',
              'pressure_min', 'pressure_med', 'pressure_max', 'rel_humid_min', 'rel_humid_med',
              'rel_humid_max', 'thermal_range', 'rainy_days']
VARS_LAG_CLIMA = VARS_CLIMA + ['vpd']
MESES_ANTECEDENCIA_FORECAST = 1  # previsão emitida 1 mês antes do mês-alvo (horizonte 2)
K_VIZINHOS_IMPUTACAO = 3
SO_DESDE_2021 = {'access_count', 'access_count_t1', 'w1_acessos_afya_t1', 'afya_log_diff'}

T0 = time.time()
RELATORIO: list[tuple[str, bool, str]] = []


def log(msg: str) -> None:
    print(f'[{time.time() - T0:6.0f}s] {msg}', flush=True)


def checar(nome: str, ok: bool, detalhe: str = '') -> None:
    RELATORIO.append((nome, bool(ok), detalhe))
    print(f'    {"OK  " if ok else "FALHA"} | {nome}' + (f' — {detalhe}' if detalhe else ''), flush=True)


def ler_csv(nome: str, colunas: list[str] | None = None, tipos: dict | None = None) -> pa.Table:
    return pacsv.read_csv(
        DIR_VAR / nome,
        read_options=pacsv.ReadOptions(block_size=1 << 26),
        convert_options=pacsv.ConvertOptions(include_columns=colunas, column_types=tipos or {}),
    )


# =============================================================================
# 1. MALHA MUNICIPAL (GeoPackage lido com sqlite3, sem geopandas)
# =============================================================================
def _gpkg_para_wkb(blob: bytes) -> memoryview:
    """Remove o cabeçalho GeoPackage (GP + versão + flags + srs + envelope) e devolve o WKB."""
    if blob[:2] != b'GP':
        raise ValueError('geometria não está no formato GeoPackage')
    envelope = (blob[3] >> 1) & 0b111
    return memoryview(blob)[8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope]:]


def _wkb_para_aneis(wkb: memoryview) -> list[np.ndarray]:
    """Lê Polygon/MultiPolygon WKB 2D e devolve a lista de anéis (arrays n x 2 de lon/lat)."""
    aneis: list[np.ndarray] = []

    def ler(off: int) -> int:
        ordem = '<' if wkb[off] == 1 else '>'
        tipo = struct.unpack_from(ordem + 'I', wkb, off + 1)[0] % 1000
        off += 5
        if tipo == 3:
            n_aneis = struct.unpack_from(ordem + 'I', wkb, off)[0]
            off += 4
            for _ in range(n_aneis):
                n = struct.unpack_from(ordem + 'I', wkb, off)[0]
                off += 4
                aneis.append(np.frombuffer(wkb, dtype=np.dtype(ordem + 'f8'), count=2 * n, offset=off).reshape(n, 2))
                off += 16 * n
            return off
        if tipo == 6:
            n_pol = struct.unpack_from(ordem + 'I', wkb, off)[0]
            off += 4
            for _ in range(n_pol):
                off = ler(off)
            return off
        raise ValueError(f'tipo WKB não suportado: {tipo}')

    ler(0)
    return aneis


def ler_malha(caminho: Path) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    con = sqlite3.connect(caminho)
    tabela, col_geom = con.execute('SELECT table_name, column_name FROM gpkg_geometry_columns').fetchone()
    linhas = con.execute(f'SELECT geocode, "{col_geom}" FROM "{tabela}"').fetchall()
    con.close()
    geocodes = np.array([int(g) for g, _ in linhas], dtype=np.int64)
    aneis = [_wkb_para_aneis(_gpkg_para_wkb(b)) for _, b in linhas]
    return geocodes, aneis


def centroides(aneis_por_mun: list[list[np.ndarray]]) -> np.ndarray:
    """Centroide por área (fórmula do polígono) somando anéis com área com sinal."""
    out = np.zeros((len(aneis_por_mun), 2))
    for i, aneis in enumerate(aneis_por_mun):
        a_tot = cx = cy = 0.0
        for r in aneis:
            x, y = r[:, 0], r[:, 1]
            x1, y1 = np.roll(x, -1), np.roll(y, -1)
            cr = x * y1 - x1 * y
            a = cr.sum() / 2
            if a == 0:
                continue
            a_tot += a
            cx += ((x + x1) * cr).sum() / 6
            cy += ((y + y1) * cr).sum() / 6
        out[i] = (cx / a_tot, cy / a_tot) if a_tot != 0 else np.concatenate(aneis).mean(axis=0)
    return out


def contiguidade_queen(aneis_por_mun: list[list[np.ndarray]]) -> sparse.csr_matrix:
    """Queen = compartilham ao menos 1 vértice (mesmo critério do libpysal.Queen)."""
    ids, xs, ys = [], [], []
    for i, aneis in enumerate(aneis_por_mun):
        p = np.concatenate(aneis)
        xs.append(p[:, 0]); ys.append(p[:, 1]); ids.append(np.full(len(p), i, np.int32))
    x, y, idv = np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)
    o = np.lexsort((idv, y, x))
    x, y, idv = x[o], y[o], idv[o]
    rep = np.r_[False, (x[1:] == x[:-1]) & (y[1:] == y[:-1]) & (idv[1:] == idv[:-1])]
    x, y, idv = x[~rep], y[~rep], idv[~rep]
    mesmo = (x[1:] == x[:-1]) & (y[1:] == y[:-1])
    inicio = np.flatnonzero(np.r_[True, ~mesmo])
    tam = np.diff(np.r_[inicio, len(x)])
    pares_a, pares_b = [], []
    for s in inicio[tam == 2]:
        pares_a.append(idv[s]); pares_b.append(idv[s + 1])
    for s, k in zip(inicio[tam > 2], tam[tam > 2]):
        g = idv[s:s + k]
        for i in range(k):
            for j in range(i + 1, k):
                pares_a.append(g[i]); pares_b.append(g[j])
    a, b = np.array(pares_a), np.array(pares_b)
    n = len(aneis_por_mun)
    m = sparse.coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    m.data[:] = 1.0
    m.setdiag(0)
    m.eliminate_zeros()
    return m


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def normalizar_linhas(b: sparse.csr_matrix) -> sparse.csr_matrix:
    s = np.asarray(b.sum(axis=1)).ravel()
    inv = np.divide(1.0, s, out=np.zeros_like(s), where=s > 0)
    return sparse.diags(inv) @ b


# =============================================================================
# 2. UTILITÁRIOS DA GRADE [nó x semana]
# =============================================================================
class Grade:
    def __init__(self, geocodes: np.ndarray, epiweeks: np.ndarray, datas: pd.DatetimeIndex):
        self.geo = geocodes
        self.ew = epiweeks
        self.datas = datas
        self.N, self.T = len(geocodes), len(epiweeks)

    def idx_no(self, g: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.geo, g)
        i = np.clip(i, 0, self.N - 1)
        return np.where(self.geo[i] == g, i, -1)

    def idx_sem(self, e: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.ew, e)
        i = np.clip(i, 0, self.T - 1)
        return np.where(self.ew[i] == e, i, -1)

    def espalhar(self, nome: str, geo, ew, valores, dtype=np.float32, checar_dup: bool = True) -> np.ndarray:
        """Coloca valores na matriz [N, T] por (geocode, epiweek), exigindo unicidade."""
        ni, ti = self.idx_no(np.asarray(geo)), self.idx_sem(np.asarray(ew))
        ok = (ni >= 0) & (ti >= 0)
        if checar_dup:
            chave = ni[ok].astype(np.int64) * self.T + ti[ok]
            n_dup = len(chave) - len(np.unique(chave))
            checar(f'{nome}: sem (geocode, semana) repetido na fonte', n_dup == 0, f'{n_dup} repetidos')
        m = np.full((self.N, self.T), np.nan, dtype=dtype)
        m[ni[ok], ti[ok]] = np.asarray(valores)[ok]
        return m


def imputar_idw(m: np.ndarray, faltantes: np.ndarray, doadores: np.ndarray, cent: np.ndarray, k: int) -> dict:
    """Preenche as células NaN dos nós `faltantes` com a média ponderada pelo inverso da
    distância dos k nós doadores (sem nenhum NaN) mais próximos. Células com dado ficam intactas."""
    info = {}
    for i in faltantes:
        d = haversine_km(cent[i, 0], cent[i, 1], cent[doadores, 0], cent[doadores, 1])
        viz = doadores[np.argsort(d)[:k]]
        w = 1.0 / np.sort(d)[:k]
        est = (m[viz] * w[:, None]).sum(axis=0) / w.sum()
        m[i] = np.where(np.isnan(m[i]), est, m[i])
        info[int(i)] = [int(v) for v in viz]
    return info


# =============================================================================
# 3. MONTAGEM DAS FONTES
# =============================================================================
def montar() -> dict:
    P: dict = {}

    # ---------------------------------------------------------------- dengue
    log('Lendo dengue.csv ...')
    tb = ler_csv('dengue.csv', ['geocode', 'date', 'epiweek', 'casos', 'uf_code', 'target_city'])
    tb = tb.filter(pc.less_equal(tb['epiweek'], EW_TESTE[1]))
    den = tb.to_pandas()
    den['date'] = pd.to_datetime(den['date'])
    checar('dengue: toda data é domingo', bool((den['date'].dt.weekday == 6).all()))
    checar('dengue: 1 data por epiweek', den.groupby('epiweek')['date'].nunique().max() == 1)

    # calendário: dengue (>=201001) + clima (histórico pré-2010)
    log('Lendo climate.csv (histórico + clima observado) ...')
    tc = ler_csv('climate.csv', ['date', 'epiweek', 'geocode'] + VARS_CLIMA)
    tc = tc.filter(pc.and_(pc.greater_equal(tc['epiweek'], EW_HIST_INI), pc.less_equal(tc['epiweek'], EW_TESTE[1])))
    cli = tc.to_pandas()
    del tc
    cli['date'] = pd.to_datetime(cli['date'])
    cal = pd.concat([den[['epiweek', 'date']], cli[['epiweek', 'date']]]).drop_duplicates()
    checar('calendário: dengue e clima concordam (1 data por epiweek)', cal.groupby('epiweek')['date'].nunique().max() == 1)
    cal = cal.drop_duplicates('epiweek').sort_values('epiweek').reset_index(drop=True)
    passos = cal['date'].diff().dt.days.dropna().unique()
    checar('calendário: semanas contínuas de 7 dias', list(passos) == [7.0], f'passos={passos}')
    checar('calendário: começa em EW_HIST_INI e termina no fim do teste',
           cal.epiweek.iloc[0] == EW_HIST_INI and cal.epiweek.iloc[-1] == EW_TESTE[1])

    geos = np.sort(den.loc[den['uf_code'] != UF_EXCLUIDA, 'geocode'].unique()).astype(np.int64)
    G = Grade(geos, cal['epiweek'].to_numpy(np.int64), pd.DatetimeIndex(cal['date']))
    P['G'] = G
    i0 = int(G.idx_sem(np.array([EW_TREINO[0]]))[0])
    P['i0'] = i0
    log(f'Grade: {G.N} municípios x {G.T} semanas (histórico desde {EW_HIST_INI}; dengue desde índice {i0})')

    P['casos'] = G.espalhar('dengue', den['geocode'], den['epiweek'], den['casos'], np.float64)
    bloco = P['casos'][:, i0:]
    checar('dengue: grade completa 201001–fim do teste (sem buracos)', not np.isnan(bloco).any(),
           f'{int(np.isnan(bloco).sum())} células vazias')
    soma_raw = den.loc[(den['uf_code'] != UF_EXCLUIDA) & (den['epiweek'] >= EW_TREINO[0]), 'casos'].sum()
    checar('dengue: soma de casos da grade = soma do CSV', np.nansum(bloco) == soma_raw, f'{soma_raw:,.0f}')
    tc_map = den.drop_duplicates('geocode').set_index('geocode')['target_city']
    P['target_city'] = tc_map.reindex(geos).fillna(False).astype(bool).to_numpy()
    del den

    # ---------------------------------------------------------------- malha
    log('Lendo malha municipal (shape_muni.gpkg) ...')
    geo_malha, aneis = ler_malha(DIR_VAR / 'shape_muni.gpkg')
    checar('malha: 1 geometria por município', len(np.unique(geo_malha)) == len(geo_malha))
    faltam = np.setdiff1d(geos, geo_malha)
    checar('malha: todos os municípios da grade têm geometria', len(faltam) == 0, str(faltam[:5]))
    cent_all = centroides(aneis)
    B_all = contiguidade_queen(aneis)
    ilhas_all = geo_malha[np.asarray(B_all.sum(axis=1)).ravel() == 0]
    log(f'  Queen (malha completa): {B_all.nnz // 2} pares de vizinhos; ilhas: {list(ilhas_all)}')
    pos = np.searchsorted(geo_malha[np.argsort(geo_malha)], geos)
    ordem_malha = np.argsort(geo_malha)[pos]                  # posição de cada nó da grade na malha
    B1 = B_all[ordem_malha][:, ordem_malha].tocsr()           # remove ES e alinha à grade
    cent = cent_all[ordem_malha]
    P['cent'] = cent
    sem_viz = np.flatnonzero(np.asarray(B1.sum(axis=1)).ravel() == 0)
    knn = {}
    B1 = B1.tolil()
    for i in sem_viz:
        d = haversine_km(cent[i, 0], cent[i, 1], cent[:, 0], cent[:, 1])
        d[i] = np.inf
        j = int(np.argmin(d))
        B1[i, j] = 1.0
        B1[j, i] = 1.0
        knn[int(geos[i])] = (int(geos[j]), round(float(d[j]), 1))
    B1 = B1.tocsr()
    log(f'  Nós sem vizinho Queen após remover ES -> ligados ao mais próximo: {knn}')
    B2 = (B1 @ B1)
    B2.data[:] = 1.0
    B2 = B2.tolil()
    B2.setdiag(0)
    B2 = B2.tocsr()
    B2 = B2 - B2.multiply(B1)
    B2.eliminate_zeros()
    P['W1'] = normalizar_linhas(B1).tocsr()
    P['W2'] = normalizar_linhas(B2).tocsr()
    P['B1'], P['knn'] = B1, knn
    P['w1_vizinho_knn'] = np.isin(geos, list(knn.keys()) + [v[0] for v in knn.values()]).astype(np.int8)
    checar('malha: nenhum nó sem vizinho em W1', (np.asarray(B1.sum(axis=1)).ravel() > 0).all())
    checar('malha: W1 simétrica', (B1 != B1.T).nnz == 0)
    checar('malha: nenhum nó sem vizinho em W2', (np.asarray(B2.sum(axis=1)).ravel() > 0).all(),
           f'{int((np.asarray(B2.sum(axis=1)).ravel() == 0).sum())} nós sem W2')

    # ---------------------------------------------------------------- estáticos
    log('Lendo mapa regional e environ_vars ...')
    mp = ler_csv('map_regional_health.csv').to_pandas().set_index('geocode').reindex(geos)
    env = ler_csv('environ_vars.csv').to_pandas().set_index('geocode').reindex(geos)
    checar('mapa regional: todos os nós presentes', not mp['uf_code'].isna().any())
    checar('environ: koppen e bioma para todos os nós', not env[['koppen', 'biome']].isna().any().any())
    P['estatico'] = pd.DataFrame({
        'geocode': geos, 'node_id': np.arange(G.N),
        'uf': mp['uf'].to_numpy(), 'uf_code': mp['uf_code'].to_numpy(),
        'macroregion_code': mp['macroregion_code'].to_numpy(), 'macroregion_name': mp['macroregion_name'].to_numpy(),
        'uf_name': mp['uf_name'].to_numpy(),
        'macroregional_geocode': mp['macroregional_geocode'].to_numpy(), 'macroregional_name': mp['macroregional_name'].to_numpy(),
        'regional_geocode': mp['regional_geocode'].to_numpy(), 'regional_name': mp['regional_name'].to_numpy(),
        'geocode_name': mp['geocode_name'].to_numpy(),
        'koppen': env['koppen'].to_numpy(), 'biome': env['biome'].to_numpy(),
        'target_city': P['target_city'],
        'lon_centroide': cent[:, 0], 'lat_centroide': cent[:, 1],
    })

    # ---------------------------------------------------------------- chikungunya
    log('Lendo chikungunya.csv ...')
    tk = ler_csv('chikungunya.csv', ['geocode', 'epiweek', 'casos'])
    chk = tk.filter(pc.less_equal(tk['epiweek'], EW_TESTE[1])).to_pandas()
    P['chik'] = G.espalhar('chikungunya', chk['geocode'], chk['epiweek'], chk['casos'], np.float32)
    ini_chik = int(chk['epiweek'].min())
    t_chik = int(G.idx_sem(np.array([ini_chik]))[0])
    checar('chikungunya: grade completa a partir de 201401', not np.isnan(P['chik'][:, t_chik:]).any())
    P['chik'][:, :t_chik] = 0.0            # série oficial começa em 201401 (sem transmissão notificada antes)
    P['chik_disponivel'] = (G.ew >= ini_chik).astype(np.int8)
    del chk, tk

    # ---------------------------------------------------------------- população
    log('Lendo população ...')
    pop = ler_csv('datasus_population_2001_2025.csv').to_pandas()
    checar('população: sem (geocode, ano) repetido', not pop.duplicated(['geocode', 'year']).any())
    anos = np.arange(G.ew.min() // 100, G.ew.max() // 100 + 1)
    pv_pop = pop.pivot(index='geocode', columns='year', values='population').reindex(index=geos)
    pv_pop = pv_pop.reindex(columns=range(min(pv_pop.columns.min(), anos[0]), max(pv_pop.columns.max(), anos[-1]) + 1))
    faltou = pv_pop.reindex(columns=anos).isna().to_numpy()
    # ano faltante recebe o ano disponível anterior; antes do 1º ano disponível, o 1º ano
    # (população é estrutural: só afeta município criado depois de 2010)
    pv_full = pv_pop.ffill(axis=1).bfill(axis=1).reindex(columns=anos)
    checar('população: nenhum nó/ano sem valor após imputação', not pv_full.isna().any().any())
    ano_sem = (G.ew // 100) - anos[0]
    P['pop'] = pv_full.to_numpy(np.float64)[:, ano_sem]
    P['pop_imputada'] = faltou[:, ano_sem].astype(np.int8)
    imput = geos[faltou.any(axis=1)]
    log(f'  população imputada (ano mais próximo) para: {list(imput)}')
    del pop, pv_pop, pv_full

    # ---------------------------------------------------------------- clima observado
    log('Espalhando clima observado ...')
    for i, v in enumerate(VARS_CLIMA):
        P[v] = G.espalhar('clima', cli['geocode'], cli['epiweek'], cli[v], checar_dup=(i == 0))
    # checa calendário do clima x grade
    amostra = cli.sample(5000, random_state=1)
    ni, ti = G.idx_no(amostra['geocode'].to_numpy()), G.idx_sem(amostra['epiweek'].to_numpy())
    ok = ni >= 0
    checar('clima: amostra da grade = CSV (temp_med)', np.allclose(P['temp_med'][ni[ok], ti[ok]], amostra['temp_med'].to_numpy()[ok]))
    del cli
    sem_clima = np.flatnonzero(np.isnan(P['temp_med']).all(axis=1))
    doadores = np.setdiff1d(np.arange(G.N), sem_clima)
    info = {}
    for v in VARS_CLIMA:
        info = imputar_idw(P[v], sem_clima, doadores, cent, K_VIZINHOS_IMPUTACAO)
    log(f'  clima imputado (IDW {K_VIZINHOS_IMPUTACAO} vizinhos) -> {{nó: doadores}}: '
        f'{ {int(geos[k]): [int(geos[x]) for x in v] for k, v in info.items()} }')
    P['clima_imputado'] = np.isin(np.arange(G.N), sem_clima).astype(np.int8)
    checar('clima: nenhuma célula vazia após imputação das ilhas',
           not any(np.isnan(P[v]).any() for v in VARS_CLIMA))

    # ---------------------------------------------------------------- clima previsto
    log('Lendo forecasting_climate.csv ...')
    fc = ler_csv('forecasting_climate.csv').to_pandas()
    fc['reference_month'] = pd.to_datetime(fc['reference_month'])
    checar('previsão: sem (geocode, mês, horizonte) repetido',
           not fc.duplicated(['geocode', 'reference_month', 'forecast_months_ahead']).any())
    meses = pd.period_range(fc['reference_month'].min(), fc['reference_month'].max(), freq='M')
    checar('previsão: todos os meses de emissão presentes (sem buraco)', fc['reference_month'].nunique() == len(meses))
    rm = fc['reference_month']
    mi = ((rm.dt.year - meses[0].year) * 12 + (rm.dt.month - meses[0].month)).to_numpy()
    ni = G.idx_no(fc['geocode'].to_numpy())
    hi = fc['forecast_months_ahead'].to_numpy() - 1
    ok = ni >= 0
    cubo = {}
    for v_src, v_dst, fator in [('temp_med', 'temp_med_forecast', 1.0), ('umid_med', 'umid_med_forecast', 1.0),
                                ('precip_tot', 'precip_tot_forecast', 86400.0)]:
        c = np.full((G.N, len(meses), 6), np.nan, np.float32)
        c[ni[ok], mi[ok], hi[ok]] = fc[v_src].to_numpy()[ok] * fator
        cubo[v_dst] = c
    faltou_fc = np.isnan(cubo['temp_med_forecast'])                 # (N, meses, 6) antes da imputação
    com_falta = np.flatnonzero(faltou_fc.any(axis=(1, 2)))
    doad = np.flatnonzero(~faltou_fc.any(axis=(1, 2)))
    for v in cubo:
        flat = cubo[v].reshape(G.N, -1)
        imputar_idw(flat, com_falta, doad, cent, K_VIZINHOS_IMPUTACAO)
        cubo[v] = flat.reshape(G.N, len(meses), 6)
    log(f'  previsão imputada (IDW, só células vazias) para: '
        f'{ {int(geos[i]): f"{faltou_fc[i].any(axis=1).sum()} de {len(meses)} meses" for i in com_falta} }')
    checar('previsão: cubo completo após imputação', not any(np.isnan(c).any() for c in cubo.values()))
    # mês-alvo de cada semana = mês da quarta-feira; emissão = mês-alvo - 1 (fallback: próprio mês)
    alvo = (G.datas + pd.Timedelta(days=3)).to_period('M')
    emis = alvo - MESES_ANTECEDENCIA_FORECAST
    e_idx = np.array([(e - meses[0]).n for e in emis])
    a_idx = np.array([(a - meses[0]).n for a in alvo])
    valido = a_idx >= 0                          # semanas de 2009 (só histórico) não têm previsão -> NaN
    fallback = valido & (e_idx < 0)              # jan/2010: não há emissão anterior -> usa a do próprio mês
    e_idx = np.where(fallback, a_idx, e_idx)
    h = a_idx - e_idx + 1                        # horizonte na convenção do arquivo (1 = mês de emissão)
    checar('previsão: mês de emissão dentro do arquivo para toda semana >= 2010',
           (e_idx[valido] >= 0).all() and (e_idx[valido] < len(meses)).all())
    e_c, h_c = np.where(valido, e_idx, 0), np.where(valido, h, 1)
    for v in cubo:
        x = cubo[v][:, e_c, h_c - 1]
        x[:, ~valido] = np.nan
        P[v] = x
    imp = faltou_fc[:, e_c, h_c - 1].astype(np.int8)
    imp[:, ~valido] = 0
    P['forecast_imputado'] = imp
    P['forecast_months_ahead'] = np.where(valido, h, 0).astype(np.int8)
    i0 = P['i0']
    checar('previsão: nenhuma célula vazia de 201001 em diante', not any(np.isnan(P[v][:, i0:]).any() for v in cubo))
    hh = h[i0:]
    log(f'  horizonte usado (semanas >= 201001): {dict(zip(*[x.tolist() for x in np.unique(hh, return_counts=True)]))} '
        f'(h=1 só em jan/2010, sem emissão anterior disponível)')
    # checagem contra o CSV
    t_s = np.random.default_rng(2).integers(P['i0'], G.T, 300)
    n_s = np.random.default_rng(3).integers(0, G.N, 300)
    n_s = n_s[~np.isin(n_s, com_falta)]
    t_s = t_s[:len(n_s)]
    ref = fc.set_index(['geocode', 'reference_month', 'forecast_months_ahead'])['temp_med'].sort_index()
    esp = [ref.loc[(int(geos[n]), emis[t].to_timestamp() if not fallback[t] else alvo[t].to_timestamp(), int(h[t]))] for n, t in zip(n_s, t_s)]
    checar('previsão: amostra da grade = CSV (emissão = mês-alvo − 1)', np.allclose(P['temp_med_forecast'][n_s, t_s], esp))
    del fc, cubo

    # ---------------------------------------------------------------- oceano
    log('Lendo ocean_climate_oscillations.csv ...')
    oc = ler_csv('ocean_climate_oscillations.csv').to_pandas()
    oc['date'] = pd.to_datetime(oc['date'])
    oc['domingo'] = oc['date'] - pd.to_timedelta((oc['date'].dt.weekday + 1) % 7, unit='D')
    checar('oceano: no máximo 1 observação por semana', not oc['domingo'].duplicated().any())
    sem = pd.DataFrame({'domingo': G.datas})
    j = pd.merge_asof(sem, oc.sort_values('domingo')[['domingo', 'enso', 'iod', 'pdo']], on='domingo',
                      direction='backward', tolerance=pd.Timedelta(days=28))
    n_locf = int((~sem['domingo'].isin(oc['domingo'])).sum())
    for v in ('enso', 'iod', 'pdo'):
        P[v] = j[v].to_numpy(np.float32)        # série global (1 valor por semana)
    checar('oceano: toda semana da grade tem valor', not j[['enso', 'iod', 'pdo']].isna().any().any(),
           f'{n_locf} semanas preenchidas por LOCF')
    del oc

    # ---------------------------------------------------------------- Afya
    log('Lendo access_afya (diário -> semanal) ...')
    af = ler_csv('access_afya_dengue_2021_2026.csv', ['access_date', 'geocode', 'access_count']).to_pandas()
    af['access_date'] = pd.to_datetime(af['access_date'])
    n_dup = int(af.duplicated(['geocode', 'access_date']).sum())
    af = af.groupby(['geocode', 'access_date'], as_index=False)['access_count'].sum()     # soma duplicatas do dia
    d_ini, d_fim = af['access_date'].min(), af['access_date'].max()
    af['domingo'] = af['access_date'] - pd.to_timedelta((af['access_date'].dt.weekday + 1) % 7, unit='D')
    sem_af = af.groupby(['geocode', 'domingo']).agg(acc=('access_count', 'sum'), dias=('access_count', 'size')).reset_index()
    cobertas = G.datas[(G.datas >= d_ini) & (G.datas + pd.Timedelta(days=6) <= d_fim)]
    sem_af = sem_af[sem_af['domingo'].isin(cobertas)]
    checar('Afya: toda semana coberta tem os 7 dias', (sem_af['dias'] == 7).all())
    t_cob = np.flatnonzero(np.isin(G.datas, cobertas))
    m = np.full((G.N, G.T), np.nan, np.float32)
    m[:, t_cob] = 0.0                                              # município ausente do arquivo = 0 acessos
    ni = G.idx_no(sem_af['geocode'].to_numpy())
    ti = np.searchsorted(G.datas.values, sem_af['domingo'].values)
    ok = ni >= 0
    m[ni[ok], ti[ok]] = sem_af['acc'].to_numpy()[ok]
    P['access_count'] = m
    P['afya_disponivel'] = np.isin(np.arange(G.T), t_cob).astype(np.int8)
    ausentes = int((~np.isin(geos, af['geocode'].unique())).sum())
    log(f'  {n_dup} registros diários duplicados somados; 1ª semana completa: {cobertas.min().date()}; '
        f'{ausentes} municípios da grade nunca aparecem no arquivo (=0)')
    g_s = af['geocode'].drop_duplicates().sample(50, random_state=4).to_numpy()
    g_s = g_s[G.idx_no(g_s) >= 0]
    t_s = t_cob[np.random.default_rng(5).integers(0, len(t_cob), len(g_s))]
    esp = [af[(af.geocode == g) & (af.access_date >= G.datas[t]) & (af.access_date <= G.datas[t] + pd.Timedelta(days=6))].access_count.sum()
           for g, t in zip(g_s, t_s)]
    checar('Afya: amostra da grade = soma diária do CSV', np.allclose(m[G.idx_no(g_s), t_s], esp))
    del af, sem_af
    return P


# =============================================================================
# 4. ENGENHARIA DE FEATURES (tudo na série contínua)
# =============================================================================
def engenharia(P: dict) -> None:
    G, i0 = P['G'], P['i0']
    log('Engenharia de features ...')
    # --- derivadas do R
    t, rh = P['temp_med'].astype(np.float64), P['rel_humid_med'].astype(np.float64)
    P['vpd'] = (0.611 * np.exp(17.27 * t / (t + 237.3)) * (1 - rh / 100)).astype(np.float32)

    def rolling(m, k, fn):
        df = pd.DataFrame(m.T.astype(np.float64))
        r = df.rolling(k, min_periods=k)
        return (r.mean() if fn == 'mean' else r.std()).to_numpy(np.float32).T

    P['amplitude_termica_dinamica3'] = rolling(P['thermal_range'], 3, 'mean')
    P['amplitude_termica_dinamica2'] = rolling(P['thermal_range'], 2, 'mean')
    P['volatilidade_temp_4w'] = rolling(P['temp_med'], 4, 'std')
    P['volatilidade_precip_4w'] = rolling(P['precip_med'], 4, 'std')
    hti = (P['precip_med'] * P['temp_med']).astype(np.float32)
    P['_hti'] = hti                                                   # hti_lag2/3 = shift desta

    # --- população, incidência, imunidade
    P['log_pop'] = np.log(np.clip(P['pop'], 1, None)).astype(np.float32)
    inc = np.full((G.N, G.T), np.nan)
    inc[:, i0:] = P['casos'][:, i0:] / np.clip(P['pop'][:, i0:], 1, None) * 1e5
    P['incidencia'] = inc.astype(np.float32)
    alpha = 1 - np.exp(np.log(0.5) / 104)
    ewm = np.full((G.N, G.T), np.nan)
    ewm[:, i0] = inc[:, i0]
    for k in range(i0 + 1, G.T):
        ewm[:, k] = (1 - alpha) * ewm[:, k - 1] + alpha * inc[:, k]
    P['imunidade_ewm'] = ewm.astype(np.float32)

    # --- sazonalidade pelo dia do ano da quarta-feira da semana
    doy = (G.datas + pd.Timedelta(days=3)).dayofyear.to_numpy()
    P['seno_semana'] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    P['cosseno_semana'] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    # --- Afya em log-diferença
    la = np.log1p(P['access_count'].astype(np.float64))
    d = np.full_like(la, np.nan)
    d[:, 1:] = la[:, 1:] - la[:, :-1]
    P['afya_log_diff'] = d.astype(np.float32)

    # --- física (Savitzky-Golay janela 7, ordem 3, deslocada 3 semanas — igual ao notebook)
    y = np.log1p(P['casos'][:, i0:])
    desl = 3
    out = []
    for der in (0, 1, 2):
        c = savgol_filter(y, 7, 3, deriv=der, axis=1)
        causal = np.empty_like(c)
        causal[:, :desl] = c[:, [0]]
        causal[:, desl:] = c[:, :-desl]
        full = np.full((G.N, G.T), np.nan)
        full[:, i0:] = causal
        out.append(full)
    P['casos_suavizados'] = np.expm1(np.maximum(out[0], 0)).astype(np.float32)
    P['velocidade'] = out[1].astype(np.float32)
    P['aceleracao'] = out[2].astype(np.float32)

    # --- target encoding (só linhas de treino)
    it1 = int(G.idx_sem(np.array([EW_TREINO[1]]))[0]) + 1
    media_no = P['casos'][:, i0:it1].mean(axis=1)
    est = P['estatico']
    for col in ('koppen', 'biome'):
        enc = pd.Series(media_no).groupby(est[col].to_numpy()).mean()
        P[f'{col}_encoded'] = est[col].map(enc).to_numpy(np.float32)
        P[f'_enc_{col}'] = enc.to_dict()
    P['_it1'] = it1


# =============================================================================
# 5. ESPECIFICAÇÃO DAS COLUNAS (calculadas por bloco de semanas na hora de gravar)
# =============================================================================
def especificar(P: dict) -> tuple[dict, dict]:
    G, W1, W2 = P['G'], P['W1'], P['W2']
    est = P['estatico']

    def mat(nome, k=0):
        a = P[nome]
        if a.ndim == 1:                                   # série só temporal (oceano, seno...)
            return lambda s: np.broadcast_to(a[s.start - k:s.stop - k], (G.N, s.stop - s.start))
        return lambda s: a[:, s.start - k:s.stop - k]

    def espacial(W, f):
        def g(s):
            x = np.asarray(f(s), dtype=np.float64)
            falta = np.isnan(x)
            out = W @ np.where(falta, 0.0, x)
            if falta.any():
                out[(W @ falta.astype(np.float64)) > 0] = np.nan
            return out.astype(np.float32)
        return g

    C: dict = {}      # nome -> (tipo, fonte, dtype)   tipo: 'no' | 'tempo' | 'mat'
    ew_str = pd.Series(G.datas.strftime('%Y-%m-%d')).to_numpy()
    C['geocode'] = ('no', est['geocode'].to_numpy(), 'int32')
    C['date'] = ('tempo', ew_str, 'string')
    C['epiweek'] = ('tempo', G.ew, 'int32')
    C['year'] = ('tempo', G.ew // 100, 'int32')
    C['node_id'] = ('no', est['node_id'].to_numpy(), 'int32')
    C['time_index'] = ('tempo', np.arange(G.T) - P['i0'] + 1, 'int32')
    for c in ['uf', 'uf_code', 'macroregion_code', 'macroregion_name', 'uf_name', 'macroregional_geocode',
              'macroregional_name', 'regional_geocode', 'regional_name', 'geocode_name']:
        C[c] = ('no', est[c].to_numpy(), 'string' if est[c].dtype == object else 'int32')
    C['target_city'] = ('no', est['target_city'].to_numpy(), 'bool')
    C['casos.x'] = ('mat', mat('casos'), 'int32')
    C['incidencia'] = ('mat', mat('incidencia'), 'float32')
    C['casos_chikungunya'] = ('mat', mat('chik'), 'int32')
    C['population'] = ('mat', mat('pop'), 'int32')
    C['log_pop'] = ('mat', mat('log_pop'), 'float32')
    C['koppen'] = ('no', est['koppen'].to_numpy(), 'string')
    C['biome'] = ('no', est['biome'].to_numpy(), 'string')
    C['koppen_encoded'] = ('no', P['koppen_encoded'], 'float32')
    C['biome_encoded'] = ('no', P['biome_encoded'], 'float32')
    for v in VARS_CLIMA:
        C[v] = ('mat', mat(v), 'float32')
    for v in ['vpd', 'amplitude_termica_dinamica3', 'amplitude_termica_dinamica2',
              'volatilidade_temp_4w', 'volatilidade_precip_4w']:
        C[v] = ('mat', mat(v), 'float32')
    C['hti_lag3'] = ('mat', mat('_hti', 3), 'float32')
    C['hti_lag2'] = ('mat', mat('_hti', 2), 'float32')
    for v in ['temp_med_forecast', 'umid_med_forecast', 'precip_tot_forecast']:
        C[v] = ('mat', mat(v), 'float32')
    C['forecast_months_ahead'] = ('tempo', P['forecast_months_ahead'], 'int32')
    for v in ['enso', 'iod', 'pdo']:
        C[v] = ('mat', mat(v), 'float32')
    for v in ['enso', 'iod', 'pdo']:
        for k in LAGS_OCEANO:
            C[f'{v}_lag{k}'] = ('mat', mat(v, k), 'float32')
    C['access_count'] = ('mat', mat('access_count'), 'float32')
    C['afya_log_diff'] = ('mat', mat('afya_log_diff'), 'float32')
    C['imunidade_ewm'] = ('mat', mat('imunidade_ewm'), 'float32')
    C['seno_semana'] = ('mat', mat('seno_semana'), 'float32')
    C['cosseno_semana'] = ('mat', mat('cosseno_semana'), 'float32')
    for v in VARS_LAG_CLIMA:
        for k in LAGS_CLIMA:
            C[f'{v}_lag{k}'] = ('mat', mat(v, k), 'float32')
    for v in ['casos_suavizados', 'velocidade', 'aceleracao']:
        C[v] = ('mat', mat(v), 'float32')
    # ---- colunas _t1 (convenção do notebook)
    t1 = {
        'umid_med_forecast_t1': mat('umid_med_forecast', 1), 'imunidade_ewm_t1': mat('imunidade_ewm', 1),
        'precip_tot_forecast_t1': mat('precip_tot_forecast', 1), 'access_count_t1': mat('access_count', 1),
        'temp_med_forecast_t1': mat('temp_med_forecast', 1), 'pressure_min_t1': mat('pressure_min', 1),
        'casos_chikungunya_t1': mat('chik', 1), 'population_t1': mat('pop', 1),
        'casos_suavizados_t1': mat('casos_suavizados', 0), 'aceleracao_t1': mat('aceleracao', 0),
        'velocidade_t1': mat('velocidade', 0),
    }
    for k, f in t1.items():
        C[k] = ('mat', f, 'float32')
    # ---- lags espaciais (mesma lista do notebook, célula 6)
    esp_map = [
        ('casos_suavizados_t1', W1, 'w1_dengue_t1'), ('casos_chikungunya_t1', W1, 'w1_chikungunya_t1'),
        ('imunidade_ewm_t1', W1, 'w1_imunidade_t1'), ('velocidade_t1', W1, 'w1_velocidade_t1'),
        ('aceleracao_t1', W1, 'w1_aceleracao_t1'), ('temp_med_forecast_t1', W1, 'w1_temp_forecast_t1'),
        ('umid_med_forecast_t1', W1, 'w1_umid_forecast_t1'), ('precip_tot_forecast_t1', W1, 'w1_precip_forecast_t1'),
        ('pressure_min_t1', W1, 'w1_pressure_min_t1'), ('access_count_t1', W1, 'w1_acessos_afya_t1'),
        ('population_t1', W1, 'w1_population_t1'),
        ('casos_suavizados_t1', W2, 'w2_dengue_t1'), ('casos_chikungunya_t1', W2, 'w2_chikungunya_t1'),
        ('velocidade_t1', W2, 'w2_velocidade_t1'), ('precip_tot_forecast_t1', W2, 'w2_precip_forecast_t1'),
        ('umid_med_forecast_t1', W2, 'w2_umid_forecast_t1'), ('aceleracao_t1', W2, 'w2_aceleracao_t1'),
        ('temp_med_forecast_t1', W2, 'w2_temp_forecast_t1'), ('population_t1', W2, 'w2_population_t1'),
    ]
    for src, W, dst in esp_map:
        C[dst] = ('mat', espacial(W, t1[src]), 'float32')
    # ---- flags de qualidade
    C['afya_disponivel'] = ('tempo', P['afya_disponivel'], 'int8')
    C['chik_disponivel'] = ('tempo', P['chik_disponivel'], 'int8')
    C['pop_imputada'] = ('mat', mat('pop_imputada'), 'int8')
    C['clima_imputado'] = ('no', P['clima_imputado'], 'int8')
    C['forecast_imputado'] = ('mat', mat('forecast_imputado'), 'int8')
    C['w1_vizinho_knn'] = ('no', P['w1_vizinho_knn'], 'int8')

    # ------------------------------------------------------------------ FINAL_NN
    def clip1(f):
        return lambda s: np.clip(np.asarray(f(s), np.float64), 1, None)

    def log_inc(fc, fp):
        return lambda s: np.log1p(np.asarray(fc(s), np.float64) / clip1(fp)(s) * 1e5)

    def cf(nome):
        return C[nome][1]

    transform = {
        'log_pop_t1': lambda s: np.log(clip1(cf('population_t1'))(s)),
        'w1_log_pop_t1': lambda s: np.log(clip1(cf('w1_population_t1'))(s)),
        'w2_log_pop_t1': lambda s: np.log(clip1(cf('w2_population_t1'))(s)),
        'log_inc_dengue_t1': log_inc(cf('casos_suavizados_t1'), cf('population_t1')),
        'log_inc_chiku_t1': log_inc(cf('casos_chikungunya_t1'), cf('population_t1')),
        'log_inc_chiku': log_inc(cf('casos_chikungunya'), cf('population')),
        'w1_log_inc_dengue_t1': log_inc(cf('w1_dengue_t1'), cf('w1_population_t1')),
        'w2_log_inc_dengue_t1': log_inc(cf('w2_dengue_t1'), cf('w2_population_t1')),
        'access_count': lambda s: np.log1p(np.asarray(cf('access_count')(s), np.float64)),        # NOVO: log1p
        'access_count_t1': lambda s: np.log1p(np.asarray(cf('access_count_t1')(s), np.float64)),  # NOVO: log1p
        # ---- candidatas novas (entram só no FINAL_NN_v2, se o SHAP corrigido as escolher)
        # incidência OBSERVADA (não suavizada) na própria semana t e nas anteriores. Como o modelo usa a
        # linha t para prever t+1, elas são "o último dado conhecido" (a mesma informação da persistência)
        'log_inc_dengue_obs': log_inc(cf('casos.x'), cf('population')),
        'log_inc_dengue_obs_lag1': log_inc(mat('casos', 1), mat('pop', 1)),
        'log_inc_dengue_obs_lag2': log_inc(mat('casos', 2), mat('pop', 2)),
        'w1_log_inc_dengue_obs': log_inc(espacial(W1, mat('casos')), espacial(W1, mat('pop'))),
        'w2_log_inc_dengue_obs': log_inc(espacial(W2, mat('casos')), espacial(W2, mat('pop'))),
        'w1_log_inc_chiku_t1': log_inc(cf('w1_chikungunya_t1'), cf('w1_population_t1')),
        'w2_log_inc_chiku_t1': log_inc(cf('w2_chikungunya_t1'), cf('w2_population_t1')),
        'w1_acessos_afya_t1': lambda s: np.log1p(np.asarray(cf('w1_acessos_afya_t1')(s), np.float64)),
    }
    ordem_nn = ['geocode', 'date', 'epiweek', 'uf_code', 'population', 'macroregional_geocode', 'regional_geocode',
                'casos.x', 'seno_semana', 'umid_med_forecast', 'imunidade_ewm', 'precip_tot_forecast', 'access_count',
                'temp_med_forecast', 'koppen_encoded', 'time_index', 'velocidade', 'aceleracao', 'node_id',
                'imunidade_ewm_t1', 'access_count_t1', 'temp_med_forecast_t1', 'pressure_min_t1', 'w1_imunidade_t1',
                'w1_velocidade_t1', 'w1_aceleracao_t1', 'w1_temp_forecast_t1', 'w1_umid_forecast_t1',
                'w2_velocidade_t1', 'w2_precip_forecast_t1', 'w2_umid_forecast_t1', 'w2_aceleracao_t1',
                'w2_temp_forecast_t1', 'year', 'temp_min', 'temp_med', 'rel_humid_max', 'koppen', 'biome',
                'macroregion_code', 'forecast_months_ahead', 'volatilidade_temp_4w', 'log_pop', 'incidencia',
                'cosseno_semana', 'temp_min_lag4', 'pressure_max_lag3', 'log_pop_t1', 'w1_log_pop_t1',
                'w2_log_pop_t1', 'log_inc_dengue_t1', 'log_inc_chiku_t1', 'log_inc_chiku', 'w1_log_inc_dengue_t1',
                'w2_log_inc_dengue_t1', 'afya_disponivel', 'chik_disponivel']
    protegidas = {'geocode', 'date', 'epiweek', 'uf_code', 'macroregional_geocode', 'regional_geocode',
                  'macroregion_code', 'node_id', 'time_index', 'year', 'forecast_months_ahead', 'casos.x',
                  'population', 'koppen', 'biome', 'incidencia', 'log_pop', 'afya_disponivel', 'chik_disponivel'}
    if ARGS.etapa == 'nn_v2':
        extras = json.loads(ARQ_EXTRAS_NN.read_text(encoding='utf-8'))
        desconhecidas = [c for c in extras if c not in transform and c not in C]
        if desconhecidas:
            raise KeyError(f'colunas extras sem definição: {desconhecidas}')
        ordem_nn = ordem_nn + [c for c in extras if c not in ordem_nn]
        log(f'FINAL_NN_v2: {len(ordem_nn)} colunas ({len(extras)} extras do SHAP corrigido)')
    NN: dict = {}
    for nome in ordem_nn:
        if nome in transform:
            NN[nome] = ('mat', transform[nome], 'float32', True)
        else:
            tipo, fonte, dt = C[nome]
            NN[nome] = (tipo, fonte, dt, nome not in protegidas and dt.startswith('float'))
    return C, NN


# =============================================================================
# 6. GRAVAÇÃO EM BLOCOS
# =============================================================================
def _pa_array(v: np.ndarray, dt: str) -> pa.Array:
    if dt == 'string':
        return pa.array(v.astype(object), type=pa.string())
    if dt == 'bool':
        return pa.array(v.astype(bool))
    if dt.startswith('int'):
        if np.issubdtype(v.dtype, np.floating):
            if np.isnan(v).any():
                raise ValueError('NaN em coluna inteira')
            v = np.rint(v)
        return pa.array(v.astype(dt))
    v = v.astype(np.float32)
    return pa.array(v, type=pa.float32(), from_pandas=True)      # NaN -> null


def valores_bloco(G, spec_item, s: slice) -> np.ndarray:
    tipo, fonte = spec_item[0], spec_item[1]
    nb = s.stop - s.start
    if tipo == 'no':
        return np.tile(np.asarray(fonte), nb)
    if tipo == 'tempo':
        return np.repeat(np.asarray(fonte)[s], G.N)
    return np.ascontiguousarray(np.asarray(fonte(s)).T).reshape(-1)   # (semana, nó) -> ordem date, node_id


def gravar(caminho: Path, G, spec: dict, t_ini: int, t_fim: int, escala: dict | None = None) -> None:
    writer = None
    for a in range(t_ini, t_fim, BLOCO_SEMANAS):
        s = slice(a, min(a + BLOCO_SEMANAS, t_fim))
        cols = {}
        for nome, item in spec.items():
            v = valores_bloco(G, item, s)
            if escala is not None and nome in escala:
                mu, sd = escala[nome]
                v = (np.asarray(v, np.float64) - mu) / sd
                v = np.where(np.isnan(v), 0.0, v)                  # faltante -> média do treino (z = 0)
            cols[nome] = _pa_array(v, item[2])
        tab = pa.table(cols)
        if writer is None:
            writer = pq.ParquetWriter(caminho, tab.schema, compression='zstd')
        writer.write_table(tab)
    writer.close()


# =============================================================================
# 7. VALIDAÇÃO FINAL (lendo de volta os parquets)
# =============================================================================
def validar_saida(P: dict, C: dict, NN: dict, arquivos: dict, escala: dict) -> None:
    G = P['G']
    log('Validando arquivos gravados ...')
    nan_afya = {'access_count', 'access_count_t1', 'afya_log_diff', 'w1_acessos_afya_t1'}
    # _t1 da 1ª semana de treino (201001) aponta para 200953: não há dengue (imunidade) nem previsão antes de 2010
    nan_1a_semana = {'imunidade_ewm_t1', 'w1_imunidade_t1', 'temp_med_forecast_t1', 'umid_med_forecast_t1',
                     'precip_tot_forecast_t1', 'w1_temp_forecast_t1', 'w1_umid_forecast_t1', 'w1_precip_forecast_t1',
                     'w2_temp_forecast_t1', 'w2_umid_forecast_t1', 'w2_precip_forecast_t1'}
    for rotulo, (caminho, t_ini, t_fim, tipo) in arquivos.items():
        pf = pq.ParquetFile(caminho)
        n_sem = t_fim - t_ini
        chaves = pf.read(columns=['geocode', 'date', 'epiweek']).to_pandas()
        checar(f'{rotulo}: nº de linhas = municípios x semanas', len(chaves) == G.N * n_sem, f'{len(chaves):,}')
        checar(f'{rotulo}: nenhum (geocode, date) repetido', not chaves.duplicated(['geocode', 'date']).any())
        cont = chaves.groupby('geocode').size()
        checar(f'{rotulo}: cada município aparece em todas as semanas', cont.nunique() == 1 and cont.iloc[0] == n_sem)
        dt = pd.to_datetime(chaves['date'].drop_duplicates())
        checar(f'{rotulo}: todas as datas são domingo e contínuas',
               (dt.dt.weekday == 6).all() and set(dt.diff().dt.days.dropna()) == {7.0})
        checar(f'{rotulo}: janela de epiweeks correta',
               chaves.epiweek.min() == G.ew[t_ini] and chaves.epiweek.max() == G.ew[t_fim - 1])
        checar(f'{rotulo}: sem Espírito Santo', not (chaves.geocode // 100000 == UF_EXCLUIDA).any())
        tab = pf.read()
        nulos = {c: tab[c].null_count for c in tab.column_names if tab[c].null_count > 0}
        if tipo == 'COMPLETO':
            limite_1a = G.N if t_ini == P['i0'] else 0
            inesperado = {c: n for c, n in nulos.items()
                          if c not in nan_afya and not (c in nan_1a_semana and n <= limite_1a)}
            checar(f'{rotulo}: nulos só onde esperado (Afya antes de 2021; _t1 da 1ª semana de 2010)',
                   not inesperado, str(inesperado))
            if 'access_count' in nulos:
                af_nan = tab.filter(pc.is_null(tab['access_count']))['date'].to_pandas()
                checar(f'{rotulo}: access_count nulo só antes de 2021-01-03', af_nan.max() < '2021-01-03',
                       f'última data nula: {af_nan.max()}')
            RELATORIO.append((f'{rotulo}: nulos esperados', True, json.dumps(nulos)))
            # ida e volta: amostra do arquivo = matriz em memória
            rng = np.random.default_rng(7)
            linhas = rng.integers(0, tab.num_rows, 2000)
            for c in ['casos.x', 'temp_med', 'precip_tot_forecast', 'access_count', 'w1_dengue_t1', 'enso_lag8']:
                arq = tab[c].to_numpy(zero_copy_only=False)[linhas].astype(np.float64)
                mem = valores_bloco(G, C[c], slice(t_ini, t_fim))[linhas].astype(np.float64)
                checar(f'{rotulo}: arquivo = memória ({c})', np.allclose(arq, mem, equal_nan=True, rtol=1e-5))
        else:
            checar(f'{rotulo}: FINAL_NN sem nenhum nulo/NaN', not nulos, str(nulos))
            if 'treino' in rotulo:
                # Afya só existe a partir de 2021: depois do z-score o NaN vira 0, então o dp fica < 1 por construção
                df = tab.select([c for c in escala if c not in SO_DESDE_2021]).to_pandas()
                checar(f'{rotulo}: colunas escalonadas com média ~0 e dp ~1 no treino',
                       (df.mean().abs() < 0.05).all() and ((df.std() - 1).abs() < 0.05).all(),
                       f'max|média|={df.mean().abs().max():.3f}')
            cols_seca = ['velocidade', 'aceleracao', 'access_count', 'seno_semana', 'temp_min', 'w2_velocidade_t1',
                         'w1_log_pop_t1', 'w1_imunidade_t1', 'w2_log_inc_dengue_t1', 'w1_log_inc_dengue_t1',
                         'imunidade_ewm', 'imunidade_ewm_t1', 'log_inc_dengue_t1', 'log_inc_chiku', 'log_pop_t1',
                         'rel_humid_max', 'log_inc_chiku_t1', 'cosseno_semana', 'w1_velocidade_t1', 'temp_med',
                         'koppen_encoded']
            cols_pico = ['velocidade', 'aceleracao', 'access_count', 'seno_semana', 'w2_velocidade_t1', 'w1_log_pop_t1',
                         'w2_aceleracao_t1', 'access_count_t1', 'w1_log_inc_dengue_t1', 'imunidade_ewm',
                         'imunidade_ewm_t1', 'log_inc_dengue_t1', 'log_inc_chiku', 'log_pop_t1', 'cosseno_semana',
                         'w1_velocidade_t1', 'w1_aceleracao_t1', 'w2_log_inc_dengue_t1', 'w2_temp_forecast_t1',
                         'w1_temp_forecast_t1', 'log_inc_chiku_t1', 'umid_med_forecast', 'temp_min', 'koppen_encoded',
                         'w2_umid_forecast_t1', 'temp_med_forecast', 'w2_precip_forecast_t1']
            falt = [c for c in set(cols_seca + cols_pico) if c not in tab.column_names]
            checar(f'{rotulo}: contém todas as COLS_SECA/COLS_PICO do notebook', not falt, str(falt))
        del tab

    # ---- comparação opcional com o banco antigo deduplicado (valida a malha Queen)
    antigo = DIR_BASE / 'bancos' / f'train{SEASON}_dengue_eng_shap_final_spatial_v2.parquet'
    if antigo.exists():
        log(f'Comparando W1 com o banco antigo (bancos/train{SEASON}_..._spatial_v2) ...')
        old = pq.read_table(antigo, columns=['geocode', 'date', 'w1_population_t1', 'casos.x']).to_pandas()
        old = old[old['date'] == old['date'].max()]
        t = int(np.flatnonzero(G.datas == pd.Timestamp(old['date'].iloc[0]))[0])
        novo = pd.Series(valores_bloco(G, C['w1_population_t1'], slice(t, t + 1)), index=G.geo)
        excl = set(P['knn'].keys()) | {v[0] for v in P['knn'].values()}
        o = old.set_index('geocode')['w1_population_t1']
        o = o[~o.index.isin(excl)]
        dif = (o - novo.reindex(o.index)).abs() / o.clip(lower=1)
        checar('malha: W1 idêntica ao libpysal.Queen do notebook (w1_population_t1, fora das ilhas)',
               (dif < 1e-4).mean() > 0.999, f'{(dif >= 1e-4).sum()} de {len(dif)} municípios diferem')
        c_old = old.set_index('geocode')['casos.x']
        c_new = pd.Series(valores_bloco(G, C['casos.x'], slice(t, t + 1)), index=G.geo)
        checar('dengue: casos da última semana de treino = banco antigo', (c_old == c_new.reindex(c_old.index)).all())


def ajustar_escala(G, NN: dict, tr: tuple) -> dict:
    """StandardScaler ajustado SÓ nas linhas de treino (dp populacional, igual ao sklearn)."""
    log('Ajustando StandardScaler nas linhas de treino ...')
    escala = {}
    for nome, item in NN.items():
        if item[3]:
            x = np.asarray(valores_bloco(G, item, slice(*tr)), np.float64)
            mu, sd = np.nanmean(x), np.nanstd(x)
            escala[nome] = (mu, sd if sd > 0 else 1.0)
    return escala


def gravar_nn_v2(P: dict, NN: dict, tr: tuple, te: tuple) -> int:
    """FINAL_NN_v2 = mesmas colunas/transformações do FINAL_NN + colunas extras do SHAP corrigido."""
    G = P['G']
    escala = ajustar_escala(G, NN, tr)
    pd.DataFrame([(k, *v) for k, v in escala.items()], columns=['coluna', 'media_treino', 'dp_treino']) \
        .to_csv(DIR_OUT / f'scaler_FINAL_NN_v2_season{SEASON}.csv', index=False)
    spec_nn = {k: v[:3] for k, v in NN.items()}
    # 'contexto' = semanas entre o fim do treino e o início do teste (EW26–40), escalonadas com o scaler do
    # treino. Servem só para aquecer o estado recorrente na inferência: a saída da última delas é a previsão
    # da 1ª semana do teste (o modelo aprende X[t] -> y[t+1]).
    arqs = {'treino': (DIR_OUT / f'train{SEASON}_FINAL_NN_v2.parquet', *tr),
            'teste': (DIR_OUT / f'test{SEASON}_FINAL_NN_v2.parquet', *te),
            'contexto': (DIR_OUT / f'ctx{SEASON}_FINAL_NN_v2.parquet', tr[1], te[0])}
    for rot, (caminho, a, b) in arqs.items():
        log(f'Gravando {caminho.name} ...')
        gravar(caminho, G, spec_nn, a, b, escala)
    for rot, (caminho, a, b) in arqs.items():
        tab = pq.read_table(caminho)
        nulos = {c: tab[c].null_count for c in tab.column_names if tab[c].null_count > 0}
        checar(f'FINAL_NN_v2 {rot}: nº de linhas = municípios x semanas', tab.num_rows == G.N * (b - a))
        checar(f'FINAL_NN_v2 {rot}: sem nulos/NaN', not nulos, str(nulos))
        chaves = tab.select(['geocode', 'date']).to_pandas()
        checar(f'FINAL_NN_v2 {rot}: nenhum (geocode, date) repetido', not chaves.duplicated().any())
        ew = tab['epiweek'].to_numpy()
        checar(f'FINAL_NN_v2 {rot}: EW {ew.min()}–{ew.max()}', ew.min() == G.ew[a] and ew.max() == G.ew[b - 1])
        antigo = DIR_OUT / f'{"train" if rot == "treino" else "test"}{SEASON}_FINAL_NN.parquet'
        if rot != 'contexto' and antigo.exists():
            v1 = pq.read_table(antigo)
            comuns = [c for c in v1.column_names if c in tab.column_names]
            iguais = all(v1[c].equals(tab[c]) for c in comuns)
            checar(f'FINAL_NN_v2 {rot}: as {len(comuns)} colunas do FINAL_NN são idênticas às do v1', iguais)
            del v1
        if rot == 'treino':
            df = tab.select([c for c in escala if c not in SO_DESDE_2021]).to_pandas()
            checar('FINAL_NN_v2 treino: escalonadas com média ~0 e dp ~1 (fora Afya, que só existe desde 2021)',
                   (df.mean().abs() < 0.05).all() and ((df.std() - 1).abs() < 0.05).all())
        del tab
    falhas = [r for r in RELATORIO if not r[1]]
    linhas = [f'# Relatório — FINAL_NN_v2 Season {SEASON}', '',
              f'Gerado por `refazer_season.py --season {SEASON} --etapa nn_v2` em {pd.Timestamp.now():%Y-%m-%d %H:%M} · '
              f'{len(RELATORIO) - len(falhas)}/{len(RELATORIO)} checagens OK', '',
              f'Colunas ({len(NN)}): ' + ', '.join(f'`{c}`' for c in NN), '',
              '| Checagem | Resultado | Detalhe |', '|---|---|---|']
    linhas += [f'| {n} | {"✅" if ok else "❌"} | {d.replace("|", "/")} |' for n, ok, d in RELATORIO]
    (DIR_OUT / f'relatorio_FINAL_NN_v2_season{SEASON}.md').write_text('\n'.join(linhas), encoding='utf-8')
    log(f'Fim (nn_v2). {len(falhas)} falha(s).')
    return 1 if falhas else 0


# =============================================================================
# 8. MAIN
# =============================================================================
def main() -> int:
    P = montar()
    engenharia(P)
    C, NN = especificar(P)
    G = P['G']
    tr = (int(G.idx_sem(np.array([EW_TREINO[0]]))[0]), int(G.idx_sem(np.array([EW_TREINO[1]]))[0]) + 1)
    te = (int(G.idx_sem(np.array([EW_TESTE[0]]))[0]), int(G.idx_sem(np.array([EW_TESTE[1]]))[0]) + 1)
    log(f'Treino: índices {tr} ({tr[1] - tr[0]} semanas) · Teste: {te} ({te[1] - te[0]} semanas)')

    if ARGS.etapa == 'nn_v2':
        return gravar_nn_v2(P, NN, tr, te)

    spec_c = {k: v for k, v in C.items()}
    arquivos = {
        'COMPLETO treino': (DIR_OUT / f'train{SEASON}_dengue_eng_COMPLETO_spatial.parquet', *tr, 'COMPLETO'),
        'COMPLETO teste': (DIR_OUT / f'test{SEASON}_dengue_eng_COMPLETO_spatial.parquet', *te, 'COMPLETO'),
        'FINAL_NN treino': (DIR_OUT / f'train{SEASON}_FINAL_NN.parquet', *tr, 'NN'),
        'FINAL_NN teste': (DIR_OUT / f'test{SEASON}_FINAL_NN.parquet', *te, 'NN'),
    }
    for rot in ('COMPLETO treino', 'COMPLETO teste'):
        caminho, a, b, _ = arquivos[rot]
        log(f'Gravando {caminho.name} ...')
        gravar(caminho, G, spec_c, a, b)

    escala = ajustar_escala(G, NN, tr)
    pd.DataFrame([(k, *v) for k, v in escala.items()], columns=['coluna', 'media_treino', 'dp_treino']) \
        .to_csv(DIR_OUT / f'scaler_FINAL_NN_season{SEASON}.csv', index=False)
    spec_nn = {k: v[:3] for k, v in NN.items()}
    for rot in ('FINAL_NN treino', 'FINAL_NN teste'):
        caminho, a, b, _ = arquivos[rot]
        log(f'Gravando {caminho.name} ...')
        gravar(caminho, G, spec_nn, a, b, escala)

    # artefatos auxiliares
    est = P['estatico'].copy()
    for c in ('clima_imputado', 'w1_vizinho_knn'):
        est[c] = P[c]
    est['forecast_imputado_alguma_semana'] = P['forecast_imputado'][:, P['i0']:].any(axis=1).astype(np.int8)
    est['n_vizinhos_w1'] = np.asarray(P['B1'].sum(axis=1)).ravel().astype(int)
    est.to_csv(DIR_OUT / f'nos_season{SEASON}.csv', index=False, encoding='utf-8')
    sparse.save_npz(DIR_OUT / f'W1_queen_season{SEASON}.npz', P['W1'].tocsr())
    sparse.save_npz(DIR_OUT / f'W2_season{SEASON}.npz', P['W2'].tocsr())

    validar_saida(P, C, NN, arquivos, escala)

    # relatório
    falhas = [r for r in RELATORIO if not r[1]]
    linhas = [f'# Relatório de validação — Season {SEASON} corrigida', '',
              f'Gerado por `refazer_season.py --season {SEASON}` em {pd.Timestamp.now():%Y-%m-%d %H:%M} · '
              f'{len(RELATORIO) - len(falhas)}/{len(RELATORIO)} checagens OK · tempo {time.time() - T0:.0f}s', '',
              f'- Grade: **{G.N} municípios** · treino {tr[1] - tr[0]} semanas · teste {te[1] - te[0]} semanas',
              f'- Ilhas ligadas ao vizinho mais próximo (W1): `{P["knn"]}`',
              f'- Target encoding (média de casos/semana no treino) koppen: `{ {k: round(v, 3) for k, v in P["_enc_koppen"].items()} }`',
              f'- Target encoding bioma: `{ {k: round(v, 3) for k, v in P["_enc_biome"].items()} }`', '',
              '| Checagem | Resultado | Detalhe |', '|---|---|---|']
    linhas += [f'| {n} | {"✅" if ok else "❌"} | {d.replace("|", "/")} |' for n, ok, d in RELATORIO]
    for rot, (caminho, *_r) in arquivos.items():
        m = pq.ParquetFile(caminho).metadata
        linhas.append(f'\n- `{caminho.name}`: {m.num_rows:,} linhas × {m.num_columns} colunas · '
                      f'{caminho.stat().st_size / 1e6:,.1f} MB')
    (DIR_OUT / f'relatorio_validacao_season{SEASON}.md').write_text('\n'.join(linhas), encoding='utf-8')
    log(f'Fim. {len(falhas)} falha(s). Relatório: relatorio_validacao_season{SEASON}.md')
    return 1 if falhas else 0


if __name__ == '__main__':
    sys.exit(main())
