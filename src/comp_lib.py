"""
Competição de modelos · núcleo comum (dados, execução padronizada, soma até a UF, avaliação e gráficos).

Todo modelo:
  - recebe o Contexto (dados por região de saúde e por UF) e a temporada s;
  - só usa dados até a SE25 do ano s;
  - devolve, POR REGIÃO DE SAÚDE, trajetórias (M × semanas × regiões) OU quantis (semanas × regiões × 9).
Depois, igual para todos:
  - quantis → amostras (interpolação em escala log com caudas gaussianas);
  - modelos sem dependência própria entre regiões/semanas recebem a dependência histórica pela reordenação de
    Schaake (ranks de temporadas passadas + ruído);
  - soma das regiões → UF;
  - grava quantis por região e por UF.
"""
import os
import time
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import v4_lib as V

warnings.filterwarnings('ignore')
Q, QCOLS = V.Q, V.QCOLS
ZQ = norm.ppf(Q)
MACRO_BR = {1: 'Norte', 2: 'Nordeste', 3: 'Sudeste', 4: 'Sul', 5: 'Centro-Oeste'}


# =====================================================================================================================
# Contexto
# =====================================================================================================================
class Contexto:
    def __init__(self, pasta_dados, pasta_ref=None):
        self.Dr = V.Dados(pasta_dados, 'regiao')
        self.Du = V.Dados(pasta_dados, 'uf')
        self.ver_uf = V.Verdade(self.Du)
        self.ver_reg = V.Verdade(self.Dr)
        self.ver_reg.ufs = self.Dr.unidades              # na região, os "ids" são os códigos das regiões
        self.ref = V.ler_referencias_imdc3(pasta_ref, self.Du.ufs) if pasta_ref else None
        self.pasta_dados = pasta_dados

    def alvo(self, s):
        datas = V.datas_temporada(s)
        idx = np.array([self.Dr.idx_data[pd.Timestamp(d)] for d in datas])
        return datas, idx


class Saida:
    """traj: (M, W, N) casos por região · quant: (W, N, 9) · dependencia: 'propria' ou 'independente'."""

    def __init__(self, traj=None, quant=None, dependencia='independente', info=None):
        self.traj, self.quant, self.dependencia, self.info = traj, quant, dependencia, info or {}


class Modelo:
    nome = 'base'
    familia = ''
    origem = ''
    usa_gpu = False
    temporadas = None            # None = todas as da configuração
    dependencia = 'independente'

    def prever(self, ctx, s):
        raise NotImplementedError


# =====================================================================================================================
# quantis ⇄ amostras e dependência (Schaake)
# =====================================================================================================================
def amostrar_quantis(q, M):
    """q: (W, N, 9) em contagens → (M, W, N). Interpolação linear em log1p contra o z normal dos níveis,
    com extrapolação linear nas caudas (equivale a caudas log-normais)."""
    lq = np.log1p(np.maximum(np.sort(q, -1), 0))
    z = norm.ppf((np.arange(M) + 0.5) / M)
    i = np.clip(np.searchsorted(ZQ, z), 1, len(ZQ) - 1)
    t = (z - ZQ[i - 1]) / (ZQ[i] - ZQ[i - 1])
    v = lq[..., i - 1] + t * (lq[..., i] - lq[..., i - 1])       # (W, N, M)
    return np.clip(np.expm1(np.moveaxis(v, -1, 0)), 0, None).astype(np.float32)


def template_dependencia(ctx, s, M, a=0.85, seed=0):
    """campo latente (M, W, N): anomalia padronizada de log-incidência de uma temporada passada sorteada
    (dependência espacial e temporal real) + ruído, com peso a."""
    D = ctx.Dr
    rng = np.random.default_rng(seed + s)
    c = D.corte(s)
    W = len(V.datas_temporada(s))
    temps = [j for j in range(2010, s) if D.temporada_completa(j, c) and D.idx_temporada(j)[0] > 0]
    F = []
    for j in temps:
        x = D.LINC[D.idx_temporada(j)]
        x = np.vstack([x, np.repeat(x[-1:], max(0, W - len(x)), 0)])[:W]
        F.append(x)
    F = np.stack(F)
    A = (F - F.mean(0)) / (F.std(0) + 1e-6)
    js = rng.integers(0, len(temps), M)
    return (a * A[js] + np.sqrt(1 - a ** 2) * rng.standard_normal((M, W, D.N))).astype(np.float32)


def schaake(amostras, Z):
    """reordena as amostras de cada (semana, região) segundo os ranks do campo Z."""
    ordenadas = np.sort(amostras, 0)
    rk = np.argsort(np.argsort(Z, 0), 0)
    return np.take_along_axis(ordenadas, rk, 0)


def quantis(amostras):
    return V.arruma_quantis(np.moveaxis(np.quantile(amostras, Q, axis=0), 0, -1))


# =====================================================================================================================
# execução padronizada
# =====================================================================================================================
def _arq(pasta, nivel, nome, s):
    return os.path.join(pasta, nivel, f'{nome}_s{s}.parquet')


def ja_rodou(pasta, nome, s):
    return os.path.exists(_arq(pasta, 'uf', nome, s))


def rodar(ctx, modelo, s, pasta, M=500, refazer=False, seed=0):
    """roda um modelo numa temporada e grava quantis por região e por UF. Devolve o tempo (s) ou None se já existia."""
    for n in ('regiao', 'uf', 'log'):
        os.makedirs(os.path.join(pasta, n), exist_ok=True)
    if ja_rodou(pasta, modelo.nome, s) and not refazer:
        return None
    t0 = time.time()
    out = modelo.prever(ctx, s)
    datas = V.datas_temporada(s)
    D = ctx.Dr
    if out.quant is not None:
        q_reg = V.arruma_quantis(out.quant)
        amostras = amostrar_quantis(q_reg, M)
    else:
        amostras = np.asarray(out.traj, dtype=np.float32)
        q_reg = quantis(amostras)
    assert q_reg.shape == (len(datas), D.N, 9), f'{modelo.nome}: formato {q_reg.shape}'
    dep = out.dependencia if out.dependencia else modelo.dependencia
    if dep != 'propria':
        amostras = schaake(amostras, template_dependencia(ctx, s, amostras.shape[0], seed=seed))
    q_uf = quantis(amostras @ D.A_uf.T)
    reg = V.para_df(q_reg, datas, D.unidades, modelo.nome, s).rename(columns={'uf_code': 'unidade'})
    reg['uf_code'] = np.tile(D.uf_of, len(datas))
    reg.to_parquet(_arq(pasta, 'regiao', modelo.nome, s), index=False)
    V.para_df(q_uf, datas, D.ufs, modelo.nome, s).to_parquet(_arq(pasta, 'uf', modelo.nome, s), index=False)
    dt = time.time() - t0
    with open(os.path.join(pasta, 'log', f'{modelo.nome}_s{s}.json'), 'w') as f:
        json.dump(dict(modelo=modelo.nome, season=s, segundos=dt, **out.info), f, default=str)
    return dt


def ler_resultados(pasta, nivel):
    import glob
    fs = glob.glob(os.path.join(pasta, nivel, '*.parquet'))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


# =====================================================================================================================
# leitura robusta de previsões (qualquer origem: nossos modelos, ensembles, equipes externas, CSV de submissão)
# =====================================================================================================================
COLS_PREV = ['modelo', 'season', 'uf_code', 'date'] + QCOLS
ULTIMO_RELATORIO = pd.DataFrame(columns=['fonte', 'modelo', 'season', 'acao'])
_SINONIMOS = {'median': 'pred', 'mediana': 'pred', 'q0.5': 'pred', 'q_0.5': 'pred', '0.5': 'pred'}
for _q, _c in zip(Q, QCOLS):
    for _f in (f'q{_q:g}', f'q_{_q:g}', f'{_q:g}', f'q{_q}', f'q_{_q}', str(_q)):
        _SINONIMOS.setdefault(_f, _c)


def temporada_da_data(d):
    """temporada s a que a semana pertence (SE41 do ano s → SE40 do ano s+1)."""
    e = V.data_para_epiweek(d)
    return e // 100 if e % 100 >= 41 else e // 100 - 1


def padronizar_previsoes(df, nome=None, nivel='uf'):
    """converte um DataFrame de previsões para o formato comum (COLS_PREV). Aceita:
      - o nosso formato longo (modelo, season, uf_code, date, lower_95 … upper_95);
      - o formato de submissão da plataforma (date, adm_1, pred, lower_*, upper_*), com ou sem 'season'/'validation';
      - quantis nomeados por nível (q0.025 … q0.975, 'median').
    Unidade: 'unidade' (região) ou 'uf_code'/'adm_1'/'geocode'. Sem 'season', ela é deduzida da data."""
    d = df.rename(columns={c: _SINONIMOS.get(str(c).lower(), c) for c in df.columns}).copy()
    if nome is not None:
        d['modelo'] = nome
    if 'modelo' not in d.columns:
        raise ValueError("previsão sem coluna 'modelo': passe nome=...")
    cand = (['unidade'] if nivel == 'regiao' else []) + ['uf_code', 'adm_1', 'geocode', 'unidade']
    col_u = next((c for c in cand if c in d.columns), None)
    if col_u is None:
        raise ValueError(f'previsão sem coluna de unidade (uma de {cand})')
    d['uf_code'] = pd.to_numeric(d[col_u], errors='coerce')
    d['date'] = pd.to_datetime(d['date']).dt.normalize()
    falta = [c for c in QCOLS if c not in d.columns]
    if falta:
        raise ValueError(f'previsão sem os quantis {falta}: o WIS e as métricas de compra precisam dos 9 '
                         f'(mediana + intervalos de 50, 80, 90 e 95%)')
    if 'season' not in d.columns:
        if 'validation' in d.columns:
            d['season'] = 2021 + d['validation']
        else:
            sd = {x: temporada_da_data(x) for x in d['date'].unique()}
            d['season'] = d['date'].map(sd)
    d['season'] = d['season'].astype(int)
    d[QCOLS] = d[QCOLS].apply(pd.to_numeric, errors='coerce')
    d = d[COLS_PREV].dropna(subset=['uf_code'])
    d['uf_code'] = d['uf_code'].astype(int)
    q = d[QCOLS].values
    ok = ~np.isnan(q).any(1)
    d.loc[ok, QCOLS] = V.arruma_quantis(q[ok])           # ordena, ≥ 0 (linhas com NaN ficam NaN: contam como falta)
    return d


def _ler_fonte(f, nivel):
    """pasta de resultados (lê <pasta>/<nivel>/*.parquet), arquivo .parquet/.csv(.gz) ou DataFrame."""
    if isinstance(f, pd.DataFrame):
        return [('DataFrame', f)]
    if os.path.isdir(f):
        sub = os.path.join(f, nivel)
        d = ler_resultados(f, nivel) if os.path.isdir(sub) else pd.DataFrame()
        return [(f, d)] if len(d) else []
    if not os.path.exists(f):
        return []
    d = pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
    return [(f, d)]


def carregar_previsoes(fontes, ids, nivel='uf', so_completas=True, mostrar=True):
    """junta previsões de várias fontes num formato só, pronto para avaliar e fazer ensemble.

    fontes: lista de pastas, arquivos ou DataFrames, ou de pares (rótulo, fonte). Fontes inexistentes são puladas.
    ids:    as unidades esperadas (ctx.Du.ufs ou ctx.Dr.unidades).
    Regras (tudo vai para o relatório, guardado em L.ULTIMO_RELATORIO):
      - o mesmo (modelo, temporada) em duas fontes → fica o da fonte listada POR ÚLTIMO;
      - linhas repetidas dentro de uma fonte → fica a última;
      - semanas fora da temporada e unidades fora de `ids` são descartadas;
      - (modelo, temporada) sem todas as semanas × unidades (ou com NaN) → descartado se so_completas=True,
        para que todo modelo seja comparado nas mesmas células (comparação justa)."""
    global ULTIMO_RELATORIO
    ids = np.asarray(ids)
    partes, rel = [], []
    for k, item in enumerate(fontes):
        rot, f = item if isinstance(item, tuple) else (None, item)
        for orig, d in _ler_fonte(f, nivel):
            if not len(d):
                continue
            try:
                d = padronizar_previsoes(d, nivel=nivel)
            except ValueError as e:
                rel.append(dict(fonte=rot or orig, modelo='?', season=None, acao=f'IGNORADA: {e}'))
                continue
            d['fonte'] = rot or (orig if isinstance(orig, str) else f'fonte{k}')
            d['_ordem'] = k
            partes.append(d)
    if not partes:
        out = pd.DataFrame(columns=COLS_PREV + ['fonte'])
        ULTIMO_RELATORIO = pd.DataFrame(rel, columns=['fonte', 'modelo', 'season', 'acao'])
        return out
    P = pd.concat(partes, ignore_index=True)
    # 1. o mesmo (modelo, temporada) em fontes diferentes: fica a última fonte
    dono = P.groupby(['modelo', 'season'])._ordem.max().rename('_dono')
    P = P.join(dono, on=['modelo', 'season'])
    tirar = P[P._ordem != P._dono]
    for (m, s, fo), _ in tirar.groupby(['modelo', 'season', 'fonte']):
        rel.append(dict(fonte=fo, modelo=m, season=s, acao='duplicado em outra fonte: ficou a última'))
    P = P[P._ordem == P._dono].drop(columns=['_ordem', '_dono'])
    # 2. repetidas dentro da fonte
    n0 = len(P)
    P = P.drop_duplicates(['modelo', 'season', 'uf_code', 'date'], keep='last')
    if len(P) < n0:
        rel.append(dict(fonte='*', modelo='*', season=None, acao=f'{n0 - len(P)} linhas repetidas removidas'))
    # 3. só unidades esperadas e semanas da temporada
    jan = {s: set(V.datas_temporada(s)) for s in P.season.unique()}
    dentro = P.uf_code.isin(ids) & np.array([d in jan[s] for d, s in zip(P.date, P.season)])
    for (m, s), g in P[~dentro].groupby(['modelo', 'season']):
        rel.append(dict(fonte=g.fonte.iloc[0], modelo=m, season=s,
                        acao=f'{len(g)} linhas fora da temporada ou de unidades desconhecidas: descartadas'))
    P = P[dentro]
    # 4. completude
    tirar = []
    for (m, s), g in P.groupby(['modelo', 'season']):
        esperado = len(jan[s]) * len(ids)
        cheio = int((~g[QCOLS].isna().any(axis=1)).sum())
        if cheio < esperado:
            acao = 'INCOMPLETA: fora da avaliação' if so_completas else 'incompleta (mantida)'
            rel.append(dict(fonte=g.fonte.iloc[0], modelo=m, season=s,
                            acao=f'{acao} ({cheio}/{esperado} células)'))
            if so_completas:
                tirar.append((m, s))
    if tirar:
        P = P[~pd.MultiIndex.from_frame(P[['modelo', 'season']]).isin(tirar)]
    P = P.reset_index(drop=True)
    R = pd.DataFrame(rel, columns=['fonte', 'modelo', 'season', 'acao'])
    ULTIMO_RELATORIO = R                 # fora do DataFrame: .attrs quebraria o to_parquet dos derivados
    if mostrar:
        cob = P.groupby('modelo').season.nunique()
        print(f'{P.modelo.nunique()} modelos · {P.season.nunique()} temporadas · {len(P):,} linhas '
              f'({(cob == cob.max()).sum()} modelos com todas as temporadas presentes)')
        if len(R):
            print(f'Relatório ({len(R)} avisos; a tabela inteira fica em L.ULTIMO_RELATORIO):')
            print(R.head(30).to_string(index=False))
    return P


def importar_externo(previsao, nome, pasta, ids, nivel='uf'):
    """grava a previsão de um modelo de FORA (CSV/parquet/DataFrame em qualquer formato aceito por
    padronizar_previsoes) em <pasta>/<nivel>/<nome>_s<temporada>.parquet. A partir daí ele entra em tudo
    (CompUF4, ensembles, Comp5) como qualquer modelo nosso. Devolve o relatório da checagem."""
    lidos = _ler_fonte(previsao, nivel)
    if not lidos:
        raise ValueError(f'{nome}: arquivo não encontrado ou vazio: {previsao}')
    bruto = pd.concat([d for _, d in lidos], ignore_index=True)
    P = carregar_previsoes([(nome, padronizar_previsoes(bruto, nome=nome, nivel=nivel))], ids, nivel,
                           so_completas=False, mostrar=False)
    if not len(P):
        raise ValueError(f'nada aproveitável em {nome}: {ULTIMO_RELATORIO.to_dict("records")}')
    P['modelo'] = nome
    os.makedirs(os.path.join(pasta, nivel), exist_ok=True)
    for s, g in P.groupby('season'):
        g = g[COLS_PREV].copy()
        if nivel == 'regiao':
            g = g.rename(columns={'uf_code': 'unidade'})
            g.insert(3, 'uf_code', -1)                     # mesmo layout dos arquivos por região (coluna ignorada)
        g.to_parquet(_arq(pasta, nivel, nome, s), index=False)
    return ULTIMO_RELATORIO


def tempos(pasta):
    import glob
    lin = [json.load(open(f)) for f in glob.glob(os.path.join(pasta, 'log', '*.json'))]
    return pd.DataFrame(lin)


# =====================================================================================================================
# avaliação
# =====================================================================================================================
def avaliar_uf(ctx, prev_uf, incluir_ref=True, temporadas=None):
    P = prev_uf if temporadas is None else prev_uf[prev_uf.season.isin(temporadas)]
    if incluir_ref and ctx.ref is not None:
        R = ctx.ref if temporadas is None else ctx.ref[ctx.ref.season.isin(temporadas)]
        P = pd.concat([P, R], ignore_index=True)
    return V.avaliar(P, ctx.ver_uf)


def avaliar_regiao(ctx, prev_reg, temporadas=None):
    P = prev_reg if temporadas is None else prev_reg[prev_reg.season.isin(temporadas)]
    P = P.drop(columns=['uf_code']).rename(columns={'unidade': 'uf_code'})
    por, res = V.avaliar(P, ctx.ver_reg, ref=('climatologia',))
    por = por.rename(columns={'uf_code': 'unidade'})
    por['uf_code'] = pd.Series(ctx.Dr.uf_of, index=ctx.Dr.unidades).reindex(por.unidade).values
    return por, res.rename(columns={'UFs_melhor_que_climatologia': 'regioes_melhor_que_climatologia'})


def alvos_temporada(ctx, prev_uf, temporadas):
    """erro no total da temporada (log da razão, pela mediana) e na semana do pico (pela curva mediana), por UF."""
    lin = []
    for (m, s), g in prev_uf[prev_uf.season.isin(temporadas)].groupby(['modelo', 'season']):
        datas = np.sort(g.date.unique())
        q = V.de_df(g, datas, ctx.Du.ufs)
        y = ctx.ver_uf.y(datas)
        ok = ~np.isnan(y[:, 0])
        if ok.sum() < 20:
            continue
        for j, u in enumerate(ctx.Du.ufs):
            lin.append(dict(modelo=m, season=s, uf_code=u,
                            erro_log_total=np.log((q[ok, j, 4].sum() + 1) / (y[ok, j].sum() + 1)),
                            erro_sem_pico=int(np.argmax(q[ok, j, 4]) - np.argmax(y[ok, j])),
                            total_dentro_IC90=bool(q[ok, j, 1].sum() <= y[ok, j].sum() <= q[ok, j, 7].sum())))
    t = pd.DataFrame(lin)
    if len(t) == 0:
        return t, t
    res = t.groupby('modelo').agg(erro_abs_log_total=('erro_log_total', lambda x: np.mean(np.abs(x))),
                                  vies_log_total=('erro_log_total', 'mean'),
                                  erro_abs_sem_pico=('erro_sem_pico', lambda x: np.mean(np.abs(x))),
                                  vies_sem_pico=('erro_sem_pico', 'mean')).reset_index()
    return t, res


def diagnostico(ctx, prev_uf, por_uf, temporadas):
    """por modelo: viés da mediana, largura dos intervalos, cobertura, piores UFs e uma leitura automática
    do que melhorar."""
    lin = []
    for m, g in prev_uf[prev_uf.season.isin(temporadas)].groupby('modelo'):
        vies, larg = [], []
        for s, gg in g.groupby('season'):
            datas = np.sort(gg.date.unique())
            q = V.de_df(gg, datas, ctx.Du.ufs)
            y = ctx.ver_uf.y(datas)
            ok = ~np.isnan(y)
            vies.append((np.log1p(q[..., 4]) - np.log1p(y))[ok])
            larg.append((np.log1p(q[..., 7]) - np.log1p(q[..., 1]))[ok])
        vies, larg = np.concatenate(vies), np.concatenate(larg)
        pu = por_uf[por_uf.modelo == m]
        if len(pu) == 0:
            continue
        piores = pu.groupby('uf_code').WIS_log.mean().sort_values(ascending=False).head(3).index.tolist()
        ic50, ic90 = pu.IC50.mean(), pu.IC90.mean()
        v = float(np.median(vies))
        dica = []
        if v > 0.25:
            dica.append('superestima (mediana alta): reduzir viés / rever tendência ou imunidade')
        if v < -0.25:
            dica.append('subestima (mediana baixa): falta sinal de surto / magnitude')
        if ic90 < 80:
            dica.append('intervalos estreitos demais: aumentar incerteza (dispersão, sementes, fatores comuns)')
        if ic90 > 97 and ic50 > 65:
            dica.append('intervalos largos demais: reduzir incerteza')
        if not dica:
            dica.append('equilibrado: ganhos devem vir de variáveis ou de ensemble')
        lin.append(dict(modelo=m, vies_log_mediana=v, largura_log_IC90=float(np.median(larg)), IC50=ic50, IC90=ic90,
                        piores_UFs=piores, leitura='; '.join(dica)))
    return pd.DataFrame(lin)


# =====================================================================================================================
# gráficos
# =====================================================================================================================
def _regiao_br(uf):
    return MACRO_BR[int(str(uf)[0])]


def grafico_leque(prev, ver, ids, col_id, modelos, s, titulo, arquivo=None, ncol=6):
    import matplotlib.pyplot as plt
    n = len(ids)
    nl = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nl, ncol, figsize=(3.1 * ncol, 2.3 * nl), squeeze=False)
    cores = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k, u in enumerate(ids):
        ax = axs.flat[k]
        datas = None
        for i, m in enumerate(modelos):
            g = prev[(prev.modelo == m) & (prev.season == s) & (prev[col_id] == u)].sort_values('date')
            if len(g) == 0:
                continue
            ax.fill_between(g.date, g.lower_90, g.upper_90, color=cores[i % 10], alpha=.12)
            ax.plot(g.date, g.pred, color=cores[i % 10], lw=1.1, label=m)
            datas = g.date.values
        if datas is not None:
            y = ver.y(datas)[:, list(ver.ufs).index(u)]
            ax.plot(datas, y, 'k.', ms=2.5)
        ax.set_title(str(u), fontsize=8)
        ax.tick_params(labelsize=5)
    for ax in list(axs.flat)[n:]:
        ax.axis('off')
    axs.flat[0].legend(fontsize=5)
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=110)
    return fig


def mapa_calor(por_uf, modelos, arquivo=None, ref='BB'):
    """média geométrica do WIS relativo (modelo ÷ ref) por UF, como o heatmap do 3º IMDC."""
    import matplotlib.pyplot as plt
    b = por_uf[por_uf.modelo == ref][['season', 'uf_code', 'WIS']].rename(columns={'WIS': 'Wr'})
    t = por_uf[por_uf.modelo.isin(modelos)].merge(b, on=['season', 'uf_code'])
    t['lr'] = np.log(t.WIS / t.Wr)
    H = np.exp(t.groupby(['uf_code', 'modelo']).lr.mean()).unstack()
    H = H[H.columns[np.argsort(-(H < 0.95).sum(0).values)]]
    H['macro'] = [_regiao_br(u) for u in H.index]
    H = H.sort_values(['macro']).drop(columns='macro')
    fig, ax = plt.subplots(figsize=(0.55 * H.shape[1] + 3, 0.32 * H.shape[0] + 2))
    im = ax.imshow(np.log(H.values), cmap='RdYlGn_r', vmin=-0.7, vmax=0.7, aspect='auto')
    ax.set_xticks(range(H.shape[1]))
    ax.set_xticklabels(H.columns, rotation=75, fontsize=7)
    ax.set_yticks(range(H.shape[0]))
    ax.set_yticklabels(H.index, fontsize=7)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5)
    ax.set_title(f'WIS ÷ WIS do {ref} (verde < 1 = melhor)')
    fig.colorbar(im, ax=ax, shrink=.6, label='log(R)')
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=120)
    return fig, H


def grafico_estabilidade(res, destaque, arquivo=None, ref='BB'):
    import matplotlib.pyplot as plt
    col, dp = f'R_vs_{ref}', f'dp_logR_vs_{ref}'
    r = res.dropna(subset=[col])
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, x in r.iterrows():
        d = x.modelo in destaque
        ax.scatter(np.log(x[col]), x[dp], c='tab:red' if d else 'grey', s=40 if d else 16, alpha=.8)
        if d or x[col] < 0.92:
            ax.annotate(str(x.modelo).replace('IMDC:', ''), (np.log(x[col]), x[dp]), fontsize=7)
    ax.axvline(0, ls=':', c='k')
    ax.axhline(r[dp].median(), ls=':', c='k')
    ax.set_xlabel(f'média de log(R) vs {ref} (negativo = melhor)')
    ax.set_ylabel('desvio-padrão de log(R) (menor = mais estável)')
    ax.set_title('Competição (vermelho) × 3º IMDC (cinza)')
    fig.tight_layout()
    if arquivo:
        fig.savefig(arquivo, dpi=120)
    return fig


def ranking_macro(por_uf, modelos, ref='BB'):
    b = por_uf[por_uf.modelo == ref][['season', 'uf_code', 'WIS']].rename(columns={'WIS': 'Wr'})
    t = por_uf[por_uf.modelo.isin(modelos)].merge(b, on=['season', 'uf_code'])
    t['lr'] = np.log(t.WIS / t.Wr)
    t['macro'] = [_regiao_br(u) for u in t.uf_code]
    g = np.exp(t.groupby(['macro', 'modelo']).lr.mean()).reset_index(name='R')
    g['posicao'] = g.groupby('macro').R.rank()
    return g.sort_values(['macro', 'posicao'])
