"""RODADA 5 · modelo FINAL de temporada: um especialista para cada fase da temporada.

Lições da rodada 4 (doc 18 §6, teste 2022–2025):
  - nenhum modelo sozinho é bom em tudo:
      · na ENTRESSAFRA o melhor é o normal ENS_grafos3_λ (WIS 23,6; viés +49%);
      · no PICO, o MIX_escala (WIS 2.136 contra 4.065; altura do pico −4% contra −42%);
  - o detector ampliado da R4 piorou (variável NACIONAL do Atlântico norte, recorde em 2023, sobreajustada) e levou
    junto os membros novos.

O que este modelo faz:
  ① DETECTOR CORRIGIDO: candidatos = detector da R3 + detector ampliado SEM variáveis nacionais + a média dos dois;
     escolha pelo pinball nas temporadas ≤ 2021 (anos grandes valendo 3×), nas MESMAS linhas para todos.
  ② MEMBROS refeitos com esse detector: MIX_escala5, MIX_elastico5, DoisEstagios_det5 e MTGNN_cond5 (a rede da R4,
     já treinada, só prevê de novo; nada é retreinado). Entram também o normal, o MTGNN_aug e os da R3.
  ③ ENSEMBLE POR FASE PREVISTA: cada semana × UF recebe, em junho, um rótulo de fase ESPERADA (entressafra,
     intermediária ou pico, pelo nível que um modelo de referência prevê para ela contra os limiares p25/p75 da UF) e
     de regime (P(ano grande) do detector). Para cada fase × regime, os pesos dos membros são escolhidos na afinação
     pelo critério misto; a versão "troca" usa só o melhor membro de cada fase. Os pesos são suavizados no tempo
     (5 semanas), para a curva não dar saltos quando muda a fase.
  ④ CALIBRAÇÃO POR FASE PREVISTA: largura e posição ajustadas só dentro de cada fase. Assim, subir o pico não infla a
     entressafra.
  ⑤ ESCOLHA HONESTA: todos os ensembles são validados na afinação deixando uma temporada de fora; o recomendado é o
     de menor critério misto fora da amostra, decidido antes de olhar o teste.
"""
import glob
import os

import numpy as np
import pandas as pd

import v4_lib as V
import comp_lib as L
import comp_modelos as CM
import uf_lib as U
import semanal_lib as S
import r2_lib as R2
import r3_lib as R3
import r4_lib as R4

Q, QCOLS = R4.Q, R4.QCOLS
TUNING, TESTE = R4.TUNING, R4.TESTE
CORTES = R4.CORTES
salvar_tabela, json_grava, json_le, carimbo, _grava_pq = R4.salvar_tabela, R4.json_grava, R4.json_le, R4.carimbo, R4._grava_pq
NIVEIS = ('entressafra', 'intermediaria', 'alta (pico)')


# =====================================================================================================================
# ① DETECTOR CORRIGIDO
# =====================================================================================================================
def _ler_r3(pasta_c3):
    ps = []
    for f in glob.glob(os.path.join(pasta_c3, 'detector_*_*.parquet')):
        nome = os.path.basename(f)
        if nome.startswith('detector_variaveis'):
            continue
        ps.append(pd.read_parquet(f))
    if not ps:
        return None
    P = pd.concat(ps, ignore_index=True)
    P['metodo'] = 'R3:' + P.metodo
    return P


def _nacional(X, col):
    """variável igual em todas as UFs dentro de cada temporada × corte (ENSO, mar, sorotipo nacional...). A versão da
    R4 agrupava só por temporada, misturando os cortes: nada parecia nacional e o filtro não funcionou."""
    g = X[['season', 'corte', col]].dropna()
    return len(g) > 0 and float(g.groupby(['season', 'corte'])[col].std().fillna(0).max()) < 1e-9


def detector_corrigido(ctx, pasta_c3, pasta_c4, pasta_c5, temporadas=range(2014, 2026)):
    """devolve (escolha por corte, todas as previsões, tabela de escolha). Grava em cache_r5 e retoma."""
    arq_todos = os.path.join(pasta_c5, 'detector5_todos.parquet')
    arq_esc = os.path.join(pasta_c5, 'detector5_escolhido.json')
    if os.path.exists(arq_todos) and os.path.exists(arq_esc):
        P = pd.read_parquet(arq_todos)
        return json_le(arq_esc), P, pd.read_csv(os.path.join(pasta_c5, 'detector5_escolha.csv'))
    X = pd.read_parquet(os.path.join(pasta_c4, 'detector4_variaveis.parquet'))
    # ampliado SEM variáveis nacionais (iguais em todas as UFs: ENSO, IOD, PDO, mar, sorotipo nacional)
    nac = {c for c in X.columns if X[c].dtype.kind in 'fi' and c not in R3.NAO_VARIAVEIS and _nacional(X, c)}
    print('variáveis nacionais excluídas:', sorted(nac))
    Xs = X.drop(columns=sorted(nac))
    arq_sin = os.path.join(pasta_c5, 'sinais5.json')
    sinais = {k: R4.selecionar_sinais4(Xs, k, arq=arq_sin) for k in CORTES}
    P5 = R4.prever_detector4(ctx, Xs, sinais, list(temporadas), pasta_c5)
    P5['metodo'] = 'R5:' + P5.metodo
    P3 = _ler_r3(pasta_c3)
    partes = [P5] + ([P3] if P3 is not None else [])
    # média dos dois detectores "media" (robustez)
    if P3 is not None:
        a = P3[P3.metodo == 'R3:media']
        b = P5[P5.metodo == 'R5:media']
        k = ['uf_code', 'season', 'corte']
        m = a.merge(b[k + QCOLS], on=k, suffixes=('', '_b'))
        if len(m):
            m[QCOLS] = np.sort((m[QCOLS].values + m[[c + '_b' for c in QCOLS]].values) / 2, 1)
            m = m.drop(columns=[c + '_b' for c in QCOLS])
            m['metodo'] = 'R3+R5:media'
            m['p_grande'] = R3._prob_acima(m[QCOLS].values, R4.LIM_GRANDE)
            partes.append(m)
    P = pd.concat(partes, ignore_index=True)
    # escolha nas MESMAS linhas para todos os candidatos (o da R3 não tem as temporadas mais antigas)
    k = ['uf_code', 'season', 'corte']
    lin = []
    for corte, g in P[P.season <= max(TUNING)].dropna(subset=['alvo']).groupby('corte'):
        comuns = None
        for _, gm in g.groupby('metodo'):
            s = set(map(tuple, gm[k].values))
            comuns = s if comuns is None else comuns & s
        gg = g[[tuple(r) in comuns for r in g[k].values]]
        lin.append(R4.pinball_escolha(gg))
    PE = pd.concat(lin, ignore_index=True)
    esc = R4.escolher_detector4(PE)
    P.to_parquet(arq_todos)
    json_grava(esc, arq_esc)
    PE.to_csv(os.path.join(pasta_c5, 'detector5_escolha.csv'), index=False)
    return esc, P, PE


def detector_escolhido5(pasta_c5):
    esc = json_le(os.path.join(pasta_c5, 'detector5_escolhido.json'))
    assert esc, 'Rode antes a etapa do detector.'
    P = pd.read_parquet(os.path.join(pasta_c5, 'detector5_todos.parquet'))
    return esc, pd.concat([P[(P.metodo == m) & (P.corte == c)] for c, m in esc.items()], ignore_index=True)


# =====================================================================================================================
# ② MEMBROS com o detector corrigido
# =====================================================================================================================
def prever_mtgnn_salva(ctx, s, DET, pasta_ckpt, sementes=(1, 2, 3, 4, 5)):
    """MTGNN_cond da R4 com as redes JÁ TREINADAS (cache_r4/ckpt) e os cenários do detector corrigido.
    Devolve None se faltar algum peso (não retreina aqui)."""
    arqs = [os.path.join(pasta_ckpt, f'MTGNN_cond_UF_s{s}_s{k}.pt') for k in sementes]
    if not all(os.path.exists(a) for a in arqs):
        return None
    rede = U.MTGNN(sementes=tuple(sementes))
    c, datas, idx, h = CM._horizontes(ctx, s)
    redes, dd = R4.treinar_cond(ctx, rede, c, 68, pasta_ckpt, f's{s}')
    datas, saida = R4.prever_temporada_cond(ctx, redes, dd, rede, s, DET)
    if 'MTGNN_cond_UF' not in saida:
        return None
    return V.para_df(V.arruma_quantis(saida['MTGNN_cond_UF']), datas, ctx.Du.unidades, 'MTGNN_cond5_UF', s)


# =====================================================================================================================
# ③ ENSEMBLE POR FASE PREVISTA × REGIME
# =====================================================================================================================
def nivel_previsto(X, ref, D, lim):
    """fase ESPERADA de cada semana × UF: o nível que o modelo `ref` prevê (mediana) contra os limiares p25/p75 da UF
    (calculados com dados até 2017). Conhecido em junho: não usa o que aconteceu."""
    r = X[X.modelo == ref][['season', 'uf_code', 'date', 'pred']]
    pop = pd.DataFrame(D.POP, index=pd.to_datetime(D.datas), columns=D.unidades).stack().rename('pop').reset_index() \
        .rename(columns={'level_0': 'date', 'level_1': 'uf_code'})
    r = r.merge(pop, on=['date', 'uf_code'], how='left').merge(lim, on='uf_code', how='left')
    r['nivel'] = R2._fase(r.pred / r['pop'] * 1e5, r.p25, r.p75)
    return r[['season', 'uf_code', 'date', 'nivel']]


def com_balde(X, DET, niv):
    X = R4.com_regime(X, DET)
    X = X.merge(niv, on=['season', 'uf_code', 'date'], how='left')
    X['nivel'] = X.nivel.fillna('intermediaria')
    X['balde'] = X.nivel + '|' + X.regime
    return X


def _otimiza(A, R_, yy, troca=False, passo=0.1, max_iter=60):
    wl0, wa0 = V.wis(np.log1p(R_), np.log1p(yy)).mean(), V.wis(R_, yy).sum()
    pa = R4.PESO_ABS

    def f(w):
        q = np.sort(np.tensordot(A, w, (1, 0)), 1)
        return (1 - pa) * V.wis(np.log1p(q), np.log1p(yy)).mean() / wl0 + pa * V.wis(q, yy).sum() / max(wa0, 1e-9)
    K = A.shape[1]
    if troca:                                            # só o melhor membro da fase
        v = [f(np.eye(K)[i]) for i in range(K)]
        return np.eye(K)[int(np.argmin(v))], float(min(v))
    w = np.full(K, 1 / K)
    atual = f(w)
    for _ in range(max_iter):
        melhorou = False
        for i in range(K):
            for j in range(K):
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
    return w, float(atual)


CHAVES = ['season', 'uf_code', 'date']


def _alinha(X, colunas):
    """(índice com balde/nível, arr (n, K, 9)) só das linhas em que todas as colunas existem."""
    w = X[X.modelo.isin(colunas)].pivot_table(index=CHAVES + ['balde', 'nivel'], columns='modelo', values=QCOLS).dropna()
    arr = np.stack([w[c].reindex(columns=colunas).values for c in QCOLS], -1)
    return w.index.to_frame(index=False), arr


def pesos_fase(T, membros, ref='climatologia', troca=False, min_linhas=150):
    """T (com y, balde, nivel): pesos por fase × regime; se o balde tiver poucas linhas, usa os da fase."""
    idx, arr = _alinha(T, membros + [ref])
    y = idx.merge(T.drop_duplicates(CHAVES)[CHAVES + ['y']], on=CHAVES, how='left').y.values.astype(float)
    out = {}
    for col in ('nivel', 'balde'):
        for b in idx[col].astype(str).unique():
            s = (idx[col].astype(str) == b).values
            if s.sum() < min_linhas:
                continue
            w, v = _otimiza(arr[s, :-1], arr[s, -1], y[s], troca)
            out[b] = dict(zip(membros, np.round(w, 3).tolist()), _criterio=round(v, 4), _linhas=int(s.sum()))
    return out


def aplicar_fase(X, membros, pesos, nome, suav=5):
    idx, arr = _alinha(X, membros)
    K = len(membros)
    W = np.full((len(idx), K), 1 / K)
    for i, (b, n) in enumerate(zip(idx.balde.astype(str), idx.nivel.astype(str))):
        p = pesos.get(b) or pesos.get(n)
        if p:
            W[i] = [p[m] for m in membros]
    idx = idx.reset_index(drop=True)
    ordem = idx.sort_values(CHAVES).index.values
    Wd = pd.DataFrame(W[ordem], index=ordem)
    g = idx.loc[ordem, ['season', 'uf_code']]
    Ws = Wd.groupby([g.season.values, g.uf_code.values]).transform(lambda c: c.rolling(suav, center=True, min_periods=1).mean())
    W = Ws.sort_index().values
    W = W / W.sum(1, keepdims=True)
    out = idx.copy()
    out[QCOLS] = np.sort(np.einsum('nkq,nk->nq', arr, W), 1)
    out['modelo'] = nome
    return out


def ajustar_fase(ctx, XB, membros, troca=False):
    T = S.com_verdade(XB, ctx.Du)
    pesos = pesos_fase(T, membros, troca=troca)
    E = S.com_verdade(aplicar_fase(XB, membros, pesos, 'x'), ctx.Du)
    cal = R2.calibrar(E, 'nivel', R4.GRADE_LAM4, R4.GRADE_DELTA4, min_linhas=150)
    return pesos, cal


def prever_fase(XB, membros, pesos, cal, nome):
    e = aplicar_fase(XB, membros, pesos, nome)
    return e, R2.aplicar_cal(e, 'nivel', cal)


def cv_fase(ctx, XB, membros, nome, troca=False, temporadas=TUNING):
    out = []
    for s in temporadas:
        tr = XB[XB.season.isin(temporadas) & (XB.season != s)]
        pesos, cal = ajustar_fase(ctx, tr, membros, troca)
        out += list(prever_fase(XB[XB.season == s], membros, pesos, cal, nome))
    return pd.concat(out, ignore_index=True)


# =====================================================================================================================
# saídas para a DASA
# =====================================================================================================================
def resumo_temporada(X, D, s):
    """por UF: total previsto (mediana e IC90 aproximado pela soma dos quantis), semana e altura do pico."""
    g = X[X.season == s]
    lin = []
    for u, gu in g.groupby('uf_code'):
        gu = gu.sort_values('date')
        i = int(np.argmax(gu.pred.values))
        lin.append(dict(uf=R2.SIGLAS.get(u, u), total_mediana=gu.pred.sum(), total_q05=gu.lower_90.sum(),
                        total_q95=gu.upper_90.sum(), semana_pico=pd.Timestamp(gu.date.values[i]).date(),
                        pico_mediana=gu.pred.values[i], pico_q95=gu.upper_90.values[i]))
    return pd.DataFrame(lin)
