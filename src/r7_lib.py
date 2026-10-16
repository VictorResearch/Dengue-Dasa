"""RODADA 7 · ensemble por nível previsto: um membro-base fora do pico e a média do elástico com o escala no pico.

Por que (R5 e R6, teste 2022–2025):
  - no pico, os dois erram em sentidos opostos: MIX_elastico5 −12% e MIX_escala5 +12% na altura. O escala5 tem o
    menor WIS do pico de todos (2.012), mas infla a entressafra (+151%);
  - fora do pico, os melhores são o normal (ENS_grafos3_λ, entressafra 23,6), o AJ_continuo (26,4) e o elastico5 (27,8);
  - otimizar cada modelo sozinho (R6) só trocou erro entre as fases.

O ensemble (poucos parâmetros, para não sobreajustar como o ENS_fase da R5):
    z  = nível previsto em junho da semana × UF, na escala dos limiares da UF: 0 = p25, 1 = p75 (log da incidência);
         o nível vem da mediana de um membro de referência (`nivel`);
    s  = 1 / (1 + exp(−(z − c) / h))                      peso do componente de PICO (0 na entressafra, 1 no pico);
    PICO = (1 − a)·elastico5 + a·escala5                    média ponderada dos quantis;
    ENS  = (1 − s)·BASE + s·PICO                            BASE = normal, AJ, elastico5 ou escala6.
Parâmetros: base, a, c, h e nivel. São escolhidos por GRADE (determinística), com o mesmo objetivo da R6:
    minimizar o WIS do PICO ÷ elastico5 (média geométrica entre UFs), sem a entressafra e a intermediária piorarem mais
    que TOL. O próprio elastico5 está na grade (base = elastico5, a = 0): o ensemble só muda se ganhar dele.
Validação: 2015–2021 com os membros gerados pela r6_lib (antes de 2018, o normal é a climatologia) e validação aninhada
(cada temporada fora). Teste 2022–2025 com as previsões gravadas, olhado uma vez.
"""
import itertools

import numpy as np
import pandas as pd

import v4_lib as V
import r3_lib as R3
import r6_lib as R6

Q, QCOLS = R6.Q, R6.QCOLS
FASES = R6.FASES
NOMES_ARQ = {'normal': 'ENS_grafos3_λ', 'AJ': 'AJ_continuo_UF', 'elastico5': 'MIX_elastico5_UF', 'escala5': 'MIX_escala5_UF',
             'escala6': 'MIX_escala6_UF', 'elastico6': 'MIX_elastico6_UF'}
GRADE = dict(base=('normal', 'AJ', 'elastico5', 'escala6'), a=(0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0),
             c=(-0.5, 0.0, 0.5, 1.0, 1.5, 2.0), h=(0.1, 0.3, 0.6, 1.0), nivel=('elastico5', 'escala5', 'normal'))
SIMPLES = dict(base='elastico5', a=0.5, c=1.0, h=0.3, nivel='elastico5')     # escolhido a priori, sem olhar dados
REFERENCIA = 'elastico5'


# =====================================================================================================================
# membros
# =====================================================================================================================
def par_aj(par_continuo):
    """AJ_continuo = escala sempre ligado, com o fator (alvo/normal)^p e sem teto (R3.mistura_uf 'continuo')."""
    return R6.completa(dict(R6.padrao('escala'), limiar=-10.0, tau=0.0, p=float(par_continuo), nivel_L=-0.5, teto_log2=10.0))


def membros_validacao(B, PAR0, PAR_AJ, MELHOR6):
    """{s: {membro: (W, N, 9)}} pela reimplementação da R6 (números aleatórios comuns a todos os membros)."""
    out = {}
    for s, d in B.dados.items():
        out[s] = {'normal': V.arruma_quantis(d['qn'].copy()),
                  'elastico5': B.prever(s, 'elastico', R6.completa(PAR0['elastico'])),
                  'escala5': B.prever(s, 'escala', R6.completa(PAR0['escala'])),
                  'AJ': B.prever(s, 'escala', PAR_AJ)}
        for k, (modelo, par) in {'escala6': ('escala', MELHOR6.get('escala')), 'elastico6': ('elastico', MELHOR6.get('elastico'))}.items():
            if par:
                out[s][k] = B.prever(s, modelo, par)
    return out


def verdade(ctx, s, datas, p25, p75):
    """y, fase observada (0 entressafra, 1 intermediária, 2 pico; −1 sem dado), população no corte e se está completa."""
    D = ctx.Du
    idx = np.array([D.idx_data[pd.Timestamp(x)] for x in datas])
    y = ctx.ver_uf.y(datas)
    inc = y / D.POP[idx] * 1e5
    fase = np.where(np.isnan(inc), -1, np.where(inc >= p75, 2, np.where(inc < p25, 0, 1)))
    c = R3.indice_corte(D, s, 'jun')
    return dict(datas=datas, y=y, fase=fase, pop=D.POP[c], completa=R3._completa_ate(D, s) is not None and not np.isnan(y).any())


def membros_arquivo(ctx, X, temporadas, p25, p75):
    """previsões GRAVADAS (formato longo) → ({s: {membro: (W, N, 9)}}, {s: verdade}). Datas = as do elastico5."""
    D = ctx.Du
    MA, VA = {}, {}
    for s in temporadas:
        g = X[(X.modelo == NOMES_ARQ[REFERENCIA]) & (X.season == s)]
        if g.empty:
            continue
        datas = np.sort(g.date.unique())
        m = {}
        for k, nome in NOMES_ARQ.items():
            x = X[(X.modelo == nome) & (X.season == s)]
            if len(x):
                q = V.de_df(x, datas, D.unidades)
                if np.isfinite(q).all():
                    m[k] = V.arruma_quantis(q)
        MA[s], VA[s] = m, verdade(ctx, s, datas, p25, p75)
    return MA, VA


# =====================================================================================================================
# ensemble
# =====================================================================================================================
def nivel_z(q, pop, p25, p75):
    """nível previsto (mediana) na escala dos limiares da UF: 0 = p25, 1 = p75 (em log1p da incidência)."""
    l = np.log1p(np.maximum(q[..., 4], 0) / pop * 1e5)
    a, b = np.log1p(p25), np.log1p(p75)
    return (l - a) / np.maximum(b - a, 1e-6)


def peso_pico(z, c, h):
    return 1 / (1 + np.exp(-np.clip((z - c) / h, -50, 50)))


def combinar(M, pop, p25, p75, par):
    """quantis (W, N, 9) do ensemble. Devolve None se faltar membro."""
    if any(k not in M for k in (par['base'], par['nivel'], 'elastico5', 'escala5')):
        return None
    s = peso_pico(nivel_z(M[par['nivel']], pop, p25, p75), par['c'], par['h'])[..., None]
    pico = (1 - par['a']) * M['elastico5'] + par['a'] * M['escala5']
    return np.sort((1 - s) * M[par['base']] + s * pico, -1)


def acumula(q, v):
    """somas do WIS por fase × UF e erros de altura e total do pico (temporadas completas), como R6.Base6.acumula."""
    N = q.shape[1]
    w = V.wis(q, v['y'])
    A, n = np.zeros((3, N)), np.zeros((3, N))
    for f in range(3):
        m = v['fase'] == f
        A[f] += np.where(m, w, 0).sum(0)
        n[f] += m.sum(0)
    alt, tot = [], []
    if v['completa']:
        y, med = v['y'], q[..., 4]
        for j in range(N):
            if y[:, j].max() > 0:
                alt.append(med[:, j].max() / y[:, j].max() - 1)
                tot.append(med[:, j].sum() / max(y[:, j].sum(), 1) - 1)
    return dict(wis=A, n=n, alt=np.array(alt), tot=np.array(tot))


def combos(grade=GRADE):
    ks = list(grade)
    out = [dict(zip(ks, v)) for v in itertools.product(*[grade[k] for k in ks])]
    # com a = 0 e base = elastico5 o ensemble é o próprio elastico5 (c, h, nivel não importam): deixa um só
    out = [p for p in out if not (p['base'] == 'elastico5' and p['a'] == 0) or (p['c'] == 1.0 and p['h'] == 0.3 and p['nivel'] == 'elastico5')]
    return out


def avaliar_grade(MV, VV, p25, p75, lista):
    """ACC[(i, s)] para cada combinação i e temporada s, e REF[s] (o elastico5)."""
    ACC, REF = {}, {}
    for s in MV:
        REF[s] = acumula(MV[s][REFERENCIA], VV[s])
        for i, par in enumerate(lista):
            q = combinar(MV[s], VV[s]['pop'], p25, p75, par)
            if q is not None:
                ACC[i, s] = acumula(q, VV[s])
    return ACC, REF


def tabela(lista, ACC, REF, temporadas, tol):
    lin = []
    for i, par in enumerate(lista):
        a = [ACC[i, s] for s in temporadas if (i, s) in ACC]
        if len(a) < len(temporadas):
            continue
        r = R6.razoes(R6.soma_acc(a), R6.soma_acc([REF[s] for s in temporadas]))
        lin.append(dict(i=i, **par, **r, viavel=bool(r['entressafra'] <= 1 + tol and r['intermediaria'] <= 1 + tol)))
    return pd.DataFrame(lin)


def escolher(lista, ACC, REF, temporadas, tol):
    """menor WIS do pico ÷ elastico5 entre as combinações viáveis (o próprio elastico5 é sempre viável)."""
    T = tabela(lista, ACC, REF, temporadas, tol)
    v = T[T.viavel].sort_values(['alta (pico)', 'geral'])
    return int(v.i.iloc[0]), T


def validacao_aninhada(lista, ACC, REF, temporadas, tol):
    lin, am, ar = [], [], []
    for h in temporadas:
        resto = [s for s in temporadas if s != h]
        i, _ = escolher(lista, ACC, REF, resto, tol)
        r = R6.razoes(ACC[i, h], REF[h])
        lin.append(dict(temporada_fora=h, **{f'escolhido_{k}': v for k, v in lista[i].items()}, **{f'fora_{k}': v for k, v in r.items()},
                        elastico5_erro_altura_pct=R6.razoes(REF[h], REF[h])['erro_altura_mediano_pct']))
        am.append(ACC[i, h])
        ar.append(REF[h])
    return pd.DataFrame(lin), R6.razoes(R6.soma_acc(am), R6.soma_acc(ar))


def complementaridade(MV, VV, temporadas):
    """no pico de cada UF × temporada completa: o real fica entre o elastico5 e o escala5? (altura pela mediana)."""
    lin = []
    for s in temporadas:
        v = VV[s]
        if not v['completa']:
            continue
        y = v['y']
        pe, pc = MV[s]['elastico5'][..., 4].max(0), MV[s]['escala5'][..., 4].max(0)
        for j in range(y.shape[1]):
            r = y[:, j].max()
            if r > 0:
                lin.append(dict(season=s, j=j, elastico5=pe[j] / r - 1, escala5=pc[j] / r - 1, media=(pe[j] + pc[j]) / 2 / r - 1))
    T = pd.DataFrame(lin)
    if T.empty:
        return T, {}
    entre = ((T.elastico5 <= 0) & (T.escala5 >= 0)) | ((T.elastico5 >= 0) & (T.escala5 <= 0))
    res = dict(n=len(T), real_entre_os_dois_pct=100 * entre.mean(), os_dois_abaixo_pct=100 * ((T.elastico5 < 0) & (T.escala5 < 0)).mean(),
               os_dois_acima_pct=100 * ((T.elastico5 > 0) & (T.escala5 > 0)).mean(),
               erro_altura_mediano_elastico5_pct=100 * T.elastico5.median(), erro_altura_mediano_escala5_pct=100 * T.escala5.median(),
               erro_altura_mediano_media_50_50_pct=100 * T.media.median(),
               erro_abs_mediano_elastico5_pct=100 * T.elastico5.abs().median(), erro_abs_mediano_escala5_pct=100 * T.escala5.abs().median(),
               erro_abs_mediano_media_50_50_pct=100 * T.media.abs().median())
    return T, res


def para_df(q, datas, ufs, nome, s):
    return V.para_df(V.arruma_quantis(q), datas, ufs, nome, s)
