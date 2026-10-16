"""
PRODUÇÃO · os 5 melhores modelos de cada nível, escolhidos por métricas absolutas no teste 2022–2025.

SEMANAL (previsão de 1, 2, 3 e 4 semanas a partir da última semana observada)
  ordem pelo MAE médio de h = 1…4 (casos/semana, teste 2022–2025):
    1. ENS_stack_sem                  428  ← CAMPEÃO  (média ponderada de 9 membros; pesos por horizonte × fase)
    2. ARIMA_Fourier_sem              455
    3. GBM_xgb_optuna_sem             464
    4. ENS_GBM_ARIMA_sem              465  (mediana de GBM_xgb_todas_sem e ARIMA_Fourier_sem)
    5. ENS_GBM_ARIMA_MTGNNcresc_sem   477  (mediana de GBM_xgb_todas_sem, ARIMA_Fourier_sem e MTGNN_cresc_sem)
  Os 9 membros do ENS_stack_sem são retreinados com os dados até a semana de origem, com os MESMOS hiperparâmetros
  e a MESMA seleção de variáveis da pesquisa (producao/parametros/); os pesos do stacking vêm da afinação 2018–2021.

TEMPORADA (SE41 → SE40, prevista com dados até a SE25, regra do 3º IMDC)
  ordem pelo WIS absoluto e pelo MAE (teste 2022–2025):
    1. ENS_pico7_simples_UF   WIS 721, MAE 1.277, altura do pico +2%, total −0,6%   ← CAMPEÃO
    2. MIX_escala5_UF         WIS 746, MAE 1.279 (infla a entressafra: +151%)
    3. ENS_pico7_UF           WIS 902, MAE 1.452 (melhor na régua oficial: 0,808 × BB)
    4. MIX_elastico5_UF       WIS 1.010, MAE 1.506
    5. AJ_continuo_UF         WIS 1.285, MAE 1.672
  Cadeia: normal (mediana de 4 redes de grafo, ENS_grafos3_λ) → detector de anos grandes (ridge) → misturas
  (elástica, escala, contínua) → ensembles do pico (rodada 7).

Tudo roda pela área de trabalho (scripts/04_montar_area_de_trabalho.py), no Colab ou localmente.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import v4_lib as V
import comp_lib as L
import uf_lib as U
import semanal_lib as S
import r2_lib as R2
import r3_lib as R3
import r4_lib as R4
import r5_lib as R5
import r7_lib as R7

Q, QCOLS = V.Q, V.QCOLS
TOP5_SEMANAL = ['ENS_stack_sem', 'ARIMA_Fourier_sem', 'GBM_xgb_optuna_sem', 'ENS_GBM_ARIMA_sem', 'ENS_GBM_ARIMA_MTGNNcresc_sem']
CAMPEAO_SEMANAL = 'ENS_stack_sem'
TOP5_TEMPORADA = ['ENS_pico7_simples_UF', 'MIX_escala5_UF', 'ENS_pico7_UF', 'MIX_elastico5_UF', 'AJ_continuo_UF']
CAMPEAO_TEMPORADA = 'ENS_pico7_simples_UF'

# parâmetros da temporada fixados na pesquisa (doc docs/05_modelos_campeoes.md)
LAMBDA_NORMAL = {'MTGNN_UF': 0.8, 'A3TGCN_UF': 0.8, 'GCNGRU_UF': 0.8}       # largura dos intervalos (CompUF4)
MEMBROS_NORMAL = ['MTGNN_UF+λ', 'A3TGCN_UF+λ', 'GCNGRU_UF+λ', 'MTGNN_optuna10_UF']
DETECTOR = dict(corte='jun', ate_temporada=2021, k=10, rho_min=0.12, min_treino=60, primeira_temporada=2011)
MISTURAS = {'MIX_elastico5_UF': dict(w=0.5),                 # cache_r5/mix_elastico_w.json
            'MIX_escala5_UF': dict(limiar=float(np.log(3))),  # cache_r3/mistura_parametros.json (escala)
            'AJ_continuo_UF': dict(p=0.25)}                   # cache_r3/mistura_parametros.json (continuo)
ENSEMBLES_PICO = {'ENS_pico7_simples_UF': R7.SIMPLES,
                  'ENS_pico7_UF': dict(base='AJ', a=0.6, c=2.0, h=0.6, nivel='normal')}   # avaliacao_r7/ensemble_r7_*.json


def ler_parametros(pasta):
    """os .json de producao/parametros (copiados para <área de trabalho>/parametros)."""
    out = {}
    for f in glob.glob(os.path.join(pasta, '*.json')):
        with open(f, encoding='utf-8') as fh:
            out[os.path.basename(f)[:-5]] = json.load(fh)
    falta = {'pesos_stack_semanal', 'optuna_gbm_semanal_melhor', 'optuna_mtgnn_melhor', 'tsfresh_selecao'} - set(out)
    assert not falta, f'faltam parâmetros em {pasta}: {falta}'
    return out


# =====================================================================================================================
# SEMANAL
# =====================================================================================================================
SEMANAS_INCOMPLETAS = 1     # a última semana dos dados chega incompleta (atraso de notificação: ~36% dos casos em 03/2026)


def rodada_producao(D, origem=None, H=S.H_SEM, descartar=SEMANAS_INCOMPLETAS):
    """uma 'rodada' com UMA origem: por padrão a última semana COMPLETA (a última observada menos `descartar`).
    O modelo treina com alvos até c = origem − 3 (como em todas as rodadas da pesquisa) e prevê origem + 1 … origem + 4;
    com descartar = 1, o h = 1 re-estima a semana incompleta e h = 2…4 são as semanas seguintes."""
    o = int(D.ultimo_obs - descartar if origem is None else origem)
    nome = f"p{pd.Timestamp(D.datas[o]):%Y%m%d}"
    return {nome: dict(c=o + 1 - H, t_ini=o + 1, t_fim=o + H, season=int(D.season[o + 1]), origens=np.array([o]))}


def membros_semanais(ctx, par, pasta_cache, so=None):
    """os 9 membros do ENS_stack_sem (que incluem os 3 dos outros ensembles e os 2 modelos isolados do top 5).
    so: lista de nomes para rodar só parte (ex. testes rápidos)."""
    D = ctx.Du
    lim = R2.limiares(D)
    nomes = list(next(iter(par['pesos_stack_semanal'].values())).keys())
    precisa = set(so or nomes)
    tsf_M = tsf_cols = None
    if precisa & {'GBM_xgb_tsf_sem'}:
        tsf_M, tsf_cols = R2.tsfresh_cache(D, os.path.join(pasta_cache, f'tsfresh_ate{pd.Timestamp(D.datas[D.ultimo_obs]):%Y%m%d}'),
                                           janela=52, parametros='efficient')
    kw = dict(lim=lim, tsf_M=tsf_M, tsf_cols=tsf_cols)
    bp = {k: v for k, v in par['optuna_gbm_semanal_melhor'].items() if k != 'valor'}
    n_opt = int(bp.pop('n_estimators'))
    ck, pasta_ch = os.path.join(pasta_cache, 'ckpt'), os.path.join(pasta_cache, 'chronos')
    os.makedirs(pasta_ch, exist_ok=True)
    fab = {
        'GBM_xgb_todas_sem': lambda: S.GBMSemanal('xgb', 'todas'),
        'ARIMA_Fourier_sem': lambda: S.ARIMASemanal(),
        'MTGNN_cresc_sem': lambda: S.RedeSemanal(U.MTGNN(), alvo='cresc'),
        'GBM_xgb_fase_sem': lambda: R2.GBMSemanalR2('GBM_xgb_fase_sem', 'xgb', fase=True, **kw),
        'GBM_xgb_tsf_sem': lambda: R2.GBMSemanalR2('GBM_xgb_tsf_sem', 'xgb', fase=True, tsf=par['tsfresh_selecao']['colunas'], **kw),
        'GBM_xgb_fase_L1_sem': lambda: R2.GBMSemanalR2('GBM_xgb_fase_L1_sem', 'xgb', fase=True, mediana_L1=True, **kw),
        'MTGNN_cresc_escala_sem': lambda: R2.RedeSemanalR2(dict(nome='MTGNN_cresc_escala_sem', fabrica=lambda: U.MTGNN(),
                                                                alvo='cresc', escala=True, cov=None, H=4), ck),
        'GBM_xgb_optuna_sem': lambda: R2.GBMSemanalR2('GBM_xgb_optuna_sem', 'xgb', fase=True, n_estimators=n_opt, prm=bp, lim=lim),
        'Chronos2FT_sem': lambda: R2.Chronos2FTSemanal(pasta_ch, num_steps=1000),
    }
    return [fab[n]() for n in nomes if n in precisa]


def prever_semanal(ctx, membros, pasta, rodadas=None, refazer=False):
    """roda os membros nas rodadas (padrão: a de produção) e devolve as previsões em formato longo + erros.
    Um arquivo por membro × rodada em <pasta> (retoma de onde parou)."""
    rods = rodadas or rodada_producao(ctx.Du)
    erros = []
    for m in membros:
        erros += S.rodar_semanal(ctx, m, rods, pasta, refazer=refazer)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    P = S.ler_semanal([pasta])
    P = P[P.rodada.isin(list(rods))].copy()
    P['date'], P['origem'] = pd.to_datetime(P['date']), pd.to_datetime(P['origem'])
    return P, erros


def _stack(PV, pesos, nome):
    """ENS_stack_sem; se algum membro faltar (ex.: o Chronos não rodou), os pesos são renormalizados entre os presentes."""
    presentes = [m for m in next(iter(pesos.values())) if m in set(PV.modelo)]
    if not presentes:
        return PV.iloc[:0]
    p2 = {b: {m: pw[m] / max(sum(pw[x] for x in presentes), 1e-9) for m in presentes} for b, pw in pesos.items()}
    ch = ['rodada', 'h', 'date', 'uf_code', 'balde']
    st = R2.aplicar_stacking(PV, presentes, ch, 'balde', p2, nome)
    info = PV.drop_duplicates(['rodada', 'h', 'date', 'uf_code'])[['rodada', 'h', 'date', 'uf_code', 'season', 'origem']]
    return st.merge(info, on=['rodada', 'h', 'date', 'uf_code']).drop(columns='balde')


def ensembles_semanais(P, D, par):
    """P (membros, formato longo) → P + ENS_stack_sem + ENS_GBM_ARIMA_sem + ENS_GBM_ARIMA_MTGNNcresc_sem."""
    lim = R2.limiares(D)
    PV = P.copy()
    PV['fase_origem'] = R2.fase_origem(PV, D, lim)
    PV['balde'] = PV.h.astype(str) + '|' + PV.fase_origem
    partes = [P,
              S.ensemble_semanal(P, ['GBM_xgb_todas_sem', 'ARIMA_Fourier_sem'], 'ENS_GBM_ARIMA_sem'),
              S.ensemble_semanal(P, ['GBM_xgb_todas_sem', 'ARIMA_Fourier_sem', 'MTGNN_cresc_sem'], 'ENS_GBM_ARIMA_MTGNNcresc_sem'),
              _stack(PV, par['pesos_stack_semanal'], 'ENS_stack_sem')]
    out = pd.concat([p for p in partes if len(p)], ignore_index=True)
    out['date'], out['origem'] = pd.to_datetime(out['date']), pd.to_datetime(out['origem'])
    return out


def tabela_semanal(P, D, modelo=CAMPEAO_SEMANAL):
    """tabela de leitura fácil: UF × semana-alvo com mediana e faixas de 50% e 90% (casos)."""
    g = P[P.modelo == modelo].copy()
    g['uf'] = g.uf_code.map(R2.SIGLAS)
    cols = ['uf', 'uf_code', 'origem', 'h', 'date', 'lower_90', 'lower_50', 'pred', 'upper_50', 'upper_90']
    return g[cols].sort_values(['uf', 'h']).round(0)


# =====================================================================================================================
# TEMPORADA
# =====================================================================================================================
def temporada_alvo(D):
    """a temporada mais recente que já pode ser prevista: a SE25 dela está nos dados observados."""
    s = int(pd.Timestamp(D.datas[D.ultimo_obs]).year)
    while R3.indice_corte(D, s, 'jun') is None or R3.indice_corte(D, s, 'jun') > D.ultimo_obs:
        s -= 1
    return s


def modelos_normal(par, pasta_ckpt):
    """as 4 redes de grafo do normal (ENS_grafos3_λ). Precisam de GPU (~20–40 min por temporada numa L4)."""
    p = par['optuna_mtgnn_melhor']
    fab10 = lambda: U.MTGNN(d=p['d'], epocas=p['epocas'], lr=p['lr'], L=p['L'], k_top=p['k_top'], beta=p['beta'],   # noqa: E731
                            sementes=tuple(range(1, 11)))
    return {'MTGNN_UF': U.MTGNN(), 'A3TGCN_UF': U.GCNGRU(atencao=True), 'GCNGRU_UF': U.GCNGRU(),
            'MTGNN_optuna10_UF': R2.RedeTemporadaR2(dict(nome='MTGNN_optuna10_UF', fabrica=fab10, alvo='ancora',
                                                         escala=False, peso_pop=p['peso_pop'], cov=None), pasta_ckpt)}


def rodar_normal(ctx, modelos, temporadas, pasta, M=500):
    for nome, m in modelos.items():
        for s in temporadas:
            dt = L.rodar(ctx, m, s, pasta, M=M)
            if dt is not None:
                print(f'✅ {nome:20s} {s}: {dt / 60:5.1f} min', flush=True)


def normal(pasta, temporadas, lam=LAMBDA_NORMAL):
    """ENS_grafos3_λ = mediana dos quantis das 4 redes (3 delas com intervalos encolhidos por λ)."""
    partes = []
    for s in temporadas:
        for nome in ('MTGNN_UF', 'A3TGCN_UF', 'GCNGRU_UF', 'MTGNN_optuna10_UF'):
            arq = L._arq(pasta, 'uf', nome, s)
            if not os.path.exists(arq):
                raise FileNotFoundError(f'{nome} {s}: rode antes rodar_normal ({arq})')
            x = pd.read_parquet(arq)
            if nome in lam:
                x = U.ajusta_largura(x, lam[nome]).assign(modelo=f'{nome}+λ')
            partes.append(x)
    X = pd.concat(partes, ignore_index=True)
    X['date'] = pd.to_datetime(X.date)
    e3, _ = V.ensemble(X, MEMBROS_NORMAL, 'ENS_grafos3_λ', metodo='mediana')
    e3 = e3[L.COLS_PREV].copy()
    e3['date'] = pd.to_datetime(e3.date)
    return e3


def normal_climatologia(D, temporadas):
    """substituto SEM GPU (para testes): a climatologia no lugar do normal. NÃO é o modelo campeão."""
    lin = []
    for s in temporadas:
        d, q = V.base_climatologia(D, s)
        lin.append(V.para_df(q, d, D.unidades, 'ENS_grafos3_λ', s))
    X = pd.concat(lin, ignore_index=True)
    X['date'] = pd.to_datetime(X.date)
    return X


def detector(ctx, temporadas, pasta_cache, cfg=DETECTOR):
    """detector de anos grandes da R3/R5 (ridge, corte de junho): log do tamanho da temporada relativo ao normal da UF.
    Sinais escolhidos SÓ com temporadas ≤ 2021; o treino de cada temporada usa só temporadas já terminadas no corte."""
    D = ctx.Du
    corte = cfg['corte']
    ult = max(temporadas)
    pasta = os.path.join(pasta_cache, f'detector_ate{ult}')
    os.makedirs(pasta, exist_ok=True)
    X = R3.tabela_detector(ctx, list(range(cfg['primeira_temporada'], ult + 1)), pasta)
    sinais = R3.selecionar_sinais(X, corte, ate_temporada=cfg['ate_temporada'], k=cfg['k'], rho_min=cfg['rho_min'],
                                  arq=os.path.join(pasta, 'sinais.json'))
    fim = {s: R3._completa_ate(D, s) for s in sorted(X.season.unique())}
    g = X[X.corte == corte]
    lin = []
    for s in temporadas:
        c = R3.indice_corte(D, s, corte)
        te = g[g.season == s]
        tr = g[g.season.map(lambda x: (fim.get(x) or 10 ** 9) <= c) & (g.season < s)].dropna(subset=['alvo'] + sinais)
        if c is None or te.empty or len(tr) < cfg['min_treino']:
            print(f'detector {s}: sem dados suficientes')
            continue
        Xte = te[sinais].fillna(tr[sinais].median()).values
        q = np.sort(R3._quantis_detector('ridge', tr[sinais].values, tr.alvo.values, Xte), -1)
        o = te[['uf_code', 'season', 'corte', 'base', 'alvo', 'pop']].copy()
        o[QCOLS] = q
        o['metodo'], o['n_treino'] = 'R3:ridge', len(tr)
        lin.append(o)
    DET = pd.concat(lin, ignore_index=True)
    DET['p_grande'] = R3._prob_acima(DET[QCOLS].values, R3.LIM_GRANDE)
    return DET, sinais


def misturas(ctx, NORMAL, DET, temporadas, M=1000):
    """MIX_elastico5, MIX_escala5 e AJ_continuo (normal reescalado pelo tamanho previsto)."""
    out = [R4.gerar_mix_elastico(ctx, NORMAL, DET, MISTURAS['MIX_elastico5_UF']['w'], 'jun', temporadas, M=M, nome='MIX_elastico5_UF'),
           R3.gerar_mistura(ctx, NORMAL, DET, 'escala', MISTURAS['MIX_escala5_UF']['limiar'], 'jun', temporadas, M=M).assign(modelo='MIX_escala5_UF'),
           R3.gerar_mistura(ctx, NORMAL, DET, 'continuo', MISTURAS['AJ_continuo_UF']['p'], 'jun', temporadas, M=M).assign(modelo='AJ_continuo_UF')]
    X = pd.concat([x for x in out if x is not None], ignore_index=True)
    X['date'] = pd.to_datetime(X.date)
    return X


def ensembles_pico(D, NORMAL, MIX, temporadas):
    """ENS_pico7_simples_UF (campeão) e ENS_pico7_UF (rodada 7)."""
    lim = R2.limiares(D).set_index('uf_code').reindex(D.unidades)
    p25, p75 = lim.p25.values, lim.p75.values
    fontes = {'normal': NORMAL, 'AJ': MIX[MIX.modelo == 'AJ_continuo_UF'], 'elastico5': MIX[MIX.modelo == 'MIX_elastico5_UF'],
              'escala5': MIX[MIX.modelo == 'MIX_escala5_UF']}
    lin = []
    for s in temporadas:
        ref = fontes['elastico5'][fontes['elastico5'].season == s]
        if ref.empty:
            continue
        datas = np.sort(ref.date.unique())
        Mx = {k: V.arruma_quantis(V.de_df(f[f.season == s], datas, D.unidades)) for k, f in fontes.items()}
        pop = D.POP[R3.indice_corte(D, s, 'jun')]
        for nome, par in ENSEMBLES_PICO.items():
            q = R7.combinar(Mx, pop, p25, p75, par)
            if q is not None:
                lin.append(R7.para_df(q, datas, D.unidades, nome, s))
    X = pd.concat(lin, ignore_index=True)
    X['date'] = pd.to_datetime(X.date)
    return X


def prever_temporada(ctx, NORMAL, temporadas, pasta_cache, M=1000):
    """normal pronto → detector → misturas → ensembles do pico. Devolve (top 5 + normal em formato longo, detector)."""
    DET, sinais = detector(ctx, temporadas, pasta_cache)
    print('detector: sinais =', sinais)
    MIX = misturas(ctx, NORMAL, DET, temporadas, M=M)
    ENS = ensembles_pico(ctx.Du, NORMAL, MIX, temporadas)
    X = pd.concat([NORMAL, MIX, ENS], ignore_index=True)
    return X[L.COLS_PREV], DET


def resumo_temporada(X, D, s, modelo=CAMPEAO_TEMPORADA):
    """por UF: total previsto, semana e altura do pico (mediana) e faixas."""
    return R5.resumo_temporada(X[X.modelo == modelo], D, s)


def gravar(df, pasta, nome):
    os.makedirs(pasta, exist_ok=True)
    arq = os.path.join(pasta, nome)
    if arq.endswith('.parquet'):
        df.to_parquet(arq + '.tmp', index=False)
    else:
        df.to_csv(arq + '.tmp', index=False)
    os.replace(arq + '.tmp', arq)
    return arq
