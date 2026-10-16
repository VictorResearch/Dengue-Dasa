"""Gera os notebooks da RODADA 2 (rode: python _gerar_notebooks_r2.py). Ordem de uso:

  R2_1_Avaliacao.ipynb     (CPU)  A métricas de pico/entressafra · D2 calibração por horizonte e fase · E4 pesos de
                                  ensemble por horizonte e fase · tabelas novas com carimbo de data (as antigas ficam).
                                  Rode agora (avalia o que já existe) e de novo depois dos notebooks 2–5.
  R2_2_Semanal_CPU.ipynb   (CPU)  B3 variáveis de fase · C tsfresh (seleção só na afinação) · D1 peso nas semanas de
                                  pico · mediana L1 — GBMs semanais
  R2_3_Redes_GPU.ipynb     (GPU)  B1 escala por UF · B2 pré-treino nas 435 regiões + ajuste · D1 quantis altos ·
                                  D3 cabeça do total — MTGNN de temporada e semanal
  R2_4_Optuna.ipynb        (GPU)  E1 Optuna no GBM semanal e no MTGNN de temporada · E2 10 sementes
  R2_5_Chronos_GPU.ipynb   (GPU)  E3 Chronos-2 ajustado aos dados de dengue (semanal e temporada)
  R2_6a_Variaveis_GPU.ipynb (GPU) variáveis de fase/Rt e tsfresh nas redes (semanal e temporada) e no GBM do Optuna
  R2_6b_Variaveis_CPU.ipynb (CPU) ARIMA com regressores de fase (ARIMAX_fase_sem)
  R2_7_Graficos_UF.ipynb   (CPU)  gráficos das 26 UFs + Brasil (semanal e temporada) com os campeões de cada área
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

ANTES = r"""
**Antes de rodar (uma vez só)**, em `MyDrive/dengue_V4/` (que já tem tudo das rodadas anteriores, inclusive `semanal_lib.py`):
1. suba **`r2_lib.py`** (pasta `data_imdc_2026/RODADA2/`);
2. **crie à mão** as pastas `resultados_semanal_r2`, `resultados_uf_r2`, `cache_r2` e `avaliacao_r2` (evita pastas duplicadas quando há duas sessões abertas).

**Nada antigo é alterado.** Os modelos novos gravam em `resultados_semanal_r2/` e `resultados_uf_r2/`; as tabelas novas vão para `avaliacao_r2/` com a data e a hora no nome (`..._AAAAMMDD-HHMM.csv`) e uma cópia `..._ultimo.csv`. As tabelas da rodada anterior (`resultados_semanal_avaliacao/`) continuam como estão.

**Retomada:** se a internet ou a energia cair, reconecte e **rode tudo de novo**. Cada modelo × rodada/temporada vira um arquivo assim que termina (gravado como `.tmp` e renomeado); os treinos longos das redes gravam um checkpoint a cada 5 épocas e continuam da última época salva; o tsfresh grava uma UF por vez; o Optuna grava o estudo no Drive a cada tentativa; o Chronos ajustado fica salvo e não é refeito.
"""

SETUP = r"""
# @title 1 · Drive, arquivos e pacotes
import os, sys, shutil, time, glob, subprocess, json, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/r2', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'uf_lib.py', 'semanal_lib.py', 'r2_lib.py', 'dados', 'dados_uf',
                 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou subir {item} para {PASTA_DRIVE}'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
    if PACOTES:
        r = subprocess.run(f'pip install -q {PACOTES}', shell=True, capture_output=True, text=True)
        print(r.stdout[-400:], r.stderr[-400:])
sys.path.insert(0, PASTA_LOCAL)
PASTA_SEM_R2 = os.path.join(PASTA_DRIVE, 'resultados_semanal_r2')
PASTA_UF_R2 = os.path.join(PASTA_DRIVE, 'resultados_uf_r2')
PASTA_CACHE = os.path.join(PASTA_DRIVE, 'cache_r2')
PASTA_AV2 = os.path.join(PASTA_DRIVE, 'avaliacao_r2')
PASTA_COMP = os.path.join(PASTA_DRIVE, 'resultados_competicao')
PASTA_EXT = os.path.join(PASTA_DRIVE, 'resultados_externos')
ARQ_SEL = os.path.join(PASTA_DRIVE, 'resultados_semanal', 'selecao_grupos.json')
for p in (PASTA_SEM_R2, PASTA_UF_R2, PASTA_CACHE, PASTA_AV2, os.path.join(PASTA_CACHE, 'ckpt')):
    os.makedirs(p, exist_ok=True)
import importlib, v4_lib as V, comp_lib as L, comp_modelos as CM, uf_lib as U, semanal_lib as S, r2_lib as R2
for m in (V, L, CM, U, S, R2):
    importlib.reload(m)
try:
    import torch
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma', '· CPUs:', os.cpu_count())
except ImportError:
    torch = None
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'),
                   os.path.join(PASTA_LOCAL, 'dados_uf'))
D = ctx.Du
TUNING, TESTE = U.TUNING, U.TESTE
RODADAS = S.rodadas(D, felipe=True)
LIM = R2.limiares(D)                        # p25/p75 de cada UF com dados até 2017 (antes da afinação)
REFAZER = False
"""

LACO_SEM = r"""
# @title Rodar (retoma de onde parou; um erro não interrompe o resto)
erros = []
for m in MODELOS:
    erros += S.rodar_semanal(ctx, m, RODADAS, PASTA_SEM_R2, refazer=REFAZER)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
print('Erros:', len(erros))
pd.DataFrame(erros, columns=['modelo', 'rodada', 'erro']).to_csv(os.path.join(PASTA_SEM_R2, f'erros_{NOTEBOOK}.csv'), index=False)
feitos = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(PASTA_SEM_R2, '*.parquet'))}
pd.DataFrame({m.nome: {r: '●' if f'{m.nome}__{r}' in feitos else '·' for r in RODADAS} for m in MODELOS}).T
"""

LACO_TEMP = r"""
# @title Rodar temporada (L.rodar: pula o que já existe; as redes retomam do último checkpoint)
erros = []
for nome, m in MODELOS_T.items():
    for s in TEMPORADAS:
        try:
            dt = L.rodar(ctx, m, s, PASTA_UF_R2, M=500, refazer=REFAZER)
            if dt is not None:
                print(f'✅ {nome:30s} {s}: {dt / 60:5.1f} min', flush=True)
        except Exception as e:
            erros.append((nome, s, repr(e)[:500]))
            print(f'❌ {nome:30s} {s}: {repr(e)[:300]}', flush=True)
            traceback.print_exc(limit=2)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
print('Erros:', len(erros))
pd.DataFrame(erros, columns=['modelo', 'temporada', 'erro']).to_csv(os.path.join(PASTA_UF_R2, f'erros_{NOTEBOOK}.csv'), index=False)
"""

# ------------------------------------------------------------------------------------------------- R2_1 avaliação
NB1 = [
    ('markdown', "# R2 · 1 · Avaliação complementar: pico e entressafra (A), calibração por horizonte e fase (D2), pesos de ensemble (E4)\n" + ANTES +
     "\nCPU, 15–40 min. **Rode agora** (avalia os modelos que já existem) e **de novo no fim**, depois dos notebooks 2–5, para incluir os modelos novos. Cada execução grava um novo conjunto de tabelas com carimbo de data; as anteriores ficam.\n\n"
     "**Fases** (por UF, limiares calculados só com dados até 2017): *entressafra* = semana abaixo do percentil 25 da incidência; *alta (pico)* = acima do percentil 75; *intermediária* = entre os dois. "
     "**Pico:** erro na semana do pico (semanas) e na altura do pico (%), por UF × temporada. **Cobertura por quantil:** % dos casos abaixo de cada quantil (o ideal é o nível do quantil)."),
    ('code', "NOTEBOOK = 'r2_avaliacao'\nPACOTES = ''\n" + SETUP),
    ('markdown', "## A · Semanal"),
    ('code', r"""
# @title A1 · Ler todas as previsões semanais (rodada anterior + rodada 2) e os ensembles de referência
PASTAS_SEM = sorted(p for p in glob.glob(os.path.join(PASTA_DRIVE, 'resultados_semanal*'))
                    if os.path.isdir(p) and not p.endswith('_avaliacao'))
print('Pastas:', [os.path.basename(p) for p in PASTAS_SEM])
P0 = S.ler_semanal(PASTAS_SEM)
P0['date'], P0['origem'] = pd.to_datetime(P0['date']), pd.to_datetime(P0['origem'])
def ens_sem(membros, nome):
    global P0
    if all(m in set(P0.modelo) for m in membros):
        P0 = pd.concat([P0, S.ensemble_semanal(P0, membros, nome)], ignore_index=True)
    else:
        print(f'{nome}: faltam {[m for m in membros if m not in set(P0.modelo)]}')
ens_sem(['GBM_xgb_todas_sem', 'ARIMA_Fourier_sem'], 'ENS_GBM_ARIMA_sem')                         # o vencedor anterior
ens_sem(['GBM_xgb_todas_sem', 'ARIMA_Fourier_sem', 'MTGNN_cresc_sem'], 'ENS_GBM_ARIMA_MTGNNcresc_sem')  # corrige a falha
PV = S.com_verdade(P0, D)
PV['fase_origem'] = R2.fase_origem(PV, D, LIM)
PV['balde'] = PV.h.astype(str) + '|' + PV.fase_origem
print(len(PV), 'linhas ·', PV.modelo.nunique(), 'modelos')
"""),
    ('code', r"""
# @title A2 · D2: calibração por horizonte e fase na origem (λ = largura, δ = deslocamento da mediana), escolhida na afinação
ARQ_CAL = os.path.join(PASTA_AV2, 'calibracao_semanal.json')
CAL_SEM = {} if REFAZER else R2.json_le(ARQ_CAL, {})
for m in sorted(PV.modelo.unique()):
    if m in CAL_SEM or m in S.REFS_SEM:
        continue
    T = PV[(PV.modelo == m) & (PV.rodada.map(S.grupo_rodada) == 'afinacao')]
    if T.rodada.nunique() < len(TUNING):
        continue
    CAL_SEM[m] = R2.calibrar(T, 'balde')
    R2.json_grava(CAL_SEM, ARQ_CAL)                                   # grava a cada modelo (retoma)
if os.path.exists(ARQ_CAL):
    shutil.copy(ARQ_CAL, ARQ_CAL.replace('.json', f'_{R2.carimbo()}.json'))
PCAL = pd.concat([R2.aplicar_cal(PV[PV.modelo == m], 'balde', t) for m, t in CAL_SEM.items() if (PV.modelo == m).any()],
                 ignore_index=True)
ex = 'ENS_GBM_ARIMA_sem'
if ex in CAL_SEM:
    display(pd.DataFrame(CAL_SEM[ex]).T.round(3))
"""),
    ('code', r"""
# @title A3 · E4: pesos de ensemble por horizonte e fase (afinação) → ENS_stack_sem
CAND_SEM = ['GBM_xgb_todas_sem', 'ARIMA_Fourier_sem', 'MTGNN_cresc_sem', 'GBM_xgb_fase_sem', 'GBM_xgb_tsf_sem',
            'GBM_xgb_fase_L1_sem', 'MTGNN_cresc_escala_sem', 'GBM_xgb_optuna_sem', 'Chronos2FT_sem',
            'MTGNN_cresc_fase_sem', 'GBM_xgb_optuna_tsf_sem', 'ARIMAX_fase_sem']
MEMB_SEM = [m for m in CAND_SEM if m in set(PV.modelo)]
print('Membros:', MEMB_SEM)
CH = ['rodada', 'h', 'date', 'uf_code', 'balde']
T = PV[PV.rodada.map(S.grupo_rodada) == 'afinacao']
PESOS_SEM = R2.pesos_stacking(T, MEMB_SEM, CH, 'balde')
R2.json_grava(PESOS_SEM, os.path.join(PASTA_AV2, f'pesos_stack_semanal_{R2.carimbo()}.json'))
ST = R2.aplicar_stacking(PV, MEMB_SEM, CH, 'balde', PESOS_SEM, 'ENS_stack_sem')
info = PV.drop_duplicates(['rodada', 'h', 'date', 'uf_code'])[['rodada', 'h', 'date', 'uf_code', 'y', 'season', 'origem', 'fase_origem']]
ST = ST.merge(info, on=['rodada', 'h', 'date', 'uf_code'])
PA = R2.com_fase(pd.concat([PV, PCAL, ST], ignore_index=True), D, LIM)
pd.DataFrame(PESOS_SEM).T.round(2)
"""),
    ('code', r"""
# @title A4 · Tabelas semanais (gerais, por fase, pico, cobertura) com carimbo
TAB = {}
for g in ('afinacao', 'teste', 'felipe'):
    TAB[g] = S.resumo_semanal(PA, g)
    R2.salvar_tabela(TAB[g], PASTA_AV2, f'semanal_{g}')
TT = PA[PA.rodada.map(S.grupo_rodada) == 'teste']
FASE_SEM = R2.metricas_fase(TT[TT.h.isin([1, 4])], ['modelo', 'h'])
PICO_SEM = R2.erros_pico(TT[TT.h == 1], 'rodada')
COB_SEM = R2.cobertura_quantis(TT[TT.h == 1])
for nome, t in (('semanal_teste_por_fase', FASE_SEM), ('semanal_teste_pico_h1', PICO_SEM), ('semanal_teste_cobertura_h1', COB_SEM)):
    R2.salvar_tabela(t, PASTA_AV2, nome)
# visão resumida: régua geral + desempenho no pico e na entressafra (h = 1)
f1 = FASE_SEM[FASE_SEM.h == 1].pivot_table(index='modelo', columns='fase', values=['WIS_log', 'vies_pct', 'IC90'])
f1.columns = [f'{a}_{b}' for a, b in f1.columns]
RES_SEM = TAB['teste'].query('h == 1')[['modelo', 'R_vs_Climatologia_sem', 'MAE', 'WAPE', 'IC90']].merge(
    f1.reset_index(), on='modelo', how='left').merge(PICO_SEM, on='modelo', how='left').sort_values('R_vs_Climatologia_sem')
R2.salvar_tabela(RES_SEM, PASTA_AV2, 'semanal_resumo_fases_h1')
RES_SEM.head(25).round(3)
"""),
    ('markdown', "## B · Temporada"),
    ('code', r"""
# @title B1 · Ler todos os modelos de temporada (rodadas anteriores + resultados_uf_prox + resultados_uf_r2), λ dos novos
PASTAS_UF = [p for p in sorted(glob.glob(os.path.join(PASTA_DRIVE, 'resultados_uf*'))) if os.path.isdir(p)]
FONTES = [('região→UF', PASTA_COMP)] + [(os.path.basename(p), p) for p in PASTAS_UF] + [('externo', PASTA_EXT)]
PU = L.carregar_previsoes(FONTES, ctx.Du.ufs, 'uf', mostrar=False)
NOVOS_PROX = sorted(PU[PU.fonte.str.contains('prox')].modelo.unique())
NOVOS_R2 = sorted(PU[PU.fonte.str.contains('_r2')].modelo.unique())
NOVOS = NOVOS_PROX + NOVOS_R2
PU = PU[~PU.modelo.str.startswith('ENS')][L.COLS_PREV]
ens = [os.path.join(p, 'ensembles_uf.parquet') for p in PASTAS_UF if os.path.exists(os.path.join(p, 'ensembles_uf.parquet'))]
ENS0 = pd.read_parquet(ens[0])[L.COLS_PREV] if ens else PU.iloc[:0]
REF = ctx.ref[L.COLS_PREV] if ctx.ref is not None else PU.iloc[:0]
arq_f = os.path.join(PASTA_DRIVE, 'COMPARACAO_FELIPE', 'felipe_no_meu_jogo_previsoes.parquet')
FEL = pd.read_parquet(arq_f) if os.path.exists(arq_f) else PU.iloc[:0]
FEL = FEL[FEL.modelo.isin(['Felipe_XGB_ponto', 'Felipe_XGB_quantis'])][L.COLS_PREV] if len(FEL) else FEL
print('Novos (prox + r2):', NOVOS)
ARQ_LAM = os.path.join(PASTA_AV2, 'lambda_temporada.json')
LAM_T = {} if REFAZER else R2.json_le(ARQ_LAM, {})
clim = PU[(PU.modelo == 'climatologia') & PU.season.isin(TUNING)]
for m in NOVOS:
    if m in LAM_T:
        continue
    g = PU[(PU.modelo == m) & PU.season.isin(TUNING)]
    if g.season.nunique() < len(TUNING):
        print('sem afinação completa:', m); continue
    rs = []
    for l in (0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5):
        _, rr = V.avaliar(pd.concat([U.ajusta_largura(g, l).assign(modelo='x'), clim]), ctx.ver_uf, ref=('climatologia',))
        rs.append(float(rr.set_index('modelo').loc['x', 'R_vs_climatologia']))
    LAM_T[m] = (0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5)[int(np.argmin(rs))]
    R2.json_grava(LAM_T, ARQ_LAM)
CAL2 = [U.ajusta_largura(PU[PU.modelo == m], l).assign(modelo=f'{m}+λ') for m, l in LAM_T.items() if l != 1.0 and (PU.modelo == m).any()]
TODOS = pd.concat([REF, PU, ENS0, FEL] + CAL2, ignore_index=True).drop_duplicates(['modelo', 'season', 'uf_code', 'date'], keep='last')
print('λ:', LAM_T)
"""),
    ('code', r"""
# @title B2 · Ensembles de temporada: grafos com λ, + melhor MTGNN da rodada anterior, + melhor da rodada 2
def ens(membros, nome):
    global TODOS
    falta = [m for m in membros if m not in set(TODOS.modelo)]
    if falta:
        print(f'{nome}: faltam {falta}'); return
    e, _ = V.ensemble(TODOS, membros, nome, metodo='mediana')
    TODOS = pd.concat([TODOS, e], ignore_index=True)
BASE_G = ['MTGNN_UF+λ', 'A3TGCN_UF+λ', 'GCNGRU_UF+λ']
ens(BASE_G, 'ENS_grafos_λ')
def melhor_afinacao(cands):
    cands = [m for m in cands if m in set(TODOS.modelo)]
    if not cands:
        return None
    _, ra = V.avaliar(TODOS[TODOS.season.isin(TUNING) & TODOS.modelo.isin(cands + ['climatologia'])][L.COLS_PREV], ctx.ver_uf, ref=('climatologia',))
    ra = ra[(ra.temporadas == len(TUNING)) & ra.modelo.isin(cands)].dropna(subset=['R_vs_climatologia'])
    return ra.sort_values('R_vs_climatologia').modelo.iloc[0] if len(ra) else None
b_prox = melhor_afinacao([m + s for m in NOVOS_PROX for s in ('', '+λ')])
if b_prox:
    ens(BASE_G + [b_prox], 'ENS_grafos2_λ')
b_todos = melhor_afinacao([m + s for m in NOVOS for s in ('', '+λ')])
if b_todos:
    ens(BASE_G + [b_todos], 'ENS_grafos3_λ')
print('melhor novo (rodada anterior):', b_prox, '· melhor novo (todos):', b_todos)
"""),
    ('code', r"""
# @title B3 · D2 e E4 na temporada: calibração por bloco de antecedência e pesos por bloco (afinação)
c_data = {s: pd.Timestamp(D.datas[D.corte(s)]) for s in TUNING + TESTE}
TV = S.com_verdade(TODOS[TODOS.season.isin(TUNING + TESTE)], D)
TV['h'] = ((TV.date - TV.season.map(c_data)).dt.days // 7).astype(int)
TV['balde'] = R2.bloco_horizonte(TV.h.values)
_, ra = V.avaliar(TODOS[TODOS.season.isin(TUNING)][L.COLS_PREV], ctx.ver_uf, ref=('climatologia',))
TOP_AF = ra[ra.temporadas == len(TUNING)].dropna(subset=['R_vs_climatologia']).sort_values('R_vs_climatologia').modelo.head(30).tolist()
ARQ_CALT = os.path.join(PASTA_AV2, 'calibracao_temporada.json')
CAL_T = {} if REFAZER else R2.json_le(ARQ_CALT, {})
for m in TOP_AF:
    if m in CAL_T:
        continue
    CAL_T[m] = R2.calibrar(TV[(TV.modelo == m) & TV.season.isin(TUNING)], 'balde', min_linhas=50)
    R2.json_grava(CAL_T, ARQ_CALT)
TCAL = pd.concat([R2.aplicar_cal(TV[TV.modelo == m], 'balde', t) for m, t in CAL_T.items() if (TV.modelo == m).any()], ignore_index=True)
CAND_T = ['MTGNN_pesopop_UF+λ', 'MTGNN_UF+λ', 'A3TGCN_UF+λ', 'GCNGRU_UF+λ', 'TFT_pool_UF', 'DoisEstagios_pool_UF'] + ([b_todos] if b_todos else [])
MEMB_T = list(dict.fromkeys(m for m in CAND_T if m in set(TV.modelo)))
CHT = ['season', 'date', 'uf_code', 'balde']
PESOS_T = R2.pesos_stacking(TV[TV.season.isin(TUNING)], MEMB_T, CHT, 'balde')
R2.json_grava(PESOS_T, os.path.join(PASTA_AV2, f'pesos_stack_temporada_{R2.carimbo()}.json'))
STT = R2.aplicar_stacking(TV, MEMB_T, CHT, 'balde', PESOS_T, 'ENS_stack_UF')
STT = STT.merge(TV.drop_duplicates(['season', 'date', 'uf_code'])[['season', 'date', 'uf_code', 'y', 'h']], on=['season', 'date', 'uf_code'])
TV2 = pd.concat([TV, TCAL, STT], ignore_index=True)
print('membros do stacking:', MEMB_T)
pd.DataFrame(PESOS_T).T.round(2)
"""),
    ('code', r"""
# @title B4 · Tabelas da temporada (régua oficial, fases, pico, cobertura) com carimbo
TT2 = TV2[TV2.season.isin(TESTE)]
_, RES_T = V.avaliar(TT2[L.COLS_PREV], ctx.ver_uf, ref=('BB', 'climatologia'))
RES_T = RES_T[RES_T.temporadas == len(TESTE)]
pont = pd.DataFrame([dict(modelo=m, **S.metricas(g)) for m, g in TT2.groupby('modelo')])
TAB_T = RES_T[[c for c in ['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'R_vs_climatologia'] if c in RES_T]].merge(pont, on='modelo')
TTF = R2.com_fase(TT2, D, LIM)
FASE_T = R2.metricas_fase(TTF, ['modelo'])
PICO_T = R2.erros_pico(TT2, 'season')
COB_T = R2.cobertura_quantis(TT2)
f1 = FASE_T.pivot_table(index='modelo', columns='fase', values=['WIS_log', 'vies_pct', 'IC90'])
f1.columns = [f'{a}_{b}' for a, b in f1.columns]
RES_TEMP = TAB_T.merge(f1.reset_index(), on='modelo', how='left').merge(PICO_T, on='modelo', how='left') \
    .sort_values('R_vs_BB' if 'R_vs_BB' in TAB_T else 'WIS')
for nome, t in (('temporada_teste', TAB_T), ('temporada_teste_por_fase', FASE_T), ('temporada_teste_pico', PICO_T),
                ('temporada_teste_cobertura', COB_T), ('temporada_resumo_fases', RES_TEMP)):
    R2.salvar_tabela(t, PASTA_AV2, nome)
RES_TEMP.head(30).round(3)
"""),
    ('markdown', "## C · Os dois níveis lado a lado e o top 5 de cada métrica (inclusive pico e entressafra)"),
    ('code', r"""
# @title C1 · Tabela única (teste) e top 5
cols = ['modelo', 'nivel', 'R_vs_clim', 'WIS', 'WIS_log', 'MAE', 'WAPE', 'sMAPE', 'IC90', 'vies_pct',
        'WIS_log_alta (pico)', 'WIS_log_entressafra', 'erro_semana_abs_medio', 'erro_altura_mediano_pct', 'pico_no_IC90_pct']
t1 = RES_TEMP.rename(columns={'R_vs_climatologia': 'R_vs_clim'}).assign(nivel='temporada (SE25)')
ts = RES_SEM.rename(columns={'R_vs_Climatologia_sem': 'R_vs_clim'}).assign(nivel='semanal h=1')
t4 = TAB['teste'].query('h == 4').rename(columns={'R_vs_Climatologia_sem': 'R_vs_clim'}).assign(nivel='semanal h=4')
UNICA = pd.concat([t1, ts, t4], ignore_index=True)
UNICA = UNICA[[c for c in cols if c in UNICA]]
R2.salvar_tabela(UNICA, PASTA_AV2, 'tabela_unica_teste')
crit = {'R_vs_clim': True, 'WIS': True, 'MAE': True, 'WAPE': True, 'sMAPE': True,
        'WIS_log_alta (pico)': True, 'WIS_log_entressafra': True, 'erro_semana_abs_medio': True}
linhas = []
for nv, g in UNICA.groupby('nivel'):
    g = g[~g.modelo.isin(['climatologia', 'Climatologia_sem'])]
    for c, asc in crit.items():
        if c not in g:
            continue
        o = g.dropna(subset=[c]).sort_values(c, ascending=asc).head(5)
        linhas.append(dict(nivel=nv, metrica=c, **{f'{i + 1}º': f'{m} ({v:.3g})' for i, (m, v) in enumerate(zip(o.modelo, o[c]))}))
    if 'erro_altura_mediano_pct' in g:
        o = g.assign(_k=g['erro_altura_mediano_pct'].abs()).dropna(subset=['_k']).sort_values('_k').head(5)
        linhas.append(dict(nivel=nv, metrica='|erro na altura do pico|', **{f'{i + 1}º': f'{m} ({v:.3g}%)' for i, (m, v) in enumerate(zip(o.modelo, o['erro_altura_mediano_pct']))}))
TOP5 = pd.DataFrame(linhas)
R2.salvar_tabela(TOP5, PASTA_AV2, 'top5_por_metrica')
TOP5
"""),
]

# ------------------------------------------------------------------------------------------------- R2_2 semanal CPU
NB2 = [
    ('markdown', "# R2 · 2 · GBMs semanais: variáveis de fase (B3), tsfresh (C), peso no pico (D1) e mediana L1\n" + ANTES +
     "\nCPU High-RAM. tsfresh: ~20–40 min na primeira vez (grava uma UF por vez); modelos: ~2–3 h.\n\n"
     "- **B3 · fase:** Rt pela equação de renovação (intervalo serial de 1–4 semanas), variação do Rt, semanas desde a última entressafra, semanas desde o máximo das últimas 52, nível em relação ao p75 da UF, fração de semanas em alta.\n"
     "- **C · tsfresh:** ~780 características da janela de 52 semanas de log-incidência terminando na origem; seleção FRESH (teste de relevância com controle de falsas descobertas) **só com alvos anteriores ao teste**; ficam as 40 mais relevantes.\n"
     "- **D1 · peso no pico:** linhas cujo alvo está acima do p75 da UF pesam 3× no treino.\n"
     "- **Mediana L1:** a mediana vem de um modelo treinado só para ela, e os outros quantis são recentrados nela (a mediana dos modelos quantílicos ficou pior na rodada anterior)."),
    ('code', "NOTEBOOK = 'r2_semanal_cpu'\nPACOTES = 'tsfresh catboost'\n" + SETUP),
    ('code', r"""
# @title 2 · tsfresh (retoma por UF) e seleção só na afinação
TSF_M, TSF_COLS = R2.tsfresh_cache(D, os.path.join(PASTA_CACHE, 'tsfresh'), janela=52, parametros='efficient')
TSF_SEL = R2.selecionar_tsfresh(D, TSF_M, TSF_COLS, RODADAS['s2022']['c'], os.path.join(PASTA_CACHE, 'tsfresh_selecao.json'), k=40)
print(len(TSF_COLS), 'características ·', len(TSF_SEL), 'selecionadas:'); print(TSF_SEL[:15], '...')
"""),
    ('code', r"""
# @title 3 · Modelos (ordem = prioridade)
kw = dict(lim=LIM, tsf_M=TSF_M, tsf_cols=TSF_COLS)
MODELOS = [R2.GBMSemanalR2('GBM_xgb_fase_sem', 'xgb', fase=True, **kw),
           R2.GBMSemanalR2('GBM_xgb_fase_L1_sem', 'xgb', fase=True, mediana_L1=True, **kw),
           R2.GBMSemanalR2('GBM_xgb_tsf_sem', 'xgb', fase=True, tsf=TSF_SEL, **kw),
           R2.GBMSemanalR2('GBM_xgb_pico_sem', 'xgb', fase=True, peso_pico=2.0, **kw),
           R2.GBMSemanalR2('GBM_lgb_r2full_sem', 'lgb', fase=True, tsf=TSF_SEL, peso_pico=2.0, mediana_L1=True, **kw)]
[m.nome for m in MODELOS]
"""),
    ('code', LACO_SEM),
]

# ------------------------------------------------------------------------------------------------- R2_3 redes GPU
NB3 = [
    ('markdown', "# R2 · 3 · Redes de grafo: escala por UF (B1), pré-treino nas regiões (B2), quantis altos (D1), cabeça do total (D3)\n" + ANTES +
     "\nGPU L4 (ou A100). Temporada: ~3–6 h; semanal: ~1–2 h. As redes gravam checkpoint a cada 5 épocas em `cache_r2/ckpt/`; o pré-treino nas 435 regiões é feito uma vez por temporada e semente e reaproveitado.\n\n"
     "Todas as versões de temporada partem do melhor isolado da rodada anterior (**MTGNN com peso pela população**, alvo âncora) e mudam uma coisa por vez, mais uma versão com tudo junto."),
    ('code', "NOTEBOOK = 'r2_redes_gpu'\nPACOTES = ''\n" + SETUP),
    ('code', r"""
# @title 2 · Configurações
CK = os.path.join(PASTA_CACHE, 'ckpt')
base_t = dict(fabrica=lambda: U.MTGNN(), alvo='ancora', escala=False, peso_pop=0.5, cov=None)
TOT = (16, 67, 0.3)                       # D3: total das semanas 16–67 a partir da SE25 (≈ SE41→SE40), peso 0,3
CFG_T = [dict(base_t, nome='MTGNN_escala_pp_UF', escala=True),                                   # B1
         dict(base_t, nome='MTGNN_qalto_pp_UF', peso_q=R2.PESO_Q_ALTO),                          # D1
         dict(base_t, nome='MTGNN_total_pp_UF', tot=TOT),                                        # D3
         dict(base_t, nome='MTGNN_pre_pp_UF', pre=True, pre_nome='ancora', ep_pre=30, ep_ajuste=30),   # B2
         dict(base_t, nome='MTGNN_r2full_UF', escala=True, peso_q=R2.PESO_Q_ALTO, tot=TOT, pre=True,
              pre_nome='ancora_esc', ep_pre=30, ep_ajuste=30)]
base_s = dict(fabrica=lambda: U.MTGNN(), alvo='cresc', escala=False, cov=None, H=4)
CFG_S = [dict(base_s, nome='MTGNN_cresc_escala_sem', escala=True),
         dict(base_s, nome='MTGNN_cresc_qalto_sem', peso_q=R2.PESO_Q_ALTO),
         dict(base_s, nome='MTGNN_cresc_pre_sem', pre=True, pre_nome='cresc', ep_pre=30, ep_ajuste=30)]
MODELOS_T = {c['nome']: R2.RedeTemporadaR2(c, CK) for c in CFG_T}
MODELOS = [R2.RedeSemanalR2(c, CK) for c in CFG_S]
TEMPORADAS = TUNING + TESTE
list(MODELOS_T), [m.nome for m in MODELOS]
"""),
    ('markdown', "## Temporada (→ `resultados_uf_r2/`)"),
    ('code', LACO_TEMP),
    ('markdown', "## Semanal (→ `resultados_semanal_r2/`)"),
    ('code', LACO_SEM),
]

# ------------------------------------------------------------------------------------------------- R2_4 Optuna
NB4 = [
    ('markdown', "# R2 · 4 · Otimização: Optuna (E1) e mais sementes (E2)\n" + ANTES +
     "\nGPU (o XGBoost também usa a GPU). Parte A (GBM semanal): ~1–2 h para 40 tentativas. Parte B (MTGNN de temporada): ~3–6 h para 25 tentativas + o modelo final com 10 sementes.\n\n"
     "**Sem vazamento:** as tentativas são avaliadas **só nas temporadas de afinação** (2018–2021); o teste só é usado pelo modelo final, avaliado no notebook R2_1. O estudo fica em `cache_r2/*.db` e é retomado: rodar de novo continua até completar `N_TENTATIVAS`."),
    ('code', "NOTEBOOK = 'r2_optuna'\nPACOTES = 'optuna'\n" + SETUP),
    ('markdown', "## A · GBM semanal (XGBoost com variáveis de fase)"),
    ('code', r"""
# @title A · Optuna (retoma)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TENTATIVAS = 40
R_AF = {k: v for k, v in RODADAS.items() if S.grupo_rodada(k) == 'afinacao'}
def objetivo_gbm(trial):
    prm = dict(n_estimators=trial.suggest_int('n_estimators', 150, 900, step=50),
               max_depth=trial.suggest_int('max_depth', 3, 9),
               learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
               subsample=trial.suggest_float('subsample', 0.5, 1.0),
               colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
               min_child_weight=trial.suggest_float('min_child_weight', 1, 50, log=True),
               reg_lambda=trial.suggest_float('reg_lambda', 0.1, 20, log=True))
    n = prm.pop('n_estimators')
    m = R2.GBMSemanalR2('opt', 'xgb', fase=True, n_estimators=n, prm=prm, lim=LIM)
    w = []
    for rod, R in R_AF.items():
        o, q = m.prever_semanal(ctx, R)
        X = S.com_verdade(S.montar('opt', rod, R, D, o, V.arruma_quantis(q)), D)
        w.append(V.wis(np.log1p(X[V.QCOLS].values), np.log1p(X.y.values)).mean())
    return float(np.mean(w))
st, salva = R2.estudo_optuna('optuna_gbm_semanal', PASTA_CACHE)
feitas = sum(t.state.name == 'COMPLETE' for t in st.trials)
if feitas < N_TENTATIVAS:
    st.optimize(objetivo_gbm, n_trials=N_TENTATIVAS - feitas, callbacks=[salva])
print('melhor (WIS log na afinação):', round(st.best_value, 4), st.best_params)
"""),
    ('code', r"""
# @title A2 · Modelo final com os melhores hiperparâmetros (todas as rodadas → resultados_semanal_r2)
bp = dict(st.best_params)
n = bp.pop('n_estimators')
MODELOS = [R2.GBMSemanalR2('GBM_xgb_optuna_sem', 'xgb', fase=True, n_estimators=n, prm=bp, lim=LIM)]
R2.json_grava(dict(st.best_params, valor=st.best_value), os.path.join(PASTA_CACHE, 'optuna_gbm_semanal_melhor.json'))
""" + LACO_SEM.replace('# @title Rodar', '# rodar')),
    ('markdown', "## B · MTGNN de temporada (1 semente por tentativa; avaliado contra a climatologia na afinação)"),
    ('code', r"""
# @title B · Optuna (retoma)
N_TENTATIVAS_T = 25
def fabrica_mtgnn(p, sementes):
    return lambda: U.MTGNN(d=p['d'], epocas=p['epocas'], lr=p['lr'], L=p['L'], k_top=p['k_top'], beta=p['beta'],
                           sementes=sementes)
def objetivo_mtgnn(trial):
    p = dict(d=trial.suggest_categorical('d', [32, 48, 64]), epocas=trial.suggest_int('epocas', 30, 120, step=10),
             lr=trial.suggest_float('lr', 5e-4, 5e-3, log=True), L=trial.suggest_categorical('L', [52, 78, 104]),
             k_top=trial.suggest_int('k_top', 2, 8), beta=trial.suggest_float('beta', 0.02, 0.3, log=True))
    pp = trial.suggest_float('peso_pop', 0.0, 1.0)
    pasta = os.path.join(PASTA_CACHE, 'ckpt_optuna', f't{trial.number}')
    os.makedirs(pasta, exist_ok=True)
    cfg = dict(nome=f'opt{trial.number}', fabrica=fabrica_mtgnn(p, (1,)), alvo='ancora', escala=False, peso_pop=pp, cov=None)
    r = R2.R_temporada_vs_clim(ctx, R2.RedeTemporadaR2(cfg, pasta), TUNING)
    shutil.rmtree(pasta, ignore_errors=True)
    return r
stt, salva_t = R2.estudo_optuna('optuna_mtgnn_temporada', PASTA_CACHE)
feitas = sum(t.state.name == 'COMPLETE' for t in stt.trials)
if feitas < N_TENTATIVAS_T:
    stt.optimize(objetivo_mtgnn, n_trials=N_TENTATIVAS_T - feitas, callbacks=[salva_t])
print('melhor (R × climatologia na afinação):', round(stt.best_value, 4), stt.best_params)
"""),
    ('code', r"""
# @title B2 · E2: modelos finais com 10 sementes (Optuna e o melhor isolado anterior) → resultados_uf_r2
bp = dict(stt.best_params)
pp = bp.pop('peso_pop')
R2.json_grava(dict(stt.best_params, valor=stt.best_value), os.path.join(PASTA_CACHE, 'optuna_mtgnn_melhor.json'))
SEM10 = tuple(range(1, 11))
CK = os.path.join(PASTA_CACHE, 'ckpt')
MODELOS_T = {'MTGNN_optuna10_UF': R2.RedeTemporadaR2(dict(nome='MTGNN_optuna10_UF', fabrica=fabrica_mtgnn(bp, SEM10),
                                                           alvo='ancora', escala=False, peso_pop=pp, cov=None), CK),
             'MTGNN_pesopop10_UF': R2.RedeTemporadaR2(dict(nome='MTGNN_pesopop10_UF', fabrica=lambda: U.MTGNN(sementes=SEM10),
                                                            alvo='ancora', escala=False, peso_pop=0.5, cov=None), CK)}
TEMPORADAS = TUNING + TESTE
""" + LACO_TEMP.replace('# @title Rodar temporada', '# rodar temporada')),
]

# ------------------------------------------------------------------------------------------------- R2_5 Chronos
NB5 = [
    ('markdown', "# R2 · 5 · Chronos-2 ajustado aos dados de dengue (E3)\n" + ANTES +
     "\nGPU L4/A100, ~2–4 h. Para cada rodada, o Chronos-2 é ajustado (LoRA, 1.000 passos) só com as séries das 26 UFs até o corte daquela rodada e salvo em `cache_r2/chronos/` (não é refeito). "
     "Sem ajuste, ele empatou com a persistência no semanal e ficou em 1,14 × BB na temporada.\n\n"
     "Se a versão do pacote não tiver `Chronos2Pipeline.fit`, a célula avisa e para; nesse caso rode `pip install -U chronos-forecasting` e reinicie a sessão."),
    ('code', "NOTEBOOK = 'r2_chronos'\nPACOTES = '\"chronos-forecasting>=2.0\" peft && pip uninstall -y -q torchao'\n" + SETUP),
    ('code', r"""
# @title 2 · Semanal (→ resultados_semanal_r2)
PASTA_CH = os.path.join(PASTA_CACHE, 'chronos')
os.makedirs(PASTA_CH, exist_ok=True)
MODELOS = [R2.Chronos2FTSemanal(PASTA_CH, num_steps=1000)]
""" + LACO_SEM.replace('# @title Rodar', '# rodar')),
    ('code', r"""
# @title 3 · Temporada (→ resultados_uf_r2)
MODELOS_T = {'Chronos2FT_UF': R2.Chronos2FTTemporada(PASTA_CH, num_steps=1000)}
TEMPORADAS = TUNING + TESTE
""" + LACO_TEMP.replace('# @title Rodar temporada', '# rodar temporada')),
]

# ------------------------------------------------------------------------------------------------- R2_6 variáveis
TSF_LER = r"""
# @title 2 · Variáveis da rodada 2 (lidas do cache do R2_2; nada é recalculado)
SEL_ARQ = os.path.join(PASTA_CACHE, 'tsfresh_selecao.json')
assert os.path.exists(SEL_ARQ), 'Rode antes o R2_2 até a célula 2 (tsfresh) terminar.'
TSF_M, TSF_COLS = R2.tsfresh_cache(D, os.path.join(PASTA_CACHE, 'tsfresh'), janela=52, parametros='efficient')
TSF_SEL = R2.json_le(SEL_ARQ)['colunas']
print('tsfresh:', TSF_M.shape, '· selecionadas:', len(TSF_SEL))
COV_FASE = R2.fabrica_cov(LIM, fase=True)
COV_FASE_TSF = R2.fabrica_cov(LIM, fase=True, tsf=TSF_SEL, tsf_M=TSF_M, tsf_cols=TSF_COLS)
"""

NB6A = [
    ('markdown', "# R2 · 6a · Variáveis da rodada 2 nos outros modelos (GPU)\n" + ANTES +
     "\n**Pré-requisito:** o R2_2 (tsfresh e seleção) e o R2_4 (Optuna do GBM) já terminaram — este notebook só lê `cache_r2/tsfresh/`, `cache_r2/tsfresh_selecao.json` e `cache_r2/optuna_gbm_semanal_melhor.json`. Pode rodar junto com o R2_6b.\n\n"
     "GPU L4/A100, ~4–6 h. As variáveis de fase (log Rt, variação do Rt, semanas desde a entressafra e desde o máximo, nível − p75, fração de semanas em alta) e as 40 do tsfresh entram como **covariáveis padronizadas só com dados até o corte** de cada rodada/temporada (sem vazamento). "
     "Cada versão muda uma coisa em relação a um modelo que já existe, para a comparação ser direta:\n\n"
     "| Novo | Compara com | Muda |\n|---|---|---|\n"
     "| MTGNN_cresc_fase_sem | MTGNN_cresc_sem | + fase/Rt |\n"
     "| MTGNN_cresc_fasetsf_sem | MTGNN_cresc_sem | + fase/Rt + tsfresh |\n"
     "| GBM_xgb_optuna_tsf_sem | GBM_xgb_optuna_sem | + tsfresh |\n"
     "| MTGNN_fase_pp_UF | MTGNN_pesopop_UF | + fase/Rt |\n"
     "| MTGNN_fasetsf_pp_UF | MTGNN_pesopop_UF | + fase/Rt + tsfresh |\n\n"
     "Retomada: igual aos outros (arquivo por modelo × rodada; checkpoint das redes a cada 5 épocas em `cache_r2/ckpt/`)."),
    ('code', "NOTEBOOK = 'r2_variaveis_gpu'\nPACOTES = 'tsfresh'\n" + SETUP),
    ('code', TSF_LER),
    ('markdown', "## Semanal (→ `resultados_semanal_r2/`)"),
    ('code', r"""
# @title 3 · Modelos semanais
CK = os.path.join(PASTA_CACHE, 'ckpt')
bs = dict(fabrica=lambda: U.MTGNN(), alvo='cresc', escala=False, cov=None, H=4)
MODELOS = [R2.RedeSemanalR2(dict(bs, nome='MTGNN_cresc_fase_sem', cov_fn=COV_FASE), CK),
           R2.RedeSemanalR2(dict(bs, nome='MTGNN_cresc_fasetsf_sem', cov_fn=COV_FASE_TSF), CK)]
bp = R2.json_le(os.path.join(PASTA_CACHE, 'optuna_gbm_semanal_melhor.json'))
if bp is None:
    print('⚠️ sem optuna_gbm_semanal_melhor.json (R2_4 parte A) — GBM_xgb_optuna_tsf_sem fica de fora')
else:
    bp = {k: v for k, v in bp.items() if k != 'valor'}
    n = bp.pop('n_estimators')
    MODELOS.append(R2.GBMSemanalR2('GBM_xgb_optuna_tsf_sem', 'xgb', fase=True, tsf=TSF_SEL, tsf_M=TSF_M, tsf_cols=TSF_COLS,
                                   n_estimators=n, prm=bp, lim=LIM))
[m.nome for m in MODELOS]
"""),
    ('code', LACO_SEM),
    ('markdown', "## Temporada (→ `resultados_uf_r2/`)"),
    ('code', r"""
# @title 4 · Modelos de temporada
bt = dict(fabrica=lambda: U.MTGNN(), alvo='ancora', escala=False, peso_pop=0.5, cov=None)
MODELOS_T = {c['nome']: R2.RedeTemporadaR2(c, CK) for c in (dict(bt, nome='MTGNN_fase_pp_UF', cov_fn=COV_FASE),
                                                            dict(bt, nome='MTGNN_fasetsf_pp_UF', cov_fn=COV_FASE_TSF))}
TEMPORADAS = TUNING + TESTE
""" + LACO_TEMP.replace('# @title Rodar temporada', '# rodar temporada')),
]

NB6B = [
    ('markdown', "# R2 · 6b · ARIMA com regressores de fase (CPU)\n" + ANTES +
     "\nCPU comum, ~1–2 h. Pode rodar junto com o R2_6a. É o **ARIMA_Fourier_sem** (o mais preciso em MAE) mais 3 regressores: log Rt, variação do Rt e nível − p75 da UF, "
     "**defasados 4 semanas** — assim o valor usado para prever até h = 4 já é conhecido na origem (sem precisar prever o Rt). "
     "Padronização só com dados até o corte. Na temporada (68 semanas à frente) isso exigiria defasagem de 68 semanas e não faz sentido, por isso só há a versão semanal.\n\n"
     "Se o ajuste de alguma UF não convergir com os regressores, ela usa o ARIMA_Fourier normal (sem regressores)."),
    ('code', "NOTEBOOK = 'r2_variaveis_cpu'\nPACOTES = ''\n" + SETUP),
    ('code', r"""
# @title 2 · Modelo
MODELOS = [R2.ARIMAXSemanalR2(LIM, defasagem=4)]
[m.nome for m in MODELOS]
"""),
    ('code', LACO_SEM),
]

# ------------------------------------------------------------------------------------------------- R2_7 gráficos
NB7 = [
    ('markdown', "# R2 · 7 · Gráficos por UF: semanal e temporada com os campeões de cada área\n" + ANTES +
     "\nCPU comum, ~10–20 min. **Rode depois do R2_1 final** — este notebook só lê previsões e os arquivos que o R2_1 gravou em `avaliacao_r2/` (λ da temporada, calibração e os pesos dos ensembles com pesos) e recria exatamente os mesmos ensembles. Nada é treinado de novo.\n\n"
     "**Em cada figura (uma por UF e uma para o Brasil):**\n"
     "- **em cima, semanal (teste 2022–2025):** casos observados (preto), previsão de 1 semana à frente do melhor modelo semanal (**ENS_GBM_ARIMA_MTGNNcresc_sem**, azul, com a faixa de 90%) e previsão de 4 semanas à frente do melhor em 4 semanas (**ENS_stack_sem**, laranja tracejado);\n"
     "- **embaixo, temporada:** previsão feita em junho (SE25) para a temporada SE41→SE40 seguinte, nas 4 temporadas de teste: o melhor geral (**ENS_grafos3_λ**, azul com faixas de 50% e 90%), o melhor no pico entre os nossos (**ENS_stack_UF**, vermelho), o melhor para o total da temporada (**MTGNN_total_pp_UF+cal**, verde pontilhado) e o modelo-base oficial (**BB**, cinza tracejado).\n\n"
     "O Brasil é a soma das UFs (as faixas somadas quantil a quantil ficam um pouco largas demais). Saída: `avaliacao_r2/graficos_r2_<carimbo>/` com um PNG por UF, um PDF com todas e `totais_temporada.csv` (total observado × previsto por UF e temporada). Retomada: os PNGs prontos são pulados."),
    ('code', "NOTEBOOK = 'r2_graficos'\nPACOTES = ''\n" + SETUP),
    ('code', r"""
# @title 2 · Semanal: membros, ensemble de mediana e ensemble com pesos (pesos gravados pelo R2_1)
PASTAS_SEM = sorted(p for p in glob.glob(os.path.join(PASTA_DRIVE, 'resultados_semanal*'))
                    if os.path.isdir(p) and not p.endswith('_avaliacao'))
arq_ps = sorted(glob.glob(os.path.join(PASTA_AV2, 'pesos_stack_semanal_*.json')))
assert arq_ps, 'Rode antes o R2_1 (ele grava os pesos do ENS_stack_sem).'
PESOS_SEM = R2.json_le(arq_ps[-1])
MEMB_SEM = list(next(iter(PESOS_SEM.values())).keys())
MED3 = ['GBM_xgb_todas_sem', 'ARIMA_Fourier_sem', 'MTGNN_cresc_sem']
PRECISA = set(MEMB_SEM) | set(MED3)
fs = [f for p in PASTAS_SEM for f in glob.glob(os.path.join(p, '*.parquet')) if os.path.basename(f).split('__')[0] in PRECISA]
P0 = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(['modelo', 'rodada', 'h', 'date', 'uf_code'], keep='last')
P0 = P0[P0.rodada.map(S.grupo_rodada) == 'teste']
P0['date'], P0['origem'] = pd.to_datetime(P0['date']), pd.to_datetime(P0['origem'])
E1 = S.ensemble_semanal(P0, MED3, 'ENS_GBM_ARIMA_MTGNNcresc_sem')
PV = S.com_verdade(P0, D)
PV['fase_origem'] = R2.fase_origem(PV, D, LIM)
PV['balde'] = PV.h.astype(str) + '|' + PV.fase_origem
ST = R2.aplicar_stacking(PV, MEMB_SEM, ['rodada', 'h', 'date', 'uf_code', 'balde'], 'balde', PESOS_SEM, 'ENS_stack_sem')
SEM = {'1 semana à frente · ENS_GBM_ARIMA_MTGNNcresc_sem': E1[E1.h == 1].assign(date=lambda d: pd.to_datetime(d.date)),
       '4 semanas à frente · ENS_stack_sem': ST[ST.h == 4].assign(date=lambda d: pd.to_datetime(d.date))}
print('membros do stacking semanal:', MEMB_SEM, '·', {k: len(v) for k, v in SEM.items()})
"""),
    ('code', r"""
# @title 3 · Temporada: mesmas fontes, λ, ensembles e calibração do R2_1
PASTAS_UF = [p for p in sorted(glob.glob(os.path.join(PASTA_DRIVE, 'resultados_uf*'))) if os.path.isdir(p)]
FONTES = [('região→UF', PASTA_COMP)] + [(os.path.basename(p), p) for p in PASTAS_UF] + [('externo', PASTA_EXT)]
PU = L.carregar_previsoes(FONTES, ctx.Du.ufs, 'uf', mostrar=False)
PU = PU[~PU.modelo.str.startswith('ENS')][L.COLS_PREV]
ens0 = [os.path.join(p, 'ensembles_uf.parquet') for p in PASTAS_UF if os.path.exists(os.path.join(p, 'ensembles_uf.parquet'))]
ENS0 = pd.read_parquet(ens0[0])[L.COLS_PREV] if ens0 else PU.iloc[:0]
REF = ctx.ref[L.COLS_PREV] if ctx.ref is not None else PU.iloc[:0]
LAM_T = R2.json_le(os.path.join(PASTA_AV2, 'lambda_temporada.json'), {})
CAL2 = [U.ajusta_largura(PU[PU.modelo == m], l).assign(modelo=f'{m}+λ') for m, l in LAM_T.items() if l != 1.0 and (PU.modelo == m).any()]
TODOS = pd.concat([REF, PU, ENS0] + CAL2, ignore_index=True).drop_duplicates(['modelo', 'season', 'uf_code', 'date'], keep='last')
TODOS = TODOS[TODOS.season.isin(TESTE)]
arq_pt = sorted(glob.glob(os.path.join(PASTA_AV2, 'pesos_stack_temporada_*.json')))
assert arq_pt, 'Rode antes o R2_1 (ele grava os pesos do ENS_stack_UF).'
PESOS_T = R2.json_le(arq_pt[-1])
MEMB_T = list(next(iter(PESOS_T.values())).keys())
BASE_G = ['MTGNN_UF+λ', 'A3TGCN_UF+λ', 'GCNGRU_UF+λ']
FIXOS = ['MTGNN_pesopop_UF+λ', 'TFT_pool_UF', 'DoisEstagios_pool_UF'] + BASE_G
b_todos = [m for m in MEMB_T if m not in FIXOS]                  # o melhor modelo novo na afinação (o R2_1 o põe nos dois)
e3, _ = V.ensemble(TODOS, BASE_G + b_todos[:1], 'ENS_grafos3_λ', metodo='mediana')
c_data = {s: pd.Timestamp(D.datas[D.corte(s)]) for s in TESTE}
TV = S.com_verdade(TODOS, D)
TV['h'] = ((TV.date - TV.season.map(c_data)).dt.days // 7).astype(int)
TV['balde'] = R2.bloco_horizonte(TV.h.values)
STT = R2.aplicar_stacking(TV, MEMB_T, ['season', 'date', 'uf_code', 'balde'], 'balde', PESOS_T, 'ENS_stack_UF')
CAL_T = R2.json_le(os.path.join(PASTA_AV2, 'calibracao_temporada.json'), {})
TOT = R2.aplicar_cal(TV[TV.modelo == 'MTGNN_total_pp_UF'], 'balde', CAL_T['MTGNN_total_pp_UF']) if 'MTGNN_total_pp_UF' in CAL_T else TV.iloc[:0]
data = lambda d: d.assign(date=pd.to_datetime(d.date))
TEMP = {'melhor geral · ENS_grafos3_λ': data(e3), 'melhor no pico · ENS_stack_UF': data(STT),
        'melhor no total · MTGNN_total_pp_UF+cal': data(TOT), 'modelo-base oficial · BB': data(TODOS[TODOS.modelo == 'BB'])}
print('novo do ensemble:', b_todos, '·', {k: len(v) for k, v in TEMP.items()})
"""),
    ('code', r"""
# @title 4 · Figuras (uma por UF + Brasil), PDF único e totais por temporada
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import Image, display
PASTA_G = os.path.join(PASTA_AV2, f'graficos_r2_{R2.carimbo()[:8]}')
os.makedirs(PASTA_G, exist_ok=True)
OBS = pd.DataFrame(D.Y[:D.ultimo_obs + 1], index=pd.to_datetime(D.datas[:D.ultimo_obs + 1]), columns=D.unidades)
OBS[0] = OBS.sum(1)
ORDEM = [0] + sorted(D.unidades, key=lambda u: R2.SIGLAS.get(u, str(u)))
tot = []
for u in ORDEM:
    arq = os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')
    if not os.path.exists(arq) or REFAZER:
        fig = R2.grafico_uf(OBS[u], SEM, TEMP, u, arq, TESTE, ' · Brasil = soma das UFs' if u == 0 else '')
        plt.close(fig)
    tot.append(R2.totais_temporada(OBS[u], TEMP, u, TESTE))
    print('✅', R2.SIGLAS.get(u, u), flush=True)
with PdfPages(os.path.join(PASTA_G, 'graficos_UFs_r2.pdf.tmp')) as pdf:
    for u in ORDEM:
        img = plt.imread(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png'))
        f = plt.figure(figsize=(13, 8.2)); plt.imshow(img); plt.axis('off'); pdf.savefig(f, bbox_inches='tight'); plt.close(f)
os.replace(os.path.join(PASTA_G, 'graficos_UFs_r2.pdf.tmp'), os.path.join(PASTA_G, 'graficos_UFs_r2.pdf'))
TOT_T = pd.concat(tot, ignore_index=True)
TOT_T.to_csv(os.path.join(PASTA_G, 'totais_temporada.csv'), index=False)
print('Pasta:', PASTA_G)
for u in (0, 35, 31, 41, 52):
    display(Image(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')))
TOT_T.round(0)
"""),
]


def celula(tipo, src):
    src = src.strip('\n')
    c = {'cell_type': tipo, 'metadata': {}, 'source': [l + '\n' for l in src.split('\n')]}
    if tipo == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def grava(nome, celulas, gpu):
    nb = {'cells': [celula(t, s) for t, s in celulas],
          'metadata': {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                       **({'accelerator': 'GPU'} if gpu else {})},
          'nbformat': 4, 'nbformat_minor': 0}
    with open(os.path.join(AQUI, nome), 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('gerado:', nome)


grava('R2_1_Avaliacao.ipynb', NB1, False)
grava('R2_2_Semanal_CPU.ipynb', NB2, False)
grava('R2_3_Redes_GPU.ipynb', NB3, True)
grava('R2_4_Optuna.ipynb', NB4, True)
grava('R2_5_Chronos_GPU.ipynb', NB5, True)
grava('R2_6a_Variaveis_GPU.ipynb', NB6A, True)
grava('R2_6b_Variaveis_CPU.ipynb', NB6B, False)
grava('R2_7_Graficos_UF.ipynb', NB7, False)
