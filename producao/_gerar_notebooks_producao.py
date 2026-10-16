"""Gera os notebooks de PRODUÇÃO (rode: python producao/_gerar_notebooks_producao.py).

  Producao_Semanal.ipynb      GPU (L4/T4) recomendada · ~1–2 h · previsão de 1–4 semanas dos 5 melhores semanais
  Producao_Temporada.ipynb    GPU (L4) · ~1–3 h na 1ª vez · previsão da temporada dos 5 melhores de temporada
  Relatorio_Campeoes.ipynb    CPU · ~10 min · gráficos por UF/região/Brasil e o documento dos campeões (HTML)
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

SETUP = r"""
# @title 1 · Área de trabalho (Drive no Colab ou pasta local), pacotes e contexto
import os, sys, shutil, glob, json, time, subprocess, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/prod', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False          # Jupyter aberto na área de trabalho
if NO_COLAB:
    if PACOTES:
        subprocess.run(f'pip install -q {PACOTES}', shell=True, check=True)
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'uf_lib.py', 'semanal_lib.py', 'r2_lib.py', 'r3_lib.py', 'r4_lib.py',
                 'r5_lib.py', 'r6_lib.py', 'r7_lib.py', 'prod_lib.py', 'relatorio_campeoes.py', 'graficos_previsoes.py',
                 'dados', 'dados_uf', 'referencias_imdc3', 'parametros'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou {item} em {PASTA_DRIVE}: rode scripts/04_montar_area_de_trabalho.py e suba a pasta'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
sys.path.insert(0, PASTA_LOCAL)
import importlib, v4_lib as V, comp_lib as L, uf_lib as U, semanal_lib as S, r2_lib as R2, r3_lib as R3, prod_lib as PL
for m in (V, L, U, S, R2, R3, PL):
    importlib.reload(m)
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 120)
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'), os.path.join(PASTA_LOCAL, 'dados_uf'))
D = ctx.Du
TUNING, TESTE = U.TUNING, U.TESTE
P = {k: os.path.join(PASTA_DRIVE, v) for k, v in dict(PS='producao_semanal', PT='producao_temporada', REL='relatorio_campeoes',
                                                      CACHE='cache_producao').items()}
for p in P.values():
    os.makedirs(p, exist_ok=True)
PAR = PL.ler_parametros(os.path.join(PASTA_LOCAL, 'parametros'))
ULTIMA = pd.Timestamp(D.datas[D.ultimo_obs]).date()
print('última semana observada:', ULTIMA, '· UFs:', D.N)
"""

# ------------------------------------------------------------------------------------------------- semanal
SEM_MD = r"""# Produção · previsão SEMANAL (1 a 4 semanas à frente)

Roda os **5 melhores modelos semanais** (MAE médio no teste 2022–2025) a partir da **última semana completa**. A última semana dos dados chega incompleta (atraso de notificação: em 03/2026 tinha ~36% dos casos), então por padrão ela é descartada como origem (`DESCARTAR = 1`) e o h = 1 a re-estima:

| # | Modelo | MAE h = 1 / 2 / 3 / 4 (casos/semana) |
|---|---|---|
| 1 | **ENS_stack_sem** (campeão) | 270 / 383 / 475 / 583 |
| 2 | ARIMA_Fourier_sem | 252 / 398 / 525 / 646 |
| 3 | GBM_xgb_optuna_sem | 308 / 403 / 508 / 639 |
| 4 | ENS_GBM_ARIMA_sem | 271 / 414 / 526 / 648 |
| 5 | ENS_GBM_ARIMA_MTGNNcresc_sem | 267 / 418 / 544 / 680 |

O campeão é a média ponderada de **9 membros** (XGBoost em 5 versões, ARIMA + Fourier, 2 MTGNN e Chronos-2 ajustado). Cada membro é **retreinado** com os dados até a semana de origem, com os hiperparâmetros e a seleção de variáveis da pesquisa (`parametros/`). Os pesos (horizonte × fase da UF) vêm da afinação 2018–2021.

**Ambiente:** GPU (L4 ou T4). Sem GPU funciona, mas as redes e o Chronos ficam lentos; para um teste rápido, use `SO_MEMBROS` (ex.: só ARIMA e XGBoost). Se um membro falhar, os pesos do stacking são renormalizados entre os que rodaram.

**Antes:** monte a área de trabalho (`python scripts/04_montar_area_de_trabalho.py`) e suba a pasta para `MyDrive/dengue_V4/`.

**Saídas** em `producao_semanal/`: um arquivo por membro, `previsao_semanal_<data>.parquet` (top 5, 9 quantis) e `tabela_semanal_<data>.csv` (campeão, por UF)."""

SEM_C2 = r"""
# @title 2 · Rodada de produção e membros
SO_MEMBROS = None          # ex.: ['ARIMA_Fourier_sem', 'GBM_xgb_todas_sem'] para um teste rápido
DESCARTAR = 1              # últimas semanas incompletas (atraso de notificação); 0 = usar a última semana como origem
ROD = PL.rodada_producao(D, descartar=DESCARTAR)   # origem = última semana completa; h = 1 re-estima a incompleta
print(ROD)
MEMBROS = PL.membros_semanais(ctx, PAR, P['CACHE'], so=SO_MEMBROS)
print('membros:', [m.nome for m in MEMBROS])
"""

SEM_C3 = r"""
# @title 3 · Treinar, prever e combinar (retoma de onde parou)
PM, ERROS = PL.prever_semanal(ctx, MEMBROS, os.path.join(P['PS'], 'membros'), ROD)
print('erros:', ERROS)
PREV = PL.ensembles_semanais(PM, D, PAR)
PREV = PREV[PREV.modelo.isin(PL.TOP5_SEMANAL + [m.nome for m in MEMBROS])]
rot = f'{ULTIMA:%Y%m%d}'
PL.gravar(PREV, P['PS'], f'previsao_semanal_{rot}.parquet'); PL.gravar(PREV, P['PS'], 'previsao_semanal_ultimo.parquet')
TAB = PL.tabela_semanal(PREV, D)
PL.gravar(TAB, P['PS'], f'tabela_semanal_{rot}.csv'); PL.gravar(TAB, P['PS'], 'tabela_semanal_ultimo.csv')
ORIGEM = pd.Timestamp(D.datas[list(ROD.values())[0]['origens'][0]]).date()
print(f'Campeão ({PL.CAMPEAO_SEMANAL}) · origem {ORIGEM} (última semana completa) · casos previstos (mediana) por semana-alvo:')
TAB.pivot_table(index='uf', columns='date', values='pred').round(0)
"""

SEM_C4 = r"""
# @title 4 · (opcional) Reproduzir o teste 2022–2025 com o mesmo código (horas de GPU)
RODAR_BACKTEST = False
if RODAR_BACKTEST:
    ROD_T = S.rodadas(D, temporadas=TESTE, felipe=False)
    PB, _ = PL.prever_semanal(ctx, PL.membros_semanais(ctx, PAR, P['CACHE']), os.path.join(P['PS'], 'backtest'), ROD_T)
    EB = S.com_verdade(PL.ensembles_semanais(PB, D, PAR), D)
    RES = pd.DataFrame([dict(modelo=m, h=h, **S.metricas(g)) for (m, h), g in EB.groupby(['modelo', 'h'])])
    PL.gravar(RES, P['PS'], 'backtest_metricas.csv')
    display(RES[RES.modelo.isin(PL.TOP5_SEMANAL)].pivot_table(index='modelo', columns='h', values='MAE').round(0))
"""

# ------------------------------------------------------------------------------------------------- temporada
TEMP_MD = r"""# Produção · previsão da TEMPORADA (SE41 → SE40, com dados até a SE25)

Roda os **5 melhores modelos de temporada** (WIS e MAE no teste 2022–2025) para a temporada mais recente cuja SE25 já está nos dados:

| # | Modelo | WIS | MAE | Altura do pico | Total |
|---|---|---|---|---|---|
| 1 | **ENS_pico7_simples_UF** (campeão) | 721 | 1.277 | +2% | −0,6% |
| 2 | MIX_escala5_UF | 746 | 1.279 | +12% | +9% |
| 3 | ENS_pico7_UF | 902 | 1.452 | −20% | −16% |
| 4 | MIX_elastico5_UF | 1.010 | 1.506 | −12% | −8% |
| 5 | AJ_continuo_UF | 1.285 | 1.672 | −28% | −25% |

**Cadeia:**
1. **Normal** (`ENS_grafos3_λ`): mediana de 4 redes de grafo (MTGNN, A3T-GCN e GCNGRU com λ = 0,8; MTGNN do Optuna com 10 sementes). Precisa de **GPU**.
2. **Detector** de anos grandes: ridge do log do tamanho relativo ao normal da UF, com sinais escolhidos só até 2021.
3. **Misturas:** elástica (w = 0,5), de escala (limiar 3×) e contínua (p = 0,25).
4. **Ensembles do pico** (rodada 7).

`NORMAL_SEM_GPU = True` troca o normal pela climatologia **só para testar o fluxo**: não é o modelo campeão.

**Saídas** em `producao_temporada/`: `previsao_temporada_<s>.parquet` (top 5 + normal), `detector_<s>.csv` e `resumo_temporada_<s>.csv` (campeão, por UF)."""

TEMP_C2 = r"""
# @title 2 · Temporada-alvo e normal (4 redes de grafo, GPU; retoma)
S_ALVO = PL.temporada_alvo(D)
RODAR_BACKTEST = False                      # True: também as temporadas do teste (2022–2025), para conferir as métricas
NORMAL_SEM_GPU = False                      # True: climatologia no lugar do normal (só para testar o fluxo)
TEMPORADAS = sorted(set([S_ALVO] + (TESTE if RODAR_BACKTEST else [])))
print('temporada-alvo:', S_ALVO, f'({S_ALVO}/{S_ALVO + 1}, prevista com dados até a SE25 de {S_ALVO}) · temporadas:', TEMPORADAS)
if NORMAL_SEM_GPU:
    print('⚠️ normal = climatologia (teste do fluxo; NÃO é o modelo campeão)')
    NORMAL = PL.normal_climatologia(D, TEMPORADAS)
else:
    PASTA_REDES = os.path.join(P['PT'], 'redes')
    PL.rodar_normal(ctx, PL.modelos_normal(PAR, os.path.join(P['CACHE'], 'ckpt_temporada')), TEMPORADAS, PASTA_REDES)
    NORMAL = PL.normal(PASTA_REDES, TEMPORADAS)
NORMAL.groupby('season').size()
"""

TEMP_C3 = r"""
# @title 3 · Detector → misturas → ensembles do pico
X, DET = PL.prever_temporada(ctx, NORMAL, TEMPORADAS, P['CACHE'])
for s in TEMPORADAS:
    PL.gravar(X[X.season == s], P['PT'], f'previsao_temporada_{s}.parquet')
    PL.gravar(DET[DET.season == s], P['PT'], f'detector_{s}.csv')
RES = PL.resumo_temporada(X, D, S_ALVO)
d = DET[DET.season == S_ALVO].assign(uf=lambda x: x.uf_code.map(R2.SIGLAS), tamanho_x_normal=lambda x: np.exp(x.pred))
RES = RES.merge(d[['uf', 'p_grande', 'tamanho_x_normal']], on='uf', how='left')
PL.gravar(RES, P['PT'], f'resumo_temporada_{S_ALVO}.csv')
print(f'{PL.CAMPEAO_TEMPORADA} · temporada {S_ALVO}/{S_ALVO + 1}: total, semana e altura do pico por UF; P(ano grande) e tamanho previsto pelo detector')
RES.round(2)
"""

TEMP_C4 = r"""
# @title 4 · (se RODAR_BACKTEST) métricas no teste 2022–2025
if RODAR_BACKTEST:
    OF = pd.concat([X[X.season.isin(TESTE)], ctx.ref[ctx.ref.modelo == 'BB'][L.COLS_PREV]], ignore_index=True)
    _, R = V.avaliar(OF[L.COLS_PREV], ctx.ver_uf, ref=('BB',))
    TT = S.com_verdade(OF, D)
    PT = pd.DataFrame([dict(modelo=m, **S.metricas(g)) for m, g in TT.groupby('modelo')])
    TAB = R[['modelo', 'R_vs_BB', 'UFs_melhor_que_BB']].merge(PT, on='modelo').sort_values('WIS')
    PL.gravar(TAB, P['PT'], 'backtest_metricas.csv')
    display(TAB[['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'WIS', 'MAE', 'WAPE', 'sMAPE', 'IC90']].round(3))
"""

# ------------------------------------------------------------------------------------------------- relatório
REL_MD = r"""# Relatório dos modelos campeões (gráficos por UF, região e Brasil)

Junta, para cada **UF, macrorregião e Brasil**:
- **semanal (ENS_stack_sem):** o que o campeão previu 1 semana antes em todo o histórico de avaliação (2018–2026) e o leque das **próximas 4 semanas**;
- **temporada (ENS_pico7_simples_UF):** a previsão de cada temporada feita em junho (2018/19 → a mais recente).

**De onde vêm as previsões** (o que existir primeiro):
1. as saídas dos notebooks de produção (`producao_semanal/`, `producao_temporada/`, inclusive os backtests);
2. as pastas da pesquisa no Drive (`resultados_semanal*`, `resultados_uf_r7/`).

**Saída:** `relatorio_campeoes/modelos_campeoes.html` (autocontido; abre em qualquer navegador) e os PNG em `relatorio_campeoes/figuras/`. CPU, ~10 min."""

REL_C2 = r"""
# @title 2 · Previsões do campeão semanal: histórico (h = 1…4) e próximas semanas
import graficos_previsoes as G, relatorio_campeoes as RC
importlib.reload(G); importlib.reload(RC)
MEMB = list(next(iter(PAR['pesos_stack_semanal'].values())).keys())
pastas = [os.path.join(P['PS'], 'backtest')] + sorted(p for p in glob.glob(os.path.join(PASTA_DRIVE, 'resultados_semanal*'))
                                                     if os.path.isdir(p) and not p.endswith('_avaliacao'))
P0 = S.ler_semanal([p for p in pastas if os.path.isdir(p)])
P0 = P0[P0.modelo.isin(MEMB) & P0.rodada.str.match(r'^s\d{4}$')].copy()
P0['date'], P0['origem'] = pd.to_datetime(P0['date']), pd.to_datetime(P0['origem'])
HIST = PL.ensembles_semanais(P0, D, PAR) if len(P0) else P0
HIST = HIST[HIST.modelo == PL.CAMPEAO_SEMANAL]
print('histórico semanal:', HIST.rodada.unique() if len(HIST) else 'nenhum (rode o backtest do Producao_Semanal ou use as pastas da pesquisa)')
arq_f = os.path.join(P['PS'], 'previsao_semanal_ultimo.parquet')
FUT = pd.read_parquet(arq_f) if os.path.exists(arq_f) else None
FUT = FUT[FUT.modelo == PL.CAMPEAO_SEMANAL] if FUT is not None else None
print('próximas semanas:', None if FUT is None else sorted(FUT.date.dt.date.unique()))
"""

REL_C3 = r"""
# @title 3 · Previsões do campeão de temporada (produção + pesquisa)
fs = sorted(glob.glob(os.path.join(P['PT'], 'previsao_temporada_*.parquet')))
TEMP = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame(columns=L.COLS_PREV)
r7 = sorted(glob.glob(os.path.join(PASTA_DRIVE, 'resultados_uf_r7', 'uf', f'{PL.CAMPEAO_TEMPORADA}_s*.parquet')))
if r7:
    PESQ = pd.concat([pd.read_parquet(f) for f in r7], ignore_index=True)
    TEMP = pd.concat([TEMP, PESQ[~PESQ.season.isin(set(TEMP.season))]], ignore_index=True)
TEMP = TEMP[TEMP.modelo == PL.CAMPEAO_TEMPORADA].copy()
TEMP['date'] = pd.to_datetime(TEMP.date)
S_ULT = int(TEMP.season.max()) if len(TEMP) else None
print('temporadas:', sorted(TEMP.season.unique()))
"""

REL_C4 = r"""
# @title 4 · Gráficos (Brasil, 5 regiões, 26 UFs) e o documento HTML
OBS = G.observado(D)
sem = (G.agregar_previsoes(HIST) if len(HIST) else None, G.agregar_previsoes(FUT) if FUT is not None and len(FUT) else None)
tem = G.agregar_previsoes(TEMP) if len(TEMP) else None
FIG = G.figuras_todas(OBS, os.path.join(P['REL'], 'figuras'), semanal=sem if any(x is not None for x in sem) else None, temporada=tem)
TS = PL.tabela_semanal(FUT, D) if FUT is not None else None
TS = TS.pivot_table(index='uf', columns='date', values='pred').round(0).reset_index() if TS is not None else None
if TS is not None:
    TS.columns = ['UF'] + [f'{pd.Timestamp(c):%d/%m/%Y}' for c in TS.columns[1:]]
TT = PL.resumo_temporada(TEMP, D, S_ULT) if S_ULT else None
arq = RC.gerar(os.path.join(P['REL'], 'modelos_campeoes.html'), FIG, TS, TT,
               dict(ultima_semana=f'{ULTIMA:%d/%m/%Y}', temporada=f'{S_ULT}/{S_ULT + 1}' if S_ULT else '–'))
print('documento:', arq, f'({os.path.getsize(arq) / 1e6:.1f} MB)')
from IPython.display import Image, display
for u in ('Brasil', 'Sudeste', 'SP', 'PR'):
    for k in ('semanal', 'temporada'):
        if FIG.get(u, {}).get(k):
            display(Image(FIG[u][k]))
"""

NOTEBOOKS = {
    'Producao_Semanal.ipynb': [('markdown', SEM_MD), ('code', "PACOTES = 'tsfresh catboost xgboost lightgbm statsmodels \"chronos-forecasting>=2.0\" peft && pip uninstall -y -q torchao'\n" + SETUP),
                               ('code', SEM_C2), ('code', SEM_C3), ('code', SEM_C4)],
    'Producao_Temporada.ipynb': [('markdown', TEMP_MD), ('code', "PACOTES = ''\n" + SETUP), ('code', TEMP_C2), ('code', TEMP_C3), ('code', TEMP_C4)],
    'Relatorio_Campeoes.ipynb': [('markdown', REL_MD), ('code', "PACOTES = ''\n" + SETUP), ('code', REL_C2), ('code', REL_C3), ('code', REL_C4)],
}


def celula(tipo, src):
    src = src.strip('\n')
    c = {'cell_type': tipo, 'metadata': {}, 'source': [l + '\n' for l in src.split('\n')]}
    if tipo == 'code':
        c.update(execution_count=None, outputs=[])
    return c


if __name__ == '__main__':
    for nome, cels in NOTEBOOKS.items():
        nb = {'cells': [celula(t, s) for t, s in cels],
              'metadata': {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                           'accelerator': 'GPU' if 'Producao' in nome else 'None'},
              'nbformat': 4, 'nbformat_minor': 0}
        with open(os.path.join(AQUI, nome), 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print('gerado:', nome)
