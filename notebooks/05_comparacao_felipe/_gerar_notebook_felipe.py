"""Gera CompFelipe.ipynb (rode: python _gerar_notebook_felipe.py).
Parte A (CPU, ~5 min): o modelo do Felipe no MEU jogo (previsões feitas no PC por felipe_no_meu_jogo.py) avaliado
  com o MESMO `V.avaliar` do CompUF4, ao lado de todos os meus modelos e ensembles (S1–S4, WIS vs BB) + métricas
  pontuais (MAE, WAPE, sMAPE) e o top 5 de cada métrica.
Parte B (GPU, ~30–60 min): as minhas redes de grafo no jogo DELE (1 a 4 semanas à frente, modelo fixo por rodada,
  teste 2023–24 e temporada 1), comparadas com a persistência, o XGBoost dele e as minhas estimativas locais
  (meu_no_jogo_dele.py)."""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

CABECALHO = r"""
# CompFelipe · os dois jogos

**Antes de rodar**, em `MyDrive/dengue_V4/` (que já tem tudo dos notebooks CompUF):
1. suba **`semana_lib.py`** (pasta `data_imdc_2026/COMPARACAO_FELIPE/`);
2. crie a pasta **`COMPARACAO_FELIPE/`** no Drive e suba `felipe_no_meu_jogo_previsoes.parquet` e `meu_no_jogo_dele_previsoes.parquet` (gerados no PC).

| Parte | O que faz | Ambiente | Tempo |
|---|---|---|---|
| A · meu jogo | Felipe × meus modelos na régua do CompUF4 (dados até SE25, SE41→SE40, WIS vs BB, S1–S4) | CPU | ~5 min |
| B · jogo dele | minhas redes de grafo prevendo 1–4 semanas à frente (teste 2023–24 e S1) | GPU (T4/L4) | 30–60 min |

A parte A não precisa de GPU: rode só as células 1 e A se quiser. Os resultados vão para `MyDrive/dengue_V4/resultados_felipe/`. A parte B retoma de onde parou.
"""

SETUP = r"""
# @title 1 · Drive, arquivos e pacotes
import os, sys, shutil, glob, time, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/uf', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'uf_lib.py', 'semana_lib.py', 'dados', 'dados_uf',
                 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou subir {item} para {PASTA_DRIVE}'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
sys.path.insert(0, PASTA_LOCAL)
PASTA_RES = os.path.join(PASTA_DRIVE, 'resultados_uf')
PASTA_COMP = os.path.join(PASTA_DRIVE, 'resultados_competicao')
PASTA_EXT = os.path.join(PASTA_DRIVE, 'resultados_externos')
PASTA_FEL_IN = os.path.join(PASTA_DRIVE, 'COMPARACAO_FELIPE')
PASTA_FEL = os.path.join(PASTA_DRIVE, 'resultados_felipe')
os.makedirs(PASTA_FEL, exist_ok=True)
import importlib, v4_lib as V, comp_lib as L, comp_modelos as CM, uf_lib as U, semana_lib as SL
for m in (V, L, CM, U, SL):
    importlib.reload(m)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 120)
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'),
                   os.path.join(PASTA_LOCAL, 'dados_uf'))
TESTE = U.TESTE
VERDADE = pd.read_parquet(os.path.join(PASTA_LOCAL, 'dados_uf', 'tabela_uf_semanal.parquet'),
                          columns=['uf_code', 'date', 'casos']).rename(columns={'casos': 'y'})
VERDADE['date'] = pd.to_datetime(VERDADE['date'])
"""

PARTE_A = r"""
# @title A · Felipe no MEU jogo × meus modelos (S1–S4; mesma régua do CompUF4)
PASTAS_UF = [p for p in sorted(glob.glob(os.path.join(PASTA_DRIVE, 'resultados_uf*'))) if os.path.isdir(p)]
FONTES = [('região→UF', PASTA_COMP)] + [(os.path.basename(p), p) for p in PASTAS_UF] + [('externo', PASTA_EXT)]
PU = L.carregar_previsoes(FONTES, ctx.Du.ufs, 'uf', mostrar=False)
PU = PU[~PU.modelo.str.startswith('ENS')]
ens = [os.path.join(p, 'ensembles_uf.parquet') for p in PASTAS_UF if os.path.exists(os.path.join(p, 'ensembles_uf.parquet'))]
ENS = pd.read_parquet(ens[0]) if ens else pd.DataFrame(columns=L.COLS_PREV)       # ENS_UF_top* e modelos +λ
FEL = pd.read_parquet(os.path.join(PASTA_FEL_IN, 'felipe_no_meu_jogo_previsoes.parquet'))
FEL = FEL[FEL.modelo.isin(['Felipe_XGB_ponto', 'Felipe_XGB_quantis', 'persistencia_SE25'])]
TODOS = pd.concat([PU[L.COLS_PREV], ENS[L.COLS_PREV], FEL[L.COLS_PREV]], ignore_index=True)
TODOS = TODOS[TODOS.season.isin(TESTE)].drop_duplicates(['modelo', 'season', 'uf_code', 'date'], keep='last')
por, res = V.avaliar(TODOS, ctx.ver_uf, ref=('BB',))
res = res[res.temporadas == len(TESTE)]
# métricas pontuais (mediana) nas mesmas semanas
x = TODOS.merge(VERDADE, on=['uf_code', 'date'])
pont = pd.DataFrame([dict(modelo=m, **SL.metricas(g)) for m, g in x.groupby('modelo')])
A = res[['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'WIS_medio', 'IC90']].merge(pont, on='modelo')
A = A.sort_values('R_vs_BB').reset_index(drop=True)
A.to_csv(os.path.join(PASTA_FEL, 'meu_jogo_ranking.csv'), index=False)
print('Posição do Felipe no meu jogo:')
display(A[A.modelo.str.contains('Felipe|persist|BB$|ENS_UF_top5|MTGNN_UF\\+')].round(3))
display(A.head(15).round(3))
"""

TOP5 = r"""
# @title A2 · Top 5 por métrica no meu jogo (teste 2022–2025)
crit = {'R_vs_BB (WIS, régua oficial)': ('R_vs_BB', True), 'WIS médio': ('WIS_medio', True), 'MAE': ('MAE', True),
        'WAPE': ('WAPE', True), 'sMAPE': ('sMAPE', True), 'cobertura IC90 (perto de 90)': ('IC90', None)}
tops = {}
for rot, (c, asc) in crit.items():
    o = A.assign(_k=(A[c] - 90).abs()).sort_values('_k') if asc is None else A.sort_values(c, ascending=asc)
    tops[rot] = [f'{m} ({v:.3g})' for m, v in zip(o.modelo.head(5), o[c].head(5))]
TOP = pd.DataFrame(tops, index=[1, 2, 3, 4, 5]).T
TOP.to_csv(os.path.join(PASTA_FEL, 'top5_por_metrica.csv'))
TOP
"""

PARTE_B = r"""
# @title B · Minhas redes de grafo no jogo DELE (treino uma vez por rodada; 1–4 semanas à frente)
VALIDACAO = False                       # True: também 2019–2022 (como o piloto dele; dobra o tempo)
REFAZER = False
REDES = lambda: {'MTGNN_UF': U.MTGNN(), 'A3TGCN_UF': U.GCNGRU(atencao=True), 'GCNGRU_UF': U.GCNGRU(),   # noqa: E731
                 'STGCN_UF': U.STGCN()}
RODADAS = {'teste_2023-24': ('2022-12-31', '2023-01-01', '2024-12-31'),
           'S1_IMDC': ('2022-06-25', '2022-10-09', '2023-10-07')}        # S1: dados até a SE25/2022
if VALIDACAO:
    RODADAS.update({f'valid_{a}': (f'{a - 1}-12-31', f'{a}-01-01', f'{a}-12-31') for a in (2019, 2020, 2021, 2022)})
for rod, (ft, ia, fa) in RODADAS.items():
    for nome, rede in REDES().items():
        arq = os.path.join(PASTA_FEL, f'semanal_{rod}_{nome}.parquet')
        if os.path.exists(arq) and not REFAZER:
            continue
        t0 = time.time()
        try:
            SL.prever_semanal(rede, ctx.Du, ft, ia, fa, H=4).assign(avaliacao=rod).to_parquet(arq, index=False)
            print(f'{rod} · {nome}: {time.time() - t0:.0f}s', flush=True)
        except Exception:
            traceback.print_exc()
"""

AVAL_B = r"""
# @title B2 · Avaliação no jogo dele (h = 1 e h = 4) com persistência, XGBoost dele e minhas estimativas do PC
R = pd.concat([pd.read_parquet(a) for a in glob.glob(os.path.join(PASTA_FEL, 'semanal_*.parquet'))], ignore_index=True)
R = R.merge(VERDADE, on=['uf_code', 'date'])
g = R.groupby(['avaliacao', 'h', 'date', 'uf_code'])
ens = g.pred.median().rename('pred').reset_index().assign(modelo='ENS_grafos')
R = pd.concat([R[['avaliacao', 'h', 'date', 'uf_code', 'modelo', 'pred', 'y']], ens.merge(VERDADE, on=['uf_code', 'date'])])
loc = os.path.join(PASTA_FEL_IN, 'meu_no_jogo_dele_previsoes.parquet')
if os.path.exists(loc):                                                   # persistência, XGBoost dele, GBM_log, ARIMA (PC)
    pc = pd.read_parquet(loc)
    pc['date'] = pd.to_datetime(pc['date'])
    cols = [c for c in pc.columns if c not in ('date', 'uf_code', 'y', 'avaliacao', 'h')]
    pc = pc.melt(['avaliacao', 'h', 'date', 'uf_code', 'y'], cols, var_name='modelo', value_name='pred').dropna(subset=['pred'])
    R = pd.concat([R, pc], ignore_index=True)
    w = R[R.modelo.isin(['ENS_grafos', 'GBM_log_cresc', 'ARIMA_Fourier'])].pivot_table(
        index=['avaliacao', 'h', 'date', 'uf_code', 'y'], columns='modelo', values='pred').dropna()
    R = pd.concat([R, w.median(1).rename('pred').reset_index().assign(modelo='ENS_grafos+GBM+ARIMA')], ignore_index=True)
chave = R[R.modelo == 'ENS_grafos'][['avaliacao', 'h', 'date', 'uf_code']]          # mesmas linhas para todos
R = R.merge(chave, on=['avaliacao', 'h', 'date', 'uf_code'])
B = pd.DataFrame([dict(avaliacao=a, h=h, modelo=m, **SL.metricas(x)) for (a, h, m), x in R.groupby(['avaliacao', 'h', 'modelo'])])
B = B.sort_values(['avaliacao', 'h', 'MAE'])
B.to_csv(os.path.join(PASTA_FEL, 'jogo_dele_ranking.csv'), index=False)
B.round(2)
"""


def celula(tipo, src):
    src = src.strip('\n')
    c = {'cell_type': tipo, 'metadata': {}, 'source': [l + '\n' for l in src.split('\n')]}
    if tipo == 'code':
        c.update(execution_count=None, outputs=[])
    return c


nb = {'cells': [celula('markdown', CABECALHO), celula('code', SETUP),
                celula('markdown', '## A · O modelo do Felipe no meu jogo'), celula('code', PARTE_A), celula('code', TOP5),
                celula('markdown', '## B · Os meus modelos no jogo dele (precisa de GPU)'), celula('code', PARTE_B),
                celula('code', AVAL_B)],
      'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []},
                   'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
      'nbformat': 4, 'nbformat_minor': 0}
with open(os.path.join(AQUI, 'CompFelipe.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('CompFelipe.ipynb gerado')
