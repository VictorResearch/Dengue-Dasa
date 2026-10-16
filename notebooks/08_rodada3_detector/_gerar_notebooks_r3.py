"""Gera os notebooks da RODADA 3 (rode: python _gerar_notebooks_r3.py). Tudo em CPU. Ordem:

  R3_1_Detector.ipynb            ① detector do tamanho da temporada (jun, jul, ago, set, out), sinais escolhidos só na
                                 afinação, 3 métodos + média, avaliação de alarmes. ~10–20 min.
  R3_2_Mistura.ipynb             ② normal (ENS_grafos3_λ) + ③ pico → MIX_escala_UF, MIX_forma_UF, AJ_continuo_UF
                                 (parâmetros escolhidos na afinação) e as revisões mensais. ~20–40 min.
  R3_3_Avaliacao_Graficos.ipynb  régua oficial × BB, pico/entressafra/total, ganho das revisões e gráficos das 26 UFs + Brasil.
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

ANTES = r"""
**Antes de rodar (uma vez só)**, em `MyDrive/dengue_V4/` (que já tem tudo das rodadas 1 e 2):
1. suba **`r3_lib.py`** (pasta `data_imdc_2026/RODADA3/`) e confira que o `r2_lib.py` mais recente está lá;
2. **crie à mão** as pastas `resultados_uf_r3`, `revisoes_r3`, `cache_r3` e `avaliacao_r3`.

**Complementar:** nada das rodadas anteriores muda. As previsões oficiais (corte de junho) vão para `resultados_uf_r3/uf/` (o CompUF4 e o Comp5 também as enxergam). As revisões de julho a outubro vão para `revisoes_r3/uf/`, fora da régua oficial. As tabelas vão para `avaliacao_r3/`, com data e hora no nome e uma cópia `_ultimo`.

**Retomada:** se a internet ou a luz cair, reconecte e **rode tudo de novo**. Cada etapa grava o seu arquivo (variáveis, um arquivo por método × corte do detector, parâmetros em json, um arquivo por modelo × temporada, um PNG por UF) e o que já existe é pulado.

**Dados externos (opcional):** para testar sinais que não estão no FTP (previsão de ENSO do IRI, sorotipo, vacinação, positividade de exames), ponha em `cache_r3/externos_r3.csv` colunas `uf_code`, `season` e, se quiser, `corte` (`jun`/`jul`/`ago`/`set`/`out`), mais uma coluna por sinal. Cada valor deve ser o que se sabia **naquele corte**. Apague `cache_r3/detector_*` e `cache_r3/sinais_detector.json` para refazer.
"""

SETUP = r"""
# @title 1 · Drive, arquivos e pacotes
import os, sys, shutil, time, glob, subprocess, json, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/r3', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'uf_lib.py', 'semanal_lib.py', 'r2_lib.py', 'r3_lib.py',
                 'dados', 'dados_uf', 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou subir {item} para {PASTA_DRIVE}'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
sys.path.insert(0, PASTA_LOCAL)
PASTA_R3 = os.path.join(PASTA_DRIVE, 'resultados_uf_r3')
PASTA_REV = os.path.join(PASTA_DRIVE, 'revisoes_r3')
PASTA_C3 = os.path.join(PASTA_DRIVE, 'cache_r3')
PASTA_AV3 = os.path.join(PASTA_DRIVE, 'avaliacao_r3')
PASTA_AV2 = os.path.join(PASTA_DRIVE, 'avaliacao_r2')
PASTA_COMP = os.path.join(PASTA_DRIVE, 'resultados_competicao')
PASTA_EXT = os.path.join(PASTA_DRIVE, 'resultados_externos')
for p in (os.path.join(PASTA_R3, 'uf'), os.path.join(PASTA_REV, 'uf'), PASTA_C3, PASTA_AV3):
    os.makedirs(p, exist_ok=True)
import importlib, v4_lib as V, comp_lib as L, comp_modelos as CM, uf_lib as U, semanal_lib as S, r2_lib as R2, r3_lib as R3
for m in (V, L, CM, U, S, R2, R3):
    importlib.reload(m)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'),
                   os.path.join(PASTA_LOCAL, 'dados_uf'))
D = ctx.Du
TUNING, TESTE = U.TUNING, U.TESTE
REFAZER = False
def detector_escolhido():
    esc = R3.json_le(os.path.join(PASTA_C3, 'detector_escolhido.json'))
    assert esc, 'Rode antes o R3_1_Detector.'
    P = pd.concat([pd.read_parquet(os.path.join(PASTA_C3, f'detector_{m}_{k}.parquet')) for k, m in esc.items()], ignore_index=True)
    return esc, P
"""

NB1 = [
    ('markdown', "# R3 · 1 · Detector do tamanho da temporada\n" + ANTES +
     "\nCPU comum, ~10–20 min.\n\n"
     "**O que é:** uma linha por UF × temporada × corte. O alvo é o log do tamanho da temporada (SE41→SE40) **relativo ao normal da UF** (média das temporadas já terminadas). "
     "Os sinais são o que se sabe no corte: casos e espalhamento das últimas 4–13 semanas comparados com a mesma época dos anos anteriores, tendência, vizinhos e macrorregião, ENSO/IOD/PDO e a variação de 12 semanas, tamanho das últimas temporadas e anos desde a última grande.\n\n"
     "**Sem vazamento:** os sinais de cada corte são escolhidos só com temporadas até 2021. Para prever a temporada *s*, cada método treina só com temporadas que já tinham terminado no corte. "
     "Métodos: regressão ridge (com faixa pelos resíduos), regressão quantílica linear, LightGBM quantílico pequeno e a média dos três. Por corte, fica o de menor erro (pinball) na afinação.\n\n"
     "**Cortes:** jun = SE25 (regra oficial); jul = SE30; ago = SE34; set = SE38; out = SE40 (véspera do início da temporada)."),
    ('code', "NOTEBOOK = 'r3_detector'\n" + SETUP),
    ('code', r"""
# @title 2 · Variáveis e alvo (cache; retoma)
arq_ext = os.path.join(PASTA_C3, 'externos_r3.csv')
EXT = pd.read_csv(arq_ext) if os.path.exists(arq_ext) else None
print('externos:', None if EXT is None else list(EXT.columns))
X = R3.tabela_detector(ctx, list(range(2011, 2026)), PASTA_C3, externos=EXT)
print(X.shape)
X.groupby('corte').alvo.describe().round(2)
"""),
    ('code', r"""
# @title 3 · Sinais escolhidos só com temporadas ≤ 2021 (correlação de Spearman com o alvo)
ARQ_SIN = os.path.join(PASTA_C3, 'sinais_detector.json')
SINAIS = {k: R3.selecionar_sinais(X, k, ate_temporada=max(TUNING), arq=ARQ_SIN) for k in R3.CORTES}
pd.DataFrame({k: pd.Series(v['rho']) for k, v in R3.json_le(ARQ_SIN).items()}).round(2)
"""),
    ('code', r"""
# @title 4 · Previsões do detector (janela expansiva; um arquivo por método × corte) e avaliação
P = R3.prever_detector(ctx, X, SINAIS, list(range(2014, 2026)), PASTA_C3)
TD = R3.avaliar_detector(P)
R3.salvar_tabela(TD, PASTA_AV3, 'detector_avaliacao')
ESC = R3.escolher_detector(TD)
R3.json_grava(ESC, os.path.join(PASTA_C3, 'detector_escolhido.json'))
print('escolhido por corte (menor pinball na afinação):', ESC)
TD[TD.grupo != 'anterior'].pivot_table(index=['corte', 'metodo'], columns='grupo',
                                       values=['pinball', 'corr_mediana', 'IC90', 'sensibilidade', 'precisao']).round(2)
"""),
    ('code', r"""
# @title 5 · O que o detector disse em junho de cada temporada de teste (tamanho × o normal da UF)
DJ = pd.concat([P[(P.metodo == m) & (P.corte == k)] for k, m in ESC.items()])
DJ = DJ[(DJ.corte == 'jun') & DJ.season.isin(TESTE)].copy()
DJ['uf'] = DJ.uf_code.map(R2.SIGLAS)
DJ['real_x'], DJ['previsto_x'] = np.exp(DJ.alvo), np.exp(DJ.pred)
DJ['faixa90_x'] = [f'{np.exp(a):.1f}–{np.exp(b):.1f}' for a, b in zip(DJ.lower_90, DJ.upper_90)]
TAB = DJ.pivot_table(index='uf', columns='season', values=['real_x', 'previsto_x', 'p_grande']).round(2)
R3.salvar_tabela(TAB.reset_index(), PASTA_AV3, 'detector_junho_por_uf')
TAB
"""),
]

NB2 = [
    ('markdown', "# R3 · 2 · Mistura normal + pico (e revisões mensais)\n" + ANTES +
     "\nCPU comum, ~20–40 min. **Rode depois do R3_1.**\n\n"
     "- **② normal:** ENS_grafos3_λ, o melhor modelo de temporada da rodada 2, recriado do mesmo jeito que no R2_1 (λ e membros gravados em `avaliacao_r2/`).\n"
     "- **Trajetórias:** 1.000 trajetórias do normal (coerentes no tempo) e 1.000 sorteios do tamanho previsto pelo detector.\n"
     "- **Três jeitos de misturar:**\n"
     "  - `MIX_escala_UF`: quando o tamanho sorteado passa do limiar, a trajetória é reescalada para esse total, mantendo a forma do normal;\n"
     "  - `MIX_forma_UF`: igual, mas com a forma média dos anos grandes (da UF, senão da macrorregião, senão de todas). Ela já traz o início mais cedo dos anos epidêmicos;\n"
     "  - `AJ_continuo_UF`: sem limiar; toda trajetória anda uma fração *w* do caminho até o total do detector (*w* = 0 é o normal; *w* = 1 confia só no detector).\n"
     "- **Escolha dos parâmetros:** o limiar (1,5×, 2× ou 3×) e o *w* (0,25 a 1) são escolhidos pelo menor R × climatologia nas temporadas de afinação (2018–2021), no corte de junho.\n"
     "- **Revisões:** o mesmo, com o detector de jul/ago/set/out. O normal continua o de junho."),
    ('code', "NOTEBOOK = 'r3_mistura'\n" + SETUP),
    ('code', r"""
# @title 2 · Normal (ENS_grafos3_λ), BB e climatologia para afinação + teste (cache) e detector escolhido
NORMAL = R3.normal_ens_grafos3(ctx, PASTA_DRIVE, PASTA_AV2, PASTA_COMP, PASTA_EXT, PASTA_C3, TUNING + TESTE)
ESC, DET = detector_escolhido()
print(NORMAL.groupby(['modelo']).season.unique(), ESC)
"""),
    ('code', r"""
# @title 3 · Parâmetros de cada mistura (escolhidos na afinação; gravados em json)
PAR = R3.escolher_parametros(ctx, NORMAL, DET, PASTA_C3, M=500)
pd.DataFrame({v: dict(parametro=round(p['par'], 3), R_afinacao=min(r for _, r in p['grade']), R_normal_afinacao=round(p['normal'], 3))
              for v, p in PAR.items()}).T
"""),
    ('code', r"""
# @title 4 · Gerar as misturas (jun → resultados_uf_r3; jul–out → revisoes_r3). Um arquivo por modelo × temporada
for v, p in PAR.items():
    for corte in R3.CORTES:
        nome = R3.nome_mistura(v, corte)
        pasta = PASTA_R3 if corte == 'jun' else PASTA_REV
        falta = [s for s in TUNING + TESTE if REFAZER or not os.path.exists(L._arq(pasta, 'uf', nome, s))]
        if not falta:
            continue
        t0 = time.time()
        x = R3.gerar_mistura(ctx, NORMAL, DET, v, p['par'], corte, falta, M=1000)
        if x is None:
            print('sem dados:', nome); continue
        for s, g in x.assign(modelo=nome).groupby('season'):
            g.to_parquet(L._arq(pasta, 'uf', nome, s) + '.tmp', index=False)
            os.replace(L._arq(pasta, 'uf', nome, s) + '.tmp', L._arq(pasta, 'uf', nome, s))
        print(f'✅ {nome:28s} {sorted(x.season.unique())} · {time.time() - t0:.0f}s', flush=True)
"""),
]

NB3 = [
    ('markdown', "# R3 · 3 · Avaliação e gráficos da rodada 3\n" + ANTES +
     "\nCPU comum, ~10–20 min. **Rode depois do R3_2.**\n\n"
     "- **Régua oficial (teste 2022–2025, corte de junho):** R × BB, por fase (pico, intermediária, entressafra), erro no pico, erro no total da temporada e cobertura de cada quantil, comparando as misturas com o normal (ENS_grafos3_λ) e o BB.\n"
     "- **Escolha honesta:** a variante recomendada é a de menor R na **afinação**, e não a que for melhor no teste.\n"
     "- **Revisões:** quanto cada mês a mais de dados (jul → out) melhora a previsão da temporada.\n"
     "- **Gráficos:** uma figura por UF e uma do Brasil (2022–2025), com a recomendada, o normal e o BB, e o que o detector disse em junho (probabilidade de temporada grande e tamanho previsto). Saída: `avaliacao_r3/graficos_r3_<data>/` (PNGs, PDF e totais por temporada)."),
    ('code', "NOTEBOOK = 'r3_avaliacao'\n" + SETUP),
    ('code', r"""
# @title 2 · Ler normal, BB, climatologia, misturas e revisões
NORMAL = R3.normal_ens_grafos3(ctx, PASTA_DRIVE, PASTA_AV2, PASTA_COMP, PASTA_EXT, PASTA_C3, TUNING + TESTE)
MIX = pd.concat([pd.read_parquet(f) for p in (PASTA_R3, PASTA_REV) for f in glob.glob(os.path.join(p, 'uf', '*.parquet'))],
                ignore_index=True)
MIX['date'] = pd.to_datetime(MIX.date)
TODOS = pd.concat([NORMAL, MIX[L.COLS_PREV]], ignore_index=True)
PAR = R3.json_le(os.path.join(PASTA_C3, 'mistura_parametros.json'))
RECOMENDADA = R3.nome_mistura(min(PAR, key=lambda v: min(r for _, r in PAR[v]['grade'])), 'jun')
ESC, DET = detector_escolhido()
print('modelos:', sorted(TODOS.modelo.unique()), '\nrecomendada (escolhida na afinação):', RECOMENDADA)
"""),
    ('code', r"""
# @title 3 · Régua oficial no teste (corte de junho): geral, fases, pico, total e cobertura
OF = TODOS[TODOS.season.isin(TESTE) & ~TODOS.modelo.str.contains('@')]
_, RES = V.avaliar(OF[L.COLS_PREV], ctx.ver_uf, ref=('BB', 'climatologia'))
RES = RES[RES.temporadas == len(TESTE)]
TT = S.com_verdade(OF, D)
pont = pd.DataFrame([dict(modelo=m, **S.metricas(g)) for m, g in TT.groupby('modelo')])
LIM = R2.limiares(D)
FASE = R2.metricas_fase(R2.com_fase(TT, D, LIM), ['modelo'])
PICO = R2.erros_pico(TT, 'season')
COB = R2.cobertura_quantis(TT)
f1 = FASE.pivot_table(index='modelo', columns='fase', values=['WIS_log', 'vies_pct', 'IC90'])
f1.columns = [f'{a}_{b}' for a, b in f1.columns]
TAB = RES[['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'R_vs_climatologia']].merge(pont, on='modelo') \
    .merge(f1.reset_index(), on='modelo', how='left').merge(PICO, on='modelo', how='left').sort_values('R_vs_BB')
for nome, t in (('temporada_teste_r3', TAB), ('temporada_teste_por_fase_r3', FASE), ('temporada_teste_pico_r3', PICO),
                ('temporada_teste_cobertura_r3', COB)):
    R3.salvar_tabela(t, PASTA_AV3, nome)
cols = ['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'WAPE', 'IC90', 'WIS_log_alta (pico)', 'WIS_log_entressafra',
        'IC90_alta (pico)', 'erro_total_mediano_pct', 'pico_no_IC90_pct']
TAB[[c for c in cols if c in TAB]].round(3)
"""),
    ('code', r"""
# @title 4 · Revisões mensais: R × BB e erro no total da temporada por corte
lin = []
for v in PAR:
    for corte in R3.CORTES:
        nome = R3.nome_mistura(v, corte)
        g = TODOS[(TODOS.modelo == nome) & TODOS.season.isin(TESTE)]
        if g.empty:
            continue
        _, r = V.avaliar(pd.concat([g.assign(modelo='x'), NORMAL[NORMAL.modelo == 'BB']])[L.COLS_PREV], ctx.ver_uf, ref=('BB',))
        pk = R2.erros_pico(S.com_verdade(g.assign(modelo='x'), D), 'season')
        lin.append(dict(variante=v, corte=corte, R_vs_BB=float(r.set_index('modelo').loc['x', 'R_vs_BB']),
                        erro_total_mediano_pct=float(pk.erro_total_mediano_pct.iloc[0]), pico_no_IC90_pct=float(pk.pico_no_IC90_pct.iloc[0])))
REV = pd.DataFrame(lin)
R3.salvar_tabela(REV, PASTA_AV3, 'revisoes_mensais')
REV.pivot_table(index='variante', columns='corte', values=['R_vs_BB', 'erro_total_mediano_pct'])[
    [(a, k) for a in ('R_vs_BB', 'erro_total_mediano_pct') for k in R3.CORTES]].round(3)
"""),
    ('code', r"""
# @title 5 · Gráficos (26 UFs + Brasil), PDF e totais por temporada (pula os PNGs prontos)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import Image, display
PASTA_G = os.path.join(PASTA_AV3, f'graficos_r3_{R3.carimbo()[:8]}')
os.makedirs(PASTA_G, exist_ok=True)
OBS = pd.DataFrame(D.Y[:D.ultimo_obs + 1], index=pd.to_datetime(D.datas[:D.ultimo_obs + 1]), columns=D.unidades)
OBS[0] = OBS.sum(1)
PREV = {f'recomendada · {RECOMENDADA}': TODOS[TODOS.modelo == RECOMENDADA],
        'normal · ENS_grafos3_λ': TODOS[TODOS.modelo == 'ENS_grafos3_λ'],
        'modelo-base oficial · BB': TODOS[TODOS.modelo == 'BB']}
DJ = DET[(DET.corte == 'jun')]
ORDEM = [0] + sorted(D.unidades, key=lambda u: R2.SIGLAS.get(u, str(u)))
tot = []
for u in ORDEM:
    arq = os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')
    if not os.path.exists(arq) or REFAZER:
        det = DJ[DJ.uf_code == u].set_index('season') if u != 0 else None
        fig = R3.grafico_temporada_uf(OBS[u], PREV, u, arq, TESTE, det=det, titulo=' · Brasil = soma das UFs' if u == 0 else '')
        plt.close(fig)
    tot.append(R2.totais_temporada(OBS[u], PREV, u, TESTE))
with PdfPages(os.path.join(PASTA_G, 'graficos_UFs_r3.pdf.tmp')) as pdf:
    for u in ORDEM:
        img = plt.imread(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png'))
        f = plt.figure(figsize=(13, 4.6)); plt.imshow(img); plt.axis('off'); pdf.savefig(f, bbox_inches='tight'); plt.close(f)
os.replace(os.path.join(PASTA_G, 'graficos_UFs_r3.pdf.tmp'), os.path.join(PASTA_G, 'graficos_UFs_r3.pdf'))
TOT = pd.concat(tot, ignore_index=True)
TOT.to_csv(os.path.join(PASTA_G, 'totais_temporada_r3.csv'), index=False)
print('Pasta:', PASTA_G)
for u in (0, 35, 31, 41, 42, 23):
    display(Image(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')))
TOT.round(0)
"""),
]


def celula(tipo, src):
    src = src.strip('\n')
    c = {'cell_type': tipo, 'metadata': {}, 'source': [l + '\n' for l in src.split('\n')]}
    if tipo == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def grava(nome, celulas):
    nb = {'cells': [celula(t, s) for t, s in celulas],
          'metadata': {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
          'nbformat': 4, 'nbformat_minor': 0}
    with open(os.path.join(AQUI, nome), 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('gerado:', nome)


grava('R3_1_Detector.ipynb', NB1)
grava('R3_2_Mistura.ipynb', NB2)
grava('R3_3_Avaliacao_Graficos.ipynb', NB3)
