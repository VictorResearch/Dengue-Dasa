"""Gera o notebook FINAL de temporada (rode: python _gerar_notebook_r5.py).

  R5_Modelo_Final_Temporada.ipynb   CPU comum (High-RAM se puder), 40–90 min na 1ª vez; retoma de onde parou.
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

MD0 = r"""# R5 · Modelo final de temporada: um especialista para cada fase

**Ideia:** nenhum modelo sozinho é bom na temporada inteira. Na rodada 4 (teste 2022–2025):

| | Melhor na entressafra | Melhor no pico |
|---|---|---|
| Modelo | normal (ENS_grafos3_λ) | MIX_escala |
| WIS da fase | 23,6 | 2.136 (o normal: 4.065) |
| Viés da fase | +49% (o MIX_escala: +156%) | −36% (o normal: −82%) |

Este notebook monta um modelo que **troca de especialista conforme a fase esperada de cada semana**:

1. **Detector corrigido.** O detector ampliado da R4 errou 2023 (previu MG ×0,49; real ×20) por causa de uma variável nacional do Atlântico norte, recorde em 2023. Aqui os candidatos são:
   - o detector da R3;
   - o ampliado **sem variáveis nacionais**;
   - a média dos dois.

   A escolha é pelo pinball nas temporadas ≤ 2021 (anos grandes valendo 3×), nas mesmas linhas para todos.
2. **Membros refeitos com o detector corrigido:** MIX_escala5, MIX_elastico5 e DoisEstagios_det5. A MTGNN_cond5 usa as redes **já treinadas na R4** (`cache_r4/ckpt/`) e só refaz a previsão: não há retreino. Entram também o normal, a MTGNN_aug, o MIX_escala e o AJ_continuo da R3.
3. **Fase esperada de cada semana × UF:** em junho, cada semana recebe um rótulo pelo nível que um modelo de referência prevê para ela, contra os limiares p25/p75 da UF (até 2017): entressafra, intermediária ou pico. Recebe também o regime (P(ano grande) do detector: baixa, média ou alta). Nada disso usa o que aconteceu depois.
4. **Pesos por fase × regime,** escolhidos na afinação pelo critério misto (½ R × climatologia + ½ WIS absoluto; ajuste em `R4.PESO_ABS`). Os pesos são suavizados em 5 semanas para a curva não dar saltos.
   - `ENS_fase_*`: mistura ponderada;
   - `ENS_troca_*`: só o melhor membro de cada fase;
   - `_nivMIX`: a fase esperada vem do MIX_escala5, que enxerga os anos grandes; sem o sufixo, vem do normal.
5. **Calibração por fase (`+cal`):** largura e posição ajustadas dentro de cada fase. Subir o pico não mexe na entressafra.
6. **Escolha honesta:** cada ensemble é validado na afinação deixando uma temporada de fora. O **recomendado** é o de menor critério misto fora da amostra, decidido antes de olhar o teste. Depois vêm o teste (régua oficial, fases, pico, cobertura), os gráficos das 26 UFs + Brasil e a tabela da temporada mais recente.

**Antes de rodar:** suba `r5_lib.py` (pasta `data_imdc_2026/RODADA5/`) para `MyDrive/dengue_V4/`. Ele precisa do que as rodadas 3 e 4 gravaram (`cache_r3/`, `cache_r4/` com `detector4_variaveis.parquet` e `ckpt/`, `resultados_uf_r3/`, `resultados_uf_r4/`). As pastas `resultados_uf_r5`, `cache_r5` e `avaliacao_r5` são criadas sozinhas.

**Ambiente:** CPU comum (High-RAM se puder), 40–90 min na primeira vez. A GPU só acelera a etapa da MTGNN, que é opcional.

**Retomada:** se cair, **rode tudo de novo**. O detector, os membros (um arquivo por modelo × temporada), as validações e os PNGs ficam gravados e são pulados. Para refazer uma etapa, apague o arquivo dela em `cache_r5/` ou `resultados_uf_r5/uf/`."""

SETUP = r"""
# @title 1 · Drive, arquivos e pacotes
import os, sys, shutil, time, glob, subprocess, json, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/r5', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'uf_lib.py', 'semanal_lib.py', 'r2_lib.py', 'r3_lib.py',
                 'r4_lib.py', 'r5_lib.py', 'dados', 'dados_uf', 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou subir {item} para {PASTA_DRIVE}'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
sys.path.insert(0, PASTA_LOCAL)
P = {k: os.path.join(PASTA_DRIVE, v) for k, v in dict(R5='resultados_uf_r5', C5='cache_r5', AV5='avaliacao_r5', R4='resultados_uf_r4',
     C4='cache_r4', R3='resultados_uf_r3', C3='cache_r3', AV2='avaliacao_r2', COMP='resultados_competicao', EXT='resultados_externos').items()}
for p in (os.path.join(P['R5'], 'uf'), P['C5'], P['AV5']):
    os.makedirs(p, exist_ok=True)
import importlib, v4_lib as V, comp_lib as L, comp_modelos as CM, uf_lib as U, semanal_lib as S, r2_lib as R2, r3_lib as R3, r4_lib as R4, r5_lib as R5
for m in (V, L, CM, U, S, R2, R3, R4, R5):
    importlib.reload(m)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'), os.path.join(PASTA_LOCAL, 'dados_uf'))
D = ctx.Du
TUNING, TESTE = U.TUNING, U.TESTE
REFAZER = False
R4.PESO_ABS = 0.5     # critério misto = (1 − p)·R + p·WIS absoluto relativo. Suba para priorizar a altura dos picos
"""

C2 = r"""
# @title 2 · Detector corrigido (R3, ampliado sem variáveis nacionais, média dos dois)
ESC, PDET, PE = R5.detector_corrigido(ctx, P['C3'], P['C4'], P['C5'])
_, DET = R5.detector_escolhido5(P['C5'])
print('escolhido por corte:', ESC)
display(PE.pivot_table(index='metodo', columns='corte', values='pinball_escolha').round(3))
T = R4.avaliar_detector4(PDET[PDET.corte == 'jun'])
R5.salvar_tabela(T, P['AV5'], 'detector5_avaliacao')
T = T[T.grupo == 'teste'].sort_values('pinball')[['metodo', 'pinball', 'corr_mediana', 'IC90', 'sensibilidade', 'precisao', 'vies_grandes_x']]
print('junho, TESTE (só para conferir; a escolha acima não olhou o teste):'); display(T.round(2))
j = DET[(DET.corte == 'jun') & (DET.season == 2023)].copy()
j['uf'] = j.uf_code.map(R2.SIGLAS)
j['real_x'], j['previsto_x'], j['q95_x'] = np.exp(j.alvo), np.exp(j.pred), np.exp(j.upper_90)
j.sort_values('alvo', ascending=False)[['uf', 'real_x', 'previsto_x', 'q95_x', 'p_grande']].round(2).head(12)
"""

C3 = r"""
# @title 3 · Membros com o detector corrigido (um arquivo por modelo × temporada; pula os prontos)
NORMAL = R4.ler_normal(ctx, PASTA_DRIVE, P['AV2'], P['COMP'], P['EXT'], P['C3'], TUNING + TESTE)
PAR3 = R3.json_le(os.path.join(P['C3'], 'mistura_parametros.json'))
W = R4.escolher_w_elastico(ctx, NORMAL, DET, P['C5'], M=500)
print('limiar do MIX_escala (R3):', round(PAR3['escala']['par'], 3), '· w do MIX_elastico:', W['w'])
def falta(nome):
    return [s for s in TUNING + TESTE if REFAZER or not os.path.exists(L._arq(P['R5'], 'uf', nome, s))]
GERA = {'MIX_escala5_UF': lambda ss: R3.gerar_mistura(ctx, NORMAL, DET, 'escala', PAR3['escala']['par'], 'jun', ss, M=1000),
        'MIX_elastico5_UF': lambda ss: R4.gerar_mix_elastico(ctx, NORMAL, DET, W['w'], 'jun', ss, M=1000),
        'DoisEstagios_det5_UF': lambda ss: R4.gerar_dois_estagios_det(ctx, DET, 'jun', ss, M=1000)}
for nome, f in GERA.items():
    ss = falta(nome)
    if ss:
        t0 = time.time(); x = f(ss)
        R4.grava_modelo(x, P['R5'], nome)
        print(f'✅ {nome:24s} {ss} · {time.time() - t0:.0f}s', flush=True)
# MTGNN_cond5: redes da R4 já treinadas, nova previsão com o detector corrigido (CPU serve; ~1–3 min por temporada)
CK = os.path.join(P['C4'], 'ckpt')
for s in falta('MTGNN_cond5_UF'):
    try:
        t0 = time.time(); x = R5.prever_mtgnn_salva(ctx, s, DET, CK)
        if x is None:
            print(f'⚠️ {s}: pesos da MTGNN_cond não encontrados em cache_r4/ckpt; fica sem esse membro'); continue
        R4.grava_modelo(x, P['R5'], 'MTGNN_cond5_UF')
        print(f'✅ MTGNN_cond5_UF {s} · {time.time() - t0:.0f}s', flush=True)
    except Exception:
        traceback.print_exc()
"""

C4 = r"""
# @title 4 · Candidatos e fase esperada de cada semana
R3M = R4.ler_pastas(P['R3']); R3M = R3M[R3M.modelo.isin(['MIX_escala_UF', 'AJ_continuo_UF'])]
R4M = R4.ler_pastas(P['R4']); R4M = R4M[R4M.modelo.isin(['MTGNN_aug_UF'])]
R5M = R4.ler_pastas(P['R5']); R5M = R5M[~R5M.modelo.str.startswith('ENS_')]
TODOS = pd.concat([NORMAL, R3M, R4M, R5M], ignore_index=True)
CAND = ['ENS_grafos3_λ', 'MIX_escala5_UF', 'MIX_elastico5_UF', 'DoisEstagios_det5_UF', 'MTGNN_cond5_UF', 'MTGNN_aug_UF',
        'MIX_escala_UF', 'AJ_continuo_UF']
cnt = TODOS.groupby('modelo').season.nunique()
MEMBROS = [m for m in CAND if cnt.get(m, 0) == len(TUNING + TESTE)]
print('membros:', MEMBROS, '· faltando:', [m for m in CAND if m not in MEMBROS])
LIM = R2.limiares(D)
NIV = {'': R5.nivel_previsto(TODOS, 'ENS_grafos3_λ', D, LIM)}
if 'MIX_escala5_UF' in MEMBROS:
    NIV['_nivMIX'] = R5.nivel_previsto(TODOS, 'MIX_escala5_UF', D, LIM)
BASE = TODOS[TODOS.modelo.isin(MEMBROS + ['climatologia'])]
XB = {tag: R5.com_balde(BASE, DET, niv) for tag, niv in NIV.items()}
pd.crosstab(XB[''].drop_duplicates(['season', 'uf_code', 'date']).nivel, XB[''].drop_duplicates(['season', 'uf_code', 'date']).regime)
"""

C5 = r"""
# @title 5 · Validação na afinação (deixando uma temporada de fora) e escolha do recomendado
VARIANTES = {f'ENS_{tipo}{tag}_UF': (tag, tipo == 'troca') for tag in XB for tipo in ('fase', 'troca')}
CV = []
for nome, (tag, troca) in VARIANTES.items():
    arq = os.path.join(P['C5'], f'cv_{nome}.parquet')
    if os.path.exists(arq) and not REFAZER:
        CV.append(pd.read_parquet(arq)); continue
    t0 = time.time()
    x = R5.cv_fase(ctx, XB[tag], MEMBROS, nome, troca=troca)[L.COLS_PREV]
    R5._grava_pq(x, arq); CV.append(x)
    print(f'validou {nome} · {time.time() - t0:.0f}s', flush=True)
CV = pd.concat(CV, ignore_index=True)
AF = R4.criterio(ctx, pd.concat([BASE[BASE.season.isin(TUNING)][L.COLS_PREV], CV]))
R5.salvar_tabela(AF, P['AV5'], 'afinacao_criterio_misto_r5')
RECOMENDADA = AF.modelo.iloc[0]
print('RECOMENDADO (menor critério misto na afinação; ensembles fora da amostra):', RECOMENDADA)
AF.round(3)
"""

C6 = r"""
# @title 6 · Ajuste final (4 temporadas de afinação) → previsões de todas as temporadas em resultados_uf_r5
ENS, PESOS = [], {}
for nome, (tag, troca) in VARIANTES.items():
    pesos, cal = R5.ajustar_fase(ctx, XB[tag][XB[tag].season.isin(TUNING)], MEMBROS, troca)
    e, ec = R5.prever_fase(XB[tag], MEMBROS, pesos, cal, nome)
    for x in (e, ec):
        R4.grava_modelo(x[L.COLS_PREV], P['R5'], x.modelo.iloc[0]); ENS.append(x[L.COLS_PREV])
    PESOS[nome] = dict(pesos=pesos, calibracao=cal)
R5.json_grava(dict(membros=MEMBROS, variantes=PESOS, recomendada=RECOMENDADA, peso_abs=R4.PESO_ABS, detector=ESC),
              os.path.join(P['AV5'], f'modelo_final_{R5.carimbo()}.json'))
TODOS = pd.concat([TODOS] + ENS, ignore_index=True)
nome_rec = RECOMENDADA.replace('+cal', '')
if nome_rec in PESOS:
    pw = PESOS[nome_rec]['pesos']
    tab = pd.DataFrame({b: {m: v for m, v in p.items() if not m.startswith('_')} for b, p in pw.items()}).T
    print(f'pesos de {nome_rec} por fase esperada × regime (linhas) e membro (colunas):')
    display(tab.round(2))
    print('calibração por fase (λ = largura, δ = deslocamento em log):'); display(pd.DataFrame(PESOS[nome_rec]['calibracao']).round(3))
"""

C7 = r"""
# @title 7 · TESTE 2022–2025 (olhado uma vez): geral, fases, pico, total e cobertura
OF = TODOS[TODOS.season.isin(TESTE)]
_, RES = V.avaliar(OF[L.COLS_PREV], ctx.ver_uf, ref=('BB', 'climatologia'))
RES = RES[RES.temporadas == len(TESTE)]
CRIT = R4.criterio(ctx, OF, ref='BB')[['modelo', 'WIS_rel', 'criterio']].rename(columns={'WIS_rel': 'WIS_abs_rel_BB', 'criterio': 'criterio_misto_BB'})
TT = S.com_verdade(OF, D)
pont = pd.DataFrame([dict(modelo=m, **S.metricas(g)) for m, g in TT.groupby('modelo')])
FASE = R2.metricas_fase(R2.com_fase(TT, D, LIM), ['modelo'])
PICO = R2.erros_pico(TT, 'season')
COB = R2.cobertura_quantis(TT)
f1 = FASE.pivot_table(index='modelo', columns='fase', values=['WIS', 'WIS_log', 'vies_pct', 'IC90'])
f1.columns = [f'{a}_{b}' for a, b in f1.columns]
TAB = RES[['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'R_vs_climatologia']].merge(CRIT, on='modelo', how='left') \
    .merge(pont, on='modelo').merge(f1.reset_index(), on='modelo', how='left').merge(PICO, on='modelo', how='left').sort_values('criterio_misto_BB')
TAB['recomendado'] = TAB.modelo == RECOMENDADA
for nome, t in (('temporada_teste_r5', TAB), ('temporada_teste_por_fase_r5', FASE), ('temporada_teste_pico_r5', PICO),
                ('temporada_teste_cobertura_r5', COB)):
    R5.salvar_tabela(t, P['AV5'], nome)
cols = ['modelo', 'recomendado', 'R_vs_BB', 'UFs_melhor_que_BB', 'WIS_abs_rel_BB', 'criterio_misto_BB', 'WIS', 'WIS_alta (pico)', 'WIS_entressafra',
        'vies_pct_alta (pico)', 'vies_pct_entressafra', 'vies_pct_intermediaria', 'IC90', 'erro_altura_mediano_pct', 'erro_total_mediano_pct', 'pico_no_IC90_pct']
TAB[[c for c in cols if c in TAB]].round(3)
"""

C8 = r"""
# @title 8 · Gráficos (26 UFs + Brasil), PDF e totais por temporada
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import Image, display
PASTA_G = os.path.join(P['AV5'], f'graficos_r5_{R5.carimbo()[:8]}')
os.makedirs(PASTA_G, exist_ok=True)
OBS = pd.DataFrame(D.Y[:D.ultimo_obs + 1], index=pd.to_datetime(D.datas[:D.ultimo_obs + 1]), columns=D.unidades)
OBS[0] = OBS.sum(1)
PREV = {f'modelo final · {RECOMENDADA}': TODOS[TODOS.modelo == RECOMENDADA],
        'normal · ENS_grafos3_λ': TODOS[TODOS.modelo == 'ENS_grafos3_λ'],
        'modelo-base oficial · BB': TODOS[TODOS.modelo == 'BB']}
DJ = DET[DET.corte == 'jun']
ORDEM = [0] + sorted(D.unidades, key=lambda u: R2.SIGLAS.get(u, str(u)))
tot = []
for u in ORDEM:
    arq = os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')
    if not os.path.exists(arq) or REFAZER:
        det = DJ[DJ.uf_code == u].set_index('season') if u != 0 else None
        fig = R3.grafico_temporada_uf(OBS[u], PREV, u, arq, TESTE, det=det, titulo=' · modelo final' + (' · Brasil = soma das UFs' if u == 0 else ''))
        plt.close(fig)
    tot.append(R2.totais_temporada(OBS[u], PREV, u, TESTE))
with PdfPages(os.path.join(PASTA_G, 'graficos_UFs_r5.pdf.tmp')) as pdf:
    for u in ORDEM:
        img = plt.imread(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png'))
        f = plt.figure(figsize=(13, 4.6)); plt.imshow(img); plt.axis('off'); pdf.savefig(f, bbox_inches='tight'); plt.close(f)
os.replace(os.path.join(PASTA_G, 'graficos_UFs_r5.pdf.tmp'), os.path.join(PASTA_G, 'graficos_UFs_r5.pdf'))
TOT = pd.concat(tot, ignore_index=True)
TOT.to_csv(os.path.join(PASTA_G, 'totais_temporada_r5.csv'), index=False)
print('Pasta:', PASTA_G)
for u in (0, 35, 31, 41, 42, 53, 23):
    display(Image(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')))
TOT.round(0)
"""

C9 = r"""
# @title 9 · Temporada mais recente: o que o modelo final previu em junho, por UF
ULT = max(TESTE)
DJ = DET[DET.corte == 'jun']
RF = R5.resumo_temporada(TODOS[TODOS.modelo == RECOMENDADA], D, ULT)
dj = DJ[DJ.season == ULT].assign(uf=lambda x: x.uf_code.map(R2.SIGLAS))[['uf', 'p_grande', 'pred']]
RF = RF.merge(dj.assign(tamanho_x_normal=np.exp(dj.pred))[['uf', 'p_grande', 'tamanho_x_normal']], on='uf', how='left')
R5.salvar_tabela(RF, P['AV5'], f'previsao_temporada_{ULT}_{RECOMENDADA}')
print(f'Temporada {ULT}/{ULT + 1} (SE41 → SE40), prevista na SE25 de {ULT}. Faixa do total = soma dos limites semanais (conservadora).')
RF.round(2)
"""

CELULAS = [('markdown', MD0), ('code', "NOTEBOOK = 'r5_final'\n" + SETUP), ('code', C2), ('code', C3), ('code', C4),
           ('code', C5), ('code', C6), ('code', C7), ('code', C8), ('code', C9)]


def celula(tipo, src):
    src = src.strip('\n')
    c = {'cell_type': tipo, 'metadata': {}, 'source': [l + '\n' for l in src.split('\n')]}
    if tipo == 'code':
        c.update(execution_count=None, outputs=[])
    return c


nb = {'cells': [celula(t, s) for t, s in CELULAS],
      'metadata': {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
      'nbformat': 4, 'nbformat_minor': 0}
with open(os.path.join(AQUI, 'R5_Modelo_Final_Temporada.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('gerado: R5_Modelo_Final_Temporada.ipynb')
