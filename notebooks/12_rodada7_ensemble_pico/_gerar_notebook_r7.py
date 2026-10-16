"""Gera o notebook da rodada 7 (rode: python _gerar_notebook_r7.py).

  R7_Ensemble_Pico.ipynb   CPU comum, ~5–10 min; retoma de onde parou.
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

MD0 = r"""# R7 · Ensemble por nível previsto: membro-base fora do pico, elástico + escala no pico

**Por que:** a R6 mostrou que otimizar cada modelo sozinho só troca erro entre as fases. Mas, no pico, os dois modelos da R5 erram em **sentidos opostos**:

| Teste 2022–2025 | MIX_elastico5 | MIX_escala5 |
|---|---|---|
| Altura do pico | −12% | +12% |
| WIS nas semanas de pico | 2.994 | **2.012** |
| WIS na entressafra | **27,8** | 55,1 |

**O ensemble.** Cada semana × UF recebe, em junho, um nível previsto `z`, na escala dos limiares da UF: 0 = p25 e 1 = p75 da incidência histórica, até 2017. A partir daí:

- **peso do pico:** `s = 1 / (1 + exp(−(z − c)/h))`, uma rampa suave, sem saltos;
- **componente de pico:** `PICO = (1 − a)·elastico5 + a·escala5` (média ponderada dos quantis);
- **ensemble:** `ENS = (1 − s)·BASE + s·PICO`, com `BASE` = normal (ENS_grafos3_λ), AJ_continuo, elastico5 ou escala6.

São só 5 escolhas: `base`, `a`, `c`, `h` e o membro que dá o nível. Elas saem de uma **grade** determinística, com cerca de 2.000 combinações.

**Objetivo (o mesmo da R6, sem critério misto):** minimizar o **WIS do pico ÷ MIX_elastico5**, com média geométrica entre UFs, desde que a entressafra e a intermediária não piorem mais que `TOL` (2%). O próprio elastico5 está na grade, então o ensemble só é escolhido se ganhar dele.

**Validação:**
- **2015–2021:** os membros são gerados pela `r6_lib`, que reproduz a R5. Antes de 2018, o normal é a climatologia.
- **Validação aninhada:** a escolha é refeita deixando cada temporada de fora.
- **Regra de uso, decidida antes do teste:** fora da amostra, o pico precisa cair pelo menos 2% sem estourar a restrição.
- **`ENS_pico7_simples_UF`:** uma versão de referência fixada **a priori**, sem olhar dados: base elastico5, 50/50 no pico, rampa em z = 1.
- **Teste 2022–2025:** olhado uma vez, com as previsões gravadas e as métricas-padrão lado a lado.

**Antes de rodar:**
- Suba `r7_lib.py` (pasta `data_imdc_2026/RODADA7/`) para `MyDrive/dengue_V4/`. O `r6_lib.py` já deve estar lá.
- O notebook usa:
  - `cache_r3/`, `cache_r5/` e `cache_r6/` (melhor_*.json);
  - `resultados_uf_r3/`, `resultados_uf_r5/` e `resultados_uf_r6/`.

**Ambiente:** CPU comum, ~5–10 min. **Retomada:** os membros da validação e a grade ficam em `cache_r7/`. Se cair, rode tudo de novo."""

SETUP = r"""
# @title 1 · Drive, arquivos e ajustes da rodada
import os, sys, shutil, time, glob, json, pickle, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/r7', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'uf_lib.py', 'semanal_lib.py', 'r2_lib.py', 'r3_lib.py',
                 'r4_lib.py', 'r5_lib.py', 'r6_lib.py', 'r7_lib.py', 'dados', 'dados_uf', 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou subir {item} para {PASTA_DRIVE}'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
sys.path.insert(0, PASTA_LOCAL)
P = {k: os.path.join(PASTA_DRIVE, v) for k, v in dict(R7='resultados_uf_r7', C7='cache_r7', AV7='avaliacao_r7', R6='resultados_uf_r6', C6='cache_r6',
     R5='resultados_uf_r5', C5='cache_r5', C4='cache_r4', R3='resultados_uf_r3', C3='cache_r3', AV2='avaliacao_r2',
     COMP='resultados_competicao', EXT='resultados_externos').items()}
for p in (os.path.join(P['R7'], 'uf'), P['C7'], P['AV7']):
    os.makedirs(p, exist_ok=True)
import importlib, v4_lib as V, comp_lib as L, comp_modelos as CM, uf_lib as U, semanal_lib as S, r2_lib as R2, r3_lib as R3, r4_lib as R4, r5_lib as R5, r6_lib as R6, r7_lib as R7
for m in (V, L, CM, U, S, R2, R3, R4, R5, R6, R7):
    importlib.reload(m)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'), os.path.join(PASTA_LOCAL, 'dados_uf'))
D = ctx.Du
TUNING, TESTE = U.TUNING, U.TESTE
REFAZER = False
# ---- ajustes da rodada
VALIDACAO = list(range(2015, 2022))   # temporadas da escolha (antes de 2018 o normal é a climatologia)
M_VAL = 500                           # trajetórias por UF dos membros reimplementados na validação
TOL = 0.02                            # quanto a entressafra e a intermediária podem piorar em relação ao elastico5
GANHO_MIN = 0.02                      # ganho mínimo no pico, fora da amostra, para recomendar o ensemble
"""

C2 = r"""
# @title 2 · Membros na validação (2015–2021), gerados pela r6_lib (gravados em cache_r7)
NORMAL = R4.ler_normal(ctx, PASTA_DRIVE, P['AV2'], P['COMP'], P['EXT'], P['C3'], TUNING + TESTE)
_, DET5 = R5.detector_escolhido5(P['C5'])
X3 = pd.read_parquet(os.path.join(P['C3'], 'detector_variaveis.parquet'))
a4 = os.path.join(P['C4'], 'detector4_variaveis.parquet')
X4 = pd.read_parquet(a4) if os.path.exists(a4) else None
MP = R3.json_le(os.path.join(P['C3'], 'mistura_parametros.json'))
W_R5 = float(R3.json_le(os.path.join(P['C5'], 'mix_elastico_w.json'))['w'])
PAR0 = {'elastico': R6.padrao('elastico', w_r5=W_R5), 'escala': R6.padrao('escala', limiar_r5=float(MP['escala']['par']))}
PAR_AJ = R7.par_aj(MP['continuo']['par'])
MELHOR6 = {m: R3.json_le(os.path.join(P['C6'], f'melhor_{m}.json')) for m in ('elastico', 'escala')}
MELHOR6 = {m: v for m, v in MELHOR6.items() if v}
print('R5: w =', W_R5, '· limiar escala =', round(MP['escala']['par'], 3), '· p do AJ =', MP['continuo']['par'], '· R6 disponível:', list(MELHOR6))
arq = os.path.join(P['C7'], f'membros_validacao_M{M_VAL}.pkl')
if os.path.exists(arq) and not REFAZER:
    with open(arq, 'rb') as f:
        MV, VV, P25, P75 = pickle.load(f)
else:
    t0 = time.time()
    B = R6.Base6(ctx, NORMAL, DET5, X3, X4, VALIDACAO, M=M_VAL)
    MV = R7.membros_validacao(B, PAR0, PAR_AJ, MELHOR6)
    VV = {s: {k: d[k] for k in ('datas', 'y', 'fase', 'pop', 'completa')} for s, d in B.dados.items()}
    P25, P75 = B.p25, B.p75
    with open(arq + '.tmp', 'wb') as f:
        pickle.dump((MV, VV, P25, P75), f)
    os.replace(arq + '.tmp', arq)
    del B
    print(f'membros gerados · {time.time() - t0:.0f}s')
print({s: sorted(m) for s, m in MV.items()})
"""

C3 = r"""
# @title 3 · Diagnóstico: no pico de cada UF × temporada, o real fica ENTRE o elastico5 e o escala5? (validação)
TC, RC = R7.complementaridade(MV, VV, VALIDACAO)
print('2015–2021 (membros reimplementados; erros de altura pela mediana, em %):')
display(pd.Series(RC).round(1).to_frame('valor'))
TC['uf'] = [R2.SIGLAS.get(D.unidades[j], j) for j in TC.j]
R7.R6.salvar_tabela(TC, P['AV7'], 'complementaridade_validacao')
print('Por temporada (mediana entre UFs, %):')
display((100 * TC.groupby('season')[['elastico5', 'escala5', 'media']].median()).round(1))
"""

C4 = r"""
# @title 4 · Grade, validação aninhada e decisão — ANTES de olhar o teste
LISTA = R7.combos() + [R7.SIMPLES]
I_SIMPLES = len(LISTA) - 1
t0 = time.time()
ACC, REF = R7.avaliar_grade(MV, VV, P25, P75, LISTA)
print(f'{len(LISTA)} combinações × {len(MV)} temporadas · {time.time() - t0:.0f}s')
I_MELHOR, GR = R7.escolher(LISTA, ACC, REF, VALIDACAO, TOL)
R6.salvar_tabela(GR, P['AV7'], 'grade_validacao')
print('Melhor combinação em 2015–2021 (dentro da amostra). Fases = WIS ÷ elastico5; erros em %:')
display(GR[GR.i.isin([I_MELHOR, I_SIMPLES])].assign(qual=lambda x: np.where(x.i == I_MELHOR, 'escolhida', 'simples (a priori)')).round(3))
print('Top 10 viáveis:')
display(GR[GR.viavel].sort_values('alta (pico)').head(10).round(3))
ANI, TOT = R7.validacao_aninhada(LISTA, ACC, REF, VALIDACAO, TOL)
R6.salvar_tabela(ANI, P['AV7'], 'validacao_aninhada')
print('\nValidação aninhada (escolha refeita sem a temporada; avaliada nela):')
display(ANI.round(3))
print('TODAS as dobras juntas (fora da amostra):', {k: round(v, 3) for k, v in TOT.items()})
RECOMENDAR = bool(TOT['alta (pico)'] <= 1 - GANHO_MIN and TOT['entressafra'] <= 1 + TOL and TOT['intermediaria'] <= 1 + TOL)
print(f'→ recomendar o ensemble no lugar do MIX_elastico5? {"SIM" if RECOMENDAR else "NÃO"} '
      f'(regra: pico ≤ {1 - GANHO_MIN:.2f}, entressafra e intermediária ≤ {1 + TOL:.2f}, fora da amostra)')
PAR7 = LISTA[I_MELHOR]
R6.json_grava(dict(escolhida=PAR7, simples=R7.SIMPLES, aninhada=TOT, recomendar=RECOMENDAR, tol=TOL, ganho_min=GANHO_MIN, validacao=VALIDACAO),
              os.path.join(P['AV7'], f'ensemble_r7_{R6.carimbo()}.json'))
print('escolhida:', PAR7)
"""

C5 = r"""
# @title 5 · Previsões do ensemble (2018–2025) a partir das previsões GRAVADAS dos membros → resultados_uf_r7
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image, display
R3M = R4.ler_pastas(P['R3']); R5M = R4.ler_pastas(P['R5']); R6M = R4.ler_pastas(P['R6'])
XM = pd.concat([NORMAL, R3M, R5M, R6M], ignore_index=True)
XM = XM[XM.modelo.isin(list(R7.NOMES_ARQ.values()) + ['BB'])].drop_duplicates(['modelo', 'season', 'uf_code', 'date'], keep='last')
MA, VA = R7.membros_arquivo(ctx, XM, TUNING + TESTE, P25, P75)
print('membros por temporada:', {s: sorted(m) for s, m in MA.items()})
NOMES7 = {'ENS_pico7_UF': PAR7, 'ENS_pico7_simples_UF': R7.SIMPLES}
for nome, par in NOMES7.items():
    lin = []
    for s, M in MA.items():
        q = R7.combinar(M, VA[s]['pop'], P25, P75, par)
        if q is None:
            print(f'⚠️ {nome} {s}: falta membro'); continue
        lin.append(R7.para_df(q, VA[s]['datas'], D.unidades, nome, s))
    if not lin:
        print(f'⚠️ {nome} não gerado: falta o membro {par["base"]} nas previsões gravadas'); continue
    R4.grava_modelo(pd.concat(lin, ignore_index=True), P['R7'], nome)
    print(f'✅ {nome}: {sorted(set(pd.concat(lin).season))}')
# a rampa escolhida
z = np.linspace(-1.5, 3, 200)
fig, ax = plt.subplots(figsize=(7, 3.4))
for nome, par in NOMES7.items():
    ax.plot(z, R7.peso_pico(z, par['c'], par['h']), label=f"{nome}: base={par['base']}, a={par['a']}, nível={par['nivel']}")
ax.axvline(0, color='#999', lw=.6, ls=':'); ax.axvline(1, color='#999', lw=.6, ls=':')
ax.text(0, 1.02, 'p25', ha='center', fontsize=8); ax.text(1, 1.02, 'p75', ha='center', fontsize=8)
ax.set_xlabel('nível previsto da semana (0 = p25 da UF, 1 = p75)'); ax.set_ylabel('peso do componente de pico')
ax.legend(fontsize=7, frameon=False, loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); arq = os.path.join(P['AV7'], 'rampa_peso_pico.png'); fig.savefig(arq, dpi=110); plt.close(fig)
display(Image(arq))
"""

C6 = r"""
# @title 6 · TESTE 2022–2025 (olhado uma vez): métricas-padrão lado a lado
R7M = R4.ler_pastas(P['R7'])
MOD = ['ENS_pico7_UF', 'ENS_pico7_simples_UF', 'MIX_elastico5_UF', 'MIX_escala5_UF', 'MIX_escala6_UF', 'AJ_continuo_UF', 'ENS_grafos3_λ', 'BB']
TODOS = pd.concat([XM, R7M], ignore_index=True)
TODOS = TODOS[TODOS.modelo.isin(MOD)].drop_duplicates(['modelo', 'season', 'uf_code', 'date'], keep='last')
OF = TODOS[TODOS.season.isin(TESTE)]
_, RES = V.avaliar(OF[L.COLS_PREV], ctx.ver_uf, ref=('BB',))
RES = RES[RES.temporadas == len(TESTE)]
TT = R2.com_fase(S.com_verdade(OF, D), D, R2.limiares(D))
pont = pd.DataFrame([dict(modelo=m, **S.metricas(g)) for m, g in TT.groupby('modelo')])
FASE = R2.metricas_fase(TT, ['modelo'])
f1 = FASE.pivot_table(index='modelo', columns='fase', values=['WIS', 'vies_pct'])
f1.columns = [f'{a}_{b}' for a, b in f1.columns]
PICO = R2.erros_pico(TT, 'season')
TAB = RES[['modelo', 'R_vs_BB', 'UFs_melhor_que_BB']].merge(pont[['modelo', 'WIS', 'MAE', 'WAPE', 'sMAPE', 'IC90']], on='modelo') \
    .merge(f1.reset_index(), on='modelo', how='left').merge(PICO, on='modelo', how='left')
TAB['ordem'] = TAB.modelo.map({m: i for i, m in enumerate(MOD)}); TAB = TAB.sort_values('ordem').drop(columns='ordem')
RZ = R6.razoes_longo(TT, [m for m in MOD if m != 'MIX_elastico5_UF' and m != 'BB'], 'MIX_elastico5_UF')
_, RCT = R7.complementaridade({s: MA[s] for s in TESTE if s in MA}, {s: VA[s] for s in TESTE if s in VA}, TESTE)
for nome, t in (('teste_r7', TAB), ('teste_r7_por_fase', FASE), ('teste_r7_pico', PICO), ('teste_r7_razao_vs_elastico5', RZ)):
    R6.salvar_tabela(t, P['AV7'], nome)
cols = ['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'WIS', 'WIS_alta (pico)', 'WIS_entressafra', 'WIS_intermediaria', 'vies_pct_alta (pico)',
        'vies_pct_entressafra', 'MAE', 'WAPE', 'sMAPE', 'erro_altura_mediano_pct', 'erro_total_mediano_pct', 'pico_no_IC90_pct', 'IC90']
print('TESTE 2022–2025 · R_vs_BB < 1 = melhor que o modelo-base oficial; WIS e MAE em casos/semana; WAPE e erros em %')
display(TAB[[c for c in cols if c in TAB]].round(3))
print('WIS por fase ÷ MIX_elastico5 (média geométrica entre UFs; < 1 = melhor que o elastico5):')
display(RZ.round(3))
print('Complementaridade no teste (temporadas completas):', {k: round(v, 1) for k, v in RCT.items()})
"""

C7 = r"""
# @title 7 · Gráficos (26 UFs + Brasil): ensemble × membros
from matplotlib.backends.backend_pdf import PdfPages
PASTA_G = os.path.join(P['AV7'], f'graficos_r7_{R6.carimbo()[:8]}')
os.makedirs(PASTA_G, exist_ok=True)
OBS = pd.DataFrame(D.Y[:D.ultimo_obs + 1], index=pd.to_datetime(D.datas[:D.ultimo_obs + 1]), columns=D.unidades)
OBS[0] = OBS.sum(1)
PREV = {'ENS_pico7 (R7, leque)': TODOS[TODOS.modelo == 'ENS_pico7_UF'],
        'MIX_elastico5 (R5)': TODOS[TODOS.modelo == 'MIX_elastico5_UF'],
        'MIX_escala5 (R5)': TODOS[TODOS.modelo == 'MIX_escala5_UF'],
        'ENS_pico7_simples': TODOS[TODOS.modelo == 'ENS_pico7_simples_UF']}
ORDEM = [0] + sorted(D.unidades, key=lambda u: R2.SIGLAS.get(u, str(u)))
for u in ORDEM:
    arq = os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')
    if not os.path.exists(arq) or REFAZER:
        fig = R3.grafico_temporada_uf(OBS[u], PREV, u, arq, TESTE, titulo=' · ensemble R7' + (' · Brasil = soma das UFs' if u == 0 else ''))
        plt.close(fig)
with PdfPages(os.path.join(PASTA_G, 'graficos_UFs_r7.pdf.tmp')) as pdf:
    for u in ORDEM:
        img = plt.imread(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png'))
        f = plt.figure(figsize=(13, 4.6)); plt.imshow(img); plt.axis('off'); pdf.savefig(f, bbox_inches='tight'); plt.close(f)
os.replace(os.path.join(PASTA_G, 'graficos_UFs_r7.pdf.tmp'), os.path.join(PASTA_G, 'graficos_UFs_r7.pdf'))
print('Pasta:', PASTA_G)
for u in (0, 35, 31, 41, 42, 53, 23):
    display(Image(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')))
"""

C8 = r"""
# @title 8 · Temporada mais recente: o que o ensemble previu em junho, por UF
ULT = max(TESTE)
for nome in ('ENS_pico7_UF', 'ENS_pico7_simples_UF'):
    RF = R5.resumo_temporada(TODOS[TODOS.modelo == nome], D, ULT)
    R6.salvar_tabela(RF, P['AV7'], f'previsao_temporada_{ULT}_{nome}')
    print(f'\n{nome} · temporada {ULT}/{ULT + 1} (SE41 → SE40), prevista na SE25 de {ULT}. Faixa do total = soma dos limites semanais (conservadora).')
    display(RF.round(0))
"""

CELULAS = [('markdown', MD0), ('code', "NOTEBOOK = 'r7_ensemble'\n" + SETUP), ('code', C2), ('code', C3), ('code', C4),
           ('code', C5), ('code', C6), ('code', C7), ('code', C8)]


def celula(tipo, src):
    src = src.strip('\n')
    c = {'cell_type': tipo, 'metadata': {}, 'source': [l + '\n' for l in src.split('\n')]}
    if tipo == 'code':
        c.update(execution_count=None, outputs=[])
    return c


if __name__ == '__main__':
    nb = {'cells': [celula(t, s) for t, s in CELULAS],
          'metadata': {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
          'nbformat': 4, 'nbformat_minor': 0}
    with open(os.path.join(AQUI, 'R7_Ensemble_Pico.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('gerado: R7_Ensemble_Pico.ipynb')
