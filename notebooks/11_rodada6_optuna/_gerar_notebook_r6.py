"""Gera o notebook da rodada 6 (rode: python _gerar_notebook_r6.py).

  R6_Otimizacao_Pico.ipynb   CPU comum, ~45 min–1 h na 1ª vez (Optuna); retoma de onde parou.
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

MD0 = r"""# R6 · Otimização do pico: MIX_elastico e MIX_escala

**Ponto de partida (R5, teste 2022–2025):**

| Modelo | Altura do pico | Total | Viés na entressafra |
|---|---|---|---|
| MIX_elastico5_UF | **−12%** (fica baixo) | −8% | +85% |
| MIX_escala5_UF | +12% | +9% | **+151%** (infla) |

**Objetivo:**
- o **MIX_elastico** deve subir no pico sem estragar o resto;
- o **MIX_escala** deve parar de inflar a entressafra sem perder o pico.

## O que o notebook faz

1. **Os dois modelos são reescritos em `r6_lib.py` com todos os parâmetros abertos.** Os valores padrão reproduzem a R5, e a célula 2 confere isso. A tabela completa de parâmetros sai na célula 3.
   - **MIX_elastico:**
     - `w`: peso do tamanho previsto;
     - faixas de nível da elasticidade;
     - elasticidade estimada só com os anos acima do normal;
     - teto da elasticidade;
     - expoente `γ` (concentra o crescimento no pico);
     - afinação da forma `η` nos anos grandes (pico mais alto e estreito, mesmo total);
     - antecipação do pico nos anos grandes.
   - **MIX_escala:**
     - limiar;
     - troca suave `τ`, no lugar de liga/desliga;
     - expoente `p` do fator;
     - nível `L` abaixo do qual o fator tem teto;
     - o teto.
2. **Um detector para cada modelo.** Cada estudo escolhe o seu:
   - o detector da R5 como está;
   - ou um ridge reajustado aqui, com fonte das variáveis (R3 ou R3 + R4 sem as nacionais novas), janela do normal da UF, `k`, ρ mínimo, α e peso dos anos grandes.

   Nos dois casos entram o alargamento da faixa, a correção de viés `β` e a cauda do sorteio. O reajuste é causal: escolhe as variáveis e treina só com temporadas terminadas antes de junho.
3. **Objetivo sem critério misto.** Cada número é o WIS do modelo na fase dividido pelo WIS da R5 na mesma UF, com média geométrica entre UFs (cada UF pesa igual). Por exemplo, **0,85 = 15% menos erro que a R5 naquela fase**.
   - **MIX_elastico:** minimizar o WIS das semanas de **pico**.
   - **MIX_escala:** minimizar o WIS das semanas de **entressafra**.
   - **Restrição nos dois:** as outras fases (e a intermediária) não podem piorar mais que `TOL` (5%).

   As fases são definidas pelo observado: pico ≥ p75 da UF, entressafra < p25 (limiares até 2017).
4. **Validação planejada para não sobreajustar.**
   - **Busca em 2015–2021:** 7 temporadas, com os anos grandes de 2015, 2016 e 2019. Antes de 2018, o normal é a climatologia, porque o ENS_grafos3_λ só existe a partir daí. A R5 de referência é calculada no mesmo normal, então a comparação é justa.
   - **Validação aninhada:** a busca inteira é refeita 7 vezes, deixando uma temporada de fora em cada uma, e avaliada na temporada que ficou de fora. Isso mede o ganho real fora da amostra.
   - **Regra de uso, decidida antes do teste:** o modelo otimizado só é recomendado se, fora da amostra, ganhar ao menos 2% na fase-alvo sem estourar a restrição.
   - **Teste 2022–2025:** olhado uma vez, no fim, com as métricas-padrão lado a lado (R × BB, WIS por fase, MAE, WAPE, sMAPE, erro do pico e do total, cobertura).

**Antes de rodar:**
- Suba `r6_lib.py` (pasta `data_imdc_2026/RODADA6/`) para `MyDrive/dengue_V4/`.
- O notebook usa o que as rodadas 3 e 5 gravaram: `cache_r3/` (detector e normal), `cache_r5/` (detector escolhido e `w`), `resultados_uf_r5/` e `resultados_uf_r3/`.
- `cache_r4/detector4_variaveis.parquet` é opcional: sem ele, o detector reajustado usa só as variáveis da R3.

**Ambiente:** CPU comum. Leva cerca de 45 min–1 h na primeira vez, quase tudo no Optuna do MIX_elastico (~1,5 s por tentativa; o MIX_escala leva ~0,15 s). Se quiser ir mais rápido, baixe `N_ANINHADA` na célula 1.

**Retomada:** cada estudo do Optuna fica gravado em `cache_r6/*.db` e é retomado de onde parou. As dobras da validação aninhada e as previsões finais também ficam gravadas. Se cair, **rode tudo de novo**."""

SETUP = r"""
# @title 1 · Drive, arquivos, pacotes e ajustes da rodada
import os, sys, shutil, time, glob, subprocess, json, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/r6', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optuna'], check=True)
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'uf_lib.py', 'semanal_lib.py', 'r2_lib.py', 'r3_lib.py',
                 'r4_lib.py', 'r5_lib.py', 'r6_lib.py', 'dados', 'dados_uf', 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou subir {item} para {PASTA_DRIVE}'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
sys.path.insert(0, PASTA_LOCAL)
P = {k: os.path.join(PASTA_DRIVE, v) for k, v in dict(R6='resultados_uf_r6', C6='cache_r6', AV6='avaliacao_r6', R5='resultados_uf_r5',
     C5='cache_r5', C4='cache_r4', R3='resultados_uf_r3', C3='cache_r3', AV2='avaliacao_r2', COMP='resultados_competicao',
     EXT='resultados_externos').items()}
for p in (os.path.join(P['R6'], 'uf'), P['C6'], P['AV6']):
    os.makedirs(p, exist_ok=True)
import importlib, v4_lib as V, comp_lib as L, comp_modelos as CM, uf_lib as U, semanal_lib as S, r2_lib as R2, r3_lib as R3, r4_lib as R4, r5_lib as R5, r6_lib as R6
for m in (V, L, CM, U, S, R2, R3, R4, R5, R6):
    importlib.reload(m)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'), os.path.join(PASTA_LOCAL, 'dados_uf'))
D = ctx.Du
TUNING, TESTE = U.TUNING, U.TESTE
REFAZER = False
# ---- ajustes da rodada
VALIDACAO = list(range(2015, 2022))   # temporadas da busca (antes de 2018 o normal é a climatologia)
M_BUSCA = 200                         # trajetórias por UF na busca (os mesmos números aleatórios em todas as tentativas)
N_FINAL = 400                         # tentativas do Optuna na busca final, por modelo
N_ANINHADA = 120                      # tentativas por dobra na validação aninhada (7 dobras por modelo)
TOL = 0.05                            # quanto as outras fases podem piorar em relação à R5 (5%)
GANHO_MIN = 0.02                      # ganho mínimo fora da amostra na fase-alvo para recomendar a versão otimizada
"""

C2 = r"""
# @title 2 · Dados, parâmetros da R5 e conferência da reimplementação
NORMAL = R4.ler_normal(ctx, PASTA_DRIVE, P['AV2'], P['COMP'], P['EXT'], P['C3'], TUNING + TESTE)
_, DET5 = R5.detector_escolhido5(P['C5'])
X3 = pd.read_parquet(os.path.join(P['C3'], 'detector_variaveis.parquet'))
a4 = os.path.join(P['C4'], 'detector4_variaveis.parquet')
X4 = pd.read_parquet(a4) if os.path.exists(a4) else None
W_R5 = float(R3.json_le(os.path.join(P['C5'], 'mix_elastico_w.json'))['w'])
LIMIAR_R5 = float(R3.json_le(os.path.join(P['C3'], 'mistura_parametros.json'))['escala']['par'])
PAR0 = {'elastico': R6.padrao('elastico', w_r5=W_R5), 'escala': R6.padrao('escala', limiar_r5=LIMIAR_R5)}
print(f'R5: w do MIX_elastico = {W_R5} · limiar do MIX_escala = {LIMIAR_R5:.3f} (×{np.exp(LIMIAR_R5):.1f} o normal)')
B = R6.Base6(ctx, NORMAL, DET5, X3, X4, VALIDACAO, M=M_BUSCA)
# conferência: com os parâmetros da R5, a reimplementação reproduz as previsões gravadas na R5?
R5M = R4.ler_pastas(P['R5'])
conf = []
for modelo, nome in (('elastico', 'MIX_elastico5_UF'), ('escala', 'MIX_escala5_UF')):
    for s in [s for s in TUNING if s in B.dados]:
        g = R5M[(R5M.modelo == nome) & (R5M.season == s)]
        if g.empty:
            continue
        q = B.prever(s, modelo, R6.completa(PAR0[modelo]))
        q5 = V.de_df(g, B.dados[s]['datas'], D.unidades)
        rel = np.abs(q[..., 4] - q5[..., 4]) / (q5[..., 4] + 10)
        conf.append(dict(modelo=nome, temporada=s, dif_mediana_pct=100 * np.nanmedian(rel), dif_p90_pct=100 * np.nanpercentile(rel, 90),
                         total_R6_sobre_R5=np.nansum(q[..., 4]) / np.nansum(q5[..., 4])))
print('reimplementação × R5 gravada (diferença na mediana; a R5 usou 1.000 trajetórias, a busca usa', M_BUSCA, '):')
display(pd.DataFrame(conf).round(3))
dt = {}
for m in PAR0:
    t0 = time.time(); B.acumula(m, R6.completa(PAR0[m]), VALIDACAO); dt[m] = time.time() - t0
n_tent = N_FINAL + N_ANINHADA * (len(VALIDACAO) - 1)          # a validação aninhada usa 6 das 7 temporadas por dobra
print('uma tentativa ≈ ' + ' · '.join(f'{m} {v:.1f}s' for m, v in dt.items()) + f' · estimativa total do Optuna ≈ {n_tent * sum(dt.values()) / 60:.0f} min')
"""

C3 = r"""
# @title 3 · Busca final (Optuna, 2015–2021): melhor configuração de cada modelo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image, display
EST, MELHOR, TENT = {}, {}, {}
for modelo in ('elastico', 'escala'):
    t0 = time.time()
    st = R6.otimizar(B, modelo, VALIDACAO, f'r6_{modelo}', P['C6'], N_FINAL, PAR0[modelo], TOL)
    par, v, t = R6.escolher(st, PAR0[modelo])
    EST[modelo], MELHOR[modelo], TENT[modelo] = st, par, t
    R6.json_grava(par, os.path.join(P['C6'], f'melhor_{modelo}.json'))
    R6.salvar_tabela(R6.tentativas(st), P['AV6'], f'tentativas_{modelo}')
    alvo, restr = R6.OBJETIVO[modelo]
    print(f'\n=== MIX_{modelo} · {len(R6.concluidas(st))} tentativas · {time.time() - t0:.0f}s')
    print(f'objetivo = WIS de "{alvo}" ÷ R5 → {v:.3f}  (restrição: {", ".join(restr)} ≤ {1 + TOL:.2f})')
    display(R6.tabela_parametros(PAR0[modelo], par))
# razões da melhor configuração na busca, por grupo de temporadas (normal = climatologia × ENS_grafos3_λ)
GRUPOS = {'2015–2017 (normal = climatologia)': [s for s in VALIDACAO if s < 2018],
          '2018–2021 (normal = ENS_grafos3_λ)': [s for s in VALIDACAO if s >= 2018], 'todas': VALIDACAO}
lin = []
for modelo in MELHOR:
    for gnome, ss in GRUPOS.items():
        if ss:
            r = R6.razoes(B.acumula(modelo, MELHOR[modelo], ss), B.acumula(modelo, R6.completa(PAR0[modelo]), ss))
            r0 = R6.razoes(B.acumula(modelo, R6.completa(PAR0[modelo]), ss), B.acumula(modelo, R6.completa(PAR0[modelo]), ss))
            lin.append(dict(modelo=modelo, temporadas=gnome, **r, R5_erro_altura_pct=r0['erro_altura_mediano_pct'],
                            R5_erro_total_pct=r0['erro_total_mediano_pct']))
BUSCA = pd.DataFrame(lin)
R6.salvar_tabela(BUSCA, P['AV6'], 'busca_melhor_vs_R5')
print('\nNa busca (DENTRO da amostra; a validação honesta é a próxima célula). Fases = WIS ÷ R5; erros em %:')
display(BUSCA.round(3))
# fronteira pico × entressafra de todas as tentativas
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, modelo in zip(axs, ('elastico', 'escala')):
    TR = R6.tentativas(EST[modelo])
    ax.scatter(TR['entressafra'], TR['alta (pico)'], s=12, c=np.where(TR.viavel, '#2a6fdb', '#c8c8c8'), label='tentativas (azul = viável)')
    ax.scatter([1], [1], marker='s', s=60, color='#1a1a1a', label='R5')
    if TENT[modelo] is not None:
        ax.scatter([TENT[modelo].user_attrs['entressafra']], [TENT[modelo].user_attrs['alta (pico)']], marker='*', s=220, color='#c0392b', label='escolhida')
    ax.axhline(1, color='#999', lw=.6); ax.axvline(1, color='#999', lw=.6)
    if modelo == 'elastico':
        ax.axvline(1 + TOL, color='#c0392b', lw=.8, ls='--')     # restrição na entressafra
    else:
        ax.axhline(1 + TOL, color='#c0392b', lw=.8, ls='--')     # restrição no pico
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.5, 2); ax.set_ylim(0.5, 2)   # pontos fora dessa faixa ficam de fora do quadro
    ax.set_xticks([0.5, 0.7, 1, 1.4, 2]); ax.set_yticks([0.5, 0.7, 1, 1.4, 2])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('WIS da entressafra ÷ R5 (menor = melhor)'); ax.set_ylabel('WIS do pico ÷ R5 (menor = melhor)')
    ax.set_title(f'MIX_{modelo} · tracejado = limite da restrição ({1 + TOL:.2f})', loc='left', fontsize=10); ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
arq = os.path.join(P['AV6'], 'fronteira_pico_entressafra.png'); fig.savefig(arq, dpi=110); plt.close(fig)
display(Image(arq))
"""

C4 = r"""
# @title 4 · Validação aninhada (cada temporada fora da busca) e decisão — ANTES de olhar o teste
ANI, USAR = {}, {}
for modelo in ('elastico', 'escala'):
    t0 = time.time()
    T, tot = R6.validacao_aninhada(B, modelo, VALIDACAO, P['C6'], N_ANINHADA, PAR0[modelo], TOL)
    R6.salvar_tabela(T, P['AV6'], f'validacao_aninhada_{modelo}')
    ANI[modelo] = tot
    alvo, restr = R6.OBJETIVO[modelo]
    USAR[modelo] = bool(tot[alvo] <= 1 - GANHO_MIN and all(tot[r] <= 1 + TOL for r in restr))
    print(f'\n=== MIX_{modelo} · {time.time() - t0:.0f}s · por temporada deixada de fora (fases = WIS ÷ R5):')
    display(T.round(3))
    print('TODAS as dobras juntas (fora da amostra):', {k: round(v, 3) for k, v in tot.items()})
    print(f'→ recomendar a versão otimizada? {"SIM" if USAR[modelo] else "NÃO"}  '
          f'(regra: {alvo} ≤ {1 - GANHO_MIN:.2f} e {", ".join(restr)} ≤ {1 + TOL:.2f} fora da amostra)')
"""

C5 = r"""
# @title 5 · Previsões finais (1.000 trajetórias) das temporadas 2018–2025 em resultados_uf_r6
NOMES = {'elastico': 'MIX_elastico6_UF', 'escala': 'MIX_escala6_UF'}
falta = {m: [s for s in TUNING + TESTE if REFAZER or not os.path.exists(L._arq(P['R6'], 'uf', NOMES[m], s))] for m in NOMES}
ss = sorted({s for v in falta.values() for s in v})
if ss:
    t0 = time.time()
    B2 = R6.Base6(ctx, NORMAL, DET5, X3, X4, ss, M=1000)
    for m, sm in falta.items():
        for s in sm:
            R4.grava_modelo(B2.para_df(s, B2.prever(s, m, MELHOR[m]), NOMES[m]), P['R6'], NOMES[m])
        print(f'✅ {NOMES[m]} {sm} · {time.time() - t0:.0f}s', flush=True)
    del B2
R6.json_grava(dict(parametros=MELHOR, R5=PAR0, aninhada=ANI, recomendar=USAR, tol=TOL, ganho_min=GANHO_MIN, validacao=VALIDACAO),
              os.path.join(P['AV6'], f'modelo_r6_{R6.carimbo()}.json'))
print('recomendação (decidida antes do teste):', USAR)
"""

C6 = r"""
# @title 6 · TESTE 2022–2025 (olhado uma vez): métricas-padrão lado a lado
R3M = R4.ler_pastas(P['R3']); R5M = R4.ler_pastas(P['R5']); R6M = R4.ler_pastas(P['R6'])
MOD = ['MIX_elastico6_UF', 'MIX_elastico5_UF', 'MIX_escala6_UF', 'MIX_escala5_UF', 'ENS_grafos3_λ', 'AJ_continuo_UF', 'BB']
TODOS = pd.concat([NORMAL, R3M[R3M.modelo == 'AJ_continuo_UF'], R5M[R5M.modelo.isin(MOD)], R6M], ignore_index=True)
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
RZ = pd.concat([R6.razoes_longo(TT, ['MIX_elastico6_UF'], 'MIX_elastico5_UF'), R6.razoes_longo(TT, ['MIX_escala6_UF'], 'MIX_escala5_UF')])
for nome, t in (('teste_r6', TAB), ('teste_r6_por_fase', FASE), ('teste_r6_pico', PICO), ('teste_r6_razao_vs_R5', RZ)):
    R6.salvar_tabela(t, P['AV6'], nome)
cols = ['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'WIS', 'WIS_alta (pico)', 'WIS_entressafra', 'vies_pct_alta (pico)',
        'vies_pct_entressafra', 'MAE', 'WAPE', 'sMAPE', 'erro_altura_mediano_pct', 'erro_total_mediano_pct', 'pico_no_IC90_pct', 'IC90']
print('TESTE 2022–2025 · R_vs_BB < 1 = melhor que o modelo-base oficial; WIS, MAE em casos/semana; WAPE e erros em %')
display(TAB[[c for c in cols if c in TAB]].round(3))
print('Versão R6 ÷ R5 por fase (WIS, média geométrica entre UFs; < 1 = R6 melhor):')
display(RZ.round(3))
"""

C7 = r"""
# @title 7 · Gráficos (26 UFs + Brasil): R6 × R5 dos dois modelos
from matplotlib.backends.backend_pdf import PdfPages
PASTA_G = os.path.join(P['AV6'], f'graficos_r6_{R6.carimbo()[:8]}')
os.makedirs(PASTA_G, exist_ok=True)
OBS = pd.DataFrame(D.Y[:D.ultimo_obs + 1], index=pd.to_datetime(D.datas[:D.ultimo_obs + 1]), columns=D.unidades)
OBS[0] = OBS.sum(1)
PREV = {'MIX_elastico6 (R6, leque)': TODOS[TODOS.modelo == 'MIX_elastico6_UF'],
        'MIX_elastico5 (R5)': TODOS[TODOS.modelo == 'MIX_elastico5_UF'],
        'MIX_escala5 (R5)': TODOS[TODOS.modelo == 'MIX_escala5_UF'],
        'MIX_escala6 (R6)': TODOS[TODOS.modelo == 'MIX_escala6_UF']}
ORDEM = [0] + sorted(D.unidades, key=lambda u: R2.SIGLAS.get(u, str(u)))
for u in ORDEM:
    arq = os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')
    if not os.path.exists(arq) or REFAZER:
        fig = R3.grafico_temporada_uf(OBS[u], PREV, u, arq, TESTE, titulo=' · R6 × R5' + (' · Brasil = soma das UFs' if u == 0 else ''))
        plt.close(fig)
with PdfPages(os.path.join(PASTA_G, 'graficos_UFs_r6.pdf.tmp')) as pdf:
    for u in ORDEM:
        img = plt.imread(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png'))
        f = plt.figure(figsize=(13, 4.6)); plt.imshow(img); plt.axis('off'); pdf.savefig(f, bbox_inches='tight'); plt.close(f)
os.replace(os.path.join(PASTA_G, 'graficos_UFs_r6.pdf.tmp'), os.path.join(PASTA_G, 'graficos_UFs_r6.pdf'))
print('Pasta:', PASTA_G)
for u in (0, 35, 31, 41, 42, 53, 23):
    display(Image(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')))
"""

C8 = r"""
# @title 8 · Temporada mais recente: o que cada versão previu em junho, por UF
ULT = max(TESTE)
for nome in ('MIX_elastico6_UF', 'MIX_escala6_UF'):
    RF = R5.resumo_temporada(TODOS[TODOS.modelo == nome], D, ULT)
    R6.salvar_tabela(RF, P['AV6'], f'previsao_temporada_{ULT}_{nome}')
    print(f'\n{nome} · temporada {ULT}/{ULT + 1} (SE41 → SE40), prevista na SE25 de {ULT}. Faixa do total = soma dos limites semanais (conservadora).')
    display(RF.round(0))
"""

CELULAS = [('markdown', MD0), ('code', "NOTEBOOK = 'r6_otimizacao'\n" + SETUP), ('code', C2), ('code', C3), ('code', C4),
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
    with open(os.path.join(AQUI, 'R6_Otimizacao_Pico.ipynb'), 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('gerado: R6_Otimizacao_Pico.ipynb')
