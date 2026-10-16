"""Gera R5_Graficos_Comparacao.ipynb: os melhores modelos recentes lado a lado, por UF (rode: python _gerar_notebook_graficos.py)."""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

MD = r"""# R5 · Comparação visual dos melhores modelos de temporada, por UF

CPU comum, ~5–10 min. Só lê as previsões que já estão no Drive (rodadas 2, 3 e 5); não treina nada.

**O que sai em `avaliacao_r5/comparacao_<data>/`:**
1. **Uma figura por UF (e Brasil)**, em duas partes:
   - **em cima**, a curva semanal observada (preta) e a mediana prevista em junho por 6 modelos, nas temporadas 2022/23 a 2025/26;
   - **embaixo**, a altura do pico de cada temporada: barras dos modelos contra o pico real (linha preta).
2. **`erro_pico_por_UF.csv`** e um **mapa de calor**: erro da altura do pico (%) por UF × modelo, na mediana das 4 temporadas. Negativo = previu abaixo.
3. **`metricas_por_UF.csv`**: MAE, WAPE e erro do total por UF × modelo.
4. **PDF** com tudo.

Todas as previsões foram feitas na SE25 (junho), antes da temporada começar."""

SETUP = r"""
# @title 1 · Drive e dados
import os, sys, shutil, glob
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/r5g', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'uf_lib.py', 'semanal_lib.py', 'r2_lib.py', 'r3_lib.py',
                 'r4_lib.py', 'r5_lib.py', 'dados', 'dados_uf', 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
sys.path.insert(0, PASTA_LOCAL)
import v4_lib as V, comp_lib as L, uf_lib as U, semanal_lib as S, r2_lib as R2, r3_lib as R3, r4_lib as R4
P = {k: os.path.join(PASTA_DRIVE, v) for k, v in dict(R5='resultados_uf_r5', R3='resultados_uf_r3', C3='cache_r3',
     AV2='avaliacao_r2', AV5='avaliacao_r5', COMP='resultados_competicao', EXT='resultados_externos').items()}
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'), os.path.join(PASTA_LOCAL, 'dados_uf'))
D = ctx.Du
TESTE = U.TESTE
# modelos comparados (nome no arquivo → rótulo, cor). Troque/acrescente à vontade.
MODELOS = {
    'ENS_grafos3_λ':      ('normal · ENS_grafos3_λ (R2)', '#2a6fdb'),
    'AJ_continuo_UF':     ('AJ_continuo (R3)', '#8e44ad'),
    'MIX_elastico5_UF':   ('MIX_elastico5 (R5)', '#c0392b'),
    'MIX_escala5_UF':     ('MIX_escala5 (R5)', '#e67e22'),
    'ENS_fase_nivMIX_UF': ('ENS_fase_nivMIX (R5)', '#16a085'),
    'BB':                 ('BB · modelo-base oficial', '#7f8c8d'),
}
NORMAL = R4.ler_normal(ctx, PASTA_DRIVE, P['AV2'], P['COMP'], P['EXT'], P['C3'], TESTE)
X = pd.concat([NORMAL, R4.ler_pastas(P['R3']), R4.ler_pastas(P['R5'])], ignore_index=True)
X = X[X.modelo.isin(MODELOS) & X.season.isin(TESTE)].drop_duplicates(['modelo', 'season', 'uf_code', 'date'])
print('modelos encontrados:', sorted(X.modelo.unique()), '· faltando:', [m for m in MODELOS if m not in set(X.modelo)])
OBS = pd.DataFrame(D.Y[:D.ultimo_obs + 1], index=pd.to_datetime(D.datas[:D.ultimo_obs + 1]), columns=D.unidades)
OBS[0] = OBS.sum(1)
PASTA_G = os.path.join(P['AV5'], f'comparacao_{R4.carimbo()[:8]}')
os.makedirs(PASTA_G, exist_ok=True)
"""

C2 = r"""
# @title 2 · Pico, total e erros por UF × modelo × temporada
def serie(df, u):
    return R2._serie(df, u)['pred'] if len(df) else pd.Series(dtype=float)

lin = []
for u in [0] + list(D.unidades):
    for s in TESTE:
        ini, fim = pd.Timestamp(f'{s}-10-01'), pd.Timestamp(f'{s + 1}-10-15')
        o = OBS[u][(OBS.index >= ini) & (OBS.index <= fim)]
        for m in MODELOS:
            p = serie(X[(X.modelo == m) & (X.season == s)], u)
            if not len(p):
                continue
            oo = o.reindex(p.index)
            ok = oo.notna()
            if ok.sum() < 8:
                continue
            lin.append(dict(uf=R2.SIGLAS.get(u, 'Brasil'), uf_code=u, season=s, modelo=m,
                            pico_real=oo[ok].max(), pico_prev=p[ok].max(),
                            total_real=oo[ok].sum(), total_prev=p[ok].sum(),
                            MAE=float(np.abs(p[ok] - oo[ok]).mean()),
                            semanas=int(ok.sum())))
E = pd.DataFrame(lin)
E['erro_pico_pct'] = 100 * (E.pico_prev - E.pico_real) / E.pico_real.clip(lower=1)
E['erro_total_pct'] = 100 * (E.total_prev - E.total_real) / E.total_real.clip(lower=1)
E['WAPE_pct'] = 100 * E.MAE * E.semanas / E.total_real.clip(lower=1)
ULT_OBS = OBS.index.max()
PARCIAL = [s for s in TESTE if ULT_OBS < pd.Timestamp(f'{s + 1}-09-30')]
E['temporada_completa'] = ~E.season.isin(PARCIAL)
E.to_csv(os.path.join(PASTA_G, 'metricas_por_UF_temporada.csv'), index=False)
print('temporadas ainda em andamento (fora do mapa e das médias):', PARCIAL, '· último dado:', ULT_OBS.date())
M = E[E.temporada_completa].groupby(['uf', 'modelo']).agg(erro_pico_mediano_pct=('erro_pico_pct', 'median'), erro_total_mediano_pct=('erro_total_pct', 'median'),
                                    MAE=('MAE', 'mean'), WAPE_pct=('WAPE_pct', 'mean')).reset_index()
M.to_csv(os.path.join(PASTA_G, 'metricas_por_UF.csv'), index=False)
# quem chegou mais perto do pico real em cada UF (menor |erro| mediano)
melhor = M.loc[M.groupby('uf').erro_pico_mediano_pct.apply(lambda v: v.abs().idxmin())][['uf', 'modelo', 'erro_pico_mediano_pct']]
print('Quantas UFs cada modelo foi o que mais se aproximou do pico real:')
melhor.modelo.map(lambda m: MODELOS[m][0]).value_counts()
"""

C3 = r"""
# @title 3 · Mapa de calor: erro da altura do pico (%) por UF × modelo (mediana das temporadas 2022–2025)
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display
H = M.pivot(index='uf', columns='modelo', values='erro_pico_mediano_pct')[[m for m in MODELOS if m in set(M.modelo)]]
H = H.loc[['Brasil'] + sorted(i for i in H.index if i != 'Brasil')]
fig, ax = plt.subplots(figsize=(10, 11))
im = ax.imshow(H.values.clip(-100, 100), cmap='RdBu_r', vmin=-100, vmax=100, aspect='auto')
ax.set_xticks(range(H.shape[1])); ax.set_xticklabels([MODELOS[m][0] for m in H.columns], rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index, fontsize=9)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f'{v:+.0f}', ha='center', va='center', fontsize=7.5, color='white' if abs(v) > 60 else '#222')
ax.set_title('Erro da altura do pico previsto em junho (%, mediana das temporadas completas)\nazul = previu abaixo · vermelho = acima · 0 = acertou', loc='left', fontsize=11)
fig.colorbar(im, ax=ax, shrink=.5, label='erro do pico (%)')
fig.tight_layout()
arq_mapa = os.path.join(PASTA_G, '00_mapa_erro_pico.png')
fig.savefig(arq_mapa, dpi=120); plt.close(fig)
display(Image(arq_mapa))
"""

C4 = r"""
# @title 4 · Uma figura por UF (curvas + picos por temporada) e PDF
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'.replace(',', '.'))
def figura(u):
    nome = R2.SIGLAS.get(u, 'Brasil')
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw=dict(height_ratios=[3, 1.6]))
    ini, fim = pd.Timestamp(f'{TESTE[0]}-10-01'), pd.Timestamp(f'{TESTE[-1] + 1}-10-15')
    o = OBS[u][(OBS.index >= ini) & (OBS.index <= fim)]
    a1.fill_between(o.index, 0, o.values, color='#dddddd', lw=0)
    a1.plot(o.index, o.values, color='#111111', lw=1.6, label='casos observados')
    for m, (rot, cor) in MODELOS.items():
        for k, s in enumerate(TESTE):
            p = serie(X[(X.modelo == m) & (X.season == s)], u)
            if len(p):
                a1.plot(p.index, p.values, color=cor, lw=2.2 if m == 'MIX_elastico5_UF' else 1.4,
                        ls='--' if m == 'BB' else '-', label=rot if k == 0 else None)
    for s in TESTE:
        a1.axvline(pd.Timestamp(f'{s}-10-08'), color='#bbbbbb', lw=.8, ls=':')
    a1.set_title(f'{nome} · temporada prevista em junho (SE25) por cada modelo, 2022/23 a 2025/26', loc='left', fontsize=12)
    a1.set_ylabel('casos por semana'); a1.yaxis.set_major_formatter(fmt)
    a1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7))); a1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    a1.legend(fontsize=8.5, ncol=4, frameon=False, loc='upper left'); a1.spines[['top', 'right']].set_visible(False); a1.set_ylim(bottom=0)
    # picos por temporada
    g = E[E.uf_code == u]
    ms = [m for m in MODELOS if m in set(g.modelo)]
    w = 0.8 / max(len(ms), 1)
    for k, s in enumerate(TESTE):
        gs = g[g.season == s]
        if gs.empty:
            continue
        for j, m in enumerate(ms):
            r = gs[gs.modelo == m]
            if len(r):
                a2.bar(k - 0.4 + w * (j + .5), r.pico_prev.iloc[0], w, color=MODELOS[m][1])
        a2.hlines(gs.pico_real.iloc[0], k - 0.45, k + 0.45, color='#111111', lw=2.2)
    a2.set_xticks(range(len(TESTE))); a2.set_xticklabels([f'{s}/{str(s + 1)[2:]}' + (' (em andamento)' if s in PARCIAL else '') for s in TESTE])
    a2.set_title('altura do pico: barras = previsto · traço preto = real', loc='left', fontsize=10)
    a2.yaxis.set_major_formatter(fmt); a2.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    arq = os.path.join(PASTA_G, f'{nome}.png')
    fig.savefig(arq, dpi=110); plt.close(fig)
    return arq
ORDEM = [0] + sorted(D.unidades, key=lambda u: R2.SIGLAS.get(u, str(u)))
ARQS = [figura(u) for u in ORDEM]
with PdfPages(os.path.join(PASTA_G, 'comparacao_modelos_UFs.pdf')) as pdf:
    for a in [arq_mapa] + ARQS:
        img = plt.imread(a); f = plt.figure(figsize=(14, 8)); plt.imshow(img); plt.axis('off'); pdf.savefig(f, bbox_inches='tight'); plt.close(f)
print('Pasta:', PASTA_G)
for u in (0, 35, 31, 41, 42, 53, 23, 29):
    display(Image(os.path.join(PASTA_G, f"{R2.SIGLAS.get(u, 'Brasil')}.png")))
"""


def celula(tipo, src):
    src = src.strip('\n')
    c = {'cell_type': tipo, 'metadata': {}, 'source': [l + '\n' for l in src.split('\n')]}
    if tipo == 'code':
        c.update(execution_count=None, outputs=[])
    return c


nb = {'cells': [celula('markdown', MD), celula('code', SETUP), celula('code', C2), celula('code', C3), celula('code', C4)],
      'metadata': {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
      'nbformat': 4, 'nbformat_minor': 0}
with open(os.path.join(AQUI, 'R5_Graficos_Comparacao.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('gerado: R5_Graficos_Comparacao.ipynb')
