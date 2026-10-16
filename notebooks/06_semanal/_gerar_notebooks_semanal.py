"""Gera os notebooks da comparação SEMANAL × TEMPORADA (rode: python _gerar_notebooks_semanal.py).

  Semanal1_CPU.ipynb        seleção de variáveis (afinação) + modelos semanais de CPU: referências, GBMs (todas /
                            selecionadas / só casos), ARIMA e AutoARIMA (filtro de Kalman), VAR, modelos do Felipe
  Semanal2_GPU.ipynb        redes de grafo semanais (H=4, H=68, alvo em crescimento, com covariáveis), NHITS/TFT,
                            Chronos-2 e os PRÓXIMOS PASSOS DA TEMPORADA (MTGNN v2: crescimento, misto, peso pela população)
  Semanal3_Avaliacao.ipynb  λ, ensembles semanais (GBM+ARIMA, grafos, grafos+GBM+ARIMA, top-k), ensembles de temporada
                            (só grafos com λ, por porte da UF), tabelas por nível e a tabela única temporada × semanal
1 e 2 podem rodar ao mesmo tempo. O 3 avalia o que existir (pode rodar várias vezes)."""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

ANTES = r"""
**Antes de rodar (uma vez só)**, em `MyDrive/dengue_V4/` (que já tem `dados/`, `dados_uf/`, `referencias_imdc3/`, `v4_lib.py`, `comp_lib.py`, `comp_modelos.py`, `uf_lib.py`):
1. suba **`semanal_lib.py`** (pasta `data_imdc_2026/SEMANAL/`);
2. **crie à mão** as pastas **`resultados_semanal`** e **`resultados_uf_prox`** no Drive (assim, duas sessões abertas ao mesmo tempo não criam cópias com o mesmo nome);
3. (opcional, para o Felipe na temporada) a pasta `COMPARACAO_FELIPE/` com `felipe_no_meu_jogo_previsoes.parquet`.

**Retomada:** cada modelo × rodada vira um arquivo assim que termina (gravado num `.tmp` e renomeado, então uma queda no meio nunca deixa arquivo pela metade). Se a sessão cair (internet, energia, limite do Colab), **rode de novo todas as células**: o que já existe é pulado. A seleção de variáveis também grava o estado a cada passo.
"""

SETUP = r"""
# @title 1 · Drive, arquivos e pacotes
import os, sys, shutil, time, glob, subprocess, json, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/sem', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'uf_lib.py', 'semanal_lib.py', 'dados', 'dados_uf',
                 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou subir {item} para {PASTA_DRIVE}'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
    if PACOTES:
        r = subprocess.run(f'pip install -q {PACOTES}', shell=True, capture_output=True, text=True)
        print(r.stdout[-400:], r.stderr[-400:])
sys.path.insert(0, PASTA_LOCAL)
PASTA_SEM = os.path.join(PASTA_DRIVE, 'resultados_semanal')       # previsões semanais (1 arquivo por modelo × rodada)
PASTA_PROX = os.path.join(PASTA_DRIVE, 'resultados_uf_prox')      # próximos passos da temporada (lida pelo CompUF4/Comp5)
PASTA_AVAL = os.path.join(PASTA_DRIVE, 'resultados_semanal_avaliacao')
PASTA_RES = os.path.join(PASTA_DRIVE, 'resultados_uf')
PASTA_COMP = os.path.join(PASTA_DRIVE, 'resultados_competicao')
PASTA_EXT = os.path.join(PASTA_DRIVE, 'resultados_externos')
for p in (PASTA_SEM, PASTA_PROX, PASTA_AVAL):
    os.makedirs(p, exist_ok=True)
ARQ_SEL = os.path.join(PASTA_SEM, 'selecao_grupos.json')
import importlib, v4_lib as V, comp_lib as L, comp_modelos as CM, uf_lib as U, semanal_lib as S
for m in (V, L, CM, U, S):
    importlib.reload(m)
try:
    import torch
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma', '· CPUs:', os.cpu_count())
except ImportError:
    torch = None
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'),
                   os.path.join(PASTA_LOCAL, 'dados_uf'))
TUNING, TESTE = U.TUNING, U.TESTE
REFAZER = False                         # True refaz o que já existe (normalmente: False)
"""

RODADAS = r"""
# @title 2 · Rodadas (8 temporadas + a do Felipe)
RODADAS = S.rodadas(ctx.Du, felipe=True)
D = ctx.Du
pd.DataFrame([dict(rodada=k, grupo=S.grupo_rodada(k), treino_ate=str(pd.Timestamp(D.datas[R['c']]).date()),
                   primeira_semana=str(pd.Timestamp(D.datas[R['t_ini']]).date()),
                   ultima_semana=str(pd.Timestamp(D.datas[R['t_fim']]).date())) for k, R in RODADAS.items()])
"""

LACO_SEM = r"""
# @title Rodar (retoma de onde parou; um erro não interrompe o resto)
erros = []
for m in MODELOS:
    erros += S.rodar_semanal(ctx, m, RODADAS, PASTA_SEM, refazer=REFAZER)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
print('Erros:', len(erros))
pd.DataFrame(erros, columns=['modelo', 'rodada', 'erro']).to_csv(os.path.join(PASTA_SEM, f'erros_{NOTEBOOK}.csv'), index=False)
"""

PROGRESSO = r"""
# @title Progresso (● = feito)
feitos = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(PASTA_SEM, '*.parquet'))}
pd.DataFrame({m.nome: {r: '●' if f'{m.nome}__{r}' in feitos else '·' for r in RODADAS} for m in MODELOS}).T
"""

# ------------------------------------------------------------------------------------------------------ notebook 1
NB1 = [
    ('markdown', "# Semanal 1 · CPU: seleção de variáveis e modelos semanais estatísticos e de árvores\n"
     "Prevê **1 a 4 semanas à frente** nas 8 temporadas (afinação 2018–21, teste 2022–25) e na rodada do Felipe "
     "(2023–24), com o modelo treinado uma vez por rodada.\n" + ANTES +
     "\n| Etapa | O que faz | Tempo (CPU High-RAM) |\n|---|---|---|\n"
     "| 3 · seleção | seleção para trás por grupo de variáveis, **só na afinação** (grava a cada passo) | 30–60 min |\n"
     "| 4 · modelos | referências, Felipe (XGB/LGBM/CatBoost, ponto e quantis), GBMs (só casos / todas / selecionadas; LightGBM, XGBoost, CatBoost), ARIMA_Fourier, VAR, AutoARIMA | 2–4 h |\n"),
    ('code', "NOTEBOOK = 'semanal_cpu'\nPACOTES = 'catboost'\n" + SETUP),
    ('code', RODADAS),
    ('markdown', "## 3 · Seleção de variáveis (modelos que rodam melhor com variáveis curadas)\n"
     "Começa com **todos** os grupos do `uf_lib` (clima, oceano, vizinhos, imunidade física, espalhamento, chik, "
     "Afya, estáticas, dispersão, SEAS5) e tira, um por vez, o grupo cuja retirada mais melhora o erro da mediana "
     "(log, h = 1–4) nas rodadas de **afinação**. Para quando nenhuma retirada melhora mais que 0,2%. O resultado "
     "(`selecao_grupos.json`) é usado pelos modelos `_sel` aqui e no notebook 2."),
    ('code', r"""
# @title 3 · Seleção (retoma de onde parou)
GRUPOS_SEL = S.selecionar_grupos(ctx, RODADAS, ARQ_SEL)
print('Grupos escolhidos:', GRUPOS_SEL)
est = json.load(open(ARQ_SEL))
pd.DataFrame(est['historico'])
"""),
    ('markdown', "## 4 · Modelos semanais (CPU)\n"
     "- **Todas as variáveis:** GBM_*_todas (os modelos de árvore aguentam todas).\n"
     "- **Variáveis curadas:** GBM_lgb_sel (grupos da etapa 3), nas versões crescimento e âncora.\n"
     "- **Só casos:** GBM_lgb_casos, ARIMA, VAR, referências e os modelos do Felipe (as variáveis dele)."),
    ('code', r"""
# @title 4 · Lista (ordem = prioridade)
MODELOS = [S.PersistenciaQ(), S.ClimatologiaSem(),
           S.FelipeSemanal('xgb'), S.FelipeSemanal('lgbm'), S.FelipeSemanal('cat'), S.FelipeSemanal('xgb', quantis=True),
           S.GBMSemanal('lgb', 'todas'), S.GBMSemanal('lgb', 'sel', arq_sel=ARQ_SEL), S.GBMSemanal('lgb', 'casos'),
           S.ARIMASemanal(), S.VARSemanal(),
           S.GBMSemanal('lgb', 'sel', alvo='ancora', arq_sel=ARQ_SEL),
           S.GBMSemanal('xgb', 'todas'), S.GBMSemanal('cat', 'todas'),
           S.ARIMASemanal(auto=True)]
[m.nome for m in MODELOS]
"""),
    ('code', LACO_SEM),
    ('code', PROGRESSO),
]

# ------------------------------------------------------------------------------------------------------ notebook 2
NB2 = [
    ('markdown', "# Semanal 2 · GPU: redes semanais e próximos passos da temporada\n" + ANTES +
     "\n| Parte | O que faz | Tempo (GPU L4) |\n|---|---|---|\n"
     "| A · grafos semanais | MTGNN, A3T-GCN, GCN-GRU, STGCN (H=4); MTGNN com H=68 (= o modelo de temporada usado na semana seguinte); MTGNN com alvo em crescimento; MTGNN com todas as covariáveis e com as selecionadas | 1–2 h |\n"
     "| B · NHITS, TFT, Chronos-2 semanais | sem covariáveis, com todas e com as selecionadas; Chronos-2 zero-shot | 2–3 h |\n"
     "| C · temporada, próximos passos | MTGNN v2: alvo em crescimento, alvo misto e perda ponderada pela população (raiz), nas 8 temporadas → `resultados_uf_prox` | 2–3 h |\n\n"
     "Os modelos `_sel` usam a seleção do notebook 1; se ela ainda não existir, eles são pulados (rode de novo depois)."),
    ('code', "NOTEBOOK = 'semanal_gpu'\nPACOTES = 'neuralforecast \"chronos-forecasting>=2.0\"'\n" + SETUP),
    ('code', RODADAS),
    ('markdown', "## A · Redes de grafo semanais"),
    ('code', r"""
# @title A · Lista
TEM_SEL = os.path.exists(ARQ_SEL) and json.load(open(ARQ_SEL)).get('final') is not None
MODELOS = [S.RedeSemanal(U.MTGNN()), S.RedeSemanal(U.GCNGRU(atencao=True)), S.RedeSemanal(U.GCNGRU()),
           S.RedeSemanal(U.STGCN()),
           S.RedeSemanal(U.MTGNN(), H_treino=68),                 # o modelo de temporada, usado para a semana seguinte
           S.RedeSemanal(U.MTGNN(), alvo='cresc'),               # o truque que funcionou no jogo do Felipe
           S.RedeSemanal(U.MTGNN(), cov='todas')]
if TEM_SEL:
    MODELOS.append(S.RedeSemanal(U.MTGNN(), cov='sel', arq_sel=ARQ_SEL))
else:
    print('⚠️ seleção do notebook 1 ainda não existe: MTGNN_sel fica para a próxima execução')
[m.nome for m in MODELOS]
"""),
    ('code', LACO_SEM),
    ('code', PROGRESSO),
    ('markdown', "## B · NHITS, TFT e Chronos-2 semanais"),
    ('code', r"""
# @title B · Lista
MODELOS = [S.NFSemanal('NHITS'), S.NFSemanal('TFT'), S.NFSemanal('NHITS', cov='todas'), S.NFSemanal('TFT', cov='todas'),
           S.Chronos2Semanal()]
if TEM_SEL:
    MODELOS += [S.NFSemanal('NHITS', cov='sel', arq_sel=ARQ_SEL), S.NFSemanal('TFT', cov='sel', arq_sel=ARQ_SEL)]
[m.nome for m in MODELOS]
"""),
    ('code', LACO_SEM),
    ('code', PROGRESSO),
    ('markdown', "## C · Temporada: próximos passos (MTGNN v2)\n"
     "Mesma régua dos CompUF (dados até a SE25, SE41→SE40). Grava em `resultados_uf_prox/`, que o `CompUF4`, o `Comp5` "
     "e o notebook 3 leem junto com as outras pastas `resultados_uf*`."),
    ('code', r"""
# @title C · Rodar (retoma de onde parou)
PROX = {m.nome: m for m in [S.MTGNNv2('cresc'), S.MTGNNv2('misto'), S.MTGNNv2('ancora', peso_pop=0.5)]}
erros = []
for nome, m in PROX.items():
    for s in TUNING + TESTE:
        try:
            dt = L.rodar(ctx, m, s, PASTA_PROX, M=500, refazer=REFAZER)
            if dt is not None:
                print(f'✅ {nome:28s} {s}: {dt / 60:5.1f} min', flush=True)
        except Exception as e:
            erros.append((nome, s, repr(e)[:500]))
            print(f'❌ {nome:28s} {s}: {repr(e)[:300]}', flush=True)
            traceback.print_exc(limit=2)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
print('Erros:', len(erros))
pd.DataFrame(erros, columns=['modelo', 'temporada', 'erro']).to_csv(os.path.join(PASTA_PROX, 'erros_prox.csv'), index=False)
"""),
]

# ------------------------------------------------------------------------------------------------------ notebook 3
NB3 = [
    ('markdown', "# Semanal 3 · Avaliação: semanal, temporada e os dois lado a lado\n" + ANTES +
     "\nRoda em CPU (~10–20 min) e pode ser repetido a qualquer momento: avalia o que já existir. As tabelas vão para "
     "`resultados_semanal_avaliacao/`.\n\n"
     "**Regras de escolha:** tudo (λ, membros de ensemble, porte da UF) é escolhido na **afinação (2018–2021)**; o "
     "**teste (2022–2025)** só é olhado no fim. A régua comum aos dois níveis é o **WIS relativo à climatologia** "
     "(a mesma climatologia por semana do ano); na temporada também vale o WIS relativo ao BB (régua oficial)."),
    ('code', "NOTEBOOK = 'semanal_avaliacao'\nPACOTES = 'catboost'\n" + SETUP),
    ('markdown', "## A · Semanal"),
    ('code', r"""
# @title A1 · Ler as previsões semanais e cobertura
D = ctx.Du
P0 = S.ler_semanal(sorted(p for p in glob.glob(os.path.join(PASTA_DRIVE, 'resultados_semanal*')) if os.path.isdir(p)
                          and not p.endswith('_avaliacao')))
P0['date'], P0['origem'] = pd.to_datetime(P0['date']), pd.to_datetime(P0['origem'])
print(len(P0), 'linhas ·', P0.modelo.nunique(), 'modelos')
P0.groupby(['modelo', 'rodada']).size().unstack().notna().replace({True: '●', False: '·'})
"""),
    ('code', r"""
# @title A2 · λ dos intervalos (afinação) e ensembles semanais
Pv = S.com_verdade(P0, D)
BASE = [m for m in P0.modelo.unique() if m not in S.REFS_SEM]
LAM_SEM = S.calibrar_lambda(Pv, BASE)
print('λ escolhidos na afinação:', {k: v for k, v in LAM_SEM.items() if v != 1.0})
CAL = [U.ajusta_largura(P0[P0.modelo == m], l).assign(modelo=f'{m}+λ') for m, l in LAM_SEM.items() if l != 1.0]
P1 = pd.concat([P0] + CAL, ignore_index=True)
# melhor GBM na afinação (para o ensemble GBM + ARIMA)
ra = S.resumo_semanal(S.com_verdade(P1, D), 'afinacao')
media_h = (ra.groupby('modelo')['R_vs_Climatologia_sem'].mean().sort_values()
           if len(ra) and 'R_vs_Climatologia_sem' in ra else pd.Series(dtype=float))   # vazio até existir afinação
gbm = [m for m in media_h.index if m.startswith('GBM_')]
def existe(*ms):
    return all(m in set(P1.modelo) for m in ms)
ENS = {}
if gbm and existe('ARIMA_Fourier_sem'):
    ENS['ENS_GBM_ARIMA_sem'] = [gbm[0], 'ARIMA_Fourier_sem']
grafos = [m for m in ('MTGNN_sem', 'A3TGCN_sem', 'GCNGRU_sem') if existe(m)]
if len(grafos) == 3:
    ENS['ENS_grafos_sem'] = grafos
    if 'ENS_GBM_ARIMA_sem' in ENS:
        ENS['ENS_grafos_GBM_ARIMA_sem'] = grafos + ENS['ENS_GBM_ARIMA_sem']
cand, vistos = [], set()                                    # um representante por modelo (com ou sem λ)
for m in media_h.index:
    b = m.replace('+λ', '')
    if m not in S.REFS_SEM and b not in vistos:
        vistos.add(b); cand.append(m)
for k in (3, 5):
    if len(cand) >= k:
        ENS[f'ENS_sem_top{k}'] = cand[:k]                   # escolhidos na afinação
for nome, mb in ENS.items():
    print(f'{nome}: {mb}')
P2 = pd.concat([P1] + [S.ensemble_semanal(P1, mb, nome) for nome, mb in ENS.items()], ignore_index=True)
PV = S.com_verdade(P2, D)
"""),
    ('code', r"""
# @title A3 · Tabelas semanais (afinação, teste e a rodada do Felipe)
TAB_SEM = {g: S.resumo_semanal(PV, g) for g in ('afinacao', 'teste', 'felipe')}
for g, t in TAB_SEM.items():
    t.to_csv(os.path.join(PASTA_AVAL, f'semanal_{g}.csv'), index=False)
COLS = ['modelo', 'h', 'rodadas', 'R_vs_Climatologia_sem', 'R_vs_Persistencia_UF', 'UFs_melhor_que_Climatologia_sem',
        'WIS', 'WIS_log', 'MAE', 'WAPE', 'sMAPE', 'IC50', 'IC90', 'r', 'vies_pct']
for h in (1, 4):
    print(f'\n=== TESTE 2022–2025 · h = {h} ===')
    display(TAB_SEM['teste'].query('h == @h')[[c for c in COLS if c in TAB_SEM['teste']]].head(25).round(3))
print('\n=== JOGO DO FELIPE (teste dele, 2023–2024) · h = 1 ===')
display(TAB_SEM['felipe'].query('h == 1')[[c for c in COLS if c in TAB_SEM['felipe']]].round(3))
"""),
    ('markdown', "## B · Temporada (com os próximos passos)"),
    ('code', r"""
# @title B1 · Ler todos os modelos de temporada (inclusive resultados_uf_prox e o Felipe)
PASTAS_UF = [p for p in sorted(glob.glob(os.path.join(PASTA_DRIVE, 'resultados_uf*'))) if os.path.isdir(p)]
FONTES = [('região→UF', PASTA_COMP)] + [(os.path.basename(p), p) for p in PASTAS_UF] + [('externo', PASTA_EXT)]
PU = L.carregar_previsoes(FONTES, ctx.Du.ufs, 'uf', mostrar=False)
NOVOS = sorted(PU[PU.fonte.str.contains('prox')].modelo.unique())
PU = PU[~PU.modelo.str.startswith('ENS')][L.COLS_PREV]
ens = [os.path.join(p, 'ensembles_uf.parquet') for p in PASTAS_UF if os.path.exists(os.path.join(p, 'ensembles_uf.parquet'))]
ENS0 = pd.read_parquet(ens[0])[L.COLS_PREV] if ens else PU.iloc[:0]
arq_f = os.path.join(PASTA_DRIVE, 'COMPARACAO_FELIPE', 'felipe_no_meu_jogo_previsoes.parquet')
FEL = pd.read_parquet(arq_f) if os.path.exists(arq_f) else PU.iloc[:0]
FEL = FEL[FEL.modelo.isin(['Felipe_XGB_ponto', 'Felipe_XGB_quantis'])][L.COLS_PREV] if len(FEL) else FEL
print('Modelos novos (próximos passos):', NOVOS)
# λ dos modelos novos (afinação), como no CompUF4
GRADE_LAM = [0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5]
clim = PU[(PU.modelo == 'climatologia') & PU.season.isin(TUNING)]
CAL2 = []
for m in NOVOS:
    g = PU[(PU.modelo == m) & PU.season.isin(TUNING)]
    if g.season.nunique() < len(TUNING):
        print('sem afinação completa:', m); continue
    rs = []
    for l in GRADE_LAM:
        _, rr = V.avaliar(pd.concat([U.ajusta_largura(g, l).assign(modelo='x'), clim]), ctx.ver_uf, ref=('climatologia',))
        rs.append(float(rr.set_index('modelo').loc['x', 'R_vs_climatologia']))
    l = GRADE_LAM[int(np.argmin(rs))]
    print(f'{m:28s} λ = {l}')
    if l != 1.0:
        CAL2.append(U.ajusta_largura(PU[PU.modelo == m], l).assign(modelo=f'{m}+λ'))
REF = ctx.ref[L.COLS_PREV] if ctx.ref is not None else PU.iloc[:0]     # BB e equipes do 3º IMDC (régua oficial)
TODOS = pd.concat([REF, PU, ENS0, FEL] + CAL2, ignore_index=True).drop_duplicates(['modelo', 'season', 'uf_code', 'date'], keep='last')
"""),
    ('code', r"""
# @title B2 · Ensembles de temporada: só grafos com λ, + MTGNN v2, + TFT/NHITS, e por porte da UF
def ens(membros, nome, metodo='mediana'):
    global TODOS
    falta = [m for m in membros if m not in set(TODOS.modelo)]
    if falta:
        print(f'{nome}: faltam {falta}'); return
    e, _ = V.ensemble(TODOS, membros, nome, metodo=metodo)
    TODOS = pd.concat([TODOS, e], ignore_index=True)
    print(f'{nome}: {sorted(e.season.unique())}')
ens(['MTGNN_UF+λ', 'A3TGCN_UF+λ', 'GCNGRU_UF+λ'], 'ENS_grafos_λ')
# melhor MTGNN v2 na afinação entra no ensemble de grafos
_, ra = V.avaliar(TODOS[TODOS.season.isin(TUNING) & TODOS.modelo.isin([m + s for m in NOVOS for s in ('', '+λ')] + ['climatologia'])],
                  ctx.ver_uf, ref=('climatologia',))
ra = ra[ra.temporadas == len(TUNING)].dropna(subset=['R_vs_climatologia']).sort_values('R_vs_climatologia') if len(ra) else ra
if len(ra):
    display(ra[['modelo', 'R_vs_climatologia']].round(3))
    ens(['MTGNN_UF+λ', 'A3TGCN_UF+λ', 'GCNGRU_UF+λ', ra.modelo.iloc[0]], 'ENS_grafos2_λ')
ens(['ENS_grafos_λ', 'TFT_pool_UF', 'NHITS_pool_UF'], 'ENS_grafos_TFT_NHITS')
CAND = [m for m in ('ENS_grafos_λ', 'ENS_grafos2_λ', 'TFT_pool_UF', 'NHITS_pool_UF', 'ENS_grafos_TFT_NHITS',
                    'MTGNN_pesopop_UF', 'MTGNN_pesopop_UF+λ', 'ENS_UF_top5') if m in set(TODOS.modelo)]
EP, TAB_PORTE, ESC = S.ensemble_por_porte(TODOS, CAND, ctx.ver_uf, D, TUNING)
TODOS = pd.concat([TODOS, EP], ignore_index=True)
print('Escolha por porte (0 = UFs pequenas … 2 = grandes):', ESC)
TAB_PORTE
"""),
    ('code', r"""
# @title B3 · Tabela da temporada (teste 2022–2025): régua oficial + as mesmas métricas do semanal
TT = TODOS[TODOS.season.isin(TESTE)]
por_t, res_t = V.avaliar(TT, ctx.ver_uf, ref=('BB', 'climatologia'))
res_t = res_t[res_t.temporadas == len(TESTE)]
TV = S.com_verdade(TT, D)
pont = pd.DataFrame([dict(modelo=m, **S.metricas(g)) for m, g in TV.groupby('modelo')])
TAB_TEMP = res_t[[c for c in ['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'R_vs_climatologia'] if c in res_t]].merge(pont, on='modelo') \
    .sort_values('R_vs_BB' if 'R_vs_BB' in res_t else 'WIS').reset_index(drop=True)
TAB_TEMP.to_csv(os.path.join(PASTA_AVAL, 'temporada_teste.csv'), index=False)
TAB_TEMP.head(30).round(3)
"""),
    ('markdown', "## C · Os dois níveis lado a lado (mesmas semanas-alvo: SE41→SE40 de 2022 a 2025)"),
    ('code', r"""
# @title C1 · Tabela única: temporada (16–68 semanas de antecedência) × semanal (1 e 4 semanas)
cols = ['modelo', 'nivel', 'R_vs_clim', 'WIS', 'WIS_log', 'MAE', 'WAPE', 'sMAPE', 'IC90', 'r', 'vies_pct']
t1 = TAB_TEMP.rename(columns={'R_vs_climatologia': 'R_vs_clim'}).assign(nivel='temporada (SE25)')
ts = TAB_SEM['teste'].rename(columns={'R_vs_Climatologia_sem': 'R_vs_clim'})
t2 = pd.concat([ts.query('h == @h').assign(nivel=f'semanal h={h}') for h in (1, 2, 3, 4)])
UNICA = pd.concat([t1, t2], ignore_index=True)[cols]
UNICA.to_csv(os.path.join(PASTA_AVAL, 'tabela_unica_teste.csv'), index=False)
for nv in ['temporada (SE25)', 'semanal h=1', 'semanal h=4']:
    print(f'\n=== {nv} · top 10 pela régua comum (WIS relativo à climatologia) ===')
    display(UNICA[UNICA.nivel == nv].sort_values('R_vs_clim').head(10).round(3))
print('\n=== Felipe × meus melhores, nos dois níveis ===')
melhores = UNICA.sort_values('R_vs_clim').groupby('nivel').head(3)
display(pd.concat([melhores, UNICA[UNICA.modelo.str.contains('Felipe|Persist|Climatologia|climatologia|^BB$')]])
        .drop_duplicates().sort_values(['nivel', 'R_vs_clim']).round(3))
"""),
    ('code', r"""
# @title C2 · Top 5 de cada métrica, em cada nível (teste)
crit = {'R_vs_clim': True, 'WIS': True, 'WIS_log': True, 'MAE': True, 'WAPE': True, 'sMAPE': True, 'r': False}
linhas = []
for nv, g in UNICA.groupby('nivel'):
    g = g[~g.modelo.isin(['climatologia', 'Climatologia_sem'])]
    for c, asc in crit.items():
        o = g.dropna(subset=[c]).sort_values(c, ascending=asc).head(5)
        linhas.append(dict(nivel=nv, metrica=c, **{f'{i + 1}º': f'{m} ({v:.3g})' for i, (m, v) in enumerate(zip(o.modelo, o[c]))}))
    o = g.assign(_k=(g.IC90 - 90).abs()).sort_values('_k').head(5)
    linhas.append(dict(nivel=nv, metrica='IC90 perto de 90', **{f'{i + 1}º': f'{m} ({v:.3g})' for i, (m, v) in enumerate(zip(o.modelo, o.IC90))}))
TOP5 = pd.DataFrame(linhas)
TOP5.to_csv(os.path.join(PASTA_AVAL, 'top5_por_metrica.csv'), index=False)
TOP5
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


grava('Semanal1_CPU.ipynb', NB1, False)
grava('Semanal2_GPU.ipynb', NB2, True)
grava('Semanal3_Avaliacao.ipynb', NB3, False)
