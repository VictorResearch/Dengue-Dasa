"""Gera os notebooks dos modelos direto por UF (rode: python _gerar_notebooks_uf.py).

  CompUF1_portados.ipynb   Fase 1: os modelos da competição rodando direto por UF (CPU; INLA opcional)
  CompUF2_CPU.ipynb        Fase 3 e 6: auto.arima e auto.ARIMAX (Python e R), 6 VARs (macro, espacial, BVAR Minnesota;
                           com e sem ENSO), INLA não linear, dois estágios, GBMs (LightGBM, XGBoost, CatBoost) com
                           covariáveis, pooling, aumento de dados e ablação por grupo de variáveis (CPU)
  CompUF3_GPU.ipynb        Fases 4 e 5: AGCRN e Chronos por UF, redes do neuralforecast com pooling (com e sem
                           exógenas futuras), STID, redes de grafo (GCN-GRU, A3T-GCN, STGCN, MTGNN), Chronos-2 (GPU L4)
  CompUF4_avaliacao.ipynb  Fase 7: afinação (2018–2021) → escolha → teste (2022–2025), UF × região, ablação,
                           calibração da largura, ensembles (CPU)

1, 2 e 3 podem rodar ao mesmo tempo (sessões separadas). O 4 avalia o que existir.
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.join(AQUI, '..', 'COMPETICAO')

CABECALHO = r"""
**Antes de rodar (uma vez só)**, em `MyDrive/dengue_V4/` (que já tem `dados/`, `referencias_imdc3/` e os arquivos da competição):
1. suba **`uf_lib.py`** e **`uf_autoarima.R`** (pasta `data_imdc_2026/UF/`);
2. suba a **pasta `UF/dados/` com o nome `dados_uf/`** (5 arquivos `.parquet` + `vizinhanca_uf.npz`);
3. substitua **`comp_modelos.py`** e **`comp_inla.R`** pelas versões novas (de `COMPETICAO/`): elas trazem o INLA não linear e a correção do `val_size`.

Os resultados vão para `MyDrive/dengue_V4/resultados_uf/`. Se a sessão cair, rode de novo: o que já foi feito é pulado.
⚠️ Ao abrir vários notebooks ao mesmo tempo, **espere a célula 1 do primeiro terminar** antes de rodar os outros: sessões simultâneas podem criar duas pastas `resultados_uf` no Drive. Se isso acontecer, renomeie a cópia para `resultados_uf_2`; o `CompUF4` e o `Comp5` leem todas as pastas `resultados_uf*`.

**Temporadas:** 2018–2021 = **afinação** (é onde se escolhe tudo); 2022–2025 = **teste** (S1–S4, olhado uma vez no `CompUF4`).

| Notebook | O que roda | Ambiente | Tempo estimado |
|---|---|---|---|
| `CompUF1_portados` | os 13 modelos da competição, direto por UF | CPU High-RAM | 1–2 h (com INLA) |
| `CompUF2_CPU` | auto.arima / auto.ARIMAX (Python e R), 6 VARs, INLA não linear, dois estágios, GBMs com covariáveis, ablação | CPU High-RAM | 4–6 h |
| `CompUF3_GPU` | AGCRN, Chronos, 6 redes com pooling (com e sem exógenas), STID, 4 redes de grafo, Chronos-2 | GPU L4 | 8–12 h |
| `CompUF4_avaliacao` | escolha em 2018–2021, teste em 2022–2025, ensembles | CPU | ~15 min |
"""

SETUP = r"""
# @title Drive, arquivos e pacotes
import os, sys, shutil, time, glob, subprocess, json, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/uf', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'comp_inla.R', 'uf_lib.py', 'uf_autoarima.R', 'dados',
                 'dados_uf', 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou subir {item} para {PASTA_DRIVE}'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
    if PACOTES:
        r = subprocess.run(f'pip install -q {PACOTES}', shell=True, capture_output=True, text=True)
        print(r.stdout[-400:], r.stderr[-400:])
sys.path.insert(0, PASTA_LOCAL)
PASTA_RES = os.path.join(PASTA_DRIVE, 'resultados_uf')
PASTA_COMP = os.path.join(PASTA_DRIVE, 'resultados_competicao')
os.makedirs(os.path.join(PASTA_RES, 'graficos'), exist_ok=True)
import importlib, v4_lib as V, comp_lib as L, comp_modelos as CM, uf_lib as U
for m in (V, L, CM, U):
    importlib.reload(m)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma', '· CPUs:', os.cpu_count())
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'),
                   os.path.join(PASTA_LOCAL, 'dados_uf'))
TUNING, TESTE, TODAS = U.TUNING, U.TESTE, U.TODAS_UF
M = 500                                  # trajetórias por UF
REFAZER = False
"""

LACO = r"""
# @title Rodar (retoma de onde parou; um modelo com erro não interrompe os outros)
erros = []
for nome, temps in PLANO.items():
    for s in temps:
        try:
            dt = L.rodar(ctx, MODELOS[nome], s, PASTA_RES, M=M, refazer=REFAZER)
            if dt is not None:
                print(f'✅ {nome:34s} {s}: {dt / 60:5.1f} min', flush=True)
        except Exception as e:
            erros.append((nome, s, repr(e)))
            print(f'❌ {nome:34s} {s}: {repr(e)[:400]}', flush=True)
            traceback.print_exc(limit=2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
print('Erros:', len(erros))
pd.DataFrame(erros, columns=['modelo', 'temporada', 'erro']).to_csv(os.path.join(PASTA_RES, f'erros_{NOTEBOOK}.csv'), index=False)
t = L.tempos(PASTA_RES)
t[t.modelo.isin(list(PLANO))].groupby('modelo').segundos.agg(['count', 'mean', 'sum']).div([1, 60, 60]).round(1) \
    .rename(columns={'count': 'temporadas', 'mean': 'min/temporada', 'sum': 'min total'})
"""

# célula de instalação do INLA: a mesma do Comp2 (já corrigida no Colab), lida do notebook gerado
_c2 = json.load(open(os.path.join(COMP, 'Comp2_INLA.ipynb'), encoding='utf-8'))
INSTALA_INLA = ''.join(_c2['cells'][2]['source'])
assert 'MOTOR_OK' in INSTALA_INLA and '/content/R_libs' in INSTALA_INLA
INLA_CEL = "USAR_INLA = True   # False pula a instalação e os modelos INLA\nif USAR_INLA:\n" + \
           '\n'.join('    ' + l if l.strip() else l for l in INSTALA_INLA.split('\n')) + \
           "\nINLA_KW = dict(anos=6, M=300, threads=os.cpu_count(), r_libs=R_LIBS, pasta_script=PASTA_LOCAL) if USAR_INLA else None"


# célula do R + forecast (o auto.arima "original"); independe do INLA
R_CEL = r"""
# @title R + pacote forecast (para o auto.arima em R; ~1 min na primeira vez)
R_LIBS = '/content/R_libs' if NO_COLAB else None
if NO_COLAB:
    os.makedirs(R_LIBS, exist_ok=True); os.environ['R_LIBS_USER'] = R_LIBS
    if shutil.which('Rscript') is None:
        print('Instalando o R (~2 min)...'); subprocess.run('apt-get -qq update && apt-get -qq install -y r-base-core > /dev/null', shell=True)
    tem = lambda: 'TRUE' in subprocess.run(['Rscript', '-e', 'cat(requireNamespace("forecast", quietly=TRUE))'],
                                           capture_output=True, text=True).stdout
    if not tem():
        r = subprocess.run(['Rscript', '-e', 'options(Ncpus=4); install.packages("forecast", lib=Sys.getenv("R_LIBS_USER"), '
                            'repos="https://packagemanager.posit.co/cran/__linux__/jammy/latest")'], capture_output=True, text=True)
        print(r.stderr[-800:])
    print('forecast disponível:', tem())
RSCRIPT = shutil.which('Rscript') or 'Rscript'
R_KW = dict(rscript=RSCRIPT, r_libs=R_LIBS, pasta_script=PASTA_LOCAL)
"""


def salva(nome, celulas):
    gpu = 'GPU' in nome
    nb = {'cells': [], 'metadata': {'accelerator': 'GPU' if gpu else 'None',
                                    'colab': {'provenance': [], **({'gpuType': 'L4'} if gpu else {})},
                                    'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                                    'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 0}
    for tipo, txt in celulas:
        txt = txt.strip('\n')
        linhas = [l + '\n' for l in txt.split('\n')]
        linhas[-1] = linhas[-1].rstrip('\n')
        c = {'cell_type': tipo, 'metadata': {}, 'source': linhas}
        if tipo == 'code':
            c.update(execution_count=None, outputs=[])
        nb['cells'].append(c)
    with open(os.path.join(AQUI, nome), 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('gerado', nome, len(celulas), 'células')


# =====================================================================================================================
salva('CompUF1_portados.ipynb', [
    ('markdown', "# Modelos direto por UF · 1/4 · os modelos da competição rodando por UF (Fase 1)\n" + CABECALHO + r"""
**Pergunta que este notebook responde:** para cada modelo, prever **direto por UF** é melhor ou pior do que prever as 435 regiões e somar (o que a competição fez)? A comparação sai no `CompUF4`.

**Ambiente:** CPU com bastante RAM (*High-RAM*). Os modelos de GPU (AGCRN e Chronos) estão no `CompUF3_GPU`.
Cada modelo ganha o sufixo `_UF` (ex.: `ARIMA_Fourier_UF`).
"""),
    ('code', "NOTEBOOK = 'uf_portados'\nPACOTES = 'statsforecast'\n" + SETUP),
    ('code', INLA_CEL),
    ('code', r"""
MODELOS = {m.nome: m for m in U.portados_cpu(INLA_KW)}
print(list(MODELOS))
PLANO = {nome: TODAS for nome in MODELOS}
"""),
    ('code', LACO),
])

# =====================================================================================================================
salva('CompUF2_CPU.ipynb', [
    ('markdown', "# Modelos direto por UF · 2/4 · estatísticos e árvores com covariáveis (Fases 3 e 6)\n" + CABECALHO + r"""
**Ambiente:** CPU *High-RAM*. Dá para dividir em sessões com `RODAR_AGORA`.

| Grupo | Modelos | Ideia |
|---|---|---|
| auto.arima | `AutoARIMA_py_UF`, `AutoARIMA_r_UF` | **tudo automático**: d (KPSS), p, q e drift pelo AICc; sazonalidade por Fourier com K = 1..6 também pelo AICc. `py` = statsforecast, `r` = o `forecast::auto.arima` original |
| auto.arima estacionário | `AutoARIMA_d0_py_UF`, `AutoARIMA_d0_r_UF` | o mesmo, mas com d = 0 e sem drift (como o vencedor da competição): impede a extrapolação de tendência por 68 semanas |
| auto.ARIMAX | `AutoARIMAX_py_UF`, `AutoARIMAX_r_UF`, `AutoARIMAX_d0_py_UF`, `AutoARIMAX_d0_r_UF` | + ENSO (persistência da SE25) e anomalias de temperatura e chuva (SEAS5 por 6 meses, depois 0) |
| VAR (várias UFs juntas) | `VAR_macro_UF`, `VAR_espacial_UF`, `BVAR_Minnesota_UF` (+ versões `X` com ENSO) | anomalias de log-incidência; *macro* = um VAR por grande região, ordem pelo BIC; *espacial* = defasagens próprias + média dos vizinhos; *BVAR* = todas as 26 UFs com encolhimento de Minnesota |
| INLA não linear | `GHR_NL_INLA_UF` | BYM2 nas 26 UFs + efeitos não lineares (RW1 nos decis) das anomalias de clima + ENSO |
| Dois estágios | `DoisEstagios_UF`, `DoisEstagios_pool_UF`, `DoisEstagios_pool_sel_UF` | GBM prevê tamanho e pico da temporada com variáveis da SE25; a curva vem das temporadas passadas; *sel* = só as variáveis aprovadas na avaliação |
| GBMs | `GBM_lgb_casos_UF`, `GBM_lgb_selecionadas_UF`, `GBM_lgb_todas_UF`, `GBM_{xgb,cat}_selecionadas_UF`, `GBM_lgb_todas_pool_UF` | quantílicos diretos (68 semanas); *casos* = sem covariáveis, *selecionadas* = conjunto padrão, *todas* = tudo; *pool* = treina também com as 435 regiões |
| Aumento de dados | `GBM_{lgb,cat}_selecionadas_aum_UF` | linhas extras com temporadas 1,5–4× maiores (Fase 6) |
| Ablação | `GBM_lgb_todas_sem_<grupo>_UF` | um grupo de variáveis a menos por vez; **só 2018–2021** |

**Comparabilidade:** todo modelo com exógenas futuras usa o MESMO conjunto `U.EXOG_FUTURAS` = ENSO (persistido da SE25) + anomalias de temperatura e chuva (SEAS5 por 6 meses, depois 0); os VARX usam só o ENSO. Cada família tem a versão **sem** exógenas, para medir o ganho.
"""),
    ('code', "NOTEBOOK = 'uf_cpu'\nPACOTES = 'statsforecast catboost'\n" + SETUP),
    ('code', INLA_CEL),
    ('code', R_CEL),
    ('code', r"""
mods = []
for est in (False, True):                                       # d livre (auto.arima puro) e d = 0 sem drift
    for ex in ((), U.EXOG_FUTURAS):                             # ARIMA e ARIMAX
        mods += [U.AutoARIMA('py', ex, estacionario=est), U.AutoARIMA('r', ex, estacionario=est, **R_KW)]
mods += [U.VARModelo(t, exog=x, M=M) for t in ('macro', 'espacial', 'bvar') for x in (False, True)]
if INLA_KW is not None:
    mods.append(U.Porta(CM.ModeloINLA('ghr_nl', **INLA_KW)))
mods += [U.DoisEstagios(pool=False, M=M), U.DoisEstagios(pool=True, M=M), U.DoisEstagios(pool=True, selecao=True, M=M),
         U.GBMDireto('lgb', None), U.GBMDireto('lgb', 'selecionadas'), U.GBMDireto('lgb', 'todas'),
         U.GBMDireto('xgb', 'selecionadas'), U.GBMDireto('cat', 'selecionadas'), U.GBMDireto('lgb', pool=True),
         U.GBMDireto('lgb', 'selecionadas', aumento=0.15), U.GBMDireto('cat', 'selecionadas', aumento=0.15)]
abl = U.ablacao('lgb')
MODELOS = {m.nome: m for m in mods + abl}
PLANO_COMPLETO = {**{m.nome: TODAS for m in mods}, **{m.nome: TUNING for m in abl}}
RODAR_AGORA = list(PLANO_COMPLETO)      # ex.: só os ARIMA numa sessão: [k for k in PLANO_COMPLETO if k.startswith('AutoARIMA')]
PLANO = {k: v for k, v in PLANO_COMPLETO.items() if k in RODAR_AGORA}
print(len(PLANO), 'modelos ·', sum(len(v) for v in PLANO.values()), 'rodadas')
"""),
    ('code', LACO),
])

# =====================================================================================================================
salva('CompUF3_GPU.ipynb', [
    ('markdown', "# Modelos direto por UF · 3/4 · redes, grafos e modelos de fundação (Fases 4 e 5)\n" + CABECALHO + r"""
**Ambiente:** GPU **L4**. Dá para dividir em sessões com `RODAR_AGORA`.

| Grupo | Modelos | Ideia |
|---|---|---|
| Portados | `AGCRN_UF`, `Chronos_UF` | os da competição, direto por UF |
| neuralforecast com *pooling* | `TFT_pool_UF`, `NHITS_pool_UF`, `TiDE_pool_UF`, `TSMixerx_pool_UF`, `DeepAR_pool_UF`, `PatchTST_pool_UF` | treinados com 435 regiões + 26 UFs (461 séries), prevendo as UFs; Fourier (futuro), anomalias de clima (passado), nível e população (estáticas); 3 sementes |
| Identidade espaço-temporal | `STID_pool_UF`, `STID_UF` | MLP com embedding da unidade e da semana (empata com GNNs na literatura) |
| neuralforecast sem exógenas | `TFT_pool_semcov_UF`, `NHITS_pool_semcov_UF`, `TiDE_pool_semcov_UF` | as mesmas, só com Fourier: mede quanto as exógenas futuras (ENSO, tanom, panom) ajudam |
| Grafo · T-GCN | `GCNGRU_UF` | GRU por UF + 2 camadas de convolução no grafo de fronteiras das 26 UFs |
| Grafo · A3T-GCN | `A3TGCN_UF` | o mesmo + atenção temporal sobre os estados ocultos (Bai et al. 2021) |
| Grafo · STGCN | `STGCN_UF` | blocos "sanduíche" convolução temporal com GLU → grafo → temporal (Yu et al. 2018) |
| Grafo · MTGNN | `MTGNN_UF` | grafo aprendido (direcionado, top-4) somado ao de fronteiras, mix-hop + convoluções dilatadas (Wu et al. 2020) |
| Fundação | `Chronos2_UF`, `Chronos2_cov_UF` | Chronos-2 zero-shot, sem e com covariáveis (anomalias + SEAS5 + ENSO) |
| Opcional | `TimesFM_UF` | **experimental** (API nova do TimesFM 3, não testada): ligue com `USAR_TIMESFM = True` |

⚠️ As redes do neuralforecast, o Chronos-2 e o TimesFM **não puderam ser testados no PC** (sem GPU). STID e as 4 redes de grafo foram testados com 1 época. Um erro vai para `erros_uf_gpu.csv` e os outros modelos seguem: mande o arquivo que eu corrijo.
"""),
    ('code', "USAR_TIMESFM = False\nNOTEBOOK = 'uf_gpu'\nPACOTES = 'neuralforecast \"chronos-forecasting>=2.0\"' + (' timesfm' if USAR_TIMESFM else '')\n" + SETUP),
    ('code', r"""
cfg_agcrn = dict(n_sementes=2, epocas_max=40, paciencia=6, frac_origens=0.5, n_traj=M)
mods = U.portados_gpu(cfg_agcrn)
mods += [U.NFPooled(a) for a in ('TFT', 'NHITS', 'TiDE', 'TSMixerx', 'DeepAR', 'PatchTST')]
mods += [U.NFPooled(a, covariaveis=False) for a in ('TFT', 'NHITS', 'TiDE')]
mods += [U.STID(pool=True), U.STID(pool=False)]
mods += [U.GCNGRU(), U.GCNGRU(atencao=True), U.STGCN(), U.MTGNN()]
mods += [U.Chronos2(False), U.Chronos2(True)]
if USAR_TIMESFM:
    mods.append(U.TimesFM())
MODELOS = {m.nome: m for m in mods}
PLANO_COMPLETO = {m.nome: TODAS for m in mods}
RODAR_AGORA = list(PLANO_COMPLETO)      # ex.: [k for k in PLANO_COMPLETO if 'pool' in k]
PLANO = {k: v for k, v in PLANO_COMPLETO.items() if k in RODAR_AGORA}
print(list(PLANO))
"""),
    ('code', LACO),
])

# =====================================================================================================================
salva('CompUF4_avaliacao.ipynb', [
    ('markdown', "# Modelos direto por UF · 4/4 · escolha (2018–2021) → teste (2022–2025)\n" + CABECALHO + r"""
**Protocolo (definido antes de olhar o teste):**
1. Tudo é escolhido nas temporadas de **afinação (2018–2021)**, com a régua **R vs climatologia** (o BB oficial só existe para 2022–2025).
2. Entram na escolha só modelos com as 4 temporadas de afinação.
3. Para cada um dos 15 melhores, escolhe-se a **largura dos intervalos** (λ) que minimiza o R na afinação.
4. Candidatos finais: o melhor modelo sozinho e a **mediana dos quantis dos k melhores** (k = 3, 5, 7), misturando "UF direto" e "região→UF".
5. **O final é o de menor R na afinação.** Só depois disso o teste (2022–2025, contra o BB e as 28 equipes do 3º IMDC) é calculado.

Lê `resultados_uf/` (UF direto) e `resultados_competicao/` (região→UF).
"""),
    ('code', "NOTEBOOK = 'uf_avaliacao'\nPACOTES = ''\n" + SETUP),
    ('markdown', "## 1. Previsões disponíveis"),
    ('code', r"""
# Modelos de FORA (outra equipe, outro notebook, um CSV no formato da plataforma: date, adm_1, pred, lower_*, upper_*):
# liste aqui {nome: arquivo}. Eles são checados, gravados em resultados_externos/ e entram em tudo como os nossos.
EXTERNOS = {}            # ex.: {'EquipeX': os.path.join(PASTA_DRIVE, 'externos', 'equipex.csv')}
PASTA_EXT = os.path.join(PASTA_DRIVE, 'resultados_externos')
for nome, arq in EXTERNOS.items():
    print(nome, '→', L.importar_externo(arq, nome, PASTA_EXT, ctx.Du.ufs).to_dict('records') or 'ok')
# Todas as pastas resultados_uf* (sessões abertas ao mesmo tempo podem criar cópias com o mesmo nome no Drive;
# renomeie a cópia para resultados_uf_2, _3 … e ela entra aqui).
PASTAS_UF = [p for p in sorted(glob.glob(os.path.join(PASTA_DRIVE, 'resultados_uf*'))) if os.path.isdir(p)]
print('Pastas de resultados por UF:', [os.path.basename(p) for p in PASTAS_UF])
# Leitura padronizada: duplicatas → fica a fonte listada por último; previsões incompletas ficam fora (relatório abaixo)
FONTES = [('região→UF', PASTA_COMP)] + [(f'UF direto · {os.path.basename(p)}', p) for p in PASTAS_UF] + [('externo', PASTA_EXT)]
PU = L.carregar_previsoes(FONTES, ctx.Du.ufs, 'uf')
PU['fonte'] = PU.fonte.str.replace(r' · resultados_uf.*', '', regex=True)
PU = PU[~PU.modelo.str.startswith('ENS')]
cob = PU.groupby(['fonte', 'modelo']).season.agg(lambda s: ''.join('●' if a in set(s) else '·' for a in TODAS))
print('temporadas', TODAS[0], '→', TODAS[-1], '(● = existe)')
cob.to_frame('2018 … 2025')
"""),
    ('markdown', "## 2. Afinação (2018–2021): R vs climatologia por UF"),
    ('code', r"""
por_t, res_t = V.avaliar(PU[PU.season.isin(TUNING)], ctx.ver_uf, ref=('climatologia',))
fonte = PU.drop_duplicates('modelo').set_index('modelo').fonte
res_t['fonte'] = res_t.modelo.map(fonte)
res_t.to_csv(os.path.join(PASTA_RES, 'afinacao_ranking.csv'), index=False)
ELEGIVEIS = res_t[res_t.temporadas == len(TUNING)].dropna(subset=['R_vs_climatologia']).sort_values('R_vs_climatologia')
print(len(ELEGIVEIS), f'modelos com as {len(TUNING)} temporadas de afinação completas')
assert len(ELEGIVEIS), 'nenhum modelo com as 4 temporadas de afinação completas (veja o relatório da seção 1)'
ELEGIVEIS[['fonte', 'modelo', 'R_vs_climatologia', 'UFs_melhor_que_climatologia', 'dp_logR_vs_climatologia',
           'WIS_log_medio', 'IC50', 'IC90']].round(3).head(40)
"""),
    ('markdown', "## 3. UF direto × região→UF (mesmo modelo, afinação)"),
    ('code', r"""
r = res_t.set_index('modelo').R_vs_climatologia
pares = [(m[:-3], m) for m in r.index if m.endswith('_UF') and m[:-3] in r.index]
cmp_ = pd.DataFrame([(a, r[a], r[b]) for a, b in pares], columns=['modelo', 'região→UF', 'UF direto'])
cmp_['melhor'] = np.where(cmp_['UF direto'] < cmp_['região→UF'], 'UF direto', 'região→UF')
cmp_.round(3).sort_values('UF direto')
"""),
    ('markdown', "## 4. Ablação: quanto cada grupo de variáveis ajuda (GBM LightGBM, afinação)\n> Razão > 1: tirar o grupo **piora** (o grupo ajuda). Razão < 1: tirar o grupo **melhora** (o grupo atrapalha)."),
    ('code', r"""
base = 'GBM_lgb_todas_UF'
if base in r.index:
    abl = pd.DataFrame([(g, r.get(f'GBM_lgb_todas_sem_{g}_UF', np.nan) / r[base]) for g in U.GRUPOS],
                       columns=['grupo retirado', 'R sem o grupo / R com todos'])
    display(abl.sort_values('R sem o grupo / R com todos', ascending=False).round(3))
else:
    print('Rode antes o CompUF2_CPU (modelo', base, ')')
"""),
    ('markdown', "## 5. Calibração da largura (λ escolhido na afinação)"),
    ('code', r"""
GRADE_LAM = [0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]
TOP = ELEGIVEIS.modelo.head(15).tolist()
clim = PU[(PU.modelo == 'climatologia') & PU.season.isin(TUNING)]
lam = {}
for m in TOP:
    g = PU[(PU.modelo == m) & PU.season.isin(TUNING)]
    rs = []
    for l in GRADE_LAM:
        _, rr = V.avaliar(pd.concat([U.ajusta_largura(g, l).assign(modelo='x'), clim]), ctx.ver_uf, ref=('climatologia',))
        rs.append(float(rr.set_index('modelo').loc['x', 'R_vs_climatologia']))
    lam[m] = GRADE_LAM[int(np.argmin(rs))]
    print(f'{m:40s} λ = {lam[m]:4.2f} · R: {rs[GRADE_LAM.index(1.0)]:.3f} → {min(rs):.3f}')
CAL = pd.concat([U.ajusta_largura(PU[PU.modelo == m], l).assign(modelo=f'{m}+λ') for m, l in lam.items() if l != 1.0]
                + [PU.iloc[:0]], ignore_index=True)
"""),
    ('markdown', "## 6. Candidatos e escolha final (só com a afinação)"),
    ('code', r"""
TODOS = pd.concat([PU, CAL], ignore_index=True)
_, res_c = V.avaliar(pd.concat([TODOS[TODOS.season.isin(TUNING) & TODOS.modelo.isin(TOP + list(CAL.modelo.unique()))],
                               clim]), ctx.ver_uf, ref=('climatologia',))
res_c = res_c[res_c.modelo != 'climatologia']
melhor_versao = {}
for m in TOP:                                         # versão (com ou sem λ) de cada modelo
    v = [x for x in (m, f'{m}+λ') if x in set(res_c.modelo)]
    melhor_versao[m] = min(v, key=lambda x: float(res_c.set_index('modelo').loc[x, 'R_vs_climatologia']))
ordem = sorted(melhor_versao.values(), key=lambda x: float(res_c.set_index('modelo').loc[x, 'R_vs_climatologia']))
ENS = []
for k in sorted({min(k, len(ordem)) for k in (3, 5, 7)} - {1}):    # com poucos modelos, não repete o mesmo ensemble
    e, _ = V.ensemble(TODOS[TODOS.modelo.isin(ordem[:k])], ordem[:k], f'ENS_UF_top{k}', 'mediana')
    ENS.append(e)
ENS = pd.concat(ENS + [PU.iloc[:0][L.COLS_PREV]], ignore_index=True)
# grava os ensembles e as versões com λ: o Comp5 (negócio) e avaliações futuras leem daqui
pd.concat([ENS, CAL], ignore_index=True)[L.COLS_PREV].to_parquet(os.path.join(PASTA_RES, 'ensembles_uf.parquet'),
                                                                 index=False)
CAND = pd.concat([TODOS[TODOS.modelo == ordem[0]], ENS], ignore_index=True)
_, res_f = V.avaliar(pd.concat([CAND, clim])[lambda d: d.season.isin(TUNING)], ctx.ver_uf, ref=('climatologia',))
res_f = res_f[res_f.modelo != 'climatologia'].sort_values('R_vs_climatologia')
FINAL = res_f.iloc[0].modelo
print('Membros por ordem (afinação):', ordem[:7])
print('>>> ESCOLHA FINAL (antes de olhar o teste):', FINAL)
res_f[['modelo', 'temporadas', 'R_vs_climatologia', 'dp_logR_vs_climatologia', 'IC50', 'IC90']].round(3)
"""),
    ('markdown', "## 7. Teste (2022–2025): contra o BB oficial e as 28 equipes do 3º IMDC"),
    ('code', r"""
TESTE_DF = pd.concat([TODOS, ENS], ignore_index=True)
TESTE_DF = TESTE_DF[TESTE_DF.season.isin(TESTE)]
por_s, res_s = L.avaliar_uf(ctx, TESTE_DF, temporadas=TESTE)
res_s['destaque'] = np.where(res_s.modelo == FINAL, '◀ FINAL', np.where(res_s.modelo.str.startswith('IMDC:'), '3º IMDC', ''))
res_s = res_s[res_s.temporadas == len(TESTE)]
res_s.to_csv(os.path.join(PASTA_RES, 'teste_ranking.csv'), index=False)
res_s[['destaque', 'modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'dp_logR_vs_BB', 'R_vs_climatologia', 'IC50', 'IC90']].round(3).head(45)
"""),
    ('code', r"""
# @title Por temporada (o final e as referências)
ref = [FINAL, 'ARIMA_Fourier', 'climatologia', 'IMDC:emap_epidematicos_sarimax_state', 'IMDC:lncc_lncc_arp26_dengue']
t = por_s[por_s.modelo.isin(ref)]
b = por_s[por_s.modelo == 'BB'][['season', 'uf_code', 'WIS']].rename(columns={'WIS': 'WIS_BB'})
t = t.merge(b, on=['season', 'uf_code'])
t.assign(lr=np.log(t.WIS / t.WIS_BB)).groupby(['modelo', 'season']).lr.mean().pipe(np.exp).unstack().round(3)
"""),
    ('code', r"""
# @title Gravar a previsão final (para o Comp5 de negócio e para a submissão)
fin = pd.concat([TODOS, ENS])[lambda d: d.modelo == FINAL].copy()
fin.to_parquet(os.path.join(PASTA_RES, 'final_uf.parquet'), index=False)
arqs = V.exportar_submissao(fin, FINAL, os.path.join(PASTA_RES, 'submissao'))
print('final:', FINAL, '·', len(fin), 'linhas · arquivos de submissão:', arqs)
V.grafico_leque(pd.concat([fin, PU[PU.modelo.isin(['climatologia', 'ARIMA_Fourier'])]]), ctx.ver_uf,
                [FINAL, 'ARIMA_Fourier', 'climatologia'], TESTE[0],
                arquivo=os.path.join(PASTA_RES, 'graficos', f'leque_final_{TESTE[0]}.png'))
"""),
])
