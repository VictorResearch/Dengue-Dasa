"""Gera os notebooks da competição (rode: python _gerar_notebook_comp.py).

  Comp1_CPU.ipynb        10 modelos estatísticos/ML/mecanísticos, 10 temporadas · ambiente CPU (~1–1,5 h)
  Comp2_INLA.ipynb       BB_INLA e GHR_INLA, 2020–2025 · ambiente CPU com bastante RAM (tempo a medir)
  Comp3_GPU.ipynb        XGBoost, DeepAR, NHITS, TFT, TSMixer, Chronos, AGCRN · GPU L4 (~5–8 h)
  Comp4_Avaliacao.ipynb  rankings por UF e por região, gráficos, diagnóstico e ensembles (roda a qualquer momento)

Os três primeiros podem rodar AO MESMO TEMPO, em sessões separadas do Colab (cada um grava arquivos próprios).
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

CABECALHO = r"""
**Antes de rodar (uma vez só):** em `MyDrive/dengue_V4/` (a pasta do V4, que já tem `dados/` e `referencias_imdc3/`) suba `comp_lib.py`, `comp_modelos.py` e `comp_inla.R`, e **substitua o `v4_lib.py`** pela versão nova de `data_imdc_2026/V4/`.
Todos os resultados vão para `MyDrive/dengue_V4/resultados_competicao/`. Se a sessão cair, rode de novo: o que já foi feito é pulado.

| Notebook | Modelos | Ambiente |
|---|---|---|
| `Comp1_CPU` | climatologia, canal_endemico, AR_longo, SURGE, ARIMA_Fourier, ETS_MSTL, Prophet, LightGBM_direto, Renovacao_S, Tamanho_x_forma | CPU |
| `Comp2_INLA` | BB_INLA, GHR_INLA | CPU com bastante RAM |
| `Comp3_GPU` | XGBoost_quantis, DeepAR, NHITS, TFT, TSMixer, Chronos, AGCRN | GPU L4 |
| `Comp4_Avaliacao` | rankings, gráficos, diagnóstico, ensembles | qualquer (CPU) |

Os notebooks 1, 2 e 3 podem rodar **ao mesmo tempo**, cada um numa sessão. O 4 pode rodar a qualquer momento: ele avalia o que já existir.
"""

SETUP = r"""
# @title Drive, arquivos e pacotes
import os, sys, shutil, time, glob, subprocess, json, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/comp', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'comp_inla.R', 'dados', 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou subir {item} para {PASTA_DRIVE}'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
    if PACOTES:
        r = subprocess.run(f'pip install -q {PACOTES}', shell=True, capture_output=True, text=True)
        print(r.stdout[-400:], r.stderr[-400:])
sys.path.insert(0, PASTA_LOCAL)
PASTA_RES = os.path.join(PASTA_DRIVE, 'resultados_competicao')
os.makedirs(os.path.join(PASTA_RES, 'graficos'), exist_ok=True)
import importlib, v4_lib as V, comp_lib as L, comp_modelos as CM
for m in (V, L, CM):
    importlib.reload(m)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma', '· CPUs:', os.cpu_count())
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
ctx = L.Contexto(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'))
TODAS = list(range(2016, 2026))          # 2016–2021: calibração dos pesos do ensemble · 2022–2025: S1–S4
RECENTES = list(range(2020, 2026))       # para os modelos mais caros
OFICIAIS = V.TEMPORADAS_OFICIAIS
M = 500                                  # trajetórias por região
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
                print(f'✅ {nome:18s} {s}: {dt / 60:5.1f} min', flush=True)
        except Exception as e:
            erros.append((nome, s, repr(e)))
            print(f'❌ {nome:18s} {s}: {repr(e)[:400]}', flush=True)
            traceback.print_exc(limit=2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
print('Erros:', len(erros))
pd.DataFrame(erros, columns=['modelo', 'temporada', 'erro']).to_csv(os.path.join(PASTA_RES, f'erros_{NOTEBOOK}.csv'), index=False)
t = L.tempos(PASTA_RES)
t[t.modelo.isin(list(PLANO))].groupby('modelo').segundos.agg(['count', 'mean', 'sum']).div([1, 60, 60]).round(1) \
    .rename(columns={'count': 'temporadas', 'mean': 'min/temporada', 'sum': 'min total'})
"""


def salva(nome, celulas):
    nb = {'cells': [], 'metadata': {'accelerator': 'GPU' if 'GPU' in nome else 'None',
                                    'colab': {'provenance': [], **({'gpuType': 'L4'} if 'GPU' in nome else {})},
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
salva('Comp1_CPU.ipynb', [
    ('markdown', "# Competição · 1/4 · modelos de CPU (10 modelos, 10 temporadas)\n" + CABECALHO +
     "\n**Ambiente:** CPU (não gasta GPU). **Tempo estimado:** 1–1,5 h. Nos testes do PC, por temporada: climatologia e canal ~1 s, AR_longo ~10 s, SURGE ~3 s, ETS ~12 s, Prophet ~40 s, LightGBM ~1,3 min, ARIMA ~2 min, Renovação ~5 s, Tamanho×forma ~8 s."),
    ('code', "NOTEBOOK = 'cpu'\nPACOTES = 'statsforecast'\n" + SETUP),
    ('code', r"""
MODELOS = {m.nome: m for m in [CM.Climatologia(), CM.CanalEndemico(), CM.ARLongo(), CM.Surge(), CM.ARIMAFourier(),
                               CM.ETSMSTL(), CM.ProphetModelo(), CM.LightGBMDireto(), CM.RenovacaoS(), CM.TamanhoForma()]}
PLANO = {nome: TODAS for nome in MODELOS}
"""),
    ('code', LACO),
])

# =====================================================================================================================
salva('Comp2_INLA.ipynb', [
    ('markdown', "# Competição · 2/4 · modelos bayesianos em R-INLA (BB oficial e GHR/BSC)\n" + CABECALHO + r"""
**Ambiente:** CPU com bastante RAM (em *Alterar tipo de ambiente*, escolha **High-RAM**).

**Como está otimizado:**
- ajuste só com as linhas observadas (6 temporadas de treino por região de saúde);
- modo `compact` do INLA (o mais rápido) com estratégia `gaussian` e `eb`;
- amostras só dos efeitos, com o preditor montado no R.

**BB:** semana e ano por macrorregião do país (5), como no original. **GHR:** BYM2 no grafo das 435 regiões + semana e ano por UF + clima.

**Primeiro rode só 1 temporada** (`TESTE_RAPIDO = True`) para medir o tempo. Depois rode o resto. Na primeira sessão, o INLA é instalado no disco local do Colab e uma cópia compactada fica no Drive (`R_libs_inla.tar.gz`); nas próximas sessões ele é restaurado em ~1 min. A célula de instalação termina com um **teste do motor** (`MOTOR_OK`): se ele falhar, não adianta rodar o resto.
"""),
    ('code', "NOTEBOOK = 'inla'\nPACOTES = ''\n" + SETUP),
    ('code', r"""
# @title Instalar o R-INLA (só na primeira vez; cópia compactada fica no Drive)
# O Drive não deixa executar programas (o motor inla.mkl.run dá "Permission denied").
# Por isso o INLA roda do disco local do Colab (/content/R_libs); o Drive só guarda um .tar.gz,
# que preserva as permissões de execução.
R_LIBS = '/content/R_libs'; os.makedirs(R_LIBS, exist_ok=True)
os.environ['R_LIBS_USER'] = R_LIBS
TAR_DRIVE = os.path.join(PASTA_DRIVE, 'R_libs_inla.tar.gz')
ANTIGO_DRIVE = os.path.join(PASTA_DRIVE, 'R_libs')          # instalação antiga, direto no Drive


def tem_inla():
    r = subprocess.run(['Rscript', '-e', 'cat(requireNamespace("INLA", quietly=TRUE))'], capture_output=True, text=True)
    return 'TRUE' in r.stdout


def libera_execucao():
    b = os.path.join(R_LIBS, 'INLA', 'bin')
    if os.path.isdir(b):
        subprocess.run(['chmod', '-R', 'a+rx', b])


if shutil.which('Rscript') is None:
    print('Instalando o R (~2 min)...'); subprocess.run('apt-get -qq update && apt-get -qq install -y r-base-core > /dev/null', shell=True)
print(subprocess.run(['Rscript', '--version'], capture_output=True, text=True).stderr.split('\n')[0])
if not tem_inla() and os.path.exists(TAR_DRIVE):
    print('Restaurando o INLA do Drive (~1 min)...'); subprocess.run(['tar', '-xzf', TAR_DRIVE, '-C', '/content'])
if not tem_inla() and os.path.isdir(os.path.join(ANTIGO_DRIVE, 'INLA')):
    print('Copiando a instalação antiga do Drive para o disco local (~2 min)...')
    subprocess.run(f'cp -r "{ANTIGO_DRIVE}"/. "{R_LIBS}"/', shell=True)
libera_execucao()
if not tem_inla():
    print('Instalando o INLA (10–20 min, só na primeira vez)...')
    script = r'''
    lib <- Sys.getenv("R_LIBS_USER"); options(timeout = 3600, Ncpus = 4)
    options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"))
    install.packages(c("Matrix", "sp", "fmesher", "foreach", "withr", "MatrixModels"), lib = lib)
    ok <- function() requireNamespace("INLA", quietly = TRUE)
    try(install.packages("INLA", lib = lib, repos = c(getOption("repos"), INLA = "https://inla.r-inla-download.org/R/stable"), dependencies = FALSE))
    if (!ok()) try(install.packages("https://inla.r-inla-download.org/R/stable/src/contrib/INLA_25.10.19.tar.gz", lib = lib, repos = NULL, type = "source"))
    if (!ok()) try(install.packages("https://inla.r-inla-download.org/R/stable/src/contrib/INLA_25.06.07.tar.gz", lib = lib, repos = NULL, type = "source"))
    cat("INLA:", ok(), "\n")'''
    r = subprocess.run(['Rscript', '-e', script], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    libera_execucao()
print('INLA disponível:', tem_inla())

# Pacotes que o INLA usa em tempo de execução ('sn' é exigido pelo inla.posterior.sample)
EXTRAS = ['sn', 'mnormt', 'numDeriv', 'quantreg', 'SparseM', 'MatrixModels']
falta = subprocess.run(['Rscript', '-e', 'p <- commandArgs(TRUE); cat(p[!sapply(p, requireNamespace, quietly=TRUE)])', *EXTRAS],
                       capture_output=True, text=True).stdout.split()
if falta:
    print('Instalando pacotes auxiliares:', falta)
    subprocess.run(['Rscript', '-e', 'install.packages(commandArgs(TRUE), lib=Sys.getenv("R_LIBS_USER"), '
                    'repos="https://packagemanager.posit.co/cran/__linux__/jammy/latest")', *falta], capture_output=True)
    if os.path.exists(TAR_DRIVE):
        os.remove(TAR_DRIVE)                                  # regrava a cópia do Drive já com os auxiliares
if tem_inla() and not os.path.exists(TAR_DRIVE):
    print('Guardando cópia compactada no Drive para as próximas sessões...')
    subprocess.run(['tar', '-czf', '/content/R_libs_inla.tar.gz', '-C', '/content', 'R_libs'])
    shutil.copy('/content/R_libs_inla.tar.gz', TAR_DRIVE)

# Teste de fumaça: o motor do INLA precisa rodar de verdade (não basta o pacote carregar)
teste = r'''
suppressMessages(library(INLA)); set.seed(1)
d <- data.frame(y = rpois(50, 5), x = rnorm(50))
r <- inla(y ~ x, family = "poisson", data = d, inla.mode = "compact", control.compute = list(config = TRUE))
a <- inla.posterior.sample(5, r)
cat("MOTOR_OK", round(r$summary.fixed[1, "mean"], 2), "| amostras:", length(a), "| versão", as.character(packageVersion("INLA")), "\n")'''
r = subprocess.run(['Rscript', '-e', teste], capture_output=True, text=True)
print(r.stdout[-500:] if 'MOTOR_OK' in r.stdout else ('❌ O motor do INLA não rodou:\n' + r.stdout[-1500:] + r.stderr[-2500:]))
"""),
    ('code', r"""
TESTE_RAPIDO = True            # True: roda só a temporada 2022 do BB para medir o tempo; depois mude para False
kw = dict(anos=6, M=300, threads=os.cpu_count(), r_libs=R_LIBS, pasta_script=PASTA_LOCAL)
MODELOS = {'BB_INLA': CM.ModeloINLA('bb', **kw), 'GHR_INLA': CM.ModeloINLA('ghr', **kw)}
PLANO = {'BB_INLA': [2022]} if TESTE_RAPIDO else {'BB_INLA': RECENTES, 'GHR_INLA': RECENTES}
"""),
    ('code', LACO),
    ('code', r"""
# @title Tempo do INLA por temporada (modo usado e segundos de ajuste)
for f in sorted(glob.glob(os.path.join(PASTA_RES, 'log', '*INLA*.json'))):
    d = json.load(open(f)); print(d['modelo'], d['season'], round(d['segundos']), 's ·', d.get('tempo_R', '').replace('\n', ' '))
"""),
])

# =====================================================================================================================
salva('Comp3_GPU.ipynb', [
    ('markdown', "# Competição · 3/4 · modelos de GPU (XGBoost, redes neurais, Chronos, AGCRN)\n" + CABECALHO + r"""
**Ambiente:** GPU **L4**. Tempo estimado por temporada:
- XGBoost e Chronos: ~1–2 min;
- cada rede do neuralforecast: ~5–15 min;
- AGCRN com 2 sementes: ~30–50 min.

O total é de ~5–8 h. Dá para dividir em sessões com `RODAR_AGORA`.

**Ainda não testados no PC** (o PC não tem GPU): as quatro redes do neuralforecast e o Chronos. Um modelo que der erro fica registrado em `erros_gpu.csv` e os outros continuam. Mande o erro que eu corrijo.
"""),
    ('code', "NOTEBOOK = 'gpu'\nPACOTES = 'neuralforecast chronos-forecasting'\n" + SETUP),
    ('code', r"""
cfg_agcrn = dict(n_sementes=2, epocas_max=40, paciencia=6, frac_origens=0.5, n_traj=M)
MODELOS = {m.nome: m for m in [CM.XGBoostQuantis(), CM.Chronos(), CM.NeuralForecastModelo('DeepAR'),
                               CM.NeuralForecastModelo('NHITS'), CM.NeuralForecastModelo('TFT'),
                               CM.NeuralForecastModelo('TSMixer'), CM.AGCRN(**cfg_agcrn)]}
PLANO_COMPLETO = {'XGBoost_quantis': TODAS, 'Chronos': TODAS, 'DeepAR': RECENTES, 'NHITS': RECENTES, 'TFT': RECENTES,
                  'TSMixer': RECENTES, 'AGCRN': RECENTES}
RODAR_AGORA = list(PLANO_COMPLETO)      # ex.: ['XGBoost_quantis', 'Chronos', 'DeepAR'] numa sessão, o resto em outra
PLANO = {k: v for k, v in PLANO_COMPLETO.items() if k in RODAR_AGORA}
"""),
    ('code', LACO),
])

# =====================================================================================================================
AVAL = [
    ('markdown', "# Competição · 4/4 · avaliação, gráficos, diagnóstico e ensembles\n" + CABECALHO + r"""
Roda a qualquer momento e avalia o que já existir em `resultados_competicao/`.

- **Por UF (a tarefa oficial):** WIS relativo ao **BB oficial**, média geométrica nas 4 safras (<1 = melhor). Os estáveis do 3º IMDC ficaram em 0,87–0,92. As 28 equipes aparecem como `IMDC:`.
- **Por região de saúde:** WIS relativo à climatologia, nas 435 regiões.
- ⚠️ Compare só modelos com `temporadas = 4`.
- ⚠️ A S4 vai só até 08/03/2026 nos nossos dados.
"""),
    ('code', "NOTEBOOK = 'avaliacao'\nPACOTES = ''\n" + SETUP),
    ('markdown', "## 1. Ranking por UF"),
    ('code', r"""
PU = L.ler_resultados(PASTA_RES, 'uf')
print('Modelos com resultado:', sorted(PU.modelo.unique()))
por_uf, res_uf = L.avaliar_uf(ctx, PU, temporadas=OFICIAIS)
res_uf['origem'] = np.where(res_uf.modelo.str.startswith('IMDC:') | (res_uf.modelo == 'BB'), '3º IMDC', 'competição')
cols = ['origem', 'modelo', 'temporadas', 'R_vs_BB', 'UFs_melhor_que_BB', 'dp_logR_vs_BB', 'R_vs_climatologia',
        'WIS_log_medio', 'IC50', 'IC80', 'IC90', 'IC95']
res_uf.to_csv(os.path.join(PASTA_RES, 'ranking_uf.csv'), index=False)
por_uf.to_csv(os.path.join(PASTA_RES, 'wis_uf_temporada.csv'), index=False)
completos = res_uf[(res_uf.origem == 'competição') & (res_uf.temporadas == len(OFICIAIS))].modelo.tolist()
res_uf[cols].round(3).head(45)
"""),
    ('code', r"""
# @title Mapa de calor modelo × UF, estabilidade e ranking por macrorregião
import matplotlib.pyplot as plt
fig, H = L.mapa_calor(por_uf, completos, arquivo=os.path.join(PASTA_RES, 'graficos', 'mapa_calor_uf.png')); plt.show()
L.grafico_estabilidade(res_uf[res_uf.temporadas == len(OFICIAIS)], completos,
                       arquivo=os.path.join(PASTA_RES, 'graficos', 'estabilidade.png')); plt.show()
rk = L.ranking_macro(por_uf, completos)
rk.to_csv(os.path.join(PASTA_RES, 'ranking_macrorregiao.csv'), index=False)
rk[rk.posicao <= 3].pivot(index='posicao', columns='macro', values='modelo')
"""),
    ('code', r"""
# @title Alvos da temporada: erro no total e na semana do pico
alv, res_alv = L.alvos_temporada(ctx, PU, OFICIAIS)
res_alv.to_csv(os.path.join(PASTA_RES, 'alvos_temporada.csv'), index=False)
res_alv.sort_values('erro_abs_log_total').round(2)
"""),
    ('markdown', "## 2. Ranking por região de saúde"),
    ('code', r"""
PR = L.ler_resultados(PASTA_RES, 'regiao')
por_reg, res_reg = L.avaliar_regiao(ctx, PR, temporadas=OFICIAIS)
res_reg.to_csv(os.path.join(PASTA_RES, 'ranking_regiao.csv'), index=False)
por_reg.to_csv(os.path.join(PASTA_RES, 'wis_regiao_temporada.csv'), index=False)
res_reg[['modelo', 'temporadas', 'R_vs_climatologia', 'regioes_melhor_que_climatologia', 'dp_logR_vs_climatologia',
         'WIS_log_medio', 'IC50', 'IC90', 'IC95']].round(3)
"""),
    ('code', r"""
# @title Distribuição por região: log(WIS ÷ climatologia)
b = por_reg[por_reg.modelo == 'climatologia'][['season', 'unidade', 'WIS']].rename(columns={'WIS': 'Wc'})
t = por_reg[por_reg.modelo != 'climatologia'].merge(b, on=['season', 'unidade'])
t['lr'] = np.log((t.WIS + 1e-9) / (t.Wc + 1e-9))
g = t.groupby(['modelo', 'unidade']).lr.mean().reset_index()
ordem_m = list(g.groupby('modelo').lr.median().sort_values().index)
fig, ax = plt.subplots(figsize=(10, 0.35 * len(ordem_m) + 1.5))
dados_bp = [g[g.modelo == m].lr.clip(-2, 2) for m in ordem_m]
try:
    ax.boxplot(dados_bp, vert=False, tick_labels=ordem_m, showfliers=False)   # matplotlib ≥ 3.9
except TypeError:
    ax.boxplot(dados_bp, vert=False, labels=ordem_m, showfliers=False)
ax.axvline(0, ls=':', c='k'); ax.set_xlabel('log(WIS ÷ WIS da climatologia) por região (negativo = melhor)')
fig.tight_layout(); fig.savefig(os.path.join(PASTA_RES, 'graficos', 'boxplot_regioes.png'), dpi=120); plt.show()
"""),
    ('markdown', "## 3. Gráficos de previsão"),
    ('code', r"""
# @title Leque por UF: 4 melhores + climatologia, em cada safra
top = res_uf[(res_uf.origem == 'competição') & (res_uf.temporadas == len(OFICIAIS))].sort_values('R_vs_BB').modelo.head(4).tolist()
for s in OFICIAIS:
    L.grafico_leque(PU, ctx.ver_uf, list(ctx.Du.ufs), 'uf_code', top + ['climatologia'], s,
                    f'Temporada {s}/{s + 1} · UF', os.path.join(PASTA_RES, 'graficos', f'leque_uf_{s}.png')); plt.show()
"""),
    ('code', r"""
# @title Leque por região de saúde: 24 regiões mais populosas (melhor modelo × climatologia)
D = ctx.Dr
maiores = D.unidades[np.argsort(-D.POP[D.ultimo_obs])[:24]]
for s in OFICIAIS:
    L.grafico_leque(PR.drop(columns='uf_code'), ctx.ver_reg, list(maiores), 'unidade', [top[0], 'climatologia'], s,
                    f'Temporada {s}/{s + 1} · 24 maiores regiões de saúde',
                    os.path.join(PASTA_RES, 'graficos', f'leque_regiao_{s}.png')); plt.show()
"""),
    ('markdown', "## 4. Diagnóstico: o que melhorar em cada modelo"),
    ('code', r"""
diag = L.diagnostico(ctx, PU, por_uf, OFICIAIS)
diag = diag.merge(res_uf[['modelo', 'R_vs_BB']], on='modelo', how='left').sort_values('R_vs_BB')
diag.to_csv(os.path.join(PASTA_RES, 'diagnostico_modelos.csv'), index=False)
diag.round(2)
"""),
    ('code', r"""
# @title Fidelidade: nossa reimplementação × a previsão publicada pela equipe original
pares = {'BB_INLA': 'BB', 'AR_longo': 'IMDC:lncc_lncc_arp26_dengue', 'SURGE': 'IMDC:lncc_surge_model26_dengue',
         'ARIMA_Fourier': 'IMDC:emap_epidematicos_sarimax_fixed', 'Prophet': 'IMDC:emap_epidematicos_prophet_fixed',
         'GHR_INLA': 'IMDC:bsc_ghr', 'LightGBM_direto': 'IMDC:purdue_neuralearth', 'canal_endemico': 'IMDC:cornell_bentolab',
         'XGBoost_quantis': 'IMDC:emap_xgbsillas', 'DeepAR': 'IMDC:emap_lstm', 'Renovacao_S': 'IMDC:nus_nus-cerm',
         'Tamanho_x_forma': 'IMDC:ifgw_inframind-proteus', 'Chronos': 'IMDC:rki_rki_zki_ph'}
r = res_uf.set_index('modelo')
pd.DataFrame([(a, b, r.R_vs_BB.get(a), r.R_vs_BB.get(b)) for a, b in pares.items()],
             columns=['nossa versão', 'original (3º IMDC)', 'R_vs_BB nossa', 'R_vs_BB original']).round(3)
"""),
    ('markdown', r"""
## 5. Ensembles

Entram os modelos que têm **todas** as temporadas de 2020 a 2025.
- **mediana:** mediana dos quantis entre os membros.
- **pesos:** pesos calculados só com temporadas **anteriores** à prevista, encolhidos 50% para pesos iguais.
- **top-5 da temporada anterior:** mediana dos 5 melhores (WIS em log) na temporada anterior. É uma escolha honesta, que não olha a temporada prevista.
"""),
    ('code', r"""
com_recentes = [m for m in PU.modelo.unique()
                if set(RECENTES) <= set(PU[PU.modelo == m].season.unique()) and not m.startswith('ENS')]
print('Membros:', com_recentes)
assert len(com_recentes) >= 2, 'Ainda não há modelos suficientes com 2020–2025: rode antes os notebooks 1 a 3.'
ens = []
for met in ('mediana', 'pesos'):
    e, pw = V.ensemble(PU, com_recentes, f'ENS_{met}', met, ctx.ver_uf, ctx.Du)
    ens.append(e)
    if len(pw):
        display(pw[pw.season.isin(OFICIAIS)].round(2))
por_todas, _ = L.avaliar_uf(ctx, PU[PU.modelo.isin(com_recentes)], incluir_ref=False)
for s in OFICIAIS:
    ant = por_todas[(por_todas.season == s - 1)]
    top5 = ant.groupby('modelo').WIS_log.mean().nsmallest(5).index.tolist()
    g = PU[(PU.season == s) & PU.modelo.isin(top5)]
    datas = np.sort(g.date.unique())
    q = np.median(np.stack([V.de_df(g[g.modelo == m], datas, ctx.Du.ufs) for m in top5]), 0)
    ens.append(V.para_df(V.arruma_quantis(q), datas, ctx.Du.ufs, 'ENS_top5_temporada_anterior', s))
    print(s, '· top-5 da temporada anterior:', top5)
E = pd.concat(ens, ignore_index=True)
E.to_parquet(os.path.join(PASTA_RES, 'ensembles_uf.parquet'), index=False)
_, res_e = L.avaliar_uf(ctx, pd.concat([PU, E]), temporadas=OFICIAIS)
res_e = res_e[res_e.temporadas == len(OFICIAIS)]
res_e.to_csv(os.path.join(PASTA_RES, 'ranking_uf_com_ensembles.csv'), index=False)
res_e[['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'dp_logR_vs_BB', 'WIS_log_medio', 'IC50', 'IC90']].round(3).head(25)
"""),
    ('code', r"""
print('Arquivos em', PASTA_RES); print(sorted(os.listdir(PASTA_RES)))
"""),
]
salva('Comp4_Avaliacao.ipynb', AVAL)

# =====================================================================================================================
NEGOCIO = [
    ('markdown', "# Competição · 5 · métricas de negócio (compra de testes)\n" + r"""
**Pergunta:** quantos testes comprar para o período, de forma que **não falte**, mas também **não sobre demais**?

**Não precisa rodar nenhum modelo de novo.** Este notebook lê os quantis que os notebooks 1 a 4 já gravaram em `resultados_competicao/`, reconstrói trajetórias e calcula as métricas. **Ambiente:** CPU padrão; leva alguns minutos.

**Como a compra é decidida (problema do jornaleiro):** compra-se o quantil **τ\* = custo de faltar ÷ (custo de faltar + custo de sobrar)** da previsão do **total do período**. Exemplo: faltar custa 3× mais que sobrar → τ\* = 0,75 → compra-se o percentil 75 do total (≈ 25% de chance de faltar).

**O total do período não é a soma dos quantis semanais**, porque somar os P75 de cada semana superestima o total. Por isso o notebook transforma os quantis em trajetórias (o mesmo método da soma das regiões: amostras + reordenação de Schaake com a dependência histórica entre semanas e unidades) e soma as trajetórias.

| Métrica | O que mede | Bom quando |
|---|---|---|
| `custo_vs_climatologia` | custo real da compra (falta × custo de faltar + sobra × custo de sobrar) ÷ o mesmo custo usando a climatologia | **< 1** · **é a métrica para escolher o modelo** |
| `falta_freq_pct` | % das unidades (UF ou região) × temporadas em que **faltou** teste | perto de (1 − τ\*) × 100 |
| `falta_pct` / `sobra_pct` | testes que faltaram / sobraram, em % do total real | baixos |
| `WAPE_total_pct` | erro da mediana do total, em % do total real | baixo |
| `vies_total_pct` | a mediana do total está acima (+) ou abaixo (−) do real | perto de 0 |
| `MAE_semanal` | erro médio da mediana, em casos por unidade e semana (dominado pelas UFs grandes) | baixo |
| `WAPE_semanal_pct` | soma dos erros semanais ÷ soma dos casos (o "erro percentual" estável) | baixo |
| `sMAPE_semanal_pct` | erro percentual simétrico (0–200%); instável com zeros, ignora os pares em que real e previsto são 0 | baixo, mas leia com cuidado |
| `WIS_semanal`, `IC90_semanal_pct` | métrica do desafio e cobertura do intervalo de 90% | WIS baixo; IC90 ≈ 90 |

⚠️ Uma temporada só (por exemplo, só a S1) tem pouca amostra: 26 UFs. Confira depois com as 4 temporadas.
"""),
    ('code', "NOTEBOOK = 'negocio'\nPACOTES = ''\n" + SETUP),
    ('markdown', "## Configuração"),
    ('code', r"""
TEMPORADAS_AVAL = [2022]          # S1 = 2022 (SE41/2022 → SE40/2023). Para as 4: [2022, 2023, 2024, 2025]
PERIODOS = {                      # índices das semanas da temporada (0 = SE41); None = até o fim
    'temporada (SE41→SE40)': (0, None),
    '1º semestre (SE41→SE14)': (0, 26),
    '2º semestre (SE15→SE40)': (26, None),
}
CUSTO_FALTA, CUSTO_SOBRA = 3.0, 1.0      # faltar 1 teste custa 3× mais que sobrar 1 → τ* = 0,75
TESTES_POR_CASO = 1.0                    # converte casos notificados em testes (ajuste se souber a taxa real)
M_NEG = 500                              # trajetórias por unidade
INCLUIR_IMDC = True                      # equipes do 3º IMDC e BB oficial (só por UF; só nas temporadas S1–S4)
INCLUIR_ENSEMBLES = True                 # ensembles do Comp4 e do CompUF4 (só por UF)
PASTAS_EXTRA = ([p for p in sorted(glob.glob(os.path.join(PASTA_DRIVE, 'resultados_uf*'))) if os.path.isdir(p)]  # CompUF1–4
                + [os.path.join(PASTA_DRIVE, 'resultados_externos')])  # modelos de fora (L.importar_externo); pode não existir
# Escolha honesta do modelo para a empresa: escolhe em AFINACAO, confirma em TESTE (veja a seção 5).
# Para usá-la, ponha em TEMPORADAS_AVAL as temporadas das duas listas, ex.: list(range(2018, 2026)).
AFINACAO, TESTE = [2018, 2019, 2020, 2021], [2022, 2023, 2024, 2025]

TAU = CUSTO_FALTA / (CUSTO_FALTA + CUSTO_SOBRA)
TAU_GRADE = sorted(set(np.round(np.arange(0.05, 0.96, 0.05), 3)) | {round(TAU, 3)})
PASTA_NEG = os.path.join(PASTA_RES, 'negocio'); os.makedirs(PASTA_NEG, exist_ok=True)
SIGLA = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN',
         25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC',
         43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'}
print(f'τ* = {TAU:.3f} → compra-se o percentil {100 * TAU:.0f} do total previsto')
"""),
    ('code', r"""
# @title Funções: trajetórias a partir dos quantis e métricas por unidade
def template(D, s, M, a=0.85, seed=0):
    # igual a L.template_dependencia, mas para qualquer nível (região ou UF)
    rng = np.random.default_rng(seed + s)
    c = D.corte(s)
    W = len(V.datas_temporada(s))
    temps = [j for j in range(2010, s) if D.temporada_completa(j, c) and D.idx_temporada(j)[0] > 0]
    F = []
    for j in temps:
        x = D.LINC[D.idx_temporada(j)]
        F.append(np.vstack([x, np.repeat(x[-1:], max(0, W - len(x)), 0)])[:W])
    F = np.stack(F)
    A = (F - F.mean(0)) / (F.std(0) + 1e-6)
    js = rng.integers(0, len(temps), M)
    return (a * A[js] + np.sqrt(1 - a ** 2) * rng.standard_normal((M, W, D.N))).astype(np.float32)


def trajetorias(q, D, s, M):
    am = L.amostrar_quantis(q, M)                        # (M, W, N), marginais exatas de cada semana
    return L.schaake(am, template(D, s, M)[:, :q.shape[0]])


def metricas_unidades(q, Y, D, ids, s):
    # q (W, N, 9) e Y (W, N) da temporada inteira → DataFrame com uma linha por unidade e período
    traj = trajetorias(q, D, s, M_NEG)
    saida = []
    for per, (a, b) in PERIODOS.items():
        qq, yy, tt = q[a:b], Y[a:b], traj[:, a:b]
        obs = ~np.isnan(yy[:, 0])
        if obs.sum() == 0:
            continue
        med, yo = qq[obs, :, 4], yy[obs]
        err = med - yo
        den = np.abs(yo) + np.abs(med)
        tem = den > 0
        smape = np.where(tem, 2 * np.abs(err) / np.where(tem, den, 1), 0)
        cob = (yo >= qq[obs, :, 1]) & (yo <= qq[obs, :, 7])
        T = tt[:, obs].sum(1)                            # (M, N): total do período em cada trajetória
        d = pd.DataFrame({'unidade': ids, 'season': s, 'periodo': per, 'completo': bool(obs.all()),
                          'n_sem': int(obs.sum()), 'soma_y': yo.sum(0), 'soma_abs': np.abs(err).sum(0),
                          'soma_err': err.sum(0), 'soma_smape': smape.sum(0), 'n_smape': tem.sum(0),
                          'soma_wis': V.wis(qq[obs], yo).sum(0), 'n_cob90': cob.sum(0),
                          'y_total': yo.sum(0), 'mediana_total': np.median(T, 0)})
        qT = np.quantile(T, TAU_GRADE, axis=0)
        for k, t in enumerate(TAU_GRADE):
            d[f'q{t:.3f}'] = qT[k]
        saida.append(d)
    return pd.concat(saida, ignore_index=True)
"""),
    ('markdown', "## 1. Calcular (alguns minutos)"),
    ('code', r"""
# Leitura padronizada: aceita qualquer modelo que tenha gravado os 9 quantis (nossos, ensembles, externos).
# Duplicatas → fica a fonte listada por último; previsões incompletas ficam fora (comparação justa). Ver o relatório.
fontes_uf = [('competição', PASTA_RES)]
if INCLUIR_IMDC and ctx.ref is not None:
    fontes_uf.append(('3º IMDC', ctx.ref))
if INCLUIR_ENSEMBLES:
    fontes_uf += [('ensembles', os.path.join(p, 'ensembles_uf.parquet')) for p in [PASTA_RES] + PASTAS_EXTRA]
for p in PASTAS_EXTRA:
    fontes_uf += [(os.path.basename(p), p), ('final', os.path.join(p, 'final_uf.parquet'))]
PU = L.carregar_previsoes(fontes_uf, ctx.Du.ufs, 'uf')
PR = L.carregar_previsoes([('competição', PASTA_RES)], ctx.Dr.unidades, 'regiao')
PU, PR = PU[PU.season.isin(TEMPORADAS_AVAL)], PR[PR.season.isin(TEMPORADAS_AVAL)]

NIVEIS = {'UF': (PU, ctx.Du, ctx.Du.ufs, ctx.ver_uf), 'região': (PR, ctx.Dr, ctx.Dr.unidades, ctx.ver_reg)}
partes, pulados = [], []
for nivel, (P, D, ids, ver) in NIVEIS.items():
    for (m, s), g in P.groupby(['modelo', 'season']):
        datas = V.datas_temporada(s)
        q = V.de_df(g, datas, ids)
        if np.isnan(q).any():                         # semanas ou unidades faltando: fica de fora (comparação justa)
            pulados.append((nivel, m, s, int(np.isnan(q[..., 4]).sum())))
            continue
        d = metricas_unidades(V.arruma_quantis(q), ver.y(datas), D, ids, s)
        d.insert(0, 'modelo', m); d.insert(0, 'nivel', nivel)
        partes.append(d)
    print(f'{nivel}: {P.modelo.nunique()} modelos', flush=True)
U = pd.concat(partes, ignore_index=True)
U.to_parquet(os.path.join(PASTA_NEG, 'metricas_por_unidade.parquet'), index=False)
if pulados:
    print('Ficaram de fora (previsão incompleta):'); display(pd.DataFrame(pulados, columns=['nivel', 'modelo', 'season', 'células vazias']))
print('Linhas:', len(U), '· temporadas:', sorted(U.season.unique()))
"""),
    ('code', r"""
# @title Resumo por modelo (escolha do modelo = menor custo_vs_climatologia)
CHAVE = ['nivel', 'periodo', 'season', 'unidade']


def resumo(U, tau):
    c = f'q{tau:.3f}'
    U = U.copy()
    U['_falta'] = np.maximum(U.y_total - U[c], 0)
    U['_sobra'] = np.maximum(U[c] - U.y_total, 0)
    U['_custo'] = (CUSTO_FALTA * U._falta + CUSTO_SOBRA * U._sobra) * TESTES_POR_CASO
    U['_faltou'] = (U.y_total > U[c]).astype(float)
    U['_ae_tot'] = (U.mediana_total - U.y_total).abs()
    ref = U[U.modelo == 'climatologia'][CHAVE + ['_custo']].rename(columns={'_custo': '_custo_ref'})
    U = U.merge(ref, on=CHAVE, how='left')
    lin = []
    for (nv, per, m), g in U.groupby(['nivel', 'periodo', 'modelo']):
        gc = g[g.completo]
        gr = gc[gc._custo_ref.notna()]
        ytot = gc.y_total.sum()
        lin.append(dict(
            nivel=nv, periodo=per, modelo=m, temporadas=g.season.nunique(), unidades=g.unidade.nunique(),
            custo_vs_climatologia=gr._custo.sum() / gr._custo_ref.sum() if len(gr) else np.nan,
            falta_freq_pct=100 * gc._faltou.mean(), falta_pct=100 * gc._falta.sum() / ytot,
            sobra_pct=100 * gc._sobra.sum() / ytot, WAPE_total_pct=100 * gc._ae_tot.sum() / ytot,
            vies_total_pct=100 * (gc.mediana_total.sum() - ytot) / ytot,
            MAE_semanal=g.soma_abs.sum() / g.n_sem.sum(), WAPE_semanal_pct=100 * g.soma_abs.sum() / g.soma_y.sum(),
            sMAPE_semanal_pct=100 * g.soma_smape.sum() / max(g.n_smape.sum(), 1),
            vies_semanal_pct=100 * g.soma_err.sum() / g.soma_y.sum(),
            WIS_semanal=g.soma_wis.sum() / g.n_sem.sum(), IC90_semanal_pct=100 * g.n_cob90.sum() / g.n_sem.sum(),
            custo_total=gc._custo.sum()))
    return pd.DataFrame(lin).sort_values(['nivel', 'periodo', 'custo_vs_climatologia'])


R = resumo(U, round(TAU, 3))
R.to_csv(os.path.join(PASTA_NEG, f'resumo_tau{TAU:.2f}.csv'), index=False)
print(f'τ* = {TAU:.2f} · falta_freq ideal ≈ {100 * (1 - TAU):.0f}% · temporadas: {TEMPORADAS_AVAL}')
for (nv, per), g in R.groupby(['nivel', 'periodo'], sort=False):
    print(f'\n=== {nv} · {per} ===')
    display(g.drop(columns=['nivel', 'periodo', 'custo_total']).round(3).head(20))
"""),
    ('markdown', r"""
## 2. Os mesmos números para outros τ

Se você não sabe o custo de faltar em relação ao de sobrar, compare os modelos em vários τ. Um modelo que vence em todos é a escolha robusta.
"""),
    ('code', r"""
PER0 = list(PERIODOS)[0]
tabs = []
for t in (0.5, 0.75, 0.9):
    r = resumo(U, t)
    r = r[r.periodo == PER0][['nivel', 'modelo', 'custo_vs_climatologia', 'falta_freq_pct', 'sobra_pct']]
    tabs.append(r.set_index(['nivel', 'modelo']).add_suffix(f' τ={t}'))
T3 = pd.concat(tabs, axis=1).reset_index().sort_values(['nivel', 'custo_vs_climatologia τ=0.75'])
T3.to_csv(os.path.join(PASTA_NEG, 'comparacao_taus.csv'), index=False)
for nv, g in T3.groupby('nivel'):
    print(f'\n=== {nv} · {PER0} ==='); display(g.drop(columns='nivel').round(2).head(20))
"""),
    ('markdown', r"""
## 3. Gráficos

- **Falta × sobra (τ\*):** cada ponto é um modelo. O canto inferior esquerdo é o ideal. A linha tracejada marca a frequência de falta esperada (1 − τ\*).
- **Calibração do total:** para cada τ, a % de vezes que faltou. Um modelo bem calibrado fica sobre a diagonal (falta = 1 − τ). Abaixo dela, superestima e sobra teste; acima, subestima e falta teste.
"""),
    ('code', r"""
import matplotlib.pyplot as plt
for nv in R.nivel.unique():
    g = R[(R.nivel == nv) & (R.periodo == PER0)].dropna(subset=['custo_vs_climatologia'])
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    cor = np.where(g.modelo.str.startswith('IMDC:') | (g.modelo == 'BB'), 'lightgray',
                   np.where(g.modelo.str.startswith('ENS'), 'tab:purple', 'tab:red'))
    ax[0].scatter(g.sobra_pct, g.falta_freq_pct, c=cor, s=40)
    for _, r in g.nsmallest(12, 'custo_vs_climatologia').iterrows():
        ax[0].annotate(r.modelo.replace('IMDC:', ''), (r.sobra_pct, r.falta_freq_pct), fontsize=7)
    ax[0].axhline(100 * (1 - TAU), ls='--', c='k', lw=.8)
    ax[0].set(xlabel='testes que sobraram (% do total real)', ylabel='% de unidades em que faltou',
              title=f'{nv} · {PER0} · τ*={TAU:.2f} (vermelho = nossos; cinza = 3º IMDC; roxo = ensembles)')
    top = g.nsmallest(6, 'custo_vs_climatologia').modelo.tolist()
    if 'climatologia' not in top:
        top.append('climatologia')
    taus = [t for t in TAU_GRADE if 0.4 <= t <= 0.96]
    for m in top:
        u = U[(U.nivel == nv) & (U.periodo == PER0) & (U.modelo == m) & U.completo]
        ax[1].plot(taus, [100 * (u.y_total > u[f'q{t:.3f}']).mean() for t in taus], marker='o', ms=3, label=m)
    ax[1].plot(taus, [100 * (1 - t) for t in taus], 'k--', lw=1, label='ideal')
    ax[1].set(xlabel='τ (quantil comprado)', ylabel='% de vezes em que faltou', title='calibração do total do período')
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(PASTA_NEG, f'graficos_{nv}.png'), dpi=120); plt.show()
"""),
    ('markdown', "## 4. Compra sugerida × consumo real, por UF (melhor modelo)"),
    ('code', r"""
MODELO_COMPRA = None        # None = o de menor custo_vs_climatologia por UF; ou, por exemplo, 'ARIMA_Fourier'
g = R[(R.nivel == 'UF') & (R.periodo == PER0)].dropna(subset=['custo_vs_climatologia'])
m = MODELO_COMPRA or g.iloc[0].modelo
c = f'q{round(TAU, 3):.3f}'
for s in TEMPORADAS_AVAL:
    u = U[(U.nivel == 'UF') & (U.periodo == PER0) & (U.modelo == m) & (U.season == s)].copy()
    if not len(u):
        continue
    t = pd.DataFrame({'UF': u.unidade.map(SIGLA),
                      'real': (u.y_total * TESTES_POR_CASO).round(),
                      'compra (τ*)': (u[c] * TESTES_POR_CASO).round(),
                      'mediana prevista': (u.mediana_total * TESTES_POR_CASO).round()})
    t['diferença %'] = (100 * (t['compra (τ*)'] - t.real) / t.real.clip(lower=1)).round(0)
    t['faltou?'] = np.where(t.real > t['compra (τ*)'], 'FALTOU', '')
    t = t.sort_values('real', ascending=False)
    t.to_csv(os.path.join(PASTA_NEG, f'compra_por_uf_{m}_{s}.csv'.replace(':', '_').replace('/', '_')), index=False)
    print(f'\n{m} · temporada {s} · {PER0} · τ*={TAU:.2f} · faltou em {(t["faltou?"] != "").sum()} de {len(t)} UFs')
    print(f'Total Brasil: real {t.real.sum():,.0f} · compra {t["compra (τ*)"].sum():,.0f} '
          f'({100 * (t["compra (τ*)"].sum() / t.real.sum() - 1):+.0f}%)')
    display(t)
print('Arquivos em', PASTA_NEG, ':', sorted(os.listdir(PASTA_NEG)))
"""),
]
NEGOCIO += [
    ('markdown', "## 5. Escolha honesta para a empresa: escolhe na afinação, confirma no teste\n\n"
                 "Escolher o modelo pelas mesmas temporadas em que se mede o resultado dá números otimistas. Aqui o modelo é "
                 "escolhido pelo custo em `AFINACAO` e só depois o custo em `TESTE` é mostrado. Precisa de `TEMPORADAS_AVAL` com "
                 "as duas listas; só entram os modelos que rodaram em todas as temporadas de cada lista."),
    ('code', r"""
tem = set(U.season.unique())
if not (set(AFINACAO) <= tem and set(TESTE) <= tem):
    print('Ponha em TEMPORADAS_AVAL as temporadas de AFINACAO e TESTE, ex.: list(range(2018, 2026)), e rode de novo.')
else:
    ra = resumo(U[U.season.isin(AFINACAO)], round(TAU, 3))
    rt = resumo(U[U.season.isin(TESTE)], round(TAU, 3))
    ra, rt = ra[ra.temporadas == len(AFINACAO)], rt[rt.temporadas == len(TESTE)]
    col = ['custo_vs_climatologia', 'falta_freq_pct', 'sobra_pct', 'WAPE_total_pct']
    chave = ['nivel', 'periodo', 'modelo']
    H = ra[chave + col].merge(rt[chave + col], on=chave, suffixes=(' · afinação', ' · teste'))
    H.to_csv(os.path.join(PASTA_NEG, f'escolha_afinacao_teste_tau{TAU:.2f}.csv'), index=False)
    for (nv, per), g in H.groupby(['nivel', 'periodo'], sort=False):
        g = g.sort_values('custo_vs_climatologia · afinação')
        print(f'\n=== {nv} · {per} · escolhido na afinação: {g.iloc[0].modelo} ===')
        display(g.drop(columns=['nivel', 'periodo']).round(3).head(15))
"""),
]
NEGOCIO += [
    ('markdown', "## 6. As métricas entre si: elas escolhem o mesmo modelo?\n\n"
                 "Cada métrica ordena os modelos. Se a correlação entre as ordens for alta, tanto faz qual otimizar; se for baixa, "
                 "a escolha da métrica muda o modelo escolhido, e para a compra deve valer o **custo** (o da empresa). "
                 "`WIS_semanal` é a régua da competição; `custo_vs_climatologia` é a da compra."),
    ('code', r"""
METRICAS = {'custo_vs_climatologia': True, 'WIS_semanal': True, 'MAE_semanal': True, 'WAPE_semanal_pct': True,
            'sMAPE_semanal_pct': True, 'WAPE_total_pct': True, 'falta_pct': True, 'sobra_pct': True,
            'abs_vies_total_pct': True, 'erro_IC90': True}           # True = menor é melhor
for (nv, per), g in R.groupby(['nivel', 'periodo'], sort=False):
    g = g.dropna(subset=['custo_vs_climatologia']).copy()
    if len(g) < 4:
        continue
    g['abs_vies_total_pct'] = g.vies_total_pct.abs()
    g['erro_IC90'] = (g.IC90_semanal_pct - 90).abs()                    # distância da cobertura ideal
    rk = pd.DataFrame({m: g[m].rank(ascending=asc) for m, asc in METRICAS.items()}).set_index(g.modelo)
    print(f'\n=== {nv} · {per} · {len(g)} modelos ===')
    print('Correlação de Spearman entre as ordens (1 = escolhem igual):')
    display(rk.corr(method='spearman').round(2))
    print('Vencedor por métrica:')
    display(pd.DataFrame({m: [rk[m].idxmin()] for m in METRICAS}, index=['1º lugar']).T)
    rk.to_csv(os.path.join(PASTA_NEG, f'posicoes_por_metrica_{nv}_{per[:9].strip()}.csv'.replace(' ', '_')))
"""),
]
salva('Comp5_Negocio.ipynb', NEGOCIO)
