"""Gera os notebooks da RODADA 4 (rode: python _gerar_notebooks_r4.py). Ordem:

  R4_0_Sorotipo_SINAN.ipynb          (opcional, CPU, 30–60 min) baixa o SINAN (OpenDataSUS) e conta os sorotipos por UF × semana.
  R4_1_Detector_ampliado.ipynb       (CPU, 20–40 min) temperatura do mar (ERSST), variáveis novas, 4 métodos + média + conformes.
  R4_2_Mistura_DoisEstagios.ipynb    (CPU, 20–40 min) MIX_elastico_UF (+ revisões) e DoisEstagios_det_UF.
  R4_3_MTGNN_condicionado.ipynb      (GPU L4/A100, 2–4 h) MTGNN_cond_UF e MTGNN_aug_UF.     ← R4_2 e R4_3 em paralelo
  R4_4_Ensemble_Avaliacao.ipynb      (CPU, 15–30 min) ensemble por regime + calibração, régua no teste e gráficos.
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))

ANTES = r"""
**Antes de rodar (uma vez só)**, em `MyDrive/dengue_V4/` (que já tem tudo das rodadas 1 a 3):
1. suba **`r4_lib.py`** (pasta `data_imdc_2026/RODADA4/`) e confira que `r3_lib.py` e `r2_lib.py` estão lá;
2. as pastas `resultados_uf_r4`, `revisoes_r4`, `cache_r4` e `avaliacao_r4` são criadas sozinhas.

**Usa a rodada 3:** o normal (ENS_grafos3_λ) vem de `cache_r3/normal_ens_grafos3_bb_clim.parquet` e as misturas da R3 entram na comparação.

**Complementar:** nada das rodadas anteriores muda. As previsões oficiais (corte de junho) vão para `resultados_uf_r4/uf/`; as revisões de julho a outubro, para `revisoes_r4/uf/`; as tabelas, para `avaliacao_r4/` (com data e hora no nome e uma cópia `_ultimo`).

**Retomada:** se a internet ou a luz cair, reconecte e **rode tudo de novo**. Cada etapa grava o seu arquivo (mar, sorotipo por ano, variáveis, detector por método × corte, parâmetros, um arquivo por modelo × temporada, os pesos das redes a cada 5 épocas, um PNG por UF) e o que já existe é pulado.
"""

SETUP = r"""
# @title 1 · Drive, arquivos e pacotes
import os, sys, shutil, time, glob, subprocess, json, traceback
import numpy as np, pandas as pd
try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE, PASTA_LOCAL, NO_COLAB = '/content/drive/MyDrive/dengue_V4', '/content/r4', True
except ImportError:
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.'); NO_COLAB = False
if NO_COLAB:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'comp_lib.py', 'comp_modelos.py', 'uf_lib.py', 'semanal_lib.py', 'r2_lib.py', 'r3_lib.py',
                 'r4_lib.py', 'dados', 'dados_uf', 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Faltou subir {item} para {PASTA_DRIVE}'
        shutil.copytree(o, d, dirs_exist_ok=True) if os.path.isdir(o) else shutil.copy(o, d)
    PACOTES
sys.path.insert(0, PASTA_LOCAL)
PASTA_R4 = os.path.join(PASTA_DRIVE, 'resultados_uf_r4')
PASTA_REV4 = os.path.join(PASTA_DRIVE, 'revisoes_r4')
PASTA_C4 = os.path.join(PASTA_DRIVE, 'cache_r4')
PASTA_AV4 = os.path.join(PASTA_DRIVE, 'avaliacao_r4')
PASTA_R3 = os.path.join(PASTA_DRIVE, 'resultados_uf_r3')
PASTA_C3 = os.path.join(PASTA_DRIVE, 'cache_r3')
PASTA_AV2 = os.path.join(PASTA_DRIVE, 'avaliacao_r2')
PASTA_COMP = os.path.join(PASTA_DRIVE, 'resultados_competicao')
PASTA_EXT = os.path.join(PASTA_DRIVE, 'resultados_externos')
for p in (os.path.join(PASTA_R4, 'uf'), os.path.join(PASTA_REV4, 'uf'), PASTA_C4, os.path.join(PASTA_C4, 'ckpt'), PASTA_AV4):
    os.makedirs(p, exist_ok=True)
import importlib, v4_lib as V, comp_lib as L, comp_modelos as CM, uf_lib as U, semanal_lib as S, r2_lib as R2, r3_lib as R3, r4_lib as R4
for m in (V, L, CM, U, S, R2, R3, R4):
    importlib.reload(m)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
ctx = U.ContextoUF(os.path.join(PASTA_LOCAL, 'dados'), os.path.join(PASTA_LOCAL, 'referencias_imdc3'),
                   os.path.join(PASTA_LOCAL, 'dados_uf'))
D = ctx.Du
TUNING, TESTE = U.TUNING, U.TESTE
REFAZER = False
R4.PESO_ABS = 0.5     # critério misto = (1 − p)·R × referência + p·WIS absoluto relativo. Suba (ex.: 0.7) para priorizar a altura dos picos
"""


def setup(pacotes=''):
    linha = f"subprocess.run('pip install -q {pacotes}', shell=True)" if pacotes else 'pass'
    return SETUP.replace('PACOTES', linha)


# ------------------------------------------------------------------------------------------------- R4_0 sorotipo
NB0 = [
    ('markdown', "# R4 · 0 · Sorotipo do SINAN (opcional)\n" + ANTES +
     "\nCPU comum, **30–60 min** na primeira vez (depois é instantâneo). Rode **antes** do R4_1. Se pular, o detector segue sem o sorotipo.\n\n"
     "**O que faz:** baixa as notificações de dengue de 2010 a 2026 do OpenDataSUS (`DENGBRaa.csv.zip`, 15–190 MB por ano), lê só 5 colunas (UF de residência, semana dos sintomas, sorotipo e classificação) e conta, por UF × semana, os casos com sorotipo identificado (DENV-1 a DENV-4). Fica um arquivo por ano em `cache_r4/sorotipo/`. O zip é apagado depois de lido.\n\n"
     "**Por que:** a troca de sorotipo encontra uma população sem imunidade para ele. Em Singapura, a competição entre sorotipos somada ao clima previu os surtos grandes ([Finch et al. 2025](https://doi.org/10.1038/s41467-025-66411-6)). No Brasil, o DENV-3 voltou em 2023 ([Ferreira et al. 2026](https://doi.org/10.1016/j.lana.2026.101521)).\n\n"
     "**Cuidados (sem vazamento):**\n"
     "- o detector só usa semanas até **8 semanas antes** do corte, imitando o tempo até o resultado do exame entrar no sistema;\n"
     "- os arquivos são os **finais**, mais completos do que os que existiam na época. O ganho medido aqui é, portanto, um teto otimista;\n"
     "- só uma fração pequena dos casos tem sorotipo; por isso as frações de cada UF são puxadas para a fração nacional."),
    ('code', "NOTEBOOK = 'r4_sorotipo'\n" + setup()),
    ('code', r"""
# @title 2 · Baixar e contar (um arquivo por ano; retoma)
SOR = R4.baixar_sorotipo(range(2010, 2027), PASTA_C4)
SOR['ano'] = SOR.sem_pri // 100
tab = SOR.groupby('ano')[['DENV1', 'DENV2', 'DENV3', 'DENV4']].sum()
print('Casos com sorotipo identificado por ano de início dos sintomas:')
display(tab)
(tab.div(tab.sum(1), axis=0) * 100).round(1)
"""),
    ('code', r"""
# @title 3 · Sorotipo dominante por UF nos últimos anos (conferência)
SOR['uf'] = SOR.uf_code.map(R2.SIGLAS)
d = SOR[SOR.ano >= 2021].groupby(['uf', 'ano'])[['DENV1', 'DENV2', 'DENV3', 'DENV4']].sum()
(d.div(d.sum(1), axis=0) * 100).round(0).unstack('ano').fillna(0).astype(int)
"""),
]

# ------------------------------------------------------------------------------------------------- R4_1 detector
NB1 = [
    ('markdown', "# R4 · 1 · Detector ampliado\n" + ANTES +
     "\nCPU comum, 20–40 min. Rode depois do R4_0 (se for usar o sorotipo).\n\n"
     "**O que mudou em relação ao detector da R3:** o alvo é o mesmo (log do tamanho da temporada relativo ao normal da UF). Entram variáveis novas:\n\n"
     "| Grupo | Variáveis | Fonte |\n|---|---|---|\n"
     "| Estabilidade entre surtos e alerta precoce | autocorrelação, variância e as tendências de ambas nas últimas 26 semanas; dispersão; piso relativo | Rypdal 2019; Brett 2020 |\n"
     "| Imunidade da população | tamanho relativo das últimas 6 temporadas com esquecimento (τ = 1 e 3 temporadas); ataque acumulado em 3 temporadas | McGough 2021 |\n"
     "| Seca e clima observado | índice de seca tipo SPEI (chuva − evapotranspiração, 13 e 26 semanas); seca máxima no último ano; anomalias de temperatura, chuva e umidade | Lowe 2021 |\n"
     "| Clima previsto | SEAS5 emitida até o corte: anomalias de temperatura, chuva e umidade para 1–3 e 4–6 meses à frente | ECMWF (já no projeto) |\n"
     "| Mar | Índico (IOBW), Atlântico tropical norte e sul, Atlântico sudoeste e Niño 3.4 (ERSST v5): média de 3 meses e variação em 6 | Chen 2024; Beal 2025 |\n"
     "| Sorotipo (se o R4_0 rodou) | fração de cada sorotipo, sorotipo emergente, sorotipos antes raros | Finch 2025 |\n\n"
     "**Métodos:** os 3 da R3, mais **duas etapas** (primeiro P(ano grande), depois o tamanho dado grande ou não; Francisco 2024), mais a média. Cada um também ganha uma versão **conforme** (`+conf`), em que cada quantil é corrigido pelos erros fora da amostra das temporadas já terminadas. Isso alonga a cauda de cima quando o detector vinha errando os anos grandes para baixo.\n\n"
     "**Escolha, por corte:** fica o de menor pinball nas temporadas ≤ 2021, com os **anos grandes valendo 3×**. Só a afinação (2018–2021) tinha poucos anos grandes e premiava o LightGBM, que não extrapola: no teste de fumaça ele previu SC 2023 em ×9, quando o real foi ×206.\n\n"
     "**Sem vazamento:**\n"
     "- os sinais são escolhidos só com temporadas ≤ 2021 (as variáveis nacionais, iguais em todas as UFs, precisam de correlação maior: |ρ| ≥ 0,30);\n"
     "- o treino é em janela expansiva;\n"
     "- as anomalias de clima e do mar usam só anos anteriores;\n"
     "- o mar usa o mês anterior ao do corte (o último publicado)."),
    ('code', "NOTEBOOK = 'r4_detector'\n" + setup('netCDF4')),
    ('code', r"""
# @title 2 · Temperatura do mar (ERSST v5, NOAA; ~150 MB uma vez) e sorotipo (se houver)
try:
    OC = R4.indices_oceano(PASTA_C4)
except Exception:
    traceback.print_exc(); OC = None
SOR = R4.ler_sorotipo(PASTA_C4)
arq_ext = os.path.join(PASTA_C4, 'externos_r4.csv')
EXT = pd.read_csv(arq_ext) if os.path.exists(arq_ext) else None
print('mar:', None if OC is None else f'{OC.mes.min().date()} → {OC.mes.max().date()}',
      '· sorotipo:', None if SOR is None else f'{len(SOR)} linhas UF × semana', '· externos:', None if EXT is None else list(EXT.columns))
if OC is not None:
    display(OC.tail(8).round(2))
"""),
    ('code', r"""
# @title 3 · Variáveis e alvo (cache; para refazer, apague cache_r4/detector4_* e cache_r4/sinais4.json)
X = R4.tabela_detector4(ctx, list(range(2011, 2026)), PASTA_C4, OC, SOR, EXT)
NOVAS = [c for c in X.columns if c.startswith(('ews_', 'piso', 'imun', 'ataque', 'anom_s1', 'seca', 'clim_', 'seas_', 'sor_'))
         or c.split('_')[0] in R4.REGIOES_SST]
print(X.shape, '· variáveis novas:', len(NOVAS))
X[X.corte == 'jun'][NOVAS].describe().T.round(2)
"""),
    ('code', r"""
# @title 4 · Sinais escolhidos só com temporadas ≤ 2021
ARQ_SIN = os.path.join(PASTA_C4, 'sinais4.json')
SINAIS = {k: R4.selecionar_sinais4(X, k, ate_temporada=max(TUNING), arq=ARQ_SIN) for k in R4.CORTES}
t = pd.DataFrame({k: pd.Series(v['rho']) for k, v in R4.json_le(ARQ_SIN).items()}).round(2)
t['nova'] = t.index.isin(NOVAS)
t.sort_values('jun', key=abs, ascending=False)
"""),
    ('code', r"""
# @title 5 · Detector (janela expansiva) + conformes, avaliação e escolha por corte
P = R4.prever_detector4(ctx, X, SINAIS, list(range(2014, 2026)), PASTA_C4)
P.to_parquet(os.path.join(PASTA_C4, 'detector4_todos.parquet'))
TD = R4.avaliar_detector4(P)
PE = R4.pinball_escolha(P)
TD = TD.merge(PE, on=['metodo', 'corte'], how='left')
R4.salvar_tabela(TD, PASTA_AV4, 'detector4_avaliacao')
ESC = R4.escolher_detector4(PE)
R4.json_grava(ESC, os.path.join(PASTA_C4, 'detector4_escolhido.json'))
print('escolhido por corte (menor pinball nas temporadas ≤ 2021, anos grandes valendo 3×):', ESC)
display(PE.pivot_table(index='metodo', columns='corte', values='pinball_escolha').round(3))
# comparação com o detector escolhido na R3 (mesmas métricas)
E3 = R3.json_le(os.path.join(PASTA_C3, 'detector_escolhido.json'), {})
P3 = pd.concat([pd.read_parquet(os.path.join(PASTA_C3, f'detector_{m}_{k}.parquet')).assign(metodo=f'R3:{m}') for k, m in E3.items()]) if E3 else None
T3 = R4.avaliar_detector4(P3) if P3 is not None else None
esc4 = pd.concat([TD[(TD.metodo == m) & (TD.corte == k)].assign(metodo=f'R4:{m}') for k, m in ESC.items()])
CMP = pd.concat([T3, esc4]) if T3 is not None else esc4
R4.salvar_tabela(CMP, PASTA_AV4, 'detector_r3_x_r4')
CMP[CMP.grupo != 'anterior'].pivot_table(index=['corte', 'metodo'], columns='grupo',
    values=['pinball', 'corr_mediana', 'IC90', 'sensibilidade', 'precisao', 'vies_grandes_x']).round(2)
"""),
    ('code', r"""
# @title 6 · O que o detector disse em junho de cada temporada de teste (× o normal da UF)
_, DET = R4.detector_escolhido4(PASTA_C4)
DJ = DET[(DET.corte == 'jun') & DET.season.isin(TESTE)].copy()
DJ['uf'] = DJ.uf_code.map(R2.SIGLAS)
DJ['real_x'], DJ['previsto_x'], DJ['q95_x'] = np.exp(DJ.alvo), np.exp(DJ.pred), np.exp(DJ.upper_90)
TAB = DJ.pivot_table(index='uf', columns='season', values=['real_x', 'previsto_x', 'q95_x', 'p_grande']).round(2)
R4.salvar_tabela(TAB.reset_index(), PASTA_AV4, 'detector4_junho_por_uf')
TAB
"""),
]

# ------------------------------------------------------------------------------------------------- R4_2 mistura
NB2 = [
    ('markdown', "# R4 · 2 · MIX_elastico e DoisEstagios_det\n" + ANTES +
     "\nCPU comum, 20–40 min. **Rode depois do R4_1.** Pode rodar **ao mesmo tempo** que o R4_3 (GPU).\n\n"
     "- **`MIX_elastico_UF`:** corrige o defeito do `MIX_escala_UF` da R3. Ele acertava a altura do pico, mas inflava a entressafra (+156%) porque multiplicava a curva inteira pelo mesmo fator.\n"
     "  - Agora cada semana × UF cresce conforme a sua **elasticidade histórica**, que depende do **nível normal** daquela semana: quanto o log da incidência sobe quando a temporada é maior, medido nas temporadas já terminadas.\n"
     "  - No histórico até 2021, ela vale ~0,1 nas semanas quase sem casos e ~1 no pico. O total continua o do detector.\n"
     "  - O *w* (0,5, 0,75 ou 1: quanto confiar no detector) é escolhido na afinação pelo **critério misto** = ½ R × climatologia + ½ WIS absoluto relativo.\n"
     "- **`DoisEstagios_det_UF`:** o *retreino do item 1*.\n"
     "  - Tamanho: o do detector ampliado.\n"
     "  - Forma: sorteada entre temporadas passadas **parecidas em tamanho**, da própria UF (peso 1) ou da macrorregião (peso 0,4). Assim, quando o detector diz \"ano grande\", entram as formas dos anos grandes (início mais cedo, pico mais estreito).\n"
     "  - Ainda: deslocamento de ±2 semanas e ruído binomial negativo.\n"
     "- **Revisões:** o `MIX_elastico_UF` é refeito com o detector de jul/ago/set/out e vai para `revisoes_r4/`."),
    ('code', "NOTEBOOK = 'r4_mistura'\n" + setup()),
    ('code', r"""
# @title 2 · Normal (cache da R3) e detector ampliado
NORMAL = R4.ler_normal(ctx, PASTA_DRIVE, PASTA_AV2, PASTA_COMP, PASTA_EXT, PASTA_C3, TUNING + TESTE)
ESC, DET = R4.detector_escolhido4(PASTA_C4)
print(NORMAL.groupby('modelo').season.unique(), ESC)
"""),
    ('code', r"""
# @title 3 · w do MIX_elastico (afinação, critério misto; gravado em json)
W = R4.escolher_w_elastico(ctx, NORMAL, DET, PASTA_C4, M=500)
print('w escolhido:', W['w'])
pd.DataFrame(W['tabela'])
"""),
    ('code', r"""
# @title 4 · Gerar MIX_elastico_UF (jun + revisões) e DoisEstagios_det_UF (um arquivo por modelo × temporada)
TAREFAS = [('MIX_elastico_UF', c) for c in R4.CORTES] + [('DoisEstagios_det_UF', 'jun')]
for nome0, corte in TAREFAS:
    nome = nome0 if corte == 'jun' else f'{nome0}@{corte}'
    pasta = PASTA_R4 if corte == 'jun' else PASTA_REV4
    falta = [s for s in TUNING + TESTE if REFAZER or not os.path.exists(L._arq(pasta, 'uf', nome, s))]
    if not falta:
        continue
    t0 = time.time()
    if nome0 == 'MIX_elastico_UF':
        x = R4.gerar_mix_elastico(ctx, NORMAL, DET, W['w'], corte, falta, M=1000)
    else:
        x = R4.gerar_dois_estagios_det(ctx, DET, corte, falta, M=1000)
    if x is None:
        print('sem dados:', nome); continue
    R4.grava_modelo(x, pasta, nome)
    print(f'✅ {nome:28s} {sorted(x.season.unique())} · {time.time() - t0:.0f}s', flush=True)
"""),
    ('code', r"""
# @title 5 · Conferência rápida no teste (o veredito completo é no R4_4)
X = pd.concat([NORMAL, R4.ler_pastas(PASTA_R4)], ignore_index=True)
X = X[X.season.isin(TESTE) & X.modelo.isin(['ENS_grafos3_λ', 'BB', 'MIX_elastico_UF', 'DoisEstagios_det_UF'])]
R4.criterio(ctx, X, ref='BB').round(3)
"""),
]

# ------------------------------------------------------------------------------------------------- R4_3 MTGNN
NB3 = [
    ('markdown', "# R4 · 3 · MTGNN condicionado ao tamanho da temporada (GPU)\n" + ANTES +
     "\nGPU **L4** (ou A100), 2–4 h para 8 temporadas × 5 sementes. **Rode depois do R4_1.** Pode rodar **ao mesmo tempo** que o R4_2. Os pesos das redes são gravados a cada 5 épocas em `cache_r4/ckpt/` e o treino retoma de onde parou.\n\n"
     "**O que muda em relação à MTGNN_pesopop da rodada 2** (mesma rede, mesmas entradas, mesma perda ponderada pela população):\n"
     "1. **Condição:** duas entradas novas, o tamanho relativo da temporada principal da janela e um indicador de \"conhecido\". No treino, entra o tamanho real das temporadas já terminadas.\n"
     "2. **Cópias sem condição (20%):** a mesma rede também aprende a prever sem saber o tamanho. Essa versão é a `MTGNN_aug_UF`.\n"
     "3. **Aumento com anos grandes sintéticos (60%):**\n"
     "   - a temporada principal do futuro é multiplicada por *k* entre 2 e 8, com o passado igual e a condição + log *k*;\n"
     "   - assim a rede aprende a desenhar temporadas maiores que as já vistas, inclusive a forma do pico (Al Bannoud et al. 2026: até −80% de erro em São Paulo 2024);\n"
     "   - o teto de segurança sobe de 3× para 10× o maior valor já visto.\n"
     "4. **Previsão:**\n"
     "   - 24 cenários de tamanho tirados do detector ampliado (correlação de 0,6 entre UFs, porque anos grandes costumam ser regionais);\n"
     "   - a rede roda uma vez por cenário e as trajetórias são misturadas;\n"
     "   - resultado: `MTGNN_cond_UF`.\n\n"
     "**Sem vazamento:** em cada temporada, a rede é treinada só com dados até a SE25. A condição de treino usa apenas temporadas terminadas até lá. Na previsão, o tamanho vem do detector, que também só viu o passado."),
    ('code', "NOTEBOOK = 'r4_mtgnn'\n" + setup()),
    ('code', r"""
# @title 2 · Detector ampliado e configuração
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '⚠️ sem GPU (vai ser MUITO lento)')
ESC, DET = R4.detector_escolhido4(PASTA_C4)
SEMENTES = (1, 2, 3, 4, 5)
REDE = U.MTGNN(sementes=SEMENTES)
CK = os.path.join(PASTA_C4, 'ckpt')
NOMES = ('MTGNN_cond_UF', 'MTGNN_aug_UF')
"""),
    ('code', r"""
# @title 3 · Treinar e prever (uma temporada por vez; pula o que já existe)
for s in TUNING + TESTE:
    if not REFAZER and all(os.path.exists(L._arq(PASTA_R4, 'uf', n, s)) for n in NOMES):
        continue
    t0 = time.time()
    c, datas, idx, h = CM._horizontes(ctx, s)
    redes, dd = R4.treinar_cond(ctx, REDE, c, 68, CK, f's{s}')
    datas, saida = R4.prever_temporada_cond(ctx, redes, dd, REDE, s, DET)
    for nome, q in saida.items():
        R4.grava_modelo(V.para_df(V.arruma_quantis(q), datas, D.unidades, nome, s), PASTA_R4, nome)
    del redes, dd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f'✅ {s}: {list(saida)} · {(time.time() - t0) / 60:.1f} min', flush=True)
"""),
    ('code', r"""
# @title 4 · Conferência rápida no teste (o veredito completo é no R4_4)
NORMAL = R4.ler_normal(ctx, PASTA_DRIVE, PASTA_AV2, PASTA_COMP, PASTA_EXT, PASTA_C3, TUNING + TESTE)
X = pd.concat([NORMAL, R4.ler_pastas(PASTA_R4)], ignore_index=True)
X = X[X.season.isin(TESTE) & X.modelo.isin(['ENS_grafos3_λ', 'BB'] + list(NOMES))]
R4.criterio(ctx, X, ref='BB').round(3)
"""),
]

# ------------------------------------------------------------------------------------------------- R4_4 ensemble + avaliação
NB4 = [
    ('markdown', "# R4 · 4 · Ensemble por regime, avaliação e gráficos\n" + ANTES +
     "\nCPU comum, 15–30 min. **Rode depois do R4_2 e do R4_3.** Se o R4_3 ainda não terminou, roda com os membros que existirem e é só rodar de novo depois.\n\n"
     "- **Membros:** normal (ENS_grafos3_λ), MIX_escala_UF e AJ_continuo_UF (R3), MIX_elastico_UF, DoisEstagios_det_UF, MTGNN_cond_UF e MTGNN_aug_UF.\n"
     "- **Regimes:** a probabilidade de ano grande que o detector deu em junho define três faixas: baixa (< 0,35), média e alta (≥ 0,65).\n"
     "- **`ENS_regime_UF`:** em cada faixa, os pesos dos membros (média dos quantis) são escolhidos na afinação (2018–2021) pelo **critério misto**. Assim, em ano tranquilo pesa mais o normal; em ano de alarme, pesam mais os modelos de pico.\n"
     "- **`ENS_regime_UF+cal`:** por faixa, a largura e a posição da faixa de previsão são calibradas na afinação, que é a versão por faixas da calibração conforme. A posição mexe no máximo ×1,5.\n"
     "- **Validação honesta antes do teste:** os dois ensembles são refeitos 4 vezes, cada vez deixando uma temporada da afinação de fora. A **recomendada** é o modelo de menor critério misto na afinação, contando os ensembles só fora da amostra. Ela é decidida antes de olhar o teste.\n"
     "- **Régua no teste (2022–2025):**\n"
     "  - R × BB (régua oficial), critério misto contra o BB e WIS absoluto;\n"
     "  - fases (pico, intermediária, entressafra), erro na altura e na semana do pico, erro no total e cobertura;\n"
     "  - as revisões mensais do MIX_elastico.\n"
     "- **Gráficos:** 26 UFs + Brasil, com a recomendada, o normal e o BB, mais o que o detector disse em junho."),
    ('code', "NOTEBOOK = 'r4_avaliacao'\n" + setup()),
    ('code', r"""
# @title 2 · Ler tudo
NORMAL = R4.ler_normal(ctx, PASTA_DRIVE, PASTA_AV2, PASTA_COMP, PASTA_EXT, PASTA_C3, TUNING + TESTE)
R3M = R4.ler_pastas(PASTA_R3)
R3M = R3M[R3M.modelo.isin(['MIX_escala_UF', 'AJ_continuo_UF'])]
R4M = R4.ler_pastas(PASTA_R4)
R4M = R4M[~R4M.modelo.str.startswith('ENS_regime')]
REV = R4.ler_pastas(PASTA_REV4)
ESC, DET = R4.detector_escolhido4(PASTA_C4)
TODOS = pd.concat([NORMAL, R3M, R4M], ignore_index=True)
CAND = ['ENS_grafos3_λ', 'MIX_escala_UF', 'AJ_continuo_UF', 'MIX_elastico_UF', 'DoisEstagios_det_UF', 'MTGNN_cond_UF', 'MTGNN_aug_UF']
cnt = TODOS.groupby('modelo').season.nunique()
MEMBROS = [m for m in CAND if cnt.get(m, 0) == len(TUNING + TESTE)]
print('membros com as 8 temporadas:', MEMBROS, '\nfaltando:', [m for m in CAND if m not in MEMBROS])
"""),
    ('code', r"""
# @title 3 · Afinação (2018–2021): critério misto de cada modelo contra a climatologia
AF = R4.criterio(ctx, TODOS[TODOS.season.isin(TUNING) & TODOS.modelo.isin(MEMBROS + ['climatologia'])])
R4.salvar_tabela(AF, PASTA_AV4, 'afinacao_criterio_misto')
AF.round(3)
"""),
    ('code', r"""
# @title 4 · Ensemble por regime: validação na afinação (deixando uma temporada de fora), ajuste final e recomendada
XR = R4.com_regime(TODOS[TODOS.modelo.isin(MEMBROS + ['climatologia'])], DET)
CV = R4.cv_regime(ctx, XR, MEMBROS)                                   # fora da amostra dentro de 2018–2021
AFc = R4.criterio(ctx, pd.concat([TODOS[TODOS.season.isin(TUNING) & TODOS.modelo.isin(MEMBROS + ['climatologia'])], CV[L.COLS_PREV]]))
R4.salvar_tabela(AFc, PASTA_AV4, 'afinacao_criterio_misto_com_ensemble')
RECOMENDADA = AFc.modelo.iloc[0]                                      # decidida ANTES de olhar o teste
print('recomendada (menor critério misto na afinação; ensembles fora da amostra):', RECOMENDADA)
display(AFc.round(3))
PESOS, CAL = R4.ajustar_regime(ctx, XR[XR.season.isin(TUNING)], MEMBROS)   # ajuste final com as 4 temporadas
ENS, ENSc = R4.prever_regime(XR, MEMBROS, PESOS, CAL)
R4.json_grava(dict(membros=MEMBROS, pesos=PESOS, calibracao=CAL, faixas=list(R4.FAIXAS), recomendada=RECOMENDADA),
              os.path.join(PASTA_AV4, f'ensemble_regime_{R4.carimbo()}.json'))
for nome, x in (('ENS_regime_UF', ENS), ('ENS_regime_UF+cal', ENSc)):
    R4.grava_modelo(x[L.COLS_PREV], PASTA_R4, nome)
print('pesos por faixa de P(ano grande):'); display(pd.DataFrame(PESOS).round(2))
print('calibração por faixa (λ = largura, δ = deslocamento em log):'); pd.DataFrame(CAL).round(3)
"""),
    ('code', r"""
# @title 5 · Régua no TESTE (corte de junho): geral, critério misto, fases, pico, total e cobertura
TODOS = pd.concat([TODOS, ENS[L.COLS_PREV], ENSc[L.COLS_PREV]], ignore_index=True)
OF = TODOS[TODOS.season.isin(TESTE)]
_, RES = V.avaliar(OF[L.COLS_PREV], ctx.ver_uf, ref=('BB', 'climatologia'))
RES = RES[RES.temporadas == len(TESTE)]
CRIT = R4.criterio(ctx, OF, ref='BB')[['modelo', 'WIS_rel', 'criterio']].rename(columns={'WIS_rel': 'WIS_abs_rel_BB', 'criterio': 'criterio_misto_BB'})
TT = S.com_verdade(OF, D)
pont = pd.DataFrame([dict(modelo=m, **S.metricas(g)) for m, g in TT.groupby('modelo')])
LIM = R2.limiares(D)
FASE = R2.metricas_fase(R2.com_fase(TT, D, LIM), ['modelo'])
PICO = R2.erros_pico(TT, 'season')
COB = R2.cobertura_quantis(TT)
f1 = FASE.pivot_table(index='modelo', columns='fase', values=['WIS', 'WIS_log', 'vies_pct', 'IC90'])
f1.columns = [f'{a}_{b}' for a, b in f1.columns]
TAB = RES[['modelo', 'R_vs_BB', 'UFs_melhor_que_BB', 'R_vs_climatologia']].merge(CRIT, on='modelo', how='left') \
    .merge(pont, on='modelo').merge(f1.reset_index(), on='modelo', how='left').merge(PICO, on='modelo', how='left').sort_values('criterio_misto_BB')
for nome, t in (('temporada_teste_r4', TAB), ('temporada_teste_por_fase_r4', FASE), ('temporada_teste_pico_r4', PICO),
                ('temporada_teste_cobertura_r4', COB)):
    R4.salvar_tabela(t, PASTA_AV4, nome)
cols = ['modelo', 'R_vs_BB', 'WIS_abs_rel_BB', 'criterio_misto_BB', 'WIS', 'WAPE', 'vies_pct', 'IC90', 'WIS_alta (pico)', 'vies_pct_alta (pico)',
        'vies_pct_entressafra', 'IC90_alta (pico)', 'erro_altura_mediano_pct', 'erro_total_mediano_pct', 'pico_no_IC90_pct']
TAB[[c for c in cols if c in TAB]].round(3)
"""),
    ('code', r"""
# @title 6 · Revisões mensais do MIX_elastico (R × BB e erro no total por corte)
lin = []
for corte in R4.CORTES:
    nome = 'MIX_elastico_UF' if corte == 'jun' else f'MIX_elastico_UF@{corte}'
    g = pd.concat([TODOS, REV])[lambda x: (x.modelo == nome) & x.season.isin(TESTE)]
    if g.empty:
        continue
    _, r = V.avaliar(pd.concat([g.assign(modelo='x'), NORMAL[NORMAL.modelo == 'BB']])[L.COLS_PREV], ctx.ver_uf, ref=('BB',))
    pk = R2.erros_pico(S.com_verdade(g.assign(modelo='x'), D), 'season')
    lin.append(dict(corte=corte, R_vs_BB=float(r.set_index('modelo').loc['x', 'R_vs_BB']),
                    erro_altura_mediano_pct=float(pk.erro_altura_mediano_pct.iloc[0]),
                    erro_total_mediano_pct=float(pk.erro_total_mediano_pct.iloc[0]), pico_no_IC90_pct=float(pk.pico_no_IC90_pct.iloc[0])))
REVT = pd.DataFrame(lin)
R4.salvar_tabela(REVT, PASTA_AV4, 'revisoes_mensais_r4')
REVT.round(3)
"""),
    ('code', r"""
# @title 7 · Gráficos (26 UFs + Brasil), PDF e totais por temporada
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import Image, display
PASTA_G = os.path.join(PASTA_AV4, f'graficos_r4_{R4.carimbo()[:8]}')
os.makedirs(PASTA_G, exist_ok=True)
OBS = pd.DataFrame(D.Y[:D.ultimo_obs + 1], index=pd.to_datetime(D.datas[:D.ultimo_obs + 1]), columns=D.unidades)
OBS[0] = OBS.sum(1)
PREV = {f'recomendada · {RECOMENDADA}': TODOS[TODOS.modelo == RECOMENDADA],
        'normal · ENS_grafos3_λ': TODOS[TODOS.modelo == 'ENS_grafos3_λ'],
        'modelo-base oficial · BB': TODOS[TODOS.modelo == 'BB']}
DJ = DET[DET.corte == 'jun']
ORDEM = [0] + sorted(D.unidades, key=lambda u: R2.SIGLAS.get(u, str(u)))
tot = []
for u in ORDEM:
    arq = os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png')
    if not os.path.exists(arq) or REFAZER:
        det = DJ[DJ.uf_code == u].set_index('season') if u != 0 else None
        fig = R3.grafico_temporada_uf(OBS[u], PREV, u, arq, TESTE, det=det,
                                      titulo=' · rodada 4' + (' · Brasil = soma das UFs' if u == 0 else ''))
        plt.close(fig)
    tot.append(R2.totais_temporada(OBS[u], PREV, u, TESTE))
with PdfPages(os.path.join(PASTA_G, 'graficos_UFs_r4.pdf.tmp')) as pdf:
    for u in ORDEM:
        img = plt.imread(os.path.join(PASTA_G, f'{R2.SIGLAS.get(u, u)}.png'))
        f = plt.figure(figsize=(13, 4.6)); plt.imshow(img); plt.axis('off'); pdf.savefig(f, bbox_inches='tight'); plt.close(f)
os.replace(os.path.join(PASTA_G, 'graficos_UFs_r4.pdf.tmp'), os.path.join(PASTA_G, 'graficos_UFs_r4.pdf'))
TOT = pd.concat(tot, ignore_index=True)
TOT.to_csv(os.path.join(PASTA_G, 'totais_temporada_r4.csv'), index=False)
print('Pasta:', PASTA_G)
for u in (0, 35, 31, 41, 42, 53):
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


def grava(nome, celulas, gpu=False):
    meta = {'colab': {'provenance': []}, 'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}}
    if gpu:
        meta.update(accelerator='GPU', colab={'provenance': [], 'gpuType': 'L4'})
    nb = {'cells': [celula(t, s) for t, s in celulas], 'metadata': meta, 'nbformat': 4, 'nbformat_minor': 0}
    with open(os.path.join(AQUI, nome), 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('gerado:', nome)


grava('R4_0_Sorotipo_SINAN.ipynb', NB0)
grava('R4_1_Detector_ampliado.ipynb', NB1)
grava('R4_2_Mistura_DoisEstagios.ipynb', NB2)
grava('R4_3_MTGNN_condicionado.ipynb', NB3, gpu=True)
grava('R4_4_Ensemble_Avaliacao.ipynb', NB4)
