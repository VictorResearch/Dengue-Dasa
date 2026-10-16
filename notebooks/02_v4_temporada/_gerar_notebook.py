"""Gera Modelo_v4.ipynb (rode: python _gerar_notebook.py)."""
import json
import os

C = []


def md(t):
    C.append({'cell_type': 'markdown', 'metadata': {}, 'source': t.strip('\n')})


def code(t):
    C.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': t.strip('\n')})


md(r"""
# Modelo V4 · previsão da temporada inteira (3rd IMDC)

Este notebook segue o plano do `_MAPA_DO_PROJETO/04_ESTADO_DA_ARTE.md`. A tarefa é a **oficial**: com dados só até a **SE25**, prever as semanas **SE41 → SE40** da temporada, com mediana e intervalos de 50/80/90/95%. A nota é o **WIS relativo ao modelo-base BB** do desafio.

| Etapa | O que faz | Onde |
|---|---|---|
| 1 | Avaliador oficial (WIS, conferido contra os valores publicados) + modelos-base + previsões publicadas do 3º IMDC para comparar | seção 1 |
| 5 | Variáveis de temporada: taxa de ataque de 1, 2 e 3 temporadas, incidência acumulada, tendência de 8 semanas, anomalia climática prolongada, SEAS5, ENSO, Richards (estilo Episcanner), vizinhos | seção 2 |
| 2 | **Versão A**: AGCRN direta multi-horizonte (1 a 68 semanas de uma vez). **Versão B**: tamanho da temporada (LightGBM) × forma da curva | seções 3 e 4 |
| 3 | Soma município/região → UF por **trajetórias correlacionadas** (já dentro de A e B) + recalibração opcional com temporadas passadas | seção 5 |
| 4 | Ensemble por quantis (média, mediana ou pesos encolhidos) | seção 6 |

**Antes de rodar:** suba para `MyDrive/dengue_V4/` o arquivo `v4_lib.py` e as pastas `dados/` e `referencias_imdc3/` (tudo sai de `F:\AmbientePython\Projetos\data_imdc_2026\V4\`).
**Ambiente:** GPU **L4** (melhor custo) ou A100. A T4 também funciona, só que mais devagar.
**Tempo:** a seção 1 e a versão B levam cerca de 10 min sem GPU. A versão A é o que consome GPU (estimativas na seção 4). Tudo o que fica pronto é salvo no Drive, e rodar de novo retoma do ponto em que parou.
""")

md("## 0. Ambiente e configuração")
code(r"""
# @title Montar o Drive, copiar para o disco local e importar a biblioteca
import os, sys, shutil, time, glob
import numpy as np, pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
    PASTA_DRIVE = '/content/drive/MyDrive/dengue_V4'
    PASTA_LOCAL = '/content/v4'
except ImportError:                      # rodando no PC
    PASTA_DRIVE = PASTA_LOCAL = os.path.abspath('.')

if PASTA_LOCAL != PASTA_DRIVE:
    os.makedirs(PASTA_LOCAL, exist_ok=True)
    for item in ('v4_lib.py', 'dados', 'referencias_imdc3'):
        o, d = os.path.join(PASTA_DRIVE, item), os.path.join(PASTA_LOCAL, item)
        assert os.path.exists(o), f'Não achei {o}. Suba a pasta V4 para o Drive.'
        if os.path.isdir(o):
            shutil.copytree(o, d, dirs_exist_ok=True)
        else:
            shutil.copy(o, d)
PASTA_DADOS = os.path.join(PASTA_LOCAL, 'dados')
PASTA_REF = os.path.join(PASTA_LOCAL, 'referencias_imdc3')
PASTA_SAIDA = os.path.join(PASTA_DRIVE, 'resultados_v4')          # no Drive: sobrevive a quedas da sessão
PASTA_PREV = os.path.join(PASTA_SAIDA, 'previsoes')
os.makedirs(PASTA_PREV, exist_ok=True)
sys.path.insert(0, PASTA_LOCAL)

try:
    import lightgbm
except ImportError:
    !pip -q install lightgbm
import importlib, v4_lib as V
importlib.reload(V)
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma (CPU)')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
""")

code(r"""
# @title Configuração
TEMPORADAS = list(range(2016, 2026))        # 2016–2021 = backtests (usados para pesos do ensemble); 2022–2025 = S1–S4
OFICIAIS = V.TEMPORADAS_OFICIAIS            # [2022, 2023, 2024, 2025]

RODAR_BASES = True
RODAR_B = True
RODAR_ABLACAO_B = True                      # etapa 5: tira um grupo de variáveis por vez (só B no nível UF, ~5 min)
RODAR_A_UF = True                           # Versão A com 26 nós (UF): barata
RODAR_A_REGIAO = True                       # Versão A com 435 nós (regiões de saúde): a mais cara
TEMPORADAS_A_UF = TEMPORADAS
TEMPORADAS_A_REGIAO = [2020, 2021, 2022, 2023, 2024, 2025]   # 2020–21 só para calibrar pesos do ensemble

CFG_B = dict(n_traj=1000, forma='media', desloc_max=3)       # melhor variante nos testes locais (ver LEIA-ME)
CFG_A = dict(n_sementes=2, epocas_max=40, paciencia=6, frac_origens=0.5, lote=8, retreino_final=True,
             n_traj=1000, L=104, H=68, hid=64, emb=16, lr=2e-3)
RECALIBRAR = False                          # etapa 3 opcional; nos testes locais piorou (ver seção 5)
REFAZER = False                             # True: ignora previsões já salvas e recalcula tudo


def arq_prev(modelo, s):
    return os.path.join(PASTA_PREV, f'{modelo.replace(":", "_").replace("+", "_")}_s{s}.parquet')


def ja_tem(modelo, s):
    return (not REFAZER) and os.path.exists(arq_prev(modelo, s))


def salvar_prev(df, modelo, s):
    df.to_parquet(arq_prev(modelo, s), index=False)


def todas_prev():
    fs = glob.glob(os.path.join(PASTA_PREV, '*.parquet'))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
""")

md(r"""
## 1. Etapa 1 · avaliador oficial e modelos-base

- **WIS** idêntico ao do 3º IMDC. Conferido localmente: bate com a coluna `wis` publicada pelos organizadores (razão 1,000 nas validações 1 e 2).
- **R_vs_BB** = média geométrica, entre UFs e temporadas, de WIS do modelo ÷ WIS do BB. **Abaixo de 1 é melhor que o modelo-base.** No 3º IMDC os modelos estáveis ficaram entre 0,87 e 0,92.
- **UFs_melhor_que_BB** = número de UFs em que o modelo venceu o BB (média geométrica das 4 safras).
- **dp_logR** = variabilidade do desempenho entre UFs e safras (menor = mais estável; é o eixo y do gráfico de estabilidade do desafio).
- ⚠️ A S4 (2025/26) só tem dados até 08/03/2026 no `dengue.csv`. Os organizadores avaliaram com mais semanas, então os números da S4 diferem dos oficiais.
""")
code(r"""
# @title Dados, verdade e previsões publicadas do 3º IMDC
Dr = V.Dados(PASTA_DADOS, 'regiao')
Du = V.Dados(PASTA_DADOS, 'uf')
ver = V.Verdade(Du)
REF = V.ler_referencias_imdc3(PASTA_REF, Du.ufs)
print('Previsões publicadas:', REF.modelo.nunique(), 'modelos')
""")
code(r"""
# @title Modelos-base (climatologia, canal endêmico NB, AR recursivo com trajetórias, AR longo por UF)
BASES = {'climatologia': V.base_climatologia, 'canal_endemico': V.base_canal_endemico,
         'ar_recursivo': V.base_ar_recursivo, 'ar_longo': V.base_ar_longo}
if RODAR_BASES:
    for nome, f in BASES.items():
        for s in TEMPORADAS:
            if ja_tem(nome, s):
                continue
            datas, q = f(Du, s)
            salvar_prev(V.para_df(q, datas, Du.ufs, nome, s), nome, s)
        print('ok', nome)
P = todas_prev()
_, res = V.avaliar(pd.concat([P[P.season.isin(OFICIAIS)], REF]), ver)
cols = ['modelo', 'temporadas', 'R_vs_BB', 'UFs_melhor_que_BB', 'dp_logR_vs_BB', 'R_vs_climatologia', 'WIS_log_medio', 'IC50', 'IC80', 'IC90', 'IC95']
res[cols].round(3).head(20)
""")

md(r"""
## 2. Etapa 5 · variáveis de temporada

Todas são calculadas **como vistas na SE25 do ano j**, para cada unidade (região de saúde ou UF):

| Grupo | Variáveis | Racional |
|---|---|---|
| Taxa de ataque 1, 2 e 3 temporadas | `ar1_parcial` (temporada que está terminando, até a SE25, com o pico), `ar2`, `ar3`, `ar4` | suscetíveis e imunidade (Lee 2021; NUS no 3º IMDC) |
| Incidência acumulada | `inc_acum_4temp`, `inc_acum_decaimento` (peso ½ por temporada), `inc_ano_se1_25` | imunidade acumulada |
| Tendência das últimas 8 semanas | `inc_ult8`, `inc_ult4`, `tend8` (inclinação do log), `pico_parcial_sem`, `razao_ult8_pico` | transmissão residual na entressafra (Lauer 2018) |
| Anomalia climática prolongada | anomalias padronizadas de temperatura (média e mínima), chuva e umidade em 12 e 26 semanas; semanas quentes e secas | Lowe 2021; Barcellos 2024 |
| SEAS5 | anomalia da previsão emitida em junho (jul–dez) | clima previsto sem vazamento |
| ENSO | média das últimas 12 semanas | ENSO com defasagem |
| Richards (estilo Episcanner) | tamanho, taxa de crescimento e semana do pico da temporada anterior (truncada na SE25) e da de 2 anos antes | calculados do próprio `dengue.csv` com o método do Episcanner (a API oficial exige chave; assim não há dado externo a declarar) |
| Vizinhos e UF | média dos vizinhos (vizinhança Queen agregada) e valores da UF | sincronia espacial |

A tabela abaixo mostra as variáveis da S1 (2022) e a correlação de cada uma com o alvo (log da taxa de ataque da temporada) nas temporadas de treino.
""")
code(r"""
# @title Olhar as variáveis (S1, nível região)
c = Dr.corte(2022); anom = V._anomalias_clima(Dr, c)
treino = [j for j in range(2012, 2022) if Dr.temporada_completa(j, c)]
X = pd.concat([V.features_temporada(Dr, j, c, anom).assign(alvo=V.alvo_temporada(Dr, j)) for j in treino])
corr = X.drop(columns=['uf_code', 'macro']).corr(method='spearman')['alvo'].drop('alvo').sort_values()
display(corr.to_frame('Spearman com o alvo').round(2).T)
V.features_temporada(Dr, 2022, c, anom).describe().T[['mean', 'std', 'min', 'max']].round(2)
""")

md(r"""
## 3. Etapa 2 · Versão B: tamanho × forma

1. O **LightGBM** prevê o log da taxa de ataque da temporada por unidade, com as variáveis da etapa 5.
2. Os **resíduos fora da amostra** (deixando uma temporada de fora de cada vez) são guardados como **vetor da temporada inteira**.
3. Cada trajetória sorteia uma temporada passada e usa o vetor de resíduos dela. Isso preserva a correlação real entre regiões: quando um ano surpreende, surpreende várias regiões juntas.
4. A **forma** da curva é a média das temporadas passadas (encolhida para a forma da UF quando a região tem poucos casos), deslocada ±3 semanas.
5. Há ruído de observação Gama-Poisson, e a soma das regiões dá a UF.

Roda nos dois níveis. Nos testes locais, o nível UF foi melhor.
""")
code(r"""
# @title Rodar a Versão B (regiões e UF)
if RODAR_B:
    for nivel, D in (('regiao', Dr), ('uf', Du)):
        nome = f'B_{nivel}'
        mb = V.ModeloB(**CFG_B)
        for s in TEMPORADAS:
            if ja_tem(nome, s):
                continue
            t0 = time.time()
            datas, q = mb.prever(D, s)
            salvar_prev(V.para_df(q, datas, Du.ufs, nome, s), nome, s)
            print(f'{nome} {s}: {time.time() - t0:.0f}s')
        if hasattr(mb, 'importancia_'):
            print(f'\nImportância (splits) — {nome}, última temporada rodada:')
            print(mb.importancia_.head(15).to_string())
P = todas_prev()
_, res = V.avaliar(pd.concat([P[P.season.isin(OFICIAIS)], REF[REF.modelo == 'BB']]), ver)
res[cols].round(3)
""")

code(r"""
# @title Etapa 5 · ablação: quanto cada grupo de variáveis ajuda (B no nível UF, todas as temporadas)
if RODAR_ABLACAO_B:
    linhas = []
    for grupo, vars_ in [('(nenhum: modelo completo)', [])] + list(V.GRUPOS_VARS.items()):
        nome = 'Babl_' + grupo.split(':')[0].strip('( ')
        exc = tuple(CFG_B.get('excluir_vars', ('macro', 'iod_12s', 'pdo_12s', 'enso_delta'))) + tuple(vars_)
        mb = V.ModeloB(**{**CFG_B, 'n_traj': 500, 'excluir_vars': exc})
        prev = []
        for s in TEMPORADAS:
            datas, q = mb.prever(Du, s)
            prev.append(V.para_df(q, datas, Du.ufs, nome, s))
        prev = pd.concat(prev)
        base = P[P.modelo == 'climatologia']
        _, r = V.avaliar(pd.concat([prev, base]), ver, ref=('climatologia',))
        rr = r[r.modelo == nome].iloc[0]
        linhas.append(dict(grupo_retirado=grupo, R_vs_climatologia=rr['R_vs_climatologia'], WIS_log=rr['WIS_log_medio']))
        print(linhas[-1])
    abl = pd.DataFrame(linhas)
    abl['efeito_de_retirar'] = abl.R_vs_climatologia / abl.R_vs_climatologia.iloc[0] - 1
    abl.to_csv(os.path.join(PASTA_SAIDA, 'ablacao_etapa5.csv'), index=False)
    display(abl.round(3))
    print('efeito_de_retirar > 0: retirar o grupo PIORA (o grupo ajuda). < 0: o grupo atrapalha.')
""")

md(r"""
## 4. Etapa 2 · Versão A: AGCRN direta multi-horizonte

- **Entrada:** as últimas 104 semanas de cada nó: log da incidência, anomalias de clima, semana do ano e ENSO. Também entram as variáveis de temporada "rolantes" da origem.
- **Saída:** 68 horizontes de uma vez. A rede prevê o **desvio em relação à âncora sazonal** (média histórica da mesma semana, usando só o passado), então começa igual à climatologia e aprende a se afastar dela.
- **Treino com origens sintéticas:** toda semana vira origem, e só entram alvos ≤ o corte.
- **Parada antecipada:** origens em torno da SE25 do ano anterior. Depois vem um retreino com todos os dados pelo mesmo número de épocas.
- **Incerteza:** Binomial Negativa por nó e horizonte, mais fatores multiplicativos compartilhados (nacional, por UF e por região). As trajetórias saem correlacionadas e são somadas até a UF.

**Custo estimado:** A_uf ≈ 2–4 min por temporada × 10 temporadas. A_regiao ≈ 15–30 min por temporada e semente numa L4, × 6 temporadas × 2 sementes, ou seja, 3–6 h. Com 190 unidades dá com folga: a L4 gasta ~4,8 UC/h. Cada temporada é salva assim que termina.
""")
code(r"""
# @title Rodar a Versão A (UF e regiões)
historicos = []
for nivel, D, temporadas, rodar in (('uf', Du, TEMPORADAS_A_UF, RODAR_A_UF), ('regiao', Dr, TEMPORADAS_A_REGIAO, RODAR_A_REGIAO)):
    if not rodar:
        continue
    nome = f'A_{nivel}'
    for s in temporadas:
        if ja_tem(nome, s):
            continue
        t0 = time.time()
        ma = V.ModeloA(**CFG_A, pasta_ckpt=PASTA_SAIDA)
        datas, q = ma.prever(D, s)
        salvar_prev(V.para_df(q, datas, Du.ufs, nome, s), nome, s)
        h = pd.concat(ma.historicos_).assign(modelo=nome, season=s)
        h.to_csv(os.path.join(PASTA_SAIDA, f'historico_{nome}_s{s}.csv'), index=False)
        print(f'✅ {nome} {s}: {(time.time() - t0) / 60:.1f} min')
        torch.cuda.empty_cache()
P = todas_prev()
_, res = V.avaliar(pd.concat([P[P.season.isin(OFICIAIS)], REF[REF.modelo == 'BB']]), ver)
res[cols].round(3)
""")

md(r"""
## 5. Etapa 3 · soma até a UF e recalibração

A soma até a UF **já está dentro** dos modelos:
- **B:** vetor de resíduos de uma temporada inteira sorteada, com a mesma forma em todas as regiões da trajetória.
- **A:** fatores multiplicativos compartilhados por país, UF e região.

Nos dois casos as regiões sobem e descem juntas dentro de cada trajetória, e a soma dessas trajetórias dá a variância certa na UF. É isso que o `k` da v3 fazia de forma aproximada.

**Recalibração (opcional, `RECALIBRAR`):** usa só temporadas passadas, com as semanas observáveis até o corte, e escolhe o fator das larguras que minimiza o interval score. Nos testes locais **piorou** o WIS. As temporadas de calibração (2016–21) eram de outro regime (anos baixos e covid), e a de 2024 exige intervalos enormes que sobram nas outras. Fica disponível para conferir com A.
""")
code(r"""
# @title (Opcional) recalibração com temporadas passadas
if RECALIBRAR:
    P = todas_prev()
    alvo = [m for m in P.modelo.unique() if not m.startswith('ENS') and '+' not in m]
    conf, rel = V.aplicar_conformal(P, ver, alvo, Du, sufixo='+larg', min_temporadas=3)
    for (m, s), g in conf.groupby(['modelo', 'season']):
        salvar_prev(g, m, s)
    display(rel[rel.season.isin(OFICIAIS)].round(2))
""")

md(r"""
## 6. Etapa 4 · ensemble

- **media**: média dos quantis (Vincentization).
- **mediana**: mediana dos quantis entre os membros (robusta a um membro ruim).
- **pesos**: pesos que minimizam o WIS nas temporadas **anteriores** à prevista (observadas até o corte), **encolhidos 50% para pesos iguais**. Sem pesos aprendidos numa temporada só, que foi a lição do IMDC 2024.

Os membros ausentes numa temporada fazem o ensemble daquela temporada ser pulado. Por isso `A_regiao` só entra nos conjuntos a partir de 2020.
""")
code(r"""
# @title Montar e avaliar ensembles
P = todas_prev()
P = P[~P.modelo.str.startswith('ENS')]
disponiveis = set(P.modelo.unique())
CONJUNTOS = {
    'ENS_bases': ['climatologia', 'canal_endemico', 'ar_recursivo', 'ar_longo'],
    'ENS_bases_B': ['climatologia', 'canal_endemico', 'ar_recursivo', 'ar_longo', 'B_uf'],
    'ENS_bases_A': ['climatologia', 'canal_endemico', 'ar_recursivo', 'ar_longo', 'A_uf'],
    'ENS_bases_AB': ['climatologia', 'canal_endemico', 'ar_recursivo', 'ar_longo', 'A_uf', 'B_uf'],
    'ENS_tudo': ['climatologia', 'canal_endemico', 'ar_recursivo', 'ar_longo', 'A_uf', 'A_regiao', 'B_uf', 'B_regiao'],
    'ENS_AB': ['A_uf', 'A_regiao', 'B_uf', 'B_regiao'],
}
ens, pesos_all = [], []
for nome, mem in CONJUNTOS.items():
    if not set(mem) <= disponiveis:
        print('pulando', nome, '— faltam', set(mem) - disponiveis)
        continue
    for met in ('media', 'mediana', 'pesos'):
        e, pw = V.ensemble(P, mem, f'{nome}_{met}', met, ver, Du)
        ens.append(e)
        if len(pw):
            pesos_all.append(pw.assign(ensemble=nome))
E = pd.concat(ens, ignore_index=True)
for (m, s), g in E.groupby(['modelo', 'season']):
    salvar_prev(g, m, s)
if pesos_all:
    display(pd.concat(pesos_all)[lambda d: d.season.isin(OFICIAIS)])
""")

md(r"""
## 7. Resultado final: o V4 contra o 3º IMDC

⚠️ **Viés de seleção:** escolher o melhor entre muitas variantes olhando as mesmas 4 safras deixa o número do vencedor otimista. Use como critério principal um modelo definido **antes** (por exemplo `ENS_tudo_pesos`, cujos pesos só usam o passado) e confira a estabilidade (`dp_logR`), não só a média.
""")
code(r"""
# @title Tabela final (S1–S4) com os concorrentes publicados
P = todas_prev()
por_uf, res = V.avaliar(pd.concat([P[P.season.isin(OFICIAIS)], REF]), ver)
res['origem'] = np.where(res.modelo.str.startswith('IMDC:') | (res.modelo == 'BB'), '3º IMDC', 'V4')
completo = res.temporadas == len(OFICIAIS)
print('⚠️ Modelos com menos de 4 safras (ex.: A_regiao se TEMPORADAS_A_REGIAO não cobrir 2022–2025) não são comparáveis:')
print(res.loc[~completo, ['modelo', 'temporadas']].to_string(index=False))
res = res[completo]
res.to_csv(os.path.join(PASTA_SAIDA, 'resumo_v4_vs_imdc.csv'), index=False)
por_uf.to_csv(os.path.join(PASTA_SAIDA, 'wis_por_uf_temporada.csv'), index=False)
display(res[['origem'] + cols].round(3).head(30))
""")
code(r"""
# @title Gráfico de estabilidade (igual ao do 3º IMDC) e leque do melhor modelo V4
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 6))
r_ = res.dropna(subset=['R_vs_BB'])
for _, r in r_.iterrows():
    v4 = r.origem == 'V4'
    ax.scatter(np.log(r.R_vs_BB), r.dp_logR_vs_BB, c='tab:red' if v4 else 'grey', s=40 if v4 else 20, alpha=.8)
    if v4 or r.R_vs_BB < 0.93:
        ax.annotate(r.modelo.replace('IMDC:', ''), (np.log(r.R_vs_BB), r.dp_logR_vs_BB), fontsize=7)
ax.axvline(0, ls=':', c='k'); ax.axhline(r_.dp_logR_vs_BB.median(), ls=':', c='k')
ax.set_xlabel('média de log(R) vs BB  (negativo = melhor que o modelo-base)'); ax.set_ylabel('desvio-padrão de log(R)  (menor = mais estável)')
ax.set_title('V4 (vermelho) × 3º IMDC (cinza)'); plt.show()
fig.savefig(os.path.join(PASTA_SAIDA, 'estabilidade_v4.png'), dpi=120)

melhor = res[(res.origem == 'V4')].sort_values('R_vs_BB').modelo.iloc[0]   # só modelos com as 4 safras
print('Melhor V4 nas safras oficiais:', melhor)
for s in OFICIAIS:
    V.grafico_leque(P, ver, [melhor, 'climatologia'], s, arquivo=os.path.join(PASTA_SAIDA, f'leque_{melhor}_{s}.png'))
    plt.show()
""")
code(r"""
# @title Exportar a submissão do modelo escolhido e as linhas para o metricas.csv
MODELO_ESCOLHIDO = melhor          # troque se preferir um critério definido antes (ex.: 'ENS_tudo_pesos')
arqs = V.exportar_submissao(P[P.season.isin(OFICIAIS)], MODELO_ESCOLHIDO, os.path.join(PASTA_SAIDA, 'submissao'))
print('Arquivos:', arqs)
lin = por_uf[por_uf.modelo == MODELO_ESCOLHIDO].groupby('season').agg(WIS=('WIS', 'mean'), WIS_log=('WIS_log', 'mean'),
      MAE=('MAE', 'mean'), IC50=('IC50', 'mean'), IC90=('IC90', 'mean'), IC95=('IC95', 'mean')).reset_index()
lin['temporada_S'] = lin.season.map(V.TEMPORADA_DE_VALIDACAO)
lin.to_csv(os.path.join(PASTA_SAIDA, f'metricas_{MODELO_ESCOLHIDO}.csv'), index=False)
display(lin.round(2))
""")

nb = {'cells': C, 'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'L4', 'provenance': []},
                               'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                               'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 0}
for c in nb['cells']:
    c['source'] = [l + '\n' for l in c['source'].split('\n')]
    c['source'][-1] = c['source'][-1].rstrip('\n')
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Modelo_v4.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Modelo_v4.ipynb gerado com', len(C), 'células')
