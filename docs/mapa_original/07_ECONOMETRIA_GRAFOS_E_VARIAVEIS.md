# 07 · Econometria, redes de grafo e seleção de variáveis (01/10/2026)

> **Pedido:** trocar a grade de 9 ARIMAs pelo **auto.arima** (R e Python), fazer o mesmo com o ARIMAX, testar **VAR** para prever várias UFs ao mesmo tempo, ler as pastas do Drive (*Links de estudo › Economics / Econometria* e *Articles*) e avaliar **quais variáveis** usar, com modelos comparáveis entre si.
> **Código:** `data_imdc_2026/UF/uf_lib.py`, `uf_autoarima.R`, `avaliar_variaveis.py`. **Notebooks:** `CompUF2_CPU` (estatísticos) e `CompUF3_GPU` (redes). Ver também [06](06_NIVEL_ESTADUAL_PESQUISA_E_PLANO.md).

---

## 1. O que foi lido no Drive (e o que não foi)

| Pasta | Lido | Observação |
|---|---|---|
| Econometria / Economics: ARIMA e SARIMA | `Forecasting_sARIMA`, `sarima`, *Seasonality and Seasonal-ARIMA* (SPUR), *SARIMA no R* (Oike), `Econometria_restateinsight` | Box-Jenkins, testes de raiz unitária, critérios de informação, diagnóstico dos resíduos |
| Econometria: livro-texto | `ITER.pdf` | **Leitura cortada na página 123.** Os capítulos de séries temporais (14–16) **não** vieram |
| VAR | *Chapter 12 VAR (Time Series Analysis With R)*, `VAR_Model`, *VAR using R* (R-bloggers), **Lütkepohl, *New Introduction to Multiple Time Series Analysis* (2005)** | Lütkepohl lido **até a página 95** (VAR estável, estimação, ordem, Granger, previsão). VAR restrito, VARX e BVAR vêm dos capítulos posteriores e da literatura geral |
| GARCH e finanças | vários tutoriais de GARCH, *Bauwens (2006) Multivariate GARCH*, *Handbook of Financial Time Series*, *Analysis of Financial Time Series* | **Não lidos** além do título: tratam de volatilidade de retornos financeiros, fora do nosso problema |
| Articles: grafos e LSTM | T-GCN, A3T-GCN, STGCN, MTGNN (*Connecting the Dots*), DCRNN, GMAN, ASTGCN, AGCRN, GCRN (*Structured Sequence Modeling*), *Transfer GNN for Pandemic Forecasting*, EvolveGCN, GC-LSTM, *Learning to Represent the Evolution of Dynamic Graphs* (DyGGNN), 2s-AGCN | *Predicting Temporal Sets* e *Predicting Path Failure* só **de passagem** (outro tipo de tarefa) |

### Lições aplicadas

1. **Deixar o algoritmo escolher, mas com critério de informação.** Hyndman-Khandakar (auto.arima): d pelo teste KPSS, p, q e drift pelo AICc. A grade manual de 9 combinações foi removida.
2. **Sazonalidade longa (52,18 semanas) não se modela com SARIMA de período 52.** É lento, instável e o período não é inteiro. A forma recomendada é a **regressão harmônica dinâmica**: termos de Fourier + erros ARIMA. O número de pares K também é escolhido pelo AICc (K = 1..6).
3. **Diferenciar uma série de 68 semanas à frente é arriscado.** Com d = 1 e drift, a tendência das últimas semanas é projetada por mais de um ano. No teste de fumaça, o auto.arima livre ficou ≈1,9 × BB; com d = 0 e sem drift, ≈1,1 (com K = 1). Por isso há as duas versões: a **livre** (o que foi pedido) e a **estacionária** (`_d0`, como o ARIMA_Fourier vencedor da competição).
4. **VAR com muitas séries explode em parâmetros.** Um VAR(p) com 26 UFs tem 26²·p coeficientes. Os livros indicam três saídas, e as três foram implementadas:
   - **blocos menores:** um VAR por macrorregião, com ordem pelo BIC;
   - **restrições:** VAR espacial, em que cada UF usa só as próprias defasagens e a média dos vizinhos de fronteira;
   - **encolhimento bayesiano:** BVAR com prior de Minnesota/Litterman.
5. **Exógenas precisam ter valor no futuro.** No ARIMAX e no VARX, o futuro das exógenas tem que ser conhecido ou previsto. Regra comum a todos: ENSO persistido da SE25 e anomalias de clima da SEAS5 por até 6 meses, depois 0.
6. **Diagnósticos clássicos** (Ljung-Box, ARCH, estabilidade, Granger) ficaram registrados como checagem para o modelo final. Não entram na escolha automática, que é feita pelo WIS fora da amostra em 2018–2021.

---

## 2. Modelos estatísticos novos (`CompUF2_CPU`)

| Modelo | Nome(s) | Como funciona |
|---|---|---|
| auto.arima | `AutoARIMA_py_UF`, `AutoARIMA_r_UF` | por UF: log1p(incidência) dos últimos 8 anos; para K = 1..6, `auto.arima(y, xreg = Fourier(K))`; fica o de menor AICc. `py` = statsforecast; `r` = `forecast::auto.arima` original, via `uf_autoarima.R` |
| auto.arima d = 0 | `AutoARIMA_d0_py_UF`, `AutoARIMA_d0_r_UF` | o mesmo, com d = 0 e sem drift (p, q e K continuam automáticos) |
| auto.ARIMAX | `AutoARIMAX_[d0_]{py,r}_UF` | + `EXOG_FUTURAS` (ENSO, anomalia de temperatura, anomalia de chuva) |
| VAR por macrorregião | `VAR_macro_UF`, `VAR_macroX_UF` | anomalias de log-incidência (log − climatologia da semana); um VAR por N/NE/SE/S/CO; p ≤ 4 pelo BIC (amostra comum, como o `vars::VARselect`); X = + ENSO |
| VAR espacial | `VAR_espacial_UF`, `VAR_espacialX_UF` | MQO por equação: p defasagens próprias + p defasagens da média dos vizinhos (matriz de fronteiras normalizada) |
| BVAR Minnesota | `BVAR_Minnesota_UF`, `BVAR_MinnesotaX_UF` | as 26 UFs juntas, p = 3; prior centrada em AR(1) com coeficiente δ = 0,8; encolhimento λ = 0,2 e θ = 0,5 (outras UFs encolhidas mais) |

Todos os VARs geram **trajetórias** simulando com a covariância dos resíduos. Assim a soma das UFs e o total da temporada saem com a correlação certa.

**Testes de fumaça (PC, 2022; NÃO são estimativas):** os 17 modelos novos rodaram, mais as 2 variantes `_d0`. Tempos: auto.arima py ~47 s, R ~17 s (Kmax = 2); VARs < 1 s; redes de grafo 4–33 s (1 época).

---

## 3. Redes de grafo (`CompUF3_GPU`)

Base comum (`_RedeGrafoUF`):
- as mesmas janelas de entrada;
- grafo de fronteiras normalizado (A + I);
- perda pinball nos 9 quantis;
- saída direta das 68 semanas: âncora + resíduo, com teto;
- 3 sementes, combinadas pela mediana.

Assim, a diferença entre as redes é só a arquitetura.

| Rede | Artigo | Adotada? | Por quê |
|---|---|---|---|
| T-GCN (`GCNGRU_UF`) | Zhao et al. 2019 | ✅ | GCN + GRU: a base mais simples de grafo + recorrência |
| A3T-GCN (`A3TGCN_UF`) | Bai et al. 2021 | ✅ | T-GCN + atenção temporal sobre os estados ocultos: pesa semanas passadas importantes (início da subida) |
| STGCN (`STGCN_UF`) | Yu et al. 2018 | ✅ | sem recorrência; blocos "sanduíche" (convolução temporal com GLU → grafo → temporal). Treina rápido e é estável com poucas séries |
| MTGNN (`MTGNN_UF`) | Wu et al. 2020 | ✅ | **aprende o grafo** (direcionado, top-4 vizinhos por UF) e soma ao de fronteiras: pode descobrir ligações por mobilidade (ex.: DF–GO–SP) que a fronteira não mostra; mix-hop + convoluções dilatadas (1–16) para a memória longa |
| AGCRN | Bai et al. 2020 | já existe | é o `AGCRN_UF` portado da competição (grafo adaptativo) |
| DCRNN | Li et al. 2018 | ❌ | a difusão bidirecional serve a grafos de trânsito direcionados; o encoder-decoder seq2seq de 68 passos é caro e acumula erro. O MTGNN cobre o grafo direcionado |
| GMAN | Zheng et al. 2020 | ❌ por ora | a "atenção de transformação" para horizontes longos é a ideia mais relevante, mas o modelo pede muitos nós e muitos dados; com 26 UFs × ~15 temporadas, sobreajusta. Candidato se as redes ganharem do ARIMA |
| ASTGCN | Guo et al. 2019 | ❌ | atenção espacial + temporal ≈ A3T-GCN + MTGNN, já cobertos |
| GCRN | Seo et al. 2018 | ❌ | ≈ T-GCN (convolução no grafo dentro da célula recorrente) |
| Transfer GNN (MPNN+TL) | Panagopoulos et al. 2021 | ideia já usada | a transferência entre países ≈ nosso **pooling** regiões + UFs (`NFPooled`, `GBMDireto(pool=True)`, `STID_pool`) |
| EvolveGCN, GC-LSTM, DyGGNN | 2019–2020 | ❌ | preveem **links** em grafos que mudam; nossa tarefa é prever valores nos nós de um grafo fixo |
| 2s-AGCN | Shi et al. 2019 | ❌ | reconhecimento de ações em esqueletos (classificação) |
| Temporal Sets, Path Failure | 2019 | ❌ | previsão de conjuntos e de falhas de caminho: outro tipo de saída |

Também foram adicionadas 3 redes do neuralforecast **sem** exógenas (`TFT/NHITS/TiDE_pool_semcov_UF`), para medir quanto as exógenas futuras ajudam.

---

## 4. Avaliação das variáveis (`UF/avaliar_variaveis.py`)

**Desenho:**
- Nível de temporada: com o que se sabe na SE25, quanto cada grupo melhora a previsão do log do total da temporada em relação à média da própria UF?
- Origem móvel: para prever s, treina só com as temporadas j ≤ s − 2.
- **Afinação 2015–2021**, teste 2022–2024.
- Modelos Ridge e LightGBM.
- Análises "só o grupo", "todas menos o grupo" e Spearman.

Habilidade = 1 − MAE / MAE da referência (> 0 = ajuda). MAE da referência ≈ 1,07 em log.

| Grupo (só ele + referência) | Afinação 2015–21 (Ridge / LGBM) | Teste 2022–24 (Ridge / LGBM) | Decisão |
|---|---|---|---|
| **Recente** (últimas 4–8 sem., tendência, pico parcial) | **+5,0% / +5,0%** | **+5,6% / +12,8%** | ✅ o melhor e o mais estável (já entra em todos os modelos pelas variáveis da origem) |
| Richards (estilo Episcanner) | +4,9% / +3,3% | −0,3% / −5,4% | ⚠️ ajuda na afinação e não se sustenta no teste; fica no GBM `todas` |
| Clima observado (anomalias 12/26 sem.) | +2,7% / +2,1% | −2,7% / −1,6% | ✅ entra (a escolha é feita na afinação), com ressalva |
| Vizinhos pela razão agregada (`vizp_*`, novos) | +2,1% / −3,9% | +5,4% / +6,0% | ✅ substituem os do V4 |
| Vizinhos do V4 (média simples dos logs) | −0,1% / −7,5% | +4,9% / −6,8% | ❌ |
| Espalhamento (% municípios com casos) | +0,3% / −0,9% | +4,2% / −0,2% | ✅ neutro a positivo |
| Estáticas (bioma, Köppen, urbanização) | 0,0% / −2,7% | −0,1% / +5,8% | ✅ (informação de identidade da UF, útil no pooling) |
| Previsão sazonal SEAS5 como variável de origem | −3,9% / −4,4% | −0,2% / −13,1% | ❌ como variável de tamanho; segue como exógena futura de curto prazo |
| Oceano (ENSO, IOD, PDO) | −6,5% / −0,6% | +8,5% / +8,5% | ⚠️ instável; entra nos selecionados porque o ENSO tem a maior correlação (`enso_delta` ρ = 0,389) e já é exógena futura comum |
| Histórico de taxas de ataque (2–4 temporadas, acumuladas) | −1,4% / −9,1% | +0,2% / −12,2% | ❌ (confirma a ablação do V4: −6%) |
| **Todas juntas** | **−16,4%** / +0,6% | +13,1% / +11,2% | ⚠️ sobreajusta na afinação (Ridge); fica como variante `todas` para comparar com `selecionadas` |

Leitura: a escolha usa só a afinação, e o teste aparece aqui apenas como registro. O teste (2022–24) favorece mais variáveis do que a afinação, porque 2023/24 foi uma temporada recorde, em que ENSO, vizinhos e nível recente "avisavam". Por isso as duas variantes (`selecionadas` e `todas`) seguem para o Colab, e a decisão final sai no `CompUF4`, com a afinação de 2018–2021.

Correlações de Spearman mais fortes com o desvio do tamanho: `enso_delta` 0,389 e `inc_ult8` 0,306.

### Conjuntos padronizados (para os modelos serem comparáveis)

| Conjunto | Onde | Conteúdo |
|---|---|---|
| `EXOG_FUTURAS` | AutoARIMAX, NFPooled, Chronos2_cov, INLA GHR_NL (VARX: só ENSO) | ENSO persistido + anomalias de temperatura e chuva (SEAS5 por 6 meses, depois 0) |
| `casos` | GBM, redes | só a própria série (níveis, tendência, taxa de ataque da temporada anterior) |
| `SELECIONADAS` | GBM, DoisEstagios `_sel` | clima, oceano, vizinhos (razão agregada), imunidade física, espalhamento, estáticas. Ficam **fora**: SEAS5 como variável de origem, taxas de ataque 2–4, acumuladas e vizinhos do V4 |
| `todas` | GBM e ablação | tudo, para medir se a seleção ajuda |

Cada família tem uma versão sem covariáveis. A pergunta "as covariáveis ajudam?" é respondida **dentro** de cada família, com a mesma régua (afinação 2018–2021 → teste 2022–2025 no `CompUF4`).

---

## 5. Pendências

- [ ] Rodar `CompUF1–4` no Colab (subir `uf_lib.py`, `uf_autoarima.R` e os notebooks novos).
- [ ] Se o auto.arima livre (d e drift) perder muito para o `_d0`, registrar como achado: extrapolar a tendência por 68 semanas não funciona.
- [ ] Terminar a leitura do ITER (cap. 14–16) e do Lütkepohl (VAR restrito, VECM, BVAR) se o VAR se mostrar competitivo.
- [ ] GMAN como candidato, só se as redes de grafo ficarem perto do ARIMA.
