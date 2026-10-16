# 04 · Estado da arte: como melhorar o modelo (literatura + desafio IMDC)

> **Criado em 29/09/2026.** Revisão feita no PubMed, na web e nos repositórios públicos do próprio desafio (1º, 2º e 3º IMDC).
> O Consensus não pôde ser usado: a cota mensal acabou (renova em 01/10).
> Os documentos irmãos são [00_LEIA_PRIMEIRO.md](00_LEIA_PRIMEIRO.md) (estado atual) e [03_AVALIACAO.md](03_AVALIACAO.md) (análise crítica).
> Quando uma frase é **minha inferência** e não vem de um artigo, ela está marcada com *(inferência)*.

---

## 0. Resumo em 10 pontos

1. **O resultado do 3º IMDC já saiu** e é a melhor evidência que existe para o seu problema. Foram 31 equipes, com avaliação por **WIS relativo a um modelo-base** (o BB/PROCC, uma Binomial Negativa bayesiana simples). Os modelos mais estáveis foram **ensembles simples e bem calibrados**, um **AR(92) só com casos** e modelos **por temporada**. Nenhum deles é uma rede de grafos. Ver §2.
2. **O seu modelo resolve uma tarefa diferente da tarefa do desafio.** Ele prevê 1 semana à frente usando os casos observados da semana anterior. O desafio pede de 16 a 67 semanas à frente a partir da SE25. A régua honesta, portanto, **não é a persistência (MAE 185)**. É o modelo-base do desafio e a climatologia. Ver §1.1.
3. **Na previsão de temporada, o histórico de casos domina e o clima ajuda pouco.** O resultado aparece no sprint de 2024 (PNAS 2026), no desafio de Iquitos e Porto Rico (PNAS 2019) e na Tailândia (PNAS 2018). O seu SHAP (71–82% da importância vem da família "histórico de dengue") concorda com isso.
4. **A previsão climática sazonal (ECMWF SEAS5) só alcança 6 meses.** Emitida em junho, ela cobre até dezembro, antes do pico (SE 10–15). O que resta de sinal climático para o pico vem de ENSO e de anomalias persistentes. *(inferência a partir da documentação dos dados)*
5. **A variável que mais falta no seu modelo é a imunidade da população.** Isso inclui sorotipo, tamanho das temporadas anteriores e suscetíveis. A informação de sorotipo aumentou o poder preditivo em Singapura (Nat Commun 2025). A reentrada do DENV-3 no Brasil (2023–25) e a vacinação desde 2024 mudam a dinâmica.
6. **Somar municípios para chegar ao estado exige amostras conjuntas, não quantis somados.** O modelo que mais aparece no topo das edições (GHR/BSC) modela no nível de **região de saúde** e soma **amostras da preditiva posterior**. A literatura de reconciliação probabilística para contagens (bayesRecon, BUIS) oferece a alternativa formal ao MinT.
7. **Calibração conformal foi o maior ganho isolado** do modelo mais estável do 3º IMDC (WIS 1281 → 1216). Ela corrige exatamente o seu problema atual de IC90 = 83% e IC95 = 88%.
8. **Grafos grandes e adaptativos não têm vantagem demonstrada** em séries epidemiológicas ruidosas. Modelos com "identidade espacial" simples empatam com as GNNs ou as superam, e entre os estados brasileiros o efeito de rede é fraco (PNAS 2025).
9. **Ensemble é a recomendação mais consistente de toda a literatura** (FluSight, COVID Hub, IMDC, dengue global). Com poucas temporadas de validação, os pesos devem ser **iguais ou fortemente encolhidos**. Pesos aprendidos em 1 temporada falharam no IMDC 2024.
10. **Plano recomendado (§5):** (a) montar o avaliador oficial (WIS, 4 safras, regra da SE25) com os modelos-base; (b) transformar o AGCRN em previsor direto de temporada; (c) somar por amostras e calibrar com conformal; (d) juntar tudo num ensemble com modelos simples.

---

## 1. O modelo atual visto pela literatura

### 1.1 Tarefa e métrica: o ponto mais importante

| | Seu modelo hoje | O que o desafio avalia |
|---|---|---|
| Horizonte | 1 semana (t−1 → t) | SE41 a SE40, emitido na SE25: **16 a 67 semanas** à frente |
| Informação na entrada | Casos observados até a semana anterior, inclusive dentro do teste | Nada depois da SE25 |
| Métrica principal | MAE, r, cobertura | **WIS** (Weighted Interval Score), relativo ao modelo-base **BB/PROCC**, com média geométrica entre as 4 validações |
| Referência a vencer | Persistência (MAE 185) | BB/PROCC. Na prática, a climatologia e o canal endêmico já são difíceis de vencer |

- Com horizonte de 1 semana e casos observados na entrada, a persistência é quase imbatível. **Isso não diz nada sobre a tarefa real.** Numa previsão de temporada, "repetir a SE25" seria péssimo, porque a SE25 fica na entressafra.
- O **r de Pearson** alto (0,98) vem do horizonte curto: a curva prevista acompanha a observada com 1 semana de atraso. Na previsão de temporada, esse número vai cair, e isso é esperado.
- **Ordem de grandeza para mirar:** no backtest do modelo mais estável do 3º IMDC (26 UFs × 4 safras), o WIS normalizado (Σ WIS / Σ casos) ficou assim:
  - ensemble + conformal: **0,555**;
  - climatologia (quantis históricos por semana): **0,606**;
  - "sazonal ingênuo": 0,666;
  - persistência: 0,760.

  Ou seja, **o melhor modelo ganhou só ~8% da climatologia.** ([repositório PURDUE](https://github.com/kamrul28890/3rd_imdc_purdue_neuralearth))

### 1.2 Arquitetura (AGCRN bimodal, 5.492 nós)

- A AGCRN ([Bai et al., NeurIPS 2020](https://arxiv.org/abs/2007.02842)) foi desenhada para **tráfego**: séries densas, sem zeros, horizonte de minutos a horas e milhares de passos por nó.
- **Dengue municipal é o oposto.** A série é esparsa (muitos zeros), tem ~600 semanas e só ~15 temporadas por município.
- A matriz de adjacência aprendida `softmax(ReLU(E·Eᵀ))` tem 5.492² ≈ 30 milhões de entradas. Os pesos por nó (`weights_pool`) dão a cada município parâmetros próprios. *(inferência)* Com ~15 temporadas por nó, essa estrutura é pouco identificável e tende a decorar o passado.
- **O que a literatura diz sobre grafos:**
  - Revisão de GNNs em epidemiologia ([Liu et al., KDD 2024](https://arxiv.org/abs/2403.19852)): escalabilidade, incerteza e horizonte longo continuam como problemas abertos. Os modelos híbridos (GNN + SIR/SEIR) são a direção mais promissora.
  - O [STID](https://arxiv.org/abs/2208.05233) mostrou que um MLP com "identidade espacial e temporal" (embeddings de nó e de semana) empata com STGNNs complexas ou as supera, com custo muito menor.
  - Em epidemias, [Fujita et al. 2025](https://doi.org/10.1371/journal.pone.0331611) chegam à mesma conclusão: a estrutura de grafo aumenta a complexidade sem ganho proporcional e pode sobreajustar.
  - [Wu et al., PNAS 2025](https://doi.org/10.1073/pnas.2422335122) (dengue em 180+ províncias): o VAR, que usa os vizinhos, funciona na Tailândia, com províncias pequenas e próximas. No Brasil, "os efeitos de rede entre estados são muito mais fracos".
  - Em São Paulo, 27 capitais, [Silva et al. 2026](https://doi.org/10.1007/s00484-026-03300-7) mostraram uma GRU sofrendo de subajuste, atraso temporal e incapacidade de antecipar picos. O CatBoost ganhou na maioria das cidades.
- **O que isso NÃO quer dizer:** que a informação espacial seja inútil. Na sua v2, as variáveis de vizinhança (`w1_`, `w2_`) apareceram no SHAP. E [Quandelacy et al. 2025](https://doi.org/10.1126/scitranslmed.adq4326) mostram forte sincronia de dengue entre locais próximos. O ponto é que a informação espacial entra melhor como **variável ou efeito aleatório** (BYM2, média dos vizinhos) do que como **grafo livre com 30 milhões de arestas**.

### 1.3 Variáveis

| O seu achado (SHAP corrigido) | O que a literatura diz |
|---|---|
| O histórico de casos domina (71–82%) | Igual. Na Tailândia, um modelo parcimonioso com a **incidência pré-temporada** venceu a mediana de 10 anos em 65% das previsões provinciais e reduziu o MAE em 19% ([Lauer et al., PNAS 2018](https://doi.org/10.1073/pnas.1714457115)). No desafio de Porto Rico e Iquitos, modelos com mecanismos biológicos tiveram **menos** habilidade na média ([Johansson et al., PNAS 2019](https://doi.org/10.1073/pnas.1909865116)) |
| O clima tem papel secundário (5–10%) | No sprint de 2024, incluir clima "não melhorou consistentemente" a previsão de temporada ([Araujo et al., PNAS 2026](https://doi.org/10.1073/pnas.2508989123)). O efeito do clima é **não linear, defasado e heterogêneo no espaço**: seca aumenta o risco de 3 a 5 meses depois e chuva extrema de 0 a 3 meses depois, com efeito diferente por nível de urbanização ([Lowe et al., Lancet Planet Health 2021](https://doi.org/10.1016/S2542-5196(20)30292-8)) |
| ENSO/PDO funcionam como "identificador de ano" | Coerente. Com poucas temporadas, índices de baixa frequência viram rótulos de ano. O ENSO tem correlação máxima com ~12 semanas de defasagem ([PUC-Rio, 3º IMDC](https://github.com/Dududidicao99/3rd_imdc_pucrio_arbocaster)) e só deve entrar com defasagem fixa e poucos graus de liberdade |
| `imunidade_ewm` entra no modelo | É a variável certa, mas a literatura vai além: **sorotipo** (Singapura: +54% de habilidade com clima e **+60% com sorotipo** em relação à linha de base sazonal, [Finch et al., Nat Commun 2025](https://doi.org/10.1038/s41467-025-66411-6)); **ter tido surto antes dobra a chance de outro** ([Lee et al., PLoS NTD 2021](https://doi.org/10.1371/journal.pntd.0009773)); **renovação com depleção de suscetíveis** (modelo NUS no 3º IMDC) |
| Afya e chikungunya abaixo do ruído | Nenhum modelo de topo dependia deles |

**Variáveis que faltam e têm respaldo:**
- Tamanho das 1 a 3 temporadas anteriores por UF ou região de saúde (proxy de suscetíveis).
- Taxa de ataque acumulada.
- Parâmetros do Episcanner (R0, pico, tamanho total por município e ano; [Araujo et al., RSOS 2025](https://doi.org/10.1098/rsos.241261)).
- Proporção de DENV-3 e sorotipo dominante ([Ferreira et al., Lancet Reg Health Am 2026](https://doi.org/10.1016/j.lana.2026.101521)).
- Cobertura vacinal (Qdenga desde 2024; [efetividade em SP](https://doi.org/10.1016/j.eclinm.2026.104181); vacina Butantan em [Nat Med 2026](https://doi.org/10.1038/s41591-026-04255-3)).

⚠️ **Regra do desafio:** qualquer dado fora do FTP ou da API Mosqlimate precisa ser enviado aos organizadores. O Episcanner e os dados do Infodengue (Rt etc.) estão na API Mosqlimate.

### 1.4 Conciliação município → estado

**O que a v3 faz hoje:**
- média estadual = soma das médias municipais × fator de viés por UF;
- variância estadual = soma das variâncias × k, com k escolhido na validação para acertar IC50 e IC90;
- o resultado é representado por uma Binomial Negativa.

Isso é um **bottom-up com inflação de variância**: o `k` absorve de forma grosseira a correlação entre municípios. É uma solução honesta, mas limitada, por três motivos.
- Um único `k` por regime não captura correlação diferente entre UFs (SP e AC têm estruturas espaciais muito diferentes).
- Ele não gera **trajetórias**, e trajetórias são necessárias para calcular pico, semana do pico e total da temporada (os organizadores usam cópula LNAR(1) para isso).
- Ele é calibrado em 1 temporada só.

**O que a literatura oferece** (detalhes em §3.4):

| Abordagem | Onde aparece | Ideia |
|---|---|---|
| **Somar amostras conjuntas** da preditiva | GHR/BSC (vencedor de região no 2º IMDC) | O modelo tem efeitos aleatórios espaciais e temporais compartilhados (BYM2, passeio aleatório semanal cíclico, efeito de ano). As amostras já nascem correlacionadas. É só somar amostra por amostra |
| **Cópula gaussiana** sobre os resíduos | Organizadores do 3º IMDC (no tempo, LNAR(1)) | Marginais por município + correlação estimada nos resíduos de validação → amostras correlacionadas |
| **Reconciliação probabilística para contagens** | [bayesRecon](https://cran.r-project.org/package=bayesRecon) (Corani et al. 2024; Zambon et al. 2024) | Previsões-base em vários níveis (município, região, UF), combinadas pela regra de Bayes (BUIS/MCMC) em distribuições coerentes |
| **MinT** | Wickramasuriya et al., JASA 2019 | Ótimo para a média, mas os pesos devem vir de resíduos **fora do teste**. O seu MinT antigo usava o teste. Com previsões-base de um único nível, o MinT não acrescenta nada (vocês já viram isso) |
| **Ensemble multi-resolução** | [OGEM, Epidemics 2025](https://doi.org/10.1016/j.epidem.2025.100869) | Modelos treinados em estado, região e país, combinados com pesos otimizados |

---

## 2. O que os três IMDC ensinaram

### 2.1 1º IMDC (2024): 5 UFs, 6 equipes ([Araujo et al., PNAS 2026](https://doi.org/10.1073/pnas.2508989123))

- Nenhum modelo venceu sempre. Na temporada recorde de 2024, **nenhum** modelo e **nenhum** ensemble acertou.
- Os modelos **bayesianos** (GHR, BB) ficaram entre os melhores no CRPS. Os autores destacam "as vantagens dessa metodologia para gerar previsões probabilísticas em comparação com machine learning e deep learning".
- **Clima não melhorou consistentemente** a previsão de temporada.
- O ensemble com pesos aprendidos em 2023 **errou em 2024 e em 2025**: os modelos bons em anos grandes superestimaram 2025. A lição é **não confiar em pesos aprendidos em 1 ou 2 temporadas**.
- Métricas usadas: CRPS, log score e interval score. As previsões foram aproximadas por uma log-normal. O ensemble usou *log pooling*.
- Os dados de 2024 podem estar contaminados por **Oropouche** notificado como dengue.

### 2.2 2º IMDC (2025): 26 UFs, 19 modelos ([resultados](https://github.com/Mosqlimate-project/2nd_IMDC_sprint_results))

| Região | Top 3 (WIS) |
|---|---|
| Norte | LNCC-AR(p), Beat it (NB bayesiano), Imperial-TFT |
| Nordeste | **GHR** (INLA), LNCC-AR(p), Cornell (NB só com casos) |
| Centro-Oeste | **GHR**, LSTM-RF, Dengue Oracle M2 |
| Sudeste | **Dengue Oracle M2** (LSTM), LNCC-SURGE, Dengue Oracle M1 |
| Sul | Dengue Oracle M1, LSTM-RF, LNCC-AR(p) |

- O ensemble oficial usou os **5 melhores modelos por UF**.

### 2.3 3º IMDC (2026), o seu desafio: 26 UFs, 31 equipes ([resultados](https://github.com/Mosqlimate-project/3rd_IMDC_results))

- **Avaliação:** WIS de cada modelo dividido pelo WIS do **BB/PROCC**, em cada validação (4 safras), resumido pela média geométrica por UF.
- **Modelos "estáveis"** (melhores que o modelo-base e com pouca variação entre UFs e safras): **PURDUE, LNCC_ARP26, EMAP_PROPHET, RKI_PH e IFGW**.
- **Mais UFs acima do modelo-base:** PURDUE (20), LNCC_ARP26 (18), PUCRIO, EMAP_PROPHET e RKI_PH (17 cada), IFGW (16) e CORNELL (15).
- **Mais UFs em 1º lugar:** PUCRIO (6), ISI, LNCC_ARP26 e EMAP_LSTM (3 cada).
- **UFs mais difíceis:** AM e GO, onde poucos modelos venceram o modelo-base.

**Como são os modelos de topo (lidos nos READMEs):**

| Modelo | Escala | Método | Clima? | Incerteza |
|---|---|---|---|---|
| **PURDUE** (Neural Earth) | UF | Ensemble por **mediana de quantis** (Vincentization) de climatologia + **LightGBM quantílico (CQR)** + **GRU com cabeça NB** (5 sementes) + modelo mecanístico de renovação. **Painel de "origens sintéticas"**: cada semana vira origem, com horizontes de 1 a 67 | Sim (SEAS5 filtrado pela data de emissão) | **Recalibração conformal** das caudas (o maior ganho isolado) |
| **LNCC_ARP26** | UF | **AR(92)** na série log2 de casos, simulado com ruído e filtrado por SSA | **Não** | Monte Carlo, 10 mil trajetórias |
| **PUCRIO** (Arbocaster) | UF | ETS, SARIMA(X), MLP, ESN → **stacking com GBM quantílico por quantil** | Sim, mas o clima futuro = média histórica da semana | MC Dropout, conjunto de sementes, GBM quantílico |
| **IFGW** (InfraMIND-Proteus) | UF | **2 camadas:** (1) CatBoost prevê **características da temporada** (taxa de ataque, amplitude do pico, semana do pico) com variáveis da SE25; (2) **modelo mecanístico de renovação** calibrado por Bayes e ajustado a essas características | Sim (SEAS5) | Posterior + ruído NB |
| **RKI_PH** | UF | geo-LSTM + XGBoost + **Chronos** (zero-shot) | – | – |
| **EMAP_PROPHET** | UF | Prophet | – | – |
| **CORNELL** | UF | **Canal endêmico:** NB ajustada, para cada UF × semana, aos ~15 valores históricos | Não | Quantis da NB |
| **BB/PROCC** (modelo-base) | macrorregião | **NB com efeitos aleatórios de semana e de ano por macrorregião** (INLA), sem covariáveis ([Freitas et al. 2025](https://doi.org/10.1016/j.idm.2025.07.014)) | Não | Monte Carlo |
| **GHR/BSC** | **Região de saúde** | INLA, NB, **BYM2** no grafo de regiões, passeio aleatório **cíclico** semanal + efeito anual, clima SEAS5 + Niño3.4 previsto + climatologia depois do horizonte | Sim | **Amostras da posterior somadas até a UF** |
| **NUS-CERM** | UF | Stack: renovação com **depleção de suscetíveis**, **TimesFM-3**, GAM espacial (vizinhos por fronteira), modelo estrutural (Kalman), GLM NB harmônico | Sim | Vincentization com pesos *leave-one-season-out* encolhidos |
| **EMAP_LSTM** (Dengue Oracle) | Região de saúde → UF | LSTM de 2 camadas, **1 modelo por macrorregião** treinado com as regiões de saúde como amostras, saída **log-normal**, **treinado com WIS**, entrada de 89 semanas | ENSO, bioma | MC Dropout |

**Lições diretas para você:**
1. Dos 7 modelos mais robustos, **pelo menos 3 não usam clima** e **nenhum usa grafo aprendido**.
2. Os modelos que **decompõem o problema em "tamanho da temporada" + "forma da curva"** (IFGW, LNCC-SURGE, e o NUS com renovação) vão bem.
3. **O modelo profundo sozinho (GRU NB) foi o pior componente do PURDUE:** cobertura IC95 de 75%, "melhor em temporadas normais, falha no outlier de 2024". Dentro do ensemble ele ajudou.
4. **A cidade é mais difícil que a UF:** no desafio municipal, o modelo "climatológico por quantis" foi o mais forte do PURDUE.

---

## 3. Literatura por tema

### 3.1 Previsão de temporada (horizonte longo)

- **Métodos diretos vs. recursivos.** A previsão recursiva (realimentar a própria previsão) acumula erro e fica instável em horizontes longos. O PUC-Rio relatou colapsos e explosões do ESN e do SARIMAX e precisou "sanear" as previsões. A estratégia **direta multi-horizonte** (um modelo recebe o horizonte h como entrada, ou tem uma saída por h) e o **painel de origens sintéticas** (PURDUE) evitam isso.
- **Alvos de temporada.** O desafio de Porto Rico e Iquitos definiu **pico, semana do pico e total** como alvos. A habilidade no início da temporada foi baixa e foi pior justamente nos anos grandes ([Johansson 2019](https://doi.org/10.1073/pnas.1909865116)). Os organizadores do 3º IMDC calculam essas grandezas com cópula LNAR(1) + modelo de Richards.
- **Incidência pré-temporada** como preditor do total ([Lauer 2018](https://doi.org/10.1073/pnas.1714457115)).
- **Formas da curva:** a curva de Richards (Episcanner) e o "surto típico" alinhado pelo pico e multiplicado por um ganho log-normal (LNCC-SURGE) são modelos de forma muito simples que já competem.
- **Clima previsto:** superensemble bayesiano com previsão sazonal no Vietnã, com habilidade até 3 meses ([Colón-González 2021](https://doi.org/10.1371/journal.pmed.1003542)). Machala, no Equador, com previsão sazonal e ENSO ([Lowe 2017](https://doi.org/10.1016/S2542-5196(17)30064-5)).

### 3.2 Modelos bayesianos espaço-temporais (o padrão da área)

- **Estrutura típica (INLA):** `casos ~ NB(μ)`, com `log μ = log(pop) + efeito cíclico da semana (RW2 cíclico) + efeito do ano (iid) + BYM2 espacial (+ interação espaço-tempo) + DLNM do clima`.
- Referências:
  - [Lowe et al. 2021](https://doi.org/10.1016/S2542-5196(20)30292-8): DLNM + BYM, 558 microrregiões;
  - [Lee et al. 2021](https://doi.org/10.1371/journal.pntd.0009773): GAM espaço-temporal com conectividade e urbanização;
  - pacotes R **GHRexplore / GHRmodel / GHRpredict** ([CRAN](https://www.rdocumentation.org/packages/GHRmodel/versions/0.1.1)).
- **Por que vão bem no desafio:**
  - incerteza coerente por construção;
  - encolhimento hierárquico: regiões pequenas "emprestam força" das grandes;
  - amostras conjuntas que se somam direto até a UF;
  - o efeito de ano captura anos atípicos como incerteza extra em vez de viés.
- **Modelo-base do desafio (BB):** só efeitos de semana e de ano por macrorregião ([Freitas 2025](https://doi.org/10.1016/j.idm.2025.07.014)). Vencê-lo com folga não é trivial.

### 3.3 Deep learning e GNNs espaço-temporais

| Linha | Exemplos | Relevância para você |
|---|---|---|
| **Neural puro** | Cola-GNN, STAN ([JAMIA 2021](https://doi.org/10.1093/jamia/ocaa322)), STGCN para dengue ([Siabi 2026](https://doi.org/10.1038/s41598-026-36225-7)), DengueGNN ([2026](https://doi.org/10.1038/s41598-026-43073-y)), LSTM + SHAP por UF ([Chen 2025](https://doi.org/10.1186/s12889-025-22106-7)), LSTM + mobilidade ([Chen 2025b](https://doi.org/10.1016/j.idm.2025.11.002)), LSTM vs. RF em 790 cidades ([Mussumeci & Coelho 2020](https://doi.org/10.1016/j.sste.2020.100372)) | Quase todos avaliam **horizonte curto** e comparam com modelos-base fracos |
| **Híbrido mecanístico** | MepoGNN (SIR de metapopulação), EINNs ([AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/26690)), CausalGNN, MPSTAN ([Entropy 2024](https://doi.org/10.3390/e26040278)), PINN para dengue no Brasil ([Li et al., JRSI 2026](https://doi.org/10.1098/rsif.2026.0071)) | Melhor para horizonte longo: a dinâmica SIR impõe a forma da curva e a depleção de suscetíveis |
| **Estatística + GNN** | [Fritz et al. 2022](https://doi.org/10.1038/s41598-022-07757-5): GNN dentro de modelo espaço-temporal estatístico | Mantém a interpretação e a incerteza |
| **Contagens esparsas** | **STZINB-GNN** ([Zhuang et al. 2022](https://arxiv.org/abs/2208.05908)): GNN com saída ZINB | Diretamente aplicável ao nível municipal, com excesso de zeros |
| **Baselines simples** | STID ([arXiv](https://arxiv.org/abs/2208.05233)), Fujita 2025 | Mostram que embeddings de nó + MLP bastam na maioria dos casos |
| **Modelos de fundação** | Chronos (RKI, 2º IMDC), **TimesFM-3** (NUS), **Mantis** (treinado só em simulações mecanísticas; ficou entre os 2 melhores em 16 doenças, [PNAS 2026](https://doi.org/10.1073/pnas.2602542123)) | Membros baratos de ensemble, zero-shot |
| **Revisões** | [Liu et al. 2024](https://arxiv.org/abs/2403.19852), [Rodríguez et al. 2022](https://arxiv.org/abs/2207.09370), [Chen et al. 2026](https://doi.org/10.3389/fpubh.2026.1830634) | Problemas em aberto: horizonte longo, UQ, escala, integração mecanística |

### 3.4 Agregação e reconciliação (município → região → estado)

- **Bottom-up** (o que você faz) é ótimo quando o nível de baixo é bem previsto e a correlação é modelada. Com muito ruído e zeros no município, costuma perder para modelar num nível intermediário. É por isso que o GHR e o Dengue Oracle usam **região de saúde** (~450 unidades), e não município.
- **Top-down / middle-out:** prever região ou UF diretamente e usar o município só como variável ou para distribuir o total.
- **Reconciliação ótima (MinT)** (Wickramasuriya, Athanasopoulos & Hyndman, JASA 2019): combina previsões-base de **todos os níveis** usando a covariância dos erros **estimada fora do teste**.
- **Reconciliação probabilística:**
  - por projeção: Panagiotelis et al., EJOR 2023;
  - gaussiana: [Wickramasuriya, JBES 2024](https://ideas.repec.org/a/taf/jnlbes/v42y2024i1p272-285.html);
  - **para contagens**, via condicionamento bayesiano: [Corani et al. 2024](https://arxiv.org/abs/2207.09322); BUIS de Zambon et al. 2024 (pacote [bayesRecon](https://cran.r-project.org/package=bayesRecon)).
- **Aplicações em doenças:** dengue no Sri Lanka em 3 níveis ([arXiv 2112.00992](https://arxiv.org/abs/2112.00992)), com o melhor método variando por nível; ensemble multi-resolução para hospitalizações ([OGEM 2025](https://doi.org/10.1016/j.epidem.2025.100869)).
- **Correlação espacial:** a variância da soma depende das covariâncias. A sincronia forte entre locais próximos ([Quandelacy 2025](https://doi.org/10.1126/scitranslmed.adq4326)) explica por que somar variâncias independentes dá intervalos estaduais estreitos demais. O seu `k` > 1 é o reflexo disso.

### 3.5 Incerteza, calibração e avaliação

- **WIS** ([Bracher et al. 2021](https://doi.org/10.1371/journal.pcbi.1008618)): aproxima o CRPS a partir de quantis e é a métrica oficial. **Atenção:** em escala de contagem, o WIS é dominado pelas UFs e semanas grandes (SP, MG, PR no pico).
- **WIS em escala log** ([Bosse et al. 2023](https://doi.org/10.1371/journal.pcbi.1011393)): funciona como erro relativo, mede se o modelo acertou a **taxa de crescimento** e muda o ranking dos modelos. Use as duas: a oficial para decidir e a log para diagnosticar.
- **Conformal:**
  - CQR (Romano et al. 2019): alargamento multiplicativo dos quantis ajustado em temporadas de validação (usado pelo PURDUE);
  - **Conformal PID** ([Angelopoulos, Candès & Tibshirani, NeurIPS 2023](https://arxiv.org/abs/2307.16895)): corrigiu a cobertura do ensemble oficial de COVID do CDC de 20% para 70%, com nominal de 80%, numa onda.
- **Ensembles:**
  - média de quantis (Vincentization) vs. mistura linear vs. *log pooling* ([Araujo 2026](https://doi.org/10.1073/pnas.2508989123));
  - pesos treinados vs. não treinados: o ganho dos treinados é pequeno e instável ([Ray 2022](https://doi.org/10.1016/j.ijforecast.2022.06.005), [Frutos 2026](https://doi.org/10.1016/j.idm.2026.04.002));
  - **mais de 3 modelos** garantem robustez, com retorno decrescente ([Fox 2024](https://doi.org/10.3201/eid3009.240026));
  - importância de cada modelo dentro do ensemble ([Kim 2026](https://doi.org/10.1016/j.ijforecast.2025.12.006));
  - pacote R hubEnsembles ([Shandross 2026](https://doi.org/10.1002/sim.70333)).
- **O que um modelo-base simples faz dentro do ensemble:** baixa variância. Um ensemble de modelos fracos + modelo-base supera a persistência de forma consistente ([Wu 2025](https://doi.org/10.1073/pnas.2422335122)).

### 3.6 Fatores de transmissão e qualidade de dados

- **Clima:**
  - efeito defasado e não linear, com seca e chuva extrema em defasagens diferentes ([Lowe 2021](https://doi.org/10.1016/S2542-5196(20)30292-8));
  - anomalias térmicas prolongadas explicam a expansão para o Centro-Oeste e o Sul ([Barcellos 2024](https://doi.org/10.1038/s41598-024-56044-y));
  - extremos compostos e em cascata ([Fletcher 2025](https://doi.org/10.1016/j.lanplh.2025.06.003)).
- **Imunidade e sorotipo:** ver §1.3.
- **Conectividade:** a relação com a rede urbana não é linear, com risco menor nas metrópoles do que nas capitais regionais ([Lee 2021](https://doi.org/10.1371/journal.pntd.0009773)).
- **Dados:**
  - atraso de notificação e nowcasting ([Koemen 2025](https://doi.org/10.1016/j.epidem.2025.100880), [Xiao 2025](https://doi.org/10.1371/journal.pntd.0012501)). No desafio, os dados são consolidados, mas a SE25 de cada safra pode estar incompleta;
  - contaminação por Oropouche em 2024;
  - ES com problemas de notificação.

---

## 4. Diagnóstico resumido: o que manter, o que mudar

| Componente | Manter | Mudar |
|---|---|---|
| Binomial Negativa com offset log(pop) | ✅ É o padrão da área | Considerar ZINB/hurdle no município ou modelar em região de saúde |
| Bancos V8 corrigidos | ✅ | Acrescentar variáveis de temporada (tamanho das anteriores, Episcanner) |
| AGCRN com 5.492 nós e grafo adaptativo | – | Trocar por um previsor **direto de temporada**. O grafo pode virar variável de vizinhança fixa (W1/W2, já existe) ou BYM2 |
| Especialistas SECA/PICO | 🟡 | Na previsão de temporada, a divisão perde sentido: o horizonte já cobre as duas fases. Um termo sazonal cíclico resolve |
| Losses equilibradas (v3) | ✅ Correto usar regra de pontuação própria | Treinar direto no alvo oficial (WIS ou quantis) ou em NB NLL por horizonte |
| Pós-processamento calibrado na validação (v3) | ✅ A ideia está certa (nada do teste) | Calibrar em **várias** temporadas (*leave-one-season-out*) e usar conformal em vez de um `k` único |
| Soma estadual NB com variância × k | 🟡 | Somar **amostras correlacionadas** (efeito compartilhado ou cópula) e comparar com reconciliação BUIS |
| Métricas MAE, r, IC | 🟡 | **WIS e WIS relativo ao BB** (oficiais), WIS em log, cobertura 50/80/90/95 e erro no total e no pico da temporada |

---

## 5. Plano de ação recomendado (em ordem)

### Etapa 1 · A régua certa (antes de qualquer modelo novo)

1. **Avaliador oficial:** uma função que, para cada safra S1–S4, usa só dados até a SE25, prevê SE41→SE40 e calcula o WIS por UF e semana, o **WIS relativo ao modelo-base** (média geométrica entre as 4 safras) e a cobertura. Use as colunas `train_k`/`target_k` do `dengue.csv` para definir as janelas, como o PURDUE, e não datas digitadas à mão.
2. **Três modelos-base** (cada um tem poucas linhas):
   - **Canal endêmico NB** (Cornell): para cada UF × semana epidemiológica, ajustar uma NB aos valores das temporadas anteriores.
   - **Climatologia por quantis:** quantis empíricos da incidência histórica na mesma semana.
   - **BB/PROCC:** NB com efeitos de semana e de ano por macrorregião (R-INLA, [repositório](https://github.com/lsbastos/3rd_imdc_procc_bb_model)). As previsões dele já estão no repositório de resultados e podem ser usadas direto como denominador.
3. **Critério de sucesso:** média geométrica do WIS relativo **< 1** e desvio pequeno entre UFs e safras, que é o quadrante "estável".

### Etapa 2 · Transformar o seu modelo em previsor de temporada

Há duas rotas. A **A** é a mais próxima do que você já tem; a **B** é a que a literatura favorece.

**Rota A: AGCRN direto multi-horizonte**
- Entrada: a sequência até a SE25 (por exemplo, as 52 a 104 semanas anteriores).
- Saída: 67 horizontes de uma vez (cabeça "seq2seq" ou uma saída por horizonte). Treino com **origens sintéticas**: cada semana do treino é uma origem, e o alvo são as semanas h = 16…67 à frente.
- Variáveis futuras: só as conhecidas na SE25 (semana do ano, SEAS5 até 6 meses, climatologia depois disso).
- Reduzir os nós para **região de saúde (~450)**. O grafo adaptativo fica viável (450² ≈ 200 mil) e os zeros diminuem. Depois, distribuir para município se quiser o desafio municipal opcional.
- Saída NB (ou quantis) e loss NB NLL + WIS por horizonte.

**Rota B: decompor "tamanho" × "forma"** (IFGW, LNCC-SURGE, Lauer 2018)
1. Prever, por UF ou região de saúde, a **taxa de ataque da temporada** e a **semana do pico** com variáveis da SE25: incidência de SE1–25, tamanho das 1 a 3 temporadas anteriores, `imunidade_ewm`, ENSO com defasagem, anomalia térmica. Use LightGBM quantílico ou INLA.
2. Gerar a **forma semanal**: curva típica da UF alinhada pelo pico (Richards ou média alinhada), ou uma **renovação com depleção de suscetíveis** calibrada.
3. Multiplicar e amostrar → trajetórias → quantis.

### Etapa 3 · Agregação e calibração

1. Gerar **S = 1.000 trajetórias** por unidade com **correlação** entre elas, de uma destas formas:
   - efeito aleatório compartilhado por UF ou região no próprio modelo;
   - cópula gaussiana com correlação dos resíduos de log-incidência nas temporadas de validação;
   - no INLA, amostras da posterior.
2. Somar amostra por amostra → quantis da UF. Assim saem de brinde o total da temporada, o pico e a semana do pico.
3. Comparar com: (a) o seu pós v3 atual; (b) um modelo direto na UF; (c) reconciliação **BUIS** (bayesRecon) entre município ou região e UF.
4. **Calibração conformal:** fatores multiplicativos por quantil, ajustados *leave-one-season-out* (S1–S3 para calibrar a S4, e assim por diante). Isso ataca direto o IC90 de 83% e o IC95 de 88%.

### Etapa 4 · Ensemble

- Membros sugeridos (de diversidade alta e custo baixo):
  - o seu modelo novo;
  - canal endêmico NB;
  - BB ou um INLA estilo GHR;
  - LightGBM quantílico com origens sintéticas;
  - AR em log por UF (estilo LNCC);
  - um modelo de fundação zero-shot (Chronos ou TimesFM).
- Combinar pela **mediana dos quantis** (Vincentization), depois aplicar recalibração conformal.
- Pesos: iguais, ou aprendidos *leave-one-season-out* com **encolhimento** para pesos iguais. Nunca pesos aprendidos em uma única temporada.

### Etapa 5 · Variáveis novas (só depois que as etapas 1 a 3 estiverem funcionando)

| Variável | Racional | Fonte |
|---|---|---|
| Taxa de ataque das 1 a 3 temporadas anteriores (UF e região) | Proxy de suscetíveis ([Lee 2021](https://doi.org/10.1371/journal.pntd.0009773), NUS) | `dengue.csv` |
| Incidência acumulada SE1–25 e tendência nas últimas 8 semanas | Incidência pré-temporada ([Lauer 2018](https://doi.org/10.1073/pnas.1714457115)) | `dengue.csv` |
| Parâmetros do Episcanner (R0, pico, total) da temporada anterior | Resumo da dinâmica ([Araujo 2025](https://doi.org/10.1098/rsos.241261)) | API Mosqlimate |
| ENSO com defasagem de ~12 semanas + SEAS5 de jun a dez | Clima previsto sem vazamento | FTP do desafio |
| Anomalia de temperatura prolongada | Expansão geográfica ([Barcellos 2024](https://doi.org/10.1038/s41598-024-56044-y)) | `climate.csv` |
| Proporção de sorotipos (DENV-3) | Imunidade específica ([Finch 2025](https://doi.org/10.1038/s41467-025-66411-6)) | ⚠️ Fora do FTP: precisa ser enviada aos organizadores |

---

## 6. Armadilhas a evitar

- **Vazamento:**
  - filtrar o clima previsto pela **data de emissão** (`reference_month` ≤ junho), nunca pela data-alvo;
  - não usar nada entre a SE26 e a SE40 da própria safra;
  - o arquivo `ctx` da v2 (SE26–40) **não** pode entrar na previsão de temporada.
- **Semana 53:** 2025 tem SE53. Use as datas do `dengue.csv`, não `%G%V`.
- **Escala do WIS:** SP, MG e PR dominam o WIS absoluto. Se a média geométrica por UF é o que conta no ranking, **cada UF pesa igual**, e modelos que acertam AC, AP e RR contam tanto quanto os que acertam SP.
- **Temporada outlier (2023-24):** ela está na S2 e no treino da S3 e S4. Modelos treinados com ela tendem a superestimar as temporadas seguintes, como aconteceu no IMDC24 em 2025.
- **Pesos de ensemble:** com 4 safras, qualquer otimização de pesos sobreajusta.
- **População:** para 2026-27 é preciso extrapolar a população (o PURDUE fez isso).

---

## 7. Referências principais

Do desafio (primeira leitura recomendada):
- Araujo EC et al. *Leveraging probabilistic forecasts for dengue preparedness and control: The 2024 Dengue Forecasting Sprint in Brazil.* PNAS 2026. [doi:10.1073/pnas.2508989123](https://doi.org/10.1073/pnas.2508989123)
- Resultados do [3º IMDC](https://github.com/Mosqlimate-project/3rd_IMDC_results) e do [2º IMDC](https://github.com/Mosqlimate-project/2nd_IMDC_sprint_results) · repositórios de [PURDUE](https://github.com/kamrul28890/3rd_imdc_purdue_neuralearth), [PUC-Rio](https://github.com/Dududidicao99/3rd_imdc_pucrio_arbocaster), [LNCC ARP26](https://github.com/pesquefGH/3rd_imdc_lncc_lncc_arp26_dengue), [IFGW](https://github.com/InfraMIND-models/3rd_imdc_ifgw_inframind-proteus), [GHR/BSC](https://github.com/dievillano/3rd_imdc_bsc_ghr), [NUS-CERM](https://github.com/SungmokJung/3rd_imdc_nus_nus-cerm), [Cornell](https://github.com/BentoLab-DiseaseDynamics/3rd_imdc_cornell_bentolab), [BB/PROCC](https://github.com/lsbastos/3rd_imdc_procc_bb_model), [EMAP LSTM](https://github.com/eduardocorrearaujo/3rd_imdc_emap_lstm)
- Freitas LP et al. *A statistical model for forecasting probabilistic epidemic bands for dengue cases in Brazil.* Infect Dis Model 2025. [doi:10.1016/j.idm.2025.07.014](https://doi.org/10.1016/j.idm.2025.07.014)
- Araujo EC et al. *Large-scale epidemiological modelling: scanning for mosquito-borne diseases spatio-temporal patterns in Brazil.* R Soc Open Sci 2025. [doi:10.1098/rsos.241261](https://doi.org/10.1098/rsos.241261)

Previsão de dengue e desafios:
- Johansson MA et al. PNAS 2019. [doi:10.1073/pnas.1909865116](https://doi.org/10.1073/pnas.1909865116)
- Lauer SA et al. PNAS 2018. [doi:10.1073/pnas.1714457115](https://doi.org/10.1073/pnas.1714457115)
- Wu D et al. PNAS 2025. [doi:10.1073/pnas.2422335122](https://doi.org/10.1073/pnas.2422335122)
- Colón-González FJ et al. PLoS Med 2021. [doi:10.1371/journal.pmed.1003542](https://doi.org/10.1371/journal.pmed.1003542)
- Lowe R et al. Lancet Planet Health 2017. [doi:10.1016/S2542-5196(17)30064-5](https://doi.org/10.1016/S2542-5196(17)30064-5)
- Mussumeci E, Coelho FC. Spat Spatiotemporal Epidemiol 2020. [doi:10.1016/j.sste.2020.100372](https://doi.org/10.1016/j.sste.2020.100372)
- Chen X et al. BMC Public Health 2025. [doi:10.1186/s12889-025-22106-7](https://doi.org/10.1186/s12889-025-22106-7) · Infect Dis Model 2025. [doi:10.1016/j.idm.2025.11.002](https://doi.org/10.1016/j.idm.2025.11.002)
- Silva et al. Int J Biometeorol 2026. [doi:10.1007/s00484-026-03300-7](https://doi.org/10.1007/s00484-026-03300-7)
- Charles LE et al. One Health 2026. [doi:10.1016/j.onehlt.2026.101563](https://doi.org/10.1016/j.onehlt.2026.101563)
- Li et al. J R Soc Interface 2026. [doi:10.1098/rsif.2026.0071](https://doi.org/10.1098/rsif.2026.0071)

Fatores de transmissão:
- Lowe R et al. Lancet Planet Health 2021. [doi:10.1016/S2542-5196(20)30292-8](https://doi.org/10.1016/S2542-5196(20)30292-8)
- Lee SA et al. PLoS NTD 2021. [doi:10.1371/journal.pntd.0009773](https://doi.org/10.1371/journal.pntd.0009773)
- Finch E et al. Nat Commun 2025. [doi:10.1038/s41467-025-66411-6](https://doi.org/10.1038/s41467-025-66411-6)
- Quandelacy TM et al. Sci Transl Med 2025. [doi:10.1126/scitranslmed.adq4326](https://doi.org/10.1126/scitranslmed.adq4326)
- Barcellos C et al. Sci Rep 2024. [doi:10.1038/s41598-024-56044-y](https://doi.org/10.1038/s41598-024-56044-y)
- Fletcher C et al. Lancet Planet Health 2025. [doi:10.1016/j.lanplh.2025.06.003](https://doi.org/10.1016/j.lanplh.2025.06.003)
- Ferreira et al. Lancet Reg Health Am 2026 (DENV-3). [doi:10.1016/j.lana.2026.101521](https://doi.org/10.1016/j.lana.2026.101521)

GNNs e deep learning:
- Liu Z et al. *A Review of Graph Neural Networks in Epidemic Modeling.* KDD 2024. [arXiv:2403.19852](https://arxiv.org/abs/2403.19852)
- Rodríguez A et al. *Data-Centric Epidemic Forecasting: A Survey.* [arXiv:2207.09370](https://arxiv.org/abs/2207.09370) · EINNs, AAAI 2023
- Bai L et al. AGCRN, NeurIPS 2020. [arXiv:2007.02842](https://arxiv.org/abs/2007.02842) · Shao Z et al. STID, CIKM 2022. [arXiv:2208.05233](https://arxiv.org/abs/2208.05233)
- Fujita et al. PLoS One 2025. [doi:10.1371/journal.pone.0331611](https://doi.org/10.1371/journal.pone.0331611) · Fritz C et al. Sci Rep 2022. [doi:10.1038/s41598-022-07757-5](https://doi.org/10.1038/s41598-022-07757-5)
- Zhuang D et al. STZINB-GNN. [arXiv:2208.05908](https://arxiv.org/abs/2208.05908)
- Dudley et al. *Mantis.* PNAS 2026. [doi:10.1073/pnas.2602542123](https://doi.org/10.1073/pnas.2602542123)

Reconciliação, avaliação e ensembles:
- Wickramasuriya SL, Athanasopoulos G, Hyndman RJ. MinT. JASA 2019 · Panagiotelis A et al. EJOR 2023 · Wickramasuriya SL. JBES 2024
- Corani G et al. *Probabilistic reconciliation of count time series.* [arXiv:2207.09322](https://arxiv.org/abs/2207.09322) · [bayesRecon](https://cran.r-project.org/package=bayesRecon)
- Bracher J et al. PLoS Comput Biol 2021. [doi:10.1371/journal.pcbi.1008618](https://doi.org/10.1371/journal.pcbi.1008618)
- Bosse NI et al. PLoS Comput Biol 2023. [doi:10.1371/journal.pcbi.1011393](https://doi.org/10.1371/journal.pcbi.1011393)
- Romano Y et al. CQR, NeurIPS 2019. [arXiv:1905.03222](https://arxiv.org/abs/1905.03222) · Angelopoulos A et al. Conformal PID, NeurIPS 2023. [arXiv:2307.16895](https://arxiv.org/abs/2307.16895)
- Ray EL et al. Int J Forecast 2022. [doi:10.1016/j.ijforecast.2022.06.005](https://doi.org/10.1016/j.ijforecast.2022.06.005) · Fox SJ et al. Emerg Infect Dis 2024. [doi:10.3201/eid3009.240026](https://doi.org/10.3201/eid3009.240026) · Shandross L et al. Stat Med 2026. [doi:10.1002/sim.70333](https://doi.org/10.1002/sim.70333)
- Xu S et al. OGEM. Epidemics 2025. [doi:10.1016/j.epidem.2025.100869](https://doi.org/10.1016/j.epidem.2025.100869)
