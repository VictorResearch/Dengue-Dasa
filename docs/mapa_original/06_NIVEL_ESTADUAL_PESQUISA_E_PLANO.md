# 06 · Modelos direto por UF: pesquisa e plano de execução

> **Criado em 30/09/2026.** Complementa o [04_ESTADO_DA_ARTE.md](04_ESTADO_DA_ARTE.md), que não é repetido aqui. A pesquisa nova foi feita no PubMed (2024–2026) e na web (FluSight, benchmarks de 2025–2026, modelos de fundação). O Consensus continuava sem cota.
> **Status (01/10, fim do dia): as 7 fases foram executadas e avaliadas.** Resultados em [09](09_RESULTADOS_UF_E_LICOES.md); avaliação robusta e negócio em [08](08_AVALIACAO_ROBUSTA_E_NEGOCIO.md).
>
> **Status anterior:** pesquisa e plano prontos. **A fase 2 (tabelas por UF) foi feita em 30/09:** `data_imdc_2026/UF/` (script `preparar_tabela_uf.py`, LEIA-ME com o método de cada variável, 38/38 checagens OK). **Em 01/10 foram criados os notebooks** de todas as fases: `CompUF1_portados`, `CompUF2_CPU`, `CompUF3_GPU` e `CompUF4_avaliacao`, com o código em `UF/uf_lib.py`. O LEIA-ME de `data_imdc_2026/UF/` diz o que subir e o que foi testado no PC. **Nenhuma estimativa foi gerada ainda**; elas saem no Colab.
>
> **Atualização de 01/10 (tarde):** a grade do ARIMA e o ARIMAX manual viraram o **auto.arima completo** (Python e R; d livre e d = 0). Foram incluídos 6 **VARs**, as redes de grafo **A3T-GCN, STGCN e MTGNN** e os conjuntos padronizados de variáveis. Detalhes, leituras do Drive e a avaliação das variáveis em [07](07_ECONOMETRIA_GRAFOS_E_VARIAVEIS.md).
> Quando uma frase é **minha inferência** e não vem de uma fonte, ela está marcada com *(inferência)*.

---

## 0. Resumo

1. **Sim, hoje todos os 15 modelos da competição rodam por região de saúde (435) e somam até a UF.** As exceções são os do V4 (`A_uf`, `B_uf`, modelos-base por UF), avaliados no notebook do V4, e as 28 equipes do 3º IMDC, que previram direto por UF.
2. **Rodar direto por UF é possível com pouca mudança de código.** Basta um "contexto UF", em que o nível de modelagem passa a ser o das 26 UFs, que já tem todos os arquivos (`painel_uf`, `seas5_uf`, `adj_uf`, `richards_uf`). A soma até a UF vira identidade. Ver §2.
3. **Há um indício de que o nível UF ajuda os modelos de aprendizado de máquina.** No V4, a AGCRN direto por UF fez 1,121 × BB contra 1,365 por região. A versão B fez 1,238 contra 1,382. Os modelos de topo do 3º IMDC (PURDUE, LNCC, IFGW, PUC-Rio) são todos por UF.
4. **O que a literatura nova acrescenta** (§3):
   - **Modelos globais de gradient boosting quantílico**, treinados juntos em várias localidades e várias fontes, venceram o FluSight 2023/24 (Flusion). O ganho veio de juntar dados, não de complexidade.
   - **Modelos de fundação com covariáveis** (Chronos-2, TabPFN-TS, TimesFM 2.5) passaram a ganhar com covariáveis em 2025. O ganho de modelar várias séries juntas foi pequeno.
   - **Grafos de vizinhança geográfica ajudam pouco** em epidemias (SpatialEpiBench 2026, PNAS 2025 para estados do Brasil).
   - **Aumento de dados com surtos sintéticos** melhorou muito a previsão de temporadas fora do histórico (SP 2024).
5. **Plano** (§5), em 7 fases e 4 notebooks novos:
   - porta os 15 modelos para UF;
   - monta um banco de variáveis por UF com tudo o que existe;
   - testa ~14 abordagens novas: árvores globais, ARIMAX, INLA com DLNM, redes do neuralforecast, STID e GNN pequena, modelos de fundação com covariáveis, híbrido mecanístico;
   - junta os níveis num ensemble.
   - **Toda escolha é feita em 2018–2021 e confirmada em 2022–2025**, para não repetir o problema de escolher olhando o teste.
6. **Meta para "pau a pau":** R vs BB ≤ 0,87 nas 4 temporadas (o 1º do 3º IMDC teve 0,870 e o nosso ARIMA região→UF fez 0,865), com dp log R ≤ 0,35. **Meta esticada:** ≤ 0,83 com o ensemble multinível.

---

## 1. O que roda hoje e em que nível

| Grupo | Nível de modelagem | Como chega na UF | Melhor R vs BB |
|---|---|---|---|
| 15 modelos da competição (Comp1–3) | Região de saúde (435) | Soma de trajetórias (Schaake ou próprias) | ARIMA_Fourier 0,865 |
| V4: `A_uf`, `B_uf`, bases por UF | **UF (26)** | Direto | ENS_bases_A_mediana 0,938; A_uf 1,121; B_uf 1,238 |
| V4: `A_regiao`, `B_regiao` | Região | Soma | A_regiao 1,365; B_regiao 1,382 |
| 28 equipes do 3º IMDC | Quase todas por UF (GHR e EMAP LSTM por região) | Direto | EMAP SARIMAX 0,870 |

**O que isso já sugere** *(inferência)*: para modelos estatísticos com sazonalidade fixa (ARIMA), a região funciona bem, porque a soma de 435 séries cancela ruído. Para modelos que aprendem com variáveis (árvores, redes), a UF tem séries mais suaves e mais sinal por linha, e o V4 mostrou 18–22% de ganho no nível UF. A pergunta "UF direto ou região somada?" precisa ser respondida **por modelo**, e a Fase 1 faz isso.

---

## 2. Viabilidade: adaptar os modelos atuais para UF direto

**Mecanismo único:** um `ContextoUF` em que `ctx.Dr` aponta para os dados por UF (`V.Dados(pasta, 'uf')`) e `ver_reg` = `ver_uf`.
- `D.A_uf` vira a matriz identidade (26 × 26), então `rodar()` grava região = UF sem somar nada.
- O `template_dependencia` passa a usar a dependência histórica **entre UFs**.
- Os resultados vão para `resultados_uf/`, com sufixo `_UF` no nome do modelo (`ARIMA_Fourier_UF`). Assim os dois níveis aparecem lado a lado no ranking.

| Modelo | Funciona por UF? | O que muda |
|---|---|---|
| climatologia, canal_endemico, AR_longo, SURGE | ✅ | Nada (só usam `D`) |
| ARIMA_Fourier, ETS_MSTL, Prophet | ✅ | Nada; ficam 26 séries em vez de 435 (mais rápido) |
| LightGBM_direto, XGBoost_quantis | ✅ | Nada no código. O painel fica 17× menor; dá para aumentar árvores e origens |
| Tamanho_x_forma (V4 B) | ✅ | Já testado no V4 (`B_uf`) |
| AGCRN (V4 A) | ✅ | Já testado no V4 (`A_uf` 1,121) |
| Renovacao_S | ✅ | `sig_reg` e `sig_uf` passam a valer para a mesma unidade; somar as duas variâncias ou zerar `sig_reg` |
| BB_INLA | ✅ | `macro_br = uf_of // 10` funciona igual; o efeito de unidade vira efeito de UF |
| GHR_INLA | ✅ | BYM2 no grafo das 26 UFs (`adj_uf.npz` já existe); `ano_uf` continua |
| DeepAR, NHITS, TFT, TSMixer | ✅ | 26 séries é pouco para redes. **Treinar com regiões + UFs juntas** e prever só as UFs (ver §3.2) |
| Chronos | ✅ | Nada |
| Variáveis de temporada (`features_temporada`) | ✅ | O bloco "contexto da UF" já é pulado quando `nivel == 'uf'` |

**Conclusão:** é possível com um arquivo novo pequeno (`comp_uf.py` com o `ContextoUF`), sem mexer nos modelos. As únicas mudanças reais são o pooling das redes e o ajuste da Renovação.

---

## 3. Pesquisa nova (o que o 04 não cobria)

### 3.1 Árvores globais e quantílicas (LightGBM, XGBoost, CatBoost)

- **Flusion** venceu o FluSight 2023/24 (gripe, 50+ estados dos EUA). É um ensemble de 2 modelos de **gradient boosting quantílico** com conjuntos de variáveis diferentes e um AR bayesiano, **todos treinados juntos com dados de várias localidades e de 3 fontes de vigilância**. Segundo os autores, o desempenho "veio principalmente" do GBQR treinado em conjunto ([Ray et al., arXiv 2407.19054](https://arxiv.org/abs/2407.19054)). No FluSight 2024/25, os 10 melhores ficaram a 0,04 de WIS relativo uns dos outros ([CDC](https://www.cdc.gov/flu-forecasting/evaluation/2024-2025-report.html)).
- **M5 (varejo, 42 mil séries):** quase todos os 50 primeiros usaram LightGBM. O vencedor fez uma média simples de LightGBMs treinados em **3 níveis de agregação**, misturando estratégias direta e recursiva ([Wielding Occam's razor](https://arxiv.org/pdf/2102.13209)).
- **Dengue nas 27 capitais (2026):** o CatBoost direto, com um modelo por horizonte, superou uma GRU multi-saída na maioria das capitais. A GRU teve subajuste, atraso e não antecipou picos ([Int J Biometeorol 2026](https://doi.org/10.1007/s00484-026-03300-7)).
- **No 3º IMDC:**
  - o PURDUE usou LightGBM quantílico com recalibração conformal (CQR) num painel de origens sintéticas;
  - o IFGW usou CatBoost para prever as **características da temporada** (taxa de ataque, pico, semana do pico) a partir da SE25.
- **Lição para nós:** o nosso LightGBM_direto (1,032) e o XGBoost (1,275) são **por região**, sem pooling entre níveis e com alvo de 68 horizontes. A literatura sugere quatro mudanças:
  - **juntar** regiões e UFs no mesmo treino, com o nível como variável;
  - prever **o desvio da âncora sazonal**, com quantis calibrados;
  - **separar tamanho e forma da temporada** (estilo IFGW);
  - fazer **média de vários GBMs** com conjuntos de variáveis diferentes (estilo Flusion e M5).

### 3.2 Redes profundas multi-série

- **TFT, DeepAR, N-HiTS, TSMixer, TiDE e PatchTST** estão todos no `neuralforecast`, com covariáveis históricas, futuras e estáticas.
- **Problema central: 26 UFs × ~15 temporadas é pouco dado para uma rede.**
  - O EMAP LSTM (2º lugar regional no 2º IMDC) contornou isso treinando **um modelo por macrorregião com as regiões de saúde como amostras**, com saída log-normal e **treino pelo WIS**.
  - No PURDUE, a GRU foi o componente mais fraco sozinha (IC95 de 75%), mas ajudou dentro do ensemble.
- **TFT em dengue e malária** (África do Sul e Vietnã) teve R² até 0,90, mas em horizonte curto e sem comparação com modelos-base fortes ([IJERPH 2026](https://doi.org/10.3390/ijerph23010075)).
- **LSTM com SHAP nos 27 estados do Brasil** usou clima defasado e vizinhos, mas é horizonte curto ([BMC Public Health 2025](https://doi.org/10.1186/s12889-025-22106-7)).
- **Rio de Janeiro (2025):**
  - a LSTM com clima foi o melhor modelo de aprendizado de máquina;
  - **para horizonte longo, o melhor foi o Prophet com clima**;
  - o ARIMA foi o melhor só com casos, e o SARIMAX com clima melhorou sobre ele;
  - o ensemble LSTM + ARIMA ganhou dos dois ([Trop Med Health 2025](https://doi.org/10.1186/s41182-025-00723-7)).
- **Lição para nós:**
  - treinar as redes com **regiões + UFs** (461 séries), prevendo as UFs;
  - alvo em log da incidência, com **perda quantílica ou WIS**;
  - **várias sementes**;
  - usar as redes como **membros de ensemble**, não como modelo único.

### 3.3 Modelos espaço-temporais (grafos e bayesianos)

- **SpatialEpiBench (2026):** 11 bases epidêmicas, avaliação contínua padronizada. A maioria dos métodos espaço-temporais (EpiGNN, Cola-GNN etc.) **perdeu para o "último valor"** em 1 dia a 1 mês, mesmo com conhecimento epidemiológico embutido. A **adjacência geográfica trouxe pouco** ([arXiv 2605.06530](https://arxiv.org/abs/2605.06530)).
- **Estados do Brasil:** os efeitos de rede entre estados são fracos ([PNAS 2025](https://doi.org/10.1073/pnas.2422335122)). Na Tailândia, com províncias pequenas, o VAR com vizinhos funcionou.
- **Com 26 nós, o grafo é pequeno e bem definido**, ao contrário das 5.492 cidades do Modelo 2. Os candidatos razoáveis são:
  - **STID** (MLP com embeddings de UF e de semana), que empata com GNNs;
  - uma **GNN leve** (A3T-GCN ou Graph WaveNet) com grafo fixo de fronteira mais um aprendido de baixo posto.
  - A expectativa é que ganhem pouco, mas entram porque o pedido inclui testar *(inferência)*.
- **Bayesianos (INLA):** o GHR (BSC) ficou entre os melhores no 1º e 2º IMDC. O **GHRmodel** atual usa temperatura, chuva e **indicadores de seca** (SPEI), com DLNM ([BSC](https://ess.bsc.es/news/new-models-improve-dengue-forecasting-in-brazil-amid-unprecedented-outbreaks)).
  - **A nossa GHR por região já ficou melhor que a GHR oficial (1,233 × 1,528).**
  - Por UF, vale testar **BYM2 nas 26 UFs + DLNM do clima + ENSO**.
  - A nossa BB_INLA não reproduz o BB oficial (1,555). Precisa ser corrigida para servir de controle.

### 3.4 Modelos de fundação com covariáveis (novidade de 2025)

- **Chronos-2** (Amazon, out/2025): previsão univariada, **multivariada e com covariáveis**, zero-shot, com "atenção de grupo" entre séries.
  - No fev-bench, as covariáveis melhoraram bastante o desempenho (47,0% contra 40,9% de taxa de vitória).
  - A modelagem multivariada ajudou pouco (55,0% contra 54,4%) ([arXiv 2510.15821](https://arxiv.org/abs/2510.15821); [fev-bench](https://arxiv.org/pdf/2509.26468)).
- **TabPFN-TS:** trata a previsão como regressão tabular. Com 11 milhões de parâmetros, ficou **em 1º no GIFT-Eval** e **é o melhor zero-shot com covariáveis dinâmicas** no fev-bench ([arXiv 2501.02945](https://arxiv.org/abs/2501.02945)).
- **TimesFM 2.5** (Google): 200 milhões de parâmetros, contexto de 16 mil pontos, quantis contínuos e **covariáveis via XReg**. Uma versão anterior entrou no stack do NUS no 3º IMDC.
- **Mantis:** treinado **só com simulações mecanísticas** (400 milhões de dias simulados). Superou 39 modelos ajustados por especialistas, inclusive todos do COVID-19 Forecast Hub ([arXiv 2508.12260](https://arxiv.org/abs/2508.12260)).
- **O nosso Chronos (Bolt, univariado) fez 1,431, com explosões no Sul (4,40).** Para os modelos de fundação, o plano é:
  - usar **Chronos-2 com covariáveis**, TabPFN-TS e TimesFM 2.5;
  - prever em **log da incidência**;
  - **limitar a previsão** (teto de 3× o máximo histórico, como no AR_longo).

### 3.5 Híbridos mecanísticos e temporadas extremas

- **PINN para dengue nos estados do Brasil:** rede informada pela física que integra vigilância e modelo mecanístico, com parâmetros de intervenção que variam no tempo, mais XGBoost por estado para os efeitos do clima ([Li et al., JRSI 2026](https://doi.org/10.1098/rsif.2026.0071)). O horizonte é de 12 semanas.
- **Dois estágios (IFGW):** o GBM prevê as características da temporada, e um modelo de renovação calibrado gera a curva. O NUS usou renovação com **depleção de suscetíveis** (a nossa Renovacao_S é dessa família e foi a 1ª no Sul).
- **Aumento de dados com surtos sintéticos** ("virtual data augmentation", SP 2024): reduziu o RMSE em até 80% quando o teste passou do máximo do treino ([Acta Trop 2026](https://doi.org/10.1016/j.actatropica.2026.107992)). É exatamente o problema do surto do Sul (2022/23) e de 2024.

### 3.6 Variáveis: o que a literatura nova diz

- **Brasil e Panamá (One Health, 2026, mensal):** no Brasil, **as variáveis ambientais dominaram** (temperatura, variabilidade da chuva, uso do solo, aptidão do mosquito, demografia). No Panamá, dominou o histórico de transmissão. O ensemble reduziu o MAE em 24% contra o "sazonal ingênuo" ([One Health 2026](https://doi.org/10.1016/j.onehlt.2026.101563)).
- **Isso contrasta com o IMDC** (o clima não melhorou a previsão de temporada) e com o nosso resultado (nenhum modelo com covariáveis venceu o ARIMA). A diferença provável é o horizonte e a granularidade: mensal, com alvo de risco espacial, e não semanal 16–67 semanas à frente *(inferência)*.
- **O que vale para a temporada** (04 + achados novos):
  - nível atual e tendência;
  - tamanho das temporadas anteriores (suscetíveis);
  - ENSO com defasagem fixa;
  - anomalias de clima prolongadas;
  - seca (SPEI, usado no GHRmodel);
  - sorotipo, se disponível.
- **Covariáveis futuras:** só podem entrar se conhecidas na SE25 (previsão SEAS5 emitida até a SE25, ou climatologia).

### 3.7 Combinar níveis

- **Ensemble multi-resolução:** modelos treinados em estado, região e país, combinados com pesos otimizados ([OGEM, Epidemics 2025](https://doi.org/10.1016/j.epidem.2025.100869)).
- **O vencedor do M5 juntou modelos de 3 níveis de agregação.**
- **Para nós:** os membros "UF direto" e "região→UF" erram de formas diferentes, o que dá **diversidade** ao ensemble *(inferência)*. A combinação mais simples e robusta é a **mediana dos quantis** dos k melhores, com k e membros escolhidos em 2018–2021.

---

## 4. Variáveis disponíveis por UF

| Fonte | Variáveis | Já existe por UF? | Observação |
|---|---|---|---|
| `V4/dados/painel_uf` | casos, população, 7 de clima (temp min/méd/máx, chuva, dias de chuva, umidade, amplitude térmica) | ✅ | Ponderado pela população de 2015 |
| `V4/dados/seas5_uf` | previsão sazonal ECMWF (temperatura, chuva, umidade) | ✅ | Usar só emissões até a SE25 |
| `V4/dados/oceano` | ENSO, IOD, PDO | ✅ | Nacional |
| `V4/dados/richards_uf` | parâmetros de Richards (estilo Episcanner), ajuste completo e truncado na SE25 | ✅ | |
| `features_temporada` (v4_lib) | taxa de ataque 1–4 temporadas, incidência acumulada, tendência de 8 semanas, clima prolongado, vizinhos | ✅ | Gerado em tempo de execução |
| Banco completo V8 (`COMPLETO_spatial`, municipal) | pressão, VPD, umidade mín/máx, chuva mín/máx, chikungunya, Afya (2021+), Köppen, bioma | ❌ precisa agregar | Média ponderada pela população; Köppen e bioma como % da população em cada classe |
| FINAL_NN_v2 | mesmas famílias, já escalonadas por município | ❌ | Melhor agregar do COMPLETO (escala original) |
| API Mosqlimate (permitida pelas regras) | Episcanner oficial, dados Infodengue (Rt, receptividade) | ❌ baixar | Conferir no regulamento se cada variável é permitida antes de usar na submissão |

**Seca (SPEI):** não está nos bancos. Dá para calcular uma versão simplificada (anomalia padronizada de chuva acumulada em 3 e 6 meses) com o clima que já temos.

---

## 5. Plano de execução

### Regras que valem para todas as fases

1. **Separação de temporadas:**
   - escolher hiperparâmetros, variáveis e membros do ensemble em **2018–2021 (afinação)**;
   - medir em **2022–2025 (teste)** uma única vez, no fim;
   - o 2020 da pandemia fica na afinação, mas sinalizado.
2. **Mesma régua:** WIS relativo ao BB por UF, R vs climatologia por região, Comp5 (custo de compra), estabilidade (dp log R) e cobertura.
3. **Critério de escolha definido antes:** o modelo final é o de menor R vs climatologia médio em 2018–2021. Em empate (diferença < 0,02), vence o mais estável.
4. **Sem vazamento:** variável futura só se for conhecida na SE25. O teto de previsão é 3× o máximo histórico da UF.
5. **Tudo no Colab**, retomando de onde parou, com erros em `erros_*.csv`, como na competição.

### Fases

| Fase | O que faz | Notebook | Ambiente | Tempo estimado |
|---|---|---|---|---|
| **0 · Infra** | `comp_uf.py` (`ContextoUF`), pasta `resultados_uf/`, sufixo `_UF`, Comp4 e Comp5 lendo os dois níveis, temporadas de afinação 2018–2021 | – (código) | – | – |
| **1 · Portar os 15 modelos** | Roda os 15 modelos atuais direto por UF em 2018–2025. Responde "UF direto ou região→UF?" modelo a modelo | `CompUF1_portados` | CPU (+ GPU para AGCRN e Chronos) | 1–2 h CPU + ~1 h GPU |
| **2 · Banco de variáveis por UF** | Agrega o COMPLETO_spatial por UF (ponderado pela população), cria SPEI simplificado, Köppen e bioma em %, junta SEAS5, ENSO e variáveis de temporada → `painel_uf_completo.parquet` com grupos de variáveis definidos | `CompUF0_dados` | CPU | ~20 min |
| **3 · Árvores globais** | LightGBM, XGBoost e CatBoost **quantílicos diretos**, com **pooling regiões + UFs**, alvo = desvio da âncora sazonal, origens sintéticas. Variantes: (a) só casos, (b) + grupos de variáveis (ablação por grupo), (c) **dois estágios**: GBM prevê tamanho, pico e semana do pico e a Renovação ou o Richards desenha a curva. Média de 3 GBMs com conjuntos de variáveis diferentes (estilo Flusion) | `CompUF2_CPU` | CPU High-RAM | 2–3 h |
| **3 · Estatísticos com covariáveis** | **ARIMAX/SARIMAX por UF** (ENSO defasado + anomalia de clima; família do 1º do 3º IMDC), ARIMA_Fourier_UF com K e anos afinados, **INLA-UF**: BYM2 em 26 UFs + DLNM (temperatura, chuva, SPEI) + ENSO, e a BB_INLA corrigida | `CompUF2_CPU` | CPU | 1 h |
| **4 · Redes e espaço-temporais** | neuralforecast com covariáveis (hist, futuras e estáticas), treinado em regiões + UFs: **TFT, N-HiTS, TiDE, TSMixerx, DeepAR, PatchTST** (perda MQLoss, 3 sementes); **STID** (MLP com embeddings de UF e semana); **GNN leve** (Graph WaveNet ou A3T-GCN no grafo das 26 UFs); **AGCRN_UF** | `CompUF3_GPU` | GPU L4 | 6–10 h (dá para dividir com `RODAR_AGORA`) |
| **5 · Modelos de fundação** | **Chronos-2 com covariáveis**, **TabPFN-TS**, **TimesFM 2.5** com XReg; zero-shot em log da incidência, com teto. Se der certo, um ajuste fino leve do Chronos-2 nas regiões + UFs | `CompUF3_GPU` | GPU L4 | ~1 h |
| **6 · Robustez e calibração** | (a) **aumento de dados** com temporadas sintéticas grandes (curva histórica × ganho 2–5×) nos 2–3 melhores modelos de ML; (b) **calibração conformal** da largura, afinada em 2018–2021 (no V4, o conformal piorou o WIS, então só fica se melhorar na afinação) | `CompUF2_CPU` / `CompUF3_GPU` | CPU / GPU | 1–2 h |
| **7 · Ensemble multinível e veredito** | Mediana dos quantis dos k melhores (k e membros escolhidos em 2018–2021) com **membros dos dois níveis**; comparação com o ARIMA região→UF (0,865) e com o 3º IMDC; Comp5 de negócio; leques, mapa de calor, estabilidade | `CompUF4_avaliacao` | CPU | minutos |

### O que cada fase responde

| Pergunta | Fase |
|---|---|
| Modelar direto na UF é melhor que somar regiões? Para quais modelos? | 1 |
| As variáveis do banco completo ajudam quando o modelo é por UF? Quais grupos? | 3 (ablação) |
| Árvores globais com pooling chegam ao nível do ARIMA? | 3 |
| Redes, GNN e modelos de fundação com covariáveis acrescentam algo? | 4 e 5 |
| Dá para não errar tanto em temporadas fora do histórico (Sul 2022/23)? | 6 |
| Qual é o modelo ou ensemble final? | 7 |

### Ordem e custo

- **Fase 0 → 1 → 2** primeiro: são baratas e já respondem à pergunta principal (UF ou região).
- **Fases 3, 4 e 5 podem rodar ao mesmo tempo**, em sessões separadas: CPU, GPU e GPU.
- **Custo total estimado no Colab:** ~10–15 h de GPU L4 (~50–75 unidades) + ~5 h de CPU. Cabe nas ~190 unidades do Colab Pro, com folga para refazer.

### Riscos

| Risco | Como tratar |
|---|---|
| Sobreajuste ao escolher entre muitas variantes | Escolha só em 2018–2021; teste olhado uma única vez |
| Redes com pouco dado | Pooling regiões + UFs, várias sementes, uso como membro de ensemble |
| Modelos de fundação explodindo (como o Chronos-Bolt) | Log da incidência + teto de 3× o máximo histórico |
| Variáveis não permitidas pelo regulamento | Conferir antes de submeter; se não forem permitidas, manter só os modelos com dados do FTP e da API |
| Afinação em 2018–2021 não representar 2022–2025 (2024 foi recorde) | Reportar também por temporada; preferir modelos estáveis |
| Bibliotecas novas (Chronos-2, TabPFN-TS, TimesFM 2.5) mudando de API | Fixar versões no notebook; falha vira linha em `erros_*.csv` |

---

## 6. Referências novas (além das do 04)

- Ray et al. **Flusion: Integrating multiple data sources for accurate influenza predictions.** [arXiv 2407.19054](https://arxiv.org/abs/2407.19054)
- CDC. **FluSight 2024–2025 Evaluation.** [cdc.gov](https://www.cdc.gov/flu-forecasting/evaluation/2024-2025-report.html)
- **Wielding Occam's razor: Fast and frugal retail forecasting** (M5). [arXiv 2102.13209](https://arxiv.org/pdf/2102.13209)
- Ansari et al. **Chronos-2: From Univariate to Universal Forecasting.** [arXiv 2510.15821](https://arxiv.org/abs/2510.15821)
- **fev-bench: A Realistic Benchmark for Time Series Forecasting.** [arXiv 2509.26468](https://arxiv.org/pdf/2509.26468)
- Hoo et al. **From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting.** [arXiv 2501.02945](https://arxiv.org/abs/2501.02945)
- **TimesFM 2.5** (revisão). [dev.to](https://dev.to/andrew-ooo/timesfm-25-review-googles-time-series-foundation-model-14kp)
- Dudley et al. **Mantis: A Simulation-Grounded Foundation Model for Disease Forecasting.** [arXiv 2508.12260](https://arxiv.org/abs/2508.12260)
- **SpatialEpiBench: Benchmarking Spatial Information and Epidemic Priors in Forecasting.** [arXiv 2605.06530](https://arxiv.org/abs/2605.06530)
- **EpiLearn: A Python Library for Machine Learning in Epidemic Modeling.** [arXiv 2406.06016](https://arxiv.org/abs/2406.06016)
- **Comparative evaluation of machine learning strategies for short-term dengue forecasting in Brazilian capital municipalities.** Int J Biometeorol 2026. [doi](https://doi.org/10.1007/s00484-026-03300-7)
- **Environmental and epidemiological drivers of dengue forecasting across heterogeneous transmission settings in Brazil and Panama.** One Health 2026. [doi](https://doi.org/10.1016/j.onehlt.2026.101563)
- **Machine Learning forecasting of dengue in São Paulo using virtual data augmentation…** Acta Trop 2026. [doi](https://doi.org/10.1016/j.actatropica.2026.107992)
- Li et al. **Integrating machine learning and data-driven modelling: predicting dengue outbreaks… in Brazil.** J R Soc Interface 2026. [doi](https://doi.org/10.1098/rsif.2026.0071)
- **Assessing dengue forecasting methods… Rio de Janeiro.** Trop Med Health 2025. [doi](https://doi.org/10.1186/s41182-025-00723-7)
- **Application of a Temporal Fusion Transformer… Malaria and Dengue.** IJERPH 2026. [doi](https://doi.org/10.3390/ijerph23010075)
- **Forecasting dengue across Brazil with LSTM neural networks and SHAP…** BMC Public Health 2025. [doi](https://doi.org/10.1186/s12889-025-22106-7)
- **Ensemble approaches for short-term dengue fever forecasts: A global evaluation study.** PNAS 2025. [doi](https://doi.org/10.1073/pnas.2422335122)
- BSC. **New models improve dengue forecasting in Brazil amid unprecedented outbreaks.** [bsc.es](https://ess.bsc.es/news/new-models-improve-dengue-forecasting-in-brazil-amid-unprecedented-outbreaks)
