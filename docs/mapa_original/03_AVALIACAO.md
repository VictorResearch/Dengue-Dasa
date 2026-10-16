# 03 · Avaliação crítica do progresso

> Baseado no docx de avaliação, nas saídas do notebook, no `modelos/resultados_agcrn.db` (recalculei as métricas), na inspeção dos pesos `.pt` e de todos os `.parquet`, e em **2 baselines que calculei agora** para a Season 1.
> Tudo se refere à **Season 1 (2022-23)**, porque foi a única avaliada.
>
> **Atualizado em 29/09/2026.** A seção 0 abaixo diz o que mudou desde a avaliação original. As seções 1 a 6 são a avaliação de 28/09 e ficam como registro. Os números atuais estão em [00_LEIA_PRIMEIRO.md §4](00_LEIA_PRIMEIRO.md#4-métricas-atuais-season-1-2022-23).

---

## 0. Situação em 29/09/2026

### Problemas da seção 4: o que já foi resolvido

| Problema | Situação | Como |
|---|---|---|
| 4.1 Avaliação 1 semana à frente, não a safra inteira | ❌ **Aberto** | Continua igual em todas as versões. É o maior passo que falta para uma submissão válida |
| 4.2 Datas deslocadas em +7 dias | ✅ Resolvido na v3 | O pós-processamento da v3 usa a data real do banco. O notebook v2 ainda usa `%G%V`, que também quebraria na S4: 2025 tem uma semana 53 que o ISO não tem |
| 4.3 Desalinhamento de 1 semana | ✅ **Confirmado e resolvido** (v2 em diante) | O arquivo `ctx` aquece a rede e a semana t−1 passa a prever a semana t |
| 4.4 Duplicatas e colunas vazias do R | ✅ Resolvido | Bancos V8 refeitos dos brutos, 79 checagens por safra |
| 4.5 `ffill/bfill` do futuro para o passado | ✅ Resolvido | Não existe mais nos V8 |
| 4.6 Lags climáticos diferentes entre safras | ✅ Resolvido | Sempre lags 2 a 5 |
| 4.7 A SECA nunca era validada | ✅ Resolvido (v2) | Validação = últimas 52 semanas do treino (21 de SECA e 31 de PICO), com early stopping |
| 4.8 Resultados acumulados no SQLite | ✅ Resolvido | Os resultados da safra são substituídos. Cada versão grava na sua pasta (`resultados_modelo2_v2`, `_v3`) |
| 4.9 Só a S1 avaliada; pesos não salvos | 🟡 Parcial | Bancos S2–S4 prontos e pesos salvos no Drive, mas S2–S4 ainda não foram treinadas |
| **Novo:** MinT calculava os pesos com os casos reais do teste | ✅ Resolvido (v3) | Trocado por uma calibração feita só na temporada de validação |
| **Novo:** losses assimétricas criavam viés | 🟡 Corrigido no código (v3) | Resultado do treino completo pendente |
| **Novo:** estado = soma das medianas; intervalos = soma dos quantis | ✅ Resolvido (v3) | Soma estadual por Binomial Negativa, com variância calibrada na validação |

### O que mudou nas conclusões das seções 2 e 3

| Conclusão de 28/09 | Agora |
|---|---|
| "Loss assimétrica ajuda" | ❌ **Derrubada.** O diagnóstico mostrou o PICO superestimando 3–5× no começo da temporada e a SECA subestimando ~0,65×. A v3 usa a mesma loss equilibrada nos dois especialistas |
| "MinT dinâmico: não testado" | ❌ **Descartado.** A entrada já era coerente, então o MinT não mudava nada de útil, e ele usava os casos reais do teste. Sem ele, o erro honesto da v2 era MAE 829. Com a calibração na validação, caiu para 243 |
| "Separar regimes SECA × PICO" | ✅ Mantido, mas agora cada especialista também treina no outro regime (peso 0,25) para evitar o salto na troca entre eles |
| "Afya está entre as mais importantes" | ❌ **Derrubada.** Era efeito do valor constante antes de 2021. Com a soma semanal correta, a Afya fica abaixo do nível de ruído no SHAP |
| "O clima tem papel secundário" | ✅ Confirmado com o clima corrigido: 5–10% da importância. Poucas variáveis climáticas passam no critério |
| "O que sustenta o modelo é a própria curva de casos" | ✅ Confirmado e reforçado: a **incidência observada da semana**, que não estava no modelo, é a variável mais importante |
| "Nenhum modelo supera a persistência" | 🟡 Continua valendo (melhor = 243 × 185 da persistência), mas a distância caiu de 4× para 1,3× |

### Listas de variáveis atuais (SHAP corrigido, 4 safras)

- **SECA (20):** `velocidade`, `aceleracao`, `log_inc_dengue_t1`, `log_inc_dengue_obs`, `log_inc_dengue_obs_lag1`, `imunidade_ewm`, `log_pop_t1`, `log_inc_dengue_obs_lag2`, `w1_log_inc_dengue_obs`, `w2_log_inc_dengue_obs`, `w1_log_inc_dengue_t1`, `w2_temp_forecast_t1`, `w2_log_inc_dengue_t1`, `cosseno_semana`, `w1_log_pop_t1`, `precip_tot_forecast`, `rel_humid_max`, `temp_min_lag2`, `w1_pressure_min_t1`, `temp_min_lag4`
- **PICO (20):** `velocidade`, `aceleracao`, `log_inc_dengue_obs`, `log_inc_dengue_t1`, `log_inc_dengue_obs_lag1`, `w1_log_inc_dengue_obs`, `w2_log_inc_dengue_obs`, `log_inc_dengue_obs_lag2`, `cosseno_semana`, `log_pop_t1`, `imunidade_ewm`, `temp_med`, `w1_log_pop_t1`, `seno_semana`, `w1_velocidade_t1`, `w2_velocidade_t1`, `w2_log_inc_dengue_t1`, `precip_max_lag3`, `biome_encoded`, `koppen_encoded`

Método e ranking completo: `data_imdc_2026/SHAP_corrigido/relatorio_shap_corrigido.md`.

### Bancos que podem ser apagados agora

Os bancos V8 (`S1_corrigido/` a `S4_corrigido/`) substituem V1–V7. Sobram ~15 GB livres no F:. Dá para apagar tudo o que a seção 5 marca com 🗑. Se não for mais preciso reproduzir os modelos de junho, V5b, V7 e V3 também podem sair. Os V8 são regeráveis a partir de `variaveis/` em ~5 min por safra. **Não apague `variaveis/`.**

---

## 1. Quadro de resultados (Season 1) · avaliação original de 28/09

### Nível estadual (o desafio obrigatório)

| # | Modelo | Banco | MAE | RMSE | sMAPE | Pearson r | IC50 | IC90 | IC95 |
|---|---|---|---|---|---|---|---|---|---|
| M1 | AGCRN 12 feats + MinT simples | V3 | 830 | 2.607 | 68,2% | 0,668 | – | – | – |
| M1′ | idem (recalculado do `.db`) | V3 | 831 | 2.741 | 64,9% | 0,661 | 66,6% | 95,0% | 97,6% |
| **M2** ★ | **AGCRN bimodal seca/pico + MinT dinâmico** | **V7** | **476** | **1.349** | **63,7%** | **0,927** | 59,5% | 95,9% | 98,2% |
| M3 | Pico com MHA, sem validação | V7 | 979 | 3.433 | 95,3% | 0,393 | 33,7% | 74,5% | 82,2% |
| ? | Célula 49 do notebook (execução não identificada\*) | V7 | 1.001 | 3.407 | 98,0% | 0,925 | 66,1% | 99,4% | 99,8% |
| **B1** | **Persistência: ŷ(t) = y(t−1)** · usa dados observados | V4 | **197** | **616** | **26,5%** | **0,984** | – | – | – |
| **B2** | **Climatologia: mediana por SE de 2015–22** · não usa nada do teste | V4 | 773 | 2.683 | 78,3% | 0,767 | – | – | – |

### Nível municipal

| # | MAE | RMSE | sMAPE | r | IC50 | IC95 |
|---|---|---|---|---|---|---|
| M1 | 5,23 | 50,5 | 74,6% | 0,474 | – | – |
| **M2** ★ | **3,89** | 51,3 | **39,5%** | **0,562** | 85,7% | 97,7% |
| M3 | 4,79 | 60,4 | 49,1% | 0,251 | 77,7% | 87,6% |
| B1 persistência | **1,75** | **14,5** | **37,4%** | **0,973** | – | – |
| B2 climatologia | 4,71 | 58,4 | 50,3% | 0,343 | – | – |

\* A célula 49 gravou os resultados no SQLite com `if_exists='append'`. O gráfico da célula 50 tem linhas "serrilhadas", que é o padrão de **duas execuções da mesma safra misturadas**. Trate esse número como não confiável.

### Como ler esses números

1. **O M2 foi de fato o melhor dos três modelos**, com folga. Em relação ao M1, o MAE estadual caiu 43% e o r subiu de 0,67 para 0,93.
2. **Mas nenhum modelo supera a persistência ingênua (B1).** Os modelos recebem casos observados da semana *t* do próprio período de teste (ver §4.1). Nessas condições, repetir o valor da semana anterior já dá MAE 197 e r 0,98. **O AGCRN está pior do que copiar a última semana.**
3. **A comparação justa para o desafio é a B2 (climatologia)**, que prevê a safra inteira sem ver nada depois da EW25. O M1 (830) e o M3 (979) ficam **abaixo da climatologia**. O M2 (476) fica acima, mas só porque usa informação que não estaria disponível na submissão real.
4. **Intervalos:** no nível municipal, o IC50 cobre 68–86% dos casos (deveria cobrir ~50%). Os intervalos estão **largos demais**. A cobertura municipal também parece melhor do que é por causa dos zeros: 0 ∈ [0, x] sempre. No estadual, o M2 está razoavelmente calibrado (IC50 59,5%, IC95 98%).
5. **sMAPE municipal** premia prever zero em semanas com zero casos. Use com cautela para comparar modelos.

---

## 2. Quais escolhas de modelagem funcionaram

As mudanças de M1 para M2 foram feitas **todas de uma vez**: banco V3 → V7, features, arquitetura, loss, critério de parada, reconciliação e deduplicação. Por isso **não dá para atribuir o ganho a uma mudança isolada**. O que dá para dizer é isto:

| Escolha | Evidência | Veredito |
|---|---|---|
| **Separar regimes seca × pico** (2 especialistas com features próprias) | M1 → M2: r estadual 0,67 → 0,93. O timing dos picos passou a bater | ✅ Mais promissora, mas precisa de teste isolado |
| **Loss assimétrica** (pico pune subestimar; seca pune falso alarme) | Faz parte do pacote M2. Mesmo assim, o M2 ainda subestima o Sul (PR, SC, RS) | ✅/❓ Ajuda, sem isolar |
| **Features espaciais W1/W2 + log-incidência + StandardScaler** (V7) | Chegaram junto com o M2 | ✅/❓ Coerente com a dinâmica de contágio, sem isolar |
| **Deduplicar as linhas** | O M1 usou dados duplicados: o "observado" estadual tem picos falsos de 6× em 2023-01-01 e 2023-10-01 | ✅ Obrigatório |
| **Early stopping pela loss** (em vez do MAE) | M2 usa loss, M1 usa MAE | ❓ Sem isolar |
| **MinT dinâmico (pico/seca)** vs MinT simples | Nunca comparados no mesmo modelo | ❓ Não testado |
| **Offset log(pop) + cabeça `r` com velocidade e aceleração** | Presente em todos | ✅ Boa prática para NB |
| **Multi-Head Attention no pico** (M3) | r estadual caiu para 0,39. As previsões colapsaram perto de zero | ❌ Piorou |
| **Treinar sem validação, até a última época** (M3) | Mesma execução. Provável sobreajuste ou colapso | ❌ Piorou |
| **Loss de pico multiplicativa** (×1,5 quando subestima, M3) | Mesma execução | ❌ Junto com as duas acima |
| **Ensemble AGCRN + LightGBM + Ridge + Optuna** (v1) | Nunca rodou até o fim | ⏸ Abandonado sem avaliação |

**Resumo:** a melhor configuração conhecida é o **M2** (V7 + bimodal + losses assimétricas + MinT dinâmico). O M3 deve ser descartado como está. Os pesos do M2 **não estão salvos**. Os `.pt` em `modelos/` são do M3.

---

## 3. Quais variáveis foram mais importantes

Não há valores de SHAP salvos. As saídas foram impressas no Colab e não ficaram no notebook. O que existe são as **listas selecionadas** em cada rodada, gravadas no código. Pela frequência com que aparecem:

### Selecionadas em todas as rodadas (núcleo estável)

| Variável | Tema | Rodadas |
|---|---|---|
| `velocidade`, `aceleracao` (e `casos_suavizados`) | Dinâmica da própria epidemia | todas |
| `access_count` (Afya) | Infodemiologia | SHAP geral, SECA, PICO |
| `imunidade_ewm` / `imunidade_ewm_t1` | Imunidade acumulada | todas |
| `seno_semana` (e `cosseno_semana` na 2ª rodada) | Sazonalidade | todas |
| `casos_chikungunya` → `log_inc_chiku(_t1)` | Co-circulação de arbovírus | todas |
| `koppen_encoded` | Clima regional (estático) | todas |

### Apareceram com a engenharia espacial (2ª rodada, em SECA e PICO)

`w1_log_inc_dengue_t1`, `w2_log_inc_dengue_t1`, `w1_velocidade_t1`, `w2_velocidade_t1`, `log_inc_dengue_t1`, `log_pop_t1`, `w1_log_pop_t1`, `temp_min`

### Específicas de cada regime

- **Só na SECA:** `w1_imunidade_t1`, `rel_humid_max`, `temp_med`. O clima basal e a imunidade dos vizinhos importam mais na entressafra.
- **Só no PICO:** `w1/w2_aceleracao_t1`, `access_count_t1`, `w1/w2_temp_forecast_t1`, `w2_umid_forecast_t1`, `w2_precip_forecast_t1`, `umid_med_forecast`, `temp_med_forecast`. No pico entram a aceleração espacial e o clima previsto.

### Leitura epidemiológica

O que sustenta o modelo é **a própria curva de casos**: nível, velocidade, aceleração, a mesma coisa nos vizinhos e a imunidade. Sazonalidade e Afya vêm em seguida. **O clima tem papel secundário.** Faz sentido em horizonte de 1 semana, em que o passado recente domina.

### ⚠️ Três ressalvas que mudam a interpretação

1. **Vazamento na seleção:** o SHAP prevê `casos.x` na semana *t* usando `casos_suavizados_t1`, `velocidade` e `aceleracao`. Essas três são calculadas **com o próprio y(t)** (o `_t1` delas é `shift(0)`). Por isso a importância delas está inflada.
2. **`access_count` é suspeita:** antes de 2021, o valor é o 1º dado de 2021 repetido (bfill), ou seja, uma **constante por município** que funciona como proxy de tamanho ou conectividade da cidade. Além disso, é o valor de **um dia** (o domingo), não a soma da semana. O alvo do SHAP é **casos absolutos** (não incidência), o que favorece variáveis de escala.
3. **Variáveis que nunca tiveram chance:** ENSO/IOD/PDO estão 100% vazias (= 0), e o clima previsto é uma função-degrau que muda poucas vezes por ano (83% preenchido por ffill/bfill). **Não é que o clima não importe. As variáveis climáticas estavam quebradas.**

---

## 4. Problemas críticos encontrados

Em ordem de gravidade. Os três primeiros impedem uma submissão válida.

### 4.1 A avaliação não reproduz o desafio (one-step-ahead com dados do futuro)
O desafio exige prever **EW41 → EW40 usando só dados até a EW25** ([Rules](../data_imdc_2026/Rules%20_.html) §5.1). No pipeline, o modelo recebe a cada semana do teste features calculadas com os **casos observados daquela semana**: `velocidade`, `aceleracao` e `log_inc_dengue_t1` vêm de `extrair_fisica(df_test)`, e `w1/w2_*` dos vizinhos. Com isso ele só prevê **1 semana à frente**. Na prática isso é *nowcasting*, não previsão da safra. A B1 mostra o efeito: com essa informação, copiar a última semana já ganha de todos os modelos.
**Consequência:** as métricas são otimistas e o pipeline **não consegue gerar a submissão** (em 10/10/2022 não se conhecem os casos de março de 2023). O protótipo v0 tinha um "loop de forecasting" recursivo que se perdeu no caminho.

### 4.2 Datas da submissão deslocadas em +7 dias
`montar_submissao_agcrn` converte `epiweek` com `'%G%V-%w'` (semana **ISO**, domingo = **fim** da semana ISO). O `dengue.csv` usa o domingo de **início** da SE. Resultado: o `.db` vai de 2022-10-16 a 2023-10-08, quando o oficial é **2022-10-09 a 2023-10-01**. A plataforma rejeitaria a submissão (a janela exigida é EW41–EW40) ou ela sairia deslocada. Em alguns anos a numeração ISO também difere da numeração das SE.

### 4.3 Possível desalinhamento de 1 semana entre previsão e observado
No treino, `modelo(X[t])` é comparado com `Y[t+1]`. Na submissão, a saída de `X[t]` é gravada na **semana t** e comparada com `casos_reais` da semana t. Pelo código, cada linha do `.db` compara "previsão de t+1" com "observado de t". Vale conferir com um gráfico de um município só: a curva prevista deve aparecer **adiantada 1 semana**.

### 4.4 Junções do R geram duplicatas e colunas vazias
Detalhes em [02 §3](02_BANCOS_DE_DADOS.md#3-qualidade-das-colunas-de-origem-problemas-herdados-do-join-em-r). Em resumo:
- Join exato com o **clima previsto mensal** → 83% NA e cada semana que cai no dia 1 do mês fica **6× duplicada** (~15% das linhas).
- **ENSO/IOD/PDO 100% NA**, porque as datas não caem em domingo.
- **Afya** pega o valor de 1 dia, e antes de 2021 é constante.
- `hti_lag2` é igual a `hti_lag3`.
- O **M1 foi treinado e avaliado com as duplicatas**: o "observado" estadual tem picos falsos de 6×.
- **V6/V7 foram gerados a partir de V5a (com duplicatas)**, não de V5b. O notebook remove as duplicatas com `keep='first'`, que fica com um horizonte de previsão arbitrário.

### 4.5 `ffill().bfill()` por município na engenharia
O `bfill` copia valores **do futuro para o passado**: o Afya de 2021 vai para 2010, a chikungunya de 2014 vai para 2010. Também cria features constantes que se passam por informativas.

### 4.6 Lags climáticos inconsistentes entre safras
As safras 1–2 usam lags 3–5 e as safras 3–4 usam lags 2–4 (o código mudou entre execuções). Isso não afeta V7 (que só guarda `temp_min_lag4` e `pressure_max_lag3`, presentes em todas), mas afeta V2 e V5.

### 4.7 Validação interna do modelo bimodal não valida a SECA
A validação são as **últimas 10 semanas do treino (SE 16–25)**, que estão **todas no regime PICO** (SE ≥ 47 ou ≤ 25). O especialista SECA nunca é validado. Por isso aparece `L_Val=0.000` na célula 61, e o early stopping dele nunca atua.
Os limites de regime também são **inconsistentes** entre as etapas: modelo 47–25, MinT de backup 49–22, SHAP 49–22 / 23–48.

### 4.8 Resultados acumulados no SQLite
`to_sql(..., if_exists='append')` sem identificar a execução. Reexecuções da mesma safra se misturam, o que provavelmente explica a célula 49/50. Salve com `run_id` e nome do modelo, ou use `replace`.

### 4.9 Cobertura experimental
- **Só a S1 foi avaliada.** Não dá para saber se o M2 generaliza para a S2 (2023-24, o maior surto da série), a S3 ou a S4.
- **Não havia baseline** até agora.
- Os pesos do M2 não foram salvos.

### 4.10 Menores
- `time_index` é calculado com as duplicatas.
- Queen gera 2 "ilhas" (2605459 Fernando de Noronha, 3520400 Ilhabela), com vizinhança vazia e lags espaciais = 0.
- Nomes `_t1` são enganosos nas variáveis físicas.
- Na célula 33, `except ImportFalse:` é erro de digitação.

---

## 5. Quais bancos podem ser eliminados

Tamanho total de `data_imdc_2026/`: **38,6 GB**.

| Banco | Tamanho | Situação | Recomendação |
|---|---|---|---|
| `./resultados_agcrn.db` | 0 B | Vazio | 🗑 **Apagar** |
| `Dengue dasa/datanalysis in dengue.R` | 49 B | Esboço, só `library()` | 🗑 **Apagar** (a versão real está em `data_imdc_2026/`) |
| Uma das duas cópias do docx | 2,7 MB | Idênticas | 🗑 Manter só uma |
| `variaveis/access_afya_…csv` | 296 MB | Mesma coisa que o `.csv.gz` | 🗑 Manter só o `.gz` |
| **V5a** `bancosDEF/*_COMPLETO_spatial` | **9,1 GB** | Versão com bug (duplicatas), substituída por V5b | 🗑 **Apagar**, depois de regenerar V6/V7 a partir de V5b |
| **V2** `./*_dengue_eng` | **7,6 GB** | Intermediário. Todas as colunas dele estão em V5. Lags inconsistentes | 🗑 **Apagar** (ou arquivar externamente) |
| **V4** `bancos/*_spatial_v2` | 2,0 GB | Intermediário, contido em V5 | 📦 Arquivar ou apagar |
| **V6** `bancosDEF/*_LIMPO` | 2,5 GB | Intermediário entre V5 e V7, com duplicatas | 🗑 Apagar (é regenerável em minutos) |
| **V3** `./*_shap_final` | 0,35 GB | Base do M1, com duplicatas e ES | 📦 Manter só se quiser reproduzir o M1 |
| **V1** `./*_dengue` | 2,6 GB | Raiz do pipeline, **com os bugs de join** | ♻ Manter, mas **regenerar** com os joins corrigidos |
| **V5b** `./*_COMPLETO_spatial` | 8,6 GB | Maior banco correto (sem duplicatas) | ✅ Manter como "banco mestre" enquanto o pipeline não é refeito |
| **V7** `bancosDEF/*_FINAL_NN` | 2,6 GB | Usado pelo melhor modelo, mas derivado de V5a | ✅ Manter até regenerar a partir de V5b |
| `variaveis/*` (brutos) | 3,3 GB | Fonte de tudo | ✅ Manter sempre |

**Economia imediata sem perder nada irreproduzível:** V5a + V2 + V6 + CSV duplicado ≈ **19,5 GB**. Com V4 e V3, chega a ~22 GB.

Antes de apagar V5a, regenere V6/V7 a partir de V5b (é só trocar o caminho de entrada em `TestesomenteAGCRN.py`, de `bancosDEF/…COMPLETO` para `./…COMPLETO`) e confira se o notebook do M2 continua rodando.

---

## 6. Próximos passos sugeridos (se o projeto continuar)

1. **Corrigir os joins no R:** clima previsto por mês (`floor_date(date, "month") == reference_month` + escolher o horizonte correto em relação à EW25 da safra), oscilações oceânicas por semana mais próxima, Afya somado por semana. Regenerar V1 → V7 sem duplicatas.
2. **Redefinir a tarefa como previsão da safra inteira** a partir da EW25. As opções são: (a) recursiva, realimentando as próprias previsões; (b) direta, com features só até a EW25 + clima previsto + sazonalidade. Reavaliar contra a **B2 (climatologia)**, que é a régua honesta.
3. **Corrigir as datas** (usar o `date` do `dengue.csv`, não `%G%V`) e o alinhamento t / t+1.
4. **Experimentos isolados:** M1 vs M1 + bimodal vs M1 + W1/W2 vs M1 + losses, cada um sobre o mesmo banco corrigido. Assim fica claro de onde vem o ganho.
5. **Rodar S1–S4** e salvar os pesos e as métricas com `run_id`.
6. **Refazer o SHAP** sem as features calculadas com y(t), com alvo em incidência, e só depois de consertar o clima.
