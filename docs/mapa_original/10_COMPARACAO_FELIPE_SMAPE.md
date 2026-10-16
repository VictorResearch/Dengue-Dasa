# 10 · Comparação com o modelo do Felipe ("SMAPE de 16%") (08/10/2026)

> **Pergunta:** o Felipe (github.com/felipe-matsuoka123/Dengue, commit 6f11b6d de 01/10) diz ter SMAPE de 16% por UF. Os dados de alguém estão errados? O modelo dele é melhor? Estamos medindo a mesma coisa?
> **Arquivos:** `data_imdc_2026/COMPARACAO_FELIPE/replica_felipe.py` (35 s de CPU) e `replica_felipe_resultados.csv`.

## 1. O que o código dele faz

- **Dados:** microdados do SINAN (`DENGBR*.csv.zip`) → casos prováveis por **UF de notificação** (`SG_UF`) e semana de início dos sintomas; tira descartados (5) e chikungunya (13). O ES fica de fora a partir de 2020 (o ES saiu do SINAN e foi para o e-SUS VS).
- **Modelo:** XGBoost, LightGBM e CatBoost com lags 1, 2, 4, 8 e 52, médias de 4 e 8 semanas, sen/cos, população e UF; mais a persistência.
- **Previsão:** **1 semana à frente** (`horizonte_semanas = 1`), usando o valor observado da semana anterior.
- **Avaliação:** ano civil (validação 2019–2022, teste 2023–2024). As métricas são MAE, RMSE e **WAPE**. **O repositório não calcula SMAPE.**

## 2. Os dados (rodei o código dele em 2022–2023)

Rodei `01_extrair_dengue.py` e `03_limpar_dengue.py --inicio 2022 --fim 2023` (2 zips de ~40 MB) e comparei com a nossa `tabela_uf_semanal` (Mosqlimate) na temporada 1:
- correlação semanal por UF de **0,9994**;
- total do Brasil 3,5% menor no dele (1,44 contra 1,49 milhão);
- por UF, de −11% (TO) a +9% (RN, AL). Isso é o efeito de **UF de notificação × UF de residência**: a Mosqlimate conta pela residência;
- o ES não existe no dele.

**Conclusão: nenhum dos dois está errado.** São a mesma fonte com definições um pouco diferentes. Isso não explica uma diferença grande no SMAPE.

## 3. O modelo dele nos nossos dados (resultados)

sMAPE = média de 2|p−y|/(|p|+|y|), em %, na escala de 0–200. A versão de 0–100 é a metade. Sem o ES.

| Protocolo | Modelo | MAE | WAPE | sMAPE | sMAPE (0–100) | sMAPE mediano |
|---|---|---|---|---|---|---|
| Dele: h = 1, teste 2023–24 | persistência | 507 | **16,6%** | 22,8% | 11,4% | 17,1% |
| | XGBoost | 1.165 | 38,1% | 22,3% | 11,2% | 15,2% |
| h = 1, temporada 1 (SE41/22–SE40/23) | persistência | 185 | **16,8%** | 24,8% | 12,4% | 17,1% |
| | XGBoost | 136 | 12,3% | 24,6% | 12,3% | **15,5%** |
| h = 4, temporada 1 | XGBoost | 388 | 35,2% | 49,9% | | |
| h = 13, temporada 1 | XGBoost | 778 | 70,6% | 84,1% | | |
| **Regra do IMDC** (dados até SE25, prever SE41→SE40, h = 16 a 68) | XGBoost dele (direto) | 911 | 82,6% | **96,4%** | 48,2% | |
| | persistência da SE25 | 1.214 | 110% | 106,9% | | |
| Equipes do 3º IMDC, temporada 1 | melhor (pucrio_arbocaster) | 575 | 52,1% | **59,6%** | | |
| | BB (modelo-base) | 757 | 68,7% | 72,4% | | |
| | emap sarimax_state | 895 | 81,2% | 75,5% | | |

## 4. Respostas

1. **Dados:** os dois estão certos. A diferença é de definição (notificação × residência; ES ausente no dele), corr. 0,9994.
2. **O modelo dele não é melhor.** Na regra da competição, o mesmo XGBoost tem sMAPE de 96%, pior que o BB (72%) e que a melhor equipe (60%). Na regra dele, h = 1, os boosters **perdem para a persistência** no teste 2023–24 (MAE 1.165 contra 507): as árvores não extrapolam o recorde de 2024.
3. **São desfechos diferentes.** Ele mede o erro de prever a **próxima semana** sabendo a semana atual, o que é quase um nowcast: só repetir a semana anterior já dá WAPE de ~17%. A competição mede a **temporada inteira, prevista de 16 a 68 semanas antes**, com WIS sobre 9 quantis. Para medir a mesma coisa, ele precisaria usar origem na SE25, horizonte SE41→SE40, temporadas da Mosqlimate (não o ano civil), casos por residência com o ES, quantis e WIS em relação ao BB.
4. **O "16%" se reproduz, mas só com h = 1.** Na temporada 1, a 1 semana: o WAPE da persistência é 16,8%, o sMAPE mediano do XGBoost é 15,5% e o sMAPE de 0–100 é 12,3%. O "SMAPE de 16%" quase certamente é o **WAPE** do relatório dele, ou o sMAPE na escala de 0–100 / mediano, sempre a 1 semana.

**Pendente:** calcular no Colab o sMAPE do nosso `ENS_UF_top5` na temporada 1, a partir de `final_uf.parquet`. Pelo WIS ele fica em 0,959 × BB na S1, então o sMAPE esperado fica perto do BB, uns 70%. → Agora isso está na parte A do `CompFelipe` (§5).

---

## 5. Jogos cruzados (08/10/2026)

**Arquivos** (em `data_imdc_2026/COMPARACAO_FELIPE/`):
- `felipe_no_meu_jogo.py`: 39 min de CPU, porque o PC estava dividido com outro job;
- `meu_no_jogo_dele.py`: 4 min;
- `semana_lib.py` e `CompFelipe.ipynb` (gerado por `_gerar_notebook_felipe.py`) para o Colab;
- as tabelas `*_resumo.csv` e as previsões `*_previsoes.parquet`.

**Validação da régua local:** a minha avaliação no PC reproduz o CompUF4 do Colab. A emap sarimax_state dá 0,870, a lncc 0,897 e a purdue 0,904, os mesmos valores do `teste_ranking.csv`. Por isso os números do Felipe abaixo são diretamente comparáveis aos nossos.

### 5.1 O Felipe no MEU jogo

Regras: dados até a SE25, previsão SE41→SE40 (h = 16…68), teste 2022–2025 (S1–S4), WIS em relação ao BB.

| Modelo | R × BB | UFs melhores que o BB | S1 · S2 · S3 · S4 | MAE | sMAPE |
|---|---|---|---|---|---|
| **MTGNN_UF+λ (meu melhor isolado)** | **0,846** | 20 | — | (CompFelipe A) | (CompFelipe A) |
| **ENS_UF_top5 (meu final)** | **0,853** | 20 | 0,959 · 0,980 · 0,792 · 0,712 | (CompFelipe A) | (CompFelipe A) |
| emap sarimax_state (melhor do IMDC) | 0,870 | 17 | 0,958 · 0,906 · 0,835 · 0,790 | 1.745 | 67,0% |
| BB | 1,000 | — | | 1.787 | 73,0% |
| pucrio arbocaster | 1,073 | 16 | 1,215 · 0,740 · 1,148 · 1,284 | **1.230** | **58,5%** |
| Felipe_XGB_quantis (as features dele + 9 quantis) | **1,256** | 6 | 1,297 · 1,011 · 1,491 · 1,272 | 2.007 | 74,9% |
| persistência da SE25 | 2,392 | 1 | | 2.213 | 93,4% |
| Felipe_XGB_ponto (o modelo dele, como está) | **3,303** | 0 | 3,264 · 1,709 · 5,535 · 3,857 | 2.949 | 96,0% |

- Como está, o modelo dele dá **só um valor por semana**. Sem intervalos, o WIS vira o erro absoluto e ele fica 3,3 vezes pior que o BB.
- Mesmo com quantis (1,256), ele perde para o BB, para a climatologia (0,993) e para 18 equipes do IMDC. O meu MTGNN+λ erra **33% menos** (0,846 / 1,256).
- **Por quê:** as features dele são lags de contagem e, a 16–68 semanas, o lag 1 já não diz quase nada. As árvores em contagens não extrapolam: em 2024 ficaram presas ao teto do treino. Também não há âncora sazonal nem informação dos vizinhos.

### 5.2 Eu no jogo DELE

Regras: prever 1 semana à frente, modelo fixo por rodada (treino até 31/12 do ano anterior), dados observados até a origem, ano civil. Estimativas locais com as minhas famílias de modelo; as redes de grafo rodam na parte B do `CompFelipe`.

| h = 1 | teste 2023–24: MAE · WAPE · sMAPE | temporada 1: MAE · WAPE · sMAPE | validações 2019–22: WAPE |
|---|---|---|---|
| persistência (o melhor dele no teste) | 507 · 16,6% · 22,8% | 185 · 16,8% · 24,8% | 15,3–17,8% |
| Felipe_XGB | 1.165 · **38,1%** · 22,3% | 136 · 12,3% · 24,6% | 15,1–20,3% |
| **GBM_log_cresc (estilo meu)** | **324 · 10,6% · 19,0%** | 115 · 10,4% · 22,8% | **10,7–12,7%** |
| ARIMA_Fourier um passo | 361 · 11,8% · 20,4% | 115 · 10,4% · 22,5% | 10,5–13,3% |
| ENS_GBM_ARIMA | 334 · 10,9% · 19,2% | **112 · 10,1% · 22,2%** | 10,2–12,7% |
| ENS_mediana (+ persistência) | 348 · 11,4% · 19,8% | 116 · 10,6% · 22,4% | 10,3–13,2% |

| h = 4 | teste 2023–24: MAE · WAPE · sMAPE | temporada 1: MAE · WAPE · sMAPE |
|---|---|---|
| persistência | 1.730 · 56,5% · 51,9% | 605 · 54,8% · 52,5% |
| Felipe_XGB | 1.537 · 50,2% · 43,7% | 388 · 35,2% · 49,9% |
| **GBM_log_cresc** | **1.028 · 33,6% · 36,2%** | **254 · 23,1% · 39,5%** |

- No jogo dele, o meu jeito de modelar ganha em **todas** as rodadas: MAE **36% menor** que o melhor dele (a persistência) e **72% menor** que o XGBoost dele no teste 2023–24.
- **O que faz a diferença:**
  - prever o **crescimento em log** (log1p da incidência na semana-alvo menos o da última semana observada), o que deixa extrapolar o recorde de 2024;
  - âncora sazonal (mesma semana nos anos anteriores);
  - vizinhos no grafo de fronteiras;
  - perda L1 (mediana).
- O XGBoost dele tem WAPE de 38% no teste porque, prevendo contagens, **não passa do máximo visto no treino**. Isso aparece no que ele mesmo escreveu: em 2023–24, os boosters perdem para a persistência.

### 5.3 O que funciona e o que não funciona

| Funciona | Não funciona |
|---|---|
| Alvo em log / crescimento relativo (os dois jogos) | Árvores em contagens brutas: não extrapolam epidemias recordes |
| Âncora sazonal + resíduo (as redes, o GBM_log) | Lags curtos a 16–68 semanas: o sinal some |
| Espaço (grafo, vizinhos) | Previsão só pontual na régua do WIS (×3,3) |
| Quantis com perda pinball e λ = 0,8 | Escolher pelo sMAPE: favorece quem subprevê picos |
| Ensembles simples (mediana) | Comparar números de jogos diferentes (16% × 70%) |

### 5.4 Top 5 dos meus modelos por métrica (teste 2022–2025; fonte: `teste_ranking.csv` e Comp5)

| Métrica | 1º | 2º | 3º | 4º | 5º |
|---|---|---|---|---|---|
| **R × BB (WIS, régua oficial)** | MTGNN_UF+λ 0,846 | ENS_UF_top7 0,850 | ENS_UF_top3 0,850 | ENS_UF_top5 0,853 | A3TGCN_UF+λ 0,855 |
| UFs melhores que o BB | ENS_UF_top3 21 | A3TGCN_UF+λ 21 | MTGNN_UF+λ 20 | ENS_UF_top5/top7 20 | VAR_macro_UF 20 |
| WIS em log (erro relativo) | GBM_lgb_sel_aum_UF 0,389 | GBM_cat_sel_aum_UF 0,393 | ARIMA_Fourier 0,399 | XGBoost_quantis 0,400 | GBM_xgb_sel_UF 0,403 |
| WIS médio em casos (pesa os estados grandes) | TFT_pool_UF 773 | Renovacao_S_UF 919 | NHITS_pool_UF 925 | Tamanho_x_forma_UF 979 | TFT_pool_semcov_UF 1.005 |
| Calibração (IC90 perto de 90%) | MTGNN_UF 89,9 | A3TGCN_UF 89,3 | DoisEstagios_UF 90,8 | GCNGRU_UF 88,8 | AutoARIMA_d0_r 86,7 |
| **Custo da compra (τ = 0,75, teste)** | DoisEstagios_pool_sel_UF 0,846 | DoisEstagios_pool_UF 0,871 | A3TGCN_UF 0,899 | MTGNN_UF 0,904 | DoisEstagios_UF 0,913 |
| MAE / WAPE / sMAPE semanais | ver `top5_por_metrica.csv` (parte A2 do CompFelipe) | | | | |

**Leitura:**
- **O WIS médio em casos discorda da régua oficial.** O BB tem 1.085 e o MTGNN+λ tem 1.168: no total de casos, os meus melhores **erram mais que o BB nos estados grandes** (SP, MG e PR em 2024).
- Os que mais acertam em casos (TFT_pool, NHITS_pool; no IMDC, a pucrio com 644) são ruins no WIS relativo.
- A média geométrica dá o mesmo peso a cada UF. Para a empresa, se ela vende mais em SP e MG, a métrica certa pesa por volume.

### 5.5 Inventário: o que já foi testado em cada jogo (08/10/2026)

- **Semanal (h = 1), já testado:**
  - AGCRN M1–M3, v2 e v3.1 (só a S1, tabela do `00` §4);
  - Ridge AR de 1 semana (anexo `experimento_1semana_vs_temporada`, S1–S4 com WIS);
  - persistência e climatologia;
  - hoje: GBM_log_cresc, ARIMA_Fourier um passo, ENS_GBM_ARIMA, ENS_mediana, Felipe XGB/LGBM.
- **4 semanas:** persistência, Felipe XGB, GBM_log_cresc.
- **Temporada:** os 83 modelos do CompUF + IMDC + Felipe.
- **Nunca testados no semanal:** as redes de grafo (CompFelipe B), VAR, AutoARIMA, NHITS/TFT e Chronos.
- **Extrapolação sem retreinar:**
  - **dá certo** para o ARIMA/VAR (aplicar o filtro com os parâmetros fixos) e para o Chronos (só inferência);
  - **não dá** para os GBMs de temporada (treinados para h = 16–68) nem para os modelos de dois estágios, tamanho × forma etc.;
  - **as redes de grafo** são treinadas com origens em todas as semanas e saídas para h = 1…68, então a saída h = 1 existe. Mas os pesos não foram salvos, então é preciso retreinar (H = 68 reproduz o modelo de temporada; H = 4 é a versão especializada).
- **Decisão:** não retreinar tudo nos dois objetivos.
  - Rodar primeiro o CompFelipe A (CPU) e B (GPU, de preferência com H = 4 e H = 68).
  - Opcional e barato: VAR/AutoARIMA um passo, Chronos sem treino, NHITS/TFT com h = 1–4.

## 6. Próximos passos

**Prevendo semanas à frente (temporada, a régua oficial):**
1. **Ensemble só de grafos com λ** (MTGNN + A3TGCN + GCNGRU), escolhido na afinação; é o que já ganha.
2. **Corrigir os estados grandes:**
   - empilhar (*stacking*) por tamanho da UF, juntando grafos com TFT_pool/NHITS_pool, que ganham no WIS em casos;
   - ou treinar a rede com pinball **ponderado pela população** além do log.
   - Mostrar as duas réguas (geométrica e em casos).
3. **MTGNN v2** (já planejado no 09): 5 sementes, λ por horizonte, cabeça auxiliar do total da temporada e pooling com as regiões.
4. **Estudar a pucrio arbocaster:** é a melhor em casos (WIS 644, sMAPE 58%), mas instável (R 1,07). O README está em `anexos/`. Pode entrar como membro do ensemble só nos estados grandes.
5. **Alvo em crescimento também na temporada:** testar o MTGNN prevendo o resíduo em relação à âncora × o último nível (o truque que funcionou no jogo dele).

**Prevendo a próxima semana (o jogo dele, útil para a empresa acompanhar):**
1. **Linha de base pronta:** ENS_GBM_ARIMA (WAPE ~10–11% a 1 semana, ~34% a 4 semanas). Ela já vence o melhor dele com folga.
2. **Redes de grafo a 1–4 semanas** (CompFelipe B) e ensemble grafos + GBM_log + ARIMA.
3. **Quantis a 1–4 semanas** (GBM quantílico ou conformal sobre o GBM_log) e WIS de curto prazo, para a decisão semanal de reposição.
4. **Atraso de notificação:** os 10% supõem dados consolidados. Em tempo real, a última semana chega incompleta. Testar com *vintages* ou com o *nowcast* do InfoDengue antes de prometer isso à empresa.
5. **Um modelo multi-horizonte** (h como variável, 1–12 semanas) que se emende com a previsão da temporada: o curto prazo corrige a trajetória sazonal a cada semana.
