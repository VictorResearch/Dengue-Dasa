# 02 · Bancos de dados: genealogia, versões e dicionário

> Tudo aqui foi conferido lendo os schemas e contando linhas em todos os `.parquet` (28/09/2026).
> Pasta-base: `F:\AmbientePython\Projetos\data_imdc_2026\` (abreviada como `./`).
> **Atualizado em 29/09/2026:** os bancos em uso hoje são os **V8** (seções 7, 8 e 9). As seções 1 a 6 descrevem os bancos antigos (V1–V7) e ficam como histórico.

---

## 1. Árvore genealógica

```
                         variaveis/*.csv  (V0 · brutos)
                                  │  R: datanalysis in dengue.R  (joins + VPD/HTI/volatilidade)
                                  ▼
                 ./train{k}_dengue.parquet · ./test{k}_dengue.parquet     (V1 · 69 col.)
                                  │  notebook cél.12: engenharia_de_features + extrair_fisica
                                  ▼
                 ./train{k}_dengue_eng.parquet                             (V2 · 138 col.)
                                  │  SHAP geral → TestesomenteAGCRN.py "FILTRAGEM PÓS-SHAP"
                                  ▼
                 ./train{k}_dengue_eng_shap_final.parquet                  (V3 · 22 col.)  ──► MODELO 1
                                  │  notebook cél.5–7: dedup, sem ES, lags _t1, W1/W2
                                  ▼
                 ./bancos/train{k}_dengue_eng_shap_final_spatial_v2        (V4 · 53 col.)
                                  │  R loop: V4 ⟕ (colunas de V2 que V4 não tem)
                    ┌─────────────┴───────────────┐
                    ▼                             ▼
  ./bancosDEF/*_COMPLETO_spatial (V5a)   ./*_COMPLETO_spatial (V5b)            (170 col.)
   18/06 · SEM distinct → duplicatas      19/06 · COM distinct → limpo
                    │  TestesomenteAGCRN.py: filtro união-SHAP (geral+seca+pico)
                    ▼
  ./bancosDEF/*_dengue_eng_LIMPO_spatial                                   (V6 · 55 col.)
                    │  TestesomenteAGCRN.py: log1p(incidência) + StandardScaler (fit no treino)
                    ▼
  ./bancosDEF/*_FINAL_NN                                                   (V7 · 55 col.)  ──► MODELOS 2 e 3
```

`{k}` = safra 1..4. Cada versão existe em 8 arquivos (train/test × 4 safras). **V6 e V7 vêm de V5a (com duplicatas), não de V5b.** Dá para ver pela contagem de linhas: V6 e V7 têm 4.179.412 linhas na S1, o mesmo que V5a. O notebook remove as duplicatas na hora de carregar.

---

## 2. Ficha de cada versão

### Recortes temporais (iguais em todas as versões)

| Safra | Treino | Teste | Semanas de teste disponíveis |
|---|---|---|---|
| S1 | 201001 – 202225 | 202241 – 202340 | 52 |
| S2 | 201001 – 202325 | 202341 – 202440 | 52 |
| S3 | 201001 – 202425 | 202441 – 202540 | 52 |
| S4 | 201001 – 202525 | 202541 – **202610** | **23** (o `dengue.csv` termina em 08/03/2026) |

Entre o fim do treino (EW25) e o começo do teste (EW41) há **15 semanas sem dados em nenhum banco**. É assim por regra do desafio.

### Tabela de versões

| Versão | Arquivos | Col. | Linhas S1 treino / teste | Duplic. (geocode, date) | ES? | Tamanho (8 arq.) | Gerado por | Data |
|---|---|---|---|---|---|---|---|---|
| **V0** | `variaveis/*.csv/.gpkg` | – | – | – | sim | 3,3 GB | download | mar–jun |
| **V1** | `./{train,test}{k}_dengue.parquet` | 69 | 4.238.770 / 345.340 | **612.700 / 55.700** | sim | 2,6 GB | `datanalysis in dengue.R` | 16/06 |
| **V2** | `./*_dengue_eng.parquet` | 138 | 4.238.770 / 345.340 | idem V1 | sim | 7,6 GB | notebook cél. 12 | 16–17/06 |
| **V3** | `./*_dengue_eng_shap_final.parquet` | 22 | 4.238.770 / 345.340 | idem V1 | sim | 0,35 GB | `TestesomenteAGCRN.py` l.968+ | 17/06 |
| **V4** | `./bancos/*_shap_final_spatial_v2.parquet` | 53 | 3.575.292 / 285.584 | **0** | não | 2,0 GB | notebook cél. 7 | 17/06 |
| **V5a** | `./bancosDEF/*_COMPLETO_spatial.parquet` | 170 | 4.179.412 / 340.504 | **604.120 / 54.920** | não | 9,1 GB | R (loop final, 1ª execução) | 18/06 |
| **V5b** | `./*_COMPLETO_spatial.parquet` | 170 | 3.575.292 / 285.584 | **0** | não | 8,6 GB | R (loop final, com `distinct`) | 19/06 |
| **V6** | `./bancosDEF/*_LIMPO_spatial.parquet` | 55 | 4.179.412 / 340.504 | 604.120 / 54.920 | não | 2,5 GB | `TestesomenteAGCRN.py` | 19/06 |
| **V7** | `./bancosDEF/*_FINAL_NN.parquet` | 55 | 4.179.412 / 340.504 | 604.120 / 54.920 | não | 2,6 GB | `TestesomenteAGCRN.py` | 19/06 |

Municípios: 5.570 com ES (V1–V3), **5.492 sem ES** (V4–V7).

### Diferenças entre safras dentro da mesma versão

- **V2 e V5 (a/b):** as safras **1–2** têm lags climáticos `_lag3/_lag4/_lag5`. As safras **3–4** têm `_lag2/_lag3/_lag4` (o código mudou no meio do processamento). O `_lag5` não existe em S3–S4 e o `_lag2` não existe em S1–S2.
- **V3 (teste):** `uf_code`, `population` e os geocódigos regionais estão como `double` (no treino são `int32`).
- **V1 teste S4:** `population` tem **43,5% de NA** (não há população de 2026). Esses NAs foram preenchidos por ffill em V2.

---

## 3. Qualidade das colunas de origem (problemas herdados do join em R)

Medido em `train4_dengue.parquet` (V1), que é a base de tudo:

| Variável(is) | % NA | Causa | Consequência nos bancos seguintes |
|---|---|---|---|
| `enso`, `iod`, `pdo` | **100%** | `ocean_climate_oscillations.csv` tem datas semanais que **não caem em domingo** (ex.: 1993-01-04). O join exato por `date` nunca casa | Viram **0 constante** em V2 (`fillna(0)`). Todos os `*_lag6..9` são inúteis |
| `temp_med_forecast`, `umid_med_forecast`, `precip_tot_forecast`, `forecast_months_ahead` | **83%** | `forecasting_climate` é mensal (`reference_month` = dia 1). O join `date = reference_month` só casa quando **o domingo cai no dia 1**, e aí casa **6 vezes** (horizontes 1–6) | (a) **duplica** a semana 6× → ~15% de linhas duplicadas; (b) nas demais semanas o valor é preenchido por **ffill/bfill** → vira uma função-degrau que muda poucas vezes por ano |
| `access_count` (Afya) | 77% (treino) · 19% (teste) | Série diária, casada pelo domingo (pega **o valor de 1 dia**, não a soma da semana). Só existe a partir de 2021 | Antes de 2021 o **bfill** repete o 1º valor de 2021 → vira **constante por município** entre 2010 e 2020 |
| `casos.y` / `casos_chikungunya` | 26% | Série de chikungunya começa em 2014 | bfill → 2010–2013 recebem o valor de 2014 |
| `hti_lag2` | – | Bug no R: é `lag(precip,3)*lag(temp,3)`, **igual a `hti_lag3`** | Coluna redundante |
| `amplitude_termica_dinamica*`, `volatilidade_*` | ~0% | Médias móveis calculadas **sobre as linhas duplicadas** | Levemente distorcidas nas semanas duplicadas |

As duplicatas aparecem, por exemplo, em **2023-01-01** e **2023-10-01** no teste da S1, cada uma com 6 cópias por município.

---

## 4. Qual banco alimenta qual modelo

| Modelo | Banco de treino/teste | Dedup? | Features usadas |
|---|---|---|---|
| Modelo 1 (AGCRN simples) | **V3** `_eng_shap_final` | ❌ (treinou e avaliou com duplicatas) | 12: física (3) + SHAP (9) |
| Modelo 2 (bimodal) | **V7** `FINAL_NN` | ✅ no carregamento | SECA 21 / PICO 27 (`COLS_SECA`, `COLS_PICO`) |
| Modelo 3 (MHA) | **V7** `FINAL_NN` | ✅ no carregamento | idem M2 |
| Inferência de recuperação (cél. 38) | **V7** | ✅ | idem M2 |
| SHAP por regime | **V5a** `bancosDEF/COMPLETO` | ❌ | todas as numéricas |
| Modelo 2 enxuto (29/09) | **V8** `S1_corrigido/*_FINAL_NN` | ✅ sem duplicatas por construção | mesmas listas do M2 de junho |
| **Modelo 2 v2 e v3** (29/09) | **V8** `S{N}_corrigido/*_FINAL_NN_v2` + `ctx{N}_FINAL_NN_v2` | ✅ | SECA 20 / PICO 20 (listas do SHAP corrigido) |
| SHAP corrigido (29/09) | **V8** `S{N}_corrigido/*_COMPLETO_spatial` | ✅ | 142 candidatas + 3 sondas de ruído |

---

## 5. Dicionário de variáveis

Legenda das colunas de presença: ✔ = presente em todas as safras · `S1-2` / `S3-4` = só nessas safras · vazio = ausente.
**Origem** = em que etapa a variável nasce. **`_t1`** no nome quer dizer "valor conhecido na semana *t*, usado para prever *t+1*".

> ⚠️ **Nome enganoso:** as variáveis físicas `casos_suavizados_t1`, `velocidade_t1` e `aceleracao_t1` são geradas com `shift(0)`, ou seja, **são o valor da própria semana t**, não de t−1. As outras `_t1` (clima, Afya, população, chikungunya) usam `shift(1)`.

Total: **193 variáveis distintas** no conjunto V1–V7.

### Chaves e identificação (16)

| Variável | Tipo | Descrição | Origem | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `geocode` | int32 | Código IBGE do município (7 dígitos) · chave | dengue.csv | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `date` | string | Data da semana epidemiológica (domingo inicial) · chave | dengue.csv | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `epiweek` | int32 | Semana epidemiológica AAAASS · chave | dengue.csv | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `uf` | string | Sigla da UF | dengue.csv | ✔ | ✔ | ✔ | ✔ | ✔ |  |  |
| `macroregional_geocode` | int32 | Código da macrorregião de saúde (usado na máscara do MHA) | dengue.csv | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `regional_geocode` | int32 | Código da região de saúde | dengue.csv | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `uf_code` | int32 | Código IBGE da UF (32 = ES, removido a partir de V4) | dengue.csv | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `year` | double | Ano (de epiweek) · chave do join com população | R | ✔ | ✔ |  |  | ✔ | ✔ | ✔ |
| `macroregion_code` | int32 | Código da grande região (1=Norte … 5=Sul) | map_regional_health.csv | ✔ | ✔ |  |  | ✔ | ✔ | ✔ |
| `macroregion_name` | string | Nome da grande região | map_regional_health.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `uf_name` | string | Nome da UF | map_regional_health.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `macroregional_name` | string | Nome da macrorregião de saúde | map_regional_health.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `regional_name` | string | Nome da região de saúde | map_regional_health.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `geocode_name` | string | Nome do município | map_regional_health.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `time_index` | int64 | Índice temporal por município (cumcount, **conta duplicatas**) | eng |  | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `node_id` | int64 | Índice do nó no grafo (0…5491, ordem de geocode) | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |

### Alvo e casos (4)

| Variável | Tipo | Descrição | Origem | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `casos.x` | int32 | **ALVO**: casos de dengue notificados na semana (sufixo .x vem do join com chikungunya) | dengue.csv | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `casos.y` | int32 | Casos de chikungunya na semana (NA antes de 2014) | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `incidencia` | double | casos/pop × 100 mil **na semana t** (não é feature do modelo) | eng |  | ✔ |  |  | ✔ | ✔ | ✔ |
| `casos_chikungunya` | double | = casos.y renomeado | chikungunya.csv |  |  | ✔ | ✔ | ✔ | ✔ |  |

### Flags e lixo de merge (descartáveis) (21)

| Variável | Tipo | Descrição | Origem | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `target_city.x` | string | Flag: município-alvo do desafio municipal (dengue) | dengue.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `train_1.x` | string | Flag do edital: semana no treino da safra 1 (dengue) · lixo de merge | dengue.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `target_1.x` | string | Flag do edital: semana no alvo da safra 1 (dengue) · lixo de merge | dengue.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `train_2.x` | string | Flag do edital: semana no treino da safra 2 (dengue) · lixo de merge | dengue.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `target_2.x` | string | Flag do edital: semana no alvo da safra 2 (dengue) · lixo de merge | dengue.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `train_3.x` | string | Flag do edital: semana no treino da safra 3 (dengue) · lixo de merge | dengue.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `target_3.x` | string | Flag do edital: semana no alvo da safra 3 (dengue) · lixo de merge | dengue.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `train_4.x` | string | Flag do edital: semana no treino da safra 4 (dengue) · lixo de merge | dengue.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `target_4.x` | string | Flag do edital: semana no alvo da safra 4 (dengue) · lixo de merge | dengue.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `disease.x` | string | Texto "dengue" (constante) | dengue.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `target_city.y` | string | Flag: município-alvo (chikungunya) | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `train_1.y` | string | Flag do edital: semana no treino da safra 1 (chikungunya) · lixo de merge | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `target_1.y` | string | Flag do edital: semana no alvo da safra 1 (chikungunya) · lixo de merge | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `train_2.y` | string | Flag do edital: semana no treino da safra 2 (chikungunya) · lixo de merge | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `target_2.y` | string | Flag do edital: semana no alvo da safra 2 (chikungunya) · lixo de merge | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `train_3.y` | string | Flag do edital: semana no treino da safra 3 (chikungunya) · lixo de merge | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `target_3.y` | string | Flag do edital: semana no alvo da safra 3 (chikungunya) · lixo de merge | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `train_4.y` | string | Flag do edital: semana no treino da safra 4 (chikungunya) · lixo de merge | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `target_4.y` | string | Flag do edital: semana no alvo da safra 4 (chikungunya) · lixo de merge | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `disease.y` | string | Texto "chikungunya" (constante) | chikungunya.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `accessed_disease` | string | Texto "Dengue" (Afya) | access_afya | ✔ | ✔ |  |  | ✔ |  |  |

### Clima observado e derivados (R) (21)

| Variável | Tipo | Descrição | Origem | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `temp_min` | double | Temperatura mínima semanal (°C) | climate.csv | ✔ | ✔ |  |  | ✔ | ✔ | ✔ |
| `temp_med` | double | Temperatura média semanal (°C) | climate.csv | ✔ | ✔ |  |  | ✔ | ✔ | ✔ |
| `temp_max` | double | Temperatura máxima semanal (°C) | climate.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `precip_min` | double | Precipitação mínima diária na semana (mm) | climate.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `precip_med` | double | Precipitação média diária (mm) | climate.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `precip_max` | double | Precipitação máxima diária (mm) | climate.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `pressure_min` | double | Pressão atmosférica mínima (escala normalizada ~0,95) | climate.csv | ✔ | ✔ | ✔ | ✔ | ✔ |  |  |
| `pressure_med` | double | Pressão média | climate.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `pressure_max` | double | Pressão máxima | climate.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `rel_humid_min` | double | Umidade relativa mínima (%) | climate.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `rel_humid_med` | double | Umidade relativa média (%) | climate.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `rel_humid_max` | double | Umidade relativa máxima (%) | climate.csv | ✔ | ✔ |  |  | ✔ | ✔ | ✔ |
| `thermal_range` | double | Amplitude térmica semanal (°C) | climate.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `rainy_days` | int32 | Nº de dias com chuva na semana | climate.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `vpd` | double | Déficit de pressão de vapor (kPa) = es(temp_med)·(1−UR/100) | R | ✔ | ✔ |  |  | ✔ |  |  |
| `amplitude_termica_dinamica3` | double | Média móvel de 3 semanas da amplitude térmica | R | ✔ | ✔ |  |  | ✔ |  |  |
| `amplitude_termica_dinamica2` | double | Média móvel de 2 semanas da amplitude térmica | R | ✔ | ✔ |  |  | ✔ |  |  |
| `volatilidade_temp_4w` | double | Desvio-padrão móvel de 4 semanas de temp_med | R | ✔ | ✔ |  |  | ✔ | ✔ | ✔ |
| `volatilidade_precip_4w` | double | Desvio-padrão móvel de 4 semanas de precip_med | R | ✔ | ✔ |  |  | ✔ |  |  |
| `hti_lag3` | double | Índice hidrotérmico = precip_med(t−3)·temp_med(t−3) | R | ✔ | ✔ |  |  | ✔ |  |  |
| `hti_lag2` | double | **Bug**: idêntico a hti_lag3 (deveria usar lag 2) | R | ✔ | ✔ |  |  | ✔ |  |  |

### Clima previsto (ECMWF) (7)

| Variável | Tipo | Descrição | Origem | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `forecast_months_ahead` | int32 | Horizonte (1–6 meses) da previsão climática. **Causa das duplicatas** | forecasting_climate.csv | ✔ | ✔ |  |  | ✔ | ✔ | ✔ |
| `temp_med_forecast` | double | Temperatura média PREVISTA (ECMWF) para o mês. 83% NA na origem → ffill/bfill | forecasting_climate.csv | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `umid_med_forecast` | double | Umidade média PREVISTA (ECMWF). 83% NA → ffill/bfill | forecasting_climate.csv | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `precip_tot_forecast` | double | Precipitação total PREVISTA (ECMWF). 83% NA → ffill/bfill | forecasting_climate.csv | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `umid_med_forecast_t1` | double | umid_med_forecast da semana anterior (shift(1) por município) | espacial |  |  |  | ✔ | ✔ |  |  |
| `precip_tot_forecast_t1` | double | precip_tot_forecast da semana anterior (shift(1) por município) | espacial |  |  |  | ✔ | ✔ |  |  |
| `temp_med_forecast_t1` | double | temp_med_forecast da semana anterior (shift(1) por município) | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |

### Oscilações oceânicas (3)

| Variável | Tipo | Descrição | Origem | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `enso` | double | Índice ENSO. **100% NA** (datas não casam) | ocean_climate_oscillations.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `iod` | double | Índice IOD. **100% NA** | ocean_climate_oscillations.csv | ✔ | ✔ |  |  | ✔ |  |  |
| `pdo` | double | Índice PDO. **100% NA** | ocean_climate_oscillations.csv | ✔ | ✔ |  |  | ✔ |  |  |

### Ambiente, população e infodemiologia (11)

| Variável | Tipo | Descrição | Origem | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `koppen` | string | Classe climática Köppen (texto) | environ_vars.csv | ✔ | ✔ |  |  | ✔ | ✔ | ✔ |
| `biome` | string | Bioma (texto) | environ_vars.csv | ✔ | ✔ |  |  | ✔ | ✔ | ✔ |
| `population` | int32 | População residente (DATASUS, anual; 2026 ausente → ffill) | datasus_population | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `access_count` | double | Acessos a conteúdo de dengue na plataforma Afya **no domingo** (valor de 1 dia). Antes de 2021 → bfill (constante) | access_afya | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `log_pop` | double | log(população) · offset da NB (não padronizado) | eng |  | ✔ |  |  | ✔ | ✔ | ✔ |
| `afya_log_diff` | double | Δ log1p(access_count) semana a semana | eng |  | ✔ |  |  | ✔ |  |  |
| `biome_encoded` | double | Target encoding do bioma (média de casos no treino) | eng |  | ✔ |  |  | ✔ |  |  |
| `koppen_encoded` | double | Target encoding do Köppen (média de casos no treino) | eng |  | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `access_count_t1` | double | access_count da semana anterior (shift(1) por município) | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `population_t1` | double | population da semana anterior (shift(1) por município) | espacial |  |  |  | ✔ | ✔ | ✔ |  |
| `log_pop_t1` | double | log(population_t1) | FINAL_NN |  |  |  |  |  |  | ✔ |

### Lags climáticos e macroclimáticos (criados em V2) (72)

| Variável | Tipo | Descrição | Origem | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `temp_min_lag3` | double | temp_min de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `temp_min_lag4` | double | temp_min de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ | ✔ | ✔ |
| `temp_min_lag5` | double | temp_min de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `temp_med_lag3` | double | temp_med de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `temp_med_lag4` | double | temp_med de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `temp_med_lag5` | double | temp_med de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `temp_max_lag3` | double | temp_max de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `temp_max_lag4` | double | temp_max de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `temp_max_lag5` | double | temp_max de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `precip_min_lag3` | double | precip_min de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `precip_min_lag4` | double | precip_min de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `precip_min_lag5` | double | precip_min de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `precip_med_lag3` | double | precip_med de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `precip_med_lag4` | double | precip_med de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `precip_med_lag5` | double | precip_med de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `precip_max_lag3` | double | precip_max de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `precip_max_lag4` | double | precip_max de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `precip_max_lag5` | double | precip_max de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `thermal_range_lag3` | double | thermal_range de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `thermal_range_lag4` | double | thermal_range de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `thermal_range_lag5` | double | thermal_range de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `pressure_min_lag3` | double | pressure_min de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `pressure_min_lag4` | double | pressure_min de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `pressure_min_lag5` | double | pressure_min de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `pressure_med_lag3` | double | pressure_med de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `pressure_med_lag4` | double | pressure_med de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `pressure_med_lag5` | double | pressure_med de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `pressure_max_lag3` | double | pressure_max de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ | ✔ | ✔ |
| `pressure_max_lag4` | double | pressure_max de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `pressure_max_lag5` | double | pressure_max de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `rel_humid_min_lag3` | double | rel_humid_min de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `rel_humid_min_lag4` | double | rel_humid_min de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `rel_humid_min_lag5` | double | rel_humid_min de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `rel_humid_med_lag3` | double | rel_humid_med de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `rel_humid_med_lag4` | double | rel_humid_med de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `rel_humid_med_lag5` | double | rel_humid_med de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `rel_humid_max_lag3` | double | rel_humid_max de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `rel_humid_max_lag4` | double | rel_humid_max de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `rel_humid_max_lag5` | double | rel_humid_max de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `rainy_days_lag3` | double | rainy_days de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `rainy_days_lag4` | double | rainy_days de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `rainy_days_lag5` | double | rainy_days de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `vpd_lag3` | double | vpd de 3 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `vpd_lag4` | double | vpd de 4 semanas antes (merge_asof, tolerância 7 dias) | eng |  | ✔ |  |  | ✔ |  |  |
| `vpd_lag5` | double | vpd de 5 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S1-2 |  |  | S1-2 |  |  |
| `enso_lag6` | double | enso de 6 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `enso_lag7` | double | enso de 7 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `enso_lag8` | double | enso de 8 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `enso_lag9` | double | enso de 9 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `iod_lag6` | double | iod de 6 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `iod_lag7` | double | iod de 7 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `iod_lag8` | double | iod de 8 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `iod_lag9` | double | iod de 9 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `pdo_lag6` | double | pdo de 6 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `pdo_lag7` | double | pdo de 7 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `pdo_lag8` | double | pdo de 8 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `pdo_lag9` | double | pdo de 9 semanas antes (merge_asof, tolerância 14 dias) · **sempre 0** (origem 100% NA) | eng |  | ✔ |  |  | ✔ |  |  |
| `temp_min_lag2` | double | temp_min de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `temp_med_lag2` | double | temp_med de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `temp_max_lag2` | double | temp_max de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `precip_min_lag2` | double | precip_min de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `precip_med_lag2` | double | precip_med de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `precip_max_lag2` | double | precip_max de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `thermal_range_lag2` | double | thermal_range de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `pressure_min_lag2` | double | pressure_min de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `pressure_med_lag2` | double | pressure_med de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `pressure_max_lag2` | double | pressure_max de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `rel_humid_min_lag2` | double | rel_humid_min de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `rel_humid_med_lag2` | double | rel_humid_med de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `rel_humid_max_lag2` | double | rel_humid_max de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `rainy_days_lag2` | double | rainy_days de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |
| `vpd_lag2` | double | vpd de 2 semanas antes (merge_asof, tolerância 7 dias) | eng |  | S3-4 |  |  | S3-4 |  |  |

### Epidemiológicas: física, imunidade, sazonalidade (15)

| Variável | Tipo | Descrição | Origem | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `imunidade_ewm` | double | Proxy de imunidade: EWM da incidência, meia-vida 104 semanas (inclui t) | eng |  | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `seno_semana` | double | sin(2π·semanaISO/52) · sazonalidade | eng |  | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `cosseno_semana` | double | cos(2π·semanaISO/52) · sazonalidade | eng |  | ✔ |  |  | ✔ | ✔ | ✔ |
| `casos_suavizados` | double | Casos suavizados por Savitzky-Golay (janela 7, ordem 3, em log1p), deslocado 3 semanas para ser causal | física |  | ✔ | ✔ | ✔ | ✔ |  |  |
| `velocidade` | double | 1ª derivada SG de log1p(casos), causal · "velocidade" da epidemia | física |  | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `aceleracao` | double | 2ª derivada SG de log1p(casos), causal · "aceleração" | física |  | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| `imunidade_ewm_t1` | double | imunidade_ewm da semana anterior (shift(1) por município) | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `pressure_min_t1` | double | pressure_min da semana anterior (shift(1) por município) | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `casos_chikungunya_t1` | double | casos_chikungunya da semana anterior (shift(1) por município) | espacial |  |  |  | ✔ | ✔ | ✔ |  |
| `casos_suavizados_t1` | double | = casos_suavizados em **t** (shift(0), nome enganoso) | espacial |  |  |  | ✔ | ✔ | ✔ |  |
| `aceleracao_t1` | double | = aceleracao em **t** (shift(0), nome enganoso) | espacial |  |  |  | ✔ | ✔ |  |  |
| `velocidade_t1` | double | = velocidade em **t** (shift(0), nome enganoso) | espacial |  |  |  | ✔ | ✔ |  |  |
| `log_inc_dengue_t1` | double | log1p(casos_suavizados_t1 / population_t1 × 1e5) | FINAL_NN |  |  |  |  |  |  | ✔ |
| `log_inc_chiku_t1` | double | log1p(incidência de chikungunya em t−1) | FINAL_NN |  |  |  |  |  |  | ✔ |
| `log_inc_chiku` | double | log1p(incidência de chikungunya em t) | FINAL_NN |  |  |  |  |  |  | ✔ |

### Espaciais W1/W2 (23)

| Variável | Tipo | Descrição | Origem | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|---|---|---|---|---|---|---|---|---|---|---|
| `w1_dengue_t1` | double | Média W1 (vizinhos de fronteira) de casos_suavizados_t1 | espacial |  |  |  | ✔ | ✔ | ✔ |  |
| `w1_chikungunya_t1` | double | Média W1 de casos_chikungunya_t1 | espacial |  |  |  | ✔ | ✔ |  |  |
| `w1_imunidade_t1` | double | Média W1 de imunidade_ewm_t1 | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `w1_velocidade_t1` | double | Média W1 da velocidade | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `w1_aceleracao_t1` | double | Média W1 da aceleração | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `w1_temp_forecast_t1` | double | Média W1 de temp_med_forecast_t1 | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `w1_umid_forecast_t1` | double | Média W1 de umid_med_forecast_t1 | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `w1_precip_forecast_t1` | double | Média W1 de precip_tot_forecast_t1 | espacial |  |  |  | ✔ | ✔ |  |  |
| `w1_pressure_min_t1` | double | Média W1 de pressure_min_t1 | espacial |  |  |  | ✔ | ✔ |  |  |
| `w1_acessos_afya_t1` | double | Média W1 de access_count_t1 | espacial |  |  |  | ✔ | ✔ |  |  |
| `w1_population_t1` | double | Média W1 de population_t1 | espacial |  |  |  | ✔ | ✔ | ✔ |  |
| `w2_dengue_t1` | double | Média W2 (vizinhos de 2ª ordem) de casos_suavizados_t1 | espacial |  |  |  | ✔ | ✔ | ✔ |  |
| `w2_chikungunya_t1` | double | Média W2 de casos_chikungunya_t1 | espacial |  |  |  | ✔ | ✔ |  |  |
| `w2_velocidade_t1` | double | Média W2 da velocidade | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `w2_precip_forecast_t1` | double | Média W2 de precip_tot_forecast_t1 | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `w2_umid_forecast_t1` | double | Média W2 de umid_med_forecast_t1 | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `w2_aceleracao_t1` | double | Média W2 da aceleração | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `w2_temp_forecast_t1` | double | Média W2 de temp_med_forecast_t1 | espacial |  |  |  | ✔ | ✔ | ✔ | ✔ |
| `w2_population_t1` | double | Média W2 de population_t1 | espacial |  |  |  | ✔ | ✔ | ✔ |  |
| `w1_log_pop_t1` | double | log(w1_population_t1) | FINAL_NN |  |  |  |  |  |  | ✔ |
| `w2_log_pop_t1` | double | log(w2_population_t1) | FINAL_NN |  |  |  |  |  |  | ✔ |
| `w1_log_inc_dengue_t1` | double | log1p(w1_dengue_t1 / w1_population_t1 × 1e5) | FINAL_NN |  |  |  |  |  |  | ✔ |
| `w2_log_inc_dengue_t1` | double | log1p(w2_dengue_t1 / w2_population_t1 × 1e5) | FINAL_NN |  |  |  |  |  |  | ✔ |

---

## 6. Arquivos auxiliares

| Arquivo | O que é | Estado |
|---|---|---|
| `./modelos/resultados_agcrn.db` | SQLite com `municipal_results` (340.504 linhas) e `estadual_results` (1.352 linhas). **Uma execução do Modelo 1, S1** (as métricas batem com o docx) | Útil como registro. Tem as linhas duplicadas de 2023-01-01 e 2023-10-01 |
| `./resultados_agcrn.db` | **0 bytes** | Apagar |
| `./modelos/modelo_agcrn_{SECA,PICO}_s1.pt` e `modelos_agcrn_s1.zip` | Pesos do **Modelo 3** (PICO com MHA, versão do docx). SECA tem 21 features e PICO 27 | São os únicos pesos salvos. **Não há pesos do Modelo 2** |
| `variaveis/access_afya_dengue_2021_2026.csv` e `.csv.gz` | Mesmo conteúdo, com e sem compressão | Pode manter só o `.gz` |
| `variaveis/shape_*.gpkg` | Malhas: municípios (usada para W1/W2), regionais e macrorregionais | Manter |

---

## 7. V8 · Season 1 corrigida (`./S1_corrigido/`) — criada em 28/09/2026

Substitui, **só para a S1**, os bancos V5 (`COMPLETO_spatial`) e V7 (`FINAL_NN`). É gerada do zero a partir de `variaveis/` pelo script [`S1_corrigido/refazer_season1.py`](../data_imdc_2026/S1_corrigido/refazer_season1.py), sem passar por V1–V7.

```
variaveis/*.csv + shape_muni.gpkg ──► refazer_season1.py ──► S1_corrigido/train1|test1_dengue_eng_COMPLETO_spatial.parquet (170 col.)
                                                          └─► S1_corrigido/train1|test1_FINAL_NN.parquet (55 col. originais + 2 flags)
```

- **0 duplicatas** (grade 5.492 × 651 treino / × 52 teste), datas contínuas em domingo, soma de casos idêntica ao `dengue.csv`.
- Joins corrigidos: previsão climática (mês de emissão e horizonte certos), ENSO/IOD/PDO (antes 100% zero), Afya (soma semanal, NaN + flag antes de 2021), ilhas sem clima (antes 0 °C), `hti_lag2`.
- Features na série contínua: a imunidade EWM, os lags e a física não reiniciam mais no teste.
- Mesmos nomes de colunas do FINAL_NN antigo: o notebook roda só trocando o caminho.
- Validação: 79 checagens internas + 38 de auditoria independente, todas OK. Detalhes em [`S1_corrigido/LEIA-ME.md`](../data_imdc_2026/S1_corrigido/LEIA-ME.md).

---

## 8. V8 · Seasons 2, 3 e 4 corrigidas + FINAL_NN_v2 (`./S{N}_corrigido/`) — criadas em 29/09/2026

Mesma lógica da seção 7, generalizada em [`refazer_season.py`](../data_imdc_2026/refazer_season.py) (`--season N`). Janelas oficiais (`train_k`/`target_k` do `dengue.csv`): S2 treino até 202325, teste 202341–202440; S3 até 202425, teste 202441–202540; S4 até 202525, teste 202541–202610. Cada uma passou nas 79 checagens. Na S4, a população de 2026 é imputada com a de 2025 (`pop_imputada`).

```
refazer_season.py --season N                ──► S{N}_corrigido/train|test{N}_dengue_eng_COMPLETO_spatial + train|test{N}_FINAL_NN
SHAP_corrigido/shap_walkforward_corrigido.py ──► listas COLS_SECA/COLS_PICO + colunas_extras_nn.json
refazer_season.py --season N --etapa nn_v2  ──► S{N}_corrigido/train|test|ctx{N}_FINAL_NN_v2.parquet
```

- **FINAL_NN_v2** = as 57 colunas do FINAL_NN (idênticas, conferido) + as colunas novas escolhidas pelo SHAP corrigido. A principal é a incidência **observada** (`log_inc_dengue_obs`, `_lag1`, `_lag2` e as médias dos vizinhos W1 e W2).
- **ctx{N}_FINAL_NN_v2** = as semanas EW26–40 entre treino e teste, escalonadas com o scaler do treino. Servem só para aquecer o estado da rede na inferência alinhada.
- Método e resultado do SHAP: [`SHAP_corrigido/LEIA-ME.md`](../data_imdc_2026/SHAP_corrigido/LEIA-ME.md) e `relatorio_shap_corrigido.md`.

---

## 9. Colunas novas do FINAL_NN_v2 e arquivo de contexto (29/09/2026)

O `FINAL_NN_v2` tem **as 57 colunas do FINAL_NN V8** (idênticas, conferido) e mais as colunas que o SHAP corrigido escolheu e que não existiam antes. Todas passam por z-score com a média e o desvio do treino. Se um valor estiver faltando, ele vira 0 (a média).

| Coluna | O que é | Por que entrou |
|---|---|---|
| `log_inc_dengue_obs` | log1p(casos da semana t / população × 100 mil), **sem suavizar** | A mais importante no SHAP (22–48% da importância). É o último dado conhecido, a mesma informação da persistência |
| `log_inc_dengue_obs_lag1` | Idem, semana t−1 | Estável nas 4 safras |
| `log_inc_dengue_obs_lag2` | Idem, semana t−2 | Estável nas 4 safras |
| `w1_log_inc_dengue_obs` | Incidência observada dos vizinhos de fronteira (W1): soma dos casos / soma da população | Contágio espacial |
| `w2_log_inc_dengue_obs` | Idem, vizinhos de 2ª ordem (W2) | Contágio espacial |
| `temp_min_lag2` | Temperatura mínima de 2 semanas antes | Clima (SECA) |
| `w1_pressure_min_t1` | Pressão mínima média dos vizinhos W1, semana anterior | Clima (SECA) |
| `precip_max_lag3` | Precipitação máxima de 3 semanas antes | Clima (PICO) |
| `biome_encoded` | Bioma, codificado pela média de casos no treino | Estático (PICO) |

**Arquivos por safra** em `S{N}_corrigido/`:

| Arquivo | Semanas | Para quê |
|---|---|---|
| `train{N}_FINAL_NN_v2.parquet` | 201001 até a EW25 da safra | Treino. As últimas 52 semanas viram validação dentro do notebook |
| `ctx{N}_FINAL_NN_v2.parquet` | EW26 a EW40 (15 semanas) | Só aquecem o estado da rede antes do teste. A última delas prevê a 1ª semana do teste |
| `test{N}_FINAL_NN_v2.parquet` | EW41 a EW40 do ano seguinte (S4: até 202610) | Teste |
| `scaler_FINAL_NN_v2_season{N}.csv` | – | Média e desvio de cada coluna, para desfazer o z-score |

Como gerar: `python refazer_season.py --season N --etapa nn_v2`, que lê a lista de colunas extras de `SHAP_corrigido/colunas_extras_nn.json`.
