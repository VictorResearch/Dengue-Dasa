# SHAP walk-forward corrigido — seleção de features do Modelo 2

Gerado por `shap_walkforward_corrigido.py` em 2026-09-29 12:36 · seasons [1, 2, 3, 4] · 142 candidatas + 3 sondas de ruído · 100 árvores · tempo 1892s

## Qualidade dos modelos auxiliares (validação de cada fold)

Deviance de Poisson na validação: XGBoost vs. modelo só com offset (taxa média × população).

| Season | Regime | Fold | Validação (EW alvo) | Deviance modelo | Só offset | Ganho |
|---|---|---|---|---|---|---|
| S1 | seca | 1 | 201326–201546 | 0.806 | 3.713 | 78.3% |
| S1 | seca | 2 | 201626–201846 | 0.370 | 1.827 | 79.7% |
| S1 | seca | 3 | 201926–202146 | 0.610 | 3.488 | 82.5% |
| S1 | pico | 1 | 201308–201612 | 1.903 | 27.598 | 93.1% |
| S1 | pico | 2 | 201613–201918 | 0.967 | 15.176 | 93.6% |
| S1 | pico | 3 | 201919–202223 | 1.531 | 22.110 | 93.1% |
| S2 | seca | 1 | 201331–201635 | 0.744 | 3.658 | 79.7% |
| S2 | seca | 2 | 201636–201940 | 0.456 | 2.539 | 82.1% |
| S2 | seca | 3 | 201941–202245 | 0.632 | 3.584 | 82.4% |
| S2 | pico | 1 | 201316–201649 | 1.726 | 23.905 | 92.8% |
| S2 | pico | 2 | 201650–202011 | 1.023 | 18.120 | 94.4% |
| S2 | pico | 3 | 202012–202324 | 1.606 | 22.896 | 93.0% |
| S3 | seca | 1 | 201336–201645 | 0.647 | 3.150 | 79.5% |
| S3 | seca | 2 | 201646–202034 | 0.514 | 2.917 | 82.4% |
| S3 | seca | 3 | 202035–202344 | 0.638 | 3.513 | 81.8% |
| S3 | pico | 1 | 201324–201713 | 1.619 | 21.530 | 92.5% |
| S3 | pico | 2 | 201714–202103 | 1.100 | 17.766 | 93.8% |
| S3 | pico | 3 | 202104–202425 | 2.835 | 68.858 | 95.9% |
| S4 | seca | 1 | 201341–201734 | 0.642 | 3.110 | 79.4% |
| S4 | seca | 2 | 201735–202128 | 0.526 | 2.963 | 82.2% |
| S4 | seca | 3 | 202129–202443 | 0.767 | 4.962 | 84.5% |
| S4 | pico | 1 | 201352–201748 | 1.548 | 21.379 | 92.8% |
| S4 | pico | 2 | 201749–202124 | 1.150 | 18.556 | 93.8% |
| S4 | pico | 3 | 202125–202522 | 2.937 | 69.340 | 95.8% |

## Regime SECA — lista recomendada (20 features)

| # | Feature | Família | Seasons no Top 25 | SHAP médio (% do total) | Permutação (Δ deviance rel.) | Na lista atual? | Coluna nova? |
|---|---|---|---|---|---|---|---|
| 1 | `velocidade` | dengue (histórico) | 1/4 | 0.3% | -0.01% | sim |  |
| 2 | `aceleracao` | dengue (histórico) | 4/4 | 1.7% | +0.77% | sim |  |
| 3 | `log_inc_dengue_t1` | dengue (histórico) | 4/4 | 27.9% | +64.26% | sim |  |
| 4 | `log_inc_dengue_obs` | dengue (histórico) | 4/4 | 21.7% | +157.21% | **não** | **sim** |
| 5 | `log_inc_dengue_obs_lag1` | dengue (histórico) | 4/4 | 9.4% | +13.67% | **não** | **sim** |
| 6 | `imunidade_ewm` | imunidade | 4/4 | 5.3% | +8.49% | sim |  |
| 7 | `log_pop_t1` | população | 4/4 | 3.0% | +2.32% | sim |  |
| 8 | `log_inc_dengue_obs_lag2` | dengue (histórico) | 4/4 | 2.9% | +2.80% | **não** | **sim** |
| 9 | `w1_log_inc_dengue_obs` | dengue (histórico) | 4/4 | 2.8% | +1.02% | **não** | **sim** |
| 10 | `w2_log_inc_dengue_obs` | dengue (histórico) | 4/4 | 1.5% | +0.45% | **não** | **sim** |
| 11 | `w1_log_inc_dengue_t1` | dengue (histórico) | 4/4 | 1.3% | +0.26% | sim |  |
| 12 | `w2_temp_forecast_t1` | clima previsto | 4/4 | 1.2% | +0.72% | **não** |  |
| 13 | `w2_log_inc_dengue_t1` | dengue (histórico) | 4/4 | 1.2% | +0.28% | sim |  |
| 14 | `cosseno_semana` | sazonalidade | 4/4 | 0.5% | +1.00% | sim |  |
| 15 | `w1_log_pop_t1` | população | 3/4 | 0.5% | +0.12% | sim |  |
| 16 | `precip_tot_forecast` | clima previsto | 4/4 | 0.5% | +0.21% | **não** |  |
| 17 | `rel_humid_max` | clima observado | 4/4 | 0.4% | +0.49% | sim |  |
| 18 | `temp_min_lag2` | clima observado | 3/4 | 0.2% | +0.61% | **não** | **sim** |
| 19 | `w1_pressure_min_t1` | clima observado | 3/4 | 0.2% | +0.30% | **não** | **sim** |
| 20 | `temp_min_lag4` | clima observado | 3/4 | 0.1% | +0.71% | **não** |  |

**Saem da lista atual (11):**

| Feature | Seasons no Top | SHAP médio | Permutação | Por quê |
|---|---|---|---|---|
| `access_count` | 0/4 | 0.0% | +0.00% | embaralhá-la não piora a validação mais do que embaralhar ruído |
| `seno_semana` | 2/4 | 0.2% | +0.13% | fora do Top 25 na maioria das seasons |
| `temp_min` | 2/4 | 0.8% | +0.81% | redundante com `temp_med` (|r| > 0.95) |
| `w2_velocidade_t1` | 0/4 | 0.2% | -0.00% | embaralhá-la não piora a validação mais do que embaralhar ruído |
| `w1_imunidade_t1` | 2/4 | 0.3% | +0.28% | fora do Top 25 na maioria das seasons |
| `imunidade_ewm_t1` | 0/4 | 1.4% | +1.23% | redundante com `imunidade_ewm` (|r| > 0.95) |
| `log_inc_chiku` | 0/4 | 0.2% | -0.01% | embaralhá-la não piora a validação mais do que embaralhar ruído |
| `log_inc_chiku_t1` | 0/4 | 0.0% | +0.01% | embaralhá-la não piora a validação mais do que embaralhar ruído |
| `w1_velocidade_t1` | 1/4 | 0.3% | +0.04% | embaralhá-la não piora a validação mais do que embaralhar ruído |
| `temp_med` | 2/4 | 0.8% | +0.44% | redundante com `temp_min` (|r| > 0.95) |
| `koppen_encoded` | 1/4 | 0.2% | +0.02% | fora do Top 25 na maioria das seasons |

## Regime PICO — lista recomendada (20 features)

| # | Feature | Família | Seasons no Top 25 | SHAP médio (% do total) | Permutação (Δ deviance rel.) | Na lista atual? | Coluna nova? |
|---|---|---|---|---|---|---|---|
| 1 | `velocidade` | dengue (histórico) | 4/4 | 0.8% | +2.83% | sim |  |
| 2 | `aceleracao` | dengue (histórico) | 4/4 | 0.6% | +0.32% | sim |  |
| 3 | `log_inc_dengue_obs` | dengue (histórico) | 4/4 | 47.7% | +980.71% | **não** | **sim** |
| 4 | `log_inc_dengue_t1` | dengue (histórico) | 4/4 | 12.7% | +23.37% | sim |  |
| 5 | `log_inc_dengue_obs_lag1` | dengue (histórico) | 4/4 | 9.0% | +23.26% | **não** | **sim** |
| 6 | `w1_log_inc_dengue_obs` | dengue (histórico) | 4/4 | 4.2% | +4.20% | **não** | **sim** |
| 7 | `w2_log_inc_dengue_obs` | dengue (histórico) | 4/4 | 3.4% | +2.18% | **não** | **sim** |
| 8 | `log_inc_dengue_obs_lag2` | dengue (histórico) | 4/4 | 2.4% | +3.26% | **não** | **sim** |
| 9 | `cosseno_semana` | sazonalidade | 4/4 | 2.0% | +6.92% | sim |  |
| 10 | `log_pop_t1` | população | 4/4 | 1.9% | +1.87% | sim |  |
| 11 | `imunidade_ewm` | imunidade | 4/4 | 1.6% | +10.10% | sim |  |
| 12 | `temp_med` | clima observado | 4/4 | 1.0% | +2.38% | **não** |  |
| 13 | `w1_log_pop_t1` | população | 4/4 | 0.6% | +0.25% | sim |  |
| 14 | `seno_semana` | sazonalidade | 4/4 | 0.5% | +0.71% | sim |  |
| 15 | `w1_velocidade_t1` | dengue (histórico) | 4/4 | 0.4% | +0.62% | sim |  |
| 16 | `w2_velocidade_t1` | dengue (histórico) | 4/4 | 0.4% | +0.34% | sim |  |
| 17 | `w2_log_inc_dengue_t1` | dengue (histórico) | 3/4 | 0.4% | +0.06% | sim |  |
| 18 | `precip_max_lag3` | clima observado | 4/4 | 0.1% | +0.08% | **não** | **sim** |
| 19 | `biome_encoded` | estático | 4/4 | 0.1% | +0.30% | **não** | **sim** |
| 20 | `koppen_encoded` | estático | 3/4 | 0.1% | +0.26% | sim |  |

**Saem da lista atual (15):**

| Feature | Seasons no Top | SHAP médio | Permutação | Por quê |
|---|---|---|---|---|
| `access_count` | 0/4 | 0.0% | +0.00% | embaralhá-la não piora a validação mais do que embaralhar ruído |
| `w2_aceleracao_t1` | 0/4 | 0.1% | +0.16% | fora do Top 25 na maioria das seasons |
| `access_count_t1` | 0/4 | 0.0% | +0.00% | embaralhá-la não piora a validação mais do que embaralhar ruído |
| `w1_log_inc_dengue_t1` | 0/4 | 0.2% | -0.22% | embaralhá-la não piora a validação mais do que embaralhar ruído |
| `imunidade_ewm_t1` | 0/4 | 1.1% | +1.27% | redundante com `imunidade_ewm` (|r| > 0.95) |
| `log_inc_chiku` | 0/4 | 0.0% | +0.00% | embaralhá-la não piora a validação mais do que embaralhar ruído |
| `w1_aceleracao_t1` | 0/4 | 0.1% | +0.05% | fora do Top 25 na maioria das seasons |
| `w2_temp_forecast_t1` | 2/4 | 0.2% | +0.24% | fora do Top 25 na maioria das seasons |
| `w1_temp_forecast_t1` | 0/4 | 0.1% | +0.09% | redundante com `w2_temp_forecast_t1` (|r| > 0.95) |
| `log_inc_chiku_t1` | 0/4 | 0.0% | -0.01% | embaralhá-la não piora a validação mais do que embaralhar ruído |
| `umid_med_forecast` | 0/4 | 0.1% | +0.12% | fora do Top 25 na maioria das seasons |
| `temp_min` | 0/4 | 0.4% | +0.57% | redundante com `temp_med` (|r| > 0.95) |
| `w2_umid_forecast_t1` | 0/4 | 0.1% | +0.08% | fora do Top 25 na maioria das seasons |
| `temp_med_forecast` | 0/4 | 0.1% | +0.07% | redundante com `w2_temp_forecast_t1` (|r| > 0.95) |
| `w2_precip_forecast_t1` | 1/4 | 0.1% | +0.09% | fora do Top 25 na maioria das seasons |

## Sondas de ruído (limiar)

Colunas aleatórias incluídas nas candidatas. Uma variável só é considerada útil se, embaralhada, piorar a validação mais do que a pior sonda em pelo menos 2 dos 3 folds, e se tiver |SHAP| mediano acima do das sondas. Máximos observados:

| Regime | Sonda | SHAP máx. | Permutação máx. |
|---|---|---|---|
| pico | `sonda_embaralhada` | 0.03% | +0.069% |
| pico | `sonda_normal` | 0.03% | +0.126% |
| pico | `sonda_uniforme` | 0.04% | +0.101% |
| seca | `sonda_embaralhada` | 0.15% | +0.351% |
| seca | `sonda_normal` | 0.22% | +0.273% |
| seca | `sonda_uniforme` | 0.22% | +0.419% |

## Importância por família (SHAP somado, média das seasons e folds)

| Família | SECA | PICO |
|---|---|---|
| dengue (histórico) | 70.7% | 82.2% |
| clima observado | 9.5% | 5.4% |
| oceano | 2.6% | 2.9% |
| imunidade | 7.4% | 2.9% |
| população | 3.6% | 2.7% |
| sazonalidade | 0.8% | 2.5% |
| clima previsto | 4.5% | 1.1% |
| estático | 0.3% | 0.3% |
| chikungunya | 0.3% | 0.1% |
| sonda (ruído) | 0.2% | 0.0% |
| Afya | 0.1% | 0.0% |

## Ranking completo por season (Top 25 selecionadas)

- **S1 pico**: `log_inc_dengue_obs`, `log_inc_dengue_t1`, `log_inc_dengue_obs_lag1`, `w1_log_inc_dengue_obs`, `w2_log_inc_dengue_obs`, `log_inc_dengue_obs_lag2`, `cosseno_semana`, `log_pop_t1`, `imunidade_ewm`, `temp_med`, `velocidade`, `aceleracao`, `w1_log_pop_t1`, `seno_semana`, `w1_velocidade_t1`, `w2_log_inc_dengue_t1`, `w2_velocidade_t1`, `pdo`, `w2_temp_forecast_t1`, `thermal_range`, `w1_pressure_min_t1`, `w1_imunidade_t1`, `precip_max_lag3`, `biome_encoded`, `pdo_lag6`
- **S1 seca**: `log_inc_dengue_t1`, `log_inc_dengue_obs`, `log_inc_dengue_obs_lag1`, `imunidade_ewm`, `log_pop_t1`, `w1_log_inc_dengue_obs`, `log_inc_dengue_obs_lag2`, `aceleracao`, `w1_log_inc_dengue_t1`, `w2_log_inc_dengue_obs`, `w2_temp_forecast_t1`, `w2_log_inc_dengue_t1`, `temp_min`, `cosseno_semana`, `precip_tot_forecast`, `w1_log_pop_t1`, `temp_med_lag3`, `rel_humid_max`, `w1_velocidade_t1`, `w1_imunidade_t1`, `volatilidade_precip_4w`, `temp_min_lag2`, `seno_semana`, `pressure_med_lag3`, `temp_min_lag4`
- **S2 pico**: `log_inc_dengue_obs`, `log_inc_dengue_t1`, `log_inc_dengue_obs_lag1`, `w1_log_inc_dengue_obs`, `w2_log_inc_dengue_obs`, `log_inc_dengue_obs_lag2`, `cosseno_semana`, `log_pop_t1`, `imunidade_ewm`, `temp_med`, `velocidade`, `aceleracao`, `w1_log_pop_t1`, `seno_semana`, `w2_velocidade_t1`, `w2_log_inc_dengue_t1`, `w1_velocidade_t1`, `w2_temp_forecast_t1`, `enso_lag7`, `w1_imunidade_t1`, `koppen_encoded`, `w2_log_pop_t1`, `thermal_range`, `precip_max_lag3`, `biome_encoded`
- **S2 seca**: `log_inc_dengue_t1`, `log_inc_dengue_obs`, `log_inc_dengue_obs_lag1`, `imunidade_ewm`, `log_pop_t1`, `log_inc_dengue_obs_lag2`, `w1_log_inc_dengue_obs`, `aceleracao`, `w2_log_inc_dengue_obs`, `w1_log_inc_dengue_t1`, `w2_log_inc_dengue_t1`, `w2_temp_forecast_t1`, `temp_min`, `temp_med_lag5`, `cosseno_semana`, `w1_log_pop_t1`, `precip_tot_forecast`, `rel_humid_max`, `temp_min_lag2`, `seno_semana`, `w1_pressure_min_t1`, `koppen_encoded`, `precip_min_lag5`, `temp_max_lag3`, `temp_max_lag2`
- **S3 pico**: `log_inc_dengue_obs`, `log_inc_dengue_t1`, `log_inc_dengue_obs_lag1`, `w1_log_inc_dengue_obs`, `w2_log_inc_dengue_obs`, `log_inc_dengue_obs_lag2`, `log_pop_t1`, `cosseno_semana`, `imunidade_ewm`, `temp_med`, `velocidade`, `aceleracao`, `w1_log_pop_t1`, `seno_semana`, `w1_velocidade_t1`, `w2_velocidade_t1`, `w2_log_inc_dengue_t1`, `enso_lag7`, `enso`, `koppen_encoded`, `precip_max_lag3`, `iod_lag8`, `biome_encoded`, `w1_pressure_min_t1`, `w2_precip_forecast_t1`
- **S3 seca**: `log_inc_dengue_t1`, `log_inc_dengue_obs`, `log_inc_dengue_obs_lag1`, `imunidade_ewm`, `w1_log_inc_dengue_obs`, `log_pop_t1`, `log_inc_dengue_obs_lag2`, `aceleracao`, `w2_log_inc_dengue_obs`, `w1_log_inc_dengue_t1`, `w2_temp_forecast_t1`, `w2_log_inc_dengue_t1`, `temp_med`, `temp_med_lag5`, `precip_tot_forecast`, `cosseno_semana`, `w1_log_pop_t1`, `velocidade`, `rel_humid_max`, `temp_min_lag2`, `temp_med_lag3`, `temp_min_lag4`, `volatilidade_temp_4w`, `w1_pressure_min_t1`
- **S4 pico**: `log_inc_dengue_obs`, `log_inc_dengue_t1`, `log_inc_dengue_obs_lag1`, `w1_log_inc_dengue_obs`, `w2_log_inc_dengue_obs`, `log_inc_dengue_obs_lag2`, `cosseno_semana`, `log_pop_t1`, `imunidade_ewm`, `temp_med`, `velocidade`, `w1_log_pop_t1`, `aceleracao`, `seno_semana`, `w1_velocidade_t1`, `enso`, `w2_velocidade_t1`, `iod_lag8`, `precip_max_lag3`, `iod_lag6`, `biome_encoded`, `w2_log_pop_t1`, `koppen_encoded`, `pdo_lag7`, `precip_max_lag2`
- **S4 seca**: `log_inc_dengue_t1`, `log_inc_dengue_obs`, `log_inc_dengue_obs_lag1`, `imunidade_ewm`, `log_pop_t1`, `log_inc_dengue_obs_lag2`, `w1_log_inc_dengue_obs`, `aceleracao`, `w2_log_inc_dengue_obs`, `w2_temp_forecast_t1`, `w2_log_inc_dengue_t1`, `w1_log_inc_dengue_t1`, `temp_med`, `cosseno_semana`, `precip_tot_forecast`, `rel_humid_max`, `iod_lag6`, `w1_imunidade_t1`, `volatilidade_precip_4w`, `w1_pressure_min_t1`, `temp_max_lag4`, `precip_max_lag2`, `temp_max_lag5`, `temp_med_lag2`, `temp_min_lag4`

Arquivos: `importancia_detalhada_folds.csv` (SHAP e permutação por fold), `importancia_por_season.csv`, `estabilidade_entre_seasons.csv`, `listas_features_modelo2.py`, `colunas_extras_nn.json`.