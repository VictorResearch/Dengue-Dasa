# Gerado por SHAP_corrigido/shap_walkforward_corrigido.py em 2026-09-29 12:36 (seasons [1, 2, 3, 4])
# Critério: Top 25 por season e regime, acima das sondas de ruído na permutação (>= 2 de 3 folds), sem redundância (|r| > 0.95), presente em >= 3 de 4 seasons. Ordem: importância SHAP média.
# Requer train{N}_FINAL_NN_v2.parquet se usar alguma destas colunas novas: ['log_inc_dengue_obs', 'log_inc_dengue_obs_lag1', 'log_inc_dengue_obs_lag2', 'w1_log_inc_dengue_obs', 'w2_log_inc_dengue_obs', 'temp_min_lag2', 'w1_pressure_min_t1', 'precip_max_lag3', 'biome_encoded']

COLS_FISICA = ['velocidade', 'aceleracao']

# 🌵 SECA (20 features)
COLS_SECA = COLS_FISICA + [
    'log_inc_dengue_t1', 'log_inc_dengue_obs', 'log_inc_dengue_obs_lag1', 'imunidade_ewm',
    'log_pop_t1', 'log_inc_dengue_obs_lag2', 'w1_log_inc_dengue_obs', 'w2_log_inc_dengue_obs',
    'w1_log_inc_dengue_t1', 'w2_temp_forecast_t1', 'w2_log_inc_dengue_t1', 'cosseno_semana',
    'w1_log_pop_t1', 'precip_tot_forecast', 'rel_humid_max', 'temp_min_lag2',
    'w1_pressure_min_t1', 'temp_min_lag4',
]

# 🌊 PICO (20 features)
COLS_PICO = COLS_FISICA + [
    'log_inc_dengue_obs', 'log_inc_dengue_t1', 'log_inc_dengue_obs_lag1', 'w1_log_inc_dengue_obs',
    'w2_log_inc_dengue_obs', 'log_inc_dengue_obs_lag2', 'cosseno_semana', 'log_pop_t1',
    'imunidade_ewm', 'temp_med', 'w1_log_pop_t1', 'seno_semana',
    'w1_velocidade_t1', 'w2_velocidade_t1', 'w2_log_inc_dengue_t1', 'precip_max_lag3',
    'biome_encoded', 'koppen_encoded',
]
