library(dplyr)
library(readr)
library(readxl)
library(sf)
library(stringr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(writexl)
library(sfarrow)
# Sys.setenv(NOT_CRAN = "true")
# install.packages("arrow", type = "binary")
library(arrow)
library(reticulate)
setwd("E:/AmbientePython/Projetos/data_imdc_2026")

# Criar bancos de dados ####
dengue = read.csv("dengue.csv") %>% mutate(year = as.numeric(substr(epiweek, 1, 4)))
clim = read.csv("climate.csv")
envi = read.csv("environ_vars.csv")
regihealth = read.csv("map_regional_health.csv")
pop = read.csv("datasus_population_2001_2025.csv")
ocean = read.csv("ocean_climate_oscillations.csv")
chico = read.csv("chikungunya.csv")
climate = read.csv("forecasting_climate.csv")
climate = climate %>% rename(temp_med_forecast = "temp_med", precip_tot_forecast = "precip_tot", umid_med_forecast = "umid_med")
afya = read.csv("access_afya_dengue_2021_2026.csv")
#mun = st_read(file.path("shape_muni.gpkg"))
df_new = dengue %>%
  mutate(
    # Pega o resto da divisão por 100 (ex: 202415 %% 100 = 15)
    semana_ano = semana_epi %% 100
  )
  left_join(clim, by = c("geocode" = "geocode", "epiweek" = "epiweek", "date" = "date"))%>% 
  left_join(envi, by = c("geocode" = "geocode", "uf_code" = "uf_code"))%>% 
  left_join(regihealth, by = c("geocode" = "geocode", "uf_code" = "uf_code", "uf" = "uf", 
                               "regional_geocode" = "regional_geocode", "macroregional_geocode" = "macroregional_geocode")
            )%>% 
  left_join(pop, by = c("geocode" = "geocode", "year" = "year"))%>%
  left_join(chico, by = c("epiweek" = "epiweek","date" = "date","geocode" = "geocode", "uf_code" = "uf_code", "uf" = "uf", 
                          "regional_geocode" = "regional_geocode", "macroregional_geocode" = "macroregional_geocode")
  )%>% 
  left_join(climate, by = c("geocode"="geocode", "date" = "reference_month")) %>%
  left_join(afya, by = c("geocode"="geocode", "date" = "access_date", "uf" = "uf"))%>%
  #left_join(mun, by = c("geocode" = "geocode", "uf" = "uf", "uf_code" = "uf_code")) %>% 
  left_join(ocean, by = c("date" = "date"))

  install.packages("zoo")
  library(zoo)
df_NEW <- df_new %>%
  # 1. Ordenamento espacial e temporal para garantir janelas corretas
  group_by(geocode) %>%
  arrange(geocode, epiweek) %>%
  
  mutate(
    # =========================================================================
    # 1. Déficit de Pressão de Vapor (VPD)
    # Pressão de vapor de saturação (es) em kPa e VPD resultante
    # =========================================================================
    es_kpa = 0.611 * exp((17.27 * temp_med) / (temp_med + 237.3)),
    vpd = es_kpa * (1 - (rel_humid_med / 100)),
    
    # =========================================================================
    # 2. Amplitude Térmica Dinâmica
    # Média móvel da amplitude térmica nas últimas 3 semanas
    # align = "right" garante que não há data leakage (apenas passado e presente)
    # =========================================================================
    amplitude_termica_dinamica3 = rollmean(thermal_range, k = 3, fill = NA, align = "right"),
    amplitude_termica_dinamica2 = rollmean(thermal_range, k = 2, fill = NA, align = "right"),
    # =========================================================================
    # 3. Desvio Padrão Móvel (Rolling Volatility)
    # Volatilidade da temperatura média nas últimas 4 semanas
    # =========================================================================
    volatilidade_temp_4w = rollapply(temp_med, width = 4, FUN = sd, fill = NA, align = "right"),
    volatilidade_precip_4w = rollapply(precip_med, width = 4, FUN = sd, fill = NA, align = "right"),
    
    # =========================================================================
    # 4. Índice Hidrotérmico (HTI)
    # Interação entre chuva e temperatura defasadas em 3 semanas (ciclo vetor)
    # =========================================================================
    hti_lag3 = lag(precip_med, 3) * lag(temp_med, 3),
    hti_lag2 = lag(precip_med, 3) * lag(temp_med, 3)
  ) %>%
  ungroup() %>%

  # Limpeza de variáveis intermediárias
  select(
    -es_kpa
  )


View(df_new)
names(chico)
names(df)
train1 = df_NEW %>% filter(epiweek >= 201001 & epiweek <= 202225)
train2 = df_NEW %>% filter(epiweek >= 201001 & epiweek <= 202325)
train3 = df_NEW %>% filter(epiweek >= 201001 & epiweek <= 202425)
train4 = df_NEW %>% filter(epiweek >= 201001 & epiweek <= 202525)
test1 = df_NEW %>% filter(epiweek >= 202241 & epiweek <= 202340)
test2 = df_NEW %>% filter(epiweek >= 202341 & epiweek <= 202440)
test3 = df_NEW %>% filter(epiweek >= 202441 & epiweek <= 202540)
test4 = df_NEW %>% filter(epiweek >= 202541 & epiweek <= 202640)

testprior = read_parquet("test1_dengue_eng.parquet")
test = read_parquet("bancos/test1_dengue_eng_shap_final_spatial_v2.parquet")
cat(names(test))
cat(names(testprior))

View(read_parquet("test1_dengue_eng.parquet"))
write_parquet(train1, "train1_dengue.parquet")
write_parquet(train2, "train2_dengue.parquet")
write_parquet(train3, "train3_dengue.parquet")
write_parquet(train4, "train4_dengue.parquet")
write_parquet(test1, "test1_dengue.parquet")
write_parquet(test2, "test2_dengue.parquet")
write_parquet(test3, "test3_dengue.parquet")
write_parquet(test4, "test4_dengue.parquet")

names(dengue)
names(clim)
names(envi)
names(regihealth)
names(pop)
View(df_new)


# Instale os pacotes caso não tenha: install.packages(c("arrow", "dplyr"))
library(arrow)
library(dplyr)

# Definição das Chaves de Cruzamento (Join Keys) fixas
join_keys <- c("epiweek", "date", "geocode", "uf_code", "uf", 
               "regional_geocode", "macroregional_geocode")

# Loop pelas 4 safras (folds) de dados
for (i in 1:4) {
  
  cat(sprintf("\n🔄 Iniciando processamento da Safra %d...\n", i))
  
  # 1. Definição dos nomes dos arquivos dinamicamente para a iteração atual
  # Nota: Caso os seus arquivos antigos de treino não tenham o número no nome 
  # (ex: apenas "train_dengue_eng.parquet"), ajuste a variável file_train_old abaixo.
  file_train_new <- sprintf("bancos/train%d_dengue_eng_shap_final_spatial_v2.parquet", i)
  file_test_new  <- sprintf("bancos/test%d_dengue_eng_shap_final_spatial_v2.parquet", i)
  
  file_train_old <- sprintf("train%d_dengue_eng.parquet", i)
  file_test_old  <- sprintf("test%d_dengue_eng.parquet", i)
  
  # 2. Leitura dos Data Frames
  train_new <- read_parquet(file_train_new)
  test_new  <- read_parquet(file_test_new)
  train_old <- read_parquet(file_train_old)
  test_old  <- read_parquet(file_test_old)
  
  # 3. Remoção do Espírito Santo (uf_code == 32)
  train_new <- train_new %>% filter(uf_code != 32)
  test_new  <- test_new %>% filter(uf_code != 32)
  train_old <- train_old %>% filter(uf_code != 32)
  test_old  <- test_old %>% filter(uf_code != 32)
  
  # 4. Lógica de Prevenção de Duplicatas
  # Verifica colunas sobrepostas dinamicamente para o Treino
  colunas_novas_tr <- names(train_new)
  colunas_para_trazer_old_tr <- setdiff(names(train_old), colunas_novas_tr)
  
  # Verifica colunas sobrepostas dinamicamente para o Teste
  colunas_novas_te <- names(test_new)
  colunas_para_trazer_old_te <- setdiff(names(test_old), colunas_novas_te)
  
  # 🔥 A CURA: Filtra os dados antigos e remove qualquer duplicação de chave primária antes do join
  train_old_clean <- train_old %>% 
    select(all_of(c(join_keys, colunas_para_trazer_old_tr))) %>%
    distinct(geocode, date, .keep_all = TRUE)
  
  test_old_clean  <- test_old %>% 
    select(all_of(c(join_keys, colunas_para_trazer_old_te))) %>%
    distinct(geocode, date, .keep_all = TRUE)
  
  # 5. Cruzamento dos Dados (Merge)
  # Agora o join será um casamento perfeito de 1 para 1
  train_merged <- train_new %>% left_join(train_old_clean, by = join_keys)
  test_merged  <- test_new %>% left_join(test_old_clean, by = join_keys)

  # 6. Nomenclatura e Salvamento dos Novos Arquivos Unificados
  out_train <- sprintf("train%d_dengue_eng_COMPLETO_spatial.parquet", i)
  out_test  <- sprintf("test%d_dengue_eng_COMPLETO_spatial.parquet", i)
  
  write_parquet(train_merged, out_train)
  write_parquet(test_merged, out_test)
  
  # Relatório do ciclo no console
  cat(sprintf("✅ Safra %d concluída e salva no disco!\n", i))
  cat("   -> Treino Merged:", nrow(train_merged), "linhas |", ncol(train_merged), "colunas\n")
  cat("   -> Teste Merged :", nrow(test_merged), "linhas |", ncol(test_merged), "colunas\n")
}

cat("\n🚀 Todos os 8 arquivos (4 treinos e 4 testes) foram processados com sucesso!\n")