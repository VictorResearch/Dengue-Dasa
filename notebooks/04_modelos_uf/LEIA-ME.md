# Modelos direto por UF: tabelas e notebooks

## Notebooks (01/10/2026)

> **Rodados no Colab em 01/10.** Resultados e lições: `_MAPA_DO_PROJETO/09_RESULTADOS_UF_E_LICOES.md`. Os resultados estão em duas pastas, `resultados_uf` e `resultados_uf_2`; o CompUF4 e o Comp5 leem as duas. Para modelos de fora, use `EXTERNOS` no CompUF4 ou `L.importar_externo` (ver o 08).

Gerados por `_gerar_notebooks_uf.py`. O código dos modelos está em `uf_lib.py`.

**O que subir para `MyDrive/dengue_V4/`:**
1. `uf_lib.py` e `uf_autoarima.R`;
2. a pasta `UF/dados/` com o nome **`dados_uf/`**;
3. `comp_modelos.py` e `comp_inla.R` **novos** (de `COMPETICAO/`);
4. os 4 notebooks `CompUF*`;
5. o `Comp5_Negocio.ipynb` novo, que passou a ler também os resultados por UF.

| Notebook | O que roda | Ambiente | Tempo estimado |
|---|---|---|---|
| `CompUF1_portados` | os 13 modelos da competição direto por UF (`*_UF`), 2018–2025 | CPU High-RAM | 1–2 h |
| `CompUF2_CPU` | 8 auto.arima (Python e R × ARIMA/ARIMAX × d livre/d = 0), 6 VARs, INLA não linear, 3 dois estágios, 8 GBMs, ablação (10 grupos, só 2018–2021) | CPU High-RAM | 4–6 h |
| `CompUF3_GPU` | AGCRN_UF, Chronos_UF, 6 redes do neuralforecast com pooling + 3 sem exógenas, 2 STID, 4 redes de grafo (GCN-GRU, A3T-GCN, STGCN, MTGNN), 2 Chronos-2 (TimesFM opcional) | GPU L4 | 8–12 h |
| `CompUF4_avaliacao` | escolha em 2018–2021 → teste em 2022–2025; UF × região; ablação; calibração da largura; ensembles; previsão final e submissão | CPU | ~15 min |

- **Ordem:** 1, 2 e 3 podem rodar ao mesmo tempo, em sessões separadas. O 4 vem depois, e o `Comp5_Negocio` depois dele.
- **Resultados:** `MyDrive/dengue_V4/resultados_uf/`.

**O que foi testado no PC.** Só testes de fumaça, com configurações mínimas: 8 árvores, 1 época, 1 temporada. **Não são estimativas.**

| Modelo | Testado? |
|---|---|
| ContextoUF, TabelaUF, Porta (climatologia) | ✅ |
| GBM LightGBM e XGBoost: todas as variáveis, só casos, ablação, aumento de dados, pooling | ✅ |
| DoisEstagios (com e sem pooling) | ✅, depois de corrigir as colunas que só existem nas regiões |
| auto.arima e auto.ARIMAX, Python (statsforecast) e R (forecast), d livre e d = 0 | ✅ (Kmax = 1–2 no teste; no Colab é 6). R 4.5.1 local |
| VAR macro, VAR espacial, BVAR Minnesota (com e sem ENSO) | ✅ (< 1 s cada) |
| STID (com e sem pooling), GCN-GRU, A3T-GCN, STGCN, MTGNN | ✅ (CPU, 1 época) |
| DoisEstagios com seleção, GBM `selecionadas` | ✅ |
| CompUF4 inteiro (afinação, UF × região, ablação, λ, ensembles, teste, gravação) | ✅ ensaio com 6 modelos baratos, 2018–2023; corrigido 1 bug |
| CatBoost | ❌ não está instalado no PC; o notebook instala no Colab |
| INLA (GHR_NL) | ❌ o INLA não roda bem no Windows; usa o mesmo mecanismo do GHR, que já rodou no Colab |
| neuralforecast com pooling, Chronos-2 | ❌ sem GPU nem os pacotes no PC; APIs conferidas na documentação |
| TimesFM | ❌ **experimental**, desligado por padrão: a API mudou na versão 3 |
| TabPFN-TS | não incluído: o modo local exige aceitar a licença no site da Prior Labs, o que precisa ser feito por você |

Se algo falhar no Colab, o erro vai para `resultados_uf/erros_<notebook>.csv` e os outros modelos continuam.

**Exógenas comuns (comparabilidade).** Todo modelo que usa exógenas no futuro usa o mesmo `U.EXOG_FUTURAS` = ENSO persistido da SE25 + anomalias de temperatura e chuva (SEAS5 emitida em junho por até 6 meses, depois 0). Usam esse conjunto: AutoARIMAX, NFPooled, Chronos2_cov e o INLA GHR_NL. Os VARX usam só o ENSO. Cada família tem a versão sem exógenas. Os modelos de árvore usam variáveis da origem: `casos` (nada), `selecionadas` (o conjunto padrão, escolhido em `avaliar_variaveis.py` só com 2015–2021) ou `todas`.

**Avaliação das variáveis:** `avaliar_variaveis.py` → `dados/avaliacao_variaveis.md`. O resumo está em `_MAPA_DO_PROJETO/07_ECONOMETRIA_GRAFOS_E_VARIAVEIS.md`.

---

# Tabelas unificadas por UF

> Criado em 30/09/2026 por `preparar_tabela_uf.py`, a partir dos **brutos** de `data_imdc_2026/variaveis/`. Não usa os bancos municipais já processados, para não herdar agregações erradas.
> As checagens estão em `dados/relatorio_validacao_uf.md`. O plano que usa estas tabelas está em `_MAPA_DO_PROJETO/06_NIVEL_ESTADUAL_PESQUISA_E_PLANO.md` (fase 2).

## Arquivos (`dados/`)

| Arquivo | Uma linha por | Para quê |
|---|---|---|
| `tabela_uf_semanal.parquet` | UF × semana (2010-01-03 → última semana com casos) | Modelos semanais: redes, árvores com origens sintéticas, ARIMAX, INLA |
| `tabela_uf_temporada.parquet` | UF × temporada (2011 → 2025) | Modelos de temporada (dois estágios: tamanho, pico e semana do pico). Variáveis **conhecidas na SE25** + alvos |
| `tabela_uf_estatica.parquet` | UF | Frações de bioma e Köppen, grau de vizinhança, contato de fronteira |
| `seas5_uf.parquet` | UF × mês de emissão × horizonte (1–6) | Previsão sazonal ECMWF. Para a SE25, use só emissões até junho do ano da temporada |
| `vizinhanca_uf.npz` | – | `W1` (fronteira, 0/1), `W1_contato` (nº de pares de municípios vizinhos na fronteira), `W2` (vizinhos de 2ª ordem), `ufs` |

- **26 UFs:** o ES fica fora, como no desafio, e também não entra como vizinho.
- **Chave:** `uf_code` + `date` (domingo da semana epidemiológica).
- **Temporada `s`:** SE41 do ano `s` até a SE40 de `s+1`. O corte para prever a temporada `s` é a SE25 do ano `s`.

## Como cada tipo de variável foi agregado

O princípio: **somar só o que é contagem; recalcular o resto no nível da UF, ou calcular no município antes de agregar quando a variável não é linear.**

| Tipo | Exemplos | Como | Por quê |
|---|---|---|---|
| **Contagens** | `casos`, `casos_chikungunya`, `afya_acessos`, `pop` | **Soma** dos municípios | São extensivas |
| **Taxas** | `incidencia`, `log_inc_dengue`, `log_inc_chiku`, `afya_log_acessos_100k` | **Soma das contagens ÷ soma da população** | A média de taxas dá o mesmo peso ao município de 1 mil e ao de 10 milhões de habitantes |
| **Clima intensivo** | `temp_med`, `rel_humid_med`, `pressure_med`, `precip_med` (mm/dia), `rainy_days`, `thermal_range` | **Média ponderada pela população** de 2015 (pesos fixos). Município sem dado sai da conta e os pesos são renormalizados | Mede o clima a que as pessoas estão expostas. Pesos fixos evitam saltos na virada de ano. Faltante **nunca** vira 0 (uma ilha "a 0 °C" puxaria a média) |
| **Mínimos e máximos semanais** | `temp_min`, `temp_max`, `precip_max`, `rel_humid_min`... | Média ponderada dos mínimos e máximos **municipais** | É "o mínimo típico vivido pela população". O mínimo dos mínimos refletiria só o município mais extremo |
| **Clima não linear** | `vpd`, `hti` (chuva × temperatura), `volatilidade_temp_4w`, `volatilidade_precip_4w` | **Calculado em cada município, depois agregado** | Média de função ≠ função da média (Jensen). O relatório mostra quanto a "VPD da média" erraria |
| **Clima linear no tempo** | `*_lag1..5`, `amplitude_termica_dinamica2/3` | Na série da UF | Com pesos fixos, deslocar e agregar comutam (o relatório confere) |
| **Dispersão dentro da UF** (nova) | `temp_med_dp_mun`, `precip_med_dp_mun` | Desvio-padrão ponderado entre municípios | Heterogeneidade climática que a média esconde |
| **Física dos casos** | `casos_suavizados`, `velocidade`, `aceleracao` | Savitzky-Golay (janela 7, ordem 3) em log1p(casos **da UF**), deslocado 3 semanas | log e derivada não comutam com a soma. O relatório confere que o valor em t só usa dados até t |
| **Imunidade** | `imunidade_ewm` | EWM (meia-vida de 104 semanas) da incidência **da UF** | Mesma definição do municipal, refeita sobre a taxa agregada |
| **Espalhamento** (nova) | `pct_mun_com_casos`, `pct_pop_mun_com_casos` | % dos municípios (e da população) com ≥ 1 caso na semana | Só existe no nível agregado: mede o quanto a epidemia está espalhada |
| **Urbanização** (nova) | `pct_pop_mun_100k` | % da população em municípios com ≥ 100 mil habitantes, por ano | Aproximação de urbanização com o dado que existe |
| **Vizinhos** | `w1_*`, `w1c_*`, `w2_*` | **Recalculados no grafo das UFs.** Incidência = **soma dos casos dos vizinhos ÷ soma da população dos vizinhos**. Clima dos vizinhos = média ponderada pela população de cada vizinho. Física e imunidade calculadas sobre a série **somada** dos vizinhos | A média dos `w1_` municipais misturaria vizinhos dentro da mesma UF. `w1` = fronteira; `w1c` = pesado pelo nº de pares de municípios na fronteira (mais contato, mais peso); `w2` = exatamente 2 passos |
| **Previsão sazonal (semanal)** | `temp_med_forecast`, `umid_med_forecast`, `precip_tot_forecast` | Média ponderada pela população; semana → mês-alvo (quarta-feira); emissão = mês-alvo − 1 (horizonte 2); chuva convertida para mm/dia | Mesma regra do `refazer_season.py`. ⚠️ Na tarefa de temporada, o horizonte de 1 mês **não está disponível na SE25**: use a `seas5_uf` com a emissão certa |
| **Nacionais** | `enso`, `iod`, `pdo` e lags 6–9 | Repetidos em todas as UFs | Não há o que agregar |
| **Calendário** | `seno_semana`, `cosseno_semana`, `epiweek`, `woy`, `season` | Pela data | – |
| **Estáticas** | `pct_pop_bioma_*`, `pct_pop_koppen_*`, `pct_pop_koppen_grupo_A/B/C`, `bioma_predominante`, `koppen_predominante` | **% da população em cada classe** (pesos de 2015) | A moda perde a mistura (MG tem Cerrado e Mata Atlântica). O **target encoding** municipal não foi refeito: com 26 UFs ele vazaria o alvo |
| **Qualidade** | `cobertura_clima_pop`, `pop_extrapolada`, `chik_disponivel`, `afya_disponivel`, `forecast_mes_emissao_anterior` | Flags e frações | Para saber quando um valor é parcial ou imputado |

## Tabela por temporada

- **Variáveis:** as mesmas de `features_temporada` do V4 (taxa de ataque 1–4 temporadas, incidência acumulada, tendência de 8 semanas, anomalias de clima de 12 e 26 semanas, SEAS5, ENSO/IOD/PDO, Richards), todas calculadas **com dados até a SE25 da própria temporada**. As anomalias usam a climatologia até essa SE25.
- **Acrescentadas:**
  - **`vizp_w1_*` e `vizp_w1c_*`**: vizinhos pela razão agregada. O V4 usa `viz_*`, a média simples dos logs dos vizinhos, que trata o AP igual a SP. `viz_*` foi mantida só para comparar;
  - espalhamento médio das últimas 8 semanas;
  - urbanização;
  - frações de bioma e Köppen.
- **Alvos** (só nas temporadas completas):
  - `alvo_casos_temporada` e `alvo_log_inc_temporada`;
  - `alvo_pico_casos`;
  - `alvo_semana_pico` (semanas desde a SE41).

## Cuidados ao usar

1. **Vazamento:**
   - na tabela semanal, cada linha usa só dados até aquela semana;
   - para prever a temporada `s`, filtre `date <= SE25 do ano s`;
   - a previsão sazonal semanal com horizonte de 1 mês **não** vale para horizontes longos.
2. **Afya** só existe a partir de 2021 (NaN antes, com `afya_disponivel = 0`). **Chikungunya**, a partir de 2014 (0 antes, com `chik_disponivel = 0`).
3. **População de 2026–2027** é extrapolada como no V4 (crescimento do último ano, limitado a ±3%), com `pop_extrapolada = 1`.
4. **Regulamento:** os dados vêm do FTP do desafio. Antes de submeter, confira se cada fonte é permitida.
5. **Unidade da chuva observada:** `precip_min/med/max` seguem o bruto. A documentação do desafio diz "mm/h", mas a magnitude não bate (AM teria 26 mm/h em média), então trate como escala relativa e prefira anomalias padronizadas. A agregação é linear e está correta em qualquer unidade. **Não compare com `precip_tot_forecast`**, que foi convertida para mm/dia e bate com a climatologia (AM ≈ 7 mm/dia, CE ≈ 2,5).
6. **Municípios sem clima no bruto:** 2605459 (Fernando de Noronha), 2916104 e 2919926 (ilhas da BA). Eles saem da média e os pesos são renormalizados; a `cobertura_clima_pop` fica ≥ 0,997.

## Resultado da geração (30/09/2026)

- **38 de 38 checagens OK** (`dados/relatorio_validacao_uf.md`), em ~40 s. Entre elas:
  - casos, população e 7 variáveis de clima **iguais ao `painel_uf` do V4**;
  - média ponderada refeita à mão;
  - lag da UF = UF dos lags municipais;
  - vizinho refeito à mão;
  - física causal;
  - alvos da temporada = soma das semanas;
  - corte na SE25.
- **Efeito de Jensen na VPD:** calcular a "VPD da média" em vez da "média da VPD" erraria 1,0% em média.
- **Tamanho:**
  - `tabela_uf_semanal`: 21.970 linhas (26 UFs × 845 semanas) × 214 colunas;
  - `tabela_uf_temporada`: 390 linhas (26 UFs × 2011–2025) × 81 colunas;
  - `seas5_uf`: 30.420 linhas.
- **Vizinhos pela razão agregada × média simples do V4:** diferença de até ~0,6 em log por temporada. Exemplo: SP em 2024, `vizp_w1_ar1_parcial` 8,55 contra `viz_ar1_parcial` 7,91.
