# 05 · Os modelos campeões e o top 5 de cada nível

Escolhidos pelas **métricas absolutas** (erro em casos) no teste 2022–2025. Código: [producao/prod_lib.py](../producao/prod_lib.py). Parâmetros: [producao/parametros/](../producao/parametros/).

## Semanal (1–4 semanas à frente)

### Top 5 (MAE em casos/semana, teste 2022–2025)

| # | Modelo | h = 1 | h = 2 | h = 3 | h = 4 | Média |
|---|---|---|---|---|---|---|
| 1 | **ENS_stack_sem** | 270 | **383** | **475** | **583** | **428** |
| 2 | ARIMA_Fourier_sem | **252** | 398 | 525 | 646 | 455 |
| 3 | GBM_xgb_optuna_sem | 308 | 403 | 508 | 639 | 464 |
| 4 | ENS_GBM_ARIMA_sem | 271 | 414 | 526 | 648 | 465 |
| 5 | ENS_GBM_ARIMA_MTGNNcresc_sem | 267 | 418 | 544 | 680 | 477 |
| — | persistência (referência) | 354 | 637 | 914 | 1.186 | 773 |

### Campeão: `ENS_stack_sem`

| h | MAE | WAPE | sMAPE | WIS × clim. | WIS × persist. | IC90 | Viés | r |
|---|---|---|---|---|---|---|---|---|
| 1 | 270 | 12,3% | 20,4 | 0,224 | 0,884 | 90,1% | −6,5% | 0,992 |
| 2 | 383 | 17,5% | 25,3 | 0,293 | 0,794 | 89,1% | −8,9% | 0,981 |
| 3 | 475 | 21,7% | 29,6 | 0,355 | 0,746 | 88,8% | −8,7% | 0,967 |
| 4 | 583 | 26,6% | 33,5 | 0,404 | 0,699 | 88,5% | −12,5% | 0,954 |

**Arquitetura:** média ponderada dos 9 quantis de 9 membros. Os pesos dependem do **horizonte** (1–4) e da **fase da UF na semana de origem** (entressafra < p25, intermediária, pico ≥ p75): 12 conjuntos de pesos, escolhidos na afinação 2018–2021 por busca coordenada no WIS em log ([parametros/pesos_stack_semanal.json](../producao/parametros/pesos_stack_semanal.json)).

| Membro | O que é | Parâmetros |
|---|---|---|
| `GBM_xgb_todas_sem` | XGBoost quantílico com todas as variáveis; alvo = crescimento log da semana-alvo sobre a origem | 300 árvores |
| `GBM_xgb_fase_sem` | + variáveis de fase (Rt, variação do Rt, semanas desde a entressafra, nível ÷ p75) | idem |
| `GBM_xgb_fase_L1_sem` | + mediana de um modelo próprio (perda L1), quantis recentrados nela | idem |
| `GBM_xgb_tsf_sem` | + 40 variáveis do tsfresh (seleção FRESH só até 2021) | [tsfresh_selecao.json](../producao/parametros/tsfresh_selecao.json) |
| `GBM_xgb_optuna_sem` | XGBoost com fase e hiperparâmetros do Optuna (40 tentativas na afinação) | [optuna_gbm_semanal_melhor.json](../producao/parametros/optuna_gbm_semanal_melhor.json): 750 árvores, profundidade 4, lr 0,053 |
| `ARIMA_Fourier_sem` | SARIMAX por UF em log1p da incidência + Fourier K = 3, 8 anos de treino, sem drift | — |
| `MTGNN_cresc_sem` | rede de grafo MTGNN (grafo aprendido + fronteiras), alvo = crescimento | 3 sementes |
| `MTGNN_cresc_escala_sem` | idem, com a série de cada UF padronizada | idem |
| `Chronos2FT_sem` | Chronos-2 ajustado com LoRA (1.000 passos) às 26 séries até o corte | GPU |

Pesos mais altos: na entressafra, `GBM_xgb_optuna_sem` (0,81 em h = 1); no pico em h = 1, ARIMA (0,31), XGBoost Optuna (0,31) e XGBoost todas (0,21); em h = 3–4, Chronos-2 (0,41–0,51).

## Temporada (SE41 → SE40, prevista na SE25)

### Top 5 (teste 2022–2025)

| # | Modelo | WIS | MAE | WAPE | sMAPE | WIS × BB | Altura do pico | Total | Pico no IC90 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **ENS_pico7_simples_UF** | **721** | **1.277** | **58,3%** | 67,5 | 0,935 | **+2,2%** | **−0,6%** | 85,6% |
| 2 | MIX_escala5_UF | 746 | 1.279 | 58,4% | 70,4 | 1,288 | +11,5% | +8,6% | 93,3% |
| 3 | ENS_pico7_UF | 902 | 1.452 | 66,3% | **65,7** | **0,808** | −20,0% | −15,6% | 76,0% |
| 4 | MIX_elastico5_UF | 1.010 | 1.506 | 68,8% | 70,2 | 0,859 | −11,8% | −7,8% | 74,0% |
| 5 | AJ_continuo_UF | 1.285 | 1.672 | 76,3% | 70,0 | 0,828 | −28,2% | −24,6% | 64,4% |
| — | BB (modelo-base oficial) | 1.224 | 1.787 | 81,6% | 73,0 | 1 | −36,3% | −23,5% | 73,1% |

**Escolha entre os dois ensembles do pico:** o `ENS_pico7_simples_UF` (fixado a priori, sem olhar dados) tem as melhores métricas absolutas e acerta a altura do pico e o total. O `ENS_pico7_UF` (escolhido por grade) é o melhor na régua relativa oficial e na entressafra, mas fica 20% abaixo no pico. A validação aninhada não recomendou o `ENS_pico7_UF` (ver [01_historico.md](01_historico.md), fase 12).

### Campeão: `ENS_pico7_simples_UF` (cadeia)

| Etapa | O que faz | Parâmetros |
|---|---|---|
| 1 · Normal `ENS_grafos3_λ` | mediana dos quantis de 4 redes de grafo treinadas para a temporada (saída direta de 68 semanas, âncora sazonal + resíduo, perda pinball) | `MTGNN_UF`, `A3TGCN_UF`, `GCNGRU_UF` com λ = 0,8; `MTGNN_optuna10_UF` (d = 64, 30 épocas, lr 0,0043, janela 78, top-2 vizinhos, β 0,135, peso pela população 0,79, 10 sementes) |
| 2 · Detector | ridge do log do tamanho da temporada ÷ normal da UF; 10 sinais por correlação de Spearman só até 2021 (casos relativos a 4/8/13 semanas, espalhamento, macrorregião, temporada anterior parcial, variação do ENSO, vizinhos, temporadas desde o último ano grande, incidência de 8 semanas); 9 quantis pelos resíduos de treino × 1,15 | corte de junho; ≥ 60 linhas de treino |
| 3a · `MIX_elastico5_UF` | 1.000 trajetórias do normal; cada uma vai até o tamanho sorteado do detector com peso w, e cada semana cresce pela sua elasticidade histórica (0,09 nas semanas sem casos → 0,99 no pico) | w = 0,5 |
| 3b · `MIX_escala5_UF` | se o tamanho sorteado passa do limiar, reescala a trajetória inteira | limiar = log 3 (3× o normal) |
| 4 · Rampa | z = nível previsto da semana (mediana do elástico) na escala da UF: 0 = p25, 1 = p75; s = 1 / (1 + exp(−(z − 1)/0,3)) | c = 1, h = 0,3 |
| 5 · Ensemble | ENS = (1 − s)·elástico + s·(½·elástico + ½·escala) | a = 0,5 |

`ENS_pico7_UF` usa a mesma cadeia com base = `AJ_continuo_UF` (normal × (tamanho ÷ normal)^0,25), a = 0,6, c = 2,0, h = 0,6 e o nível vindo do normal.
