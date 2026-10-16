# Produção

Os 5 melhores modelos de cada nível (métricas absolutas, teste 2022–2025), prontos para uso operacional.

| Arquivo | O que é |
|---|---|
| `prod_lib.py` | código de produção: rodada semanal com origem na última semana completa, os 9 membros do campeão semanal, os 3 ensembles semanais, o normal de temporada (4 redes de grafo), o detector, as misturas e os ensembles do pico |
| `Producao_Semanal.ipynb` | previsão de 1–4 semanas (GPU recomendada; ~1–2 h) |
| `Producao_Temporada.ipynb` | previsão da temporada SE41 → SE40 com dados até a SE25 (GPU; ~1–3 h na 1ª vez) |
| `Relatorio_Campeoes.ipynb` | gráficos de Brasil, 5 regiões e 26 UFs e o documento HTML dos campeões (CPU; ~10 min) |
| `relatorio_campeoes.py` | gerador do documento (métricas, diagramas de arquitetura, gráficos embutidos) |
| `_gerar_notebooks_producao.py` | gera os três notebooks (edite aqui) |
| `parametros/` | o que a pesquisa escolheu na afinação: pesos do stacking semanal, hiperparâmetros do Optuna (GBM e MTGNN), seleção do tsfresh, λ |

| Nível | Top 5 (1º = campeão) |
|---|---|
| Semanal | ENS_stack_sem · ARIMA_Fourier_sem · GBM_xgb_optuna_sem · ENS_GBM_ARIMA_sem · ENS_GBM_ARIMA_MTGNNcresc_sem |
| Temporada | ENS_pico7_simples_UF · MIX_escala5_UF · ENS_pico7_UF · MIX_elastico5_UF · AJ_continuo_UF |

Como rodar, saídas, a semana incompleta e quando retreinar: [docs/06_producao.md](../docs/06_producao.md). Arquitetura e métricas dos campeões: [docs/05_modelos_campeoes.md](../docs/05_modelos_campeoes.md).
