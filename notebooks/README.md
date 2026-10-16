# Notebooks da pesquisa, em ordem

Cada pasta é uma etapa. Dentro dela: os notebooks (`.ipynb`), o gerador (`_gerar_*.py`, que é onde se edita o notebook) e o `LEIA-ME.md` da etapa (o que roda, ambiente, tempo, o que precisa estar pronto antes e o que saiu). As bibliotecas ficam em [`../src/`](../src/). Para rodar, monte a área de trabalho ([docs/03_como_rodar.md](../docs/03_como_rodar.md)).

| # | Pasta | Pergunta da etapa | Resultado principal (teste) |
|---|---|---|---|
| 01 | [01_agcrn_municipal](01_agcrn_municipal/) | Uma rede de grafos em 5.492 municípios prevê bem a próxima semana? | MAE estadual 243 (persistência: 185). Inclui o notebook de junho (`historico_junho/`), o SHAP corrigido e as análises da v2 e da v3 |
| 02 | [02_v4_temporada](02_v4_temporada/) | Como prever a temporada inteira a partir da SE25? | 0,938 × BB (mediana das bases + AGCRN direta) |
| 03 | [03_competicao](03_competicao/) | Qual família de modelos funciona melhor por região de saúde? | ARIMA_Fourier 0,865 × BB; AGCRN 1,378. Resultados em `resultados_colab_2026-09-30/` |
| 04 | [04_modelos_uf](04_modelos_uf/) | E direto por UF, com redes de grafo, econometria e INLA? | MTGNN_UF+λ 0,846; ENS_UF_top5 0,853 |
| 05 | [05_comparacao_felipe](05_comparacao_felipe/) | O "SMAPE de 16%" de um modelo externo é melhor? | Não: é 1 semana à frente; na regra do desafio, sMAPE 96% |
| 06 | [06_semanal](06_semanal/) | Como os modelos se saem a 1–4 semanas? | ENS_GBM_ARIMA_sem 0,224 × climatologia; temporada 0,835 |
| 07 | [07_rodada2_pico_entressafra](07_rodada2_pico_entressafra/) | Fase, tsfresh, Optuna, Chronos-2, stacking ajudam no pico e na entressafra? | ENS_stack_sem (h = 4: MAE 583); ENS_grafos3_λ 0,831. Tabelas finais em `resultados_finais/` |
| 08 | [08_rodada3_detector](08_rodada3_detector/) | Um detector de anos grandes em junho corrige o pico? | MIX_escala: altura do pico −4%; AJ_continuo 0,828 |
| 09 | [09_rodada4_literatura](09_rodada4_literatura/) | Sorotipo, mar, seca, elasticidade e redes condicionadas ajudam? | elasticidade por nível (0,09 → 0,99); o detector ampliado sobreajustou 2023 |
| 10 | [10_rodada5_modelo_por_fase](10_rodada5_modelo_por_fase/) | Detector corrigido + especialistas por fase | MIX_elastico5: WIS 1.010, altura −12% |
| 11 | [11_rodada6_optuna](11_rodada6_optuna/) | Otimizar os parâmetros do pico resolve? | Não: só troca erro entre fases |
| 12 | [12_rodada7_ensemble_pico](12_rodada7_ensemble_pico/) | Elástico + escala no pico | **ENS_pico7_UF 0,808 × BB**; **ENS_pico7_simples** altura +2% |

**Dependências entre as etapas** e tempos: [docs/03_como_rodar.md](../docs/03_como_rodar.md). **Histórico com todas as métricas:** [docs/01_historico.md](../docs/01_historico.md).

**Observação sobre a etapa 01:** os notebooks de junho (`historico_junho/`) usam bases antigas que tinham erros (duplicatas, clima desalinhado); estão aqui como registro. Os da v2 e da v3 usam as bases refeitas por `scripts/01_refazer_municipal.py`.
