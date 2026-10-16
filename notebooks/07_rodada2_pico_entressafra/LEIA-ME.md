# RODADA 2 · melhorar o pico e a entressafra (11/10/2026)

É complementar: **não altera nada** das rodadas anteriores. Os modelos novos gravam em `resultados_semanal_r2/` e `resultados_uf_r2/`. As tabelas novas vão para `avaliacao_r2/`, sempre com data e hora no nome, mais uma cópia `_ultimo`. Código em `r2_lib.py`; os notebooks são gerados por `_gerar_notebooks_r2.py`.

## Como ativar

1. **No Drive (`MyDrive/dengue_V4/`):**
   - suba `r2_lib.py`;
   - crie à mão as pastas `resultados_semanal_r2`, `resultados_uf_r2`, `cache_r2` e `avaliacao_r2`.
   - O `semanal_lib.py` e as pastas das rodadas anteriores já devem estar lá.
2. **Suba os 5 notebooks** e rode nesta ordem:

| Ordem | Notebook | Ambiente | O que faz | Tempo |
|---|---|---|---|---|
| 1 | `R2_1_Avaliacao` | CPU | A, D2, E4 sobre o que já existe; tabelas com carimbo | 15–40 min |
| 2 | `R2_2_Semanal_CPU` | CPU High-RAM | B3, C (tsfresh), D1, mediana L1 nos GBMs semanais | 3–4 h |
| 3 | `R2_3_Redes_GPU` | GPU L4/A100 | B1, B2, D1, D3 nas redes (temporada e semanal) | 4–8 h |
| 4 | `R2_4_Optuna` | GPU | E1 (Optuna no GBM e no MTGNN) e E2 (10 sementes) | 5–8 h |
| 5 | `R2_5_Chronos_GPU` | GPU | E3 (Chronos-2 ajustado) | 2–4 h |
| 6 | `R2_6a_Variaveis_GPU` | GPU | fase/Rt e tsfresh nas redes (semanal e temporada) e no GBM do Optuna. Precisa do R2_2 e do R2_4 prontos | 4–6 h |
| 7 | `R2_6b_Variaveis_CPU` | CPU | ARIMA com regressores de fase (ARIMAX_fase_sem). Pode rodar junto com o 6a | 1–2 h |
| 8 | `R2_1_Avaliacao` de novo | CPU | inclui tudo e grava um conjunto novo de tabelas | 15–40 min |
| 9 | `R2_7_Graficos_UF` | CPU | gráficos das 26 UFs + Brasil (semanal e temporada) com os campeões; PDF e totais por temporada | 10–20 min |

   Os notebooks 2, 3, 4 e 5 podem rodar em sessões simultâneas. Abra uma por vez e espere a célula 1 terminar antes de abrir a próxima.
3. **Se cair:** reconecte e rode tudo de novo. A retomada funciona assim:
   - cada modelo × rodada/temporada vira um arquivo quando termina, e o que já existe é pulado;
   - as redes gravam checkpoint a cada 5 épocas em `cache_r2/ckpt/` e continuam da última época salva;
   - o pré-treino nas regiões é salvo e reaproveitado;
   - o tsfresh grava uma UF por vez;
   - o Optuna salva o estudo (`cache_r2/*.db`) a cada tentativa e completa as que faltam;
   - o Chronos ajustado fica em `cache_r2/chronos/` e não é refeito;
   - a calibração e o λ da avaliação são gravados em json a cada modelo.

## O que cada tópico testa

| Tópico | Onde | Modelos / tabelas |
|---|---|---|
| A · pico e entressafra | R2_1 | `*_por_fase`, `*_pico`, `*_cobertura`, `*_resumo_fases` (WIS, viés e IC90 por fase; erro na semana e na altura do pico; cobertura de cada quantil) |
| B1 · escala por UF | R2_3 | MTGNN_escala_pp_UF, MTGNN_cresc_escala_sem |
| B2 · pré-treino nas 435 regiões | R2_3 | MTGNN_pre_pp_UF, MTGNN_cresc_pre_sem |
| B3 · variáveis de fase (Rt etc.) | R2_2 | GBM_xgb_fase_sem |
| C · tsfresh | R2_2 | GBM_xgb_tsf_sem (as 40 mais relevantes, selecionadas só na afinação) |
| D1 · peso no pico / quantis altos | R2_2, R2_3 | GBM_xgb_pico_sem, MTGNN_qalto_pp_UF, MTGNN_cresc_qalto_sem |
| D2 · calibração por horizonte e fase | R2_1 | modelos `+cal` (λ e deslocamento da mediana por balde, escolhidos na afinação) |
| D3 · cabeça do total | R2_3 | MTGNN_total_pp_UF |
| tudo junto | R2_2, R2_3 | GBM_lgb_r2full_sem, MTGNN_r2full_UF |
| E1 · Optuna | R2_4 | GBM_xgb_optuna_sem, MTGNN_optuna10_UF |
| E2 · 10 sementes | R2_4 | MTGNN_pesopop10_UF, MTGNN_optuna10_UF |
| E3 · Chronos-2 ajustado | R2_5 | Chronos2FT_sem, Chronos2FT_UF |
| E4 · pesos por horizonte e fase | R2_1 | ENS_stack_sem, ENS_stack_UF |
| extras | R2_1 | ENS_GBM_ARIMA_MTGNNcresc_sem (corrige o ensemble semanal com grafos), ENS_grafos3_λ |
