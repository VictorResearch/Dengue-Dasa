# SEMANAL · os modelos de temporada também na versão semanal (08/10/2026)

Objetivo: comparar **todos** os modelos nos dois jogos.
- **Temporada:** dados até a SE25, previsão SE41→SE40 (16–68 semanas de antecedência), WIS relativo ao BB.
- **Semanal:** prever 1–4 semanas à frente com os dados até a semana atual. Inclui a rodada do Felipe, 2023–24.

Código em `semanal_lib.py`; os notebooks são gerados por `_gerar_notebooks_semanal.py`.

## Como ativar (passo a passo)

1. **No Drive, em `MyDrive/dengue_V4/`:**
   - suba `semanal_lib.py`;
   - crie à mão as pastas `resultados_semanal` e `resultados_uf_prox`. Assim, duas sessões abertas ao mesmo tempo não criam pastas duplicadas;
   - opcional: crie a pasta `COMPARACAO_FELIPE/` e suba `felipe_no_meu_jogo_previsoes.parquet`, para o Felipe entrar na tabela da temporada.
2. **Abra os três notebooks no Colab** (Arquivo › Fazer upload de notebook):
   - **`Semanal1_CPU`:** Ambiente de execução › Alterar o tipo › CPU, com High-RAM se tiver. Rodar tudo (Ctrl+F9). Leva 2–5 h.
   - **`Semanal2_GPU`:** GPU (L4 ou T4). Rodar tudo. Leva 5–8 h e pode rodar **ao mesmo tempo** que o 1.
   - **`Semanal3_Avaliacao`:** CPU. Rode quando quiser ver o resultado. Avalia o que já existir e pode ser repetido.
3. **Se cair** (internet, energia, limite do Colab): reconecte e **rode tudo de novo**.
   - O que terminou é pulado: cada modelo × rodada vira um arquivo, gravado num `.tmp` e só então renomeado.
   - A seleção de variáveis grava o estado a cada avaliação.
   - Os modelos de temporada (parte C do notebook 2) usam o `L.rodar`, que também pula o que existe.
4. **Ordem recomendada:** comece o 1 e o 2 juntos. Os modelos `_sel` do notebook 2 (MTGNN, NHITS e TFT com as variáveis selecionadas) só entram depois que a etapa 3 do notebook 1 terminar. Se o 2 chegar lá antes, ele pula esses modelos e avisa; rode o 2 de novo no fim.

## Protocolo semanal

- **Rodadas:** as 8 temporadas (afinação 2018–21, teste 2022–25) e `felipe_2023-24` (treino até 31/12/2022).
- **Treino:** uma vez por rodada, só com alvos até 4 semanas antes da SE41.
- **Previsão:** de cada origem, com os dados observados até ela e sem reajustar o modelo (o protocolo do Felipe), para h = 1…4.
- **Escolhas:** λ, membros dos ensembles e seleção de variáveis são escolhidos só na afinação.
- **Régua comum aos dois níveis:** WIS relativo à climatologia (a mesma climatologia por semana do ano). Também saem MAE, RMSE, WAPE, sMAPE (0–200 e 0–100), sMAPE mediano, r, viés, IC50/80/90/95 e o WIS relativo à persistência.

## Modelos

| Notebook | Modelo | Variáveis |
|---|---|---|
| 1 (CPU) | Persistencia_UF, Climatologia_sem | referências |
| 1 | Felipe_XGB / LGBM / CAT (ponto) e Felipe_XGB_q (quantis) | as dele |
| 1 | GBM_lgb_todas, GBM_xgb_todas, GBM_cat_todas | **todas** (todos os grupos do `uf_lib`, inclusive SEAS5) |
| 1 | GBM_lgb_sel (crescimento e âncora) | **selecionadas**: seleção para trás por grupo na afinação → `selecao_grupos.json` |
| 1 | GBM_lgb_casos, ARIMA_Fourier_sem, AutoARIMA_d0_sem, VAR_macro_sem | só casos |
| 2 (GPU) | MTGNN / A3TGCN / GCNGRU / STGCN `_sem` | as entradas das redes de temporada |
| 2 | MTGNN_H68_sem | o modelo de **temporada** usado para a semana seguinte (extrapolação) |
| 2 | MTGNN_cresc_sem; MTGNN_todas_sem; MTGNN_sel_sem | alvo em crescimento; + todas; + selecionadas |
| 2 | NHITS / TFT (`semcov`, `todas`, `sel`), Chronos2_sem | idem |
| 2 · C (temporada) | MTGNN_cresc_UF, MTGNN_misto_UF, MTGNN_pesopop_UF | próximos passos → `resultados_uf_prox` |
| 3 | ENS_GBM_ARIMA_sem, ENS_grafos_sem, ENS_grafos_GBM_ARIMA_sem, ENS_sem_top3/5, `+λ` | ensembles semanais |
| 3 | ENS_grafos_λ, ENS_grafos2_λ, ENS_grafos_TFT_NHITS, ENS_porte | ensembles de temporada |

## Saídas (em `MyDrive/dengue_V4/resultados_semanal_avaliacao/`)

- `semanal_{afinacao,teste,felipe}.csv`;
- `temporada_teste.csv`;
- `tabela_unica_teste.csv`: temporada × semanal h = 1…4, nas mesmas semanas-alvo;
- `top5_por_metrica.csv`.

A pasta `resultados_uf_prox` também entra sozinha no `CompUF4` e no `Comp5`, que leem `resultados_uf*`. Por isso o custo da compra dos modelos novos sai no Comp5.
