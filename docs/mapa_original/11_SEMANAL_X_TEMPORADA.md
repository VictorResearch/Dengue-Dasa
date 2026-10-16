# 11 · Retreino semanal de todos os modelos + próximos passos da temporada (08/10/2026)

> **Pedido:** retreinar os modelos de temporada na versão semanal (1–4 semanas à frente) para comparar tudo nos dois níveis, com todas as métricas, nas 8 temporadas. Na mesma leva:
> - o teste do Felipe (parte B do CompFelipe);
> - os próximos passos do [10 §6](10_COMPARACAO_FELIPE_SMAPE.md).
>
> **Código:** `data_imdc_2026/SEMANAL/` (`semanal_lib.py`, `_gerar_notebooks_semanal.py`, 3 notebooks e o LEIA-ME com o passo a passo de ativação).
> **Status:** a rodar no Colab. Só foi feito um teste de fumaça no PC (1 rodada, configurações mínimas, 3 min, sem erros): a retomada funciona e a persistência na rodada do Felipe repete o MAE 507 e o WAPE 16,6% do 10 §5.2.

## Resultados (11/10/2026; Semanal3_Avaliacao)

Os modelos de CPU e de GPU (redes de grafo, NHITS, TFT, Chronos-2 e MTGNN v2) foram todos avaliados. Os rankings são os mesmos na afinação e no teste.

**Semanal, teste 2022–25, h = 1** (R = WIS relativo à climatologia; Persist. = WIS relativo à persistência):

| Modelo | R | × Persist. | MAE | WAPE | sMAPE | IC90 |
|---|---|---|---|---|---|---|
| **ENS_GBM_ARIMA_sem** (GBM_xgb_todas + ARIMA) | **0,224** | 0,884 | 285 | 13,0% | 20,2% | **91,2** |
| ENS_sem_top3 / top5 | 0,226 | 0,894 | 326–329 | 14,9–15,0% | 20,7–20,8% | 90 |
| GBM_xgb_todas_sem | 0,229 | 0,903 | 311 | 14,2% | 20,6% | 91,4 |
| GBM_lgb (todas / sel / casos) + λ | 0,230–0,232 | 0,91 | 330–354 | 15–16% | 21% | 87–90 |
| **ARIMA_Fourier_sem / AutoARIMA_d0_sem** | 0,232–0,233 | 0,92 | **252** | **11,5%** | 20,7% | 89,6 |
| MTGNN_cresc_sem (+λ) | 0,235–0,238 | 0,93–0,94 | 281 | 12,8% | 21,3% | 89–92 |
| VAR_macro_sem | 0,245 | 0,97 | 271 | 12,4% | 21,9% | 85 |
| Chronos2_sem | 0,253 | 1,00 | 358 | 16,3% | 21,2% | 88 |
| Persistencia_UF | 0,253 | 1 | 354 | 16,2% | 22,8% | 88 |
| NHITS / TFT sem covariáveis | 0,26–0,27 | 1,04–1,06 | 328–338 | 15,0–15,4% | 22,2–22,4% | 85–91 |
| TFT_todas_sem | 0,44 | 1,75 | 766 | 35,0% | 34,8% | 71 |
| Felipe_XGB_q (+λ) | 0,597 (0,347) | 2,4 (1,37) | 946 | 43,2% | 22,0% | 94 (85) |
| Felipe_XGB / LGBM / CAT (ponto) | 0,51 / 0,55 / 0,69 | 2,0–2,7 | 799–890 | 36–41% | 22,9–32,4% | 0 |
| MTGNN / A3TGCN / GCNGRU / STGCN (âncora) | 0,54–0,63 | 2,1–2,5 | 816–1.009 | 37–46% | 53–57% | 81–89 |
| MTGNN_H68_sem (o de temporada, usado na semana) | 0,687 | 2,7 | 1.410 | 64,4% | 66,2% | 87 |

- **h = 4:** ENS_sem_top5 0,407; GBM_xgb_todas 0,410; ENS_GBM_ARIMA 0,411 (MAE 664, WAPE 30%, IC90 90); Chronos2 0,440; ARIMA 0,442; MTGNN_cresc 0,466; persistência 0,578 (MAE 1.186). Os pontuais do Felipe ficam acima de 1 (piores que a climatologia), com MAE de 1.246–1.367.
- **Jogo do Felipe (2023–24, h = 1):**
  - ENS_GBM_ARIMA: MAE 420 e WAPE 13,7%;
  - AutoARIMA: MAE 372 e WAPE 12,1%;
  - NHITS sem covariáveis: MAE 372;
  - persistência: MAE 507 e WAPE 16,6%;
  - os modelos do Felipe: MAE de 1.055–1.145 e WAPE de 34,5–37,4% (o melhor dele é o LGBM).
- **Temporada (teste, × BB): novos líderes.**
  - **ENS_grafos2_λ: 0,835**;
  - **MTGNN_pesopop_UF+λ: 0,839**, o melhor modelo isolado; bate o MTGNN_UF+λ (0,846);
  - ENS_grafos_λ: 0,848;
  - ENS_porte: igual ao ENS_UF_top5 (0,853). A escolha por porte ficou com o ENS_UF_top5 em todos os grupos;
  - MTGNN_cresc_UF+λ: 0,916 e MTGNN_misto_UF+λ: 0,945. O alvo em crescimento **piora** na temporada;
  - ENS_grafos_TFT_NHITS: R 0,920, mas tem o menor WIS em casos dos nossos (968). Nenhum dos nossos alcança a pucrio (699).
- **Viés:** na temporada, as medianas de quase todos os modelos ficam 70–80% abaixo do total observado em 2022–25 (2024 foi recorde). No semanal h = 1, o viés é de −4 a −13%.

**Lições:**
1. A 1 semana, o melhor é o simples e bem calibrado: ARIMA + GBM em crescimento.
2. As redes de grafo só funcionam no semanal com o alvo em crescimento (0,54 → 0,235). O modelo de temporada usado na semana seguinte (H68) é ruim (0,69).
3. Covariáveis quase não ajudam no semanal. O GBM_xgb_todas é o melhor GBM, mas por pouco. No TFT e no NHITS, as covariáveis atrapalham.
4. O Felipe fica 3× pior em MAE nos anos epidêmicos (árvores em contagens não extrapolam). Nos anos calmos (afinação), ele fica perto da persistência.
5. Na temporada, pesar a perda pela população ajuda (MTGNN_pesopop); o crescimento não ajuda.

## Desenho

- **Rodadas semanais:**
  - as 8 temporadas (afinação 2018–21, teste 2022–25): alvos SE41→SE40 e treino uma vez com alvos até 4 semanas antes da SE41;
  - `felipe_2023-24`: treino até 31/12/2022.
  - Em todas, a previsão sai de cada origem com os dados observados, para h = 1…4, sem reajustar o modelo.
- **Variáveis:**
  - os modelos que aguentam usam **todas** (GBM_*_todas, MTGNN_todas, NHITS/TFT_todas);
  - os que precisam de curadoria passam antes por uma **seleção para trás por grupo**, só na afinação (`selecao_grupos.json`), e depois pelo treino (GBM_lgb_sel, MTGNN_sel, NHITS/TFT_sel);
  - ARIMA, VAR, as referências e os modelos do Felipe usam só casos ou as variáveis dele.
- **Extrapolação temporada → semana:** `MTGNN_H68_sem` é a rede de temporada (saída de 68 semanas) usada para a semana seguinte, sem especializar.
- **Retomada:**
  - semanal: 1 arquivo por modelo × rodada, gravado como `.tmp` e depois renomeado;
  - seleção: o estado vai para um json a cada avaliação;
  - temporada: `L.rodar`.
  - Se cair, basta rodar tudo de novo.
- **Régua comum:** WIS relativo à climatologia (por semana do ano) nos dois níveis.
  - Na temporada, também o WIS relativo ao BB.
  - No semanal, também o WIS relativo à persistência probabilística.
  - Mais MAE, RMSE, WAPE, sMAPE (as duas escalas e o mediano), r, viés e IC50/80/90/95.
  - A tabela única (`tabela_unica_teste.csv`) põe a temporada e o semanal h = 1…4 nas mesmas semanas-alvo de 2022–25.

## Próximos passos incluídos

| Nível | Passo | Onde |
|---|---|---|
| Temporada | ensemble só de grafos com λ (`ENS_grafos_λ`) e com o melhor MTGNN v2 (`ENS_grafos2_λ`) | notebook 3, B2 |
| Temporada | estados grandes: `ENS_porte` (o melhor candidato por porte da UF, escolhido na afinação, entre grafos, TFT_pool, NHITS_pool e combinações) e `MTGNN_pesopop_UF` (pinball × população^0,5) | notebook 3, B2; notebook 2, C |
| Temporada | MTGNN v2 com alvo em crescimento (`MTGNN_cresc_UF`) e misto (`MTGNN_misto_UF`) | notebook 2, C → `resultados_uf_prox` |
| Semanal | ENS_GBM_ARIMA_sem (o melhor GBM na afinação + ARIMA) | notebook 3, A2 |
| Semanal | grafos a 1–4 semanas e `ENS_grafos_GBM_ARIMA_sem` | notebook 2, A; notebook 3, A2 |
| Semanal | quantis de curto prazo com WIS (todos os modelos semanais dão 9 quantis; λ escolhido na afinação) | notebooks 1–3 |

A pasta `resultados_uf_prox` também entra no CompUF4 e no Comp5, que leem `resultados_uf*`.
