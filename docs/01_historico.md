# 01 · Histórico completo dos testes

De junho a outubro de 2026, em ordem. Cada fase traz o que foi feito, as métricas, as lições e onde está o código. Os documentos de trabalho originais (com ainda mais detalhe) estão em [mapa_original/](mapa_original/).

**Como ler os números:**
- **R × BB** = WIS do modelo ÷ WIS do modelo-base oficial (BB), média geométrica por UF e temporada. Abaixo de 1 = melhor que o BB.
- **"olha o teste"** = o número foi visto antes de uma decisão. Está marcado onde aconteceu.
- Glossário completo em [04_avaliacao_e_metricas.md](04_avaliacao_e_metricas.md).

---

## Resumo da evolução

| Data | Régua | Melhor resultado | Modelo |
|---|---|---|---|
| 19/06 | MAE estadual, 1 semana (S1) | 476 (com vazamento) | AGCRN bimodal (Modelo 2) |
| 29/09 | MAE estadual, 1 semana (S1) | 243 honesto (persistência 185) | Modelo 2 v2 + pós-processamento v3 |
| 30/09 | temporada, R × BB | 0,938 → 0,865 | V4 → ARIMA_Fourier |
| 01/10 | temporada, R × BB | 0,846 | MTGNN_UF+λ |
| 11/10 | temporada, R × BB | 0,835 | ENS_grafos2_λ |
| 12/10 | temporada, R × BB | 0,831 | ENS_grafos3_λ |
| 13/10 | temporada, R × BB | 0,828 | AJ_continuo_UF |
| 13/10 | temporada, R × BB | **0,808** | ENS_pico7_UF |
| 13/10 | altura do pico (erro mediano) | −42% → **+2%** | ENS_pico7_simples_UF |
| 12/10 | semanal h = 1, R × climatologia | **0,224** (MAE 267–285) | ENS_GBM_ARIMA(_MTGNNcresc)_sem / ENS_stack_sem |

---

## Fase 1 · AGCRN semanal municipal (mar → 19/06) · [notebooks/01_agcrn_municipal](../notebooks/01_agcrn_municipal/)

**Ideia:** prever a **próxima semana** em 5.492 municípios (sem o ES) com uma rede de grafos recorrente **AGCRN** e saída **Binomial Negativa**, somar por UF e reconciliar com **MinT**.

| Data | Ciclo | O que foi feito | Resultado |
|---|---|---|---|
| mar–jun | 0 · dados | 13 arquivos brutos (dengue, chikungunya, clima ERA5 e SEAS5, oceano, população, Köppen/bioma, regionais, Afya, malhas) | — |
| 11/06 | 1 · protótipo v0 | AGCRN + "física" por Processo Gaussiano Esparso + SHAP do XGBoost sem walk-forward | sem resultado; lento e com vazamento |
| 16/06 | 2 · v1 "ambiciosa" | Savitzky-Golay, SHAP walk-forward, imunidade EWM, NB com dispersão variável, ensemble AGCRN + LightGBM + Ridge, Optuna, MinT | não chegou ao notebook final |
| 16–17/06 | 3 · bancos V1 → V3 | junção em R, engenharia (lags climáticos, física), SHAP walk-forward → 9 variáveis estáveis | 12 variáveis |
| 17/06 | 4 · Modelo 1 | AGCRN (K = 2, 64 canais), NLL da NB, MinT | MAE estadual 830, r 0,67 |
| 17–19/06 | 5 · bancos V4 → V7 | vizinhos W1/W2, SHAP separado por regime (seca/pico), StandardScaler | duplicatas reapareceram no V5a |
| 19/06 | 6 · **Modelo 2 bimodal** | dois especialistas AGCRN: PICO (SE47→25, perda que pune subestimar) e SECA (SE26→46, perda que pune falso alarme) | **MAE 476, r 0,93** |
| 19/06 | 7 · Modelo 3 | atenção espacial (4 cabeças) no PICO, treino sem validação | MAE 979, r 0,39 (colapsou) |

| Versão (Season 1, 2022-23) | MAE est. | sMAPE | r est. | MAE mun. | r mun. |
|---|---|---|---|---|---|
| M1 | 830 | 68,2% | 0,67 | 5,23 | 0,47 |
| **M2** | **476** | 63,7% | **0,93** | **3,89** | 0,56 |
| M3 | 979 | 95,3% | 0,39 | 4,79 | 0,25 |
| persistência | 197 | 26,5% | 0,98 | 1,75 | 0,97 |

**Lições:** separar pico e seca foi o maior ganho. Atenção pesada sem validação colapsou. Nada batia a persistência. (Depois se descobriu que esses números tinham vazamento no MinT e dados com duplicatas.)

## Fase 2 · Bancos refeitos e Modelo 2 v2–v3 (28–29/09)

A auditoria achou **duplicatas** nas junções do R, **clima previsto desalinhado**, **ENSO zerado**, **Afya constante** antes de 2021 e **ilhas com 0 °C**. Tudo foi refeito do zero a partir dos brutos ([scripts/01_refazer_municipal.py](../scripts/01_refazer_municipal.py), 79 checagens por safra).

| Ciclo | O que foi feito | Resultado (S1) |
|---|---|---|
| 9 · bancos V8 | grade município × semana sem duplicatas | M2 original: MAE 476 → **749** (o modelo antigo se apoiava em artefatos) |
| 10 · SHAP refeito | alvo = semana seguinte, Poisson com offset, sondas de ruído | a variável mais importante (incidência da semana) não entrava; Afya e chikungunya abaixo do ruído |
| 11 · M2 v2 | variáveis novas, previsão alinhada (t−1 prevê t), early stopping | MAE 560 com MinT; **829 honesto** (o MinT usava o teste) |
| 11b · recalibração | fator de viés por regime medido na validação | MAE 293 |
| 12 · pós v3 | fator por UF + inflação da variância, tudo na validação | **MAE 243**, r 0,98 |
| 12b · M2 v3/v3.1 | perdas equilibradas (3× a mais custa o mesmo que 3× a menos) | MAE 282; municipal MAE **2,93**, r 0,84 |

**Lições:** parte dos ganhos de junho era vazamento ou artefato. Com tudo honesto, 243 contra 185 da persistência. Viés regional: superestima Norte/Nordeste, subestima o pico do Sul.

## Fase 3 · A virada para a temporada (29–30/09) · [notebooks/02_v4_temporada](../notebooks/02_v4_temporada/)

O desafio pede a **temporada inteira (SE41 → SE40) com dados até a SE25**, avaliada por **WIS × BB**. Um experimento mostrou que a habilidade de 1 semana não se transfere:

| Mesmo modelo, usado como | WIS × climatologia |
|---|---|
| 1 semana à frente com casos observados | 0,19 |
| desenrolado da SE25 (recursivo) | 1,02 |
| direto multi-horizonte | 0,98 |
| ensemble dos três | 0,94 |

**V4:** avaliador WIS idêntico ao oficial (razão 1,000), modelos-base, versão A (AGCRN direta de 68 semanas) e versão B (tamanho × forma).

| Modelo | R × BB | UFs > BB |
|---|---|---|
| ENS_bases_A_mediana | **0,938** | 18 |
| ENS_bases_AB_mediana | 0,944 | 19 |
| AR longo | 0,980 | 14 |
| climatologia | 1,012 | 16 |
| A por UF / B por UF | 1,121 / 1,238 | 11 / 6 |

**Lições:** a mediana vence média e pesos; a versão A piora com mais treino; taxa de ataque (+6%) e SEAS5 (+1,5%) atrapalham; conformal com temporadas passadas piora; o maior erro de todos é o surto do Sul em 2022/23.

## Fase 4 · Competição de 19 modelos por região de saúde (30/09) · [notebooks/03_competicao](../notebooks/03_competicao/)

| Modelo | R × BB | UFs > BB |
|---|---|---|
| **ARIMA_Fourier** (sem covariáveis) | **0,865** | 19 |
| EMAP SARIMAX (1º do 3º IMDC) | 0,870 | 17 |
| climatologia | 0,993 | 16 |
| LightGBM_direto · SURGE · canal endêmico | 1,03–1,05 | 10–15 |
| Renovação_S (1º no Sul) | 1,167 | 11 |
| GHR_INLA · XGBoost · Tamanho_x_forma | 1,23–1,36 | 3–6 |
| **AGCRN** | 1,378 | 9 |
| Chronos · BB_INLA (nossa) · Prophet · ETS | 1,43–1,83 | 1–7 |

**Lições:** a 16–67 semanas, vale o nível atual + a sazonalidade; nenhum modelo com covariáveis bateu o ARIMA sem covariáveis; a AGCRN foi abandonada para a temporada. Ressalva: a correção do ARIMA (sem drift) foi feita depois de vê-lo falhar em 2022.

## Fase 5 · Modelos direto por UF (30/09 → 01/10) · [notebooks/04_modelos_uf](../notebooks/04_modelos_uf/)

83 modelos: auto.arima (R e Python), ARIMAX, 6 VARs (macrorregião, espacial, BVAR Minnesota), redes de grafo (MTGNN, A3T-GCN, STGCN, GCNGRU, AGCRN, STID), GBMs, neuralforecast, Chronos-2, GHR_INLA. Escolhas em 2018–2021; teste olhado uma vez.

| Modelo | Afinação (× clim.) | Teste (× BB) |
|---|---|---|
| **MTGNN_UF+λ** | 0,822 | **0,846** |
| ENS_UF_top5 (pré-registrado) | 0,798 | 0,853 |
| A3TGCN_UF+λ (mais estável) | 0,848 | 0,855 |
| ARIMA_Fourier (região→UF) | 0,922 | 0,865 |
| GCNGRU+λ · GBM_lgb_sel | 0,835 · 0,884 | 0,881 · 0,883 |
| VAR_macro · AutoARIMA_d0 | ~0,9 | 0,895 · 0,899 |
| STGCN · DoisEstagios · XGBoost | 0,82–0,85 | 0,93–1,02 |
| AGCRN_UF+λ | 0,855 | 1,030 |
| Chronos-2 (sem / com cov.) | 0,93 / 1,04 | 1,137 / 1,568 |

**Receita das redes vencedoras:** saída direta de 68 semanas, alvo = log-incidência − climatologia (âncora + resíduo), perda pinball nos 9 quantis, grafo das UFs (o MTGNN aprende um grafo direcionado), janela de 104 semanas, 3 sementes, **λ = 0,8** (intervalos 20% mais estreitos).

**Compra de testes (Comp5, problema do jornaleiro):** o campeão do WIS não é o da compra. `DoisEstagios_pool_UF` teve custo 0,871 × climatologia; o ENS_UF_top5, 0,968.

## Fase 6 · Comparação externa e retreino semanal (08–11/10) · [notebooks/05_comparacao_felipe](../notebooks/05_comparacao_felipe/) · [notebooks/06_semanal](../notebooks/06_semanal/)

**"SMAPE de 16%" externo:** os dados batem com os nossos (correlação 0,9994). O 16% é de **1 semana à frente** (só repetir a semana anterior dá WAPE 16,8%). Na regra do desafio, o mesmo XGBoost tem sMAPE de 96%, contra 72% do BB.

**Todos os modelos a 1–4 semanas, nas 8 temporadas:**

| Semanal, teste, h = 1 | R × clim. | MAE | WAPE | IC90 |
|---|---|---|---|---|
| **ENS_GBM_ARIMA_sem** | **0,224** | 285 | 13,0% | 91,2 |
| GBM_xgb_todas_sem | 0,229 | 311 | 14,2% | 91,4 |
| ARIMA_Fourier_sem | 0,232 | **252** | **11,5%** | 89,6 |
| MTGNN_cresc_sem | 0,235 | 281 | 12,8% | 89–92 |
| persistência | 0,253 | 354 | 16,2% | 88 |
| redes de grafo com âncora | 0,54–0,63 | 816–1.009 | 37–46% | 81–89 |

Na temporada: **ENS_grafos2_λ 0,835** e **MTGNN_pesopop_UF+λ 0,839** (perda pesada pela população).

**Lições:** a 1 semana, ganha o simples e calibrado; redes de grafo só servem no semanal prevendo o crescimento; covariáveis quase não ajudam.

## Fase 7 · Rodada 2: pico, entressafra e otimização (11–12/10) · [notebooks/07_rodada2_pico_entressafra](../notebooks/07_rodada2_pico_entressafra/)

Testados: métricas por fase, fase/Rt, tsfresh (seleção FRESH), peso no pico, quantis altos, cabeça do total, pré-treino nas 435 regiões, calibração por horizonte e fase, Optuna, 10 sementes, Chronos-2 com LoRA, stacking por horizonte × fase.

| Resultado (teste) | Número |
|---|---|
| semanal h = 1 | 0,224 (sem ganho) |
| semanal h = 2–4 · **ENS_stack_sem** | h = 4: 0,404; MAE 648 → 583 |
| temporada · **ENS_grafos3_λ** | **0,831 × BB** |
| MTGNN_pesopop: Optuna / 10 sementes | 0,876 / 0,890 (o resto piora o R mas melhora o pico) |
| total da temporada · MTGNN_total_pp+cal | −1% (os melhores gerais erram −37%) |

**Diagnóstico de 13/10:** em 2023/24 a temporada previu 2–10% do real no Sul/Sudeste (Brasil: 0,74 contra 6,70 milhões). Havia sinal em junho (casos de abril a junho relativos ao normal: ρ 0,46). Daí a proposta: **normal + detector de anos grandes + modelo de pico**.

## Fase 8 · Rodada 3: detector + misturas (13/10) · [notebooks/08_rodada3_detector](../notebooks/08_rodada3_detector/)

| Modelo | R × BB | WIS | WIS pico | Viés entressafra | Altura | Total |
|---|---|---|---|---|---|---|
| ENS_grafos3_λ (normal) | 0,831 | 1.343 | 4.065 | +49% | −42% | −37% |
| **AJ_continuo_UF** | **0,828** | 1.285 | 3.883 | +74% | −28% | −25% |
| **MIX_escala_UF** | 1,190 | 776 | 2.136 | +156% | **−4%** | +5% |
| BB | 1 | 1.224 | 3.651 | +108% | −36% | −24% |

**Lições:** detector + reescala acerta o pico, mas reescalar a curva inteira infla a entressafra. O detector tem correlação 0,8, mas encolhe os extremos e acerta só metade dos alarmes.

## Fase 9 · Rodada 4: literatura aplicada (13/10) · [notebooks/09_rodada4_literatura](../notebooks/09_rodada4_literatura/)

Detector ampliado (alerta precoce, imunidade com esquecimento, seca SPEI, SEAS5, temperatura do mar, **sorotipo do SINAN**), **MIX_elastico** (cada semana cresce pela sua elasticidade: 0,09 sem casos → 0,99 no pico), DoisEstagios com detector, **MTGNN condicionado** com aumento sintético e ensemble por regime.

| Modelo | R × BB | WIS | Altura | Total |
|---|---|---|---|---|
| MTGNN_aug_UF | 0,886 | 1.516 | −51% | −37% |
| MIX_elastico_UF | 0,918 | 1.491 | −43% | −42% |
| ENS_regime_UF | 1,026 | 946 | −30% | −19% |
| MTGNN_cond_UF | 1,292 | 1.430 | −52% | −42% |

**O que deu errado:** o detector usou o Atlântico tropical norte (variável nacional) que bateu recorde em 2023; MG foi previsto ×0,49 (real ×20). O filtro de variáveis nacionais tinha um bug (agrupava sem o corte).

## Fase 10 · Rodada 5: modelo final por fase (13–14/10) · [notebooks/10_rodada5_modelo_por_fase](../notebooks/10_rodada5_modelo_por_fase/)

Detector corrigido (escolhido: R3:ridge, correlação 0,82), membros refeitos, ensembles por fase esperada × regime.

| Modelo | R × BB | WIS | WIS entressafra | MAE | WAPE | Altura | Total |
|---|---|---|---|---|---|---|---|
| AJ_continuo | 0,828 | 1.285 | 26,4 | 1.672 | 76,3% | −28% | −25% |
| **MIX_elastico5_UF** | 0,859 | 1.010 | 27,8 | 1.506 | 68,8% | −12% | −8% |
| ENS_fase_nivMIX_UF | 1,006 | 876 | 37,6 | 1.466 | 66,9% | −19% | −11% |
| MIX_escala5_UF | 1,288 | 746 | 55,1 | 1.279 | 58,4% | +12% | +9% |

O critério misto (½ R + ½ WIS absoluto) foi abandonado: difícil de interpretar.

## Fase 11 · Rodada 6: Optuna nos parâmetros do pico (13–14/10) · [notebooks/11_rodada6_optuna](../notebooks/11_rodada6_optuna/)

Todos os parâmetros do MIX_elastico e do MIX_escala abertos, um detector para cada, objetivo = WIS da fase ÷ R5 com restrição de 5% nas outras fases, validação aninhada em 2015–2021.

| Fora da amostra (÷ R5) | Pico | Entressafra | Intermediária | Teste R × BB |
|---|---|---|---|---|
| MIX_elastico6 | 0,927 | 0,962 | 1,075 | 0,947 |
| MIX_escala6 | 1,137 | 0,859 | 0,802 | 0,850 |

**Lição:** a otimização só troca erro entre as fases. A validação aninhada previu isso antes do teste.

## Fase 12 · Rodada 7: ensemble do pico (13–14/10) · [notebooks/12_rodada7_ensemble_pico](../notebooks/12_rodada7_ensemble_pico/)

`ENS = (1 − s)·BASE + s·[(1 − a)·elástico + a·escala]`, com `s` uma rampa pelo nível previsto da semana. Grade de ~1.950 combinações, validação aninhada.

| Teste 2022–25 | R × BB | WIS | WIS pico | WIS entressafra | MAE | WAPE | sMAPE | Altura | Total |
|---|---|---|---|---|---|---|---|---|---|
| **ENS_pico7_UF** (base AJ, 60% escala, z = 2) | **0,808** | 902 | 2.667 | 27,3 | 1.452 | 66,3% | **65,7** | −20% | −16% |
| **ENS_pico7_simples** (a priori: 50/50, z = 1) | 0,935 | **721** | 2.061 | 29,6 | **1.277** | **58,3%** | 67,5 | **+2%** | **−0,6%** |
| MIX_elastico5 | 0,859 | 1.010 | 2.994 | 27,8 | 1.506 | 68,8% | 70,2 | −12% | −8% |

**Ressalvas:** a validação aninhada não recomendou o ENS_pico7_UF (entressafra 1,09 fora da amostra); a escolha variou entre as dobras. Na maioria das UFs e temporadas, elástico e escala erram para o mesmo lado: o ganho vem da especialização por fase.

---

## Hipóteses e o que os dados disseram

| Hipótese | Onde | Veredito |
|---|---|---|
| Especialistas separados para pico e seca | Modelo 2 | ✅ confirmada (r 0,67 → 0,93) |
| Atenção espacial no especialista de pico | Modelo 3 | ❌ colapsou |
| Afya e chikungunya ajudam | SHAP refeito | ❌ abaixo do ruído |
| MinT melhora a soma município → UF | v2 | ❌ usava o teste |
| Habilidade de 1 semana passa para a temporada | experimento | ❌ 0,19 → 1,02 |
| Direto é melhor que recursivo | experimento | ❌ empatam |
| Clima, SEAS5 e ENSO melhoram a temporada | V4, competição, CompUF | ❌ SEAS5 atrapalha; ENSO −1,7% |
| Taxa de ataque recente indica imunidade | V4 | ❌ atrapalha 6% |
| Conformal com temporadas passadas | V4 | ❌ outro regime |
| Mediana do ensemble vence média e pesos | V4, competição | ✅ |
| AGCRN serve para a temporada | competição, CompUF | ❌ 1,378 e 1,030 |
| Direto por UF é melhor que região → UF | CompUF | ⚠️ quase sempre; o ARIMA por região ganhou no teste |
| Perda pinball vence NLL da NB | CompUF | ✅ |
| Intervalos das redes largos demais | CompUF | ✅ λ = 0,8 ganha 1–4% |
| O espaço (grafo) é o que diferencia | CompUF | ✅ |
| Alvo em crescimento ajuda | semanal | ⚠️ ajuda no semanal, piora a temporada |
| Perda pesada pela população | semanal | ✅ 0,846 → 0,839 |
| Fase/Rt, tsfresh, Optuna, 10 sementes | rodada 2 | ⚠️ ganhos pequenos |
| Chronos-2 com LoRA supera o original | rodada 2 | ❌ |
| Em junho há sinal do tamanho da temporada | doc 15, R3 | ✅ correlação 0,82 (encolhe extremos) |
| Reescalar o normal acerta o pico | R3 | ✅ −4%, mas infla a entressafra |
| Entressafra cresce menos que o pico | R4 | ✅ elasticidade 0,09 → 0,99 |
| Mar e sorotipo melhoram o detector | R4 | ❌ sobreajuste em 2023 |
| MTGNN condicionado aprende anos grandes | R4, R5 | ❌ 1,29–1,44 |
| Pesos por regime ou fase resolvem tudo | R4, R5 | ⚠️ ganham no pico, inflam a entressafra |
| Temporada rolante (EAKF) ajuda em junho | R4 | ❌ descartada (sem dado na SE25) |
| Otimizar parâmetros resolve o pico | R6 | ❌ só troca erro entre fases |
| Elástico e escala se compensam | R7 | ⚠️ ganhou, mas por especialização |

## Bugs e problemas de dados encontrados

| Quando | Problema | Correção |
|---|---|---|
| 28/09 | junções do R geravam linhas duplicadas | bancos refeitos do zero |
| 28/09 | ENSO zerado, Afya constante, ilhas com 0 °C, clima previsto desalinhado | idem + checagens |
| 28/09 | datas da submissão +7 dias | corrigido no V4 |
| 29/09 | previsão de t+1 comparada com o real de t | t−1 prevê t |
| 29/09 | especialista SECA nunca validado | validação na última temporada |
| 29/09 | MinT usava o teste | calibração só na validação |
| 29/09 | perdas assimétricas (PICO 3–5×, SECA 0,65×) | perdas equilibradas |
| 29/09 | mediana da NB inflada a ⅓ do real | mediana sem inflação |
| 30/09 | AR longo explodia; ARIMA/ETS extrapolavam tendência | teto; sem drift |
| 30/09 | INLA travava no Windows e não executava a partir do Drive | rodar de `/content` |
| 30/09 | `val_size` menor que o horizonte no neuralforecast | `val_size = H` |
| 01/10 | duas pastas de resultados e comparação de temporadas diferentes | leitura robusta |
| 11/10 | torchao 0.10 do Colab travava o LoRA | desinstalar |
| 14/10 | filtro de variáveis nacionais agrupava sem o corte | corrigido |
| 14/10 | detector R3:ridge só existe a partir de 2018 | ridge reajustado como substituto |
| 16/10 | a última semana dos dados chega incompleta (~36% dos casos) | a produção usa a última semana completa como origem |
