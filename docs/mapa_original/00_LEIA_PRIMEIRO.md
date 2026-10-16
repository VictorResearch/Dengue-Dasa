# 00 · Leia primeiro: estado atual do projeto

> **Última atualização:** 30/09/2026
> Comece por este arquivo. Ele resume onde o projeto está, o que usar e os números atuais.
> Os detalhes estão em [01_LINHA_DO_TEMPO.md](01_LINHA_DO_TEMPO.md) (história), [02_BANCOS_DE_DADOS.md](02_BANCOS_DE_DADOS.md) (bancos e variáveis) e [03_AVALIACAO.md](03_AVALIACAO.md) (análise crítica).
> As métricas também estão em formato de tabela para máquina: [metricas.csv](metricas.csv).
> Revisão da literatura e dos resultados do 3º IMDC, com plano de melhorias: [04_ESTADO_DA_ARTE.md](04_ESTADO_DA_ARTE.md).
> **V4 e competição de modelos (tarefa oficial, temporada inteira):** [05_V4_E_COMPETICAO.md](05_V4_E_COMPETICAO.md). Arquivos de apoio em [anexos/](anexos/).

---

## 1. O projeto em 3 linhas

- **Objetivo:** prever casos semanais de dengue por **estado (UF)**, com intervalos de 50/80/90/95% (desafio 3rd IMDC / Mosqlimate).
- **Como:** um modelo de grafos recorrente (**AGCRN**) prevê cada um dos 5.492 municípios (sem ES) com uma distribuição **Binomial Negativa**, e depois os municípios são somados por UF.
- **Modelo principal:** o **Modelo 2**, com dois especialistas: **PICO** (semanas 47 a 25) e **SECA** (semanas 26 a 46).

---

## 2. Onde estamos agora

> ⚠️ **Desde 29/09 há duas linhas de trabalho.** A **linha oficial** é o **V4** e a **competição de modelos**, que preveem a temporada inteira a partir da SE25, como o desafio exige. A tabela abaixo descreve a linha antiga (Modelo 2, 1 semana à frente), mantida como registro.
>
> **Linha oficial, 01/10 (modelos por UF):** melhor resultado no teste S1–S4 = **ENS_UF_top5, 0,853 × BB** (escolhido só com 2018–21); melhor isolado **MTGNN_UF+λ, 0,846**. Para a compra de testes: **DoisEstagios_pool_UF** (0,871 × climatologia no custo). Ver [09](09_RESULTADOS_UF_E_LICOES.md).
>
> **Linha oficial, 30/09 (competição concluída):** melhor resultado nas S1–S4 = **ARIMA_Fourier por região de saúde, WIS 0,865 × BB**, 19 UFs melhores que o BB; também é o melhor por região (0,878 × climatologia). Fica acima do 1º do 3º IMDC (0,870) e do melhor do V4 (0,938). A AGCRN ficou em 1,378. **Decisão: continuar com o ARIMA_Fourier.** Detalhes e ressalvas em [05 §4](05_V4_E_COMPETICAO.md).

| | |
|---|---|
| **Versão atual do modelo** | **Modelo 2 v3.1**, notebook [`Modelo2_v3.ipynb`](../data_imdc_2026/S1_corrigido/Modelo2_v3.ipynb) (pós-processamento corrigido em 29/09) |
| **Melhor resultado já medido (S1)** | Estado: MAE **243** (pesos da v2 + pós v3). Município: MAE **2,93** e r **0,84** (v3). A v3.1 dá MAE estadual 282, mas é a única sem viés nos dois especialistas |
| **Próximo passo** | Rodar a v3.1 no Colab com `SO_POS_PROCESSAMENTO = True` (confirmar o MAE 282) e atacar a subestimação dos picos no Sul/Sudeste e a superestimação no Norte/Nordeste |
| **Safras com banco pronto** | S1, S2, S3 e S4 (bancos corrigidos) |
| **Safras já avaliadas** | Só a **S1 (2022-23)** |
| **Referência a vencer** | Persistência ingênua ("a semana que vem = esta semana"): MAE estadual **185** |

---

## 3. Qual arquivo usar para quê

Todos os caminhos são relativos a `F:\AmbientePython\Projetos\data_imdc_2026\`.

| Quero... | Use |
|---|---|
| Treinar e avaliar o modelo atual no Colab | `S1_corrigido/Modelo2_v3.ipynb` + os 3 arquivos `train/test/ctx1_FINAL_NN_v2.parquet` em `MyDrive/dengue_S1/` |
| Só refazer a previsão, sem treinar | O mesmo notebook, com `SO_POS_PROCESSAMENTO = True` |
| Rodar outra safra | Mude `SEASON_ALVO` e suba os arquivos de `S{N}_corrigido/` para `MyDrive/dengue_S{N}/` |
| Refazer os bancos a partir dos dados brutos | `python refazer_season.py --season N`. Depois do SHAP, rode também `--etapa nn_v2` |
| Refazer a seleção de variáveis | `python SHAP_corrigido/shap_walkforward_corrigido.py` |
| Ver o diagnóstico da v2 (viés, MinT) | `S1_corrigido/_analise_v2/` (scripts + CSVs) |
| Entender cada banco corrigido | `S1_corrigido/LEIA-ME.md`, `SHAP_corrigido/LEIA-ME.md` |

Notebooks antigos, mantidos só como histórico:
- `Modelo2_S1_enxuto.ipynb`: Modelo 2 original sobre os dados corrigidos.
- `Modelo2_v2.ipynb`: features novas, previsão alinhada e early stopping, mas ainda com o MinT.
- `Dengue dasa/Dengue_final_ufvjm (1).ipynb`: notebook de junho.

---

## 4. Métricas atuais (Season 1, 2022-23)

**Como ler:**
- **MAE** = erro médio em casos (menor é melhor).
- **r** = correlação de Pearson, que mede se a curva sobe e desce na hora certa (maior é melhor).
- **IC50/IC90/IC95** = % dos casos reais que caíram dentro do intervalo. O ideal é ficar perto de 50, 90 e 95.
- **Olha o teste?** "sim" quer dizer que a etapa de conciliação usou os casos reais do teste (vazamento). Esse número é otimista.

⚠️ **Todos os modelos preveem 1 semana à frente** e recebem os casos observados até a semana anterior. O desafio pede a safra inteira a partir da semana 25. Ver [03 §4.1](03_AVALIACAO.md#41-a-avaliação-não-reproduz-o-desafio-one-step-ahead-com-dados-do-futuro).

### Nível estadual

| Versão | O que é | MAE | RMSE | sMAPE | r | IC50 | IC90 | IC95 | Olha o teste? |
|---|---|---|---|---|---|---|---|---|---|
| M1 (jun) | AGCRN simples, dados antigos | 830 | 2.607 | 68,2% | 0,67 | – | – | – | sim (MinT) |
| M2 (jun) | Bimodal, dados antigos | 476 | 1.349 | 63,7% | 0,93 | 59,5 | 95,9 | 98,2 | sim (MinT) |
| M3 (jun) | Pico com atenção, dados antigos | 979 | 3.433 | 95,3% | 0,39 | 33,7 | 74,5 | 82,2 | sim (MinT) |
| M2 enxuto | Modelo 2 original, dados corrigidos | 749 | 1.676 | 81,1% | 0,91 | 49,1 | 95,3 | 97,7 | sim (MinT) |
| M2 v2 | Features do SHAP, previsão alinhada, early stopping | 560 | 1.501 | 62,0% | 0,97 | 62,6 | 98,2 | 99,9 | sim (MinT) |
| M2 v2 sem MinT | Mesmo modelo, soma simples das medianas | 829 | 2.599 | 64,8% | 0,98 | 60,4 | 98,2 | 99,9 | não |
| M2 v2 recalibrado | Fator de viés nacional por regime (validação) | 293 | 777 | 45,5% | 0,975 | 38,8 | 71,8 | 78,6 | não |
| **M2 v2 + pós v3** | **Fator por UF + intervalos calibrados (validação)** | **243** | **643** | **40,9%** | **0,984** | **50,1** | **82,9** | **87,9** | **não** |
| M2 v3 | Losses novas + pós v3 (ponto = mediana da NB inflada) | 318 | 819 | 84,2% | 0,974 | 42,0 | 87,7 | 91,2 | não |
| M2 v3.1 | Mesmos pesos da v3; ponto = mediana sem inflação, intervalos contêm o ponto | 282 | 801 | 42,3% | 0,975 | 54,7 | 87,6 | 91,1 | não |
| Persistência | Repete a semana anterior | 185 | 600 | 24,8% | 0,98 | – | – | – | não |
| Climatologia | Mediana histórica da semana (2015–22) | 773 | 2.683 | 78,3% | 0,77 | – | – | – | não |

### Nível municipal

| Versão | MAE | RMSE | sMAPE | r | IC50 | IC90 | IC95 |
|---|---|---|---|---|---|---|---|
| M1 (jun) | 5,23 | 50,5 | 74,6% | 0,47 | – | – | – |
| M2 (jun) | 3,89 | 51,3 | 39,5% | 0,56 | 85,7 | 96,4 | 97,7 |
| M3 (jun) | 4,79 | 60,4 | 49,1% | 0,25 | 77,7 | – | 87,6 |
| M2 enxuto | 5,49 | 64,6 | 42,0% | 0,56 | 84,8 | 96,3 | 97,7 |
| M2 v2 | 4,42 | 40,1 | 45,0% | 0,82 | 83,9 | 96,0 | 97,6 |
| M2 v2 recalibrado | 3,32 | 41,9 | 37,7% | 0,77 | 85,8 | 95,5 | 97,1 |
| M2 v2 + pós v3 | 3,28 | 41,4 | 37,1% | 0,80 | 85,6 | 95,4 | 97,0 |
| **M2 v3 / v3.1** | **2,93** | **34,0** | **35,7%** | **0,84** | 85,9 | 96,3 | 97,8 |
| Persistência | 1,75 | 14,5 | 37,4% | 0,97 | – | – | – |

No município, o IC50 fica em ~85% porque muitas semanas têm 0 caso, e o 0 sempre cai dentro do intervalo.

---

## 5. O que aprendemos (resumo)

1. **Os bancos antigos tinham erros sérios:** linhas duplicadas, previsão climática desalinhada, ENSO zerado, Afya constante antes de 2021 e ilhas com 0 °C. Tudo foi refeito do zero (bancos "V8").
2. **A informação mais importante é a incidência observada da própria semana.** O SHAP refeito mostra que a família "histórico de dengue" responde por 71–82% da importância. Afya e chikungunya ficaram abaixo do nível de ruído.
3. **O modelo comparava a previsão de t+1 com o real de t.** A previsão agora está alinhada com o treino (a semana t−1 prevê a semana t).
4. **A SECA nunca era validada** e treinava sempre até 200 épocas. A validação agora é a última temporada completa do treino, com early stopping.
5. **As losses antigas eram assimétricas e criaram viés.** O PICO previa 3 a 5 vezes o real no começo da temporada e a SECA previa ~0,65 vez o real.
6. **O MinT usava os casos reais do teste** para calcular seus pesos. Foi trocado por uma calibração feita só na validação.
7. **As losses equilibradas da v3 acabaram com o viés global:** na validação, a razão real/previsto ficou em 1,05 no PICO e 0,93 na SECA (na v2 era 0,48 e 1,06). No teste, a soma prevista do Brasil bate com a real (+2% no PICO, −6% na SECA).
8. **Mediana de uma distribuição muito inflada não serve como previsão.** Com a SECA k = 48, a mediana estadual caía para ~⅓ do real no início e no fim da temporada. A v3.1 usa a mediana sem inflação e deixa a inflação só para os intervalos.
9. **Ainda há viés regional:** o PICO superestima o Norte (+45%, mediana das UFs), o Nordeste (+35%) e o Centro-Oeste (+21%), e subestima o pico do Sul e do Sudeste (~0,8× o pico real; PR 0,6×).
10. **O modelo ainda perde para a persistência** (243–282 × 185). É a próxima coisa a vencer.

---

## 6. Histórico de mudanças (mais recente primeiro)

| Data | O que mudou | Onde | Resultado |
|---|---|---|---|
| 11/10 | **Rodada 2 (pico e entressafra, tópicos A–E)**: 5 notebooks complementares e retomáveis | [13](13_RODADA2_PICO_ENTRESSAFRA.md), `data_imdc_2026/RODADA2/` | Só fumaça no PC (80 s, sem erros; todos os caminhos menos Optuna e Chronos ajustado). Estimativas: a rodar no Colab. Tabelas novas em `avaliacao_r2/`, com carimbo; as antigas ficam |
| 11/10 | **R2_5 corrigido** (o torchao 0.10 do Colab travava o LoRA do peft) e **R2_6a/R2_6b**: variáveis de fase/Rt e tsfresh nas redes (semanal e temporada), no GBM do Optuna e no ARIMA (regressores defasados 4 semanas) | [13 §R2_6](13_RODADA2_PICO_ENTRESSAFRA.md), `RODADA2/` | Fumaça no PC: 84 s, sem erros. Estimativas no Colab |
| 12/10 | **Rodada 2 avaliada (olha o teste)** e relatório para a DASA; notebook R2_7 de gráficos por UF | [13 §Resultados](13_RODADA2_PICO_ENTRESSAFRA.md), [14](14_relatorio_rodada2_dasa.html) (artifact https://claude.ai/artifact/NeFkyaMGQLiWKRSdxt1Ea8) | Semanal h=1 igual (0,224); h=4 ENS_stack_sem 0,404 (MAE 648 → 583). Temporada: ENS_grafos3_λ 0,831 × BB; pico ENS_stack_UF 0,524 (BB 0,693); total MTGNN_total_pp_UF+cal −1% |
| 13/10 | **Diagnóstico do pico de 2024 na temporada** e proposta de detector de anos epidêmicos | [15](15_modelos_por_dentro_e_picos.html) (artifact https://claude.ai/artifact/Gk3ELYHdWriSWrmsFwDQK6) | Gráficos R2_7: em 2023/24 a temporada previu 2–10% do real no Sul/Sudeste (Brasil 0,74 mi × 6,70 mi). Sinais na SE25 (casos de abril a junho relativos ao normal da UF: ρ 0,46; variação do ENSO: ρ 0,42; SEAS5: ≈0). Ridge só com temporadas passadas previu o tamanho de 2023/24 com correlação 0,84 entre UFs (seleção de sinais ainda olhou 2023: refazer só com a afinação). Próximo: rodada 3 = detector + mistura normal/pico + revisão mensal |
| 13/10 | **Rodada 3: detector + mistura normal/pico + revisões mensais** (3 notebooks, só CPU) | [16](16_RODADA3_DETECTOR_MISTURA.md), `data_imdc_2026/RODADA3/` | Só fumaça no PC: o detector, com sinais escolhidos até 2021, tem correlação 0,8 no teste e prevê PR 2023/24 em ×18,6 (real ×23,6). Estimativas: a rodar no Colab |
| 13/10 | **Literatura sobre picos e explosões** (PubMed + Consensus) e plano da rodada 4 | [17](17_LITERATURA_PICOS_E_PROXIMOS_PASSOS.md) | Sinais a acrescentar: estabilidade entre surtos (Rypdal 2019), suscetíveis, sorotipo, IOBW, SPEI. Retreinos: DoisEstagios com detector (CPU) e MTGNN condicionado (GPU). Temporada rolante com EAKF |
| 13/10 | **Rodada 3 avaliada (olha o teste)** | [18 §1](18_RODADA4_LITERATURA_APLICADA.md), `avaliacao_r3/` | Recomendada AJ_continuo_UF 0,828 × BB (o normal tinha 0,831), ainda −28% na altura do pico. **MIX_escala_UF acerta os picos:** altura −4% (normal −42%), WIS absoluto 776 (normal 1.343), 91% dos picos no IC90. Mas R 1,19 por inflar a entressafra (+156%). Detector: corr. 0,8, precisão ~50%, encolhe os extremos (SC 2023 ×45; real ×206) |
| 13/10 | **Rodada 4: detector ampliado + MIX_elastico + DoisEstagios_det + MTGNN_cond + ensemble por regime** (5 notebooks: R4_0 sorotipo opcional, R4_1, R4_2 CPU ∥ R4_3 GPU, R4_4) | [18](18_RODADA4_LITERATURA_APLICADA.md), `data_imdc_2026/RODADA4/` | Só fumaça no PC (normal = climatologia; mar e sorotipo sintéticos). Achado: a elasticidade ao tamanho vai de 0,09 (semanas sem casos) a 0,99 (pico). Sorotipo encontrado no OpenDataSUS. EAKF não feito (não ajuda na SE25). Estimativas: a rodar no Colab |
| 13/10 | **Rodada 4 avaliada (olha o teste)** | [18 §6](18_RODADA4_LITERATURA_APLICADA.md), `avaliacao_r4/` | Pela régua R, nada superou AJ_continuo 0,828 e ENS_grafos3_λ 0,831. ENS_regime_UF tem o melhor critério misto (0,900): WIS absoluto 946 contra 1.343, 86% dos picos no IC90, mas R 1,03. O detector ampliado piorou (correlação 0,57 contra 0,79; MG 2023 ×0,49, real ×20): variável nacional do Atlântico norte (recorde em 2023) sobreajustada. Os membros novos herdaram o erro |
| 14/10 | **Rodada 5: modelo final de temporada** (1 notebook, CPU): detector corrigido, membros refeitos, ensemble por fase esperada × regime, calibração por fase, escolha validada deixando uma temporada de fora | [19](19_RODADA5_MODELO_FINAL.md), `data_imdc_2026/RODADA5/` | Bug achado: o filtro de variáveis nacionais da R4 nunca funcionou (agrupava sem o corte). Só fumaça no PC. Estimativas: a rodar no Colab |
| 13/10 | **Rodada 5 avaliada (olha o teste)** | [19 §Resultados](19_RODADA5_MODELO_FINAL.md), `avaliacao_r5/` | Detector R3:ridge (correlação 0,82). **MIX_elastico5_UF: R 0,859, 21 UFs > BB, WIS absoluto 1.010 (normal 1.343), altura do pico −12% (normal −42%), total −8%, entressafra WIS 27,8 (AJ 26,4).** É o melhor equilíbrio, mas o protocolo recomendou o ENS_grafos3_λ (afinação 0,917 contra 0,950). Os ensembles por fase ganham no pico (88% no IC90), mas inflam a entressafra (+104%) |
| 14/10 | **Rodada 6: otimização do pico do MIX_elastico e do MIX_escala** (1 notebook, CPU, Optuna): todos os parâmetros abertos, um detector para cada modelo, objetivo = WIS da fase ÷ R5 com restrição de 5% nas outras fases, busca em 2015–2021 e validação aninhada (cada temporada fora) | [20](20_RODADA6_OTIMIZACAO_PICO.md), `data_imdc_2026/RODADA6/` | Achado: o detector R3:ridge só existe a partir de 2018. Em 2015–2017, o ridge reajustado o substitui. Reimplementação confere com a R5 (1–4% na mediana com 1.000 trajetórias). Só fumaça no PC. Estimativas: a rodar no Colab |
| 13/10 | **Rodada 6 avaliada (olha o teste)** | [20 §Resultados](20_RODADA6_OTIMIZACAO_PICO.md), `avaliacao_r6/` | A validação aninhada rejeitou as duas versões, e o teste confirmou: a otimização só troca erro entre fases. **MIX_elastico6:** WIS do pico −15% (alargando a faixa; altura −9,5% contra −11,8%), mas entressafra e intermediária +14–16%; R 0,947. **MIX_escala6:** entressafra −47% e R 1,29 → 0,85 (21 UFs > BB), mas pico −21% (WIS do pico +55%). Para o ensemble do pico, o par que se compensa é o da R5 (−12% e +12%) |
| 14/10 | **Rodada 7: ensemble por nível previsto** (1 notebook, CPU, ~10 min): fora do pico, um membro-base; no pico, a média ponderada de elastico5 e escala5, com rampa suave pelo nível previsto em junho; escolha por grade (~1.950 combinações), com o objetivo WIS do pico ÷ elastico5 e restrição de 2% nas outras fases; validação aninhada 2015–2021 | [21](21_RODADA7_ENSEMBLE_PICO.md), `data_imdc_2026/RODADA7/` | Só fumaça no PC. Estimativas: a rodar no Colab |
| 13/10 | **Rodada 7 avaliada (olha o teste)** | [21 §Resultados](21_RODADA7_ENSEMBLE_PICO.md), `avaliacao_r7/` | **ENS_pico7_UF** (base AJ; 60% escala5 no pico; rampa em z = 2): **R 0,808 × BB, o melhor do projeto**, 22 UFs > BB, WIS 902, MAE 1.452, sMAPE 65,7; ganha do elastico5 nas 3 fases, mas a validação aninhada não o recomendou (entressafra 1,09, intermediária 1,11). **ENS_pico7_simples** (a priori): altura +2%, total −0,6%, WIS 721, MAE 1.277, WAPE 58%, mas R 0,935 |
| 14/10 | **Histórico completo do projeto** (de junho à R7): fases, modelos, métricas, hipóteses e bugs | [22_historico_completo.html](22_historico_completo.html) (artifact https://claude.ai/artifact/4es3dBDrzLCmoMKZq39MrV) | Documento de consulta; reúne os docs 00–21 |
| 11/10 | **Relatório para a DASA + código dos gráficos** | `12_relatorio_modelos_dengue.html` (artifact https://claude.ai/artifact/E3V9qYD6ve2tNmvPYiMVWk), `data_imdc_2026/SEMANAL/graficos_colab.py` | Top 5 semanal e de temporada com métricas explicadas e a evolução em relação ao repositório do Felipe. Os gráficos (leques semanal, de temporada e semanal × temporada) são colados no Semanal3 |
| 11/10 | **Resultados semanal × temporada** (Semanal1–3 no Colab) | [11](11_SEMANAL_X_TEMPORADA.md) | **Semanal h = 1:** ENS_GBM_ARIMA_sem 0,224 × clim. (0,884 × persistência; MAE 285; IC90 91); ARIMA tem o menor MAE (252; WAPE 11,5%). As redes com âncora são ruins na semana (0,54); com crescimento, 0,235. Felipe: MAE de 800–950. **Temporada:** novo melhor, ENS_grafos2_λ 0,835 × BB; MTGNN_pesopop_UF+λ 0,839 (o melhor isolado) |
| 08/10 | **Retreino semanal de todos os modelos + próximos passos da temporada** (3 notebooks Colab, retomáveis) | [11](11_SEMANAL_X_TEMPORADA.md), `data_imdc_2026/SEMANAL/` | Só fumaça no PC (sem erros; a persistência repete 507 / 16,6% na rodada do Felipe). Estimativas: a rodar no Colab |
| 08/10 | **Jogos cruzados com o Felipe + top 5 por métrica** | [10 §5–6](10_COMPARACAO_FELIPE_SMAPE.md), `COMPARACAO_FELIPE/` (`CompFelipe.ipynb` para o Colab) | **No meu jogo:** o Felipe com quantis tem 1,256 × BB (ponto: 3,30), contra 0,846 do MTGNN+λ (régua local validada: emap 0,870 igual ao Colab). **No jogo dele (h = 1):** o GBM_log_cresc tem MAE 324 e WAPE 10,6% no teste 2023–24, contra a persistência (507 · 16,6%) e o XGBoost dele (1.165 · 38,1%). O WIS em casos favorece TFT_pool e NHITS_pool, não os grafos. Pendente: CompFelipe A (MAE/sMAPE dos meus) e B (grafos a 1–4 semanas) |
| 08/10 | **Comparação com o modelo do Felipe ("SMAPE de 16%")**: o código dele rodou nos zips do SINAN de 2022–23, e o modelo dele foi replicado nos nossos dados | [10](10_COMPARACAO_FELIPE_SMAPE.md), `COMPARACAO_FELIPE/` | Dados equivalentes (corr. 0,9994; dele = UF de notificação, sem ES). O "16%" é a **1 semana à frente** (WAPE da persistência 16,8% na S1). Na regra do IMDC, o mesmo XGBoost tem sMAPE de 96%, contra 72% do BB e 60% da melhor equipe |
| 01/10 | **Resultados por UF (CompUF4 + Comp5)**: 83 modelos; escolha em 2018–21, teste em 2022–25 | [09](09_RESULTADOS_UF_E_LICOES.md), `resultados_uf/` | **Final ENS_UF_top5 (mediana: DoisEstagios, MTGNN, STGCN, GCNGRU com λ = 0,8 + XGBoost_UF): 0,853 × BB no teste (20 UFs)**, melhor que o ARIMA_Fourier (0,865) e que todas as equipes do 3º IMDC (0,870). Melhor isolado: MTGNN_UF+λ 0,846. Compra (τ = 0,75): DoisEstagios_pool_UF 0,886 → 0,871 × climatologia. Variáveis: nenhum grupo muda o GBM em mais de 1%; exógenas futuras não ajudam o ARIMAX |
| 01/10 | **Avaliação robusta e métricas de negócio**:<br>• leitura única de previsões (`L.carregar_previsoes`): qualquer modelo com 9 quantis, inclusive de fora (`L.importar_externo`), com remoção de duplicatas e exclusão das incompletas;<br>• `avaliar`/`ensemble` sem NaN nem zeros silenciosos;<br>• CompUF4/Comp5 leem todas as pastas `resultados_uf*`;<br>• Comp5 com escolha afinação → teste e a comparação das métricas entre si | `comp_lib.py`, `v4_lib.py`, `uf_lib.py`, `CompUF4`, `Comp5`, [08](08_AVALIACAO_ROBUSTA_E_NEGOCIO.md) | CompUF1–3 rodaram sem erros, mas os resultados ficaram em **duas pastas `resultados_uf`** (sessões simultâneas). O CompUF4 falhou por `uf_lib.py` antigo no Drive. Ensaio sintético dos dois notebooks: OK |
| 01/10 | **auto.arima, VAR, redes de grafo e seleção de variáveis**:<br>• grade do ARIMA trocada pelo auto.arima completo (Python e R), com K de Fourier pelo AICc, versões d livre e d = 0, ARIMA e ARIMAX;<br>• 6 VARs (macrorregião, espacial e BVAR Minnesota, com e sem ENSO);<br>• redes de grafo A3T-GCN, STGCN e MTGNN (dos artigos do Drive);<br>• conjuntos de variáveis padronizados (`EXOG_FUTURAS`, `SELECIONADAS`);<br>• leitura das pastas Econometria/Economics/Articles do Drive | `UF/uf_lib.py`, `UF/uf_autoarima.R`, `UF/avaliar_variaveis.py`, [07](07_ECONOMETRIA_GRAFOS_E_VARIAVEIS.md) | Fumaça OK nos 19 modelos novos (não são estimativas): auto.arima livre ≈1,9 × BB (extrapola a tendência), d = 0 ≈1,1. Variáveis (afinação 2015–21): recente +5%, Richards +4,9%, clima +2,7%, vizinhos pela razão +2,1%; SEAS5 −3,9%, taxas de ataque antigas −1,4% a −9,1%; todas juntas −16% (sobreajuste) |
| 01/10 | **Notebooks dos modelos direto por UF** (7 fases):<br>• portados da competição;<br>• grade do ARIMA e ARIMAX;<br>• INLA não linear (`ghr_nl`, novo no `comp_inla.R`);<br>• dois estágios;<br>• GBMs (LightGBM, XGBoost, CatBoost) com covariáveis, pooling, aumento de dados e ablação;<br>• 6 redes do neuralforecast com pooling, STID e GCN-GRU;<br>• Chronos-2 com covariáveis;<br>• avaliação com afinação em 2018–2021 e teste em 2022–2025.<br>O Comp5 passou a ler os resultados por UF | `data_imdc_2026/UF/` (`uf_lib.py`, `CompUF1–4`, LEIA-ME) | Só testes de fumaça no PC (14 modelos + ensaio do CompUF4). Estimativas: a rodar no Colab |
| 30/09 | **Tabelas unificadas por UF** recalculadas dos brutos:<br>• contagens somadas e taxas refeitas;<br>• clima pela média ponderada pela população, com as variáveis não lineares (VPD, HTI, volatilidade) calculadas no município antes;<br>• vizinhos refeitos no grafo das UFs pela razão agregada;<br>• bioma e Köppen como % da população;<br>• variáveis novas de espalhamento e urbanização | `data_imdc_2026/UF/` (LEIA-ME, `dados/relatorio_validacao_uf.md`) | 38/38 checagens OK. Semanal 21.970 × 214; temporada 390 × 81 (variáveis na SE25 + alvos). Achado: o `viz_*` do V4 usa a média simples dos logs dos vizinhos; a nova `vizp_*` usa a razão agregada |
| 30/09 | **Competição concluída no Colab** (Comp1–4; INLA corrigido: binário no Drive sem permissão e pacote `sn`) | `COMPETICAO/resultados_colab_2026-09-30/`, [05 §4](05_V4_E_COMPETICAO.md) | **ARIMA_Fourier 0,865 × BB** (19 UFs; 1º em 4 de 5 macrorregiões; 0,878 × climatologia por região, 343/435). Climatologia 0,993; ENS_mediana 1,003 (o mais estável); AR_longo 1,015; LightGBM 1,032; Renovação 1,167 (1º no Sul); GHR_INLA 1,233; XGBoost 1,275; **AGCRN 1,378**; Chronos 1,431; BB_INLA 1,555 (reimplementação não fiel). As redes do neuralforecast não rodaram (`val_size`, já corrigido). Nenhum modelo com covariáveis venceu o ARIMA sem covariáveis |
| 30/09 | **Competição de modelos**: 19 competidores (da revisão e do 3º IMDC) reimplementados por região de saúde, soma até a UF por trajetórias (Schaake), avaliação por UF e por região, 4 notebooks Colab | `data_imdc_2026/COMPETICAO/`, [05](05_V4_E_COMPETICAO.md) | PC (2022): AR_longo 1,03; climatologia 1,04; SURGE 1,15 (fiel ao LNCC, 1,20); Renovação é a melhor por região. `Comp1_CPU` rodado no Colab; INLA, GPU e avaliação pendentes |
| 30/09 | **V4 no Colab (treino completo)**, S1–S4, WIS relativo ao BB oficial | `MyDrive/dengue_V4/resultados_v4/` | Melhor: **ENS_bases_A_mediana 0,938** (18 UFs) e **ENS_bases_AB_mediana 0,944** (19 UFs, dp 0,265 = o mais estável entre os que vencem o BB, inclusive os do 3º IMDC). Topo do 3º IMDC: 0,870–0,924. Sozinhos: AR longo 0,980; climatologia 1,012; A_uf 1,121; B_uf 1,238; A_regiao 1,365; B_regiao 1,382. Ensemble pela mediana > média > pesos. Ablação da etapa 5: tendência de 8 semanas ajuda (+6,5%), clima prolongado (+3,1%); taxa de ataque 1–3 temporadas atrapalha (−6%), SEAS5 (−1,5%). Todos erram o surto do Sul em 2022/23 (PR, SC, RS) |
| 29/09 | **V4 (tarefa oficial: temporada inteira a partir da SE25)**: avaliador WIS conferido contra o oficial, modelos-base, versões A (AGCRN multi-horizonte) e B (tamanho × forma), variáveis de temporada, ensemble. Métrica nova: **WIS relativo ao BB** (<1 = melhor que o modelo-base; os melhores do 3º IMDC ficaram em 0,87–0,90) | `data_imdc_2026/V4/` (ver `LEIA-ME.md`) | Testes no PC, S1–S4: ensemble das bases **0,954** (18 UFs melhores que o BB); climatologia 1,012; B sozinho 1,22. Versão A **pendente** (Colab). Não é comparável com o MAE das linhas abaixo, que é 1 semana à frente |
| 29/09 | **v3.1**: ponto estadual = mediana da soma sem inflação; intervalos = união (com e sem inflação), sempre contendo o ponto. Diagnóstico dos mapas e da v3 | `S1_corrigido/Modelo2_v3.ipynb` (célula de pós-processamento), `S1_corrigido/_analise_v3/` | Mesmos pesos da v3: MAE estadual 318 → **282**, sMAPE 84% → **42%**, IC50/90/95 = 55/88/91 |
| 29/09 | **Treino completo da v3** no Colab | `resultados_modelo2_v3` (Drive) | Município: MAE **2,93**, r **0,84** (melhores até agora). Estado: MAE 318, sMAPE 84% (colapso na SECA, corrigido na v3.1). SECA com k = 48 |
| 29/09 | **Modelo 2 v3**: losses equilibradas (NLL NB + trava simétrica em log), cada especialista treina também no outro regime, pós-processamento calibrado na validação no lugar do MinT | `S1_corrigido/Modelo2_v3.ipynb` | Só o pós-processamento, com os pesos da v2: MAE estadual 243. Treino completo da v3 **pendente** |
| 29/09 | **Diagnóstico da v2**: viés de cada especialista, MinT usando o teste, estado calculado como soma de medianas. Criada a célula para refazer só a previsão | `S1_corrigido/_analise_v2/`, `S1_corrigido/celula_refazer_previsao.py` | MAE estadual 829 (honesto) → 293 (recalibrado) |
| 29/09 | **Modelo 2 v2**: features do SHAP corrigido, previsão alinhada (arquivo `ctx`), validação de 52 semanas, early stopping | `S1_corrigido/Modelo2_v2.ipynb` | MAE estadual 560 (com MinT), r 0,97. SECA parou na época 171 e PICO na 92 |
| 29/09 | **SHAP corrigido** nas 4 safras: alvo = semana seguinte, Poisson com offset de população, sondas de ruído, filtro de redundância. Geradas as listas novas e o FINAL_NN_v2 | `SHAP_corrigido/` | Afya, chikungunya e parte do clima previsto saíram. A incidência observada entrou |
| 29/09 | **Safras 2, 3 e 4 corrigidas**, geradas a partir dos brutos | `refazer_season.py`, `S2_corrigido/` a `S4_corrigido/` | 79/79 checagens OK em cada uma |
| 29/09 | Modelo 2 original rodado nos dados corrigidos da S1 | `S1_corrigido/Modelo2_S1_enxuto.ipynb` | MAE estadual 749, r 0,91 |
| 28/09 | **Notebook enxuto do Modelo 2** para o Colab: AGCRN embutida, trabalho em disco local, checkpoints | `S1_corrigido/Modelo2_S1_enxuto.ipynb` | Roda de ponta a ponta |
| 28/09 | **Season 1 corrigida** (V8), gerada a partir dos brutos, sem duplicatas | `S1_corrigido/refazer_season1.py` | 79 checagens + 38 de auditoria, todas OK |
| 28/09 | Mapa do projeto criado (documentos 01, 02 e 03) | `_MAPA_DO_PROJETO/` | – |
| 17–19/06 | Modelos 1, 2 e 3 originais | `Dengue dasa/Dengue_final_ufvjm (1).ipynb` | Ver tabela da seção 4 |

---

## 7. Pendências

**Linha oficial (V4 e competição):**
- [x] Passar a prever a **safra inteira a partir da semana 25** (V4).
- [x] `Comp1_CPU` no Colab.
- [x] `Comp2_INLA`, `Comp3_GPU` e `Comp4_Avaliacao` (30/09). Vencedor: ARIMA_Fourier.
- [x] Lapidar o ARIMA_Fourier: substituído pelo auto.arima (d livre e d = 0) e superado pelos modelos por UF ([09](09_RESULTADOS_UF_E_LICOES.md)).
- [x] Rodar as redes do neuralforecast: rodaram por UF (com pooling); ficaram entre 1,03 e 2,14 × BB no teste.
- [x] **Modelos direto por UF** (CompUF1–4 e Comp5 rodados em 01/10): final ENS_UF_top5 = 0,853 × BB; compra: DoisEstagios_pool_UF. Ver [09](09_RESULTADOS_UF_E_LICOES.md).
- [ ] MTGNN v2 (5 sementes, λ por horizonte, saída auxiliar do total, pooling com regiões) e ensemble só de redes de grafo.
- [ ] Corrigir a seção 6 do Comp5 (comparar só modelos com as mesmas temporadas).
- [ ] Generalizar para a empresa: origem e janela configuráveis, origens mensais, alvo da empresa, vintages e estoque ([08 §5](08_AVALIACAO_ROBUSTA_E_NEGOCIO.md)).
- [ ] Corrigir o viés para baixo do total (falta em 40–50% das UF × temporada; o ideal em τ = 0,75 é 25%).
- [x] Escolher o modelo final por um critério definido antes e gerar a submissão (ENS_UF_top5; CSVs em `resultados_uf/submissao/`).

**Linha antiga (Modelo 2, 1 semana à frente), como registro:**

- [x] Rodar o treino completo do **Modelo 2 v3** (S1) e registrar as métricas.
- [x] Verificar se os fatores de calibração da v3 ficam perto de 1: sim, os fatores nacionais ficaram em 1,05 (PICO) e 0,93 (SECA).
- [ ] Rodar a v3.1 no Colab (`SO_POS_PROCESSAMENTO = True`) e confirmar o MAE estadual 282.
- [ ] Viés regional do PICO: superestima o Norte, o Nordeste e o Centro-Oeste e subestima os picos do Sul e do Sudeste. Candidatos: embedding/escala por região ou calibrar com várias temporadas em vez de uma.
- [ ] Picos falsos em janeiro (AC, RO, AM, GO, DF, BA): investigar a virada do ano (subnotificação no fim de dezembro seguida de rebote).
- [ ] Pico falso em nov–dez no PR (aparece em todas as versões).
- [ ] Surtos fora de época (RR e AP em ago–set) não são capturados em nenhuma versão.
- [ ] Vencer a **persistência** (MAE estadual 185).
- [ ] Rodar as safras **S2, S3 e S4** (os bancos já estão prontos).
- [ ] Levar os intervalos estaduais ao nominal (v3.1: IC90 = 88% e IC95 = 91%).
- [ ] Liberar espaço em disco: restam ~15 GB no F:. Os bancos antigos em `bancosDEF/` e os `*_dengue_eng` podem ser apagados (ver [03 §5](03_AVALIACAO.md#5-quais-bancos-podem-ser-eliminados)).

---

## 8. Como atualizar este documento

Cada vez que rodar um experimento:

1. Acrescente **uma linha no topo** da seção 6: data, o que mudou, onde e o resultado.
2. Acrescente **uma linha** nas tabelas da seção 4 e **uma linha** no [`metricas.csv`](metricas.csv), com as mesmas colunas.
3. Atualize a seção 2 (versão atual, melhor resultado, próximo passo) e marque as pendências concluídas na seção 7.
4. Troque a data de "Última atualização" no topo.

Regras simples:
- Não apague linhas antigas; se um número se mostrar errado, mantenha a linha e explique na observação.
- Diga sempre se o número "olha o teste".
- Indique a safra.
