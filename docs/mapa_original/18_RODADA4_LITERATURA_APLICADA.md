# 18 · Rodada 3 avaliada e rodada 4 (literatura aplicada) · 13/10/2026

## 1. O que a rodada 3 mostrou (olha o teste 2022–2025, corte de junho)

Tabelas em `MyDrive/dengue_V4/avaliacao_r3/` (`temporada_teste_r3_ultimo.csv` e as de fase, pico, cobertura, revisões e detector).

| Modelo | R × BB | WIS absoluto | WIS no pico | Viés no pico | Viés na entressafra | Erro da altura do pico (mediana) | Erro do total (mediana) | Pico no IC90 |
|---|---|---|---|---|---|---|---|---|
| ENS_grafos3_λ (normal) | 0,831 | 1.343 | 4.065 | −82% | +49% | −42% | −37% | 67% |
| AJ_continuo_UF (recomendada R3) | **0,828** | 1.285 | 3.883 | −76% | +74% | −28% | −25% | 64% |
| **MIX_escala_UF** | 1,190 | **776** | **2.136** | **−36%** | **+156%** | **−4%** | **+5%** | **91%** |
| MIX_forma_UF | 1,242 | 766 | 2.113 | −35% | +173% | −4% | +5% | 91% |
| BB | 1 | 1.224 | 3.651 | −80% | +108% | −36% | −24% | 73% |

**Leitura:**
- **O detector + reescala funciona para os picos.** O `MIX_escala_UF` cortou o WIS absoluto em 42% e o WIS do pico em 47%, e a altura do pico passou a errar só −4%.
- Ele perde na régua R (média geométrica por UF) por um único defeito: multiplica a **curva inteira**, então a entressafra e a fase intermediária ficam +156% e +44% acima do real nas UFs pequenas.
- A escolha da R3 (menor R na afinação) premiou a variante tímida.
- **Cobertura:** a mediana do normal fica acima do real só 29% das vezes (subestima sempre); a do MIX_escala, 55% (calibrada).
- **Detector (junho, teste):** correlação de 0,79–0,82 e sensibilidade de 82–89%, mas precisão de 50–56% (metade dos alarmes são falsos). Encolhe os extremos: SC 2023 ×45 (real ×206), RS ×53 (real ×160), DF ×3,5 (real ×23), MG ×5,5 (real ×20).
- **Revisões jul–out:** pouco ganho (AJ 0,828 → 0,793 em outubro).

## 2. Rodada 4: o que foi feito (`data_imdc_2026/RODADA4/`)

| Recomendação (doc 17) | Implementação | Onde |
|---|---|---|
| Anos grandes são o ponto fraco | Escolha do detector com os anos grandes valendo 3×, nas temporadas ≤ 2021; critério misto (½ R + ½ WIS absoluto) para escolher misturas, pesos e a recomendada | `pinball_escolha`, `criterio`, `PESO_ABS` |
| O inverno prevê o verão | Estabilidade entre surtos e alerta precoce: autocorrelação, variância e as tendências de ambas (Kendall), dispersão, piso relativo; imunidade com esquecimento (τ = 1 e 3); ataque acumulado | `_ews`, `_imunidade` |
| Clima com antecedência real | Seca tipo SPEI (13 e 26 semanas e o máximo do último ano); anomalias de clima observado; SEAS5 emitida até o corte (1–3 e 4–6 meses); mar do ERSST v5 (IOBW, Atlântico tropical norte e sul, Atlântico sudoeste, Niño 3.4), usando só o mês já publicado | `series_causais`, `_seas5`, `indices_oceano` |
| Sorotipo | **Encontrado online:** SINAN no OpenDataSUS (`DENGBRaa.csv.zip`, 2010–2026). Coluna `SOROTIPO` por UF × semana, com atraso de 8 semanas; frações encolhidas para a nacional; sorotipo emergente e "antes raro" | `R4_0`, `baixar_sorotipo`, `_sorotipo` |
| Duas etapas | Método `duas_etapas`: logística P(grande) + ridge do tamanho para grandes e não grandes | `_duas_etapas` |
| Calibração da faixa | Detector `+conf` (cada quantil corrigido pelos erros fora da amostra de temporadas terminadas); ensemble `+cal` por regime (grade limitada a ×1,5) | `conformalizar`, `ajustar_regime` |
| Retreino 1: DoisEstagios com o detector | `DoisEstagios_det_UF`: tamanho do detector × forma de temporadas **parecidas em tamanho** (UF + macro) | `gerar_dois_estagios_det` |
| Retreino 2: MTGNN condicionado + aumento sintético | `MTGNN_cond_UF`: MTGNN_pesopop + condição (tamanho relativo da temporada principal da janela), 20% de cópias sem condição, 60% de cópias com o futuro × 2–8; previsão com 24 cenários do detector (ρ = 0,6 entre UFs). `MTGNN_aug_UF` = a mesma rede sem condição | `amostras_cond`, `treinar_cond`, `prever_temporada_cond` |
| Correção do MIX | `MIX_elastico_UF`: cada semana × UF cresce conforme a elasticidade histórica ao tamanho, por nível de incidência | `elasticidades`, `mistura_elastica_uf` |
| Ensemble com pesos que mudam conforme a probabilidade | `ENS_regime_UF`: pesos por faixa de P(grande) (< 0,35, 0,35–0,65, ≥ 0,65); validação deixando uma temporada de fora; a recomendada sai dessa validação | `pesos_regime`, `cv_regime` |
| Temporada rolante (EAKF) | **Não feita, de propósito:** na SE25 não há dado da temporada que começa na SE41, então não ajuda a previsão de junho. Depois de outubro, quem atualiza é o semanal (h = 1: 0,224) e as revisões mensais do MIX_elastico | — |

**Achado empírico que mudou o desenho:** a elasticidade de cada semana ao tamanho da temporada (temporadas ≤ 2021) depende do nível de incidência normal da semana:

| Nível normal, log1p(incidência) | 0–0,5 | 0,5–1 | 1–1,5 | 1,5–2 | 2–3 | > 3 |
|---|---|---|---|---|---|---|
| Elasticidade | 0,09 | 0,33 | 0,50 | 0,67 | 0,87 | 0,99 |

A entressafra quase não cresce nos anos grandes; o pico cresce na mesma proporção. Reescalar a curva inteira (MIX_escala) era o erro.

## 3. Ordem e ambientes

| Ordem | Notebook | Ambiente | Tempo |
|---|---|---|---|
| 0 (opcional) | R4_0_Sorotipo_SINAN | CPU | 30–60 min (uma vez) |
| 1 | R4_1_Detector_ampliado | CPU | 20–40 min |
| 2 ∥ 3 | R4_2_Mistura_DoisEstagios (CPU) **em paralelo com** R4_3_MTGNN_condicionado (GPU L4) | CPU / GPU | 20–40 min / 2–4 h |
| 4 | R4_4_Ensemble_Avaliacao | CPU | 15–30 min |

## 4. Teste de fumaça (PC; normal = climatologia; mar e sorotipo sintéticos; NÃO são estimativas)

- Fluxo completo sem erros: detector (≈40 s), mistura, DoisEstagios, regime com validação e MTGNN_cond (1 época, 1 semente, 48 s em CPU).
- **Detector:** a escolha que pesa os anos grandes ficou com a `media`; o LightGBM, escolhido só pela afinação, previa SC 2023 em ×9. Em 2023, SC ×42 (q95 ×304; real ×206) e RS ×31 (q95 ×199; real ×160).
- **MIX_elastico:** a elasticidade por nível tirou ~50 pontos do viés da entressafra em relação à elasticidade por semana.
- **Ensembles:** a calibração com deslocamento grande (grade da R2, até ×3,3) piorava muito fora da amostra (R 3,1), o que levou à grade limitada (×1,5) e à validação deixando uma temporada de fora.

## 5. Regras contra vazamento (as mesmas)

- Janela expansiva.
- Sinais, métodos, pesos, w e calibração escolhidos só com temporadas ≤ 2021.
- Anomalias de clima e do mar só com anos anteriores.
- SEAS5 e mar só com o que já tinha sido publicado no corte.
- Sorotipo com atraso de 8 semanas. Ressalva: usa os arquivos finais, mais completos que os da época.
- Teste olhado uma vez.

## 6. Resultados (Colab)

**Rodado em 13/10. Olha o teste 2022–2025.** Tabelas em `avaliacao_r4/`.

| Modelo | R × BB | WIS abs. | Crit. misto × BB | Viés no pico | Viés na entressafra | Altura do pico | Total | Pico no IC90 |
|---|---|---|---|---|---|---|---|---|
| AJ_continuo_UF (R3) | **0,828** | 1.285 | 0,939 | −76% | +74% | −28% | −25% | 64% |
| ENS_grafos3_λ (R2; recomendada na afinação) | 0,831 | 1.343 | 0,964 | −82% | +49% | −42% | −37% | 67% |
| MTGNN_aug_UF | 0,886 | 1.516 | 1,062 | −84% | +87% | −51% | −37% | 57% |
| MIX_elastico_UF | 0,918 | 1.491 | 1,068 | −81% | **+38%** | −43% | −42% | 56% |
| **ENS_regime_UF** | 1,026 | 946 | **0,900** | −64% | +87% | −30% | −19% | 86% |
| MIX_escala_UF (R3) | 1,190 | **776** | 0,912 | **−36%** | +156% | **−4%** | +5% | 91% |
| DoisEstagios_det_UF | 1,255 | 1.453 | 1,221 | −79% | +41% | −51% | −44% | 67% |
| MTGNN_cond_UF | 1,292 | 1.430 | 1,230 | −65% | +61% | −52% | −42% | 66% |
| ENS_regime_UF+cal | 1,314 | 856 | 1,007 | −56% | +128% | −15% | **+1%** | **92%** |

**Diagnóstico: o detector ampliado piorou, e isso derrubou todos os membros novos.**
- O escolhido em junho foi `duas_etapas`. No teste, comparado com o detector da R3:
  - pinball 0,35 contra 0,18;
  - correlação 0,57 contra 0,79;
  - anos grandes previstos em ×0,16 do real, contra ×0,67.
- Em junho de 2023 ele previu MG ×0,49 (real ×20), SP ×0,33 (real ×17), DF ×0,32 (real ×23) e RJ ×0,37 (real ×11).
- **Causa provável:** entraram `tna_m3` e `tna_d6` (Atlântico tropical norte; ρ = −0,35 até 2021). São variáveis **nacionais**, com só ~6 temporadas independentes. Em 2023 esse oceano bateu recorde de calor, fora da faixa do treino, e o sinal negativo "previu" um ano pequeno. É sobreajuste a uma correlação espúria que a régua |ρ| ≥ 0,30 não barrou.
- **O que funcionou:** o ensemble por regime ganhou no critério misto (0,900) e cortou o WIS absoluto em 30% (946 contra 1.343), com 86% dos picos no IC90. Mas quem puxou o ganho foi o `MIX_escala` (detector da R3). Pela régua oficial R, nada superou a R3/R2.
- **Próximo passo:** detector sem variáveis nacionais (ou com a escolha validada deixando uma temporada de fora) e ensemble/calibração por **nível previsto da semana** (pico × entressafra), em vez de um peso único por UF.
