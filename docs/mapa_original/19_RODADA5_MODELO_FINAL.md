# 19 · Rodada 5: modelo final de temporada, um especialista por fase (14/10/2026)

**Pedido:** um notebook final, com o melhor modelo de temporada, que preveja bem tanto a entressafra quanto o pico, alternando entre modelos de acordo com a época esperada.

**Onde:** `data_imdc_2026/RODADA5/`
- `r5_lib.py`
- `_gerar_notebook_r5.py` → `R5_Modelo_Final_Temporada.ipynb`
- `LEIA-ME.md`

**Ambiente:** CPU, 40–90 min. Usa os caches das rodadas 3 e 4.

## Desenho

1. **Detector corrigido.** Candidatos:
   - R3;
   - ampliado **sem variáveis nacionais**;
   - média dos dois.

   A escolha é pelo pinball nas temporadas ≤ 2021 (anos grandes 3×), nas mesmas linhas para todos.

   **Bug encontrado na R4:** a função que detecta variáveis nacionais agrupava só por temporada, misturando os cortes. Por isso nada parecia nacional, o filtro |ρ| ≥ 0,30 nunca foi aplicado e o Atlântico norte (recorde em 2023) entrou. A função foi corrigida em `r4_lib._nacional` e reescrita em `r5_lib._nacional`.
2. **Membros refeitos com o detector corrigido.** MIX_escala5, MIX_elastico5 e DoisEstagios_det5. A MTGNN_cond5 usa as redes da R4 em `cache_r4/ckpt`, só para uma nova previsão, sem retreino. Entram também o normal, a MTGNN_aug, o MIX_escala e o AJ_continuo.
3. **Fase esperada de cada semana × UF.** Em junho, é o nível previsto (pelo normal ou pelo MIX_escala5) contra os limiares p25/p75 da UF, calculados até 2017. Soma-se o regime, que é P(ano grande).
4. **Ensembles:**
   - `ENS_fase_*`: pesos por fase × regime (critério misto);
   - `ENS_troca_*`: só o melhor membro de cada fase;
   - `_nivMIX`: a fase esperada vem do MIX_escala5;
   - `+cal`: calibração por fase.

   Os pesos são suavizados em 5 semanas.
5. **Recomendado:** o menor critério misto na afinação, com os ensembles validados deixando uma temporada de fora. Depois vêm o teste, os gráficos e a tabela da temporada mais recente.

## Teste de fumaça (PC, 14/10; normal = climatologia; mar e sorotipo sintéticos; NÃO são estimativas)

- As 9 células rodaram sem erro em ~1 min com o cache pronto. A MTGNN_cond5 refez a previsão com as redes salvas: 1 semente, 2023.
- Com a correção, saíram 17 variáveis nacionais. O detector escolhido em junho foi o `R3:ridge`, que previu 2023 assim:

  | UF | Previsto | Real |
  |---|---|---|
  | SC | ×149 | ×206 |
  | RS | ×155 | ×160 |
  | PR | ×33 | ×24 |
  | SP | ×9,7 | ×16,7 |
  | MG | ×6,4 | ×20 |
- Com o normal falso (climatologia), o recomendado foi o MIX_elastico5: R 0,94 × BB, erro da altura do pico −5%, erro do total −0,3%, 82% dos picos no IC90. Os `ENS_troca` foram piores que os `ENS_fase`. Estes números servem só para conferir o código.

## Resultados (Colab)

**Rodado em 13/10. Olha o teste 2022–2025.** Tabelas em `avaliacao_r5/`.

**Detector escolhido em junho:** `R3:ridge`, escolhido sem olhar o teste. No teste: pinball 0,168, correlação 0,82 e anos grandes previstos em ×0,83 do real (o da R4 dava ×0,16).

| Modelo | R × BB | UFs > BB | WIS abs. | Crit. misto × BB | WIS entressafra | Viés entressafra | WIS pico | Altura do pico | Total | Pico no IC90 |
|---|---|---|---|---|---|---|---|---|---|---|
| AJ_continuo_UF (R3) | **0,828** | 23 | 1.285 | 0,939 | 26,4 | +74% | 3.883 | −28% | −25% | 64% |
| ENS_grafos3_λ (**recomendado pela afinação**) | 0,831 | 20 | 1.343 | 0,964 | **23,6** | **+49%** | 4.065 | −42% | −37% | 67% |
| **MIX_elastico5_UF** | 0,859 | 21 | 1.010 | **0,842** | 27,8 | +85% | 2.994 | −12% | −8% | 74% |
| ENS_fase_nivMIX_UF | 1,006 | 12 | 876 | 0,861 | 37,6 | +106% | 2.517 | −19% | −11% | 88% |
| ENS_fase_UF | 1,007 | 11 | 885 | 0,865 | 36,3 | +104% | 2.541 | −19% | −8% | 88% |
| ENS_troca_UF | 1,147 | 9 | 823 | 0,910 | 41,0 | +119% | 2.324 | −14% | +7% | 88% |
| MIX_escala5_UF | 1,288 | 4 | **746** | 0,949 | 55,1 | +151% | **2.012** | +12% | +9% | 93% |
| MTGNN_cond5_UF | 1,436 | 4 | 1.026 | 1,137 | 52,3 | +146% | 2.805 | +4% | +15% | 90% |
| BB | 1 | — | 1.224 | 1 | 39,0 | +108% | 3.651 | −36% | −24% | 73% |

**Leitura:**
- **O melhor equilíbrio no teste é o `MIX_elastico5_UF`**: o normal reescalado pelo detector corrigido, com a elasticidade por nível.
  - Pela régua R, fica só 3,7% atrás do campeão (0,859 contra 0,828) e vence o BB em 21 UFs.
  - Contra o AJ_continuo: WIS absoluto −21%, erro da altura do pico −12% (contra −28%) e erro do total −8% (contra −25%).
  - A entressafra fica praticamente igual à do AJ (WIS 27,8 contra 26,4; viés +85% contra +74%).
- **O protocolo, porém, recomendou o `ENS_grafos3_λ`.** Na afinação, o critério misto dele foi 0,917, contra 0,950 do MIX_elastico5. A afinação (2018–2021) teve poucos anos grandes, então premiou o modelo "normal".
- **Os ensembles por fase não ganharam na entressafra** (WIS 36–41, viés +104–119%). Os pesos escolhidos na afinação deram muito peso à MTGNN_aug e ao DoisEstagios nas semanas de entressafra, o que é sobreajuste a 4 temporadas. Eles ganham no pico (88% no IC90) e no WIS absoluto, mas perdem na régua R.
- **A troca pura de modelo (`ENS_troca`)** só foi melhor que a mistura no pico.

**Métricas pontuais (teste; MAE = casos por semana, média de todas as UFs e semanas):**

| Modelo | MAE | WAPE | sMAPE (0–200) | Correlação r |
|---|---|---|---|---|
| ENS_grafos3_λ | 1.771 | 80,8% | 75,3 | 0,68 |
| AJ_continuo_UF | 1.672 | 76,3% | 70,0 | 0,70 |
| MIX_elastico5_UF | 1.506 | 68,8% | 70,2 | 0,78 |
| ENS_fase_nivMIX_UF | 1.466 | 66,9% | 64,6 | 0,74 |
| MIX_escala5_UF | 1.279 | 58,4% | 70,4 | 0,82 |
| DoisEstagios_det5_UF | 1.262 | 57,6% | 70,7 | 0,79 |
| BB | 1.787 | 81,6% | 73,0 | 0,58 |

**14/10:**
- O usuário achou o critério misto confuso. Daqui em diante, os modelos são reportados pelas métricas-padrão lado a lado (R × BB, WIS, MAE, WAPE, sMAPE, erro do pico e do total).
- Comparação visual: notebook `RODADA5/R5_Graficos_Comparacao.ipynb` (`_gerar_notebook_graficos.py`):
  - uma figura por UF com 6 modelos e o pico de cada temporada;
  - mapa de calor do erro do pico por UF × modelo;
  - saída em `avaliacao_r5/comparacao_<data>/`.

## Botões para mexer no pico (14/10; inventário; a otimização é a rodada 6, no [doc 20](20_RODADA6_OTIMIZACAO_PICO.md))

O usuário quer que o MIX_elastico5 suba no pico e que o MIX_escala5 infle menos. Os dois modelos fazem a mesma conta: pegam as trajetórias do normal, sorteiam o tamanho da temporada no detector (`alvo = expm1(base + a)·pop/1e5`) e reescalam.

**Por que cada um erra:**
- **MIX_elastico5 fica baixo por quatro motivos:**
  - o detector prevê os anos grandes em ×0,83 do real;
  - o `w`, se ficar abaixo de 1, puxa o total de volta ao normal;
  - a forma do normal já vem achatada;
  - b fica em torno de 1 no pico, estimado juntando anos pequenos e grandes.
- **MIX_escala5 infla a entressafra** porque multiplica todas as semanas pelo mesmo fator quando `a` passa do limiar.

**Botões baratos (CPU, sem retreinar redes):**

| Componente | Botões |
|---|---|
| Detector | correção do viés nos anos grandes; peso dos anos grandes no treino; alargamento do resíduo (1,15); alphas do ridge; k e rho_min; base (média de todas as temporadas ou janela recente); corte das caudas em `_interp` (t ∈ [−1,5; 2,5]) |
| MIX_elastico | `w`, com grade que chegue até 1,5; `NIVEIS_ELAST`; clip de b em [0,03; 1,5]; b só dos anos acima do normal; expoente em b; afinação da forma (traj^η); antecipação do pico nos anos grandes |
| MIX_escala | limiar (log 1,5, 2 ou 3); troca suave (sigmoide) no lugar de liga/desliga; reescalar só as semanas de nível alto; teto do fator na entressafra |
| Pós-calibração | largura λ e posição δ por fase (`GRADE_LAM4`, `GRADE_DELTA4`) |

**Botões caros:** retreinar as redes do normal (GPU).

**Validação proposta:** a afinação 2018–2021 tem poucos anos grandes. O mecanismo pode ser validado em 2011–2021 com a climatologia no lugar do normal, deixando uma temporada de fora a cada vez.

**Objetivo proposto:** o WIS do pico, com a restrição de que o WIS da entressafra não piore mais que 5%.
