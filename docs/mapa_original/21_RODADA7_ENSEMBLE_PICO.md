# 21 · Rodada 7: ensemble por nível previsto, com elástico + escala no pico (14/10/2026)

**Pedido:** um ensemble entre o MIX_elastico e o MIX_escala, com média ponderada na época de pico, para que a subestimação de um compense a superestimação do outro.

**Por que o par da R5:** no teste, as versões da R6 ficaram as duas abaixo no pico (−9,5% e −21%). O par que se compensa é o da R5 (doc 20):

| | Altura do pico | WIS do pico | WIS da entressafra |
|---|---|---|---|
| MIX_elastico5 | −12% | — | 27,8 |
| MIX_escala5 | +12% | 2.012 (o menor de todos) | 55,1 |

**Onde:** `data_imdc_2026/RODADA7/`
- `r7_lib.py`
- `_gerar_notebook_r7.py` → `R7_Ensemble_Pico.ipynb`
- `LEIA-ME.md`

**Ambiente:** CPU, ~5–10 min.

## Desenho

1. **Nível previsto em junho:** `z = (log1p(inc prevista) − log1p(p25)) / (log1p(p75) − log1p(p25))`. A incidência vem da mediana do membro `nivel` (elastico5, escala5 ou normal); os limiares p25/p75 são da UF, até 2017.
2. **Peso do pico:** `s = 1/(1 + exp(−(z − c)/h))`, uma rampa suave.
3. **Combinação:** `ENS = (1 − s)·BASE + s·[(1 − a)·elastico5 + a·escala5]`, por média dos quantis. A BASE é o normal, o AJ, o elastico5 ou o escala6.
4. **Grade:** base × a (0–1) × c (−0,5 a 2) × h (0,1 a 1) × nivel, ~1.950 combinações. O elastico5 puro também está na grade.
5. **Objetivo:**
   - minimizar o WIS do pico ÷ elastico5, com média geométrica entre UFs;
   - restrição: a entressafra e a intermediária podem piorar no máximo `TOL = 2%`.
6. **Validação:**
   - 2015–2021, com os membros reimplementados pela `r6_lib` (M = 500). Antes de 2018, o normal é a climatologia, e o AJ é reproduzido como escala sempre ligado com expoente p.
   - Validação aninhada: a escolha é refeita deixando cada temporada de fora.
   - Recomendação só se, fora da amostra, o pico cair ≥ 2% sem estourar a restrição.
7. **Teste 2022–2025:** com as previsões gravadas dos membros. A referência a priori, `ENS_pico7_simples_UF`, usa base elastico5, a = 0,5, c = 1 e h = 0,3.
8. **Diagnóstico de complementaridade:** em quantas UFs × temporadas o pico real fica entre o elastico5 e o escala5, ou abaixo/acima dos dois. Na validação (célula 3) e no teste (célula 6).

## Teste de fumaça (PC, 14/10; caches falsos; NÃO são estimativas)

- As 8 células rodaram sem erro, com a grade avaliada em ~10 s.
- A célula 5 avisa, em vez de quebrar, se faltar algum membro nas previsões gravadas.

## Resultados (Colab, 13/10; `avaliacao_r7/`)

**Escolha na grade (2015–2021):** base AJ, a = 0,6, c = 2,0 (borda da grade), h = 0,6, nível pelo normal.

**Validação aninhada (fora da amostra, ÷ elastico5):**

| Pico | Entressafra | Intermediária |
|---|---|---|
| 0,951 | **1,087** | **1,108** |

Pela regra, o ensemble **não foi recomendado**. Também a escolha mudou muito de dobra para dobra: a base alternou entre elastico5, normal e AJ, e c entre 1,5 e 2.

**Diagnóstico de complementaridade:** na maioria das UFs × temporadas, o elastico5 e o escala5 erram para o mesmo lado (os dois abaixo ou os dois acima). O pico real raramente fica entre eles, porque o erro dominante é o do detector, que os dois compartilham.

**Teste 2022–2025:**

| Modelo | R × BB | UFs > BB | WIS | WIS pico | WIS entressafra | WIS intermediária | MAE | WAPE | sMAPE | Altura | Total | Pico no IC90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ENS_pico7_UF** | **0,808** | 22 | 902 | 2.667 | 27,3 | 100,9 | 1.452 | 66,3% | **65,7** | −20% | −16% | 76% |
| ENS_pico7_simples_UF | 0,935 | 16 | **721** | 2.061 | 29,6 | 123,9 | **1.277** | **58,3%** | 67,5 | **+2%** | **−0,6%** | 86% |
| MIX_elastico5 | 0,859 | 21 | 1.010 | 2.994 | 27,8 | 109,7 | 1.506 | 68,8% | 70,2 | −12% | −8% | 74% |
| MIX_escala5 | 1,288 | 4 | 746 | **2.012** | 55,1 | 196,6 | 1.279 | 58,4% | 70,4 | +12% | +9% | 93% |
| AJ_continuo | 0,828 | 23 | 1.285 | 3.883 | 26,4 | 96,4 | 1.672 | 76,3% | 70,0 | −28% | −25% | 64% |
| ENS_grafos3_λ | 0,831 | 20 | 1.343 | 4.065 | **23,6** | 97,5 | 1.771 | 80,8% | 75,3 | −42% | −37% | 67% |

WIS por fase ÷ elastico5 no teste:

| Modelo | Entressafra | Intermediária | Pico | Geral |
|---|---|---|---|---|
| ENS_pico7_UF | 0,91 | 0,93 | 0,97 | 0,91 |
| simples | 1,07 | 1,26 | 0,93 | 0,97 |

**Leitura:**
- **ENS_pico7_UF tem o melhor R × BB de todo o projeto (0,808)** e ganha do elastico5 nas três fases no teste.
- A validação, porém, previu o contrário na entressafra e na intermediária. Adotá-lo é decisão tomada olhando o teste.
- **O simples, fixado a priori, é o melhor em altura do pico (+2%), total (−0,6%), MAE e WAPE.** Em troca, piora a intermediária em 26%.

