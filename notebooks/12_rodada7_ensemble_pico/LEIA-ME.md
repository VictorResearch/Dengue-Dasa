# RODADA 7 · ensemble por nível previsto (14/10/2026)

**Por que existe:** a R6 mostrou que otimizar cada modelo sozinho só troca erro entre as fases. Mas, no pico, os dois modelos da R5 erram em sentidos opostos: MIX_elastico5 −12% e MIX_escala5 +12% na altura.

**Como o ensemble funciona.** Cada semana × UF recebe, em junho, um nível previsto `z`: 0 = p25 da UF e 1 = p75.
- **Peso do pico:** `s = 1/(1 + exp(−(z − c)/h))`.
- **Componente de pico:** `PICO = (1 − a)·elastico5 + a·escala5`, média ponderada dos quantis.
- **Ensemble:** `ENS = (1 − s)·BASE + s·PICO`, com BASE = normal, AJ, elastico5 ou escala6.

**Escolha:**
- **Método:** uma grade de ~2.000 combinações; nada de Optuna.
- **Objetivo:** WIS do pico ÷ elastico5, com a entressafra e a intermediária piorando no máximo 2%.
- **Validação:** 2015–2021, aninhada (cada temporada fora).
- **Recomendação:** só se o pico cair ≥ 2% fora da amostra.

## Como rodar

Um notebook, em CPU comum, ~5–10 min: **`R7_Ensemble_Pico.ipynb`**.

1. Suba `r7_lib.py` para `MyDrive/dengue_V4/`. O `r6_lib.py` já deve estar lá.
2. Ele usa:
   - `cache_r3/` (normal, detector_variaveis e mistura_parametros.json com `escala` e `continuo`);
   - `cache_r5/`;
   - `cache_r6/melhor_*.json`;
   - `resultados_uf_r3/` (AJ_continuo), `resultados_uf_r5/` e `resultados_uf_r6/`.
3. Saídas:
   - `resultados_uf_r7/`: ENS_pico7_UF e ENS_pico7_simples_UF;
   - `cache_r7/`: membros da validação;
   - `avaliacao_r7/`: grade, validação aninhada, complementaridade, teste, gráficos e rampa.

## Células

| Célula | O que faz |
|---|---|
| 1 | Setup e ajustes (`VALIDACAO`, `M_VAL`, `TOL`, `GANHO_MIN`) |
| 2 | Membros na validação, gerados pela `r6_lib`: normal (climatologia antes de 2018), AJ, elastico5, escala5, escala6 e elastico6 |
| 3 | **Diagnóstico:** em quantas UFs × temporadas o pico real fica entre o elastico5 e o escala5 |
| 4 | Grade, validação aninhada e decisão, **antes do teste** |
| 5 | Ensemble a partir das previsões gravadas, 2018–2025, e figura da rampa |
| 6 | **Teste 2022–2025:** métricas-padrão, WIS por fase ÷ elastico5 e diagnóstico no teste |
| 7 | Gráficos das 26 UFs + Brasil |
| 8 | Temporada mais recente |

`ENS_pico7_simples_UF` é a referência fixada a priori, sem olhar dados: base elastico5, 50/50 no pico, rampa em z = 1 com h = 0,3.
