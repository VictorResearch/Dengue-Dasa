# RODADA 6 · otimização do pico: MIX_elastico e MIX_escala (14/10/2026)

**Por que existe:** na R5 (teste 2022–2025), cada modelo erra para um lado.

| Modelo | Altura do pico | Entressafra |
|---|---|---|
| MIX_elastico5 | −12% (fica baixo) | — |
| MIX_escala5 | — | +151% (infla) |

Esta rodada otimiza todos os parâmetros dos dois modelos e dá a cada um o seu próprio detector. O objetivo é direto, sem critério misto:

| Modelo | Minimiza | Restrição |
|---|---|---|
| Elástico | WIS do pico ÷ R5 | entressafra e intermediária não pioram mais que 5% |
| Escala | WIS da entressafra ÷ R5 | pico e intermediária não pioram mais que 5% |

Detalhes no doc 20 do mapa.

## Como rodar

Um notebook só, em CPU comum: **`R6_Otimizacao_Pico.ipynb`**. Leva ~45 min a 1 h na primeira vez.

1. Suba `r6_lib.py` para `MyDrive/dengue_V4/`. O `r5_lib.py` e as libs anteriores já devem estar lá.
2. Ele usa:
   - `cache_r3/` (detector_variaveis, normal e mistura_parametros.json);
   - `cache_r5/` (detector5_* e mix_elastico_w.json);
   - `resultados_uf_r5/` e `resultados_uf_r3/`;
   - `cache_r4/detector4_variaveis.parquet`, que é opcional.
3. As pastas `resultados_uf_r6`, `cache_r6` e `avaliacao_r6` são criadas sozinhas. O Optuna é instalado na célula 1.

**Retomada:** os estudos do Optuna ficam em `cache_r6/*.db`, assim como as dobras da validação aninhada e as previsões. Se cair, rode tudo de novo.

## Células

| Célula | O que faz |
|---|---|
| 1 | Setup e ajustes: `VALIDACAO` 2015–2021, `M_BUSCA` 200, `N_FINAL` 400, `N_ANINHADA` 120, `TOL` 5%, `GANHO_MIN` 2% |
| 2 | Lê dados e parâmetros da R5. **Confere** se a reimplementação reproduz a R5 gravada e estima o tempo |
| 3 | **Busca final** (Optuna) de cada modelo: tabela de parâmetros (R5 × otimizado, com o que cada um faz), razões por grupo de temporadas e fronteira pico × entressafra |
| 4 | **Validação aninhada:** a busca é refeita deixando cada temporada de fora. Decide se recomenda a versão otimizada **antes do teste** |
| 5 | Previsões finais (1.000 trajetórias), 2018–2025, em `resultados_uf_r6` (MIX_elastico6_UF e MIX_escala6_UF) |
| 6 | **Teste 2022–2025:** métricas-padrão lado a lado e razão R6 ÷ R5 por fase |
| 7 | Gráficos das 26 UFs + Brasil (R6 × R5) e PDF |
| 8 | Temporada mais recente por UF, nas duas versões |

## Como ler os números

**WIS da fase ÷ R5** = o erro probabilístico do modelo nas semanas daquela fase, dividido pelo erro da R5 na mesma UF. A média é geométrica entre UFs, então cada UF pesa igual.
- **0,85** = 15% menos erro que a R5.
- **1,05** = 5% mais erro.

As fases vêm do observado:
- **pico:** semana com incidência ≥ p75 da UF;
- **entressafra:** semana com incidência < p25 da UF;
- **intermediária:** o resto.
