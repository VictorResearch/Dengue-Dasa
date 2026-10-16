# RODADA 5 · modelo final de temporada: um especialista para cada fase (14/10/2026)

**Por que existe:** na rodada 4 (teste 2022–2025), nenhum modelo foi bom na temporada inteira.

| | Entressafra | Pico |
|---|---|---|
| Melhor modelo | normal (ENS_grafos3_λ) | MIX_escala |
| WIS da fase | 23,6 | 2.136 (normal: 4.065) |
| Viés da fase | +49% (MIX_escala: +156%) | −36% (normal: −82%) |

Além disso, o detector ampliado errou 2023 porque usou uma variável nacional do Atlântico norte, que bateu recorde naquele ano. Detalhes no doc 18 §6 e no doc 19 do mapa.

## Como rodar

Um notebook só, em CPU comum (High-RAM se puder): **`R5_Modelo_Final_Temporada.ipynb`**. Leva 40–90 min na primeira vez.

1. Suba `r5_lib.py` para `MyDrive/dengue_V4/`.
2. Ele usa o que as rodadas 3 e 4 gravaram: `cache_r3/`, `cache_r4/` (`detector4_variaveis.parquet` e `ckpt/`), `resultados_uf_r3/` e `resultados_uf_r4/`.
3. As pastas `resultados_uf_r5`, `cache_r5` e `avaliacao_r5` são criadas sozinhas.

**Retomada:** se cair, rode tudo de novo. O que já foi gravado é pulado: detector, membros (um arquivo por modelo × temporada), validações e PNGs.

## Etapas (células)

| Célula | O que faz |
|---|---|
| 2 | **Detector corrigido.** Candidatos: o da R3, o ampliado **sem variáveis nacionais** (sem ENSO, IOD, PDO, mar e sorotipo nacional) e a média dos dois. A escolha é pelo pinball nas temporadas ≤ 2021, com os anos grandes valendo 3× e as mesmas linhas para todos |
| 3 | **Membros com o detector corrigido:** MIX_escala5, MIX_elastico5, DoisEstagios_det5 e MTGNN_cond5. A MTGNN_cond5 só refaz a previsão com as redes já treinadas na R4; se faltar algum peso, fica de fora |
| 4 | **Fase esperada de cada semana × UF**, pelo nível previsto em junho contra os p25/p75 da UF (até 2017), e o **regime** (P(ano grande)) |
| 5 | **Validação na afinação** deixando uma temporada de fora, para 4 ensembles × com/sem calibração. Escolhe o **recomendado** antes de olhar o teste |
| 6 | **Ajuste final** dos pesos por fase × regime e da calibração por fase; grava as previsões |
| 7 | **Teste** 2022–2025: régua oficial, critério misto, fases, pico, total e cobertura |
| 8 | **Gráficos** das 26 UFs + Brasil (PDF e totais) |
| 9 | **Temporada mais recente** por UF: total, semana e altura do pico, P(ano grande) |

## Ensembles

- **`ENS_fase_UF`:** pesos dos membros por fase esperada × regime, escolhidos pelo critério misto ((1 − p)·R + p·WIS absoluto; p = `R4.PESO_ABS` = 0,5). Os pesos são suavizados em 5 semanas.
- **`ENS_troca_UF`:** em cada fase × regime, usa só o melhor membro. É a "alternância" pura entre modelos.
- **`_nivMIX`:** a fase esperada vem do MIX_escala5, que enxerga os anos grandes. Sem o sufixo, ela vem do normal.
- **`+cal`:** largura (×0,8–1,5) e posição (×0,8–1,5) calibradas dentro de cada fase. Subir o pico não infla a entressafra.

## Teste de fumaça (PC)

As 9 células rodaram sem erro no PC, com o normal trocado pela climatologia e o mar e o sorotipo sintéticos. Os números estão no doc 19 §Teste de fumaça e não são estimativas. Achado: o filtro de variáveis nacionais da R4 nunca funcionou; aqui ele foi corrigido.
