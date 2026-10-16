# RODADA 3 · detector de anos epidêmicos + mistura normal/pico (13/10/2026)

Motivo: o modelo de temporada (ENS_grafos3_λ) acerta a época do pico, mas prevê uma temporada típica. Em 2023/24, previu de 2% a 10% do real no Sul e no Sudeste. Em junho já havia sinal do tamanho; o diagnóstico está em `_MAPA_DO_PROJETO/15_modelos_por_dentro_e_picos.html`.

É **complementar**: não altera nada das rodadas 1 e 2. O código está em `r3_lib.py` e os notebooks são gerados por `_gerar_notebooks_r3.py`. **Tudo roda em CPU.**

## Como ativar

1. No Drive (`MyDrive/dengue_V4/`):
   - suba `r3_lib.py` e confira que o `r2_lib.py` mais recente está lá;
   - crie à mão as pastas `resultados_uf_r3`, `revisoes_r3`, `cache_r3` e `avaliacao_r3`.
2. Rode os notebooks em ordem. Cada um depende do anterior.

| Ordem | Notebook | O que faz | Tempo |
|---|---|---|---|
| 1 | `R3_1_Detector` | Variáveis e alvo de cada UF × temporada × corte (jun, jul, ago, set, out). Escolhe os sinais com temporadas ≤ 2021. Treina 3 métodos + a média e escolhe, por corte, o de menor pinball na afinação. Mostra alarmes e o que foi previsto em junho para cada UF | 10–20 min |
| 2 | `R3_2_Mistura` | Recria o normal (ENS_grafos3_λ). Escolhe na afinação os parâmetros de `MIX_escala_UF`, `MIX_forma_UF` e `AJ_continuo_UF`. Grava junho em `resultados_uf_r3/uf/` e as revisões em `revisoes_r3/uf/` | 20–40 min |
| 3 | `R3_3_Avaliacao_Graficos` | Compara no teste, contra o BB, o resultado geral, o pico, a entressafra, o total e a cobertura. Mede o ganho das revisões mensais. Gera gráficos das 26 UFs + Brasil, PDF e totais | 10–20 min |

**Retomada:** se a conexão cair, rode tudo de novo. Variáveis, detector (um arquivo por método × corte), parâmetros, misturas (um arquivo por modelo × temporada) e PNGs ficam gravados, e o que já existe é pulado.

**Sinais externos (opcional):** coloque em `cache_r3/externos_r3.csv` as colunas `uf_code`, `season` e (opcional) `corte`, mais uma coluna por sinal. Exemplos: previsão de ENSO do IRI, sorotipo, vacinação, positividade de exames da DASA. Depois apague `cache_r3/detector_*` e `cache_r3/sinais_detector.json` e rode o R3_1 de novo.

## As três peças

- **① Detector:** prevê o log do tamanho da temporada **relativo ao normal da UF** (média das temporadas já terminadas). Sinais usados:
  - casos e espalhamento das últimas 4 a 13 semanas, comparados com a mesma época dos anos anteriores;
  - tendência de 8 semanas;
  - vizinhos e macrorregião;
  - ENSO, IOD e PDO, com a variação de 12 semanas;
  - tamanho das últimas temporadas e anos desde a última temporada grande.
  
  O detector devolve 9 quantis do tamanho e P(grande) = P(> 2× o normal).
- **② Normal:** ENS_grafos3_λ da rodada 2, sem mudança.
- **③ Pico e mistura:** são sorteadas 1.000 trajetórias do normal e 1.000 tamanhos do detector. Há três variantes:
  - `MIX_escala_UF`: quando o tamanho passa do limiar, reescala a trajetória;
  - `MIX_forma_UF`: usa a forma média dos anos grandes;
  - `AJ_continuo_UF`: aproxima toda trajetória uma fração *w* do total do detector.

**Revisões:** o detector roda de novo em SE30, SE34, SE38 e SE40. O normal continua o de junho. Como as revisões não seguem a regra oficial, ficam fora de `resultados_uf*`.

## Teste de fumaça (PC, 13/10/2026)

No teste, o normal foi substituído pela climatologia. Os números não são estimativas.

- **Detector:** sinais escolhidos só até 2021, corte de junho.
  - Correlação da mediana com o real no teste 2022–2025: 0,79–0,82. Sensibilidade a temporadas grandes: 82–89%. Precisão: 50–56%.
  - Em 2023/24, previu PR ×18,6 (real ×23,6), SC ×45 (real ×206) e SP ×7,3 (real ×16,8).
  - As revisões de setembro foram piores que as de junho na afinação. Vale olhar com cuidado no Colab.
- **Mistura sobre a climatologia:** o `AJ_continuo_UF` (*w* = 0,75) baixou o R × climatologia de 1,00 para 0,87 na afinação. As variantes com limiar pioraram (1,06–1,10).
- O fluxo completo do R3_3 (tabelas, revisões, gráficos e totais) rodou sem erro.
