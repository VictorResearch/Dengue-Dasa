# 16 · Rodada 3: detector de anos epidêmicos + mistura normal/pico (13/10/2026)

> **Pedido:** um modelo de temporada que preveja bem os picos, com três peças: modelo normal, detector que olha sinais adiantados e modelo de pico. Ver o diagnóstico no doc 15.
> **Código:** `data_imdc_2026/RODADA3/` (`r3_lib.py`, `_gerar_notebooks_r3.py`, R3_1 → R3_2 → R3_3, LEIA-ME com a ativação). Tudo roda em CPU.
> **Status:** a rodar no Colab. No PC rodou só o teste de fumaça (normal trocado pela climatologia).

## Desenho

- **Detector:**
  - **alvo:** log do tamanho da temporada menos o normal da UF;
  - **sinais:** casos e espalhamento relativos à mesma época, tendência, vizinhos, ENSO/IOD/PDO e a variação de 12 semanas, imunidade (tamanho das últimas temporadas e anos desde a última grande);
  - **cortes:** jun (SE25), jul (SE30), ago (SE34), set (SE38), out (SE40);
  - **sem vazamento:** os sinais são escolhidos com temporadas ≤ 2021. Para cada temporada, o treino usa só temporadas já terminadas no corte;
  - **métodos:** ridge, regressão quantílica linear, LightGBM quantílico e a média dos três. Por corte, fica o de menor pinball na afinação.
- **Normal:** ENS_grafos3_λ (rodada 2).
- **Mistura:**
  - `MIX_escala_UF`: limiar e reescala da trajetória;
  - `MIX_forma_UF`: limiar e forma média dos anos grandes;
  - `AJ_continuo_UF`: fração *w* do caminho até o total do detector;
  - os parâmetros são escolhidos na afinação, pelo R × climatologia.
- **Saídas:**
  - junho em `resultados_uf_r3/uf/` (entra no CompUF4/Comp5);
  - revisões em `revisoes_r3/uf/`;
  - tabelas em `avaliacao_r3/`, com carimbo.

## Fumaça (não são estimativas)

- **Detector (junho, teste 2022–2025):** correlação 0,79–0,82; sensibilidade a temporadas grandes 82–89%; precisão 50–56%. Em 2023/24: PR ×18,6 (real ×23,6), SC ×45 (real ×206), SP ×7,3 (real ×16,8).
- **Mistura sobre a climatologia:** o `AJ_continuo_UF` (*w* = 0,75) faz 0,87 × climatologia na afinação. As variantes com limiar ficam piores que 1.
