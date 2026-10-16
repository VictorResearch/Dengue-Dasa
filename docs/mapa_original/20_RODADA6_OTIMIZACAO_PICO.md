# 20 · Rodada 6: otimização do pico do MIX_elastico e do MIX_escala (14/10/2026)

**Pedido:** otimizar todos os parâmetros dos dois modelos para que:
- o **MIX_elastico** suba no pico (na R5: altura −12%, total −8%);
- o **MIX_escala** pare de inflar a entressafra (na R5: +151%).

O pedido inclui:
- um detector para cada modelo;
- Optuna;
- um objetivo sem o critério misto;
- uma validação bem planejada.

O inventário de parâmetros está no [doc 19 §Botões](19_RODADA5_MODELO_FINAL.md).

**Onde:** `data_imdc_2026/RODADA6/`
- `r6_lib.py`
- `_gerar_notebook_r6.py` → `R6_Otimizacao_Pico.ipynb`
- `LEIA-ME.md`

**Ambiente:** CPU, ~45 min a 1 h. Usa os caches das R3, R4 e R5. A R4 é opcional.

## Desenho

1. **Reimplementação com todos os parâmetros abertos.** Classe `Base6`, vetorizada em UFs × trajetórias, com números aleatórios comuns a todas as tentativas. Os valores padrão reproduzem a R5. A conferência no PC, com 1.000 trajetórias, deu diferença de 1–4% na mediana contra os arquivos da R5, que é ruído de simulação.
   - **MIX_elastico:** `w`, faixas de nível (padrão ou fino), elasticidade só dos anos acima do normal, teto de b, expoente γ em b, afinação da forma η nos anos grandes, antecipação do pico.
   - **MIX_escala:** limiar, troca suave τ, expoente p, nível L e teto do fator nas semanas abaixo de L.
2. **Um detector para cada modelo.**
   - `det_modo` = `R5` (o R3:ridge salvo) ou `refit`.
   - O `refit` é um ridge reajustado de forma causal: as variáveis são escolhidas por Spearman só com temporadas terminadas antes do corte. Os parâmetros dele são:
     - fonte R3 ou R3 + R4 (sem as nacionais novas da R4; as nacionais da R3 só entram com |ρ| ≥ 0,30);
     - janela do normal: todas as temporadas, 5 ou 8;
     - k, ρ mínimo, α e peso dos anos grandes.
   - Nos dois modos entram o alargamento, a correção de viés β (`+β·max(mediana, 0)`) e a cauda do sorteio.
3. **Objetivo.** A métrica é o WIS da fase dividido pelo WIS da R5 na mesma UF, com média geométrica entre UFs. As fases vêm do observado (p25/p75 até 2017). Nos dois modelos, a restrição é que as outras fases não piorem mais que `TOL = 5%`.

   | Modelo | Fase a minimizar | Fases com restrição |
   |---|---|---|
   | MIX_elastico | pico | entressafra e intermediária |
   | MIX_escala | entressafra | pico e intermediária |

   O Optuna usa o TPE multivariado com `constraints_func`. A primeira tentativa é a própria R5, e a escolha final é a melhor tentativa viável.
4. **Validação.**
   - **Busca em 2015–2021.** Até 2017 o normal é a climatologia; de 2018 em diante, o ENS_grafos3_λ. A R5 de referência é calculada no mesmo normal.
   - **Validação aninhada:** a busca é refeita 7 vezes, deixando uma temporada de fora em cada uma.
   - **Regra de uso, decidida antes do teste:** fora da amostra, a fase-alvo precisa cair pelo menos 2% (`GANHO_MIN`) e as restrições precisam ser respeitadas.
   - **Teste 2022–2025:** uma vez só, com as métricas-padrão (R × BB, WIS por fase, viés, MAE, WAPE, sMAPE, pico, total e IC90) e a razão R6 ÷ R5 por fase.
5. **Saídas:**
   - `resultados_uf_r6/`: MIX_elastico6_UF e MIX_escala6_UF, 2018–2025;
   - `cache_r6/`: estudos `.db`, `melhor_*.json` e `aninhada_*.json`;
   - `avaliacao_r6/`: tentativas, fronteira pico × entressafra, validação aninhada, teste, gráficos das 26 UFs + Brasil e temporada mais recente.

## Achado no desenho

**O detector R3:ridge (o da R5) só existe a partir de 2018.** As variáveis de imunidade precisam de 3 temporadas anteriores, e o ridge da R3 descartava as linhas com falta. Por isso, nas temporadas de 2015–2017:
- o modo "R5" usa o ridge reajustado com os parâmetros da R3;
- o reajuste preenche a falta com a mediana do treino, em vez de descartar a linha.

Sem isso, os anos grandes de 2015 e 2016 ficariam sem detector na validação.

## Teste de fumaça (PC, 14/10; normal = climatologia, caches falsos; NÃO são estimativas)

- As 8 células rodaram sem erro com Optuna 5.0.0.
- A retomada funcionou: na segunda execução, tudo foi pulado.
- Custo por tentativa com M = 200 e 7 temporadas: elástico 1,5 s, escala 0,13 s.
- Configuração do Colab: `N_FINAL = 400` e `N_ANINHADA = 120`.

## Resultados (Colab, 13/10; tabelas em `avaliacao_r6/`)

**Parâmetros escolhidos.** Nenhum dos dois escolheu o detector reajustado: ambos ficaram com o R3:ridge da R5. O `w` da R5 era 0,5.

| | MIX_elastico6 | MIX_escala6 |
|---|---|---|
| Detector | alargamento 1,35; **β −0,11 (desce os anos grandes)**; cauda 3,4 | alargamento 1,17; β +0,12 |
| Mistura | w 0,63; teto de b 1,21; **γ 0,51** (espalha o crescimento para fora do pico); η ≈ 1; antecipação 0,13 | limiar 1,38 (×4); τ 0,37; **p 0,41**; L 2,63; teto ×26 |

**Validação aninhada (fora da amostra, 2015–2021; WIS ÷ R5):**

| Modelo | Pico | Entressafra | Intermediária | Recomendado? |
|---|---|---|---|---|
| MIX_elastico6 | 0,927 | 0,962 | **1,075** | não (a intermediária estourou o limite de 1,05) |
| MIX_escala6 | **1,137** | 0,859 | 0,802 | não (o pico estourou o limite) |

**Teste 2022–2025 (olhado uma vez):**

| Modelo | R × BB | UFs > BB | WIS | WIS pico | WIS entressafra | Viés na entressafra | MAE | WAPE | sMAPE | Altura do pico | Total | Pico no IC90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MIX_elastico6 | 0,947 | 16 | 876 | 2.551 | 31,2 | +95% | 1.499 | 68,4% | 69,8 | −9,5% | −2,2% | 84% |
| MIX_elastico5 | 0,859 | 21 | 1.010 | 2.994 | 27,8 | +85% | 1.506 | 68,8% | 70,2 | −11,8% | −7,8% | 74% |
| MIX_escala6 | **0,850** | 21 | 1.048 | 3.122 | 31,0 | +88% | 1.531 | 69,9% | **65,8** | −21,1% | −17,0% | 76% |
| MIX_escala5 | 1,288 | 4 | **746** | **2.012** | 55,1 | +151% | **1.279** | **58,4%** | 70,4 | +11,5% | +8,6% | 93% |
| ENS_grafos3_λ | 0,831 | 20 | 1.343 | 4.065 | **23,6** | +49% | 1.771 | 80,8% | 75,3 | −41,8% | −36,8% | 67% |
| AJ_continuo | **0,828** | 23 | 1.285 | 3.883 | 26,4 | +74% | 1.672 | 76,3% | 70,0 | −28,2% | −24,6% | 64% |
| BB | 1 | — | 1.224 | 3.651 | 39,0 | +108% | 1.787 | 81,6% | 73,0 | −36,3% | −23,5% | 73% |

R6 ÷ R5 por fase no teste (média geométrica entre UFs):

| Modelo | Pico | Entressafra | Intermediária |
|---|---|---|---|
| Elástico | 0,946 | 1,141 | 1,163 |
| Escala | 0,960 | 0,529 | 0,480 |

**Leitura:**
1. **A otimização andou ao longo da fronteira, não além dela.** Toda configuração que melhora o pico piora as outras fases, e vice-versa. A validação aninhada previu isso, e a regra (decidida antes do teste) rejeitou as duas versões.
2. **O MIX_elastico6 melhorou o WIS do pico (−15%) quase só alargando a faixa.** A altura mediana mal mudou (−9,5% contra −11,8%). A correção de viés para cima não generalizou: o Optuna escolheu β negativo.
3. **O MIX_escala6 virou outro modelo:** um "elástico".
   - Com o fator encolhido (p 0,41) e o limiar ×4, ele desinflou a entressafra (viés de +151% para +88%) e ganhou na régua R (1,29 → 0,85).
   - Mas passou a subestimar o pico (−21%), e o WIS absoluto do pico subiu 55%.
4. **Para o ensemble:** as versões da R6 ficaram as duas abaixo no pico (−9,5% e −21%), então não se compensam. O par que se compensa no pico é o da R5 (−12% e +12%).

