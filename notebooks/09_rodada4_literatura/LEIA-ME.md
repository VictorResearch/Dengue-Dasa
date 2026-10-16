# RODADA 4 · detector ampliado, forma que depende do tamanho, MTGNN condicionado e ensemble por regime (13/10/2026)

**Por que existe:** a rodada 3, no teste 2022–2025, mostrou três coisas (detalhes em `_MAPA_DO_PROJETO/18_RODADA4_LITERATURA_APLICADA.md`):

- O `MIX_escala_UF` acertou a **altura dos picos**:

  | | MIX_escala_UF | Normal |
  |---|---|---|
  | Erro mediano da altura | −4% | −42% |
  | WIS absoluto | 776 | 1.343 |
  | Picos dentro do IC90 | 91% | — |

  Mesmo assim, perdeu na régua R × BB (1,19), porque inflou a entressafra (+156%).
- A recomendada da R3 (`AJ_continuo_UF`, R 0,83) continua abaixo do real: −25% no total e −28% na altura do pico.
- O detector acertou a direção (correlação 0,8), mas encolhe os extremos. Em 2023, previu SC em ×45 (real ×206).

É **complementar**: não altera nada das rodadas 1 a 3. O código está em `r4_lib.py` e os notebooks são gerados por `_gerar_notebooks_r4.py`.

## Como ativar

1. No Drive (`MyDrive/dengue_V4/`), suba `r4_lib.py` e confira que `r3_lib.py` e `r2_lib.py` estão lá. As pastas `resultados_uf_r4`, `revisoes_r4`, `cache_r4` e `avaliacao_r4` são criadas sozinhas.
2. Rode na ordem:

| Ordem | Notebook | Ambiente | Tempo | O que faz |
|---|---|---|---|---|
| 0 (opcional) | `R4_0_Sorotipo_SINAN` | CPU | 30–60 min (uma vez) | Baixa o SINAN 2010–2026 do OpenDataSUS e conta os casos com sorotipo por UF × semana |
| 1 | `R4_1_Detector_ampliado` | CPU | 20–40 min | Temperatura do mar (ERSST v5) e variáveis novas; 4 métodos + média + versões conformes; escolha nas temporadas ≤ 2021 com os anos grandes pesando 3×; comparação com o detector da R3 |
| 2 | `R4_2_Mistura_DoisEstagios` | CPU | 20–40 min | `MIX_elastico_UF` (jun + revisões jul–out) e `DoisEstagios_det_UF` |
| 3 | `R4_3_MTGNN_condicionado` | **GPU L4** | 2–4 h | `MTGNN_cond_UF` (tamanho do detector como entrada + anos grandes sintéticos) e `MTGNN_aug_UF` |
| 4 | `R4_4_Ensemble_Avaliacao` | CPU | 15–30 min | Ensemble por faixa de P(ano grande) e calibração, validados deixando uma temporada de fora; régua no teste; gráficos das 26 UFs + Brasil |

**Em paralelo:** depois do R4_1, o R4_2 (CPU) e o R4_3 (GPU) podem rodar ao mesmo tempo, em sessões diferentes. O R4_4 vem por último. Se o R4_3 ainda não terminou, o R4_4 roda com os membros que existirem; é só rodá-lo de novo depois.

**Retomada:** se a conexão cair, rode tudo de novo. Ficam gravados e são pulados:
- o mar e o sorotipo (por ano);
- as variáveis e o detector (por método × corte);
- os json dos parâmetros;
- as previsões (por modelo × temporada);
- os pesos das redes (a cada 5 épocas);
- os PNGs.

**Para refazer:**
- **detector:** apague `cache_r4/detector4_*` e `cache_r4/sinais4.json` (por exemplo, depois de rodar o R4_0);
- **w do MIX_elastico:** apague `cache_r4/mix_elastico_w.json`;
- **MIX_elastico e DoisEstagios:** apague os arquivos deles em `resultados_uf_r4/uf/`.

**Peso do pico (`R4.PESO_ABS`, célula 1):**
- O critério misto é (1 − p) × R contra a referência + p × WIS absoluto relativo.
- O padrão é p = 0,5. Suba (por exemplo, para 0,7) para priorizar a altura dos picos; desça para 0 para usar só a régua R do IMDC.
- Mudou o p? Apague o `mix_elastico_w.json` e rode R4_2 e R4_4 de novo.

**Sinais externos (opcional):** coloque em `cache_r4/externos_r4.csv` as colunas `uf_code`, `season` e (opcional) `corte`, mais uma coluna por sinal. Exemplos: vacinação, positividade de exames. Cada valor deve ser o que se sabia **naquele corte**.

## As peças

- **① Detector ampliado.** Variáveis novas, além das da R3:
  - estabilidade entre surtos e sinais de alerta precoce (Rypdal 2019; Brett 2020);
  - imunidade com esquecimento (McGough 2021);
  - seca tipo SPEI (Lowe 2021);
  - clima observado;
  - SEAS5 já emitida;
  - temperatura do mar (IOBW, Atlântico tropical norte e sul, Atlântico sudoeste, Niño 3.4; Chen 2024, Beal 2025);
  - sorotipo, se o R4_0 rodou (Finch 2025).

  Métodos: ridge, regressão quantílica, LightGBM quantílico, **duas etapas** (P(grande) e depois o tamanho; Francisco 2024), a média e as versões **conformes** (Mills 2025).
- **② MIX_elastico_UF.** O normal (ENS_grafos3_λ) vai até o total do detector, mas cada semana × UF cresce conforme a sua **elasticidade** ao tamanho da temporada. A elasticidade foi medida no histórico por nível de incidência: ~0,1 nas semanas quase sem casos e ~1 no pico. É a correção do defeito do `MIX_escala_UF`.
- **③ DoisEstagios_det_UF** (retreino do item 1): o tamanho vem do detector; a forma, de temporadas passadas **parecidas em tamanho** (da UF e da macrorregião), com deslocamento de ±2 semanas e ruído binomial negativo.
- **④ MTGNN_cond_UF** (retreino na GPU). É a MTGNN_pesopop com:
  - o tamanho relativo da temporada como entrada;
  - 20% de cópias sem condição;
  - 60% de cópias com a temporada futura multiplicada por 2 a 8 (Al Bannoud 2026).

  Na previsão, usa 24 cenários do detector. `MTGNN_aug_UF` é a mesma rede sem condição.
- **⑤ ENS_regime_UF(+cal).** Pesos por faixa de P(ano grande) (< 0,35, 0,35–0,65, ≥ 0,65) e calibração por faixa (largura ×0,8–1,5; posição ×0,8–1,5).
  - Tudo é escolhido na afinação pelo critério misto.
  - A **recomendada** é o modelo de menor critério na afinação, com os ensembles medidos fora da amostra (deixando uma temporada de fora). Ela é decidida antes de olhar o teste.

**Temporada rolante (EAKF):** ficou de fora. Na data oficial (SE25) ainda não há nenhum dado da temporada que começa na SE41, então ela não melhora a previsão de junho. Depois de outubro, quem atualiza é o semanal (já bom: h = 1 em 0,224) e as revisões mensais do `MIX_elastico`.

## Teste de fumaça (PC, 13/10/2026)

Nesse teste, o normal é a **climatologia**, e a temperatura do mar e o sorotipo são **sintéticos**. Os números servem para conferir o código e não são estimativas.

- O parser do SINAN leu um pedaço real do CSV de 2018, lido em memória.
- **Detector:** com a escolha que pesa os anos grandes, ficou a `media` em junho. Em 2023, previu SC ×42 (q95 ×304; real ×206) e RS ×31 (q95 ×199; real ×160). Escolhido só pelo pinball da afinação, o LightGBM dava SC ×9.
- **Elasticidade por nível:** a entressafra do MIX_elastico ficou 30 pontos acima do normal, contra 82 pontos na versão por semana.
- **Ensembles:** a validação deixando uma temporada de fora mostrou que, com membros fracos, o ensemble por regime e a calibração pioram fora da amostra. Por isso a recomendada sai dessa validação, e não do ajuste dentro da amostra.
- **MTGNN_cond:** com 1 época e 1 semente em CPU, treinou em 48 s e gerou as saídas `MTGNN_cond_UF` e `MTGNN_aug_UF` (52 semanas × 26 UFs × 9 quantis).
