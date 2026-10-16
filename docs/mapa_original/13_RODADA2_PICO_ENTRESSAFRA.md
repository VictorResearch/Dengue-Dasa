# 13 · Rodada 2: pico, entressafra e otimização (11/10/2026)

> **Pedido:** testar as melhorias da reflexão (tópicos A–E) para que os modelos prevejam bem no pico e na entressafra, como algo **complementar**: o histórico de tabelas fica e as novas análises ficam salvas ao lado.
> **Código:** `data_imdc_2026/RODADA2/` (`r2_lib.py`, `_gerar_notebooks_r2.py`, 5 notebooks e o LEIA-ME com a ativação passo a passo).
> **Status:** a rodar no Colab. No PC foi feito só um teste de fumaça (ver abaixo).

## Diagnóstico de partida (rodada semanal, 11/10)

- **Semanal:** WAPE de 13%, IC90 de 91%. O erro se concentra nas viradas (o h = 4 tem WAPE de 30%).
- **Temporada:**
  - as medianas ficaram cerca de 75% abaixo do total real (por causa de 2024) e o IC90 foi de 82%. O problema é a **cauda de cima**;
  - funcionaram o alvo em crescimento (no semanal), o peso pela população (na temporada), os ensembles, o λ e o espaço;
  - não funcionaram as covariáveis, o TFT/NHITS/Chronos sem ajuste, o modelo de temporada usado na semana e a escolha por porte.

## Plano em 5 passos

| Passo | Tópicos | Notebook |
|---|---|---|
| 1 | A (métricas por fase e de pico), D2 (calibração por horizonte e fase), E4 (pesos de ensemble) | R2_1 (rodar agora e no fim) |
| 2 | B3 (fase/Rt), C (tsfresh com seleção FRESH só na afinação), D1 (peso no pico), mediana L1 | R2_2 |
| 3 | B1 (escala por UF), B2 (pré-treino nas 435 regiões), D1 (quantis altos), D3 (cabeça do total) | R2_3 |
| 4 | E1 (Optuna: GBM semanal e MTGNN de temporada, só na afinação), E2 (10 sementes) | R2_4 |
| 5 | E3 (Chronos-2 ajustado com LoRA) | R2_5 |

**Definições:**
- **Fases:** limiares por UF calculados só com dados até 2017. Entressafra = abaixo do p25; alta = acima do p75.
- **Erro de pico:** diferença entre a semana do máximo previsto e a do observado, mais o erro relativo da altura, por UF × temporada (ou rodada).
- **D2:**
  - largura λ e deslocamento δ da mediana, em log1p, escolhidos por balde na afinação;
  - os baldes são h × fase na origem (semanal) e o bloco de antecedência 16–29/30–42/43–55/56–68 (temporada);
  - o δ ataca diretamente a subestimação.
- **D3:** perda pinball no log1p do total das semanas 16–67 a partir da SE25, somando as trajetórias de cada quantil.

## Resultados (R2_1 de 12/10/2026 17:22–17:26, teste 2022–2025)

Tabelas copiadas em `data_imdc_2026/RODADA2/resultados_r2_final/`; relatório para a DASA em `14_relatorio_rodada2_dasa.html`.

**Semanal:**
- **h = 1:** sem ganho. ENS_GBM_ARIMA_MTGNNcresc_sem 0,224 × climatologia, MAE 267, IC90 91,9%. O antigo ENS_GBM_ARIMA_sem também fez 0,224.
- **h = 2–4:** ENS_stack_sem (E4) é o melhor. Em h = 4: 0,404 contra 0,409; MAE 583 contra 648; viés −12,5% contra −20%.
- **GBM:** fase (0,227, MAE 291) e tsfresh (0,227, MAE 282) ajudam. Pioram o peso no pico (0,235), a mediana L1 (0,234) e o lgb r2full (0,241). O Optuna é pior em h = 1 e melhor em h = 2–4; com tsfresh, faz 0,405 em h = 4.
- **MTGNN_cresc (0,238):** escala 0,231, qalto 0,230, pré-treino 0,230 (melhor na entressafra: 0,157), fase 0,233, fase + tsfresh 0,233.
- **Sem ganho:** ARIMAX igual ao ARIMA (0,233 contra 0,232); Chronos2FT igual ao Chronos2 (0,252 contra 0,253).

**Temporada (× BB):**
- **Melhor geral:** ENS_grafos3_λ 0,831 (redes de grafo com λ + MTGNN_optuna10_UF).
- **Variantes contra MTGNN_pesopop_UF sem λ (0,889):** só o Optuna ganha (0,876). Ficam iguais ou piores: 10 sementes 0,890, fase 0,913, total 0,921, fase + tsfresh 0,922, r2full 0,928, escala 0,937, qalto 0,946, pré-treino 0,958. Todas melhoram o pico (WIS log de 0,77 para 0,66–0,70).
- **Pico entre os que vencem o BB:** ENS_stack_UF tem WIS log 0,524 e IC90 82% (BB: 0,693; ENS_grafos3: 0,801 e 61%).
- **Calibração (+cal):** piora o R (1,2–1,6), mas corrige o pico e o viés. MTGNN_total_pp_UF+cal erra o total da temporada em −1%; os melhores gerais erram em −37%.
- **Entressafra:** a melhor nossa é Chronos2FT_UF+λ (0,310).

**Gráficos:** o `R2_7_Graficos_UF` grava em `avaliacao_r2/graficos_r2_<data>/` um PNG por UF + Brasil, um PDF e `totais_temporada.csv`. Ele recria os ensembles a partir dos pesos e do λ gravados pelo R2_1.

## R2_6 · variáveis da rodada 2 em todos os modelos (11/10)

As variáveis do R2_2 antes só entravam nos GBMs semanais. O R2_6 as leva aos outros modelos, mudando uma coisa por vez:

| Notebook | Modelo novo | Compara com |
|---|---|---|
| 6a (GPU) | MTGNN_cresc_fase_sem, MTGNN_cresc_fasetsf_sem | MTGNN_cresc_sem |
| 6a (GPU) | GBM_xgb_optuna_tsf_sem | GBM_xgb_optuna_sem |
| 6a (GPU) | MTGNN_fase_pp_UF, MTGNN_fasetsf_pp_UF | MTGNN_pesopop_UF |
| 6b (CPU) | ARIMAX_fase_sem (log Rt, ΔRt, nível − p75, defasados 4 semanas) | ARIMA_Fourier_sem |

- As covariáveis são padronizadas só com dados até o corte (`R2.cov_r2`) e entram nas redes por `cfg['cov_fn']`.
- Depende do R2_2 (cache do tsfresh) e do R2_4 (melhores hiperparâmetros do GBM). O 6a e o 6b podem rodar juntos.
- O ARIMAX não tem versão de temporada: exigiria defasagem de 68 semanas.
- **Correção do R2_5:** o Colab traz torchao 0.10 e o peft recusa versões < 0.16. O notebook agora desinstala o torchao e o `r2_lib` (`_sem_torchao_velho`) desliga a checagem mesmo numa sessão já aberta.

**Pastas novas:** `resultados_semanal_r2`, `resultados_uf_r2`, `cache_r2`, `avaliacao_r2`. As tabelas antigas (`resultados_semanal_avaliacao/`) não mudam.
