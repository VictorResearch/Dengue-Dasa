# SHAP corrigido — seleção de features do Modelo 2 sobre os bancos corrigidos

## O que tem aqui

| Arquivo | O que é |
|---|---|
| `shap_walkforward_corrigido.py` | O script. Lê `S{1..4}_corrigido/train{N}_dengue_eng_COMPLETO_spatial.parquet` e roda o SHAP walk-forward por regime (~32 min em 6 núcleos, ~6 GB de RAM) |
| `relatorio_shap_corrigido.md` | Resultado comentado: listas recomendadas, o que sai e por quê, importância por família, sondas de ruído |
| `listas_features_modelo2.py` | `COLS_FISICA`, `COLS_SECA`, `COLS_PICO` prontos para colar no notebook |
| `colunas_extras_nn.json` | Colunas das listas que não existem no `FINAL_NN` antigo. O `refazer_season.py --etapa nn_v2` as acrescenta no `FINAL_NN_v2` |
| `trechos_notebook_modelo2.py` | As 4 trocas para o `Modelo2_S1_enxuto.ipynb`: listas, arquivos v2, contexto e **inferência alinhada** |
| `importancia_detalhada_folds.csv` | SHAP e permutação de cada candidata em cada season, regime e fold |
| `importancia_por_season.csv`, `estabilidade_entre_seasons.csv` | Agregações |
| `execucao_shap.log` | Saída da execução |

## Ordem para reproduzir

```bash
cd data_imdc_2026
python refazer_season.py --season 2        # (e 3, 4; a S1 já existe em S1_corrigido/)
python SHAP_corrigido/shap_walkforward_corrigido.py
python refazer_season.py --season 1 --etapa nn_v2   # (e 2, 3, 4) -> train/test/ctx{N}_FINAL_NN_v2.parquet
```

## Diferenças para o SHAP original (`TestesomenteAGCRN.py`)

1. **O alvo é a semana seguinte.** A linha *t* explica os casos de *t+1*, igual ao treino do AGCRN (`X[t] -> Y[t+1]`). O original explicava os casos da própria semana, e isso inflava variáveis calculadas com y(t).
2. **Poisson com offset `log_pop`.** Segue a estrutura da NB do modelo. O original regredia casos absolutos com erro quadrático, e o "tamanho da cidade" dominava.
3. **Regimes do modelo.** PICO = semana-alvo ≥ 47 ou ≤ 25. O original usava 49–22.
4. **Variáveis na forma que a rede recebe.** Usa log-incidência, log da população e log1p dos acessos.
5. **Sondas de ruído mais permutação.** Uma variável só é aceita se piorar a validação, quando embaralhada, mais do que colunas aleatórias.
6. **Filtro de redundância.** Variáveis com |r| > 0,95 em relação a outra já escolhida são descartadas.
7. **Candidatas novas.** Incidência observada, sem suavizar, em t, t−1 e t−2 e nos vizinhos W1/W2, além de todos os lags climáticos 2–5.

Mantido do original: XGBoost com 100 árvores e profundidade 5, 3 folds de janela expansiva, mediana dos folds, Top 25 por season e regime, e estável = presente em ≥ 3 das 4 seasons.
