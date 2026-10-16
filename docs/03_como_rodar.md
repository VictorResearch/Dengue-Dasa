# 03 · Como rodar: ambiente, ordem e dependências

## 1. Ambiente

| Parte | Onde | Requisitos |
|---|---|---|
| Download e processamento dos dados (scripts 00–04) | qualquer PC | Python 3.10+, ~16 GB de RAM (o `climate.csv` tem 1,7 GB), ~10 GB de disco |
| Modelos estatísticos, GBMs, detector, misturas, ensembles, avaliação | CPU (Colab "High-RAM" ou PC) | `requirements.txt` |
| Redes de grafo (MTGNN, A3T-GCN, GCNGRU…), Chronos-2, neuralforecast | **GPU** (Colab L4 ou A100) | `requirements.txt`; algumas pastas pedem `requirements-completo.txt` |
| INLA (BB e GHR da competição) | Colab (Linux) | R ≥ 4.3 com `INLA` e `forecast`; o notebook instala |

```bash
pip install -r requirements.txt
```

## 2. A área de trabalho

Todos os notebooks esperam uma **pasta plana** com as bibliotecas e os dados:

```
dengue_V4/
  v4_lib.py  comp_lib.py  comp_modelos.py  uf_lib.py  semanal_lib.py  r2_lib.py … r7_lib.py  prod_lib.py …
  dados/              (= dados/v4)
  dados_uf/           (= dados/uf)
  referencias_imdc3/  (= dados/referencias_imdc3)
  parametros/         (= producao/parametros)
```

Ela é montada por:

```bash
python scripts/04_montar_area_de_trabalho.py
```

- **No Google Colab (como a pesquisa foi feita):** suba `area_de_trabalho/` para o Google Drive como `MyDrive/dengue_V4/`. A 1ª célula de cada notebook monta o Drive, copia o código para o disco da sessão e grava os resultados de volta no Drive (`MyDrive/dengue_V4/resultados_*`, `cache_*`, `avaliacao_*`).
- **Localmente:** abra o Jupyter dentro de `area_de_trabalho/`. Sem o Colab, a 1ª célula usa a pasta atual como área de trabalho.

Os notebooks de pesquisa usam os arquivos `.py` da etapa e das anteriores (ex.: a rodada 7 importa `r7_lib`, `r6_lib`, `r5_lib`…). O script 04 copia todos.

**Retomada:** todos os notebooks gravam cada resultado assim que fica pronto (`.tmp` → renomeia) e pulam o que já existe. Se a sessão cair, rode de novo do início.

**Notebooks e geradores:** cada pasta de `notebooks/` tem o `.ipynb` e o script `_gerar_*.py` que o gera. Para alterar um notebook, edite o gerador e rode `python _gerar_*.py` (o `.ipynb` é sobrescrito).

## 3. Ordem da pesquisa e dependências

Cada etapa lê o que as anteriores gravaram na área de trabalho. Tempos aproximados no Colab.

| # | Pasta | Notebooks (ordem) | Ambiente | Tempo | Precisa de |
|---|---|---|---|---|---|
| 1 | `01_agcrn_municipal` | `Modelo2_S1_enxuto` → `Modelo2_v2` → `Modelo2_v3` | GPU | 1–2 h cada | `scripts/01_refazer_municipal.py --season 1` (e `--etapa nn_v2`), arquivos em `MyDrive/dengue_S1/` |
| 2 | `02_v4_temporada` | `Modelo_v4` | GPU L4 | 1–6 h | área de trabalho |
| 3 | `03_competicao` | `Comp1_CPU` ∥ `Comp2_INLA` ∥ `Comp3_GPU` → `Comp4_Avaliacao` → `Comp5_Negocio` | CPU / R / GPU | 1–4 h cada | área de trabalho |
| 4 | `04_modelos_uf` | `CompUF1` ∥ `CompUF2` ∥ `CompUF3` → `CompUF4` (→ `Comp5` de novo) | CPU / GPU | 1–12 h | etapa 3 (`comp_*`) |
| 5 | `05_comparacao_felipe` | `CompFelipe` + scripts locais | CPU | 1 h | etapa 4; repositório externo do Felipe para `replica_felipe.py` |
| 6 | `06_semanal` | `Semanal1_CPU` ∥ `Semanal2_GPU` → `Semanal3_Avaliacao` | CPU / GPU | 2–8 h | etapa 4 |
| 7 | `07_rodada2_pico_entressafra` | `R2_1` (agora) → `R2_2` ∥ `R2_3` ∥ `R2_4` ∥ `R2_5` → `R2_6a` ∥ `R2_6b` → `R2_1` (de novo) → `R2_7` | CPU / GPU | 1–6 h cada | etapas 4 e 6 |
| 8 | `08_rodada3_detector` | `R3_1` → `R3_2` → `R3_3` | CPU | 30–60 min | etapa 7 (normal `ENS_grafos3_λ`) |
| 9 | `09_rodada4_literatura` | `R4_0` (opcional) → `R4_1` → `R4_2` ∥ `R4_3` → `R4_4` | CPU / GPU | 20 min–4 h | etapa 8 |
| 10 | `10_rodada5_modelo_por_fase` | `R5_Modelo_Final_Temporada` → `R5_Graficos_Comparacao` | CPU | 40–90 min | etapas 8 e 9 |
| 11 | `11_rodada6_optuna` | `R6_Otimizacao_Pico` | CPU | 45–60 min | etapas 8 e 10 |
| 12 | `12_rodada7_ensemble_pico` | `R7_Ensemble_Pico` | CPU | 5–10 min | etapas 8, 10 e 11 |

`∥` = podem rodar ao mesmo tempo, em sessões separadas.

**Atalho para reproduzir só os campeões:** a pasta `producao/` **não depende** das etapas de pesquisa. Os parâmetros que a pesquisa escolheu estão em `producao/parametros/`, e o `Producao_*` tem uma opção de backtest que refaz o teste 2022–2025 com o mesmo código (ver [06_producao.md](06_producao.md)).

## 4. Onde ficam os resultados

| Pasta (na área de trabalho) | Quem grava | Conteúdo |
|---|---|---|
| `resultados_v4/` | etapa 2 | previsões e resumo do V4 |
| `resultados_competicao/` | etapa 3 | previsões por região e UF, avaliação, gráficos |
| `resultados_uf*/` | etapas 4, 6 e 7–12 (`resultados_uf_r2` … `_r7`) | previsões de temporada por UF, uma por modelo × temporada |
| `resultados_semanal*/` | etapas 6 e 7 | previsões semanais, uma por modelo × rodada |
| `avaliacao_*/` | avaliação de cada rodada | tabelas com carimbo de data (`*_AAAAMMDD-HHMM.csv`) e `*_ultimo.csv` |
| `cache_*/` | cada rodada | detectores, estudos do Optuna, checkpoints, parâmetros escolhidos |
| `producao_semanal/`, `producao_temporada/`, `relatorio_campeoes/` | `producao/` | previsões operacionais, tabelas e o documento dos campeões |

Os resumos finais de algumas etapas estão versionados no repositório: `notebooks/03_competicao/resultados_colab_2026-09-30/` e `notebooks/07_rodada2_pico_entressafra/resultados_finais/`.

## 5. Problemas conhecidos

- **INLA no Windows:** trava; rode a competição no Colab. No Colab, o INLA precisa ser executado a partir do disco local (`/content`), não do Drive (o notebook já faz isso).
- **Chronos-2 + LoRA:** o Colab traz `torchao` antigo, que o `peft` recusa; o notebook desinstala o `torchao`.
- **Windows e `n_jobs=-1`:** o `statsforecast` e o `tsfresh` reexecutam o script; use `if __name__ == '__main__'` ou `n_jobs=0`.
- **tsfresh:** o cache é calculado até a última semana dos dados; se os dados forem atualizados, a produção cria um cache novo (o nome da pasta tem a data).
