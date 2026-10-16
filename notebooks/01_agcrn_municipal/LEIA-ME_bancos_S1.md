# Season 1 corrigida (`S1_corrigido/`)

Reconstrução, **do zero e a partir dos CSVs brutos** (`variaveis/`), das tabelas `COMPLETO_spatial` e `FINAL_NN` da **Season 1** (treino EW 201001–202225, teste EW 202241–202340). Os bancos antigos não são usados para construir nada. Eles só aparecem numa comparação no fim da validação.

## Arquivos

| Arquivo | O que é |
|---|---|
| `refazer_season1.py` | **O script.** Gera tudo desta pasta (~3 min, ~6 GB de RAM de pico) |
| `auditar_season1.py` | Auditoria **independente**: recalcula as variáveis direto dos CSVs com pandas simples e compara com os parquets |
| `train1_dengue_eng_COMPLETO_spatial.parquet` | 3.575.292 linhas × 170 colunas · tabela "de registro", com NaN onde não há dado |
| `test1_dengue_eng_COMPLETO_spatial.parquet` | 285.584 linhas × 170 colunas |
| `train1_FINAL_NN.parquet` | 3.575.292 × 57 · mesmas 55 colunas do FINAL_NN antigo + 2 flags · **sem nenhum NaN** |
| `test1_FINAL_NN.parquet` | 285.584 × 57 |
| `scaler_FINAL_NN_season1.csv` | Média e desvio-padrão do treino de cada coluna escalonada (para desfazer o z-score) |
| `W1_queen_season1.npz`, `W2_season1.npz` | Matrizes espaciais normalizadas por linha (scipy.sparse), na ordem de `node_id` |
| `nos_season1.csv` | 1 linha por município: `node_id`, códigos, nomes, Köppen, bioma, centroide, nº de vizinhos, flags |
| `relatorio_validacao_season1.md` | 79 checagens do próprio pipeline (todas ✅) |
| `relatorio_auditoria_season1.md` | 38 checagens independentes + "antes × depois" (todas ✅) |
| `execucao.log`, `auditoria.log` | Saída completa da última execução |

## Como rodar

```bash
pip install numpy pandas scipy pyarrow
python refazer_season1.py      # gera as tabelas e o relatório de validação
python auditar_season1.py      # opcional: auditoria independente
```

O script encontra os dados sozinho (`../variaveis/`). Funciona igual no Colab: basta subir a pasta `data_imdc_2026` com `variaveis/` e esta subpasta. Não depende de geopandas nem de libpysal. O GeoPackage é lido com `sqlite3`, e a contiguidade Queen é calculada por vértices compartilhados, com resultado idêntico ao `libpysal.Queen` (conferido em 5.488 municípios).

## Como usar no notebook

Basta trocar o caminho dos arquivos. **Os nomes das colunas são os mesmos**, então `COLS_SECA`, `COLS_PICO`, `VARS` e o treino rodam sem alteração:

```python
arquivo_treino = "S1_corrigido/train1_FINAL_NN.parquet"
arquivo_teste  = "S1_corrigido/test1_FINAL_NN.parquet"
```

O filtro de ES e o `drop_duplicates` do notebook continuam funcionando, mas agora não removem nada. As linhas estão ordenadas por `(date, node_id)`, exatamente a ordem que o reshape `(T, N, F)` espera.

Recomendado: acrescentar `'afya_disponivel'` às listas de features que usam `access_count`, para a rede distinguir "0 acessos" de "sem dado antes de 2021".

## O que foi corrigido

| # | Problema no banco antigo | Como estava (S1 treino, `bancosDEF`) | Agora |
|---|---|---|---|
| 1 | **Linhas duplicadas** (join do clima previsto casava 6× nas semanas que começam no dia 1) | 604.120 linhas duplicadas (4.179.412 no total) | **0** · grade balanceada de 5.492 × 651. Duplicata é impossível por construção |
| 2 | **Clima previsto com semântica errada**: `reference_month` é o mês de *emissão*, e o join pegava todos os horizontes | 83% NA na origem + ffill/bfill; diferença média de 3,3 °C para o valor correto | Previsão **emitida no mês anterior ao mês-alvo** (horizonte 2), sempre disponível durante a semana. `precip_tot_forecast` convertida para **mm/dia** |
| 3 | **ENSO/IOD/PDO** com datas em seg/ter não casavam com o domingo | 100% NA → **0 em tudo** (dp = 0) | Data levada ao domingo da semana + LOCF (5 semanas), como manda a documentação |
| 4 | **Afya** diária casada só no domingo; bfill antes de 2021 | Valor de 1 dia; **constante por município** de 2010 a 2020 | **Soma dos 7 dias** da SE; 12 registros diários duplicados somados; 1.032 municípios ausentes = 0; antes de 2021 = NaN + `afya_disponivel` |
| 5 | **Ilhas sem clima** (Noronha, Itaparica, Madre de Deus) | **0,0 °C** de temperatura | IDW dos 3 municípios mais próximos + `clima_imputado` |
| 6 | Noronha sem previsão em 180 de 195 meses | NA → preenchido | IDW só nas células vazias + `forecast_imputado` por semana |
| 7 | ffill/bfill genérico (valores do futuro indo para o passado) | Em várias colunas | Não existe mais |
| 8 | **Teste processado isolado do treino** | EWM de imunidade reiniciava do zero no teste; lags e `_t1` vazios nas 1as semanas; física atravessava o buraco EW26–40 | Tudo calculado na **série contínua** (2009–2023) e depois recortado |
| 9 | `hti_lag2` | Igual ao `hti_lag3` (bug do R) | Lag 2 de verdade |
| 10 | seno/cosseno sobre a semana ISO do domingo | 1 semana de defasagem em relação à SE | Dia do ano da quarta-feira da semana |
| 11 | Ilhas na malha (Noronha, Ilhabela) | Lags espaciais = 0 ("vizinhos sem população") | Ligadas ao município mais próximo (simétrico) + `w1_vizinho_knn` |
| 12 | Lags climáticos | S1 tinha só 3–5 (S3–S4 tinham 2–4) | Sempre **2, 3, 4 e 5** |
| 13 | Target encoding e scaler | Ajustados com duplicatas e ES | Só nas linhas de treino limpas |

## Decisões que foram mantidas (são de modelagem, não erros)

- **Espírito Santo fora** (regra do desafio). `node_id` = ordem do geocode.
- **Física**: Savitzky-Golay (janela 7, ordem 3, em `log1p`) deslocada 3 semanas, igual à função `extrair_fisica` do notebook. Conferido: dá o mesmo valor que a função original aplicada à série contínua.
- **Sufixo `_t1`**: clima, Afya, população e chikungunya usam `shift(1)`. As físicas (`casos_suavizados_t1`, `velocidade_t1`, `aceleracao_t1`) usam `shift(0)`, como no notebook. O nome engana, mas foi mantido para compatibilidade.
- **FINAL_NN**: mesmas transformações do `TestesomenteAGCRN.py` (log da população, `log1p` da incidência local/W1/W2/chik) e o mesmo conjunto de colunas protegidas. Única mudança: `access_count` e `access_count_t1` passam por `log1p` antes do z-score, porque agora são somas semanais com cauda longa.
- **Faltantes no FINAL_NN** (Afya antes de 2021, `_t1` da 1ª semana de 2010) viram **0 depois do z-score**, ou seja, a média do treino, com flag.

## Colunas novas ou com significado alterado

| Coluna | Onde | Significado |
|---|---|---|
| `afya_disponivel` | ambas | 1 se `access_count` existe na semana (a partir de 2021-01-03) |
| `chik_disponivel` | ambas | 1 a partir de EW 201401 (antes, chikungunya = 0) |
| `clima_imputado` | COMPLETO | 1 nos 3 municípios-ilha sem ERA5 |
| `forecast_imputado` | COMPLETO | 1 nas semanas em que a previsão veio de IDW (só Noronha) |
| `pop_imputada` | COMPLETO | 1 se o ano não existe na tabela de população (nenhum caso na S1) |
| `w1_vizinho_knn` | COMPLETO | 1 nas ilhas e nos seus parceiros de vizinhança KNN |
| `access_count` | ambas | Agora é a **soma semanal** (antes, 1 dia). No FINAL_NN, `log1p` + z-score |
| `forecast_months_ahead` | ambas | Horizonte efetivamente usado: 2 (1 só em jan/2010) |
| `precip_tot_forecast` | ambas | mm/dia (a fonte está em mm/s) |
| `time_index` | ambas | Contínuo desde 201001 = 1 (treino 1–651). O teste começa em 667, não reinicia |
| `target_city` | COMPLETO | Flag única (antes vinha duplicada como `.x`/`.y`) |

Removidas do COMPLETO (lixo de merge): `train_k.x/.y`, `target_k.x/.y`, `disease.x/.y`, `target_city.y`, `accessed_disease`, `casos.y` (= `casos_chikungunya`).

## Limitação que continua valendo

As tabelas seguem o esquema **one-step-ahead** dos modelos atuais. A linha da semana *t* do teste usa casos observados até *t*, inclusive as semanas EW26–40 como histórico. Isso **não** é o formato do desafio, que exige prever a safra inteira só com dados até a EW25 (ver `_MAPA_DO_PROJETO/03_AVALIACAO.md` §4.1). Para as safras 2–4, o script pode ser adaptado trocando `EW_TREINO`/`EW_TESTE` e o sufixo `1` dos nomes de arquivo. Na S4, o teste precisa terminar em 202610 (fim do `dengue.csv`), e a população de 2026 será imputada pelo ano mais próximo (`pop_imputada`).

## Notebook do Modelo 2 para o Colab

`Modelo2_S1_enxuto.ipynb` traz só o necessário para treinar e avaliar o Modelo 2 com estes bancos. É o código original das células 22 e 41 a 50, com caminhos configuráveis, semente fixa, resultados que substituem em vez de acumular, checkpoint por época com retomada automática e pesos do SECA salvos. Foi testado de ponta a ponta (recorte de 3 UFs, incluindo a simulação de queda da sessão no meio do treino).

## Notebook v2 (`Modelo2_v2.ipynb`), criado em 29/09/2026

É o mesmo Modelo 2 do notebook enxuto, com quatro mudanças:
- Usa as features do SHAP corrigido, lidas dos arquivos `train/test/ctx1_FINAL_NN_v2.parquet`.
- A previsão do teste agora segue o mesmo alinhamento do treino (a semana *t−1* prevê a semana *t*).
- A validação usa a última temporada completa do treino, que tem semanas de SECA e de PICO.
- Tem early stopping pela loss de validação de cada especialista.

Para usar, suba os 3 parquets v2 para `MyDrive/dengue_S1/`. Os resultados vão para a pasta `resultados_modelo2_v2`.

## Notebook v3 (`Modelo2_v3.ipynb`), criado em 29/09/2026

Parte do diagnóstico da v2 (`_analise_v2/`): o PICO superestimava 3–5× no início da temporada, a SECA subestimava e o MinT usava os erros do teste para calcular os pesos.

O que muda em relação à v2:
- Os dois especialistas usam a mesma loss: NLL da Binomial Negativa + uma trava simétrica em escala log (`LAMBDA_TRAVA_LOG`).
- Cada especialista também treina nas semanas do outro regime, com peso `PESO_FORA_REGIME`.
- O MinT foi trocado por um pós-processamento calibrado só na temporada de validação: fator de viés por regime e UF, inflação da variância estadual e soma estadual correta.

Com os pesos da v2, esse pós-processamento sozinho dá MAE estadual 243 na S1 (IC50 50%, IC90 83%). O modo `SO_POS_PROCESSAMENTO = True` refaz só a previsão, sem treinar. Os resultados vão para `resultados_modelo2_v3`.
