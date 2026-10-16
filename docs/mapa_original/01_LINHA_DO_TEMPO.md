# 01 · Linha do tempo do projeto (Dengue · 3rd IMDC / Mosqlimate)

> Reconstruído em 28/09/2026 a partir dos arquivos em `F:\AmbientePython\Projetos`. **Atualizado em 29/09/2026** (ciclos 9 a 12).
> O estado atual resumido está em [00_LEIA_PRIMEIRO.md](00_LEIA_PRIMEIRO.md).
> As datas vêm dos timestamps dos arquivos. Quando algo foi **inferido**, está marcado com *(inferido)*.
> Os documentos irmãos são [02_BANCOS_DE_DADOS.md](02_BANCOS_DE_DADOS.md) e [03_AVALIACAO.md](03_AVALIACAO.md).

---

## Resumo em 30 segundos

| | |
|---|---|
| **Objetivo** | Prever casos semanais de dengue por **UF** (desafio obrigatório) com intervalos de 50/80/90/95%, nas 4 safras de validação (EW41→EW40) e na safra 2026-27. Regras: [`data_imdc_2026/Rules _.html`](../data_imdc_2026/Rules%20_.html) |
| **Abordagem** | Prever no nível **municipal** (5.492 municípios, sem ES) com uma rede de grafos recorrente **AGCRN** e saída **Binomial Negativa**, depois agregar para UF e reconciliar com **MinT** |
| **Onde você parou** | 29/09/2026. Os bancos foram refeitos (V8, safras 1–4), o SHAP foi refeito e o **Modelo 2** chegou à **v3** (losses equilibradas + calibração na validação). Melhor resultado medido na S1: MAE estadual **243**. O treino completo da v3 está pendente. Em junho, o trabalho tinha parado no Modelo 3, que piorou |
| **Safras avaliadas** | **Só a Season 1 (2022-23).** Os bancos corrigidos de S2, S3 e S4 existem desde 29/09, mas ainda não foram treinados |
| **Submissão** | Nenhum arquivo de submissão foi gerado. Os prazos do 3rd IMDC (validação 01/07, previsão 10/09) já passaram |
| **Último artefato de código** | [`data_imdc_2026/S1_corrigido/Modelo2_v3.ipynb`](../data_imdc_2026/S1_corrigido/Modelo2_v3.ipynb). O de junho era o [`Dengue dasa/Dengue_final_ufvjm (1).ipynb`](../Dengue%20dasa/Dengue_final_ufvjm%20(1).ipynb) (64 células) |
| **Registro histórico de resultados** | [`Dengue dasa/Avaliação_de_todos_os_modelos.docx`](../Dengue%20dasa/Avaliação_de_todos_os_modelos.docx), com 3 modelos, métricas e gráficos da S1 |

---

## Mapa dos arquivos de código (o que cada um é)

| Arquivo | Data | Linguagem | Papel |
|---|---|---|---|
| `data_imdc_2026/AGCRN em python.txt` | 11/06 | Python | **Protótipo v0**: AGCRN + física por *Sparse Gaussian Process* + SHAP |
| `data_imdc_2026/AGCRN em python.py` | 16/06 | Python | **Pipeline v1 "ambicioso"**: ensemble AGCRN + LightGBM + Ridge, Optuna, MinT. Não há resultados registrados |
| `data_imdc_2026/datanalysis in dengue.R` | 18/06 | R | Junta os CSVs brutos → `train/test{k}_dengue.parquet`, e depois faz o merge espacial → `*_COMPLETO_spatial.parquet` |
| `Dengue dasa/datanalysis in dengue.R` | 25/05 | R | **Esboço vazio** (só 3 `library()`). Pode apagar, a versão real é a de cima |
| `data_imdc_2026/TestesomenteAGCRN.py` | 18/06 | Python | **Fábrica de bancos**: gera `_shap_final`, faz o SHAP separado por regime (seca/pico), gera `LIMPO` e `FINAL_NN` |
| `Dengue dasa/Dengue_final_ufvjm (1).ipynb` | 27/09* | Python/Colab | **Notebook final**: física, lags, W1/W2 espaciais, MinT (4 variantes), Modelos 1-2-3, avaliação e gráficos |
| `…/Avaliação_de_todos_os_modelos.docx` | 19/06 | Word | Documentação de 3 modelos (código + métricas + gráficos). As duas cópias são **idênticas** (conferido byte a byte) |

\* 27/09 é a data do download da cópia. O conteúdo é de 17 a 19/06.

### Guia do notebook final por seção

| Células | Seção | O que tem |
|---|---|---|
| 1–3 | Início | Instalação (torch 2.4 + PyG), `VARS` (dicionário de colunas), `SEASONS`, `QUANTIS` |
| 4–7 | Engenharia espacial | Matrizes W1/W2 (Queen) e lags espaciais. **Gera `bancos/*_spatial_v2`** |
| 8–14 | Engenharia e seleção | `extrair_fisica` (Savitzky-Golay causal), `engenharia_de_features` (lags climáticos, EWM, seno/cosseno), SHAP walk-forward (top 15 e top 25). **Gera `*_dengue_eng`** |
| 16–18 | Entender o banco | Curva sazonal. Pico mediano na SE-13, entressafra da SE-25 à SE-45 |
| 19–23 | Conciliação | MinT simples, MinT-WLS, MinT dinâmico (pico/seca) e MinT dinâmico-WLS |
| 24–33 | **Modelo 1** | NB loss, AGCRN de 12 features, treino/validação, submissão, execução, avaliação e gráfico |
| 34–39 | Backups | Salvar modelo, reconciliação de backup, inferência em CPU "recuperação de desastre" |
| 40–50 | **Modelo 2** | Losses seca/pico, listas `COLS_SECA`/`COLS_PICO`, AGCRN padrão, treino bifurcado, execução e avaliação |
| 51–63 | **Modelo 3** | Loss de pico corrigida (float32), AGCRN_Pico + MHA com máscara macrorregional, treino, execução, inferência com warm-up |
| 64 | – | Download do `.db` |

---

## A história em ciclos

Cada ciclo segue a mesma estrutura: **dados → engenharia → análise/modelo → resultado → aprendizado**. O aprendizado de um ciclo é o que motiva o seguinte.

### Ciclo 0 · Coleta e entendimento das regras (mar a jun/2026)

- **Dados:** 13 arquivos brutos em `data_imdc_2026/variaveis/`: dengue, chikungunya, clima observado, clima previsto (ECMWF), oscilações oceânicas, população DATASUS, Köppen/bioma, mapa de regionais de saúde, acessos Afya e 3 malhas `.gpkg`. Os downloads foram de 22/03 a 03/06.
- **Documentação:** `Data documentation - 3rd IMDC _.pdf` e `Rules _.html` (11/06).
- **Aprendizado-chave das regras:** cada safra de validação deve ser prevista **inteira (EW41 a EW40), usando só dados até a EW25**. Isso vira um ponto crítico na avaliação (ver [03_AVALIACAO.md §4](03_AVALIACAO.md#4-problemas-críticos-encontrados)).

### Ciclo 1 · Protótipo v0 (11/06) · `AGCRN em python.txt`

- **Dados:** CSVs brutos lidos direto, só a safra 1.
- **Engenharia:** incidência, features climáticas e infodemiológicas. A "física da epidemia" (casos suavizados, velocidade, aceleração) era estimada por um **Processo Gaussiano Esparso (SGP)** para cada município.
- **Seleção:** XGBoost + SHAP simples (sem walk-forward), top 15 features, alvo = incidência.
- **Modelo:** `PreditorPandemicoAGCRN` (embedding 10) + loop de forecasting.
- **Resultado:** nenhum registrado.
- **Aprendizado *(inferido)*:** o SGP por município era lento demais e o SHAP sem walk-forward vazava informação. Os dois foram trocados na v1.

### Ciclo 2 · Pipeline v1 "completo" (16/06) · `AGCRN em python.py`

Tudo o que foi planejado está no cabeçalho do arquivo, marcado de `[C1]` a `[F5]`:
- SGP → **Savitzky-Golay** (mais rápido); SHAP **walk-forward**; **imunidade EWM** (meia-vida de 104 semanas); lags climáticos via `merge_asof`; Afya em log-diferença.
- **Binomial Negativa** com `r` dependente de velocidade e aceleração; offset `log(população)`.
- **Ensemble**: AGCRN + LightGBM (com lags do alvo) → meta-modelo **Ridge**; `r` recalibrado por MLE; **Optuna**; **MinT**.
- **Resultado:** nenhum registrado.
- **Aprendizado *(inferido)*:** o ensemble, o Optuna e o LightGBM **não chegaram ao notebook final**. Foram abandonados, provavelmente pelo custo computacional no Colab. A partir daqui o foco passa a ser só a AGCRN.

### Ciclo 3 · Base V1 → V2 → V3 (16/06 à tarde até 17/06 de manhã)

| Passo | Banco | Gerado por | O que mudou |
|---|---|---|---|
| 1 | **V1** `train/test{k}_dengue.parquet` (69 col.) | R, `datanalysis in dengue.R` | Junta os 9 CSVs e cria VPD, amplitude térmica dinâmica, volatilidades e HTI. Recorte por safra |
| 2 | **V2** `*_dengue_eng.parquet` (138 col.) | Notebook, célula 12 | Lags climáticos (3–5 semanas nas safras 1–2, **2–4 nas safras 3–4**), ENSO/IOD/PDO com lag de 6–9, EWM, seno/cosseno, target encoding (bioma, Köppen) e física Savitzky-Golay |
| 3 | SHAP walk-forward | Notebook, célula 12 / `TestesomenteAGCRN.py` | XGBoost + SHAP em 3 folds temporais, alvo = `casos.x`. Foram mantidas as **9 features estáveis em 3–4 safras** |
| 4 | **V3** `*_dengue_eng_shap_final.parquet` (22 col.) | `TestesomenteAGCRN.py` ("FILTRAGEM PÓS-SHAP") | Fica só com chaves + alvo + as 9 features SHAP. A física é recalculada no teste com 12 semanas de contexto do treino |

**Features escolhidas nesse ciclo:** `seno_semana, umid_med_forecast, imunidade_ewm, precip_tot_forecast, access_count, temp_med_forecast, pressure_min, casos_chikungunya, koppen_encoded`, mais a física (`casos_suavizados, velocidade, aceleracao`) = **12 features**.

### Ciclo 4 · Modelo 1: AGCRN simples (17/06, ~14h)

- **Dados:** V3 (`_eng_shap_final`), sem Espírito Santo. **Ainda com linhas duplicadas.**
- **Modelo:** `PreditorNBAGCRN`, com AGCRN (K=2, 64 canais ocultos, embedding de 32 por nó), dropout 0.2, cabeça `μ` com offset log-pop e cabeça `log r` com velocidade e aceleração.
- **Treino:** NB-NLL, AdamW (lr 0.004), ReduceLROnPlateau, early stopping por MAE nas últimas 10 semanas do treino, float16 + `torch.compile`.
- **Pós-processamento:** quantis NB → soma por UF → **MinT simples** (e depois MinT dinâmico, na célula 36).
- **Resultado (S1):** municipal MAE 5,23 · sMAPE 74,6% · r 0,47. Estadual MAE 830 · r 0,67 (docx). O resultado salvo está em [`modelos/resultados_agcrn.db`](../data_imdc_2026/modelos/resultados_agcrn.db).
- **Aprendizado:** o modelo **subestima muito os picos** (PR, SC, RS, MG, SP no gráfico) e superestima o começo da safra. O sMAPE alto vem do "ruído de zeros" da entressafra. Daí surgem as duas ideias do próximo ciclo: **tratar pico e seca separadamente** e **dar ao modelo informação dos vizinhos**.
  - Os "picos" pontuais da linha vermelha (observado) perto de jan/2023 no gráfico do docx são **artefato de linhas duplicadas**, não epidemia. Ver [03 §4.4](03_AVALIACAO.md#44-junções-do-r-geram-duplicatas-e-colunas-vazias).

### Ciclo 5 · Engenharia espacial e regimes: V4 → V5 → V6 → V7 (17/06 à noite até 19/06 às 8h)

| Passo | Banco | Gerado por | O que mudou |
|---|---|---|---|
| 1 | **V4** `bancos/*_shap_final_spatial_v2` (53 col.) | Notebook, células 5–7 | **Remove duplicatas** (713.960 linhas na S4) e o ES. Cria os lags `_t1` e as médias dos vizinhos **W1** (fronteira direta) e **W2** (vizinhos de 2ª ordem) |
| 2 | **V5a** `bancosDEF/*_COMPLETO_spatial` (170 col.) | R, loop no fim do script (18/06) | V4 + todas as colunas antigas de V2. **1ª execução, sem `distinct()` → duplicatas voltaram** (4,18 M linhas na S1) |
| 2b | **V5b** `data_imdc_2026/*_COMPLETO_spatial` (170 col.) | Mesmo loop, com a "CURA" `distinct()` (19/06, 7h) | Mesma coisa, **sem duplicatas** (3,58 M linhas na S1) |
| 3 | SHAP bifurcado | `TestesomenteAGCRN.py` | SHAP walk-forward **separado por regime** (seca = SE 23–48, pico = SE 49–22), top 25 de cada |
| 4 | **V6** `bancosDEF/*_LIMPO_spatial` (55 col.) | `TestesomenteAGCRN.py` | Fica só com a união das features SHAP (geral + seca + pico). **Feito a partir de V5a, com duplicatas** |
| 5 | **V7** `bancosDEF/*_FINAL_NN` (55 col.) | `TestesomenteAGCRN.py` | Contagens → `log1p(incidência)` (local, W1, W2, chikungunya), `log(pop)` e **StandardScaler** ajustado no treino |

### Ciclo 6 · Modelo 2: AGCRN bimodal "seca × pico" (19/06) ★ o melhor

- **Dados:** V7 `FINAL_NN`, deduplicado ao carregar no notebook.
- **Ideia:** dois especialistas AGCRN (embedding de 64), cada um com **suas features** e **sua loss**:
  - **PICO** (SE 47→25): `nb_loss_pico` = NLL × (1 + log1p(y)) + 0,2 × ReLU(y − μ), que pune subestimar o surto. 27 features.
  - **SECA** (SE 26→46): `nb_loss_seca` = NLL + 0,5 × ReLU(μ − y) × e^(−y), que pune o "falso alarme" onde não há casos. 21 features. weight_decay 1e-3.
  - Na hora de prever, cada semana usa o especialista do seu regime. Early stopping pela **loss** (não mais pelo MAE).
- **Reconciliação:** MinT dinâmico (uma matriz P para o pico e outra para a seca).
- **Resultado (S1, docx):** municipal **MAE 3,89 · sMAPE 39,5% · r 0,56** · cobertura IC95 97,7%. Estadual **MAE 476 · r 0,93** · IC95 98,2%.
- **Aprendizado:** o salto de r estadual (0,67 → 0,93) e a queda do MAE estadual pela metade mostram que o modelo agora acerta **o timing** dos picos. Mas ainda **subestima a amplitude** no Sul (PR, SC, RS). Os intervalos estão largos demais: IC50 cobre 85,7% no nível municipal.
- ⚠️ Os pesos deste modelo **não estão salvos** na pasta `modelos/` (ver ciclo 7).

### Ciclo 7 · Modelo 3: pico com Multi-Head Attention (19/06, tarde)

- **Mudanças em relação ao M2:** o especialista de PICO ganha uma camada de **MHA espacial** (4 cabeças) com residual + LayerNorm. A loss de pico vira **multiplicativa** (×1,5 quando subestima) e passa a ser calculada em float32. O **treino fica sem validação** (roda até 200/100 épocas e salva a última) e usa 100% do treino.
- **Resultado (S1, docx):** municipal MAE 4,79 · r **0,25** · IC95 87,6%. Estadual MAE 979 · r **0,39** · IC95 82,2%. **Pior que o Modelo 1.**
- **Leitura do gráfico:** as previsões ficam **achatadas perto de zero**. O especialista de pico colapsou e parou de acompanhar a curva.
- **Tentativa de correção (células 52–61):** máscara de atenção **por macrorregião de saúde**, projeção linear, SiLU e volta da validação. Na execução (célula 61), SECA rodou 50/50 épocas com `L_Val = 0.000` (validação vazia, ver [03 §4.7](03_AVALIACAO.md#47-validação-interna-do-modelo-bimodal-não-valida-a-seca)) e PICO parou na época 50/200 com `L_Val = 5.88`. O resultado foi para o `.db` e para `modelos_agcrn_s1.zip`, e a célula terminou com `NameError: gc`.
- **Pesos salvos:** `modelos/modelo_agcrn_{SECA,PICO}_s1.pt` e o `.zip` são **deste modelo**. Conferi que o PICO tem pesos `mha_espacial` (sem `proj_mha`, então é a **versão do docx**, sem máscara).

### Ciclo 8 · Onde parou (19/06, noite)

- **Células 62–63:** `inferencia_com_warm-up`, que aquece o estado oculto com as 10 últimas semanas do treino antes do teste, e `gerar_banco_inferencia_limpa(season_id=1)`. Começou a carregar os dados e **não há saída final**.
- **Docx** consolidado às 17:50.
- **Nada** foi rodado para S2, S3 ou S4, e nenhum CSV de submissão foi gerado.

### Ciclo 9 · Bancos refeitos do zero: V8 (28–29/09)

- **Motivo:** o mapa do projeto (28/09) achou duplicatas, previsão climática desalinhada, ENSO zerado, Afya constante e ilhas com 0 °C.
- **O que foi feito:** `refazer_season1.py` e depois `refazer_season.py --season N` montam tudo a partir dos CSVs brutos, numa grade município × semana sem duplicatas. Ficam em `S1_corrigido/` a `S4_corrigido/`, com as mesmas colunas de antes.
- **Resultado:** 79 checagens por safra + 38 de auditoria independente (S1), todas OK. Detalhes em [02 §7–9](02_BANCOS_DE_DADOS.md).
- **Primeiro teste:** o Modelo 2 original nos dados da S1 corrigida **piorou em magnitude** (MAE estadual 476 → 749) e manteve o timing (r 0,93 → 0,91). Indício de que o modelo antigo se apoiava em artefatos, como o Afya constante, que funcionava como indicador do porte da cidade.

### Ciclo 10 · SHAP refeito e descoberta do desalinhamento (29/09)

- **SHAP corrigido** (`SHAP_corrigido/`), rodado nas 4 safras:
  - alvo = casos da **semana seguinte**, como no treino da rede;
  - Poisson com offset de população;
  - regimes 47–25;
  - sondas de ruído e filtro de redundância.
- **Resultado:**
  - A variável mais importante é a **incidência observada da semana** (`log_inc_dengue_obs`), que o modelo não recebia.
  - **Afya** e **chikungunya** ficaram abaixo do nível de ruído.
  - ENSO e PDO são usadas pelo modelo auxiliar, mas atrapalham a validação: funcionam como "identificador de ano".
- **Bug encontrado:** no teste, a saída que aprendeu a prever *t+1* era comparada com *t*. A correção usa o arquivo `ctx{N}_FINAL_NN_v2.parquet` (semanas 26–40) para aquecer a rede, e a semana *t−1* passa a prever a semana *t*.

### Ciclo 11 · Modelo 2 v2 (29/09)

- **Mudanças:**
  - features novas, lidas do `FINAL_NN_v2`;
  - previsão alinhada;
  - validação = última temporada completa (52 semanas, com SECA e PICO);
  - **early stopping** pela loss (paciência 15).
- **Resultado (S1):** MAE estadual 560, r **0,97**. No município, r subiu de 0,56 para **0,82**. A SECA parou na época 171 e o PICO na 92: o treino levou ~57 min, contra ~2 h antes.
- **Diagnóstico** (`S1_corrigido/_analise_v2/`), com os pesos baixados do Drive e rodados localmente:
  - o **PICO superestima** 3–5× no começo da temporada e ~1,8× no pico;
  - a **SECA subestima** (~0,65×);
  - o **MinT usava os casos reais do teste** para calcular seus pesos e, com isso, encolhia as maiores cidades;
  - sem MinT, o erro honesto é MAE 829.
- **Correção sem treinar:** calibrar cada regime com um fator medido **na validação** e somar os municípios corretamente. MAE estadual **293** (confirmado no Colab).

### Ciclo 12 · Modelo 2 v3 (29/09) ◀ atual

- **Losses equilibradas:** NLL da Binomial Negativa + trava simétrica em escala log (prever 3× a mais custa o mesmo que 3× a menos), igual para os dois especialistas.
- **Treino:** cada especialista também aprende as semanas do outro regime, com peso 0,25.
- **Pós-processamento no lugar do MinT**, tudo ajustado só na validação:
  - fator de viés por UF, encolhido para o fator nacional;
  - inflação da variância estadual;
  - soma estadual por Binomial Negativa.
- **Resultado parcial:** só o pós-processamento, aplicado aos pesos da v2, dá MAE estadual **243**, r 0,98, IC50 50%. O treino completo da v3 está **pendente**.

### Ciclo 13 · Revisão de literatura e mudança de tarefa (29/09)

- **Revisão:** literatura (PubMed e web) e resultados publicados do 3º IMDC, em [04_ESTADO_DA_ARTE.md](04_ESTADO_DA_ARTE.md).
- **Experimento:** a habilidade em 1 semana não passa para a temporada (WIS 0,19 → ~1,0 × climatologia). Detalhes em [05 §1](05_V4_E_COMPETICAO.md).
- **Decisão:** abandonar a otimização em 1 semana e prever a **temporada inteira a partir da SE25**, com o **BB** como régua.

### Ciclo 14 · V4 (29–30/09)

- **O que foi feito:** avaliador WIS (idêntico ao oficial), modelos-base, versões A (AGCRN multi-horizonte) e B (tamanho × forma), variáveis de temporada e ensembles. Tudo em `data_imdc_2026/V4/`.
- **Colab:** melhor = **mediana de bases + A, 0,938 × BB** (topo do 3º IMDC: 0,870–0,924).
- **Aprendizado:**
  - A e B sozinhas perdem, mas ajudam no ensemble pela mediana;
  - a taxa de ataque e a SEAS5 atrapalham;
  - a recalibração com temporadas passadas piora;
  - todos erram o surto do Sul.

### Ciclo 15 · Competição de modelos (30/09) ◀ atual

- **O que foi feito:** 19 competidores por região de saúde, com a mesma soma até a UF e a mesma avaliação, em 4 notebooks do Colab. Tudo em `data_imdc_2026/COMPETICAO/`.
- **Situação:** `Comp1_CPU` rodado; INLA, GPU e avaliação pendentes.

---

## Linha do tempo compacta

```
mar–jun  Coleta dos CSVs brutos (variaveis/)
11/06    Regras + documentação · protótipo v0 (AGCRN + SGP)
16/06    R: V1 train/test_dengue · v1 "ambiciosa" (AGCRN+LGBM+Ridge+Optuna)
16–17/06 V2 _eng (lags, física, EWM) → SHAP geral → V3 _shap_final
17/06    ▶ MODELO 1 (AGCRN 12 feats + MinT)        S1: r_est 0,67
17/06    V4 spatial_v2 (dedup + W1/W2)
18/06    V5a COMPLETO (com dups) · SHAP seca/pico · TestesomenteAGCRN.py
19/06    V5b COMPLETO (sem dups) · V6 LIMPO · V7 FINAL_NN
19/06    ▶ MODELO 2 (bimodal seca/pico)  ★         S1: r_est 0,93
19/06    ▶ MODELO 3 (pico + MHA)                   S1: r_est 0,39
19/06    Docx de avaliação · inferência com warm-up (incompleta)
28/09    Mapa do projeto · V8 S1 corrigida · notebook enxuto do M2
29/09    M2 nos dados corrigidos                   S1: MAE_est 749 · r 0,91
29/09    V8 S2–S4 · SHAP corrigido · FINAL_NN_v2 · bug de alinhamento achado
29/09    ▶ M2 v2 (features novas, alinhado, early stopping)   S1: MAE_est 560 (com MinT)
29/09    Diagnóstico: viés dos especialistas + MinT olhando o teste · recalibração   S1: 293
29/09    ▶ M2 v3 (losses equilibradas + pós calibrado)   pós com pesos v2: 243
29/09    Revisão de literatura · experimento 1 semana × temporada → mudança para a tarefa oficial
30/09    ▶ V4 no Colab: mediana bases + A = 0,938 × BB (topo 3º IMDC 0,870–0,924)
30/09    ▶ Competição de 19 modelos por região de saúde · Comp1 rodado  ◀ ESTAMOS AQUI
```
