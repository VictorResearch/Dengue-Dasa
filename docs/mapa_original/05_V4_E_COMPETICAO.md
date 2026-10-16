# 05 · V4 e competição de modelos: decisões, resultados e achados (29–30/09/2026)

> Este documento guarda tudo o que foi descoberto depois da revisão de literatura ([04_ESTADO_DA_ARTE.md](04_ESTADO_DA_ARTE.md)): a mudança para a tarefa oficial, o V4, os testes no PC e no Colab, e a competição de modelos.
> Detalhes técnicos de cada pasta: [V4/LEIA-ME.md](../data_imdc_2026/V4/LEIA-ME.md) e [COMPETICAO/LEIA-ME.md](../data_imdc_2026/COMPETICAO/LEIA-ME.md).
> Arquivos de apoio (READMEs das equipes, figuras oficiais, experimento): [anexos/](anexos/).

---

## 1. Por que mudamos de tarefa (de "1 semana" para "temporada inteira")

**O desafio exige** prever SE41→SE40 usando só dados até a SE25, ou seja, de 16 a 67 semanas à frente ([regras](anexos/regras_3rd_imdc.txt)). Os modelos 1, 2 e 3 (até a v3.1) preveem 1 semana à frente com os casos da semana anterior.

**Experimento** ([anexos/experimento_1semana_vs_temporada](anexos/experimento_1semana_vs_temporada/)):
- **Dados e modelo:** 26 UFs, S1–S4, um modelo linear em log(casos).
- **Métrica:** WIS relativo à climatologia (abaixo de 1 = melhor).
- **Correção:** a 1ª versão do script tinha os pesos dos intervalos do WIS trocados. Estes são os números corrigidos.

| Forma de usar o modelo | WIS (contagens) | WIS (log) |
|---|---|---|
| 1 semana à frente com casos observados (como o Modelo 2) | 0,19 | 0,23 |
| O mesmo modelo desenrolado da SE25, com trajetórias | 1,02 | 0,80 |
| Direto multi-horizonte | 0,98 | 0,81 |
| Ensemble das três abordagens | **0,94** | **0,80** |
| Repetir a SE25 | 2,32 | 1,06 |

**Conclusões:**
1. A habilidade em 1 semana **não passa** para a temporada. Por isso, otimizar o Modelo 2 em 1 semana não ajuda no desafio.
2. Recursivo e direto empatam. O que importa é a informação sobre o **tamanho** da temporada.
3. O ensemble ganha. A persistência (MAE 185) deixa de ser régua; a régua passa a ser o **modelo-base BB** do desafio.

---

## 2. O que o 3º IMDC ensinou (resultados publicados)

- **Participação e régua:** 31 equipes, avaliadas por WIS ÷ WIS do modelo-base BB/PROCC, com média geométrica nas 4 safras.
- **Estáveis:** PURDUE, LNCC_ARP26, EMAP_PROPHET, RKI_PH e IFGW.
- **Melhor média:** EMAP SARIMAX (0,870).
- **O que os de topo fazem:**
  - ensembles simples de climatologia, LightGBM quantílico, GRU e modelo mecanístico, com calibração conformal (PURDUE);
  - AR(92) só com casos (LNCC);
  - decompor em "tamanho da temporada × forma da curva" (IFGW);
  - **nenhum** usa grafo aprendido, e vários não usam clima.
- **O nosso avaliador reproduz o oficial:** o WIS bate com a coluna `wis` publicada (razão 1,000 nas validações 1 e 2), e o ranking também (PURDUE vence o BB em 20 UFs, LNCC em 18).
- READMEs das 16 equipes lidas: [anexos/3rd_imdc_equipes/readmes](anexos/3rd_imdc_equipes/readmes/). Figuras oficiais: [anexos/3rd_imdc_equipes/figuras_oficiais](anexos/3rd_imdc_equipes/figuras_oficiais/).

---

## 3. V4 (`data_imdc_2026/V4/`)

**O que é:** o pipeline da tarefa oficial. Tem:
- **dados** agregados em 435 regiões de saúde e 26 UFs;
- **avaliador** WIS;
- **modelos-base:** climatologia, canal endêmico NB, AR recursivo e AR longo;
- **versão A:** AGCRN direta, que prevê 68 semanas de uma vez, parte da âncora sazonal e treina com origens sintéticas;
- **versão B:** LightGBM para o tamanho da temporada × forma média da curva;
- **variáveis de temporada:** a etapa 5;
- **ensembles.**

### Resultado no Colab (treino completo), S1–S4, WIS relativo ao BB

| Modelo | R_vs_BB | UFs melhores que o BB | dp log R |
|---|---|---|---|
| 3º IMDC: EMAP SARIMAX / LNCC / PURDUE | 0,870 / 0,897 / 0,904 | 17 / 18 / 20 | 0,45 / 0,41 / 0,29 |
| **ENS_bases_A_mediana** | **0,938** | 18 | 0,298 |
| **ENS_bases_AB_mediana** | **0,944** | **19** | **0,265** (o mais estável entre os que vencem o BB) |
| ENS_bases_pesos | 0,954 | 18 | 0,348 |
| AR longo | 0,980 | 14 | 0,446 |
| climatologia | 1,012 | 16 | 0,310 |
| A_uf / A_regiao | 1,121 / 1,365 | 11 / 10 | 0,68 / 1,00 |
| B_uf / B_regiao | 1,238 / 1,382 | 6 / 4 | 0,57 / 0,64 |

### Achados do V4

1. **A mediana dos membros vence a média e os pesos** em todos os conjuntos.
2. **A versão A piora com mais treino** (A_uf: 0,977 com 2 épocas → 1,121 com o treino completo). Ela se afasta da âncora e sobreajusta. Sozinha perde; na mediana do ensemble, ajuda.
3. **A versão B por região sobreajusta:** as variáveis mais usadas foram `uf_code` e `enso_12s`, que funcionam como "identificador de estado e de ano". Por UF ela é melhor, mas **superestima depois de temporadas grandes** (2025, depois do recorde de 2024 no Sul).
4. **Ablação da etapa 5** (versão B por UF, 10 temporadas):

   | Grupo retirado | Efeito de retirar |
   |---|---|
   | tendência das últimas 8 semanas | **piora 6,5%** (é o grupo que mais ajuda) |
   | clima prolongado (12 e 26 semanas) | piora 3,1% |
   | ENSO | piora 1,7% |
   | Richards (estilo Episcanner) | piora 1,2% |
   | incidência acumulada | piora 1,1% |
   | vizinhos e UF | piora 0,8% |
   | SEAS5 | **melhora 1,5%** (atrapalha) |
   | taxa de ataque de 1–3 temporadas | **melhora 6,0%** (atrapalha: com poucas temporadas, ensina "ano grande → ano grande" em vez de imunidade) |

5. **A recalibração com temporadas passadas piorou o WIS.** Testamos a conformal clássica, a otimizada pelo score e a só de largura. As temporadas de calibração (2016–21: anos baixos e covid) são de outro regime, e a de 2024 exige intervalos enormes. Por isso ela ficou desligada. O PURDUE só ganhou com conformal porque calibrou nas próprias safras de validação.
6. **O maior erro de todos os modelos é o surto do Sul em 2022/23** (PR, SC, RS), com picos de 14–20 mil casos por semana contra previsões de 500–3.000, e em menor grau MG e SP. A mediana fica quase igual à climatologia: os modelos supõem que o futuro repete a média do passado, mas a dengue está se expandindo para o Sul. [Leque da S1](../mapa%20v4%20primeiro%20teste.png).

### Problemas de dados encontrados (e contornados)

- `climate.csv` **não tem** `precip_tot` (a documentação lista); usamos `precip_med` e `rainy_days`.
- `forecasting_climate.csv` mistura dois formatos de data.
- As datas do ENSO caem em segunda ou terça; foram alinhadas ao domingo.
- A **API do Episcanner exige chave**, e o Zenodo só tem padrões agregados até 2023. Os parâmetros de Richards foram calculados do próprio `dengue.csv`, o que também evita dado externo a declarar.
- A **S4 só tem dados até 08/03/2026** no nosso `dengue.csv`. Os números da S4 diferem dos oficiais.

---

## 4. Competição de modelos (`data_imdc_2026/COMPETICAO/`)

**Desenho:**
- 19 competidores reimplementados (17 famílias da revisão e do 3º IMDC; ver o LEIA-ME da pasta), todos por região de saúde, somados até a UF.
- Os modelos que tratam cada série separadamente recebem a **dependência espacial e temporal histórica** pela reordenação de Schaake.
- Avaliação por UF (vs BB e as 28 equipes publicadas) e por região (vs climatologia).
- **Checagem de fidelidade:** cada reimplementação é comparada com a previsão estadual publicada pela equipe original.

**Notebooks do Colab:**

| Notebook | Modelos | Ambiente |
|---|---|---|
| `Comp1_CPU` | 10 modelos de CPU | CPU High-RAM (**já rodado em 30/09**) |
| `Comp2_INLA` | BB e GHR | CPU High-RAM |
| `Comp3_GPU` | XGBoost, DeepAR, NHITS, TFT, TSMixer, Chronos, AGCRN | GPU L4 |
| `Comp4_Avaliacao` | avaliação de tudo | CPU padrão |

### Testes no PC (safra 2022, R_vs_BB; referência: topo do 3º IMDC em 2022 = 0,90–0,92)

| Modelo | R | Nota |
|---|---|---|
| AR_longo | 1,03 | Explodia em log2; ganhou um teto de 3× o máximo histórico |
| climatologia | 1,04 | |
| canal_endemico | 1,09 | |
| SURGE | 1,15 | Fiel ao original do LNCC (1,20) |
| Renovacao_S | 1,18 | **Melhor por região** (0,98 vs climatologia); incerteza aumentada depois |
| LightGBM_direto | 1,24 | A 1ª versão tinha viés de +0,9 (resíduos de um só ano); refeito como regressão quantílica |
| XGBoost_quantis | 1,25 | 8 min em CPU; no Colab usa GPU |
| Tamanho_x_forma | 1,34 | Superestima |
| Prophet | 1,44 | Intervalos estreitos |
| ARIMA_Fourier / ETS_MSTL | 2,90 / 3,68 | Extrapolavam tendência; **corrigidos** (sem drift, só nível) e não reavaliados |

### Resultado da competição no Colab (30/09, S1–S4 = temporadas 2022–2025)

Cópia local das tabelas: `data_imdc_2026/COMPETICAO/resultados_colab_2026-09-30/`. Gráficos (leques, mapa de calor, estabilidade) em `MyDrive/dengue_V4/resultados_competicao/graficos/`.

**Por UF** (WIS relativo ao BB oficial, média geométrica em 26 UFs × 4 temporadas; < 1 = melhor que o BB):

| # | Modelo | R vs BB | UFs melhores que o BB | dp log R | IC50 / IC90 (%) |
|---|---|---|---|---|---|
| 1 | **ARIMA_Fourier** (nosso) | **0,865** | **19** | 0,37 | 47 / 82 |
| – | EMAP SARIMAX (1º do 3º IMDC) | 0,870 | 17 | 0,45 | 46 / 83 |
| – | LNCC ARP26 | 0,897 | 18 | 0,41 | 41 / 78 |
| – | Purdue NeuralEarth | 0,904 | 20 | 0,29 | 45 / 85 |
| – | *V4: ENS_bases_A_mediana (notebook do V4)* | *0,938* | *18* | – | – |
| 2 | climatologia | 0,993 | 16 | 0,31 | 42 / 79 |
| 3 | ENS_mediana (todos os modelos) | 1,003 | 15 | **0,25** | 55 / 88 |
| 4 | AR_longo | 1,015 | 13 | 0,47 | 47 / 82 |
| 5 | LightGBM_direto | 1,032 | 15 | 0,45 | 51 / 84 |
| 6 | SURGE | 1,050 | 10 | 0,32 | 41 / 81 |
| 7 | canal_endemico | 1,052 | 12 | 0,32 | 48 / 82 |
| 8 | ENS_top5_temporada_anterior | 1,088 | 8 | 0,43 | 53 / 87 |
| 9 | Renovacao_S | 1,167 | 11 | 0,53 | 42 / 73 |
| 10 | ENS_pesos | 1,210 | 6 | 0,56 | 53 / 88 |
| 11 | GHR_INLA | 1,233 | 3 | 0,50 | 52 / 85 |
| 12 | XGBoost_quantis | 1,275 | 6 | 0,48 | 56 / 89 |
| 13 | Tamanho_x_forma | 1,361 | 4 | 0,67 | 48 / 80 |
| 14 | **AGCRN** | **1,378** | 9 | **0,97** | 53 / 88 |
| 15 | Chronos | 1,431 | 7 | 1,10 | 49 / 83 |
| 16 | BB_INLA (nossa reimplementação) | 1,555 | 4 | 0,68 | 50 / 86 |
| 17 | Prophet | 1,631 | 4 | 1,00 | 27 / 52 |
| 18 | ETS_MSTL | 1,833 | 1 | 0,92 | 40 / 73 |

**Por região de saúde** (435 regiões, R vs climatologia): ARIMA_Fourier **0,878** (343 regiões melhores), Renovacao_S 0,955 (256), LightGBM 0,998 (235), canal 1,024, SURGE 1,025, XGBoost 1,061, AR_longo 1,093, GHR_INLA 1,115, **AGCRN 1,210**, BB_INLA 1,260, Tamanho_x_forma 1,352, Chronos 1,363, Prophet 1,635, ETS 1,642.

**Por macrorregião** (R vs BB): o ARIMA_Fourier é o 1º em 4 das 5 regiões (CO 0,84; NE 0,79; N 0,98; SE 0,80). No Sul ele é o 3º (0,97), atrás de Renovacao_S (0,95) e LightGBM (0,95). A AGCRN fica em 1,85 no Norte e 1,88 no Sul.

**Alvos da temporada** (ARIMA_Fourier): erro absoluto em log do total da temporada = 0,80 (fator típico de ~2,2×), com viés de −0,46 (subestima o total, mediana ~37% abaixo); erro na semana do pico ≈ 5 semanas. Piores UFs: RS, SC e AP.

**O que o ARIMA_Fourier é:** por região de saúde, ARIMA(p≤3, d=0, q≤3) sem drift sobre log da incidência, com 3 pares de Fourier da semana epidemiológica como sazonalidade fixa, 8 anos de treino, **sem nenhuma covariável**; somado até a UF por Schaake. É a mesma família do 1º colocado do 3º IMDC (EMAP SARIMAX), o que confirma o resultado por uma via externa.

**Ressalvas:**
- A correção do ARIMA (tirar drift e diferenciação) foi feita depois de vê-lo falhar em 2022, que é uma das 4 temporadas do teste. A mudança é genérica, mas existe esse contato.
- O critério definido antes (ENS_top5_temporada_anterior) ficou em 1,088, pior que o ARIMA sozinho. Escolher o ARIMA agora é escolher olhando o teste. O que sustenta a escolha: margem grande, vence em 4 de 5 macrorregiões e em 343 de 435 regiões, e a família já tinha vencido o 3º IMDC.
- Intervalos um pouco estreitos (IC50 47%, IC90 82%).
- **Não rodaram** DeepAR, NHITS, TFT e TSMixer: `val_size` (52) menor que o horizonte (68). Corrigido em `comp_modelos.py` (`val_size=H`), falta rodar.
- A nossa BB_INLA (1,555) ficou muito pior que o BB oficial (1,0): a reimplementação não é fiel, e os intervalos saíram largos demais. Não serve para julgar o INLA como classe. A nossa GHR_INLA (1,233) ficou melhor que a GHR oficial da BSC (1,528).

### Diagnóstico de 30/09 (decisões)

1. **Modelo para continuar:** ARIMA_Fourier por região de saúde, somado até a UF. Candidatos a complemento: Renovacao_S (1º no Sul e 2º por região) e LightGBM_direto (2º no Centro-Oeste e no Sul).
2. **AGCRN ficou para trás** nesta tarefa: 1,378 × BB por UF e 1,210 × climatologia por região, o mais instável (dp 0,97). Isso repete o V4 (A_regiao 1,365). Parar de investir nela para a temporada; o Modelo 2 continua servindo para 1 semana à frente.
3. **Variáveis:** nenhum modelo com covariáveis (clima, temporada, ENSO, taxa de ataque) superou o ARIMA sem covariáveis, e todos eles ficaram abaixo da climatologia por UF. Os ganhos da ablação do V4 (1–6%) são pequenos perto da diferença entre famílias de modelo (~40%). Conclusão: a 16–67 semanas, a informação útil é o **nível atual + a sazonalidade**. Manter só o conjunto enxuto (tendência de 8 semanas, anomalia de clima prolongada, ENSO) para um teste de ARIMAX; tirar taxa de ataque, SEAS5 e `uf_code`.
4. **Temporada × semana a semana:** prever a semana seguinte é muito mais preciso (WIS log 0,23 contra 0,80), mas não responde ao desafio, que só dá dados até a SE25. Desenrolar semana a semana a partir da SE25 empata com o direto (1,02 × 0,98 no experimento). O próprio ARIMA vencedor é recursivo: a memória de curto prazo (AR ≤ 3) some em poucas semanas, e a previsão converge para a curva sazonal no nível atual. **O que decide não é direto × recursivo, e sim âncora sazonal estável + nível atual + incerteza honesta.** A previsão de temporada é útil (13% melhor que o BB e a climatologia), mas modesta: erra o total por um fator típico de ~2×.

### Problemas técnicos registrados

- **O R-INLA no Windows** (versão 25.10.19) travou no modo `compact`, e o `classic` ficou lento demais. O script foi otimizado (ajuste só com dados observados, estratégia `gaussian`, amostras só dos efeitos, BB com as 5 grandes regiões) e **o teste real fica no Colab (Linux)**.
- **O R-INLA no Colab (30/09, versão 26.08.07):** instalou, mas todas as temporadas do BB (2020–2024) falharam com `inla.mkl.run: Permission denied`. A instalação ficava em `MyDrive/.../R_libs`, e o Google Drive montado no Colab não deixa executar programas. **Correção:** o INLA roda de `/content/R_libs` (disco local), com `chmod` no `bin/`. O Drive guarda só `R_libs_inla.tar.gz`, que preserva as permissões. A célula de instalação agora termina com um teste do motor (`MOTOR_OK`). **Lição:** nunca executar binários a partir do Drive. Na 2ª tentativa, o motor rodou (**ajuste do BB em 14–19 s por temporada**, modo compact), mas o `inla.posterior.sample` exigiu o pacote `sn`. Correção: a instalação passou a incluir `sn` e as dependências dele, e o `comp_inla.R` agora tenta de novo com `skew.corr=FALSE` se o `sn` estiver faltando.
- **Comp1, Comp2 (após as duas correções), Comp3 e Comp4 rodaram no Colab (30/09).** No Comp3, as 4 redes do neuralforecast falharam com `val_size must be ... >= horizon: 68`; corrigido (`val_size=H`).
- **No Windows, o `n_jobs=-1` do statsforecast** reexecuta o script (é preciso `if __name__ == '__main__'`). No Colab não acontece.
- **O `boxplot(labels=...)`** mudou para `tick_labels` no matplotlib ≥ 3.9. Já está tratado.

---

## 5. Próximos passos (em ordem)

1. ~~Rodar `Comp2_INLA`, `Comp3_GPU` e `Comp4_Avaliacao`.~~ Feito em 30/09 (ver o diagnóstico acima).
2. **Lapidar o ARIMA_Fourier**, escolhendo tudo nas temporadas 2016–2021 (fora do teste):
   - calibração da largura dos intervalos (cobertura hoje 47/82);
   - correção do viés negativo no total da temporada;
   - K de Fourier (2–4) e anos de treino (6/8/10);
   - ARIMAX com o conjunto enxuto de variáveis;
   - ensemble pequeno ARIMA + Renovacao_S + LightGBM (útil no Sul).
3. Rodar as 4 redes do neuralforecast (erro de `val_size` corrigido), só para completar o quadro.
3b. **`Comp5_Negocio` (criado em 30/09).** Métricas para a decisão de compra de testes, que é um problema do jornaleiro: compra-se o quantil τ* = custo de faltar ÷ (custo de faltar + custo de sobrar) da previsão do **total do período**.
   - **O que calcula:**
     - o custo realizado ÷ o custo da climatologia, que é a métrica para escolher o modelo;
     - a frequência de falta, a falta e a sobra em %;
     - a calibração do total em vários τ;
     - MAE, WAPE e sMAPE semanais;
     - a compra sugerida por UF.
   - **Totais por trajetórias:** os totais vêm de trajetórias reconstruídas dos quantis gravados (amostras + Schaake), e não da soma dos quantis.
   - **Não precisa rodar os modelos de novo.** A configuração padrão é a S1 (2022), com a temporada inteira e os dois semestres.
   - **Comparação justa:** mesmas unidades, mesma verdade e mesma reconstrução para todos; quem tiver previsão incompleta fica de fora.
   - **Ainda não rodou no Colab.**
4. Ideias antigas, agora de prioridade baixa:
   - **um membro com tendência:** climatologia com peso maior nas temporadas recentes, ou tendência do tamanho da temporada por UF. É para o Sul;
   - **B enxuta:** sem taxa de ataque, SEAS5 e `uf_code` por região;
   - **A com rédea curta:** menos épocas, penalizar o afastamento da âncora, parar pelo WIS.
3. Escolher o modelo final por um critério definido **antes** (por exemplo, mediana dos 5 melhores da temporada anterior), para não viciar a escolha nas 4 safras.
4. Gerar a submissão no formato da plataforma (`exportar_submissao` do V4).

---

## 6. Onde está cada coisa

| O quê | Onde |
|---|---|
| Revisão de literatura e 3º IMDC | [04_ESTADO_DA_ARTE.md](04_ESTADO_DA_ARTE.md) |
| READMEs das equipes, figuras e regras | [anexos/](anexos/) |
| Experimento 1 semana × temporada | [anexos/experimento_1semana_vs_temporada](anexos/experimento_1semana_vs_temporada/) |
| Código e dados do V4 | `data_imdc_2026/V4/` (resultados do Colab: `MyDrive/dengue_V4/resultados_v4/`) |
| Código da competição | `data_imdc_2026/COMPETICAO/` (resultados: `MyDrive/dengue_V4/resultados_competicao/`) |
| Testes feitos no PC | `V4/_testes_locais/`, `COMPETICAO/_testes/` |
