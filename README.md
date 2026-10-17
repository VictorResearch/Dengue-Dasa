# Previsão de dengue por UF: semanal e da temporada

Código, dados (por script) e histórico completo de um projeto de previsão de casos de dengue nas **26 UFs do Brasil** (sem o Espírito Santo), feito sobre os dados do [3º IMDC / Mosqlimate](https://sprint.mosqlimate.org/) (Infodengue Mosqlimate Dengue Challenge). Ele cobre dois problemas:

| | Pergunta | Campeão (métricas absolutas, teste 2022–2025) |
|---|---|---|
| **Semanal** | Quantos casos teremos em cada UF nas próximas 1, 2, 3 e 4 semanas? | **`ENS_stack_sem`**: média ponderada de 9 modelos. MAE de 270 casos/semana em 1 semana (persistência: 354) e 583 em 4 semanas (persistência: 1.186). |
| **Temporada** | Em junho (SE25), como será a temporada inteira seguinte (SE41 → SE40)? | **`ENS_pico7_simples_UF`**: normal de redes de grafo, corrigido por um detector de anos grandes. Erro mediano da altura do pico de +2% e do total de −0,6%; menor WIS (721) e MAE (1.277) entre todos os testados. |

Pela régua oficial do desafio (WIS relativo ao modelo-base BB, média geométrica entre UFs), o melhor modelo de temporada do projeto é o `ENS_pico7_UF`, com **0,808 × BB**. A melhor equipe do 3º IMDC fez 0,870.

---

## O que tem aqui

```
dengue-imdc-previsao/
├── README.md                     ← este arquivo
├── requirements.txt              ← pacotes do núcleo e da produção
├── requirements-completo.txt     ← + pacotes usados só em notebooks da pesquisa
├── scripts/                      ← pipeline de dados: baixa os brutos e monta as bases finais
│   ├── 00_baixar_dados.py            FTP oficial do 3º IMDC + previsões publicadas das equipes (GitHub)
│   ├── 01_refazer_municipal.py       bases municipais (fase AGCRN; opcional para o resto)
│   ├── 02_preparar_painel_v4.py      painéis por região de saúde e UF (casos, população, clima, ENSO, SEAS5, vizinhança)
│   ├── 03_preparar_tabela_uf.py      tabelas por UF (semanal, temporada, estática) com toda a engenharia de variáveis
│   └── 04_montar_area_de_trabalho.py junta código + dados + parâmetros na pasta que os notebooks usam
├── dados/
│   └── derivados/                ← vizinhança municipal (Queen) usada pelos scripts 02 e 03
├── src/                          ← bibliotecas da pesquisa (uma por etapa: v4_lib, comp_lib, uf_lib, semanal_lib, r2_lib … r7_lib)
├── notebooks/                    ← cada teste, em ordem, com o notebook, o gerador e o LEIA-ME da etapa
│   ├── 01_agcrn_municipal/           junho → setembro: AGCRN de 5.492 municípios, 1 semana à frente
│   ├── 02_v4_temporada/              virada para a temporada inteira (régua: WIS × BB)
│   ├── 03_competicao/                19 modelos por região de saúde
│   ├── 04_modelos_uf/                83 modelos direto por UF (redes de grafo, ARIMA, VAR, GBM, INLA…)
│   ├── 05_comparacao_felipe/         auditoria de um "SMAPE de 16%" externo
│   ├── 06_semanal/                   todos os modelos a 1–4 semanas, 8 temporadas
│   ├── 07_rodada2_pico_entressafra/  fase, tsfresh, Optuna, Chronos-2 com LoRA, stacking
│   ├── 08_rodada3_detector/          detector de anos epidêmicos + misturas
│   ├── 09_rodada4_literatura/        sorotipo, mar, seca, elasticidade, MTGNN condicionado
│   ├── 10_rodada5_modelo_por_fase/   detector corrigido, ensembles por fase
│   ├── 11_rodada6_optuna/            otimização dos parâmetros do pico (validação aninhada)
│   └── 12_rodada7_ensemble_pico/     ensemble do pico (os campeões de temporada)
├── producao/                     ← os 5 melhores de cada nível, prontos para uso operacional
│   ├── prod_lib.py                   código de produção (semanal e temporada)
│   ├── Producao_Semanal.ipynb        previsão de 1–4 semanas a partir da última semana completa
│   ├── Producao_Temporada.ipynb      previsão da temporada (SE41 → SE40) a partir da SE25
│   ├── Relatorio_Campeoes.ipynb      gráficos de todas as UFs/regiões e o documento dos campeões
│   ├── relatorio_campeoes.py         gerador do documento (HTML autocontido)
│   └── parametros/                   hiperparâmetros, seleção de variáveis e pesos escolhidos na afinação
├── graficos/                     ← gráficos por UF, macrorregião e Brasil (semanal e temporada)
└── docs/                         ← documentação
    ├── 01_historico.md               tudo o que foi testado, em ordem, com métricas, hipóteses e bugs
    ├── 02_dados.md                   fontes, download, processamento e variáveis
    ├── 03_como_rodar.md              ambiente, ordem dos notebooks, tempos, dependências
    ├── 04_avaliacao_e_metricas.md    protocolo de avaliação e o que cada métrica mede
    ├── 05_modelos_campeoes.md        os campeões e o top 5 de cada nível, com parâmetros
    ├── 06_producao.md                como rodar a previsão operacional e atualizar
    ├── 07_graficos.md                gráficos por UF, região e Brasil
    └── mapa_original/                os documentos de trabalho originais (00 a 22), como registro
```

---

## Início rápido

### 1. Ambiente

```bash
git clone https://github.com/alphaHunterPrimal/Dengue-Dasa.git
```

```bash
python -m venv .venv
```

Ative o ambiente (`.venv\Scripts\activate` no Windows, `source .venv/bin/activate` no Linux/macOS) e instale:

```bash
pip install -r requirements.txt
```

Os treinos pesados (redes de grafo, Chronos-2) foram feitos no **Google Colab** com GPU L4; o pipeline de dados e os modelos estatísticos rodam em qualquer PC com ~16 GB de RAM. Detalhes em [docs/03_como_rodar.md](docs/03_como_rodar.md).

### 2. Dados (~3,3 GB descompactados)

```bash
python scripts/00_baixar_dados.py
```

Baixa do FTP público do 3º IMDC (`info.dengue.mat.br`, pasta `data_imdc_2026`) e do repositório de resultados do desafio no GitHub. Depois, monte as bases finais (~10–20 min cada):

```bash
python scripts/02_preparar_painel_v4.py
```

```bash
python scripts/03_preparar_tabela_uf.py
```

### 3. Área de trabalho

```bash
python scripts/04_montar_area_de_trabalho.py
```

Cria `area_de_trabalho/` com as bibliotecas, os dados prontos e os parâmetros. **No Colab:** suba essa pasta como `MyDrive/dengue_V4/` e abra os notebooks pelo Colab. **Localmente:** abra o Jupyter dentro dela.

### 4. Previsão operacional

Abra `producao/Producao_Semanal.ipynb` e `producao/Producao_Temporada.ipynb` no Colab (GPU) e rode tudo. Depois, `producao/Relatorio_Campeoes.ipynb` gera os gráficos de cada UF, região e Brasil e o documento `modelos_campeoes.html`. Ver [docs/06_producao.md](docs/06_producao.md).

### 5. Reproduzir a pesquisa

Os notebooks de `notebooks/` seguem a ordem da pesquisa; cada pasta tem um `LEIA-ME.md` com o que roda, o ambiente, o tempo e o que precisa estar pronto antes. A sequência e as dependências estão em [docs/03_como_rodar.md](docs/03_como_rodar.md).

---

## Protocolo de avaliação (resumo)

- **Tarefa oficial:** prever cada temporada **SE41 → SE40** usando só dados até a **SE25** (junho), com 9 quantis (faixas de 50, 80, 90 e 95%).
- **Afinação 2018–2021:** todas as escolhas (variáveis, hiperparâmetros, pesos, calibração) usam só estas temporadas.
- **Teste 2022–2025:** olhado uma vez por rodada, no fim. São os números deste README.
- **Régua:** WIS (*weighted interval score*), relativo ao modelo-base oficial BB e à climatologia, mais as métricas absolutas (MAE, WAPE, sMAPE, erro da altura do pico e do total, cobertura). Ver [docs/04_avaliacao_e_metricas.md](docs/04_avaliacao_e_metricas.md).
- **Semanal:** cada temporada é uma rodada; o modelo é treinado uma vez com alvos até 4 semanas antes da SE41 e aplicado a cada semana de origem, prevendo 1–4 semanas à frente.

---

## Linha do tempo em uma tabela

| Fase | O que mudou | Melhor resultado |
|---|---|---|
| jun/2026 · AGCRN municipal | rede de grafos em 5.492 municípios, 1 semana à frente, somada por UF | MAE estadual 476 (com vazamento no MinT) |
| 29/09 · bancos refeitos | duplicatas, clima desalinhado, ENSO zerado: tudo refeito; MinT trocado por calibração honesta | MAE estadual 243 (persistência: 185) |
| 29/09 · virada | o desafio pede a temporada inteira a partir da SE25; a habilidade de 1 semana não se transfere | régua passa a ser o WIS × BB |
| 30/09 · competição | 19 modelos por região de saúde | ARIMA_Fourier 0,865 × BB |
| 01/10 · modelos por UF | 83 modelos; redes de grafo com âncora sazonal e perda pinball | MTGNN_UF+λ 0,846 |
| 08–12/10 · semanal + rodada 2 | todos os modelos a 1–4 semanas; Optuna, tsfresh, Chronos-2, stacking | semanal 0,224 × climatologia; temporada 0,831 |
| 13–14/10 · rodadas 3–7 (pico) | detector de anos grandes, misturas, ensembles por fase, Optuna, ensemble do pico | **0,808 × BB**; altura do pico +2% |

O histórico completo, com todas as arquiteturas, métricas, hipóteses (confirmadas ou refutadas) e bugs encontrados, está em [docs/01_historico.md](docs/01_historico.md).

---

## Fontes de dados e créditos

- **Casos, clima, oceano, população, malhas:** dados oficiais do 3º IMDC, organizados pelo [Infodengue](https://info.dengue.mat.br/) / [Mosqlimate](https://sprint.mosqlimate.org/) a partir do SINAN, IBGE, ERA5 e ECMWF (SEAS5). Documentação oficial: *Data documentation – 3rd IMDC*.
- **Previsões das equipes e modelo-base BB:** [Mosqlimate-project/3rd_IMDC_results](https://github.com/Mosqlimate-project/3rd_IMDC_results).
- **Sorotipo:** SINAN Dengue no OpenDataSUS (baixado dentro do notebook R4_0).
- **Temperatura do mar:** NOAA ERSST v5 (baixado dentro do notebook R4_1).

Os dados não estão neste repositório; os scripts baixam tudo das fontes originais.

**Licença:** ainda não definida pelo autor.
