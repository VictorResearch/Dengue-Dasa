# 02 · Dados: fontes, download e processamento

Nenhum dado vai no repositório. Tudo é baixado das fontes originais e processado pelos scripts de `scripts/`, sempre na mesma ordem:

```
00_baixar_dados.py  →  dados/brutos/ + dados/referencias_imdc3/
01_refazer_municipal.py (opcional)  →  dados/municipal/S{1..4}/
02_preparar_painel_v4.py  →  dados/v4/
03_preparar_tabela_uf.py  →  dados/uf/
04_montar_area_de_trabalho.py  →  area_de_trabalho/ (o que os notebooks usam)
```

## 1. Fontes

| Fonte | Conteúdo | Como chega |
|---|---|---|
| **3º IMDC (Infodengue/Mosqlimate)**, FTP anônimo `info.dengue.mat.br`, pasta `data_imdc_2026` | casos de dengue e chikungunya por município × semana epidemiológica (SINAN, por início dos sintomas, até a SE10 de 2026), clima observado ERA5 por município, previsão sazonal ECMWF SEAS5, ENSO/IOD/PDO, população DATASUS 2001–2025, Köppen e bioma, mapa município → região de saúde → macrorregião, malhas, acessos ao Afya Whitebook | `00_baixar_dados.py` |
| **Resultados do 3º IMDC** ([GitHub](https://github.com/Mosqlimate-project/3rd_IMDC_results), pasta `predictions/`) | previsões estaduais de todas as equipes e do **modelo-base BB** (a régua) | `00_baixar_dados.py` |
| **SINAN Dengue (OpenDataSUS)** | sorotipo por UF × semana (`DENGBRaa.csv.zip`) | notebook `09_rodada4_literatura/R4_0` |
| **NOAA ERSST v5** | temperatura da superfície do mar (Niño 3.4, Oceano Índico, Atlântico) | notebook `09_rodada4_literatura/R4_1` |

A documentação oficial dos dados é o *Data documentation – 3rd IMDC* (site do desafio). Regras do desafio: [sprint.mosqlimate.org](https://sprint.mosqlimate.org/).

### Atualizações de 2026

O FTP tem também arquivos `*_update_2026` (dados depois de 03/2026). **As análises do projeto usaram só os arquivos-base.** Para baixar também as atualizações: `python scripts/00_baixar_dados.py --atualizacoes`. Para usá-las na produção, é preciso juntá-las aos arquivos-base antes de rodar o script 02 (os formatos são os mesmos).

## 2. Regras do desafio que moldam os dados

- **26 UFs:** o Espírito Santo fica fora (dados com problemas de notificação) e também não entra como vizinho.
- **Temporada `s`:** da SE41 do ano `s` à SE40 de `s + 1`. **Corte:** a previsão da temporada `s` só pode usar dados até a **SE25 do ano `s`** (junho).
- **Calendário oficial:** colunas `train_k`/`target_k` do `dengue.csv` (safras de validação 1–4 = temporadas 2022 a 2025).
- **A última semana chega incompleta.** Os casos são contados pela semana de início dos sintomas; as notificações atrasam. Em 03/2026, a última semana tinha ~36% dos casos da anterior. A produção usa a última semana **completa** como origem (ver [06_producao.md](06_producao.md)).

## 3. Processamento

### `02_preparar_painel_v4.py` → `dados/v4/` (37 MB)

| Arquivo | Conteúdo |
|---|---|
| `calendario.parquet` | data, semana epidemiológica, temporada, colunas `train_k`/`target_k` |
| `painel_regiao.parquet`, `painel_uf.parquet` | casos, população e clima (média ponderada pela população) por região de saúde (435) e UF (26) |
| `oceano.parquet` | ENSO, IOD e PDO alinhados ao domingo da semana |
| `seas5_regiao.parquet`, `seas5_uf.parquet` | previsão sazonal por mês de emissão × horizonte |
| `adj_regiao.npz`, `adj_uf.npz` | vizinhança (fronteira) entre regiões e entre UFs, derivada da vizinhança Queen dos municípios |
| `richards_*.parquet` | parâmetros da curva de Richards (estilo Episcanner) por temporada, completa e truncada na SE25 |

### `03_preparar_tabela_uf.py` → `dados/uf/` (19 MB)

| Arquivo | Uma linha por | Para quê |
|---|---|---|
| `tabela_uf_semanal.parquet` | UF × semana | modelos semanais e de temporada (variáveis de origem) |
| `tabela_uf_temporada.parquet` | UF × temporada | variáveis conhecidas na SE25 + alvos da temporada |
| `tabela_uf_estatica.parquet` | UF | frações de bioma e Köppen, vizinhança |
| `seas5_uf.parquet` | UF × emissão × horizonte | previsão sazonal (chuva em mm/dia) |
| `vizinhanca_uf.npz` | — | W1 (fronteira), W1 de contato, W2 |
| `relatorio_validacao_uf.md` | — | todas as checagens (38) |

**Regras de agregação** (detalhes no cabeçalho do script e em [notebooks/04_modelos_uf/LEIA-ME.md](../notebooks/04_modelos_uf/LEIA-ME.md)):
- **contagens** (casos, população, acessos) são somadas;
- **taxas** são recalculadas como soma das contagens ÷ soma da população, nunca como média de taxas;
- **clima** é média ponderada pela população de 2015 (pesos fixos); município sem dado sai da conta, nunca vira 0;
- **variáveis não lineares** do clima (VPD, chuva × temperatura, volatilidade) são calculadas em cada município antes de agregar;
- **física da série** (Savitzky-Golay) e **imunidade** (EWM) são recalculadas na série da UF;
- **vizinhos** são recalculados no grafo das UFs, como razão agregada (casos dos vizinhos ÷ população dos vizinhos).

### `01_refazer_municipal.py` → `dados/municipal/S{N}/` (só para a fase AGCRN)

Monta as bases município × semana da fase de junho a setembro (`*_COMPLETO_spatial`, `*_FINAL_NN`, `*_FINAL_NN_v2`), com 79 checagens por safra. Corrige os erros das bases antigas: duplicatas, clima previsto desalinhado (usa a previsão emitida no mês anterior ao alvo), oscilações oceânicas com data na segunda/terça, Afya diário somado por semana (NaN antes de 2021) e ilhas sem clima (imputação pelos 3 vizinhos mais próximos).

```bash
python scripts/01_refazer_municipal.py --season 1
```

Rode uma vez por safra (1–4). Para as variáveis do SHAP corrigido, rode depois `--etapa nn_v2`. Também regenera a vizinhança municipal (`nos_season{N}.csv` e `W1_queen_season{N}.npz`), que está pronta em `dados/derivados/` (cópia da safra 1) para os scripts 02 e 03 não dependerem desta etapa.

## 4. Variáveis usadas pelos modelos

| Grupo | Exemplos | Quem usa |
|---|---|---|
| Casos da própria UF | log-incidência, últimas 4/8/13 semanas, tendência de 8 semanas, relativo à mesma época dos anos anteriores | todos |
| Vizinhança | log-incidência dos vizinhos (W1/W2), média da macrorregião | redes de grafo, GBMs, detector |
| Imunidade | tamanho das últimas temporadas relativo ao normal, temporadas desde o último ano grande | detector, GBMs |
| Fase da epidemia | Rt pela equação de renovação, variação do Rt, semanas desde a entressafra, nível ÷ p75 da UF | GBMs da rodada 2 (`fase`) |
| tsfresh | 40 características da janela de 52 semanas (seleção FRESH só até 2021) | `GBM_xgb_tsf_sem` |
| Clima | anomalias de temperatura, chuva e umidade (12 e 26 semanas), SEAS5 | GBMs `todas`, V4 B |
| Oceano | ENSO e sua variação de 12 semanas, IOD, PDO | detector, GBMs |
| Calendário | seno/cosseno da semana, Fourier (K = 3) | ARIMA, redes |

O que a pesquisa mostrou sobre variáveis: quase toda a informação útil está na própria série (nível, sazonalidade, último pico) e na vizinhança. Clima, ENSO e SEAS5 acrescentam pouco fora da amostra (ver [01_historico.md](01_historico.md)).
