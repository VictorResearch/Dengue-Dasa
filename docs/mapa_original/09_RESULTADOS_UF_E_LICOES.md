# 09 · Resultados dos modelos por UF (CompUF1–4 + Comp5) e lições (01/10/2026)

> **Fonte:** saídas do `CompUF4_avaliacao` e do `Comp5_Negocio` rodados no Colab em 01/10. São 83 modelos (UF direto + região→UF), 79 com as 4 temporadas de afinação.
> **Protocolo:** tudo escolhido em 2018–2021 (régua: WIS vs climatologia); o teste 2022–2025 (S1–S4, vs BB oficial) foi olhado uma vez só.
> **Arquivos no Drive:** `resultados_uf/{afinacao_ranking,teste_ranking}.csv`, `final_uf.parquet`, `ensembles_uf.parquet`, `resultados_competicao/negocio/`.

## 1. Régua da competição (WIS)

**Final pré-registrado: `ENS_UF_top5`**, a mediana dos quantis de 5 modelos:
- DoisEstagios_UF+λ, MTGNN_UF+λ, STGCN_UF+λ, GCNGRU_UF+λ (os quatro com λ = 0,8) e XGBoost_quantis_UF;
- **afinação: 0,798 × climatologia**;
- **teste: 0,853 × BB, 20 UFs melhores que o BB**. É melhor que o ARIMA_Fourier da competição (0,865) e que todas as equipes do 3º IMDC (a melhor, emap sarimax_state, fez 0,870).

| Modelo | Afinação (× clim.) | dp | Teste (× BB) | Observação |
|---|---|---|---|---|
| **MTGNN_UF+λ** | 0,822 | 0,30 | **0,846** | melhor modelo isolado no teste |
| ENS_UF_top3 / top7 | 0,801 / 0,806 | — | 0,850 / 0,850 | ensembles estáveis |
| **ENS_UF_top5 (final)** | 0,798 | — | 0,853 | |
| A3TGCN_UF+λ | 0,848 | 0,22 | 0,855 | o mais estável (dp 0,22) |
| ARIMA_Fourier (região→UF) | 0,922 | 0,35 | 0,865 | vencedor da competição de 30/09 |
| GCNGRU_UF+λ | 0,835 | 0,25 | 0,881 | |
| GBM_lgb_selecionadas_UF | 0,884 | 0,63 | 0,883 | |
| VAR_macro(X)_UF | 0,940 / 0,917 | ~0,49 | 0,895 / 0,892 | |
| AutoARIMA_d0 (r / py) | 0,895 / 0,898 | 0,34 | 0,899 / 0,904 | |
| ARIMA_Fourier_UF | 0,893 | 0,35 | 0,909 | |
| STGCN_UF+λ | 0,830 | 0,32 | 0,932 | |
| DoisEstagios_UF+λ | 0,818 | **0,59** | **0,970** | 1º na afinação, caiu no teste (instável) |
| XGBoost_quantis_UF | 0,847 | 0,57 | 1,021 | idem |
| GHR_NL_INLA_UF+λ | 0,850 | 0,37 | 1,017 | idem |
| AGCRN_UF (+λ) | 0,865 (0,855) | 0,55 | 1,126 (1,030) | |
| AutoARIMA livre (py / r) | 1,054 / 1,051 | 0,7 | 1,388 / 1,394 | extrapola a tendência |
| Redes do neuralforecast | 0,88–1,17 | 0,7–0,9 | 1,03–2,14 | |
| Chronos-2 (sem / com cov.) | 0,930 / 1,035 | 0,7–0,8 | 1,137 / 1,568 | |

**Por temporada** (× BB, 2022 → 2025):
- ENS_UF_top5: 0,959 · 0,980 · 0,792 · 0,712;
- ARIMA_Fourier: 1,011 · 1,021 · 0,754 · 0,719.

O ensemble ganha nos anos difíceis (2022, 2023) e empata ou perde pouco em 2024 e 2025.

**UF direto × região→UF (afinação):** prever direto por UF foi melhor para **todos** os modelos portados, exceto Prophet e Renovação. AGCRN: 1,157 → 0,865; BB_INLA: 1,196 → 0,914; XGBoost: 0,914 → 0,847. No teste, porém, o ARIMA_Fourier por região (0,865) superou o mesmo modelo por UF (0,909).

## 2. Régua da compra (Comp5, τ* = 0,75, temporada inteira, por UF)

Modelos escolhidos pelo custo na afinação e conferidos no teste (custo × climatologia; falta ideal ≈ 25%):

| Modelo | Custo afinação | Custo teste | Falta (afin. / teste) | Sobra (afin. / teste) |
|---|---|---|---|---|
| **DoisEstagios_pool_UF** | **0,886** | **0,871** | 30% / 27% | 44% / 11% |
| DoisEstagios_pool_sel_UF | 0,932 | **0,846** | 28% / 28% | 45% / 13% |
| SURGE_UF | 0,977 | 0,961 | 29% / 37% | |
| DoisEstagios_UF | 0,982 | 0,913 | 38% / 31% | |
| A3TGCN_UF | 0,989 | 0,899 | 38% / 37% | |
| ENS_UF_top5 (final do WIS) | 1,118 | 0,968 | 52% / 46% | 19% / 9% |
| MTGNN_UF | 1,099 | 0,904 | 43% / 40% | |

- **O campeão do WIS não é o campeão da compra.** O DoisEstagios_pool (pooling das 435 regiões + 26 UFs) tem WIS só razoável: 0,906 na afinação e **1,285 no teste**. Mesmo assim, é o mais consistente no custo, porque prevê o **total da temporada** diretamente e erra pouco no total.
- **Viés para baixo:** quase todos faltam mais que os 25% ideais (40–50%) e subestimam o total. O ensemble final faltou em 46–52% das UF × temporada.
- **1º semestre (SE41→SE14):** ninguém bate a climatologia na afinação. No teste, A3TGCN (0,881) e STGCN (0,875) ganham. Com 16–42 semanas de antecedência, prever o semestre é difícil.
- **Por região de saúde:** os modelos mal batem a climatologia (ARIMA_Fourier 0,98).
- **Aviso:** a seção 6 do Comp5 (as métricas entre si) misturou modelos de 8 temporadas com as equipes do IMDC (4 temporadas). As correlações de lá não são comparações justas. Corrigir: filtrar `temporadas == máximo` antes de ordenar.

## 3. Arquitetura: o que os vencedores têm em comum

Base comum das redes de grafo vencedoras (`_RedeGrafoUF`, `uf_lib.py`):
1. **Saída direta das 68 semanas:** prevê de uma vez SE41→SE40, sem rodar passo a passo e acumular erro.
2. **Âncora + resíduo:** o alvo é log1p(incidência) − climatologia da semana. A rede só aprende o desvio do normal.
3. **Perda quantílica (pinball) nos 9 quantis**, a própria régua do WIS. A AGCRN do V4 usa NLL da binomial negativa, e por isso perde: 0,865 contra 0,82–0,85.
4. **Grafo das 26 UFs** (fronteiras normalizadas). O MTGNN soma a ele um grafo **aprendido e direcionado** (top-4 vizinhos por UF).
5. **Janela de 104 semanas** (2 anos: entra o pico do ano anterior). Entradas:
   - log-incidência, 4 anomalias de clima, sen/cos da semana e ENSO;
   - 11 variáveis da origem (taxas de ataque de 1–3 anos, últimas 4/8 semanas, tendência, vizinhos, clima de 12 semanas, população);
   - embeddings da UF e da semana.
6. **Pequenas e com várias sementes:** d = 48, 60 épocas fixas, 3 sementes combinadas pela mediana. São as mais estáveis (dp 0,22–0,32, contra 0,5–0,67 dos GBMs).
7. **λ = 0,8 em todas as redes:** os intervalos saem largos demais. Encolher 20% em log melhora o WIS 1–4%.

**MTGNN** (Wu et al. 2020), o melhor isolado:
- **grafo aprendido** `relu(tanh(3(M1M2ᵀ − M2M1ᵀ)))`, assimétrico: "A influencia B" ≠ "B influencia A";
- **mix-hop:** combina 1 e 2 saltos no grafo, com pesos;
- **convoluções causais dilatadas** (1, 2, 4, 8, 16): enxerga 104 semanas com poucos parâmetros.

**A3T-GCN:** GRU + GCN + **atenção temporal** sobre todos os estados, que pode olhar direto para o pico do ano anterior. É o mais estável.

**DoisEstagios_pool**, o melhor para compra:
- estágio 1: GBM prevê o **tamanho** e o **pico** da temporada com as variáveis da SE25, treinado com regiões + UFs (461 séries × anos);
- estágio 2: a **forma** vem das temporadas passadas.

Ele acerta o total, mas erra o formato semana a semana.

## 4. Variáveis

| Evidência | Resultado |
|---|---|
| Ablação do GBM (afinação) | tirar qualquer grupo muda o R em **< 1%**: estáticas +1,0%; afya, espalhamento, seas5 ±0; chik, oceano, imunidade física −0,6 a −0,8% (atrapalham levemente) |
| GBM selecionadas × todas × casos (teste) | **0,883** × 1,054 × 0,924. A seleção protege: com todas, sobreajusta no teste |
| Exógenas futuras no ARIMAX | pioram: 0,925 contra 0,899–0,904 sem |
| VARX (ENSO) | neutro a leve ganho: macro 0,892 × 0,895; espacial 0,912 × 0,933; BVAR piora |
| Chronos-2 com covariáveis | piora muito: 1,568 contra 1,137 |
| neuralforecast com exógenas | ajuda (NHITS 1,03 contra 1,37 sem), mas a família continua ruim |
| Espaço (grafo/vizinhos) | é o que mais diferencia: as redes de grafo ganham dos mesmos insumos sem grafo (GBM, STID 1,15–1,08) |

**Lição:** quase toda a informação útil está na **própria série** (nível, sazonalidade, último pico) e na **vizinhança espacial**. Clima, ENSO e SEAS5 acrescentam pouco ou nada fora da amostra, e com muitas variáveis o modelo sobreajusta.

## 5. Próximos passos sugeridos

1. **Ensemble só de grafos com λ** (MTGNN + A3TGCN + GCNGRU), escolhido na afinação.
2. **MTGNN v2:**
   - 5 sementes;
   - λ por horizonte (calibração conformal);
   - cabeça auxiliar do **total da temporada** com perda pinball no total (alinha com a compra);
   - pooling com as regiões (o que fez o DoisEstagios_pool ser o melhor na compra);
   - ablação das entradas (só casos × casos + clima) dentro da própria rede.
3. **Para a empresa:** usar o total do DoisEstagios_pool e a forma das redes de grafo, reescalando as trajetórias da rede para a distribuição do total. Corrigir o viés para baixo (cauda superior), escolhendo pela régua de custo.
4. **Corrigir a seção 6 do Comp5** (comparar só modelos com o mesmo conjunto de temporadas).
