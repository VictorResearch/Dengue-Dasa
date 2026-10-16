# 17 · Literatura sobre previsão de picos e explosões, e próximos passos (13/10/2026)

> **Pedido:** busca extensa (PubMed e Consensus) sobre modelos para picos e explosões de dengue com semanas a meses de antecedência, e o que dá para implementar sem vazar dados do futuro.

## Respostas rápidas

- **A rodada 3 não retreina o modelo normal.** As previsões do ENS_grafos3_λ já gravadas são lidas e pós-processadas: as trajetórias são reescaladas pelo tamanho que o detector previu. Só o detector é treinado: modelos pequenos, uma vez por temporada, com janela expansiva.
- **Se for retreinar:**
  1. primeiro o **DoisEstagios** com o detector no lugar do estágio 1 (tamanho × forma; CPU, minutos);
  2. depois o **MTGNN_pesopop** condicionado ao tamanho do detector, com alvo relativo ao normal da UF e aumento sintético de temporadas grandes (GPU).
- **Ambientes:** R3_1, R3_2 e R3_3 rodam em CPU padrão, em sequência. Use High-RAM só se o R3_2 faltar memória ao ler as previsões.

## O que a literatura diz (e onde entra no projeto)

| Achado | Fonte | Como usar aqui |
|---|---|---|
| Ensembles de vários modelos superam modelos isolados. A habilidade no começo da temporada é baixa e fica ainda menor justamente nas temporadas grandes | Johansson et al. 2019, PNAS, [10.1073/pnas.1909865116](https://doi.org/10.1073/pnas.1909865116) | Manter ensemble; medir à parte o desempenho nos anos grandes |
| No IMDC 2024, nenhum modelo foi o melhor sempre, sobretudo em 2024; o ensemble foi adotado pelo Ministério | Araujo et al. 2026, PNAS, [10.1073/pnas.2508989123](https://doi.org/10.1073/pnas.2508989123) | Idem; o ensemble com pesos dependentes do regime faz sentido |
| A **estabilidade do período entre surtos** reflete o tamanho do grupo de suscetíveis e prevê o tamanho do surto seguinte com 3–4 meses de antecedência | Rypdal et al. 2019, Nat Commun | Novo sinal do detector: retorno à média, autocorrelação e variância dos casos de abril a setembro |
| Sinais de alerta precoce por desaceleração crítica (variância e autocorrelação crescentes) detectam reemergência | Brett et al. 2020, PLoS Biol, [10.1371/journal.pbio.3000697](https://doi.org/10.1371/journal.pbio.3000697) | Idem (tsfresh já calcula parte disso) |
| Clima + **ciclos de suscetibilidade** preveem anos epidêmicos em cidades do Brasil com meses de antecedência; a imunidade melhora a previsão baseada só no clima | McGough et al. 2021, J R Soc Interface | Reconstruir os suscetíveis (incidência acumulada × fator de subnotificação, com perda de imunidade) |
| **Competição entre sorotipos** + clima preveem surtos grandes com até 2 meses de antecedência (Singapura) | Finch et al. 2025, Nat Commun, [10.1038/s41467-025-66411-6](https://doi.org/10.1038/s41467-025-66411-6) | Fração de cada sorotipo por UF (LACEN/MS) como sinal externo |
| O DENV-3 (linhagem 3III_B.3.2) reemergiu no Brasil em 2023, com pelo menos 14 introduções | Ferreira et al. 2026, Lancet Reg Health Am, [10.1016/j.lana.2026.101521](https://doi.org/10.1016/j.lana.2026.101521) | Explica parte de 2024; reforça o sinal de sorotipo |
| A temperatura do Oceano Índico (índice IOBW) prevê as tendências globais de dengue | Chen et al. 2024, Science, [10.1126/science.adj4427](https://doi.org/10.1126/science.adj4427) | Acrescentar o IOBW (NOAA) ao detector |
| A temperatura do mar é o melhor preditor com 3 e 6 meses de antecedência; modelos de hidroclima ganham quando a autorregressão falha | Beal et al. 2025, GeoHealth, [10.1029/2024GH001325](https://doi.org/10.1029/2024GH001325) | ENSO e IOBW previstos (pluma do IRI) |
| Seca extrema aumenta o risco 3–5 meses depois (mais nas capitais com falta d'água); chuva extrema, 0–3 meses depois | Lowe et al. 2021, Lancet Planet Health, [10.1016/S2542-5196(20)30292-8](https://doi.org/10.1016/S2542-5196(20)30292-8) | Índice de seca (SPEI) calculado do ERA5 que já temos |
| Superensemble com previsão climática sazonal: até 6 meses à frente; detectou 69% dos surtos contra 54,5% do modelo de referência | Colón-González et al. 2021, PLoS Med, [10.1371/journal.pmed.1003542](https://doi.org/10.1371/journal.pmed.1003542) | Pesos por antecedência (já temos o E4) |
| Protótipo de alerta precoce para o Brasil (Copa de 2014), 3 meses à frente: 57% de acertos contra 33% do modelo nulo | Lowe et al. 2016, eLife, [10.7554/eLife.11285](https://doi.org/10.7554/eLife.11285) | Referência para a avaliação por categorias (baixo, médio, alto) |
| Modelos LASSO com 3 meses de antecedência em uso operacional em Singapura (surto de 2013) | Shi et al. 2016, EHP, [10.1289/ehp.1509981](https://doi.org/10.1289/ehp.1509981) | — |
| Metapopulação + filtro de Kalman por conjuntos (EAKF) prevê tamanho e pico até 10 semanas antes | Zeng et al. 2023, PLoS NTD | Atualizar a temporada semana a semana a partir de outubro (temporada rolante) |
| Aumento sintético com surtos grandes reduziu o erro em até 80% quando o teste passou do máximo do treino (São Paulo, 2024) | Al Bannoud et al. 2026, Acta Trop | Treinar as redes com temporadas reescaladas |
| Duas etapas: primeiro prever *se* haverá casos, depois *quantos* | Francisco et al. 2024, PLoS NTD | É o desenho do detector + mistura |
| Ensembles treinados + inferência conforme para as faixas | Mills et al. 2025, R Soc Open Sci | Calibrar a cauda de cima por conformal |
| A densidade populacional em escala fina explica o tamanho das ondas sucessivas (Rio, DENV-4) | Romeo-Aznar et al. 2022, Nat Commun, [10.1038/s41467-022-28231-w](https://doi.org/10.1038/s41467-022-28231-w) | Suscetíveis estratificados pela densidade dos municípios |
| Dados de busca na internet trazem pouco ganho além dos casos | Vietnã 2026, [10.1016/j.ijmedinf.2026.106675](https://doi.org/10.1016/j.ijmedinf.2026.106675); Koplewitz 2022, [10.1371/journal.pntd.0010071](https://doi.org/10.1371/journal.pntd.0010071) | Prioridade baixa; a positividade de exames da DASA deve valer mais |

## Plano (rodada 4, depois de ver a rodada 3)

1. **Detector+ (CPU):** acrescentar estabilidade entre surtos, sinais de desaceleração crítica, SPEI, IOBW, suscetíveis reconstruídos e, como externos, ENSO previsto, sorotipo, vacinação e positividade da DASA.
2. **DoisEstagios_detector (CPU):** o detector no lugar do estágio 1; a forma vem das temporadas análogas.
3. **MTGNN condicionado (GPU):** o tamanho do detector entra na cabeça da rede, o alvo é relativo ao normal da UF e o treino inclui aumento sintético de temporadas grandes.
4. **Temporada rolante (CPU):** renovação/SIR com EAKF, atualizado semana a semana de outubro a março, ligando a temporada ao semanal.
5. **Ensemble por regime e calibração conforme da cauda de cima.**

**Regra para não vazar o futuro:**
- janela expansiva;
- escolha de sinais e parâmetros só em 2018–2021;
- teste 2022–2025 olhado uma vez;
- 2025/26 avaliado prospectivamente.
