# 04 · Avaliação: protocolo e métricas

## Protocolo

| Regra | Detalhe |
|---|---|
| **Tarefa de temporada** | prever a temporada `s` (SE41/`s` → SE40/`s+1`) só com dados até a SE25/`s`: 16 a 68 semanas à frente, 9 quantis |
| **Tarefa semanal** | prever 1, 2, 3 e 4 semanas à frente a partir de cada semana de origem, com os dados até ela |
| **Afinação** | temporadas 2018–2021. Toda escolha (variáveis, hiperparâmetros, pesos, λ, limiares, detector) usa só elas |
| **Teste** | temporadas 2022–2025 (as 4 safras de validação do 3º IMDC). Olhado uma vez por rodada; onde uma decisão foi tomada depois de olhar, está marcado no histórico |
| **Janela expansiva** | cada temporada é prevista com modelos treinados só com o passado dela |
| **Rodada semanal** | o modelo é treinado uma vez por temporada, com alvos até 4 semanas antes da SE41, e aplicado a cada origem sem reajustar |
| **Validação aninhada** (rodadas 6 e 7) | a escolha inteira é refeita deixando cada temporada de 2015–2021 de fora e avaliada nela |
| **Avaliador** | `v4_lib.wis` / `v4_lib.avaliar`: reproduz o WIS oficial (razão 1,000 nas previsões publicadas) e o ranking do 3º IMDC |

## Métricas

| Métrica | O que mede | Como ler |
|---|---|---|
| **WIS** | *weighted interval score* sobre os 9 quantis (Bracher et al. 2021): pune o erro da mediana e faixas largas ou que não cobrem | menor é melhor; em casos por semana |
| **R × BB** | WIS ÷ WIS do modelo-base oficial (BB), média geométrica entre temporadas (por UF) e depois entre UFs | 0,808 = 19% menos erro que o BB, com cada UF pesando igual. É a régua do 3º IMDC |
| **R × climatologia**, **× persistência** | o mesmo, contra outras referências | usadas na afinação e no semanal |
| **MAE** | erro absoluto médio da mediana | casos por semana; dominado pelos estados grandes |
| **WAPE** | soma dos erros ÷ soma dos casos | 58% = errou o equivalente a 58% dos casos do período |
| **sMAPE** | média de 2·\|p − y\| / (\|p\| + \|y\|), escala 0–200 | trata UFs pequenas e grandes igual |
| **r** | correlação de Pearson entre mediana e observado | se a curva sobe e desce na hora certa |
| **IC50 / IC80 / IC90 / IC95** | % dos valores reais dentro de cada faixa | ideal: 50, 80, 90, 95 |
| **Viés** | soma (previsto − real) ÷ soma real | negativo = previsão baixa |
| **WIS por fase** | WIS só nas semanas de **pico** (incidência ≥ p75 da UF), **entressafra** (< p25) ou **intermediárias**; limiares com dados até 2017 | mostra onde o modelo erra |
| **WIS da fase ÷ referência** | a mesma razão por UF, média geométrica | objetivo das rodadas 6 e 7 |
| **Erro da altura do pico** | máximo da mediana prevista ÷ máximo observado − 1, mediana entre UF × temporada | 0 é o ideal; negativo = fica baixo |
| **Erro do total** | total previsto ÷ total observado − 1 | idem |
| **Pico dentro da faixa de 90%** | % das UF × temporada com o pico real dentro da faixa | ideal ≈ 90% |
| **Pinball** | perda quantílica | usada para escolher o detector (anos grandes valendo 3×) |
| **Custo de compra** | custo de comprar o quantil τ* do total (problema do jornaleiro) ÷ custo da climatologia | régua de negócio (Comp5) |

**Métricas absolutas × relativas.** A régua oficial (R × BB, média geométrica por UF) dá o mesmo peso a Roraima e a São Paulo, e premia previsões tímidas nas UFs pequenas. As métricas absolutas (WIS, MAE, WAPE) são dominadas pelos estados grandes, onde está o volume de casos. A produção escolheu os campeões pelas absolutas; o melhor pela régua oficial (`ENS_pico7_UF`, 0,808) está no top 5 de temporada.

**Critério misto** (rodadas 4 e 5): ½ R + ½ WIS absoluto relativo. Abandonado por ser difícil de interpretar.
