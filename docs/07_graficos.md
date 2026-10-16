# 07 · Gráficos por UF, macrorregião e Brasil

## Módulo principal: [graficos/graficos_previsoes.py](../graficos/graficos_previsoes.py)

| Função | O que faz |
|---|---|
| `observado(D, incompletas=1)` | casos observados por semana para as 26 UFs, as 5 macrorregiões e o Brasil; marca as semanas incompletas |
| `agregar_previsoes(P)` | acrescenta regiões e Brasil a previsões por UF (soma dos quantis) |
| `figura_semanal(OBS, hist, futuro, unidade, arq)` | painel largo com o observado e o previsto 1 semana antes no histórico + zoom dos últimos 6 meses com o leque das próximas 4 semanas |
| `figura_temporada(OBS, prev, unidade, arq)` | cada temporada (SE41 → SE40) prevista em junho, com faixas de 50% e 90%, sobre o observado |
| `figuras_todas(OBS, pasta, semanal, temporada)` | grava as figuras de Brasil, regiões e UFs em `pasta/semanal/` e `pasta/temporada/` |
| `ordem_unidades()` | Brasil, Norte, Nordeste, Sudeste, Sul, Centro-Oeste e as UFs agrupadas por região |

Exemplo (dentro da área de trabalho):

```python
import uf_lib as U, graficos_previsoes as G
ctx = U.ContextoUF('dados', 'referencias_imdc3', 'dados_uf')
OBS = G.observado(ctx.Du)
G.figura_temporada(OBS, G.agregar_previsoes(previsoes), 'Sul', 'sul.png')
```

**Regiões e Brasil** somam os quantis das UFs. A mediana somada aproxima bem a mediana da soma; as faixas somadas são **conservadoras** (mais largas que o intervalo verdadeiro da soma, porque supõem que todas as UFs erram juntas).

O notebook `producao/Relatorio_Campeoes.ipynb` usa este módulo para os dois campeões e monta o documento HTML.

## Gráficos da pesquisa

| Onde | O que mostra |
|---|---|
| `notebooks/07_rodada2_pico_entressafra/R2_7_Graficos_UF.ipynb` | por UF + Brasil: semanal (h = 1 e h = 4) e temporada dos melhores da rodada 2 |
| `notebooks/08_rodada3_detector/R3_3_Avaliacao_Graficos.ipynb` | normal × detector + mistura por UF, com P(ano grande) anotado |
| `notebooks/10_rodada5_modelo_por_fase/R5_Graficos_Comparacao.ipynb` | 6 modelos lado a lado por UF + mapa de calor do erro do pico |
| `notebooks/11_rodada6_optuna` e `12_rodada7_ensemble_pico` | R6 × R5 e ensemble × membros, por UF e Brasil; fronteira pico × entressafra |
| `graficos/graficos_colab_semanal_temporada.py` | leques semanal, de temporada e semanal × temporada da rodada semanal |
