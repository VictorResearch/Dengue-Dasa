# 06 · Produção: previsão operacional

A pasta [producao/](../producao/) roda os 5 melhores de cada nível com os dados mais recentes. Ela **não depende** dos notebooks da pesquisa: os parâmetros escolhidos na afinação estão em `producao/parametros/`.

## Passo a passo

1. **Dados novos:** baixe e processe (scripts 00, 02 e 03). Para usar dados depois de 03/2026, baixe com `--atualizacoes` e junte os `*_update_2026` aos arquivos-base antes do script 02.
2. **Área de trabalho:**

   ```bash
   python scripts/04_montar_area_de_trabalho.py
   ```

   No Colab, suba `area_de_trabalho/` como `MyDrive/dengue_V4/`.
3. **Semanal:** abra `producao/Producao_Semanal.ipynb` (GPU) e rode tudo.
4. **Temporada:** abra `producao/Producao_Temporada.ipynb` (GPU) e rode tudo. Só faz sentido a partir de junho, quando a SE25 da temporada nova está nos dados; antes disso, ele prevê a temporada anterior.
5. **Relatório:** abra `producao/Relatorio_Campeoes.ipynb` (CPU). Ele gera os gráficos de todas as UFs, regiões e Brasil e o documento `relatorio_campeoes/modelos_campeoes.html`.

## Saídas

| Arquivo (na área de trabalho) | Conteúdo |
|---|---|
| `producao_semanal/previsao_semanal_<data>.parquet` | top 5 semanal + os 9 membros: UF × semana-alvo × 9 quantis (casos) |
| `producao_semanal/tabela_semanal_<data>.csv` | campeão, por UF e semana-alvo: mediana e faixas de 50% e 90% |
| `producao_temporada/previsao_temporada_<s>.parquet` | top 5 de temporada + o normal, SE41 → SE40 |
| `producao_temporada/resumo_temporada_<s>.csv` | campeão, por UF: total previsto (mediana e faixa), semana e altura do pico, P(ano grande), tamanho previsto pelo detector |
| `producao_temporada/detector_<s>.csv` | quantis do detector por UF |
| `relatorio_campeoes/modelos_campeoes.html` | documento autocontido dos campeões, com os gráficos |
| `relatorio_campeoes/figuras/{semanal,temporada}/<unidade>.png` | os gráficos, um por unidade |

Cada arquivo com data tem também uma cópia `*_ultimo.*`.

## A última semana incompleta

Os casos são contados pela semana de início dos sintomas, e as notificações chegam com atraso. **A última semana dos dados está sempre incompleta**: em 03/2026 tinha ~36% dos casos da semana anterior (mediana entre UFs). Por isso a origem padrão da previsão semanal é a **última semana completa** (`DESCARTAR = 1` no notebook, `prod_lib.SEMANAS_INCOMPLETAS`). Assim:
- h = 1 re-estima a semana incompleta;
- h = 2, 3 e 4 são as semanas seguintes.

Nos gráficos, a semana incompleta aparece como um círculo vazio, fora da linha.

## Robustez

- **Retomada:** cada membro grava um arquivo por rodada; rodar de novo pula o que existe.
- **Membro que falha:** o `ENS_stack_sem` renormaliza os pesos entre os membros que rodaram. Os erros aparecem na saída da célula 3.
- **Teste rápido sem GPU:** `SO_MEMBROS = ['ARIMA_Fourier_sem', 'GBM_xgb_todas_sem']` no semanal; `NORMAL_SEM_GPU = True` na temporada (troca o normal pela climatologia; **não** é o campeão).
- **Conferir que o código reproduz a pesquisa:** `RODAR_BACKTEST = True` refaz o teste 2022–2025 e grava `backtest_metricas.csv`. O detector de produção reproduz o da pesquisa com diferença de 10⁻¹⁶.

## Quando retreinar

- **Semanal:** a cada semana nova; cada execução retreina todos os membros com os dados até a origem.
- **Temporada:** uma vez por ano, em junho (SE25). As revisões mensais de julho a outubro ganharam pouco na pesquisa (rodada 3).
- **Parâmetros** (pesos, hiperparâmetros, λ, misturas): foram escolhidos em 2018–2021 e conferidos em 2022–2025. Para reescolhê-los com temporadas novas, rode de novo as etapas da pesquisa (07 para o stacking semanal, 12 para os ensembles do pico).

## Limitações conhecidas

- O detector acerta a direção (correlação 0,82), mas encolhe os anos extremos e erra metade dos alarmes de ano grande.
- O `MIX_escala5_UF` (2º do top 5 de temporada) infla a entressafra em +151%: use-o para o pico, não para meses de baixa.
- As faixas das regiões e do Brasil nos gráficos são a soma das faixas das UFs: conservadoras.
