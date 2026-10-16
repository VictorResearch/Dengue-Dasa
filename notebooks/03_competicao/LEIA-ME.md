# Competição de modelos (regiões de saúde → UF)

> Criado em 30/09/2026. Todos os modelos da revisão da literatura e do 3º IMDC competem com os **nossos dados**, nas **mesmas regras** (só até a SE25, previsão SE41→SE40) e com a **mesma forma de somar** até a UF.

## Resultado (30/09, S1–S4)

**Vencedor: ARIMA_Fourier, 0,865 × BB** (19 UFs melhores; o 1º do 3º IMDC teve 0,870). Também é o melhor por região: 0,878 × climatologia, 343/435 regiões. A AGCRN ficou em 1,378. Tabelas em `resultados_colab_2026-09-30/`; análise completa em `_MAPA_DO_PROJETO/05_V4_E_COMPETICAO.md` §4. As 4 redes do neuralforecast falharam por `val_size` (já corrigido; falta rodar).

## O que subir para `MyDrive/dengue_V4/`

A pasta já tem `dados/` e `referencias_imdc3/`. Suba:
- `comp_lib.py`, `comp_modelos.py` e `comp_inla.R` (desta pasta);
- **`v4_lib.py` novo**, de `../V4/` (substitui o antigo, porque agora guarda as trajetórias por região);
- os 4 notebooks `Comp1_CPU`, `Comp2_INLA`, `Comp3_GPU` e `Comp4_Avaliacao`.

## Como rodar (1 ou mais sessões)

| Notebook | Modelos | Ambiente | Tempo estimado |
|---|---|---|---|
| `Comp1_CPU` | climatologia, canal_endemico, AR_longo, SURGE, ARIMA_Fourier, ETS_MSTL, Prophet, LightGBM_direto, Renovacao_S, Tamanho_x_forma (10 temporadas) | CPU | 1–1,5 h |
| `Comp2_INLA` | BB_INLA, GHR_INLA (2020–2025). Rode primeiro com `TESTE_RAPIDO = True`. O INLA roda do disco local (`/content/R_libs`), porque o Drive bloqueia executáveis | CPU High-RAM | a medir |
| `Comp3_GPU` | XGBoost_quantis, Chronos (10 temporadas); DeepAR, NHITS, TFT, TSMixer, AGCRN (2020–2025) | GPU L4 | 5–8 h |
| `Comp4_Avaliacao` | rankings por UF e por região, gráficos, diagnóstico, fidelidade, ensembles | CPU | minutos |
| `Comp5_Negocio` | métricas de compra de testes: custo de falta/sobra no quantil τ*, calibração do total, MAE/WAPE/sMAPE, compra sugerida por UF. **Não roda modelos**: lê os quantis já gravados. Saída em `resultados_competicao/negocio/` | CPU | minutos |

- **1, 2 e 3 podem rodar ao mesmo tempo**, em sessões separadas: cada um grava arquivos próprios em `resultados_competicao/`.
- Tudo retoma de onde parou.
- Um modelo que falhar é registrado em `erros_<notebook>.csv`, e os outros continuam.

## Arquivos

| Arquivo | O que é |
|---|---|
| `comp_lib.py` | Núcleo: dados, execução padronizada, quantis → amostras, reordenação de Schaake, soma até a UF, avaliação, diagnóstico e gráficos |
| `comp_modelos.py` | Os 19 competidores |
| `comp_inla.R` | BB e GHR em R-INLA (modo compact → classic se falhar; estratégia gaussian; amostras só dos efeitos) |
| `_gerar_notebook_comp.py` | Gera os 4 notebooks (edite aqui) |
| `_testes/` | Scripts e resultados dos testes no PC |

## O que já foi testado no PC (temporada 2022, S1)

| Modelo | Rodou? | Tempo/temp. | R_vs_BB 2022 | Observação |
|---|---|---|---|---|
| AR_longo | ✅ | 10 s | **1,03** | Viés ~0. Tinha explosões em log2; agora há teto de 3× o máximo histórico |
| climatologia | ✅ | 1 s | 1,04 | Régua |
| canal_endemico | ✅ | 2 s | 1,09 | |
| SURGE | ✅ | 3 s | 1,15 | Fiel ao original do LNCC (1,20) |
| Renovacao_S | ✅ | 5 s | 1,18 | **Melhor por região** (0,98 vs climatologia). Intervalos estreitos na UF: a incerteza foi aumentada depois do teste |
| LightGBM_direto | ✅ | 76 s | 1,24 | Refeito como regressão quantílica; viés +0,1 |
| XGBoost_quantis | ✅ | 8 min em CPU | 1,25 | No Colab usa GPU |
| Tamanho_x_forma | ✅ | 8 s | 1,34 | Superestima (+0,4), como no V4 |
| Prophet | ✅ | 37 s | 1,44 | Intervalos estreitos (IC90 56%) |
| ARIMA_Fourier | ✅ | 2 min | 2,90 | Superestimava (tendência). **Corrigido depois do teste** (sem drift, d=0); não foi rodado de novo |
| ETS_MSTL | ✅ | 12 s | 3,68 | Mesmo problema. **Corrigido depois do teste** (só nível); não foi rodado de novo |
| BB_INLA, GHR_INLA | ⚠️ | — | — | O executável do INLA para Windows travou no modo padrão, e o clássico era lento demais. O script foi otimizado (ver acima), mas **o teste real fica para o Colab (Linux)** |
| DeepAR, NHITS, TFT, TSMixer, Chronos | ❌ | — | — | Não testados (o PC não tem GPU). Instalação conferida num ambiente isolado |
| AGCRN | ✅ (no V4) | — | — | Mesma classe do V4, já testada lá |

**O notebook de avaliação (`Comp4`) também foi ensaiado** com esses resultados: tabelas, mapa de calor, estabilidade, macrorregiões, alvos da temporada, ranking por região, boxplot, leques e diagnóstico rodaram. O ensemble só roda quando existirem os resultados de 2020–2025.

Para referência, em 2022 os melhores do 3º IMDC tiveram 0,90 (EMAP Prophet) e 0,92 (LNCC ARP26).
