# V4 · Previsão da temporada inteira (3rd IMDC)

> Criado em 29/09/2026 a partir do plano do [04_ESTADO_DA_ARTE.md](../../_MAPA_DO_PROJETO/04_ESTADO_DA_ARTE.md).
> O V4 prevê a tarefa **oficial**: dados até a SE25, previsão SE41 → SE40, avaliação por **WIS relativo ao modelo-base BB**.

## Arquivos

| Arquivo | O que é | Onde roda |
|---|---|---|
| `preparar_dados_v4.py` | Monta os dados compactos a partir de `../variaveis/` (~2 min) | PC |
| `dados/` | Painéis por região de saúde (435) e por UF (26), clima ponderado pela população, ENSO alinhado ao domingo, SEAS5, vizinhança e parâmetros de Richards. 37 MB | vai para o Drive |
| `referencias_imdc3/` | Previsões publicadas de todas as equipes do 3º IMDC (inclui o BB). 9 MB | vai para o Drive |
| `v4_lib.py` | Biblioteca: avaliador, modelos-base, etapa 5, versões A e B, recalibração e ensemble | vai para o Drive |
| `Modelo_v4.ipynb` | Notebook do Colab que roda tudo em ordem | Colab |
| `_gerar_notebook.py` | Gera o notebook (edite aqui, não no `.ipynb`) | PC |

## Como rodar no Colab

1. Crie `MyDrive/dengue_V4/` e suba `v4_lib.py`, `dados/` e `referencias_imdc3/`.
2. Abra `Modelo_v4.ipynb` no Colab, escolha **GPU L4** e rode tudo.
3. Os resultados ficam em `MyDrive/dengue_V4/resultados_v4/`:
   - `previsoes/` (uma por modelo e temporada; o notebook retoma de onde parou);
   - `resumo_v4_vs_imdc.csv`, `wis_por_uf_temporada.csv`, `ablacao_etapa5.csv`, gráficos e `submissao/`.

**Custo estimado:** as bases, a versão B e a ablação levam ~10 min. A_uf leva ~30 min. A_regiao, com 6 temporadas × 2 sementes, leva 3–6 h numa L4 (~15–30 unidades). Para um primeiro teste barato, rode com `RODAR_A_REGIAO = False`.

## O que já foi medido no PC (29/09, sem a versão A completa)

Safras S1–S4, WIS relativo ao BB (abaixo de 1 = melhor que o modelo-base):

| Modelo | R_vs_BB | UFs melhores que o BB | dp log R |
|---|---|---|---|
| 3º IMDC · EMAP SARIMAX (melhor média) | 0,870 | 17 | 0,45 |
| 3º IMDC · LNCC ARP26 | 0,897 | 18 | 0,41 |
| 3º IMDC · PURDUE (mais estável) | 0,904 | 20 | 0,29 |
| **V4 · ensemble das bases, pesos só do passado** | **0,954** | **18** | 0,35 |
| **V4 · ensemble das bases + B (mediana)** | **0,962** | **18** | **0,28** |
| V4 · AR longo por UF (estilo LNCC) | 0,980 | 14 | 0,45 |
| Climatologia | 1,012 | 16 | 0,31 |
| V4 · B por UF (sozinho) | 1,225 | 6 | 0,57 |

**Lições dos testes locais:**
- **O avaliador está certo.** O WIS bate com o oficial (razão 1,000 nas validações 1–2), e o ranking reproduz o do 3º IMDC (PURDUE 20 UFs, LNCC 18).
- **A versão B sozinha perde nas contagens, mas é a melhor em escala log** (WIS log 0,39–0,41 contra 0,42 do BB). Ela erra a magnitude depois de temporadas grandes (2025, depois do recorde de 2024). Com poucas temporadas, aprende "ano grande → ano grande" em vez de imunidade.
- **B no nível UF foi melhor que por região.** A forma média da curva foi melhor que sortear a forma de uma temporada.
- **Ablação da etapa 5** (configuração de teste): a tendência de 8 semanas e o clima prolongado ajudam; a taxa de ataque de 1–3 temporadas e a SEAS5 atrapalharam. Confira no Colab com a configuração completa.
- **A recalibração com temporadas passadas piorou o WIS.** As temporadas de calibração (2016–21) eram de outro regime, e a de 2024 pede intervalos enormes. Por isso ela fica desligada (`RECALIBRAR = False`).
- **O ensemble é o que mais ajuda**, como a literatura prevê.
- **Teste de fumaça da versão A** (PC, CPU, só 2 épocas e 24 origens por época, 1 semente): `A_uf` já ficou em **0,977** nas 4 safras, e `ENS_bases_A_pesos` em **0,944** (19 UFs melhores que o BB). É o treino mínimo; o treino completo no Colab deve melhorar. `A_regiao` rodou só 2 safras no teste, então não é comparável.

## Resultado no Colab (30/09, treino completo) · S1–S4, WIS relativo ao BB oficial

| Modelo | R_vs_BB | UFs melhores que o BB | dp log R |
|---|---|---|---|
| 3º IMDC · EMAP SARIMAX / LNCC ARP26 / PURDUE | 0,870 / 0,897 / 0,904 | 17 / 18 / 20 | 0,45 / 0,41 / 0,29 |
| **ENS_bases_A_mediana** (clim + canal + AR rec. + AR longo + A_uf) | **0,938** | 18 | 0,298 |
| **ENS_bases_AB_mediana** (idem + B_uf) | **0,944** | **19** | **0,265** |
| ENS_bases_pesos | 0,954 | 18 | 0,348 |
| AR longo | 0,980 | 14 | 0,446 |
| climatologia | 1,012 | 16 | 0,310 |
| A_uf (treino completo) | 1,121 | 11 | 0,676 |
| B_uf | 1,238 | 6 | 0,571 |
| A_regiao | 1,365 | 10 | 1,000 |
| B_regiao | 1,382 | 4 | 0,637 |

- A versão A com treino completo ficou **pior** que no teste de 2 épocas (A_uf 0,977 → 1,121): ela se afasta da âncora sazonal e sobreajusta. Dentro do ensemble pela mediana, porém, ajuda.
- A mediana dos membros venceu a média e os pesos em todos os conjuntos.
- ⚠️ O melhor ensemble foi escolhido entre ~20 variantes nas mesmas 4 safras, então o número é um pouco otimista.

## Decisões de desenho

- **Episcanner:** a API oficial exige chave e o Zenodo só tem padrões agregados até 2023. Os parâmetros (tamanho, taxa de crescimento e pico da curva de Richards) são calculados do próprio `dengue.csv` com o mesmo método. Assim não há dado externo a declarar aos organizadores.
- **`climate.csv` não tem `precip_tot`** (a documentação lista, o arquivo não tem): usamos `precip_med` e `rainy_days`. A SEAS5 tem `precip_tot` próprio.
- **Soma até a UF:** trajetórias correlacionadas. Em B, por sorteio do vetor de resíduos de uma temporada inteira; em A, por fatores multiplicativos compartilhados (país, UF, região).
- **Pesos do ensemble:** só com temporadas anteriores, observadas até o corte, e encolhidos 50% para pesos iguais.
