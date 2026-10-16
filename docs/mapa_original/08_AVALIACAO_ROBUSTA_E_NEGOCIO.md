# 08 · Avaliação robusta, métricas de negócio e generalização para a empresa (01/10/2026)

> **Pedido:** o ensemble e as métricas precisam funcionar com qualquer modelo novo, inclusive de fora, e quando os testes forem ampliados. Também: as métricas servem para uma empresa que compra testes com antecedência? Os modelos generalizam para "comprar no semestre que vem"?

---

## 1. O contrato: o que um modelo precisa entregar para entrar na avaliação

Qualquer previsão com **os 9 quantis** (mediana + intervalos de 50, 80, 90 e 95%) por unidade × semana entra em tudo: CompUF4 (WIS, λ, ensembles, teste) e Comp5 (negócio). Formatos aceitos por `L.padronizar_previsoes`:
- o nosso formato longo, que é o que `L.rodar` grava em `<pasta>/uf/<modelo>_s<temporada>.parquet`;
- o formato de submissão da plataforma: `date, adm_1, pred, lower_*, upper_*`, sem precisar de `modelo` nem de `season` (a temporada é deduzida da data);
- quantis nomeados por nível (`q0.025 … q0.975`, `median`).

**Modelo de fora:** `L.importar_externo(arquivo, nome, PASTA_EXT, ctx.Du.ufs)` checa e grava em `resultados_externos/`. No CompUF4 basta listá-lo em `EXTERNOS = {nome: arquivo}`. Um modelo só com previsão pontual **não** entra: sem intervalos não há WIS nem quantil de compra.

## 2. Leitura robusta (`L.carregar_previsoes`, em `comp_lib.py`)

| Situação | O que acontece |
|---|---|
| O mesmo (modelo, temporada) em duas pastas | fica a fonte listada por último; vai para o relatório |
| Linhas repetidas | removidas |
| Semanas fora da temporada ou unidades desconhecidas | descartadas |
| Previsão incompleta (falta semana ou UF, ou tem NaN) | **fora da avaliação** naquela temporada. Antes isso virava WIS = NaN calado no `avaliar`, ou **zeros inventados** no `ensemble` |
| Pasta inexistente | pulada |

O relatório fica em `L.ULTIMO_RELATORIO`. Ele **não** fica em `.attrs`, porque `.attrs` quebrava o `to_parquet` dos dados derivados (bug achado no ensaio).

Proteções também em `v4_lib`:
- `avaliar` deixa de fora (modelo, temporada) com buracos e avisa;
- `ensemble` pula a temporada em que algum membro está incompleto, em vez de preencher com 0;
- `_alinha` remove duplicatas.

## 3. Incidente de 01/10: duas pastas `resultados_uf`

O CompUF1 e o CompUF3 foram abertos ao mesmo tempo, e cada sessão criou uma pasta `resultados_uf` (12:04:39 e 12:04:47):
- **pasta A:** CompUF2 (CPU);
- **pasta B:** CompUF1 (portados) e CompUF3 (GPU).

Os três `erros_*.csv` vieram vazios, ou seja, nenhum modelo falhou. O Colab enxerga só uma das pastas.

**Correção:**
- o CompUF4 e o Comp5 leem **todas** as pastas `resultados_uf*`;
- renomear a cópia para `resultados_uf_2`;
- os cabeçalhos dos notebooks avisam para esperar a célula 1 do primeiro notebook antes de abrir os outros.

O CompUF4 também falhou por **versões misturadas**: o notebook novo estava no Drive com o `uf_lib.py` antigo (`carregar_uf() got an unexpected keyword argument 'ids'`).

**Ensaios no PC** (previsões sintéticas; 34 s): CompUF4 e Comp5 inteiros, com 2 pastas de resultados, modelo externo em CSV, duplicata, modelo incompleto e quantis desordenados. Tudo OK.

## 4. Métricas: quais servem para a empresa

O Comp5 traz as métricas da compra:
- `custo_vs_climatologia` (custo real da compra no quantil τ*);
- `falta_freq`, `falta_pct`, `sobra_pct`;
- `WAPE_total`, `vies_total`;
- as semanais (MAE, WAPE, sMAPE, WIS, IC90).

A seção 5 nova escolhe o modelo **na afinação (2018–21)** e só depois mostra o **teste (2022–25)**. A seção 6 nova compara **as métricas entre si**: a correlação de Spearman entre as ordens dos modelos e o vencedor de cada métrica.

**Leitura do ensaio (só com as previsões do 3º IMDC de 2022; ilustrativo):**
- O custo da compra concorda muito com o WAPE do total (ρ ≈ 0,86–0,90) e razoavelmente com o WIS (ρ ≈ 0,66–0,79).
- Concorda pouco com o MAE semanal (0,31–0,47) e com o sMAPE (0,46–0,53).
- Cada métrica deu um vencedor diferente. **Para a compra, a métrica de decisão é o custo**; o WIS serve de régua geral e o WAPE do total como métrica de leitura simples.

## 5. Generalização para a empresa

| O que já generaliza | O que é específico da competição |
|---|---|
| A previsão é uma **distribuição** (quantis/trajetórias): serve para qualquer τ* e qualquer custo | **Origem fixa na SE25** (`D.corte`) e **janela fixa SE41→SE40** (`V.datas_temporada`) |
| Trajetórias com dependência (Schaake): somam para qualquer área e período | Alvo = casos prováveis notificados por UF/região, não testes consumidos pela empresa |
| Perdas próprias (pinball, NLL da binomial negativa): não estão presas ao WIS | Dados **consolidados** (revisados depois); em tempo real as últimas semanas chegam incompletas (atraso de notificação) |
| Protocolo afinação → teste e origem móvel (sem vazamento) | Só **1 previsão por ano** é avaliada (8 temporadas = poucas amostras) |
| Climatologia como régua e custo do jornaleiro | Compra única (jornaleiro), sem estoque que passa de um período para o outro, prazo de entrega ou validade |

**Plano para a empresa (não executado):**
1. **Origem e janela configuráveis:** `ctx.origem(s)` e `ctx.alvo(s)` no lugar da SE25 e da SE41→SE40, por exemplo "decido na semana t, compro para t+L … t+L+26".
   - Os modelos com origens sintéticas (GBMDireto, redes, AGCRN) e os estatísticos (auto.arima, VAR, climatologia) se adaptam quase sem mudança.
   - Os ancorados na temporada (DoisEstagios, TamanhoForma, Richards, variáveis de temporada) precisam de variáveis "como vistas na origem".
2. **Avaliação com origens mensais** (ex.: toda SE01, SE14, SE27 e SE40 de 2015–2025): cerca de 4× mais casos de teste, e na época real de decisão da empresa.
3. **Alvo da empresa:** testes vendidos/consumidos por região de atuação; ou casos × uma taxa de testes por caso estimada com o histórico da empresa. Os casos suspeitos (inclusive os descartados) também consomem teste.
4. **Atraso de notificação:** usar versões dos dados como estavam na data (*vintages*) ou um *nowcast* das últimas semanas antes de prever.
5. **Política de estoque:** simular compras mensais com estoque que sobra, prazo de entrega e validade; o custo da sobra passa a ser custo de estoque + risco de vencer.
