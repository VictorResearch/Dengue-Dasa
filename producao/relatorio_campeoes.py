"""
Gera o documento simples dos MODELOS CAMPEÕES (um HTML autocontido, com as figuras embutidas):
  - qual é o campeão semanal e o de temporada, e as métricas de cada um no teste 2022–2025;
  - a arquitetura de cada um, em diagrama;
  - os gráficos de previsão de cada UF, macrorregião e Brasil (histórico + próximas semanas / temporada);
  - as tabelas da previsão mais recente.

Uso (dentro da área de trabalho, depois do notebook producao/Relatorio_Campeoes.ipynb gerar as figuras):
    import relatorio_campeoes as RC
    RC.gerar('modelos_campeoes.html', figuras=feitos, tab_semanal=TS, tab_temporada=TT, info=dict(...))
Sem figuras (figuras=None), o documento sai com as métricas e a arquitetura e um aviso no lugar dos gráficos.
"""
import base64
import html
import os

import pandas as pd

# ---------------------------------------------------------------------------------------------------- métricas (teste)
SEMANAL_H = [  # ENS_stack_sem, teste 2022–2025 (26 UFs × semanas), por horizonte
    dict(h=1, MAE=270, WAPE=12.3, sMAPE=20.4, R_clim=0.224, R_pers=0.884, IC90=90.1, vies=-6.5, r=0.992, MAE_pers=354),
    dict(h=2, MAE=383, WAPE=17.5, sMAPE=25.3, R_clim=0.293, R_pers=0.794, IC90=89.1, vies=-8.9, r=0.981, MAE_pers=637),
    dict(h=3, MAE=475, WAPE=21.7, sMAPE=29.6, R_clim=0.355, R_pers=0.746, IC90=88.8, vies=-8.7, r=0.967, MAE_pers=914),
    dict(h=4, MAE=583, WAPE=26.6, sMAPE=33.5, R_clim=0.404, R_pers=0.699, IC90=88.5, vies=-12.5, r=0.954, MAE_pers=1186),
]
SEMANAL_TOP5 = [  # MAE por horizonte (casos/semana) e média
    ('ENS_stack_sem', 270, 383, 475, 583, 428, True),
    ('ARIMA_Fourier_sem', 252, 398, 525, 646, 455, False),
    ('GBM_xgb_optuna_sem', 308, 403, 508, 639, 464, False),
    ('ENS_GBM_ARIMA_sem', 271, 414, 526, 648, 465, False),
    ('ENS_GBM_ARIMA_MTGNNcresc_sem', 267, 418, 544, 680, 477, False),
    ('Persistência (referência)', 354, 637, 914, 1186, 773, None),
]
TEMPORADA = [  # teste 2022–2025: R×BB, UFs>BB, WIS, WIS pico, WIS entressafra, MAE, WAPE, sMAPE, altura, total, pico no IC90
    ('ENS_pico7_simples_UF', 0.935, 16, 721, 2061, 29.6, 1277, 58.3, 67.5, 2.2, -0.6, 85.6, True),
    ('MIX_escala5_UF', 1.288, 4, 746, 2012, 55.1, 1279, 58.4, 70.4, 11.5, 8.6, 93.3, False),
    ('ENS_pico7_UF', 0.808, 22, 902, 2667, 27.3, 1452, 66.3, 65.7, -20.0, -15.6, 76.0, False),
    ('MIX_elastico5_UF', 0.859, 21, 1010, 2994, 27.8, 1506, 68.8, 70.2, -11.8, -7.8, 74.0, False),
    ('AJ_continuo_UF', 0.828, 23, 1285, 3883, 26.4, 1672, 76.3, 70.0, -28.2, -24.6, 64.4, False),
    ('BB (modelo-base oficial)', 1.0, None, 1224, 3651, 39.0, 1787, 81.6, 73.0, -36.3, -23.5, 73.1, None),
]

CSS = """
:root{--bg:#f6f7f8;--surface:#fff;--ink:#17232b;--muted:#5a6a73;--line:#d8e0e4;--blue:#1f6fb2;--blue-soft:#dce9f5;
--teal:#0e7468;--teal-soft:#d9efea;--orange:#c2410c;--orange-soft:#fbe4d8;
--fd:"Literata",Georgia,serif;--fb:"Public Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--fm:"JetBrains Mono",ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10171b;--surface:#172128;--ink:#e4ecef;--muted:#9aaeb6;--line:#2c3b43;
--blue:#6aaee8;--blue-soft:#17304a;--teal:#4cc4b3;--teal-soft:#15352f;--orange:#f08a55;--orange-soft:#3d2216;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#10171b;--surface:#172128;--ink:#e4ecef;--muted:#9aaeb6;--line:#2c3b43;--blue:#6aaee8;--blue-soft:#17304a;
--teal:#4cc4b3;--teal-soft:#15352f;--orange:#f08a55;--orange-soft:#3d2216;color-scheme:dark}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font-family:var(--fb);font-size:15.5px;line-height:1.6}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:36px 72px}
h1{font-family:var(--fd);font-size:clamp(28px,4vw,40px);line-height:1.15;margin:6px 0 12px;text-wrap:balance}
h2{font-family:var(--fd);font-size:25px;margin:0 0 8px;text-wrap:balance}h3{font-size:16.5px;margin:22px 0 8px}
p{max-width:75ch}.eyebrow{font-family:var(--fm);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--teal)}
.muted{color:var(--muted)}section{border-top:1px solid var(--line);margin-top:34px;padding-top:30px}
.champs{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:20px}
.champ{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px 18px}
.champ .k{font-size:12.5px;color:var(--muted)}.champ .n{font-family:var(--fm);font-size:20px;font-weight:600;margin:2px 0 6px;overflow-wrap:anywhere}
.champ ul{margin:8px 0 0;padding-left:18px}.champ li{margin:2px 0}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface);margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{background:var(--bg);color:var(--muted);font-weight:600;font-size:12px}td.n{text-align:right;font-family:var(--fm);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}tr.best td{background:var(--teal-soft)}tr.ref td{color:var(--muted)}
.note{font-size:13px;color:var(--muted)}figure{margin:14px 0;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}
figcaption{font-size:13px;color:var(--muted);margin-top:8px;max-width:90ch}svg.dia{width:100%;min-width:760px;height:auto;color:var(--ink)}
.picker{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 18px}.picker a{font-family:var(--fm);font-size:12px;padding:3px 8px;border:1px solid var(--line);
border-radius:4px;color:var(--ink);text-decoration:none;background:var(--surface)}.picker a:hover,.picker a:focus-visible{background:var(--blue-soft);outline:none}
.uni{margin:26px 0 8px;scroll-margin-top:12px}.uni h3{margin:0 0 6px}.uni img{width:100%;max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff;margin:4px 0}
.aviso{background:var(--orange-soft);border-left:3px solid var(--orange);padding:10px 14px;border-radius:4px}
code{font-family:var(--fm);font-size:12.5px}
"""


def _img(arq):
    with open(arq, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()


def _fmt(v, d=0, pct=False, sinal=False):
    if v is None:
        return '–'
    s = f'{v:+,.{d}f}' if sinal else f'{v:,.{d}f}'
    s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
    return s + ('%' if pct else '')


def diagrama_semanal():
    """arquitetura do ENS_stack_sem: 9 membros → pesos por horizonte × fase → média ponderada dos quantis."""
    caixas = [('Árvores · XGBoost quantílico', ['GBM_xgb_todas (todas as variáveis)', 'GBM_xgb_fase (+ fase e Rt)',
                                                'GBM_xgb_fase_L1 (mediana própria)', 'GBM_xgb_tsf (+ 40 do tsfresh)',
                                                'GBM_xgb_optuna (hiperparâmetros do Optuna)']),
              ('Estatístico', ['ARIMA + Fourier (sazonalidade)']),
              ('Redes de grafo das 26 UFs', ['MTGNN_cresc (prevê o crescimento)', 'MTGNN_cresc_escala (escala por UF)']),
              ('Modelo de fundação', ['Chronos-2 ajustado (LoRA)'])]
    y, partes = 30, []
    for tit, itens in caixas:
        h = 26 + 19 * len(itens)
        partes.append(f'<rect x="250" y="{y}" width="300" height="{h}" rx="6" fill="none" stroke="currentColor" stroke-opacity=".45"/>'
                      f'<text x="262" y="{y + 18}" font-size="12.5" font-weight="700" fill="currentColor">{tit}</text>'
                      + ''.join(f'<text x="270" y="{y + 38 + 19 * i}" font-size="12" fill="currentColor">{html.escape(t)}</text>' for i, t in enumerate(itens))
                      + f'<line x1="550" y1="{y + h / 2}" x2="606" y2="230" stroke="currentColor" stroke-opacity=".5" marker-end="url(#seta)"/>')
        y += h + 12
    return f"""<svg class="dia" viewBox="0 0 1060 {y + 10}" role="img" aria-label="Arquitetura do ENS_stack_sem: nove modelos combinados por pesos que dependem do horizonte e da fase da epidemia">
<defs><marker id="seta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>
<rect x="10" y="150" width="200" height="160" rx="6" fill="none" stroke="currentColor" stroke-opacity=".45"/>
<text x="22" y="174" font-size="12.5" font-weight="700" fill="currentColor">Dados até a semana t</text>
<text x="22" y="196" font-size="12" fill="currentColor">incidência por UF (log)</text>
<text x="22" y="215" font-size="12" fill="currentColor">casos dos vizinhos</text>
<text x="22" y="234" font-size="12" fill="currentColor">clima, ENSO, SEAS5</text>
<text x="22" y="253" font-size="12" fill="currentColor">fase e Rt da epidemia</text>
<text x="22" y="272" font-size="12" fill="currentColor">semana do ano</text>
<text x="22" y="296" font-size="11" fill="currentColor" fill-opacity=".7">cada membro retreinado</text>
<line x1="210" y1="230" x2="246" y2="230" stroke="currentColor" stroke-opacity=".5" marker-end="url(#seta)"/>
<text x="216" y="222" font-size="11" fill="currentColor" fill-opacity=".7">treina</text>
<rect x="610" y="150" width="240" height="160" rx="6" fill="var(--blue-soft)" stroke="var(--blue)"/>
<text x="622" y="174" font-size="12.5" font-weight="700" fill="currentColor">Pesos (stacking)</text>
<text x="622" y="196" font-size="12" fill="currentColor">um conjunto de 9 pesos para</text>
<text x="622" y="215" font-size="12" fill="currentColor">cada horizonte (1–4 semanas)</text>
<text x="622" y="234" font-size="12" fill="currentColor">× fase da UF na semana t:</text>
<text x="622" y="253" font-size="12" fill="currentColor">entressafra · intermediária · pico</text>
<text x="622" y="276" font-size="11" fill="currentColor" fill-opacity=".7">12 combinações, escolhidas na</text>
<text x="622" y="292" font-size="11" fill="currentColor" fill-opacity=".7">afinação 2018–2021 (menor WIS)</text>
<line x1="850" y1="230" x2="876" y2="230" stroke="currentColor" stroke-opacity=".5" marker-end="url(#seta)"/>

<rect x="880" y="170" width="170" height="120" rx="6" fill="none" stroke="var(--orange)" stroke-width="1.5"/>
<text x="892" y="194" font-size="12.5" font-weight="700" fill="currentColor">Previsão</text>
<text x="892" y="216" font-size="12" fill="currentColor">semanas t+1 … t+4</text>
<text x="892" y="235" font-size="12" fill="currentColor">9 quantis por UF</text>
<text x="892" y="254" font-size="12" fill="currentColor">faixas 50·80·90·95%</text>
{''.join(partes)}
</svg>"""


def diagrama_temporada():
    return """<svg class="dia" viewBox="0 0 1060 400" role="img" aria-label="Arquitetura do ENS_pico7_simples: normal das redes de grafo, detector de anos grandes, duas misturas e uma rampa pelo nível previsto">
<defs><marker id="seta2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>
<rect x="10" y="150" width="150" height="100" rx="6" fill="none" stroke="currentColor" stroke-opacity=".45"/>
<text x="22" y="174" font-size="12.5" font-weight="700" fill="currentColor">Dados até a</text>
<text x="22" y="192" font-size="12.5" font-weight="700" fill="currentColor">SE25 (junho)</text>
<text x="22" y="214" font-size="12" fill="currentColor">casos, vizinhos,</text>
<text x="22" y="232" font-size="12" fill="currentColor">clima, imunidade</text>
<line x1="160" y1="185" x2="196" y2="95" stroke="currentColor" stroke-opacity=".5" marker-end="url(#seta2)"/>
<line x1="160" y1="215" x2="196" y2="300" stroke="currentColor" stroke-opacity=".5" marker-end="url(#seta2)"/>
<rect x="200" y="30" width="260" height="130" rx="6" fill="none" stroke="currentColor" stroke-opacity=".45"/>
<text x="212" y="52" font-size="12.5" font-weight="700" fill="currentColor">1 · Normal (ENS_grafos3_λ)</text>
<text x="212" y="73" font-size="12" fill="currentColor">mediana de 4 redes de grafo:</text>
<text x="212" y="92" font-size="12" fill="currentColor">MTGNN, A3T-GCN, GCNGRU (λ = 0,8)</text>
<text x="212" y="111" font-size="12" fill="currentColor">e MTGNN do Optuna (10 sementes)</text>
<text x="212" y="134" font-size="11" fill="currentColor" fill-opacity=".7">a temporada "típica": SE41 → SE40,</text>
<text x="212" y="150" font-size="11" fill="currentColor" fill-opacity=".7">acerta a época, subestima anos grandes</text>
<rect x="200" y="240" width="260" height="120" rx="6" fill="none" stroke="currentColor" stroke-opacity=".45"/>
<text x="212" y="262" font-size="12.5" font-weight="700" fill="currentColor">2 · Detector de anos grandes</text>
<text x="212" y="283" font-size="12" fill="currentColor">ridge: tamanho da temporada ÷</text>
<text x="212" y="302" font-size="12" fill="currentColor">normal da UF (em log), 9 quantis</text>
<text x="212" y="325" font-size="11" fill="currentColor" fill-opacity=".7">10 sinais escolhidos só até 2021</text>
<text x="212" y="341" font-size="11" fill="currentColor" fill-opacity=".7">correlação 0,82 no teste</text>
<line x1="460" y1="95" x2="506" y2="120" stroke="currentColor" stroke-opacity=".5" marker-end="url(#seta2)"/>
<line x1="460" y1="300" x2="506" y2="270" stroke="currentColor" stroke-opacity=".5" marker-end="url(#seta2)"/>
<line x1="460" y1="95" x2="506" y2="250" stroke="currentColor" stroke-opacity=".3" marker-end="url(#seta2)"/>
<line x1="460" y1="300" x2="506" y2="140" stroke="currentColor" stroke-opacity=".3" marker-end="url(#seta2)"/>
<rect x="510" y="70" width="250" height="110" rx="6" fill="var(--teal-soft)" stroke="var(--teal)"/>
<text x="522" y="92" font-size="12.5" font-weight="700" fill="currentColor">3a · MIX_elastico5</text>
<text x="522" y="113" font-size="12" fill="currentColor">o normal cresce até o tamanho</text>
<text x="522" y="132" font-size="12" fill="currentColor">previsto, mas cada semana cresce</text>
<text x="522" y="151" font-size="12" fill="currentColor">pela sua elasticidade: 0,1 sem</text>
<text x="522" y="170" font-size="12" fill="currentColor">casos → 1,0 no pico (w = 0,5)</text>
<rect x="510" y="215" width="250" height="110" rx="6" fill="var(--orange-soft)" stroke="var(--orange)"/>
<text x="522" y="237" font-size="12.5" font-weight="700" fill="currentColor">3b · MIX_escala5</text>
<text x="522" y="258" font-size="12" fill="currentColor">se o tamanho previsto passa de</text>
<text x="522" y="277" font-size="12" fill="currentColor">3× o normal, reescala a curva</text>
<text x="522" y="296" font-size="12" fill="currentColor">inteira (acerta o pico, infla</text>
<text x="522" y="315" font-size="12" fill="currentColor">a entressafra)</text>
<line x1="760" y1="125" x2="806" y2="185" stroke="currentColor" stroke-opacity=".5" marker-end="url(#seta2)"/>
<line x1="760" y1="270" x2="806" y2="215" stroke="currentColor" stroke-opacity=".5" marker-end="url(#seta2)"/>
<rect x="810" y="110" width="240" height="180" rx="6" fill="none" stroke="currentColor" stroke-width="1.5"/>
<text x="822" y="132" font-size="12.5" font-weight="700" fill="currentColor">4 · Rampa pelo nível previsto</text>
<text x="822" y="153" font-size="12" fill="currentColor">z = nível da semana na escala</text>
<text x="822" y="171" font-size="12" fill="currentColor">da UF (p25 = 0, p75 = 1)</text>
<text x="822" y="192" font-size="12" fill="currentColor">s = 1 / (1 + e^−(z−1)/0,3)</text>
<text x="822" y="216" font-size="12" font-weight="700" fill="currentColor">ENS = elástico +</text>
<text x="822" y="235" font-size="12" font-weight="700" fill="currentColor">  ½·s·(escala − elástico)</text>
<text x="822" y="259" font-size="11" fill="currentColor" fill-opacity=".7">entressafra: só o elástico</text>
<text x="822" y="275" font-size="11" fill="currentColor" fill-opacity=".7">pico: média dos dois (50/50)</text>
</svg>"""


def _tab_semanal():
    lin = ''.join(f"<tr><td class='n'>{d['h']}</td><td class='n'>{_fmt(d['MAE'])}</td><td class='n'>{_fmt(d['MAE_pers'])}</td>"
                  f"<td class='n'>{_fmt(d['WAPE'], 1, True)}</td><td class='n'>{_fmt(d['sMAPE'], 1)}</td>"
                  f"<td class='n'>{_fmt(d['R_clim'], 3)}</td><td class='n'>{_fmt(d['R_pers'], 3)}</td>"
                  f"<td class='n'>{_fmt(d['IC90'], 1, True)}</td><td class='n'>{_fmt(d['vies'], 1, True, True)}</td><td class='n'>{_fmt(d['r'], 3)}</td></tr>"
                  for d in SEMANAL_H)
    top = ''.join(f"<tr class='{'best' if b else ('ref' if b is None else '')}'><td>{html.escape(m)}</td>"
                  + ''.join(f"<td class='n'>{_fmt(v)}</td>" for v in vals) + '</tr>'
                  for m, *vals, b in SEMANAL_TOP5)
    return f"""<div class="tw"><table><tr><th>Semanas à frente</th><th>MAE</th><th>MAE da persistência</th><th>WAPE</th><th>sMAPE</th>
<th>WIS ÷ climatologia</th><th>WIS ÷ persistência</th><th>Faixa de 90% acerta</th><th>Viés</th><th>Correlação</th></tr>{lin}</table></div>
<p class="note">Teste 2022–2025, 26 UFs, todas as semanas. MAE em casos por semana; WIS ÷ referência abaixo de 1 = melhor que ela.</p>
<h3>Os 5 melhores semanais (MAE em casos por semana)</h3>
<div class="tw"><table><tr><th>Modelo</th><th>h = 1</th><th>h = 2</th><th>h = 3</th><th>h = 4</th><th>Média</th></tr>{top}</table></div>"""


def _tab_temporada():
    lin = ''
    for m, r, ufs, wis, wp, we, mae, wape, smape, alt, tot, ic, b in TEMPORADA:
        cl = 'best' if b else ('ref' if b is None else '')
        lin += (f"<tr class='{cl}'><td>{html.escape(m)}</td><td class='n'>{_fmt(r, 3)}</td><td class='n'>{_fmt(ufs)}</td>"
                f"<td class='n'>{_fmt(wis)}</td><td class='n'>{_fmt(wp)}</td><td class='n'>{_fmt(we, 1)}</td><td class='n'>{_fmt(mae)}</td>"
                f"<td class='n'>{_fmt(wape, 1, True)}</td><td class='n'>{_fmt(smape, 1)}</td><td class='n'>{_fmt(alt, 1, True, True)}</td>"
                f"<td class='n'>{_fmt(tot, 1, True, True)}</td><td class='n'>{_fmt(ic, 0, True)}</td></tr>")
    return f"""<div class="tw"><table><tr><th>Modelo</th><th>WIS ÷ BB</th><th>UFs melhores que o BB</th><th>WIS</th><th>WIS no pico</th>
<th>WIS na entressafra</th><th>MAE</th><th>WAPE</th><th>sMAPE</th><th>Altura do pico</th><th>Total da temporada</th><th>Pico dentro da faixa de 90%</th></tr>{lin}</table></div>
<p class="note">Teste 2022–2025: cada temporada (SE41 → SE40) prevista só com dados até a SE25 de junho. Altura e total: erro mediano entre UFs e temporadas
(0 é o ideal). O ENS_pico7_UF é o melhor na régua relativa oficial (0,808 × BB); o campeão de produção foi escolhido pelas métricas absolutas.</p>"""


def _tabela_df(df, colunas=None, max_linhas=40):
    if df is None or len(df) == 0:
        return ''
    d = df[colunas] if colunas else df
    cab = ''.join(f'<th>{html.escape(str(c))}</th>' for c in d.columns)
    corpo = ''
    for _, r in d.head(max_linhas).iterrows():
        cel = ''
        for v in r.values:
            if isinstance(v, (int, float)) and not isinstance(v, bool) and pd.notna(v):
                cel += f"<td class='n'>{_fmt(float(v))}</td>"
            else:
                cel += f'<td>{html.escape(str(v))}</td>'
        corpo += f'<tr>{cel}</tr>'
    return f'<div class="tw"><table><tr>{cab}</tr>{corpo}</table></div>'


def gerar(arq, figuras=None, tab_semanal=None, tab_temporada=None, info=None, fragmento=False):
    """fragmento=True: sem <!doctype>/<html>/<head>/<body> (para publicar como página que já tem o esqueleto)."""
    info = info or {}
    if figuras:
        nav = ''.join(f'<a href="#u-{html.escape(u)}">{html.escape(u)}</a>' for u in figuras)
        blocos = ''
        for u, f in figuras.items():
            blocos += f'<div class="uni" id="u-{html.escape(u)}"><h3>{html.escape(u)}</h3>'
            if f.get('semanal') and os.path.exists(f['semanal']):
                blocos += f'<img src="{_img(f["semanal"])}" alt="Previsão semanal de {html.escape(u)}">'
            if f.get('temporada') and os.path.exists(f['temporada']):
                blocos += f'<img src="{_img(f["temporada"])}" alt="Previsão de temporada de {html.escape(u)}">'
            blocos += '</div>'
        graficos = f'<div class="picker" aria-label="Ir para">{nav}</div>{blocos}'
    else:
        graficos = ('<p class="aviso">Os gráficos por estado são gerados pelo notebook <code>producao/Relatorio_Campeoes.ipynb</code>, '
                    'a partir das previsões guardadas no Drive (histórico do teste) e da previsão de produção. Rode-o para obter '
                    'esta seção preenchida, com Brasil, as 5 macrorregiões e as 26 UFs.</p>')
    ult = info.get('ultima_semana', '–')
    s = info.get('temporada', '–')
    doc = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Modelos Campeões Dengue</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,600&family=Public+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style></head><body><div class="wrap">
<div class="eyebrow">Previsão de dengue por UF · 3º IMDC / Mosqlimate</div>
<h1>Os modelos campeões: previsão semanal e da temporada</h1>
<p>Dois modelos foram escolhidos para uso operacional, pelas métricas absolutas (erro em casos) no teste de 2022 a 2025, quatro temporadas
que nenhum modelo viu durante as escolhas. Dados observados até <b>{html.escape(str(ult))}</b>; temporada mais recente prevista: <b>{html.escape(str(s))}</b>.</p>
<div class="champs">
 <div class="champ"><div class="k">Campeão semanal · 1 a 4 semanas à frente</div><div class="n">ENS_stack_sem</div>
  <ul><li>MAE de 270 casos/semana em 1 semana (persistência: 354) e 583 em 4 semanas (persistência: 1.186)</li>
  <li>erra 12% do volume em 1 semana e 27% em 4 semanas (WAPE)</li><li>a faixa de 90% contém o valor real em 88–90% das semanas</li></ul></div>
 <div class="champ"><div class="k">Campeão de temporada · SE41 → SE40, previsto em junho</div><div class="n">ENS_pico7_simples_UF</div>
  <ul><li>erro mediano da altura do pico: +2% (o modelo-base oficial erra −36%)</li><li>erro mediano do total da temporada: −0,6%</li>
  <li>menor WIS e menor MAE entre todos os modelos testados (721 e 1.277)</li></ul></div>
</div>
<section><h2>Campeão semanal: ENS_stack_sem</h2>
<p>Uma média ponderada de 9 modelos diferentes. O peso de cada um muda com o horizonte (1 a 4 semanas) e com a fase em que a UF está na
semana da previsão: na entressafra pesa mais um XGBoost; no pico, ARIMA, XGBoost e o Chronos-2. Os pesos foram escolhidos em 2018–2021.</p>
<figure>{diagrama_semanal()}<figcaption>Cada membro é retreinado com os dados até a semana da previsão; o resultado é a média dos 9 quantis
de cada membro, com os pesos da combinação (horizonte × fase).</figcaption></figure>
{_tab_semanal()}</section>
<section><h2>Campeão de temporada: ENS_pico7_simples_UF</h2>
<p>Parte de uma temporada "normal" prevista por redes de grafo e a corrige pelo tamanho que um detector de anos grandes prevê em junho.
Duas correções são misturadas conforme o nível previsto de cada semana: na entressafra vale a correção elástica, que mexe pouco nas
semanas com poucos casos; no pico, a média entre a elástica (que fica baixa) e a de escala (que passa do real).</p>
<figure>{diagrama_temporada()}<figcaption>Da esquerda para a direita: normal e detector, as duas misturas e a rampa que decide, semana a semana,
quanto peso dar ao componente de pico.</figcaption></figure>
{_tab_temporada()}</section>
<section><h2>Previsão mais recente</h2>
<h3>Próximas 4 semanas (ENS_stack_sem)</h3>{_tabela_df(tab_semanal) or '<p class="note">Gerada pelo notebook de produção semanal.</p>'}
<h3>Temporada {html.escape(str(s))} (ENS_pico7_simples_UF)</h3>{_tabela_df(tab_temporada) or '<p class="note">Gerada pelo notebook de produção de temporada.</p>'}
</section>
<section><h2>Gráficos por estado, região e Brasil</h2>
<p>Em cada unidade: <b>em cima</b>, a previsão semanal (azul: o que o modelo previu 1 semana antes, no histórico; laranja: as próximas
4 semanas); <b>embaixo</b>, a previsão de cada temporada feita em junho (verde, com faixas de 50% e 90%). Preto: casos observados.
Regiões e Brasil somam as UFs; as faixas somadas são conservadoras.</p>
{graficos}</section>
<p class="note" style="margin-top:40px">Métricas: teste 2022–2025 do projeto (docs/05_modelos_campeoes.md). Gerado por producao/relatorio_campeoes.py.</p>
</div></body></html>"""
    if fragmento:
        doc = doc[doc.index('<title>'):].replace('</style></head><body>', '</style>').replace('</div></body></html>', '</div>')
    with open(arq, 'w', encoding='utf-8') as f:
        f.write(doc)
    return arq
