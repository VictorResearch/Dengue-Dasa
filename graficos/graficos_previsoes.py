"""
Gráficos de previsão por UF, por macrorregião (Norte, Nordeste, Sudeste, Sul, Centro-Oeste) e Brasil.

  semanal:   série observada + previsões de h semanas à frente feitas no passado (teste) + leque das próximas semanas
  temporada: série observada + a previsão de cada temporada feita em junho (SE25), com leque de 50% e 90%

Regiões e Brasil: os quantis das UFs são SOMADOS. A mediana somada é uma boa aproximação; as faixas somadas são
CONSERVADORAS (mais largas que o verdadeiro intervalo da soma, porque supõem que todas as UFs erram juntas).

Uso típico (dentro da área de trabalho):
    import graficos_previsoes as G
    OBS = G.observado(D)                                    # casos por UF + regiões + Brasil
    G.figura_semanal(OBS, hist_h1, futuro, 'PR', 'pr.png')
    G.figuras_todas(OBS, ..., pasta)                        # 26 UFs + 5 regiões + Brasil
"""
import os

import numpy as np
import pandas as pd

QCOLS = ['lower_95', 'lower_90', 'lower_80', 'lower_50', 'pred', 'upper_50', 'upper_80', 'upper_90', 'upper_95']
SIGLAS = {11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO', 21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN',
          25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC',
          43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'}
REGIOES = {1: 'Norte', 2: 'Nordeste', 3: 'Sudeste', 4: 'Sul', 5: 'Centro-Oeste'}
COR = dict(obs='#1a1a1a', sem='#1f6fb2', fut='#c2410c', temp='#0e7468', faixa='#0e7468')


def unidade_de(uf_code):
    return SIGLAS.get(int(uf_code), str(uf_code))


def _com_agregados(df, valor_cols):
    """acrescenta linhas das 5 macrorregiões e do Brasil (soma das UFs) a um DataFrame com coluna uf_code."""
    d = df.copy()
    d['unidade'] = d.uf_code.map(unidade_de)
    chaves = [c for c in d.columns if c not in valor_cols + ['uf_code', 'unidade']]
    reg = d.assign(unidade=d.uf_code.astype(int) // 10).groupby(chaves + ['unidade'], as_index=False)[valor_cols].sum()
    reg['unidade'] = reg.unidade.map(REGIOES)
    br = d.groupby(chaves, as_index=False)[valor_cols].sum().assign(unidade='Brasil')
    return pd.concat([d.drop(columns='uf_code'), reg, br], ignore_index=True)


def observado(D, incompletas=1):
    """casos observados por semana: UFs, macrorregiões e Brasil (formato longo: date, unidade, y, incompleta).
    As `incompletas` últimas semanas ficam marcadas: chegam com casos faltando (atraso de notificação)."""
    n = D.ultimo_obs + 1
    y = pd.DataFrame(D.Y[:n], index=pd.to_datetime(D.datas[:n]), columns=D.unidades)
    y = y.stack().rename('y').reset_index().rename(columns={'level_0': 'date', 'level_1': 'uf_code'})
    out = _com_agregados(y, ['y'])
    limite = pd.Timestamp(D.datas[D.ultimo_obs - incompletas]) if incompletas else out.date.max()
    out['incompleta'] = out.date > limite
    return out


def _linha_obs(ax, o, rotulo=True):
    c, inc = o[~o.incompleta], o[o.incompleta]
    ax.plot(c.date, c.y, color=COR['obs'], lw=1.1, label='observado' if rotulo else None)
    if len(inc):
        ax.plot(pd.concat([c.tail(1), inc]).date, pd.concat([c.tail(1), inc]).y, color=COR['obs'], lw=1, ls=':', alpha=.6)
        ax.scatter(inc.date, inc.y, s=14, facecolor='none', edgecolor=COR['obs'], zorder=3,
                   label='semana incompleta (atraso de notificação)' if rotulo else None)


def agregar_previsoes(P):
    """previsões por UF (com uf_code e os 9 quantis) → + regiões e Brasil (soma dos quantis)."""
    P = P.copy()
    P['date'] = pd.to_datetime(P['date'])
    return _com_agregados(P, QCOLS)


def _marca_teste(ax, teste):
    if teste:
        x = pd.Timestamp(teste)
        ax.axvline(x, color='#8a8a8a', lw=.8, ls='--')
        ax.text(x, 1.0, ' início do teste (2022)', transform=ax.get_xaxis_transform(), fontsize=8, color='#6a6a6a', va='top')


def _estilo(ax, titulo):
    import matplotlib
    import matplotlib.dates as mdates
    ax.set_title(titulo, loc='left', fontsize=11)
    ax.set_ylabel('casos por semana')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#ececec')
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'.replace(',', '.')))
    ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=2)


def figura_semanal(OBS, hist, futuro, unidade, arq=None, desde='2018-10-01', h_hist=1, titulo_modelo='ENS_stack_sem', teste='2022-10-02'):
    """OBS: observado(D) · hist: previsões passadas (com h e unidade) · futuro: previsão a partir da última origem."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[3, 1]))
    o = OBS[(OBS.unidade == unidade) & (OBS.date >= pd.Timestamp(desde))].sort_values('date')
    for i, ax in enumerate((a1, a2)):
        _linha_obs(ax, o, rotulo=i == 0)
    if hist is not None and len(hist):
        h = hist[(hist.unidade == unidade) & (hist.h == h_hist)].sort_values('date')
        a1.fill_between(h.date, h.lower_90, h.upper_90, color=COR['sem'], alpha=.15, lw=0, label='faixa de 90%')
        a1.plot(h.date, h.pred, color=COR['sem'], lw=1.2, label=f'previsto {h_hist} semana antes')
        a2.fill_between(h.date, h.lower_90, h.upper_90, color=COR['sem'], alpha=.15, lw=0)
        a2.plot(h.date, h.pred, color=COR['sem'], lw=1.2)
    if futuro is not None and len(futuro):
        f = futuro[futuro.unidade == unidade].sort_values('date')
        for ax in (a1, a2):
            ax.fill_between(f.date, f.lower_90, f.upper_90, color=COR['fut'], alpha=.18, lw=0, label='próximas semanas · 90%')
            ax.fill_between(f.date, f.lower_50, f.upper_50, color=COR['fut'], alpha=.30, lw=0, label='50%')
            ax.plot(f.date, f.pred, color=COR['fut'], lw=1.8, marker='o', ms=3, label='próximas semanas · mediana')
        ini = f.date.min() - pd.Timedelta(weeks=26)
        oo = o[o.date >= ini]
        a2.set_xlim(ini, f.date.max() + pd.Timedelta(weeks=1))
        a2.set_ylim(0, 1.15 * max(float(oo.y.max() if len(oo) else 0), float(f.upper_90.max())))
    _marca_teste(a1, teste)
    _estilo(a1, f'{unidade} · {titulo_modelo} · observado e previsto (1 semana à frente) no histórico')
    import matplotlib.dates as mdates
    a2.set_title('zoom: últimos 6 meses + próximas 4 semanas', loc='left', fontsize=10)
    a2.xaxis.set_major_locator(mdates.MonthLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    a2.spines[['top', 'right']].set_visible(False)
    a2.grid(axis='y', color='#ececec')
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'.replace(',', '.')))
    fig.tight_layout()
    if arq:
        fig.savefig(arq, dpi=100)
        plt.close(fig)
    return fig


def figura_temporada(OBS, prev, unidade, arq=None, desde='2018-10-01', titulo_modelo='ENS_pico7_simples_UF', teste='2022-10-02'):
    """prev: previsões de temporada (várias temporadas; cada uma feita na SE25) com coluna unidade."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4.2))
    o = OBS[(OBS.unidade == unidade) & (OBS.date >= pd.Timestamp(desde))].sort_values('date')
    p = prev[prev.unidade == unidade]
    for i, (s, g) in enumerate(p.groupby('season')):
        g = g.sort_values('date')
        ax.fill_between(g.date, g.lower_90, g.upper_90, color=COR['faixa'], alpha=.13, lw=0, label='faixa de 90%' if i == 0 else None)
        ax.fill_between(g.date, g.lower_50, g.upper_50, color=COR['faixa'], alpha=.25, lw=0, label='faixa de 50%' if i == 0 else None)
        ax.plot(g.date, g.pred, color=COR['temp'], lw=1.6, label='previsto em junho (mediana)' if i == 0 else None)
        ax.axvline(pd.Timestamp(f'{s}-10-08'), color='#c9c9c9', lw=.7, ls=':')
    _linha_obs(ax, o)
    _marca_teste(ax, teste)
    _estilo(ax, f'{unidade} · {titulo_modelo} · cada temporada (SE41 → SE40) prevista com dados até a SE25')
    fig.tight_layout()
    if arq:
        fig.savefig(arq, dpi=100)
        plt.close(fig)
    return fig


def ordem_unidades():
    """Brasil, as 5 regiões e as UFs agrupadas por região."""
    ufs = sorted([u for u in SIGLAS if u != 32], key=lambda u: (u // 10, SIGLAS[u]))
    return ['Brasil'] + list(REGIOES.values()) + [SIGLAS[u] for u in ufs]


def figuras_todas(OBS, pasta, semanal=None, temporada=None, desde='2018-10-01'):
    """grava <pasta>/semanal/<unidade>.png e <pasta>/temporada/<unidade>.png para Brasil, regiões e UFs.
    semanal = (hist, futuro) já agregados; temporada = previsões de temporada já agregadas."""
    feitos = {}
    for u in ordem_unidades():
        feitos[u] = {}
        if semanal is not None:
            os.makedirs(os.path.join(pasta, 'semanal'), exist_ok=True)
            arq = os.path.join(pasta, 'semanal', f'{u}.png')
            figura_semanal(OBS, semanal[0], semanal[1], u, arq, desde)
            feitos[u]['semanal'] = arq
        if temporada is not None:
            os.makedirs(os.path.join(pasta, 'temporada'), exist_ok=True)
            arq = os.path.join(pasta, 'temporada', f'{u}.png')
            figura_temporada(OBS, temporada, u, arq, desde)
            feitos[u]['temporada'] = arq
    return feitos
