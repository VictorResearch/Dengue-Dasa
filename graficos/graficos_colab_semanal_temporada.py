# @title Gráficos: previsões semanais e de temporada (cole no Semanal3_Avaliacao depois das células A2 e B2)
# Usa o que já está na memória da sessão: P2 (previsões semanais), TODOS (previsões de temporada), D, TESTE, PASTA_AVAL, S, V.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

SEMANAL = {'melhor (WIS)': 'ENS_GBM_ARIMA_sem', 'mais preciso (MAE)': 'ARIMA_Fourier_sem'}
TEMPORADA = {'melhor (WIS × BB)': 'ENS_grafos2_λ', 'melhor modelo isolado': 'MTGNN_pesopop_UF+λ'}
REF_TEMP = 'BB'                                       # linha tracejada nos gráficos de temporada
UFS_PLOT = ['SP', 'MG', 'PR', 'GO', 'BA', 'RJ', 'SC', 'PE']   # 8 painéis + o Brasil
COMPARA = ['SP', 'MG', 'PR', 'GO']                     # figura semanal × temporada no mesmo painel
COR_SEM, COR_TEMP, COR_H4 = '#0d6457', '#b4680b', '#5b6fb5'

SIGLA = dict(zip([11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53],
                 'RO AC AM RR PA AP TO MA PI CE RN PB PE AL SE BA MG ES RJ SP PR SC RS MS MT GO DF'.split()))
COD = {v: k for k, v in SIGLA.items()}
PASTA_G = os.path.join(PASTA_AVAL, 'graficos')
os.makedirs(PASTA_G, exist_ok=True)
OBS = pd.DataFrame(D.Y[:D.ultimo_obs + 1], index=pd.to_datetime(D.datas[:D.ultimo_obs + 1]), columns=D.unidades)
INI = pd.Timestamp(V.datas_temporada(TESTE[0])[0]) - pd.Timedelta(weeks=2)
FIM = OBS.index.max()
INICIOS = [pd.Timestamp(V.datas_temporada(s)[0]) for s in TESTE]
milhar = FuncFormatter(lambda v, _: f'{v / 1000:.0f} mil' if abs(v) >= 1000 else f'{v:.0f}')


def checa(nome, df):
    if not (df.modelo == nome).any():
        print(f'⚠️ {nome} não encontrado. Parecidos:', sorted(m for m in df.modelo.unique() if nome.split('_')[0] in m)[:10])
        return False
    return True


def semanal(nome, h=1):
    x = P2[(P2.modelo == nome) & (P2.h == h) & P2.rodada.isin([f's{s}' for s in TESTE])].copy()
    x['date'] = pd.to_datetime(x['date'])
    return x


def temporada(nome):
    x = TODOS[(TODOS.modelo == nome) & TODOS.season.isin(TESTE)].copy()
    x['date'] = pd.to_datetime(x['date'])
    return x


def resumo(x):
    m = S.metricas(S.com_verdade(x, D))
    return f"MAE {m['MAE']:.0f} · WAPE {m['WAPE']:.1f}% · sMAPE {m['sMAPE']:.1f}% · IC90 {m['IC90']:.0f}% · viés {m['vies_pct']:+.0f}%"


def leque(ax, g, cor, rot):
    g = g.sort_values('date')
    ax.fill_between(g.date, g.lower_90, g.upper_90, color=cor, alpha=.16, lw=0, label=f'{rot}: faixa 90%')
    ax.fill_between(g.date, g.lower_50, g.upper_50, color=cor, alpha=.32, lw=0, label=f'{rot}: faixa 50%')
    ax.plot(g.date, g.pred, color=cor, lw=1.6, label=f'{rot}: mediana')


def moldura(ax, titulo):
    o = OBS.loc[INI:FIM]
    ax.set_title(titulo, fontsize=11, loc='left')
    for d in INICIOS:
        ax.axvline(d, color='0.6', lw=.8, ls=':')
    ax.yaxis.set_major_formatter(milhar)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.set_xlim(INI, FIM)
    ax.grid(alpha=.25)


def figura(x, nome, rotulo, cor, titulo_nivel, ref=None):
    fig, axs = plt.subplots(3, 3, figsize=(18, 11), sharex=True)
    for ax, sg in zip(axs.flat, UFS_PLOT):
        u = COD[sg]
        leque(ax, x[x.uf_code == u], cor, 'previsão')
        if ref is not None:
            r = ref[ref.uf_code == u].sort_values('date')
            ax.plot(r.date, r.pred, color='0.35', lw=1.1, ls='--', label=f'{REF_TEMP}: mediana')
        o = OBS.loc[INI:FIM, u]
        ax.plot(o.index, o.values, color='k', lw=1.1, label='observado')
        moldura(ax, sg)
    ax = axs.flat[-1]
    b = x.groupby('date').pred.sum()
    ob = OBS.loc[INI:FIM].sum(1)
    ax.plot(b.index, b.values, color=cor, lw=1.6)
    ax.plot(ob.index, ob.values, color='k', lw=1.1)
    moldura(ax, 'Brasil (soma das 26 UFs; só a mediana)')
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), frameon=False)
    fig.suptitle(f'{titulo_nivel} · {rotulo}: {nome}\n{resumo(x)}  (teste {TESTE[0]}–{TESTE[-1]}; linhas pontilhadas = início de cada temporada)',
                 fontsize=13, x=.01, ha='left')
    fig.tight_layout(rect=(0, .04, 1, .95))
    arq = os.path.join(PASTA_G, f"{titulo_nivel.split()[0].lower()}_{nome.replace('+', 'mais').replace('λ', 'lambda')}.png")
    fig.savefig(arq, dpi=140)
    plt.show()
    print('salvo:', arq)


# 1 · Semanal (próxima semana, h = 1): melhor e mais preciso
for rot, nome in SEMANAL.items():
    if checa(nome, P2):
        figura(semanal(nome, 1), nome, rot, COR_SEM, 'Semanal (h = 1 semana)')

# 2 · Temporada (previsão feita na SE25 para SE41→SE40): melhor e melhor isolado, com o BB tracejado
ref = temporada(REF_TEMP) if (TODOS.modelo == REF_TEMP).any() else None
for rot, nome in TEMPORADA.items():
    if checa(nome, TODOS):
        figura(temporada(nome), nome, rot, COR_TEMP, 'Temporada (16–68 semanas)', ref)

# 3 · Mesmo painel: observado × semanal h=1 × semanal h=4 × temporada
nm_s, nm_t = list(SEMANAL.values())[0], list(TEMPORADA.values())[0]
if checa(nm_s, P2) and checa(nm_t, TODOS):
    s1, s4, tp = semanal(nm_s, 1), semanal(nm_s, 4), temporada(nm_t)
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for ax, sg in zip(axs.flat, COMPARA):
        u = COD[sg]
        leque(ax, tp[tp.uf_code == u], COR_TEMP, f'temporada ({nm_t})')
        g1 = s1[s1.uf_code == u].sort_values('date')
        ax.fill_between(g1.date, g1.lower_90, g1.upper_90, color=COR_SEM, alpha=.16, lw=0, label='semanal h=1: faixa 90%')
        ax.plot(g1.date, g1.pred, color=COR_SEM, lw=1.5, label=f'semanal h=1 ({nm_s})')
        g4 = s4[s4.uf_code == u].sort_values('date')
        ax.plot(g4.date, g4.pred, color=COR_H4, lw=1.2, ls='--', label='semanal h=4: mediana')
        o = OBS.loc[INI:FIM, u]
        ax.plot(o.index, o.values, color='k', lw=1.1, label='observado')
        moldura(ax, sg)
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Mesmas semanas, três antecedências: 1 semana, 4 semanas e a temporada inteira (feita em junho)', fontsize=13, x=.01, ha='left')
    fig.tight_layout(rect=(0, .07, 1, .96))
    arq = os.path.join(PASTA_G, 'comparacao_semanal_x_temporada.png')
    fig.savefig(arq, dpi=140)
    plt.show()
    print('salvo:', arq)
