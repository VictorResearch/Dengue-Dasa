# @title REFAZER SÓ A PREVISÃO (sem treinar): recalibração por regime + soma estadual correta
# Pré-requisito: rodar as células do notebook até "GERAÇÃO DA SUBMISSÃO" e a "Avaliar Desempenho",
# PULANDO a "RODAR O MODELO 2". Usa os pesos já treinados (modelo_agcrn_{SECA,PICO}_s{N}.pt).
#
# O que muda em relação à execução normal:
#  1. RECALIBRAÇÃO: cada especialista tem um viés sistemático (PICO superestima, SECA subestima).
#     O fator de cada um = soma dos casos reais / soma das previsões na TEMPORADA DE VALIDAÇÃO
#     (últimas 52 semanas do treino). O teste não é usado para nada.
#  2. SOMA ESTADUAL: o estado recebe a distribuição da SOMA dos municípios (média = soma das médias,
#     variância = soma das variâncias, aproximada por uma Binomial Negativa). Antes, o estado era a soma das
#     medianas municipais e os intervalos eram a soma dos quantis.
#  3. SEM MinT: o MinT usava os erros do PRÓPRIO TESTE (casos_reais) para calcular os pesos, ou seja, olhava a
#     resposta. Aqui o estado é a soma dos municípios (bottom-up), coerente por construção.
import pyarrow.parquet as pq
from scipy.stats import nbinom

AQUEC = 15                                   # semanas de aquecimento do estado recorrente
torch.set_grad_enabled(False)

# ---------------------------------------------------------------- 1. dados (ordem date, node_id)
arq_tr = os.path.join(PASTA_DADOS, f"train{SEASON_ALVO}_FINAL_NN_v2.parquet")
eps = np.sort(pq.read_table(arq_tr, columns=['epiweek'])['epiweek'].to_pandas().unique())
df_val = pd.read_parquet(arq_tr, filters=[('epiweek', '>=', int(eps[-(SEMANAS_VALIDACAO + AQUEC)]))])
df_ctx = pd.read_parquet(os.path.join(PASTA_DADOS, f"ctx{SEASON_ALVO}_FINAL_NN_v2.parquet"))
df_te = pd.read_parquet(os.path.join(PASTA_DADOS, f"test{SEASON_ALVO}_FINAL_NN_v2.parquet"))
df_val, df_ctx, df_te = [d.sort_values(['date', 'node_id']).reset_index(drop=True) for d in (df_val, df_ctx, df_te)]
N = df_te['node_id'].nunique()


def _tensores(df, cols):
    n = df['date'].nunique()
    X = torch.tensor(df[cols].to_numpy(np.float32).reshape(n, N, len(cols)), device=device)
    lp = torch.tensor(df['log_pop'].to_numpy(np.float32).reshape(n, N), device=device)
    return X, lp


def _carregar(nome, cols):
    m = PreditorNBAGCRN(N, len(cols), 64, cols.index('velocidade'), cols.index('aceleracao')).to(device)
    arq = f"modelo_agcrn_{nome}_s{SEASON_ALVO}.pt"
    caminho = arq if os.path.exists(arq) else os.path.join(PASTA_SAIDA, arq)
    sd = torch.load(caminho, map_location=device, weights_only=False)
    m.load_state_dict({k.replace('_orig_mod.', ''): v.float() for k, v in sd.items()})
    print(f"📦 Pesos carregados: {caminho}")
    return m.eval()


modelo_seca, modelo_pico = _carregar('SECA', COLS_SECA), _carregar('PICO', COLS_PICO)


def _prever(df_aq, df_alvo):
    """Previsão alinhada: a saída da semana t-1 prevê a semana t. df_aq = semanas que só aquecem o estado."""
    df = pd.concat([df_aq, df_alvo], ignore_index=True)
    Xs, lp = _tensores(df, COLS_SECA)
    Xp, _ = _tensores(df, COLS_PICO)
    n_aq, n_alvo = df_aq['date'].nunique(), df_alvo['date'].nunique()
    se = df_alvo.groupby('date')['epiweek'].first().to_numpy() % 100
    mu, r = np.zeros((n_alvo, N)), np.zeros((n_alvo, N))
    hs = hp = None
    for t in range(n_aq + n_alvo - 1):
        lms, lrs, hs = modelo_seca(Xs[t], lp[t], hs)
        lmp, lrp, hp = modelo_pico(Xp[t], lp[t], hp)
        k = t + 1 - n_aq
        if k < 0:
            continue
        pico = se[k] >= 47 or se[k] <= 25
        mu[k] = torch.exp(lmp if pico else lms).squeeze(-1).float().cpu().numpy()
        r[k] = torch.exp(lrp if pico else lrs).squeeze(-1).float().cpu().numpy()
    eh_pico = (se >= 47) | (se <= 25)
    return np.maximum(mu, 1e-8), np.maximum(r, 1e-3), eh_pico


# ---------------------------------------------------------------- 2. fatores de calibração (validação)
sem_val = df_val['epiweek'].unique()
aq_val = df_val[df_val['epiweek'].isin(sem_val[:AQUEC])]
alvo_val = df_val[df_val['epiweek'].isin(sem_val[AQUEC:])]
mu_v, r_v, pico_v = _prever(aq_val, alvo_val)
y_v = alvo_val[VARS['casos_dengue']].to_numpy(np.float64).reshape(-1, N)
FATOR = {'pico': y_v[pico_v].sum() / mu_v[pico_v].sum(), 'seca': y_v[~pico_v].sum() / mu_v[~pico_v].sum()}
print(f"🎯 Fatores de calibração (validação {alvo_val['epiweek'].min()}–{alvo_val['epiweek'].max()}): "
      f"PICO × {FATOR['pico']:.3f} · SECA × {FATOR['seca']:.3f}")

# ---------------------------------------------------------------- 3. teste: previsão + calibração
mu, r, pico_t = _prever(df_ctx, df_te)
mu = mu * np.where(pico_t, FATOR['pico'], FATOR['seca'])[:, None]
QUANTIS = {'lower_95': .025, 'lower_90': .05, 'lower_80': .10, 'lower_50': .25, 'pred': .50,
           'upper_50': .75, 'upper_80': .90, 'upper_90': .95, 'upper_95': .975}

# municipal: quantis da NB de cada município
p = r / (r + mu)
df_mun_cal = df_te[['geocode', 'date', 'uf_code', VARS['casos_dengue']]].rename(columns={VARS['casos_dengue']: 'casos_reais'})
for col, q in QUANTIS.items():
    df_mun_cal[col] = nbinom.ppf(q, r, p).ravel().round(2)

# estadual: NB com a média e a variância da SOMA dos municípios
uf_no = df_te['uf_code'].to_numpy()[:N]
ufs = np.unique(uf_no)
A = (uf_no[None, :] == ufs[:, None]).astype(float)                 # (UF, município)
m_uf = mu @ A.T
v_uf = (mu + mu ** 2 / r) @ A.T
r_uf = np.where(v_uf > m_uf, m_uf ** 2 / np.maximum(v_uf - m_uf, 1e-9), 1e6)
p_uf = r_uf / (r_uf + m_uf)
datas_te = sorted(df_te['date'].unique())
df_est_cal = pd.DataFrame({'uf_code': np.tile(ufs, len(datas_te)), 'date': np.repeat(datas_te, len(ufs))})
for col, q in QUANTIS.items():
    df_est_cal[col] = nbinom.ppf(q, r_uf, p_uf).ravel().round(2)
real_uf = df_te.groupby(['uf_code', 'date'])[VARS['casos_dengue']].sum().rename('casos_reais').reset_index()
df_est_cal = df_est_cal.merge(real_uf, on=['uf_code', 'date'], how='left')

# ---------------------------------------------------------------- 4. salvar num banco NOVO e avaliar
DB_CAL = os.path.join(PASTA_LOCAL, 'resultados_agcrn_recalibrado.db')
df_mun_cal['season'] = SEASON_ALVO
df_est_cal['season'] = SEASON_ALVO
with sqlite3.connect(DB_CAL) as con:
    for tabela, df in (('municipal_results', df_mun_cal), ('estadual_results', df_est_cal)):
        if con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (tabela,)).fetchone():
            con.execute(f"DELETE FROM {tabela} WHERE season = ?", (SEASON_ALVO,))
        df.to_sql(tabela, con, if_exists='append', index=False)
csv_cal = os.path.join(PASTA_LOCAL, f"previsoes_estaduais_recalibradas_s{SEASON_ALVO}.csv")
df_est_cal.to_csv(csv_cal, index=False)
salvar_no_drive(DB_CAL)
salvar_no_drive(csv_cal)
print(f"💾 Salvo em {DB_CAL} (o banco original resultados_agcrn.db não foi alterado)")

avaliar_desempenho_season(db_path=DB_CAL, season_id=SEASON_ALVO)
