# @title [V3] Previsão + calibração na validação + soma estadual (substitui o MinT)
# Por que não o MinT: ele calculava os pesos com os erros do PRÓPRIO TESTE (casos_reais), isto é, olhava a
# resposta; e, como o estado era a soma das medianas municipais, a entrada já era coerente e o MinT só
# encolhia as maiores cidades por um efeito da regularização.
#
# O que esta célula faz, usando só a TEMPORADA DE VALIDAÇÃO (últimas SEMANAS_VALIDACAO semanas do treino):
#  1. FATOR DE VIÉS por regime (PICO/SECA) e por UF = casos reais / casos previstos na validação,
#     encolhido para o fator nacional quando a UF tem pouco volume (ENCOLHIMENTO_UF).
#  2. SOMA ESTADUAL: média = soma das médias municipais; variância = soma das variâncias × k, representada
#     por uma Binomial Negativa. k (por regime) corrige a correlação entre municípios vizinhos e é escolhido
#     para que os IC50/IC90 estaduais da validação fiquem perto do nominal.
#     O PONTO (pred) do estado é a mediana SEM a inflação; a inflação só alarga os intervalos (que sempre contêm o ponto).
#  3. Previsão alinhada: a semana t-1 prevê a semana t; o estado da rede é aquecido nas 15 semanas anteriores.
import pyarrow.parquet as pq
from scipy.stats import nbinom

AQUEC_PREV = 15
GRADE_K = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
QUANTIS_POS = {'lower_95': .025, 'lower_90': .05, 'lower_80': .10, 'lower_50': .25, 'pred': .50,
           'upper_50': .75, 'upper_80': .90, 'upper_90': .95, 'upper_95': .975}


def _nb_estadual(mu, r, A, k=1.0):
    """Quantis da soma por UF: NB com média = soma das médias e variância = soma das variâncias × k."""
    m = mu @ A.T
    v = (mu + mu ** 2 / r) @ A.T
    rr = np.where(v > m, m ** 2 / np.maximum(k * (v - m), 1e-9), 1e6)
    return {c: nbinom.ppf(q, rr, rr / (rr + m)) for c, q in QUANTIS_POS.items()}


def prever_e_agregar_calibrado(season_id: int):
    t0 = time.time()
    torch.set_grad_enabled(False)
    # ------------------------------------------------------------ dados (ordem date, node_id)
    arq_tr = os.path.join(PASTA_DADOS, f"train{season_id}_FINAL_NN_v2.parquet")
    eps = np.sort(pq.read_table(arq_tr, columns=['epiweek'])['epiweek'].to_pandas().unique())
    df_val = pd.read_parquet(arq_tr, filters=[('epiweek', '>=', int(eps[-(SEMANAS_VALIDACAO + AQUEC_PREV)]))])
    df_ctx = pd.read_parquet(os.path.join(PASTA_DADOS, f"ctx{season_id}_FINAL_NN_v2.parquet"))
    df_te = pd.read_parquet(os.path.join(PASTA_DADOS, f"test{season_id}_FINAL_NN_v2.parquet"))
    df_val, df_ctx, df_te = [d[d[VARS['uf_code']] != 32].sort_values(['date', 'node_id']).reset_index(drop=True)
                             for d in (df_val, df_ctx, df_te)]
    N = df_te['node_id'].nunique()
    uf_no = df_te[VARS['uf_code']].to_numpy()[:N]
    ufs = np.unique(uf_no)
    A = (uf_no[None, :] == ufs[:, None]).astype(float)              # (UF, município)

    # ------------------------------------------------------------ modelos
    def carregar(nome, cols):
        m = PreditorNBAGCRN(N, len(cols), 64, cols.index('velocidade'), cols.index('aceleracao')).to(device)
        arq = f"modelo_agcrn_{nome}_s{season_id}.pt"
        caminho = arq if os.path.exists(arq) else os.path.join(PASTA_SAIDA, arq)
        sd = torch.load(caminho, map_location=device, weights_only=False)
        m.load_state_dict({k.replace('_orig_mod.', ''): v.float() for k, v in sd.items()})
        print(f"📦 Pesos: {caminho}")
        return m.eval()
    mod_s, mod_p = carregar('SECA', COLS_SECA), carregar('PICO', COLS_PICO)

    def tens(df, cols):
        n = df['date'].nunique()
        return torch.tensor(df[cols].to_numpy(np.float32).reshape(n, N, len(cols)), device=device)

    def prever(df_aq, df_alvo):
        df = pd.concat([df_aq, df_alvo], ignore_index=True)
        Xs, Xp = tens(df, COLS_SECA), tens(df, COLS_PICO)
        lp = torch.tensor(df['log_pop'].to_numpy(np.float32).reshape(-1, N), device=device)
        n_aq, n_alvo = df_aq['date'].nunique(), df_alvo['date'].nunique()
        se = df_alvo.groupby('date')[VARS['semana_epi']].first().to_numpy() % 100
        pico = (se >= 47) | (se <= 25)
        mu, r = np.zeros((n_alvo, N)), np.zeros((n_alvo, N))
        hs = hp = None
        for t in range(n_aq + n_alvo - 1):
            lms, lrs, hs = mod_s(Xs[t], lp[t], hs)
            lmp, lrp, hp = mod_p(Xp[t], lp[t], hp)
            k = t + 1 - n_aq
            if k < 0:
                continue
            mu[k] = torch.exp(lmp if pico[k] else lms).squeeze(-1).float().cpu().numpy()
            r[k] = torch.exp(lrp if pico[k] else lrs).squeeze(-1).float().cpu().numpy()
        return np.maximum(mu, 1e-8), np.maximum(r, 1e-3), pico

    # ------------------------------------------------------------ calibração na validação
    sem_val = df_val[VARS['semana_epi']].unique()
    alvo_val = df_val[df_val[VARS['semana_epi']].isin(sem_val[AQUEC_PREV:])]
    mu_v, r_v, pico_v = prever(df_val[df_val[VARS['semana_epi']].isin(sem_val[:AQUEC_PREV])], alvo_val)
    y_v = alvo_val[VARS['casos_dengue']].to_numpy(np.float64).reshape(-1, N)
    yv_uf = y_v @ A.T
    fator_no, K = {}, {}
    print(f"🎯 Calibração na validação (EW {alvo_val[VARS['semana_epi']].min()}–{alvo_val[VARS['semana_epi']].max()}):")
    for nome, m in (('pico', pico_v), ('seca', ~pico_v)):
        g = y_v[m].sum() / mu_v[m].sum()
        num, den = yv_uf[m].sum(0), (mu_v[m] @ A.T).sum(0)
        w = den / (den + ENCOLHIMENTO_UF) if ENCOLHIMENTO_UF > 0 else np.zeros_like(den)
        f_uf = np.where(den > 0, w * num / np.maximum(den, 1e-9) + (1 - w) * g, g)
        fator_no[nome] = f_uf @ A
        mu_c = mu_v[m] * fator_no[nome]
        erros = []
        for k in GRADE_K:
            q = _nb_estadual(mu_c, r_v[m], A, k)
            c50 = np.mean((yv_uf[m] >= q['lower_50']) & (yv_uf[m] <= q['upper_50'])) * 100
            c90 = np.mean((yv_uf[m] >= q['lower_90']) & (yv_uf[m] <= q['upper_90'])) * 100
            erros.append(abs(c50 - 50) + abs(c90 - 90))
        K[nome] = GRADE_K[int(np.argmin(erros))]
        print(f"   {nome.upper()}: fator nacional × {g:.3f} · por UF de × {f_uf.min():.2f} a × {f_uf.max():.2f} · "
              f"inflação da variância estadual k = {K[nome]}")

    # ------------------------------------------------------------ teste
    mu, r, pico_t = prever(df_ctx, df_te)
    mu = mu * np.where(pico_t[:, None], fator_no['pico'][None, :], fator_no['seca'][None, :])
    p = r / (r + mu)
    df_mun = df_te[[VARS['municipio'], 'date', VARS['uf_code'], VARS['casos_dengue']]] \
        .rename(columns={VARS['casos_dengue']: 'casos_reais'}).reset_index(drop=True)
    for col, q in QUANTIS_POS.items():
        df_mun[col] = nbinom.ppf(q, r, p).ravel().round(2)
    datas_te = sorted(df_te['date'].unique())
    q_est = {c: np.zeros((len(datas_te), len(ufs))) for c in QUANTIS_POS}
    for nome, m in (('pico', pico_t), ('seca', ~pico_t)):
        # [V3.1] ponto = mediana da soma SEM inflação. Com k grande (ex.: SECA k=48 na S1), a mediana da NB inflada
        # despenca para perto de zero (a distribuição fica muito assimétrica) e a previsão estadual colapsava
        # no início e no fim da temporada, embora a média estivesse certa.
        # Intervalos = união do intervalo sem inflação com o inflado: nunca mais estreitos que o calibrado e
        # sempre contendo o ponto.
        q = _nb_estadual(mu[m], r[m], A, K[nome])
        q1 = _nb_estadual(mu[m], r[m], A, 1.0)
        for c in QUANTIS_POS:
            if c.startswith('lower'):
                q_est[c][m] = np.minimum(q[c], q1[c])
            elif c.startswith('upper'):
                q_est[c][m] = np.maximum(q[c], q1[c])
            else:
                q_est[c][m] = q1[c]
    df_est = pd.DataFrame({VARS['uf_code']: np.tile(ufs, len(datas_te)), 'date': np.repeat(datas_te, len(ufs))})
    for c in QUANTIS_POS:
        df_est[c] = q_est[c].ravel().round(2)
    real = df_te.groupby([VARS['uf_code'], 'date'])[VARS['casos_dengue']].sum().rename('casos_reais').reset_index()
    df_est = df_est.merge(real, on=[VARS['uf_code'], 'date'], how='left')
    torch.set_grad_enabled(True)
    print(f"✅ Previsão calibrada pronta em {time.time() - t0:.0f}s")
    return df_mun, df_est
