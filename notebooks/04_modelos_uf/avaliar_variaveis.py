"""
Avaliação das variáveis no nível de TEMPORADA (UF × temporada), com origem móvel — sem vazamento.

Pergunta: com o que se sabe na SE25 do ano s, quanto cada grupo de variáveis ajuda a prever o TAMANHO da temporada s
(log da incidência acumulada SE41/s → SE40/s+1), além da referência "média das temporadas passadas da própria UF"?

  • Para cada s em 2015–2024, treina só com temporadas j COMPLETAS antes da SE25 de s (j ≤ s−2).
  • Alvo do modelo: desvio do tamanho em relação à referência (o modelo aprende o que a referência não sabe).
  • Modelos: Ridge (linear, padronizado) e LightGBM pequeno (não linear).
  • Medida: MAE em log (≈ erro multiplicativo) e habilidade = 1 − MAE/MAE_ref. Afinação = 2015–2021; teste = 2022–2024.
  • Análises: (a) só um grupo + referência; (b) todos menos um grupo; (c) todos; (d) correlação de Spearman de cada
    variável com o desvio (agrupado, só para leitura).

Saída: dados/avaliacao_variaveis.md + .csv          Uso: python avaliar_variaveis.py   (~1–2 min)
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
AQUI = os.path.dirname(os.path.abspath(__file__))
T = pd.read_parquet(os.path.join(AQUI, 'dados', 'tabela_uf_temporada.parquet'))
T = T.sort_values(['season', 'uf_code']).reset_index(drop=True)

GRUPOS = {
    'historico (taxa de ataque 1–4 temp., acumulada)': ['ar1_parcial', 'ar2', 'ar3', 'ar4', 'inc_acum_4temp',
                                                        'inc_acum_decaimento', 'inc_ano_se1_25'],
    'recente (últimas 4–8 sem., tendência, pico parcial)': ['inc_ult8', 'inc_ult4', 'tend8', 'pico_parcial_sem',
                                                            'razao_ult8_pico'],
    'clima observado (anomalias 12/26 sem.)': [c for c in T.columns if c.startswith('anom_')] + ['semanas_quentes_26',
                                                                                              'semanas_secas_26'],
    'previsão sazonal SEAS5': ['seas_temp_med_anom', 'seas_precip_tot_anom', 'seas_umid_med_anom'],
    'oceano (ENSO, IOD, PDO)': ['enso_12s', 'iod_12s', 'pdo_12s', 'enso_delta'],
    'Richards (estilo Episcanner)': [c for c in T.columns if c.startswith('rich')],
    'vizinhos (média simples, V4)': ['viz_ar1_parcial', 'viz_ar2', 'viz_inc_ult8', 'viz_tend8'],
    'vizinhos (razão agregada, nova)': [c for c in T.columns if c.startswith('vizp_')],
    'espalhamento (% municípios com casos)': ['pct_mun_com_casos_ult8', 'pct_pop_mun_com_casos_ult8'],
    'estáticas (bioma, Köppen, urbanização, pop.)': [c for c in T.columns if c.startswith('pct_pop_bioma_')]
    + ['pct_pop_koppen_grupo_A', 'pct_pop_koppen_grupo_B', 'pct_pop_koppen_grupo_C', 'pct_pop_mun_100k',
       'grau_vizinhanca', 'log_pop'],
}
TODAS = [c for g in GRUPOS.values() for c in g]
AVAL = list(range(2015, 2025))
AFIN, TESTE = [s for s in AVAL if s <= 2021], [s for s in AVAL if s >= 2022]


def referencia(df_treino, df_alvo):
    m = df_treino.groupby('uf_code').alvo_log_inc_temporada.mean()
    return df_alvo.uf_code.map(m).fillna(df_treino.alvo_log_inc_temporada.mean()).values


def ajusta_prediz(modelo, Xtr, ytr, Xte):
    if modelo == 'ridge':
        from sklearn.linear_model import Ridge
        med = np.nanmedian(Xtr, 0)
        med = np.where(np.isnan(med), 0, med)
        Xtr, Xte = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        return Ridge(alpha=10.0).fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=10, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=0)
    return m.fit(Xtr, ytr).predict(Xte)


def avalia(cols, modelo):
    erros = []
    for s in AVAL:
        tr = T[(T.season <= s - 2) & T.alvo_completo]
        te = T[(T.season == s) & T.alvo_completo]
        if tr.season.nunique() < 3 or len(te) == 0:
            continue
        r_tr, r_te = referencia(tr, tr), referencia(tr, te)
        if cols:
            d = ajusta_prediz(modelo, tr[cols].values.astype(float), tr.alvo_log_inc_temporada.values - r_tr,
                              te[cols].values.astype(float))
        else:
            d = 0.0
        pred = r_te + d
        erros.append(pd.DataFrame({'season': s, 'uf_code': te.uf_code.values,
                                   'erro_abs': np.abs(pred - te.alvo_log_inc_temporada.values)}))
    return pd.concat(erros, ignore_index=True)


def resumo(err, ref):
    out = {}
    for nome, ss in (('afinação 2015–21', AFIN), ('teste 2022–24', TESTE)):
        e, r = err[err.season.isin(ss)].erro_abs.mean(), ref[ref.season.isin(ss)].erro_abs.mean()
        out[f'MAE log · {nome}'] = e
        out[f'habilidade · {nome}'] = 1 - e / r
    return out


ref = avalia([], 'ridge')
linhas = [dict(cenario='referência (média da UF)', modelo='—', **resumo(ref, ref))]
for modelo in ('ridge', 'lgbm'):
    linhas.append(dict(cenario='TODAS as variáveis', modelo=modelo, **resumo(avalia(TODAS, modelo), ref)))
    for g, cols in GRUPOS.items():
        linhas.append(dict(cenario=f'só: {g}', modelo=modelo, **resumo(avalia(cols, modelo), ref)))
        linhas.append(dict(cenario=f'todas menos: {g}', modelo=modelo,
                           **resumo(avalia([c for c in TODAS if c not in cols], modelo), ref)))
R = pd.DataFrame(linhas)
R.to_csv(os.path.join(AQUI, 'dados', 'avaliacao_variaveis.csv'), index=False)

# correlação simples de cada variável com o desvio do tamanho (todas as temporadas completas; só leitura)
C = T[T.alvo_completo].copy()
desvio = []
for s, g in C.groupby('season'):
    tr = C[C.season <= s - 2]
    desvio.append(pd.Series(g.alvo_log_inc_temporada.values - (referencia(tr, g) if len(tr) else np.nan), index=g.index))
C['desvio'] = pd.concat(desvio)
cor = []
for g, cols in GRUPOS.items():
    for c in cols:
        ok = C[[c, 'desvio']].dropna()
        if len(ok) > 30 and ok[c].std() > 0:
            rho, p = spearmanr(ok[c], ok.desvio)
            cor.append(dict(grupo=g, variavel=c, spearman=rho, p_valor=p, n=len(ok)))
COR = pd.DataFrame(cor).sort_values('spearman', key=np.abs, ascending=False)
COR.to_csv(os.path.join(AQUI, 'dados', 'correlacao_variaveis.csv'), index=False)

pd.set_option('display.width', 250)
with open(os.path.join(AQUI, 'dados', 'avaliacao_variaveis.md'), 'w', encoding='utf-8') as fh:
    fh.write('# Avaliação das variáveis (nível de temporada, origem móvel)\n\n')
    fh.write('Habilidade = 1 − MAE/MAE da referência (média das temporadas passadas da UF). > 0 = ajuda.\n\n')
    fh.write('```\n' + R.round(3).to_string(index=False) + '\n```')
    fh.write('\n\n## Correlação de Spearman com o desvio do tamanho (todas as temporadas completas)\n\n')
    fh.write('```\n' + COR.round(3).head(40).to_string(index=False) + '\n```\n')
print(R.round(3).to_string())
print()
print(COR.round(3).head(25).to_string())
