"""RODADA 6 · otimização dos dois modelos de temporada que mais se aproximam do pico (doc 20 do mapa).

Na R5 (teste 2022–2025):
  - MIX_elastico5_UF ficou BAIXO no pico (altura −12%, total −8%);
  - MIX_escala5_UF acertou a altura (+12%), mas INFLOU a entressafra (+151%).

O que esta rodada faz:
  ① os dois modelos são reescritos aqui com TODOS os botões abertos (valores padrão = R5, reproduzida exatamente):
       MIX_elastico: w, faixas de nível da elasticidade, elasticidade só dos anos acima do normal, teto de b,
                     expoente γ em b, afinação da forma η nos anos grandes, antecipação do pico nos anos grandes;
       MIX_escala:   limiar, troca suave (τ), expoente p do fator, nível L abaixo do qual o fator tem teto, teto;
  ② um DETECTOR PARA CADA MODELO: o da R5 como está ou um ridge REAJUSTADO (fonte das variáveis, janela do normal,
     k, ρ mínimo, α, peso dos anos grandes), com alargamento, correção de viés β e cauda do sorteio próprios;
  ③ OPTUNA (TPE com restrição) — objetivo SEM critério misto:
       MIX_elastico: minimizar o WIS das semanas de PICO;
       MIX_escala:   minimizar o WIS das semanas de ENTRESSAFRA;
     sujeito a: as outras duas fases (e a intermediária) não piorarem mais que TOL (5%).
     Cada número é a razão contra a R5 na mesma UF (média geométrica entre UFs: cada UF pesa igual).
     0,85 = 15% menos erro que a R5 naquela fase;
  ④ VALIDAÇÃO: busca em 2015–2021 (7 temporadas, com anos grandes: 2015, 2016, 2019); antes de 2018 o normal é a
     climatologia (o ENS_grafos3_λ só existe de 2018 em diante). A busca inteira é repetida deixando cada temporada de
     fora, para medir o ganho fora da amostra (validação aninhada). O teste 2022–2025 é olhado uma vez, no fim.
Sem vazamento: o detector reajustado escolhe as variáveis e treina só com temporadas terminadas antes do corte.
"""
import os
import shutil
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import v4_lib as V
import comp_lib as L
import uf_lib as U
import semanal_lib as S
import r2_lib as R2
import r3_lib as R3
import r4_lib as R4
import r5_lib as R5

Q, QCOLS, ZQ = R3.Q, R3.QCOLS, R3.ZQ
TUNING, TESTE = U.TUNING, U.TESTE
VALIDACAO = list(range(2015, 2022))
LIM_GRANDE = R3.LIM_GRANDE
FASES = ('entressafra', 'intermediaria', 'alta (pico)')
salvar_tabela, json_grava, json_le, carimbo, _grava_pq = R4.salvar_tabela, R4.json_grava, R4.json_le, R4.carimbo, R4._grava_pq
NIVEIS_ELAST = {'padrao': R4.NIVEIS_ELAST, 'fino': (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 9)}

# =====================================================================================================================
# parâmetros: padrão (= R5) e o que cada um faz
# =====================================================================================================================
DESCRICAO = {
    'det_modo': 'detector: "R5" = o escolhido na R5 (R3:ridge) como está; "refit" = ridge reajustado aqui',
    'det_fonte': 'refit: variáveis da R3 ou R3 + R4 (sem as nacionais novas da R4)',
    'det_janela': 'refit: normal da UF = média de todas as temporadas (0) ou só das últimas N',
    'det_k': 'refit: número máximo de variáveis',
    'det_rho_min': 'refit: |correlação de Spearman| mínima para a variável entrar (nacionais: 0,30)',
    'det_alpha': 'refit: força do encolhimento do ridge (maior = puxa mais para a média)',
    'det_peso_grande': 'refit: peso dos anos grandes (> 2× o normal) no treino',
    'det_alarg': 'alargamento da faixa do detector em torno da mediana',
    'det_beta': 'correção de viés: soma β × (mediana prevista, se > 0) ao log-tamanho; > 0 sobe os anos grandes',
    'det_cauda': 'até quantas larguras de faixa o sorteio extrapola além do q95 (cauda de cima)',
    'w': 'MIX_elastico: peso do tamanho previsto (1 = vai até o tamanho do detector; < 1 puxa para o normal)',
    'niveis': 'MIX_elastico: faixas de nível em que a elasticidade é estimada ("fino" = mais faixas no topo)',
    'b_so_acima': 'MIX_elastico: elasticidade estimada só com temporadas acima do normal',
    'b_teto': 'MIX_elastico: teto da elasticidade b (R5 = 1,5)',
    'gama': 'MIX_elastico: expoente em b (> 1 concentra o crescimento no pico e poupa a entressafra)',
    'eta': 'MIX_elastico: afinação da forma nos anos grandes (> 1 = pico mais alto e estreito, mesmo total)',
    'antecip': 'MIX_elastico: semanas de antecipação do pico por unidade de log-tamanho (máx. 6)',
    'limiar': 'MIX_escala: log-tamanho a partir do qual reescala (log 2 = 2× o normal)',
    'tau': 'MIX_escala: suavidade da troca (0 = liga/desliga, como na R5)',
    'p': 'MIX_escala: expoente do fator (alvo/normal)^p (1 = R5)',
    'nivel_L': 'MIX_escala: semanas com nível normal abaixo de L (log1p incidência/100 mil) têm teto no fator',
    'teto_log2': 'MIX_escala: teto do fator nessas semanas = 2^valor (10 ≈ sem teto, como na R5)',
}


def padrao(modelo, w_r5=1.0, limiar_r5=float(np.log(2))):
    """parâmetros que reproduzem a R5 (só os que o Optuna sorteia quando det_modo = 'R5')."""
    p = dict(det_modo='R5', det_alarg=1.0, det_beta=0.0, det_cauda=2.5)
    if modelo == 'elastico':
        p.update(w=float(w_r5), niveis='padrao', b_so_acima=False, b_teto=1.5, gama=1.0, eta=1.0, antecip=0.0)
    else:
        p.update(limiar=float(limiar_r5), tau=0.0, p=1.0, nivel_L=-0.5, teto_log2=10.0)
    return p


REFIT_PADRAO = dict(det_fonte='r3', det_janela=0, det_k=10, det_rho_min=0.12, det_alpha=10.0, det_peso_grande=1.0)


def completa(par):
    """acrescenta os parâmetros do detector reajustado que não foram sorteados (det_modo = 'R5')."""
    return {**REFIT_PADRAO, **par}


def espaco(trial, modelo, tem_r4=True):
    """espaço de busca (os valores da R5 estão dentro de todos os intervalos)."""
    p = {'det_modo': trial.suggest_categorical('det_modo', ['R5', 'refit'])}
    if p['det_modo'] == 'refit':
        p['det_fonte'] = trial.suggest_categorical('det_fonte', ['r3', 'r3+r4'] if tem_r4 else ['r3'])
        p['det_janela'] = trial.suggest_categorical('det_janela', [0, 5, 8])
        p['det_k'] = trial.suggest_int('det_k', 4, 16)
        p['det_rho_min'] = trial.suggest_float('det_rho_min', 0.05, 0.30)
        p['det_alpha'] = trial.suggest_float('det_alpha', 0.01, 1000.0, log=True)
        p['det_peso_grande'] = trial.suggest_float('det_peso_grande', 1.0, 6.0)
    p['det_alarg'] = trial.suggest_float('det_alarg', 0.7, 1.8)
    p['det_beta'] = trial.suggest_float('det_beta', -0.3, 0.6)
    p['det_cauda'] = trial.suggest_float('det_cauda', 1.0, 4.0)
    if modelo == 'elastico':
        p['w'] = trial.suggest_float('w', 0.5, 1.6)
        p['niveis'] = trial.suggest_categorical('niveis', ['padrao', 'fino'])
        p['b_so_acima'] = trial.suggest_categorical('b_so_acima', [False, True])
        p['b_teto'] = trial.suggest_float('b_teto', 1.0, 3.0)
        p['gama'] = trial.suggest_float('gama', 0.5, 3.0)
        p['eta'] = trial.suggest_float('eta', 1.0, 2.5)
        p['antecip'] = trial.suggest_float('antecip', 0.0, 3.0)
    else:
        p['limiar'] = trial.suggest_float('limiar', 0.0, 1.4)
        p['tau'] = trial.suggest_float('tau', 0.0, 0.5)
        p['p'] = trial.suggest_float('p', 0.4, 1.2)
        p['nivel_L'] = trial.suggest_float('nivel_L', -0.5, 4.0)
        p['teto_log2'] = trial.suggest_float('teto_log2', 0.0, 10.0)
    return completa(p)


# =====================================================================================================================
# ① dados pré-calculados por temporada (o que não depende dos parâmetros)
# =====================================================================================================================
def _elasticidades(D, TAM, s, c, W, niveis, so_acima):
    """como R4.elasticidades, mas sem o clip final (o teto é parâmetro) e com as variantes. Devolve (b, mu) em (W, N)."""
    js = [j for j in TAM.index if (R3._completa_ate(D, j) or 10 ** 9) <= c and j < s]
    if not js:
        return np.ones((W, D.N)), np.zeros((W, D.N))
    Ys = np.stack([np.log1p(np.nan_to_num(D.INC[D.idx_temporada(j)[:52]])) for j in js])
    mu = Ys.mean(0)
    estende = lambda m: np.vstack([m, np.repeat(m[-1:], max(0, W - 52), 0)])[:W]       # noqa: E731
    if len(js) < 4:
        return np.ones((W, D.N)), estende(mu)
    A = (TAM.loc[js] - TAM.loc[js].mean()).values
    dev = Ys - mu[None]
    Ab = np.broadcast_to(A[:, None, :], Ys.shape)
    mub = np.broadcast_to(mu[None], Ys.shape)
    lim = NIVEIS_ELAST[niveis]
    centros, inc = [], []
    for lo, hi in zip(lim[:-1], lim[1:]):
        m = (mub >= lo) & (mub < hi)
        if so_acima:
            m = m & (Ab > 0)
        if m.sum() < 40:
            continue
        x, y = Ab[m], dev[m]
        centros.append(float(np.median(mub[m])))
        inc.append(float((x * y).sum() / max((x * x).sum(), 1e-9)))
    if len(centros) < 2:
        return np.ones((W, D.N)), estende(mu)
    b = np.interp(mu, centros, np.maximum.accumulate(inc))
    return estende(b), estende(mu)


class Base6:
    """tudo o que não depende dos parâmetros, por temporada: trajetórias do normal (números aleatórios comuns a todas
    as tentativas), sorteios do detector, verdade, fase observada, tabelas do detector e elasticidades (em cache)."""

    def __init__(self, ctx, NORMAL, DET5, X3, X4=None, temporadas=VALIDACAO, M=200, normal='ENS_grafos3_λ'):
        D = self.D = ctx.Du
        self.M = M
        self.TAM = R3.tamanhos(D)
        lim = R2.limiares(D).set_index('uf_code').reindex(D.unidades)
        self.p25, self.p75 = lim.p25.values, lim.p75.values
        self.DET5 = DET5[DET5.corte == 'jun']
        # tabelas do detector (corte de junho) e colunas candidatas por fonte
        NV = R3.NAO_VARIAVEIS
        self.tab = {'r3': X3[X3.corte == 'jun'].copy()}
        self.cols = {'r3': [c for c in X3.columns if c not in NV and X3[c].dtype.kind in 'fi']}
        if X4 is not None:
            novas = [c for c in X4.columns if c not in X3.columns and c not in NV and X4[c].dtype.kind in 'fi']
            nac_novas = {c for c in novas if R5._nacional(X4, c)}
            self.tab['r3+r4'] = X4[X4.corte == 'jun'].copy()
            self.cols['r3+r4'] = self.cols['r3'] + [c for c in novas if c not in nac_novas]
            print(f'detector R3+R4: {len(self.cols["r3+r4"])} variáveis candidatas · nacionais novas excluídas: {sorted(nac_novas)}')
        g = self.tab['r3']
        self.nacionais = {c for c in self.cols.get('r3+r4', self.cols['r3']) if R5._nacional(self.tab.get('r3+r4', g), c)}
        self._alvos, self._rho, self._elast = {}, {}, {}
        self.dados = {}
        nm = NORMAL[NORMAL.modelo == normal]
        for s in temporadas:
            self.dados[s] = self._prepara(ctx, nm, s)
        com_r5 = [s for s in temporadas if np.isfinite(self.DET5[self.DET5.season == s][QCOLS].values.astype(float)).any()]
        print('Base6 · normal por temporada:', {s: d['normal'] for s, d in self.dados.items()}, f'· M = {M}')
        print('        detector da R5 disponível em', com_r5, '(nas outras, o modo "R5" usa o ridge reajustado com os parâmetros da R3)')

    def _prepara(self, ctx, nm, s):
        D = self.D
        c = R3.indice_corte(D, s, 'jun')
        g = nm[nm.season == s]
        if len(g):
            datas = np.sort(g.date.unique())
            qn = V.de_df(g, datas, D.unidades)
            fonte = 'normal'
        else:
            datas, qn = V.base_climatologia(D, s)
            fonte = 'climatologia'
        qn = np.sort(np.maximum(np.nan_to_num(qn), 0), -1)
        W = len(datas)
        rng = np.random.default_rng(1000 * s + 7)
        zt, za = rng.standard_normal((D.N, self.M)), rng.standard_normal((D.N, self.M))
        lq = np.log1p(qn)
        traj = np.stack([np.expm1(R3._interp(lq[:, j], zt[j])).clip(0) for j in range(D.N)]) + 1e-3     # (N, M, W)
        idx = np.array([D.idx_data[pd.Timestamp(x)] for x in datas])
        y = ctx.ver_uf.y(datas)
        inc = y / D.POP[idx] * 1e5
        fase = np.where(np.isnan(inc), -1, np.where(inc >= self.p75, 2, np.where(inc < self.p25, 0, 1)))
        return dict(datas=datas, W=W, c=c, qn=qn, traj=traj, Tn=traj.sum(-1), za=za, y=y, fase=fase, pop=D.POP[c],
                    completa=R3._completa_ate(D, s) is not None and not np.isnan(y).any(), normal=fonte)

    # ---------------------------------------------------------------- detector
    def _alvo(self, n):
        """base (normal da UF) e alvo (log-tamanho relativo) de cada temporada × UF, com janela n (0 = todas)."""
        if n not in self._alvos:
            D, TAM = self.D, self.TAM
            lin = []
            for j in sorted(self.tab['r3'].season.unique()):
                c = R3.indice_corte(D, j, 'jun')
                if c is None:
                    continue
                ant = [i for i in TAM.index if (R3._completa_ate(D, i) or 10 ** 9) <= c and i < j]
                if not ant:
                    continue
                base = TAM.loc[ant[-n:] if n else ant].mean().reindex(D.unidades).values
                alvo = TAM.loc[j].reindex(D.unidades).values - base if j in TAM.index else np.full(D.N, np.nan)
                lin.append(pd.DataFrame(dict(season=j, uf_code=D.unidades, base_n=base, alvo_n=alvo)))
            self._alvos[n] = pd.concat(lin, ignore_index=True)
        return self._alvos[n]

    def _treino(self, fonte, n, s):
        c = self.dados[s]['c'] if s in self.dados else R3.indice_corte(self.D, s, 'jun')
        X = self.tab[fonte].merge(self._alvo(n), on=['season', 'uf_code'], how='inner')
        ok = X.season.map(lambda j: (R3._completa_ate(self.D, j) or 10 ** 9) <= c) & (X.season < s)
        return X[ok].dropna(subset=['alvo_n']), X[X.season == s]

    def _rhos(self, fonte, n, s):
        k = (fonte, n, s)
        if k not in self._rho:
            tr, _ = self._treino(fonte, n, s)
            r = {}
            for col in self.cols[fonte]:
                x = tr[[col, 'alvo_n']].dropna()
                if len(x) > 40 and x[col].std() > 0:
                    r[col] = float(spearmanr(x[col], x.alvo_n)[0])
            self._rho[k] = pd.Series(r, dtype=float)
        return self._rho[k]

    def detector(self, s, par, min_treino=50):
        """(q (N, 9) do log-tamanho relativo, base (N,)) já com alargamento e correção de viés. NaN = sem detector."""
        D = self.D
        q, base = np.full((D.N, 9), np.nan), np.full(D.N, np.nan)
        if par['det_modo'] == 'R5':
            d = self.DET5[self.DET5.season == s].set_index('uf_code').reindex(D.unidades)
            if not np.isfinite(d[QCOLS].values.astype(float)).any():
                # o detector da R5 (R3:ridge) só existe a partir de 2018 (as variáveis de imunidade precisam de 3
                # temporadas anteriores): antes disso, o ridge reajustado com os parâmetros da R3
                return self.detector(s, {**par, **REFIT_PADRAO, 'det_modo': 'refit'}, min_treino)
            q, base = d[QCOLS].values.astype(float), d.base.values.astype(float)
        else:
            fonte, n = par['det_fonte'], int(par['det_janela'])
            rho = self._rhos(fonte, n, s)
            lim = np.where([c in self.nacionais for c in rho.index], 0.30, par['det_rho_min'])
            rho = rho[np.abs(rho.values) >= lim]
            sel = list(rho.abs().sort_values(ascending=False).index[:int(par['det_k'])])
            tr, te = self._treino(fonte, n, s)
            sel = [c for c in sel if tr[c].notna().sum() >= min_treino]
            if sel and len(tr) >= min_treino and len(te):
                from sklearn.linear_model import Ridge
                med_tr = tr[sel].median()
                Xtr, ytr = tr[sel].fillna(med_tr).values, tr.alvo_n.values
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                te = te.set_index('uf_code').reindex(D.unidades)
                Xte = te[sel].fillna(med_tr).values
                sw = np.where(ytr > LIM_GRANDE, par['det_peso_grande'], 1.0)
                m = Ridge(alpha=par['det_alpha']).fit((Xtr - mu) / sd, ytr, sample_weight=sw)
                res = ytr - m.predict((Xtr - mu) / sd)
                q = m.predict((Xte - mu) / sd)[:, None] + np.quantile(res, Q)[None]
                base = te.base_n.values.astype(float)
        med = q[:, [4]]
        q = med + par['det_alarg'] * (q - med) + par['det_beta'] * np.maximum(med, 0)
        return np.sort(q, 1), base

    def elast(self, s, niveis, so_acima):
        k = (s, niveis, bool(so_acima))
        if k not in self._elast:
            d = self.dados[s]
            b, mu = _elasticidades(self.D, self.TAM, s, d['c'], d['W'], niveis, so_acima)
            self._elast[k] = (b.T.copy(), mu.T.copy())                                # (N, W)
        return self._elast[k]

    # ---------------------------------------------------------------- previsão
    def prever(self, s, modelo, par):
        """quantis (W, N, 9) do modelo ('elastico' ou 'escala') na temporada s com os parâmetros `par`."""
        d = self.dados[s]
        q, base = self.detector(s, par)
        ok = np.isfinite(q).all(1) & np.isfinite(base)
        a = _sorteia(np.where(ok[:, None], q, 0.0), d['za'], par['det_cauda'])
        bf = np.where(ok, base, 0.0)
        if modelo == 'elastico':
            b, _ = self.elast(s, par['niveis'], par['b_so_acima'])
            out = _elastico(d, a, bf, par, b)
        else:
            _, mu = self.elast(s, 'padrao', False)
            out = _escala(d, a, bf, par, mu)
        qq = np.transpose(np.quantile(out, Q, axis=1), (2, 1, 0))                     # (W, N, 9)
        qq[:, ~ok] = d['qn'][:, ~ok]
        return V.arruma_quantis(qq)

    def para_df(self, s, q, nome):
        return V.para_df(q, self.dados[s]['datas'], self.D.unidades, nome, s)

    # ---------------------------------------------------------------- avaliação (somas por UF × fase)
    def acumula(self, modelo, par, temporadas):
        N = self.D.N
        A, n = np.zeros((3, N)), np.zeros((3, N))
        alt, tot = [], []
        for s in temporadas:
            d = self.dados[s]
            q = self.prever(s, modelo, par)
            w = V.wis(q, d['y'])
            for f in range(3):
                m = d['fase'] == f
                A[f] += np.where(m, w, 0).sum(0)
                n[f] += m.sum(0)
            if d['completa']:
                y, med = d['y'], q[..., 4]
                for j in range(N):
                    if y[:, j].max() > 0:
                        alt.append(med[:, j].max() / y[:, j].max() - 1)
                        tot.append(med[:, j].sum() / max(y[:, j].sum(), 1) - 1)
        return dict(wis=A, n=n, alt=np.array(alt), tot=np.array(tot))


def _sorteia(q, z, cauda):
    """q (N, 9), z (N, M) → (N, M) pela função quantil (como R3._interp, com a cauda de cima como parâmetro)."""
    i = np.clip(np.searchsorted(ZQ, z), 1, len(ZQ) - 1)
    t = np.clip((z - ZQ[i - 1]) / (ZQ[i] - ZQ[i - 1]), -1.5, cauda)
    lo, hi = np.take_along_axis(q, i - 1, 1), np.take_along_axis(q, i, 1)
    return lo + t * (hi - lo)


def _kappa(tr, b, alvo, it=40):
    """κ (N, M) tal que Σ_w tr·exp(κ·b_w) = alvo (bisseção, vetorizada em UFs × trajetórias)."""
    lo, hi = np.full(alvo.shape, -12.0), np.full(alvo.shape, 12.0)
    la = np.log(np.maximum(alvo, 1e-6))
    for _ in range(it):
        k = (lo + hi) / 2
        f = np.log(np.sum(tr * np.exp(np.clip(k[..., None] * b[:, None, :], -50, 50)), -1) + 1e-9) - la
        hi = np.where(f > 0, k, hi)
        lo = np.where(f > 0, lo, k)
    return (lo + hi) / 2


def _elastico(d, a, base, par, b):
    traj, Tn, W = d['traj'], d['Tn'], d['W']
    alvo = np.expm1(base[:, None] + a) * d['pop'][:, None] / 1e5
    alvo = np.exp(np.log(Tn) + par['w'] * (np.log(np.maximum(alvo, 1)) - np.log(Tn)))
    tr = traj
    g = np.clip(np.maximum(a, 0) / np.log(4), 0, 1)                   # 0 no ano normal → 1 a partir de 4× o normal
    if par['eta'] != 1:
        e = 1 + (par['eta'] - 1) * g
        tr = tr ** e[..., None]
        tr = tr * (Tn / tr.sum(-1))[..., None]
    if par['antecip'] > 0:
        k = np.minimum(np.rint(par['antecip'] * np.maximum(a, 0)), 6).astype(int)
        ii = np.minimum(np.arange(W)[None, None, :] + k[..., None], W - 1)
        tr = np.take_along_axis(tr, ii, -1)
    bb = np.clip(b, 0.03, par['b_teto']) ** par['gama']
    kap = _kappa(tr, bb, alvo)
    return tr * np.exp(np.clip(kap[..., None] * bb[:, None, :], -50, 50))


def _escala(d, a, base, par, mu):
    traj, Tn = d['traj'], d['Tn']
    alvo = np.expm1(base[:, None] + a) * d['pop'][:, None] / 1e5
    f = (np.maximum(alvo, 0) / Tn) ** par['p']
    if par['tau'] < 0.02:
        g = (a > par['limiar']).astype(float)
    else:
        g = 1 / (1 + np.exp(-np.clip((a - par['limiar']) / par['tau'], -50, 50)))
    alta = (mu >= par['nivel_L'])[:, None, :]
    fw = np.where(alta, f[..., None], np.minimum(f[..., None], 2.0 ** par['teto_log2']))
    return traj * (1 + g[..., None] * (fw - 1))


# =====================================================================================================================
# ② métricas: razão contra a R5 por fase (média geométrica entre UFs)
# =====================================================================================================================
def soma_acc(accs):
    return dict(wis=sum(a['wis'] for a in accs), n=sum(a['n'] for a in accs),
                alt=np.concatenate([a['alt'] for a in accs]), tot=np.concatenate([a['tot'] for a in accs]))


def razoes(acc, ref, min_semanas=3):
    """{fase: WIS do candidato ÷ WIS da R5, média geométrica entre UFs} + geral + erros mediano de altura e total (%)."""
    out = {}
    for f, nome in enumerate(FASES):
        ok = (ref['wis'][f] > 0) & (acc['n'][f] >= min_semanas)
        out[nome] = float(np.exp(np.mean(np.log(acc['wis'][f][ok] / ref['wis'][f][ok])))) if ok.any() else np.nan
    a, r = acc['wis'].sum(0), ref['wis'].sum(0)
    ok = r > 0
    out['geral'] = float(np.exp(np.mean(np.log(a[ok] / r[ok]))))
    out['erro_altura_mediano_pct'] = float(100 * np.median(acc['alt'])) if len(acc['alt']) else np.nan
    out['erro_total_mediano_pct'] = float(100 * np.median(acc['tot'])) if len(acc['tot']) else np.nan
    return out


OBJETIVO = {'elastico': ('alta (pico)', ('entressafra', 'intermediaria')),
            'escala': ('entressafra', ('alta (pico)', 'intermediaria'))}


# =====================================================================================================================
# ③ Optuna (estudo em SQLite local, copiado para o Drive; retoma)
# =====================================================================================================================
def _restricoes(trial):
    return trial.user_attrs.get('restricoes', [1.0, 1.0])


def _copia(local, drive):
    if os.path.exists(local):
        shutil.copy(local, drive + '.tmp')
        os.replace(drive + '.tmp', drive)


def estudo(nome, pasta, seed=1):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    base_local = '/content/optuna_r6' if os.path.isdir('/content') else os.path.join(pasta, '_optuna_local')
    os.makedirs(base_local, exist_ok=True)
    local, drive = os.path.join(base_local, f'{nome}.db'), os.path.join(pasta, f'{nome}.db')
    if os.path.exists(drive) and not os.path.exists(local):
        shutil.copy(drive, local)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=25,
                                         constraints_func=_restricoes)
    st = optuna.create_study(study_name=nome, storage='sqlite:///' + local.replace('\\', '/'), direction='minimize',
                             load_if_exists=True, sampler=sampler)
    return st, local, drive


def concluidas(st):
    import optuna
    return [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]


def otimizar(B, modelo, temporadas, nome, pasta, n_trials, par0, tol=0.05, seed=1, mostrar=True):
    """roda (ou retoma) o estudo até ter n_trials tentativas concluídas. A 1ª tentativa é a R5 (par0)."""
    import optuna
    st, local, drive = estudo(nome, pasta, seed)
    ref = B.acumula(modelo, completa(par0), temporadas)
    alvo, restr = OBJETIVO[modelo]
    tem_r4 = 'r3+r4' in B.tab

    def f(trial):
        par = espaco(trial, modelo, tem_r4)
        r = razoes(B.acumula(modelo, par, temporadas), ref)
        for k, v in r.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr('restricoes', [float(r[x] - (1 + tol)) if np.isfinite(r[x]) else 1.0 for x in restr])
        return r[alvo]

    if not st.trials:
        st.enqueue_trial(par0)
    falta = n_trials - len(concluidas(st))
    if falta > 0:
        def salva(study, trial):
            if trial.number % 10 == 0:
                _copia(local, drive)
            if mostrar and trial.number % 25 == 0:
                m = escolher(study, par0)[1]
                print(f'  {nome}: tentativa {trial.number} · melhor viável até agora = {m:.3f}', flush=True)
        st.optimize(f, n_trials=falta, callbacks=[salva], catch=(Exception,))
    _copia(local, drive)
    return st


def escolher(st, par0):
    """melhor tentativa VIÁVEL (todas as restrições ≤ 0). Sem nenhuma viável, fica com a R5."""
    ok = [t for t in concluidas(st) if all(v <= 0 for v in _restricoes(t)) and t.value is not None]
    if not ok:
        return completa(par0), 1.0, None
    t = min(ok, key=lambda t: t.value)
    return completa({**par0, **t.params}), float(t.value), t


def tentativas(st):
    lin = []
    for t in concluidas(st):
        lin.append(dict(numero=t.number, valor=t.value, viavel=all(v <= 0 for v in _restricoes(t)),
                        **{k: v for k, v in t.user_attrs.items() if k != 'restricoes'}, **t.params))
    return pd.DataFrame(lin)


# =====================================================================================================================
# ④ validação aninhada: a busca inteira é refeita deixando cada temporada de fora
# =====================================================================================================================
def validacao_aninhada(B, modelo, temporadas, pasta, n_trials, par0, tol=0.05):
    """para cada temporada h: busca nas outras → melhor viável → avalia em h contra a R5. Grava e retoma por dobra."""
    arq = os.path.join(pasta, f'aninhada_{modelo}.json')
    feito = json_le(arq, {})
    accs_m, accs_r, lin = [], [], []
    for h in temporadas:
        resto = [s for s in temporadas if s != h]
        chave = str(h)
        if chave not in feito:
            st = otimizar(B, modelo, resto, f'r6_{modelo}_sem_{h}', pasta, n_trials, par0, tol, seed=h, mostrar=False)
            par, v, _ = escolher(st, par0)
            feito[chave] = dict(params=par, valor_busca=v)
            json_grava(feito, arq)
            print(f'  {modelo}: dobra sem {h} pronta (busca = {v:.3f})', flush=True)
        par = feito[chave]['params']
        am, ar = B.acumula(modelo, par, [h]), B.acumula(modelo, completa(par0), [h])
        accs_m.append(am)
        accs_r.append(ar)
        lin.append(dict(temporada_fora=h, valor_na_busca=feito[chave]['valor_busca'],
                        **{f'fora_{k}': v for k, v in razoes(am, ar).items()},
                        **{f'R5_{k}': v for k, v in razoes(ar, ar).items() if k.startswith('erro')}))
    T = pd.DataFrame(lin)
    total = razoes(soma_acc(accs_m), soma_acc(accs_r))
    return T, total


# =====================================================================================================================
# ⑤ teste: razão contra a R5 por fase, a partir das previsões em formato longo
# =====================================================================================================================
def razoes_longo(TT, modelos, ref):
    """TT com y, fase (R2.com_fase) e QCOLS: WIS da fase ÷ WIS do `ref` na mesma UF, média geométrica entre UFs."""
    T = TT.assign(w=V.wis(TT[QCOLS].values.astype(float), TT.y.values.astype(float)))
    lin = []
    for m in modelos:
        d = dict(modelo=m, contra=ref)
        for f in FASES + ('geral',):
            g = T if f == 'geral' else T[T.fase == f]
            p = g.groupby(['modelo', 'uf_code']).w.sum().unstack('modelo')
            if m not in p or ref not in p:
                continue
            ok = (p[ref] > 0) & (p[m] > 0)
            d[f] = float(np.exp(np.mean(np.log(p[m][ok] / p[ref][ok]))))
        lin.append(d)
    return pd.DataFrame(lin)


def tabela_parametros(par0, melhor):
    """parâmetro × (R5, otimizado) com a descrição. Os do detector reajustado só valem com det_modo = 'refit'."""
    base = completa(par0)
    fmt = lambda v: round(v, 3) if isinstance(v, float) else v                       # noqa: E731
    lin = []
    for k in DESCRICAO:
        if k not in base:
            continue
        refit = k in REFIT_PADRAO
        v0 = '—' if refit else fmt(base[k])
        v1 = '—' if refit and melhor.get('det_modo') != 'refit' else fmt(melhor.get(k, base[k]))
        lin.append(dict(parametro=k, R5=v0, otimizado=v1, mudou=str(v0) != str(v1), o_que_faz=DESCRICAO[k]))
    return pd.DataFrame(lin)
