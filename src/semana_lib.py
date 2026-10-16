"""Redes de grafo do uf_lib (MTGNN, A3T-GCN, GCN-GRU, STGCN) nas REGRAS DO FELIPE: previsão de curto prazo
(h = 1…H semanas), modelo treinado uma vez por rodada (dados até `fim_treino`) e aplicado a cada semana do período de
avaliação com o histórico observado até a origem (sem reajustar), como o piloto dele.
Mesmas entradas, perda (pinball nos 9 quantis), âncora sazonal e sementes do `_RedeGrafoUF`; só muda o horizonte
(H = 4 em vez de 68) e as origens de previsão. Não altera o uf_lib."""
import numpy as np
import pandas as pd
import v4_lib as V
import uf_lib as U

QCOLS = V.QCOLS


def _indice(D, data, lado):
    """'fim': última semana inteira até `data`; 'ini': primeira semana que começa em `data` ou depois."""
    d = pd.to_datetime(D.datas)
    if lado == 'fim':
        return int(np.searchsorted(d, pd.Timestamp(data) - pd.Timedelta(days=6), 'right') - 1)
    return int(np.searchsorted(d, pd.Timestamp(data), 'left'))


def prever_semanal(rede, D, fim_treino, ini_av, fim_av, H=4, lote=64):
    """rede: instância de U.MTGNN / U.GCNGRU / U.STGCN. D: ctx.Du.
    Devolve DataFrame longo: modelo, h, origem, date, uf_code, 9 quantis (casos)."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    c = _indice(D, fim_treino, 'fim')
    X, S, wo, Y, Mk, _, _ = U._amostras_janela(D, c, rede.L, H, 1)
    P = V.PreparaA(D, c, L=rede.L, H=H)                               # padronizações só até c
    A = D.ADJ + np.eye(D.N)
    dg = 1 / np.sqrt(A.sum(1))
    A = (A * dg[:, None] * dg[None]).astype(np.float32)
    T_ = lambda a, tp=torch.float32: torch.tensor(a, dtype=tp, device=dev)   # noqa: E731
    Xt, St, Wt = T_(X), T_(S), T_(wo, torch.long)
    Yt, Mt = T_(Y.transpose(0, 2, 1)), T_(Mk.transpose(0, 2, 1).astype(np.float32))

    t_ini, t_fim = _indice(D, ini_av, 'ini'), min(_indice(D, fim_av, 'fim'), D.ultimo_obs)
    origens = np.arange(max(t_ini - H, rede.L), t_fim)                  # origem o: dados observados até a semana o
    Xo = np.stack([P.X[o - rede.L + 1:o + 1] for o in origens]).astype(np.float32)
    So, Wo = P.S[origens].astype(np.float32), D.woyc[origens]
    res = []
    for seed in rede.sementes:
        torch.manual_seed(seed)
        m = rede.construir(nn, torch, D.N, X.shape[-1], S.shape[-1], H, A, rede.d).to(dev)
        U._treina(m, lambda b: ((Xt[b], St[b], Wt[b]), Yt[b], Mt[b]), len(Xt), rede.epocas, rede.lr, seed, dev)
        m.eval()
        with torch.no_grad():
            res.append(np.concatenate([m(T_(Xo[i:i + lote]), T_(So[i:i + lote]), T_(Wo[i:i + lote], torch.long)).cpu().numpy()
                                       for i in range(0, len(origens), lote)]))          # (B, N, H, 9)
    r = np.sort(np.median(np.stack(res), 0), -1)
    teto = np.log1p(3 * np.nanmax(D.INC[:c + 1], 0))
    linhas = []
    for i, o in enumerate(origens):
        for k in range(1, H + 1):
            t = o + k
            if not (t_ini <= t <= t_fim):
                continue
            anc = P.CM[P.ult_woy[o, D.woyc[t]], np.arange(D.N)]
            lq = np.minimum(anc[:, None] + r[i, :, k - 1], teto[:, None])
            q = np.clip(np.expm1(lq), 0, None) * D.POP[t][:, None] / 1e5
            linhas.append(pd.DataFrame(q, columns=QCOLS).assign(modelo=rede.nome, h=k, origem=D.datas[o],
                                                                date=D.datas[t], uf_code=D.unidades))
    return pd.concat(linhas, ignore_index=True)


def metricas(df, col='pred'):
    y, p = df['y'].values, df[col].values
    den = np.abs(y) + np.abs(p)
    s = np.where(den > 0, 2 * np.abs(p - y) / np.where(den > 0, den, 1), 0)
    return dict(n=len(df), MAE=np.abs(p - y).mean(), WAPE=100 * np.abs(p - y).sum() / y.sum(), sMAPE=100 * s.mean(),
                sMAPE_0a100=50 * s.mean(), sMAPE_mediana=100 * np.median(s))
