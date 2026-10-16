import runpy, numpy as np, pandas as pd
ns = runpy.run_path('diagnostico_v3.py')
calibrar, aplicar, metr, Yuf, A, pt, GK = ns['calibrar'], ns['aplicar'], ns['metr'], ns['Yuf'], ns['A'], ns['pt'], ns['GRADE_K']
var = {}
f, K = calibrar()
um = {'pico': np.ones(A.shape[1]), 'seca': np.ones(A.shape[1])}
o, _ = aplicar(um, K, 'mediana_k1'); var['sem fator (modelo puro) + mediana k=1, IC com K da validação'] = metr(Yuf, o)
ns['GRADE_K'][:] = [k for k in GK if k <= 8]
f8, K8 = calibrar(); o, _ = aplicar(f8, K8, 'mediana_k1'); var[f'fator UF + mediana k=1, IC com K ≤ 8 {K8}'] = metr(Yuf, o)
o, _ = aplicar(um, K8, 'mediana_k1'); var[f'sem fator + mediana k=1, IC com K ≤ 8'] = metr(Yuf, o)
ns['ENC'] = 20000
f2, K2 = calibrar(); o, _ = aplicar(f2, K8, 'mediana_k1'); var['fator UF encolhido forte (20000) + mediana k=1, K ≤ 8'] = metr(Yuf, o)
print(pd.DataFrame(var).T.round(2).to_string())
