import runpy, numpy as np, pandas as pd
ns = runpy.run_path('diagnostico_agregacao.py')
Y, MU, QM, ew, datas = ns['Y'], ns['MU'], ns['QM'], ns['ew'], ns['datas']
br = pd.DataFrame({'data': datas, 'SE': ew, 'real': Y.sum(1), 'soma_medianas': QM['med'].sum(1), 'soma_mu': MU.sum(1)})
br['razao_med/real'] = br['soma_medianas'] / br['real']
br['cresc_real_%'] = br['real'].pct_change() * 100
br['regime'] = np.where((br.SE >= 47) | (br.SE <= 25), 'pico', 'seca')
print(br.round(2).to_string(index=False))
print(br.groupby('regime')[['real', 'soma_medianas', 'soma_mu']].sum().assign(razao=lambda d: d.soma_medianas / d.real).round(2))
