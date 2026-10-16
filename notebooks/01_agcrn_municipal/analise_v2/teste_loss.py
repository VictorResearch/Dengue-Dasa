import json, torch
from scipy.stats import nbinom
nb = json.load(open(r'F:\AmbientePython\Projetos\data_imdc_2026\S1_corrigido\Modelo2_v3.ipynb', encoding='utf-8'))
G = {'__name__': 'nb', 'LAMBDA_TRAVA_LOG': 0.2}
exec(''.join(nb['cells'][11]['source']), G); exec(''.join(nb['cells'][12]['source']), G)
torch.manual_seed(0)
y = torch.tensor([0., 0., 1., 5., 50., 3000., 12000.]).unsqueeze(-1)
log_mu = torch.tensor([-8., 0.5, 0.1, 1.2, 4.0, 8.2, 9.9]).unsqueeze(-1).requires_grad_()
log_r = torch.tensor([-2., 0., 1., 0.5, 2., 1., 3.]).unsqueeze(-1).requires_grad_()
nll, _ = G['_nb_nll'](log_mu, log_r, y)
mu, r = log_mu.exp().detach().numpy().ravel(), log_r.exp().detach().numpy().ravel()
ref = -nbinom.logpmf(y.numpy().ravel(), r, r / (r + mu))
print('NLL nova  :', nll.detach().numpy().ravel().round(4))
print('scipy NB  :', ref.round(4))
for nome in ('nb_loss_pico', 'nb_loss_seca'):
    L = G[nome](log_mu.half(), log_r.half(), y)      # como sob autocast fp16
    L.backward()
    print(nome, 'loss', round(L.item(), 4), '| gradientes finitos:', bool(torch.isfinite(log_mu.grad).all() and torch.isfinite(log_r.grad).all()))
    log_mu.grad = None; log_r.grad = None
# simetria da trava: prever 3x a mais e 3x a menos com y = 300
for fator in (3, 1/3):
    lm = torch.tensor([[float(torch.log(torch.tensor(300. * fator)))]])
    print(f'prever {fator:.2f}x o real (y=300): trava =', round(float(torch.nn.functional.smooth_l1_loss(torch.log1p(lm.exp()) - torch.log1p(torch.tensor([[300.]])), torch.zeros(1, 1), beta=1.0)), 3))
