import json, os
S1 = os.path.dirname(os.path.abspath(os.getcwd() + '/x'))
S1 = r'F:\AmbientePython\Projetos\data_imdc_2026\S1_corrigido'
nb = json.load(open(os.path.join(S1, 'Modelo2_v2.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells']]
G = {'__name__': 'nb'}
for i in (3, 4, 7, 14, 15, 22):
    exec(compile(cells[i], f'<celula {i}>', 'exec'), G)
os.makedirs(os.path.join(S1, '_analise_v2', 'local'), exist_ok=True)
G.update(PASTA_DADOS=S1, PASTA_SAIDA=os.path.join(S1, '_analise_v2'), PASTA_LOCAL=os.path.join(S1, '_analise_v2', 'local'),
         SEASON_ALVO=1, SEMANAS_VALIDACAO=52, salvar_no_drive=lambda f: None)
os.chdir(G['PASTA_LOCAL'])
exec(compile(open(os.path.join(S1, 'celula_refazer_previsao.py'), encoding='utf-8').read(), '<celula nova>', 'exec'), G)
