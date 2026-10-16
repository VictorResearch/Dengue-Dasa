import json, os, sqlite3
S1 = r'F:\AmbientePython\Projetos\data_imdc_2026\S1_corrigido'
nb = json.load(open(os.path.join(S1, 'Modelo2_v3.ipynb'), encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells']]
G = {'__name__': 'nb'}
for i in (3, 4, 7, 9, 11, 12, 14, 15, 21):
    exec(compile(cells[i], f'<celula {i}>', 'exec'), G)
G.update(PASTA_DADOS=S1, PASTA_SAIDA=os.path.join(S1, '_analise_v3'), SEMANAS_VALIDACAO=52, ENCOLHIMENTO_UF=2000, LAMBDA_TRAVA_LOG=0.2)
os.makedirs('local', exist_ok=True); os.chdir('local')
dm, de = G['prever_e_agregar_calibrado'](1)
dm['season'] = 1; de['season'] = 1
if os.path.exists('v3_teste.db'): os.remove('v3_teste.db')
with sqlite3.connect('v3_teste.db') as con:
    dm.to_sql('municipal_results', con, index=False); de.to_sql('estadual_results', con, index=False)
G['avaliar_desempenho_season'](db_path='v3_teste.db', season_id=1)
fora = ((de['pred'] < de['lower_50']) | (de['pred'] > de['upper_50'])).mean() * 100
print(f'pred estadual fora do IC50: {fora:.1f}% das UF-semanas')
