"""
Monta a ÁREA DE TRABALHO que os notebooks esperam: uma pasta plana com as bibliotecas, os dados prontos e os
parâmetros de produção. É essa pasta que vai para o Google Drive como MyDrive/dengue_V4/ (Colab) ou que serve de
pasta de trabalho para rodar os notebooks localmente (Jupyter aberto nela).

  <destino>/
    v4_lib.py, comp_lib.py, ..., r7_lib.py, prod_lib.py, graficos_previsoes.py, comp_inla.R, uf_autoarima.R
    dados/               ← dados/v4/          (painéis por região de saúde e UF; 02_preparar_painel_v4.py)
    dados_uf/            ← dados/uf/          (tabelas por UF; 03_preparar_tabela_uf.py)
    referencias_imdc3/   ← dados/referencias_imdc3/ (previsões publicadas do 3º IMDC; 00_baixar_dados.py)
    parametros/          ← producao/parametros/ (hiperparâmetros e pesos escolhidos na afinação)

Os resultados de cada notebook (resultados_*, cache_*, avaliacao_*) são criados DENTRO do destino.

Uso:
  python scripts/04_montar_area_de_trabalho.py                        # → ./area_de_trabalho/
  python scripts/04_montar_area_de_trabalho.py --destino "G:/Meu Drive/dengue_V4"   # Drive sincronizado no PC
"""
import argparse
import glob
import os
import shutil
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')     # console do Windows (cp1252)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTAS = {'dados': os.path.join(RAIZ, 'dados', 'v4'),
          'dados_uf': os.path.join(RAIZ, 'dados', 'uf'),
          'referencias_imdc3': os.path.join(RAIZ, 'dados', 'referencias_imdc3'),
          'parametros': os.path.join(RAIZ, 'producao', 'parametros')}
ARQUIVOS = (glob.glob(os.path.join(RAIZ, 'src', '*.py')) + glob.glob(os.path.join(RAIZ, 'src', '*.R'))
            + [os.path.join(RAIZ, 'producao', 'prod_lib.py'), os.path.join(RAIZ, 'producao', 'relatorio_campeoes.py'),
               os.path.join(RAIZ, 'graficos', 'graficos_previsoes.py')])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--destino', default=os.path.join(RAIZ, 'area_de_trabalho'))
    a = ap.parse_args()
    os.makedirs(a.destino, exist_ok=True)
    for f in ARQUIVOS:
        shutil.copy(f, a.destino)
    print(f'{len(ARQUIVOS)} arquivos de código copiados')
    faltam = []
    for nome, origem in PASTAS.items():
        if not os.path.isdir(origem) or not os.listdir(origem):
            faltam.append(f'{nome} ({os.path.relpath(origem, RAIZ)})')
            continue
        shutil.copytree(origem, os.path.join(a.destino, nome), dirs_exist_ok=True)
        print(f'{nome:18s} ← {os.path.relpath(origem, RAIZ)}')
    if faltam:
        print('⚠️ faltam (rode antes os scripts 00, 02 e 03):', ', '.join(faltam))
        return 1
    print('Área de trabalho pronta em', a.destino)
    return 0


if __name__ == '__main__':
    sys.exit(main())
