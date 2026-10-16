"""
Baixa TODOS os dados públicos usados no projeto e deixa em dados/brutos/ e dados/referencias_imdc3/.

  1. Dados oficiais do 3º IMDC (Mosqlimate/InfoDengue): FTP anônimo info.dengue.mat.br, pasta data_imdc_2026.
     Casos de dengue e chikungunya por município × semana, clima observado (ERA5) e previsto (SEAS5), oscilações
     oceânicas (ENSO, IOD, PDO), população, variáveis ambientais, mapa de regionais de saúde, malhas e acessos Afya.
     Os .csv.gz são descompactados para .csv (é o que os scripts leem). ~3,3 GB descompactados.
  2. Previsões publicadas das equipes do 3º IMDC (inclui o modelo-base oficial BB, a régua da competição):
     github.com/Mosqlimate-project/3rd_IMDC_results, pasta predictions/, arquivos dengue_state_*.csv.gz.

Fontes baixadas DENTRO dos notebooks (não precisam deste script):
  - sorotipo do SINAN (OpenDataSUS, DENGBRaa.csv.zip) → notebook 09_rodada4_literatura/R4_0;
  - temperatura do mar ERSST v5 (NOAA PSL / NCEI) → notebook 09_rodada4_literatura/R4_1.

Uso:
  python scripts/00_baixar_dados.py                    # baixa o que falta (retoma; não baixa de novo)
  python scripts/00_baixar_dados.py --listar           # só mostra o que existe no FTP e no GitHub
  python scripts/00_baixar_dados.py --atualizacoes     # também os arquivos *_update_2026 (dados depois de 03/2026)

Reprodutibilidade: as análises do projeto usaram os arquivos-base (dengue até a SE10 de 2026), SEM os *_update_2026.
"""
import argparse
import ftplib
import gzip
import json
import os
import shutil
import sys
import time
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')     # console do Windows (cp1252)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRUTOS = os.path.join(RAIZ, 'dados', 'brutos')
REFS = os.path.join(RAIZ, 'dados', 'referencias_imdc3')

FTP_HOST, FTP_PASTA = 'info.dengue.mat.br', 'data_imdc_2026'
FTP_USUARIO, FTP_SENHA = 'anonymous', 'anonymous@domain.com'      # acesso anônimo indicado na documentação oficial
ARQUIVOS_FTP = [
    'dengue.csv.gz', 'chikungunya.csv.gz', 'climate.csv.gz', 'forecasting_climate.csv.gz',
    'ocean_climate_oscillations.csv.gz', 'datasus_population_2001_2025.csv.gz', 'environ_vars.csv.gz',
    'access_afya_dengue_2021_2026.csv.gz', 'map_regional_health.csv',
    'shape_muni.gpkg', 'shape_regional_health.gpkg', 'shape_macroregional_health.gpkg',
]
ATUALIZACOES = ['dengue_update_2026.csv.gz', 'chikungunya_update_2026.csv.gz', 'climate_update_2026.csv.gz',
                'forecasting_climate_update_2026.csv.gz', 'ocean_climate_oscillations_update_2026.csv.gz']

GH_REPO, GH_REF = 'Mosqlimate-project/3rd_IMDC_results', 'main'
GH_ARVORE = f'https://api.github.com/repos/{GH_REPO}/git/trees/{GH_REF}?recursive=1'
GH_RAW = f'https://raw.githubusercontent.com/{GH_REPO}/{GH_REF}/' + '{caminho}'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def _destino(nome):
    """o .csv.gz é guardado descompactado como .csv (os scripts leem .csv)."""
    return os.path.join(BRUTOS, nome[:-3] if nome.endswith('.csv.gz') else nome)


def baixar_ftp(nomes, listar=False):
    os.makedirs(BRUTOS, exist_ok=True)
    ftp = ftplib.FTP(FTP_HOST, timeout=120)
    ftp.login(FTP_USUARIO, FTP_SENHA)
    ftp.cwd(FTP_PASTA)
    existentes = set(ftp.nlst())
    if listar:
        log(f'FTP {FTP_HOST}/{FTP_PASTA}: {len(existentes)} arquivos')
        for n in sorted(existentes):
            print('  ', n)
        ftp.quit()
        return
    for nome in nomes:
        dest = _destino(nome)
        if os.path.exists(dest):
            log(f'já existe: {os.path.relpath(dest, RAIZ)}')
            continue
        if nome not in existentes:
            log(f'⚠️ {nome} não está no FTP (o conteúdo pode ter mudado); confira com --listar')
            continue
        tmp = os.path.join(BRUTOS, nome + '.parcial')
        log(f'baixando {nome} ...')
        with open(tmp, 'wb') as f:
            ftp.retrbinary(f'RETR {nome}', f.write, blocksize=1 << 20)
        if nome.endswith('.csv.gz'):
            with gzip.open(tmp, 'rb') as gi, open(dest + '.parcial', 'wb') as go:
                shutil.copyfileobj(gi, go, length=1 << 24)
            os.remove(tmp)
            os.replace(dest + '.parcial', dest)
        else:
            os.replace(tmp, dest)
        log(f'  ok: {os.path.relpath(dest, RAIZ)} ({os.path.getsize(dest) / 1e6:,.0f} MB)')
    ftp.quit()


def baixar_referencias(listar=False):
    os.makedirs(REFS, exist_ok=True)
    with urllib.request.urlopen(GH_ARVORE, timeout=120) as r:
        arvore = json.load(r)['tree']
    caminhos = sorted(x['path'] for x in arvore
                      if x['path'].startswith('predictions/dengue_state_') and x['path'].endswith('.csv.gz'))
    if listar:
        log(f'GitHub {GH_REPO}@{GH_REF}: {len(caminhos)} previsões estaduais')
        for c in caminhos:
            print('  ', c)
        return
    for c in caminhos:
        dest = os.path.join(REFS, os.path.basename(c))
        if os.path.exists(dest):
            continue
        urllib.request.urlretrieve(GH_RAW.format(caminho=c), dest + '.parcial')
        os.replace(dest + '.parcial', dest)
    log(f'referências do 3º IMDC: {len(caminhos)} arquivos em {os.path.relpath(REFS, RAIZ)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--listar', action='store_true', help='só lista o que existe nas fontes')
    ap.add_argument('--atualizacoes', action='store_true', help='baixa também os *_update_2026')
    ap.add_argument('--sem-referencias', action='store_true', help='não baixa as previsões do 3º IMDC')
    a = ap.parse_args()
    nomes = ARQUIVOS_FTP + (ATUALIZACOES if a.atualizacoes else [])
    baixar_ftp(nomes, listar=a.listar)
    if not a.sem_referencias:
        baixar_referencias(listar=a.listar)
    if not a.listar:
        faltam = [n for n in ARQUIVOS_FTP if not os.path.exists(_destino(n))]
        log('tudo pronto.' if not faltam else f'⚠️ faltam: {faltam}')
        return 1 if faltam else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
