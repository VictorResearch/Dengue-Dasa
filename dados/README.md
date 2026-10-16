# dados/

| Pasta | Vem de | No repositório? |
|---|---|---|
| `brutos/` | `scripts/00_baixar_dados.py` (FTP do 3º IMDC) | não (~3,3 GB) |
| `referencias_imdc3/` | `scripts/00_baixar_dados.py` (GitHub do 3º IMDC) | não |
| `municipal/S{1..4}/` | `scripts/01_refazer_municipal.py` | não |
| `v4/` | `scripts/02_preparar_painel_v4.py` | não |
| `uf/` | `scripts/03_preparar_tabela_uf.py` | não |
| `derivados/` | `scripts/01_refazer_municipal.py --season 1` (cópia pronta) | **sim**: `nos_municipios.csv` (geocode → nó, com UF, região, Köppen, bioma e centróide) e `W1_queen_municipios.npz` (vizinhança Queen entre municípios, da malha oficial), usados pelos scripts 02 e 03 |

Detalhes: [docs/02_dados.md](../docs/02_dados.md).
