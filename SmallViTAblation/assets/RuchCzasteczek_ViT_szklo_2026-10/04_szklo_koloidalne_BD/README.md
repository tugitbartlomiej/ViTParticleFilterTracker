# Szkło koloidalne: symulacje dynamiki browna (kontrola dodatnia)

Model odniesienia: miękkie kule polidyspersyjne, dynamika browna (BD), stała temperatura (`stala`) albo stygnięcie (`stygnace`), kilka gęstości upakowania φ. Kod `model_szkla_eden.py`, `szklo_eden.slurm` (kopie w `07_kod/eden/2026-10-10_szklo_koloidalne/kod`). Porównanie z tokenami: `06_analizy/porownanie_szklo_20261010/README.md`, sekcje 4 i 4b.

## osie_aplikacji: te same symulacje w formacie aplikacji

Przyrostek `_duze`: duży bieg N = 4096, 10⁷ kroków. Bez przyrostka: bieg szybki N = 2048.

| oś | tryb | φ | cząstek | kroków zapisu | zakres | rozmiar |
|---|---|---|---|---|---|---|
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.56_duze` | stala | 0.56 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.58` | stala | 0.58 | 2048 | 700 | 0–6990 | 11.9 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.58_duze` | stala | 0.58 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.6` | stala | 0.6 | 2048 | 700 | 0–6990 | 11.9 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.61_duze` | stala | 0.61 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.62` | stala | 0.62 | 2048 | 700 | 0–6990 | 11.9 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.62_duze` | stala | 0.62 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.63_duze` | stala | 0.63 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.64` | stala | 0.64 | 2048 | 700 | 0–6990 | 11.9 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.64_duze` | stala | 0.64 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.66_duze` | stala | 0.66 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stala_phi_0.6_duze` | stala | 0.6 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.56_duze` | stygnace | 0.56 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.58` | stygnace | 0.58 | 2048 | 700 | 0–6990 | 11.9 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.58_duze` | stygnace | 0.58 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.6` | stygnace | 0.6 | 2048 | 700 | 0–6990 | 11.9 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.61_duze` | stygnace | 0.61 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.62` | stygnace | 0.62 | 2048 | 700 | 0–6990 | 11.9 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.62_duze` | stygnace | 0.62 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.63_duze` | stygnace | 0.63 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.64` | stygnace | 0.64 | 2048 | 700 | 0–6990 | 11.9 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.64_duze` | stygnace | 0.64 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.66_duze` | stygnace | 0.66 | 4096 | 2000 | 0–49975 | 66.8 MB |
| `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_0.6_duze` | stygnace | 0.6 | 4096 | 2000 | 0–49975 | 66.8 MB |

## eden_symulacje: surowe wyniki z Edena

| katalog | plików | rozmiar |
|---|---|---|
| `wyniki/stala` | 16010 | 0.54 GB |
| `wyniki/stygnace` | 10981 | 0.37 GB |

Sumy kontrolne z Edena: `eden_symulacje/SHA256SUMS_EDEN_szklo_bd.txt`.
