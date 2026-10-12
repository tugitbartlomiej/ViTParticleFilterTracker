# T0 i T1: nagranie uczenia krok po kroku i równanie bez sterowania (2026-09-29)

## T0: nagranie (`nagranie.py`)

Dwa lokalne nagrania uczenia małego ViT, ziarno 47, 3000 kroków, RTX 3070: `nagranie_s47` (6 min) i `nagranie_s47_sily`
(z siłami od kroku 900, 16 min). Pliki poza gitem (kilka GB).

Zgodność z Edenem (ziarno 47, ramię `bez`, kroki od 1000, zapis co 10 kroków):

| | nagranie | Eden |
|---|---|---|
| norma przyrostu latentu narzędzia na 10 kroków | 2,288 | 2,271 |
| korelacja kolejnych przyrostów (10 kroków) | −0,238 | −0,237 |
| norma latentu | 6,37 | 5,89 |

Test T0 zdany: nagranie ma tę samą statystykę ruchu co biegi z Edena.

Co krok ruch wygląda inaczej: norma przyrostu 0,40, a korelacja kolejnych przyrostów **+0,75** (pęd AdamW). Na 10
krokach ta pamięć gaśnie i odwraca się, stąd ujemna korelacja na zapisie z Edena.

Dokładny rozkład przyrostu (podmiana wag, co 10 kroków, kroki ≥ 1000): same wagi wyjaśniają 97,8 % przyrostu, same
bufory BatchNorm 0,7 %, same wagi ostatniego bloku 7,5 %. Latent porusza się więc przez zmiany wag całej sieci, nie
przez BatchNorm i nie przez sam ostatni blok.

## T1: równanie z TFT bez sterowania (`t1_bez_sterowania.py`, `wyniki/T1.json`)

Przekład: dryf do średnich ruchomych (τ = 2..128) + prędkość i przyspieszenie + pamięć z tłumionych, obracających się
jąder na prędkości + szum barwny z własnych reszt; każda z 768 współrzędnych osobno, współczynniki wspólne dla sond.

| dane | umiejętność testowa |
|---|---|
| Eden co 10 kroków, 10 ziaren, mediana | 17,6 (wspólne współczynniki), 16,8 (osobne na współrzędną) |
| nagranie co krok, sama prędkość a·v (a = 0,76) | 56,5 |
| nagranie co krok, pełne równanie T1 (osobne współczynniki; z członem N przeliczonym w T2) | 58,2 (60,2) |
| nagranie co 10 kroków, pełne równanie T1 | 17,5 |

Sprawdzono też sterowanie samą statystyką pikseli paczki (jasność, kontrast, odbicia) odtworzoną z generatora
losowego: test ujemny. Wniosek: z samej historii latentu jeden krok da się przewidzieć w około 60 %, 10 kroków w około
17 %. Reszta ruchu pochodzi z paczek danych.
