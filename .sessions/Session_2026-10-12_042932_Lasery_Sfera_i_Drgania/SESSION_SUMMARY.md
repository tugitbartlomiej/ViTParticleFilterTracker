# Session Summary: Lasery na sferze i źródło drgań cząsteczek

## Metadata
- **Date:** 2026-10-12
- **Time:** ok. 02:00–04:30
- **Status:** Completed
- **Type:** Analysis (filmy z prawdziwych danych + pomiar drgań krok po kroku)

## Objective
1. „Lasery powinny być rozmieszczone na sferze, 768, świecić do 600 cząsteczek (zielone) i do punktu centralnego (niebieskie); gdzie krzyżują się zielone wiązki, w kroku 0 są cząsteczki; przesunięcia ognisk pokaż wektorami; animacja 3D.”
2. „Co powoduje drgania cząsteczek? Temperatura jest odpowiednikiem?”
3. „Zaznacz to w kolejnym filmiku, kontynuuj ze sferą i laserami, dodaj coś, co fizycznie powoduje drgania.”
4. Wyłączenie komputera za 2 h, zapis sesji.

## Context
Poprzednia sesja (Session_2026-10-11_210254_Werdykt_Krysztal_Tokenow) ustaliła werdykt: sieć adresów narzucona przez kodowanie pozycji, lokalne sprzężenie z nakładania pól lodygi, słabe globalne przez uwagę. Powstały filmy laserowe 1–3 (matplotlib). Zostało zadanie naped_v2 na Edenie (1826939).

## Actions Taken
1. Ustalono źródło danych: pełny latent 768D zdjęcia TRAIN 0 po 806 checkpointach leży lokalnie (`assets/RuchCzasteczek_ViT_szklo_2026-10/03_.../ruch60_rp128_50tr_50val/pelne768.npy`) — Eden niepotrzebny.
2. Film 4 `film_lasery_4_sfera.py`: moderngl offscreen, render 4K → mipmapa do 1080p, HDR + tonemapping; 768 × 601 wiązek instancjonowanych z buforów SSBO; lasery na sferze Fibonacciego przydzielone (algorytm węgierski) wg środka obszaru, który neuron oświetla na końcu; strzałki g → węzeł (±300 kroków); zbliżenie na cząsteczkę narzędzia z paskiem 768 natężeń.
3. naped_v2 (1826939) zakończony: pobrano `eden/wyniki/naped_v2/`, dopisano sekcję „AKTUALIZACJA 2026-10-12” do `WERDYKT_KRYSZTAL_2026-10-11.md`.
4. Odpowiedź o temperaturze (z pomiarów fal 3–4): drgania = szum kroku wag przez Jakobian; temperatura = długość kroku (klatka ∝ lr^0,43, paczka nie grzeje przy Adam/obcinaniu gradientu); ograniczenia: szum wspólny, kierunkowy, nierównowagowy.
5. Analiza drgań krok po kroku (gęsty zapis biegu 35 000, sonda 0 = TR_0001_S1_P03/838, kroki 0–1094, PACZKI.npz z |δθ|, lr, indeksami paczek).
6. Film 5: `drgania_prep.py` (latent → scratchpad `drgania_X.npy` 1 GB, `drgania.npz` z miniaturami 15 paczek z zipów zbioru) i `film_lasery_5_drgania.py` (generator θ nad sferą, światłowody z impulsem, iskry paczki, błyski laserów = |Δz| neuronu, strzałki skoku × 4, panel udziałów, termometr |δθ|, wykres klatki).
7. `shutdown /s /t 7200` (wyłączenie ok. 06:29).

## Results

### Key Findings
- Film 4: średnie przesunięcie ogniska po 60 epokach 3,09 jednostki (narzędzie 5,09, reszta 3,02), IoU 0,93; natężeń > 0,1: 40% → 16%; najsilniejsza wiązka 2,3 → 5,1; w epokach 0–2 pole przesunięć wiruje (obrót sieci).
- Drgania (1. epoka, 768D, odstęp sąsiadów a0 = 1,49):
  - Spearman |δθ| vs |Δx| na krok 0,35 przy właściwym wyrównaniu (przesunięte o ±1 krok: 0,20 / 0,25);
  - skok tokenów ~0,39 a0 na krok (po rozgrzewce), w rozgrzewce 0,17 a0;
  - udział energii skoku (mediana kroków 300+): wspólny zdjęcia 0,48, gładkie pole 3×3 0,36, własny 0,09; w krokach 1–50 wspólny 0,83;
  - klatka (RMS wokół średniej ±10 kroków): 0,62 (kroki 5–50, |δθ| 0,04) → 0,98 (150–300, 0,13) → 0,95 (300–700) → 0,79 (700–1090, przy stałym |δθ| 0,17) — starzenie;
  - norma gradientu = 1,0 w każdym kroku (obcięcie), więc długość kroku ≈ lr.
- Samotrafienie (zdjęcie w paczce kroku 534): skok rośnie 0,36 → 0,58 a0, ale sąsiednie kroki też 0,47–0,60 — niejednoznaczne, nie pokazane w filmie.
- naped_v2: w oknie K=10 0,36 vs przypadek 0,23 (epoki 1–5), 0,14–0,16 vs 0,09 (etapy 10–60), razem 0,04 vs 0,02; prognoza poza oknem R² ≈ 0 wszędzie.

### Issues Encountered
- Za słabe zielone wiązki (gain 0,03) → 0,08 i rampa jasności t²; gwiazda wiązek cząsteczki w zbliżeniu niewidoczna → wzmocnienie ×31 dla tej cząsteczki.
- Blending w LDR: przejście na premultiplied alpha (ONE, ONE_MINUS_SRC_ALPHA).
- Segoe UI nie ma glifu „∝” → słownie; kropki dziesiętne → przecinki (`pl()`).
- Generator wycinany z kadru / podpis na tytule → większa odległość kamery, podpis obok generatora.
- `find` po całym drzewie przekroczył 2 min — niepotrzebny (PACZKI i SONDY wystarczyły).

## Files Generated/Modified
- `SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/film_lasery_4_sfera.py` → `lasery_4_sfera.mp4` (79 s, 99 MB)
- `.../film_krysztal/drgania_prep.py`, `drgania.npz` (miniatury, |δθ|, lr, paczki)
- `.../film_krysztal/film_lasery_5_drgania.py` → `lasery_5_drgania.mp4` (84 s, 86 MB)
- `.../film_krysztal/film_lasery_4.log`, `film_lasery_5.log`
- `.../eden/wyniki/naped_v2/NAPED_WAGI.json`, `naped_wagi.png` (pobrane)
- `.../WERDYKT_KRYSZTAL_2026-10-11.md` (sekcja naped_v2)
- scratchpad: `drgania_X.npy` (1 GB, tymczasowy), `drgania_analiza.py`

## Commands Used
```bash
F:/Python/3.13/python.exe -X utf8 film_lasery_4_sfera.py [proba [gain]]
F:/Python/3.13/python.exe -X utf8 drgania_prep.py <katalog_na_X>
F:/Python/3.13/python.exe -X utf8 film_lasery_5_drgania.py <drgania_X.npy> [proba]
ssh eden-cluster 'cd .../wyniki/naped_v2 && tar cf - .' | tar xf - -C naped_v2
shutdown /s /t 7200
```

## Next Steps
- [ ] Test fluktuacja–odpowiedź (temperatura efektywna): pchnąć tokeny małą siłą i porównać odpowiedź ze spontanicznymi drganiami (Eden).
- [ ] Ewentualnie film 5 dla późnej fazy (gęsty zapis co krok jest tylko do kroku 1094; później co 50).
- [ ] Opcjonalnie: zmierzone pary odpowiedzi zamiast losowych błysków w filmie siatki optycznej.
