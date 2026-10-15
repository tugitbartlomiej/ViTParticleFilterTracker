# Session Summary: szklo1_2D — PDF o ViT, filmy 13–16 (GPU), atencja vs przesunięcie centrów

## Metadata
- **Date:** 2026-10-12
- **Time:** ~15:40–17:05
- **Status:** Completed
- **Type:** Mixed (dokumentacja / animacje / analiza)

## Objective
1. PDF „jak wygląda ViT od początku do końca” do nauki.
2. Kopia filmu `lasery_2_ruch.mp4`:
   - wolniejsza i płynniejsza;
   - zamiast opisów faz: mapa atencji, centra na zdjęciu oka z narzędziem, trzy składowe ruchu, IoU;
   - potem pewność, osobne wykresy, bez słupków zmian, mozaika 6 bloków;
   - render na GPU, film krótki.
3. Zbadać związek atencji (bloki osobno i łącznie) z przesunięciem centrów, także w czasie uczenia.

## Context
Kontynuacja badania „ruch tokenów ViT jak szkło” (`SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal`). Wcześniej powstały filmy 1–12; film 2 rozkładał ruch 600 tokenów TRAIN 0 na trzy składowe: globalną, lokalną i drgania. Dane źródłowe:
- `szescian.npz`;
- 806 checkpointów `assets/MalyViT_35000_od_zera_s47/wagi`.

## Actions Taken
1. **PDF ViT** (LaTeX, lualatex, TikZ):
   - zakres: łatki, osadzenie, CLS, pozycje, blok pre-norm, uwaga z przykładem liczbowym, wiele głów, MLP, głowica;
   - tabela kształtów, liczenie parametrów, rodzina modeli, koszt;
   - trening (przepis DeiT, rozgrzewka i kosinus, pretrening);
   - ViT w ok. 45 liniach PyTorcha, porównanie z MalyViT, ćwiczenia.
   - Liczby sprawdzone wykonaniem kodu: ViT-B 86 567 656, Ti 5 717 416, S 22 050 664.
2. **Atencja i ramka modelu dla 806 checkpointów** (`atencja_train0_prep.py`, GPU):
   - definicja jak w aplikacji: blok 6, średnia głów, bez przekątnej, uwaga otrzymana;
   - IoU zgodne z pomiarem Edenu: r = 0,9975; pewność: r = 0,99994.
3. **Film 13** (matplotlib, 3,5 min): przerwany przez użytkownika jako za długi; mp4 nie powstał.
4. **Film 14** (`film_lasery_14_ruch_gpu.py`) — nowy renderer GPU:
   - moderngl z MSAA 4×;
   - studnie jako gaussy w teksturze float, tonowane shaderem;
   - linie przez shader geometrii, mapy jako tekstura z LUT;
   - tło z matplotlib rysowane raz, kodowanie h264_nvenc;
   - oś czasu v = min(0,8 + krok/250, 150) kroków na klatkę, z rozmyciem ruchu;
   - wynik: 54 s, render ~25 s.
5. **Film 15**: pewność na wykresie 7 obok IoU.
6. **Film 16** (`film_lasery_16_iou_pewnosc_osobno.py`):
   - IoU i pewność na osobnych wykresach 7 i 8;
   - słupki zmian usunięte (aktualizacja tego samego filmu, na prośbę użytkownika);
   - panel 4 jako mozaika 3×2 atencji bloków 1–6 (`atencja6_train0_prep.py`).
7. **Analiza** `analiza_atencja_centra.py`: Spearman po patchach w każdym checkpoincie (także poza narzędziem i w 3D), udział atencji na narzędziu vs stosunek przesunięć (poziomy, różnice, lagi ±3), Spearman w czasie per patch (poziomy i różnice) z testem permutacyjnym; rollout liczony na GPU.

## Results

### Key Findings
- Korelacja przestrzenna atencja ~ przesunięcie centrów, epoki 5–60:

  | mapa | ρ |
  |---|---|
  | blok 1 | −0,06 |
  | blok 2 | −0,06 |
  | blok 3 | +0,03 |
  | blok 4 | +0,12 |
  | blok 5 | +0,17 |
  | blok 6 | +0,15 |
  | średnia bloków | +0,01 |
  | rollout | 0,00 |

  Najwyższa wartość to blok 6 w epokach 1–5: +0,32. Poza narzędziem bloki 4–6 dają tylko 0,04–0,08, więc związek idzie przez narzędzie.
- Udział atencji na narzędziu na końcu: blok 5 ok. 0,75, bloki 4 i 6 ok. 0,67, rollout tylko 0,17. Stosunek przesunięć narzędzie/reszta: ok. 2,3.
- W czasie:
  - różnice z checkpointu na checkpoint nie są skorelowane (mediana per patch od −0,013 do +0,014 przy progu permutacyjnym ok. 0,01–0,016), lagi też nie;
  - jedynie poziomy udziału bloku 6 korelują ze stosunkiem przesunięć (+0,53), ale to wspólny trend.
- Wniosek: atencja i odpływ centrów to dwa skutki tej samej treści (narzędzia). Nie ma sprzężenia z checkpointu na checkpoint.
- Zastrzeżenie: wyniki dla pojedynczych bloków zależą od rzutu (blok 1: −0,06 w 2D, −0,38 w 3D). Analiza dotyczy jednego zdjęcia.
- Mozaika atencji: bloki 1–2 patrzą szeroko (rozwieracz, tęczówka), bloki 4–6 skupiają się na narzędziu.

### Issues Encountered
- Skrypt Python w heredoc stracił `\\` w węzłach TikZ; poprawione ręcznie.
- Film 13 renderowany w matplotlib (6 procesów) był za długi i za wolny; zastąpiony rendererem GPU.
- Kolizja z inną sesją: obie użyły `film_lasery_16.log`, a druga zrobiła własny film 16 (`lasery_16_pulapka_siodlo.mp4`). Render puszczony z osobnym logiem. Numeracja 16 jest zdublowana, więc następny film to 17.
- Pierwsza próba commita w tej sesji (przed kompaktowaniem) została odrzucona; w tej części sesji nic nie commitowano.

## Files Generated/Modified
Pliki w `SmallViTAblation/docs/vit_od_poczatku_do_konca/`:
- `vit_od_poczatku_do_konca.tex`
- `vit_od_poczatku_do_konca.pdf`

Pliki w `SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/`:
- `atencja_train0_prep.py` → `atencja_train0.npz` (atencja bloku 6, ramka, pewność, IoU)
- `atencja6_train0_prep.py` → `atencja6_train0.npz` (806 × 6 × 600)
- `analiza_atencja_centra.py` → `atencja_rollout_train0.npz`, `ATENCJA_CENTRA.json`, `atencja_centra.png`
- `film_lasery_13_ruch_wolny_atencja.py` (mp4 nie powstał)
- `film_lasery_14_ruch_gpu.py` → `lasery_14_ruch_gpu.mp4`
- `film_lasery_15_ruch_gpu_pewnosc.py` → `lasery_15_ruch_gpu_pewnosc.mp4`
- `film_lasery_16_iou_pewnosc_osobno.py` → `lasery_16_iou_pewnosc_osobno.mp4` (wersja finalna z mozaiką)
- logi: `film_lasery_14.log`, `film_lasery_15.log`, `film_lasery_16_iou_pewnosc.log`

Pamięć:
- nowa notatka `filmy-render-gpu-krotkie.md` (filmy na GPU, ok. 1 min, wzorzec: film 14);
- dopisany wskaźnik w `MEMORY.md`.

## Commands Used
```bash
lualatex -interaction=nonstopmode vit_od_poczatku_do_konca.tex      # x2
F:/Python/3.13/python.exe -X utf8 atencja_train0_prep.py
F:/Python/3.13/python.exe -X utf8 atencja6_train0_prep.py
F:/Python/3.13/python.exe -X utf8 film_lasery_16_iou_pewnosc_osobno.py [proba]
F:/Python/3.13/python.exe -X utf8 analiza_atencja_centra.py
ffmpeg -ss 45 -i lasery_16_iou_pewnosc_osobno.mp4 -frames:v 1 ...   # kontrola klatki
```

## Next Steps
- [ ] Powtórzyć analizę atencja ~ przesunięcie centrów na setkach zdjęć (Eden), żeby rozstrzygnąć wynik pojedynczego zdjęcia.
- [ ] Opcjonalnie: pasmo rozrzutu zamiast słupków na wykresach IoU/pewności (nowy film 17).
- [ ] Commit własnych plików po nazwie, jeśli użytkownik poprosi.
