# Podsumowanie sesji 2026-10-08/09

## Cel sesji

Realizacja sztucznego latentu z* (korekta latentu ViT, przy której detekcja na zdjęciu 0055/4019 się udaje) jako
naturalnego obrazu SD, który służy wyłącznie jako zapytanie do wyszukania prawdziwych klatek TRAIN w kuracji danych.
W drugiej części: sprawdzenie treningiem ViT, czy podmiana klatek w bazie przesuwa latent celu w stronę z*.

## Kontekst

- Repozytorium: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, katalog roboczy `SmallViTAblation`,
  branch codex/eden-nano-training-20260914. Komendy uruchamiane z projektu grantowego `F:\Studia\MCHTR\Grant\Projekt`.
- Modele: ViT K8000 (`assets/MalyViT_K8000_s47_losowe`, sha 0844a20a), SD K8000 (`assets/SD_ViT_K8000_A_control_K10500`),
  ViT K1000 (`assets/MalyViT_K1000_SD_conditioning`, sha 3df42a6a), stare SD uczone na K1000 (`assets/StableDiffusionModel`).
- Poprzednie etapy tej linii: commity 6ddf11e0, d05c5ad8, f153c754, 5d9daaa9, 8607e0dc (img2img, pętla 50 generacji,
  gradient po warunku, szukanie TRAIN, grafiki cech DINO).
- Miara postępu: ułamek = <z - z0, δ> / ||δ||², δ = z* - z0; część widoczna = rzut przez fc2 (768 do 192).

## Wykonane kroki

1. Zatrzymano biegi gradientowe (mieszanka szumów, szum wprost); bazę losowego szukania odtworzono z logu do
   `LOSOWE_SZUKANIE.json`. Uruchomiono regresję warunków początkowych (`optymalizacja_regresja.py`): 16 par
   antytetycznych, regresja grzbietowa, krok elitarny, DDIM 20. Wynik po 20 krokach i 661 generacjach (19 min):
   postęp 10,2 do 17,2 procent, część widoczna 4,5 do 12,6 procent, ostrość 390 do 373. Ten sam szum w 50 krokach DDIM:
   6,0 procent (szum dopasowuje się do liczby kroków). TRAIN top-1 dla obrazu g19: 0088/1661, postęp 13,2 procent
   (widoczny 7,0); trzy z pięciu trafień to sąsiednie klatki 1659-1661.
2. Odpowiedzi teoretyczne: ostrość obrazu z mieszanki szumów (mieszanka jest dokładnie N(0, I), więc SD rysuje normalnie),
   FastGAN wobec SD (FastGAN szybszy, ale bez warunku latentem; wcześniej 4 procent drogi na samym wejściu z), dekodery
   z dużym wejściem (RAE z NYU, dekoder ViT z tokenów DINOv2 768 x 16 x 16, straty L1, LPIPS, GAN; FeatInv; IC-GAN),
   rzut na naturalne obrazy (SDEdit, DiffPure) i granica: z* poza zasięgiem naturalnych obrazów.
3. `porownanie_metod.py` dostosowano do odtworzonej bazy i logów przerwanych biegów. Stan: losowe 10,1 procent,
   gradient mieszanki (przerwany po 8 iteracjach) 16,2 procent, regresja 17,2 procent.
4. Wersja z 5 krokami DDIM i partiami na GPU (`optymalizacja_regresja_k5.py`, agent): krok regresji 10 s zamiast 55 s;
   przerwana przez użytkownika po 9 krokach (14,7 procent).
5. `podobne_do_klatki.py`: 150 klatek TRAIN najbliższych w DINO do 0088/1661, oryginały 720 x 480 z archiwów zip,
   arkusz, CSV, JSON, wykres. 143 z 150 z filmu 0088; mediana postępu 7,3 procent, 18 klatek ponad 10 procent.
6. Eden (agent, zadanie 1824868, H100, 3 min): kontynuacja od K1000 ze stanem AdamW, 5 epok po 47 kroków,
   C0 (oryginalne 1500), A150 (150 losowych pozycji bazy zastąpione klatkami z punktu 5), C0b (powtórzenie, identyczne).
   Norma różnicy latentu celu A150 - C0 od 71 do 135, kosinus z kierunkiem do z* K1000 od -0,06 do 0,11; postęp
   celu na końcu 0,5 procent wobec -0,1; IoU celu 0 w obu ramionach; mAP@0,5 0,308 do 0,369 (C0) i 0,340 (A150), wahania
   międzyepokowe około 0,03. z* K1000 liczony lokalnie etapem 8: IoU 0,70 przy pewności 0,06.
7. Lokalnie `optymalizacja_regresja_starty.py`: 10 czystych losowych startów (ziarna 5001-5097), po 5 kroków regresji,
   η0 1,0 i σ 0,05 (większe kroki), DDIM 20, partie na GPU. 10,5 min, 1660 generacji. Średnio 4,3 do 11,7 procent;
   najlepszy start 6 (ziarno 5057): 4,2 do 16,9 procent, widoczny 10,3; w 50 krokach DDIM 13,9 procent (przenosi się
   znacznie lepiej niż bieg z punktu 1). Start 5: postęp 8,3 procent przy części widocznej 0. TRAIN dla najlepszego
   obrazu słabszy: cos DINO 0,73-0,75, najlepsza klatka 0081/19 z postępem 10,6 procent.
8. Eden (agent, zadanie 1824901, w toku): 9 ramion C0, A150, R150 (150 losowych klatek TRAIN) dla ziaren 47, 48, 49,
   wszystkie liczone od nowa na jednej karcie (pomiary A100 i H100 różnią się od startu w czwartej cyfrze).
9. Przygotowanie 1500 startów regresji w K1000 (agent): zatrzymany na warunku wstępnym. z* K1000 w naturalnym pudle
   daje pewność 0,055 przy IoU 0,70 także po 3000 krokach; bez pudła pewność 0,93, ale zmiana 11 norm i odległość od
   tokenów naturalnych 92. Pomiar czasu starego SD: 0,88 s na obraz (RTX 3070), 1500 startów około 62 h lokalnie.
10. `pewnosc_baza.py`: naturalna detekcja na 1500 obrazach bazy K1000. K1000: mediana pewności 0,77, 96 procent
    ponad 0,5, mediana IoU 0,67, 74 procent trafień. K8000: 0,93, 100 procent, 0,84, 85 procent. Zdjęcie celu:
    K1000 pewność 0,51 przy IoU 0, K8000 0,25 przy IoU 0. Wniosek: K1000 umie być pewny, problem leży w procedurze
    z* dla tego zdjęcia, nie w zasięgu latentu.

## Pliki zmienione / utworzone

### W repozytorium ViT (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation`)

Kod (niezacommitowany):
- `experiments\sd_img2img_sterowanie_latentem\kod\optymalizacja_regresja.py` (poprawka bazy i ostrości)
- `experiments\sd_img2img_sterowanie_latentem\kod\optymalizacja_regresja_k5.py` (agent)
- `experiments\sd_img2img_sterowanie_latentem\kod\optymalizacja_regresja_starty.py`
- `experiments\sd_img2img_sterowanie_latentem\kod\podobne_do_klatki.py`
- `experiments\sd_img2img_sterowanie_latentem\kod\porownanie_metod.py`
- `experiments\sd_img2img_sterowanie_latentem\kod\optymalizacja_mieszanka.py`, `optymalizacja_szumu.py`, `cechy_dino.py`,
  poprawka `grafika_podobienstwa.py` (z poprzedniej części, nadal niezacommitowane)
- `experiments\vit_k1000_podmiana150_sd\` (agent: PLAN.md, WYNIKI.md, WYNIKI.json, kod\bieg.py, dane.py, model.py,
  trening.py, przygotuj.py, przygotuj_r150.py, analiza.py, analiza_ziarna.py, z_cel_k1000.py, slurm)
- `experiments\vit_k1000_sd_regresja_1500\` (agent: PLAN.md, WYNIKI.md, WYNIKI.json, kod\z_cel_k1000_v2.py;
  moje: kod\pewnosc_baza.py)

Wyniki (`outputs\`):
- `sd_img2img_sterowanie_latentem\optymalizacja_mieszanka\LOSOWE_SZUKANIE.json`
- `sd_img2img_sterowanie_latentem\optymalizacja_regresja\` (WYNIK.json, obrazy\g00-g20.png, FILM_OPTYMALIZACJA_30fps.mp4,
  NAJLEPSZY_kroki20.png, NAJLEPSZY_kroki50.png, SZUKANIE_TRAIN_najlepszy.jpg, LATENTY.npz, TRAIN_top1_0088_1661.png,
  PODOBNE_150_0088_1661\ ze 150 klatkami, ARKUSZ.jpg, PODOBNE.csv, PODOBNE.json, WYKRES.png)
- `sd_img2img_sterowanie_latentem\optymalizacja_regresja_k5\` (częściowy log, bez WYNIK.json)
- `sd_img2img_sterowanie_latentem\optymalizacja_regresja_starty\` (WYNIK.json, obrazy\, WYKRES_STARTY.png,
  STARTY_PRZED_PO.jpg, NAJLEPSZY_kroki20.png, NAJLEPSZY_kroki50.png, SZUKANIE_TRAIN_najlepszy_start6.jpg, WAGI_NAJLEPSZE.npy)
- `sd_img2img_sterowanie_latentem\POROWNANIE_METOD.png`, `POROWNANIE_METOD.json`
- `vit_k1000_podmiana150_sd\` (ANALIZA.json, WYKRES.png, eden\pelny\{C0,A150,C0b}\, z_cel_k1000\zb_j10389_minmax_k1000.npy)
- `vit_k1000_sd_regresja_1500\z_cel\` (z* K1000 v2, warianty minmax i wagi, JSON), `pewnosc_baza\PEWNOSC_BAZA.json`, `PEWNOSC_BAZA.png`

Logi: `experiments\sd_img2img_sterowanie_latentem\kod\optymalizacja_regresja.log`, `optymalizacja_regresja_k5.log`,
`optymalizacja_regresja_starty.log`.

### Eden

- `~/smallvit_runs/2026-10-08_k1000_podmiana150_sd/` (zadania 1824863, 1824868, 1824901; pełne stany wag i AdamW
  po każdej epoce).

### Zewnętrzne (poza repozytorium ViT)

- Brak zmian w projekcie grantowym, wiki, Math i artykułach. Skrypt pomiaru czasu starego SD w scratchpadzie sesji.

## Otwarte problemy

- z* dla zdjęcia 0055/4019 w K1000: procedura etapu 8 utyka (pewność 0,055 przy IoU 0,70), choć K1000 na swojej bazie
  ma medianę pewności 0,77. Do wyboru: inna strata (większa waga pewności, dwa etapy), start z naturalnego latentu
  trafionej klatki, z* tylko na tokenach narzędzia.
- 1500 startów regresji w K1000 na Edenie czeka na decyzję o celu.
- Wynik zadania 1824901 (C0, A150, R150 x 3 ziarna) w toku przy zapisie.
- Łańcuch doboru 150 klatek był w przestrzeni K8000, trening od K1000; 143 ze 150 klatek z jednego filmu.
- Na Edenie po sprzątaniu: `od_zera/deps/timm` puste, w `od_zera/wejscia` brak `pretrain/valid/_annotations.coco.json`
  i `probe/manifest.json`; może psuć inne skrypty.
- Usunięte przez równoległą sesję foldery k1000 `outputs/latent_cel_do_obrazow/sd_interpolacja` i `sd_szukanie`
  (są w gicie), bez decyzji o przywróceniu.
- Kod i wyniki z tej sesji niezacommitowane; WYNIKI.md eksperymentu sd_img2img_sterowanie_latentem bez sekcji regresji.

## Następne kroki

1. Raport z zadania 1824901: czy A150 przesuwa latent celu w stronę z* bardziej niż R150 (średnia i rozrzut z 3 ziaren).
2. Naprawić z* K1000 dla zdjęcia celu (wariant straty), potem 1500 startów regresji na Edenie i pula klatek z wielu filmów.
3. Dopisać wyniki regresji (20 kroków, 5 kroków, 10 startów) do `experiments\sd_img2img_sterowanie_latentem\WYNIKI.md`
   i zacommitować kod z tej sesji.
