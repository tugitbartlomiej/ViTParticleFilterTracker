# Podsumowanie sesji: pętla siły tożsamości na małym ViT i kierunki tożsamości w przestrzeni cząstek

Data: 2026-09-28 (ok. 15:30) - 2026-09-29 (14:52).
Repozytorium: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, commity od `235dcd80` do `8d50ae84`.
Rozmowa prowadzona z katalogu projektu grantowego `F:\Studia\MCHTR\Grant\Projekt` (gałąź
`feature/koopman-tft-experiments`), ale cała praca merytoryczna dotyczy repozytorium ViT.

## Cel sesji

1. Zbudować mały ViT uczony od zera i sprawdzić na nim pętlę użytkownika: tożsamość narzędzia w latencie,
   prognoza jej zmiany Koopmanem i Langevinem, obraz z FastGAN-a przesunięty o tę prognozę, wyszukanie podobnych
   klatek z puli (DINOv2), dołożenie ich do uczenia. Porównanie z losowymi klatkami, kolejnymi klatkami i brakiem
   dokładania.
2. Rozstrzygnąć, czy ewentualny zysk pochodzi z prognozy, czy z samego wyszukiwania podobnych klatek.
3. Na prośbę użytkownika wyznaczyć tożsamość narzędzia w latencie na parach „z narzędziem i bez” (czarne tło,
   samo oko) i policzyć, ile kierunków wielowymiarowej przestrzeni cząstek ją koduje.

## Kontekst

- Poprzednia sesja: `Session_2026-09-28_011148_Petla_Large_Koopman_i_Domykanie_Tozsamosci` (pętla na RF-DETR
  Large nie przyspieszała uczenia, latent tożsamości niedomknięty).
- Model cząstkowy użytkownika: każdy token to cząstka o 768 współrzędnych latentu; tożsamość narzędzia ma być
  kierunkiem (lub kilkoma) w tej przestrzeni.
- Stałe reguły z tej sesji: plan przed kampanią GPU i czekanie na „tak”; nie podmieniać metody użytkownika bez
  nazwania każdej różnicy; foldery równoległych sesji tylko do odczytu; na Edenie sprzątanie przez trap, `sed` na
  końce linii, sprawdzenie wejść i kwoty `lfs quota -p 1299` przed `sbatch`.
- Eden: kampanie 1817935 (wersja 1), 1817949 (wersja 2), 1818025 (wersja 3), 1818051 (wersja 3, iteracja 2).

## Wykonane kroki

1. Mały ViT od zera (`SmallViTAblation\experiments\maly_vit_od_zera`): 2,93 mln parametrów, łodyga
   konwolucyjna, gęsta głowa, 1500 klatek bazy. Kalibracja: mAP@0,5 0,36 po 1500 krokach, 0,40-0,42 po 6000.
   Pętla tożsamość -> Koopman/Langevin -> FastGAN -> klatki (`235dcd80`).
2. Lokalnie ziarno 42: koopman 0,442 > losowe 0,422 > bez 0,392; kontrola target kontra current (`d661a8ee`).
3. Eden: analiza przed sprzątaniem (192,4 z 200 GB, nic nie usunięto, `8df08a91`), potem sprzątanie po zgodzie
   użytkownika: 193,2 -> 52,2 GB, archiwum raportów lokalnie (`91e97450`,
   `SmallViTAblation\docs\EDEN_SPRZATANIE_2026-09-28.md`).
4. Wersja Eden małego ViT (ścieżki z `MALY_VIT_*`, wyszukiwanie w numpy zgodne z FAISS). Próby 1817930 i 1817932
   padły (brak timm, ścieżki Windows na Linuksie), poprawione (`7de1c7c7`, `c29df166`).
5. Wyniki wersji 1 (Eden 1817935, ziarna 47-56 plus 42): koopman > losowe w 9 z 11 ziaren (+0,013 mAP@0,5,
   Wilcoxon p = 0,005), ale prognoza latentu nie biła braku zmiany, a FastGAN nie trafiał w cel (`3a178c23`).
   Analiza wsteczna: lista neuronów tożsamości wymienia się co ~5 epok, wektor na 768 neuronach stabilny od
   ~16 epok (cosinus 0,85-0,90) (`e5897e27`). Plan użytkownika na następne sprawdzenie (`f7efafbf`).
6. Wersja 2 (`maly_vit_wektor_tozsamosci`, Eden 1817949): wektor tożsamości na 768 neuronach, start po 21
   epokach. Koopman > losowe i > bez w 8 z 10 ziaren (+0,009, p = 0,064), ale prognoza wektora = brak zmiany
   (0 z 6 obrotów) (`ddfeae69`, `1ded306b`). Diagnozy: tożsamość metodą użytkownika po 21 epokach (rdzeń 0),
   wektor tożsamości (sieć nieuczona przechodzi ten sam test), Koopman i Langevin na stałym wektorze z 21 epok:
   błąd 0,90-0,94 x brak zmiany (`6eb7fff0`, `b7a3b308`, `8deaf528`).
7. FastGAN (`fastgan_odwrotnosc_mlp`): diagnoza chybiania, sterowanie głębszymi warstwami, sieć MLP bez zysku,
   przegląd literatury (`5f498d37`, `cf97cf3a`). Sterowanie wzdłuż wektora tożsamości o zmianę siły z Langevina:
   znak 12 z 12, 85-111 % zmiany przetrwa zaburzenia obrazu, zysk prognozy 54 % na kotwicach w kroku 1500
   (`6f6bbf06`).
8. Wersja 3 (`maly_vit_petla_sila_tozsamosci`): prognoza siły tożsamości, FastGAN wzdłuż wektora, ramiona
   tozsamosc, current, losowe, kolejne, bez; wektor LDA z Ledoit-Wolf w numpy (`d289a8ad`). Kryteria ustalone
   przed wynikiem (`831bf563`). Eden 1818025, 10 ziaren: prognoza bije brak zmiany, FastGAN realizuje zmianę,
   ale ramię z prognozą = ramię current (0,4432 wobec 0,4428, 5 z 10); przewaga nad losowaniem +0,01 z podobieństwa
   do kotwic (`d66e6eb8`). Zliczenie wygranych i sondy syntetyczne (niekonkluzywne, sondy poza rozkładem
   detektora) (`8ff21d8e`).
9. Iteracja 2 wersji 3 po zgodzie użytkownika: 14 obrotów, lista wyszukiwania 180, 20 ziaren, Eden 1818050
   (próba) i 1818051 (`e3836c66`, `c9c8ee4a`). Wynik (`97186aa0`): metoda lepsza od losowych klatek w 15 z 20
   ziaren przez cały przebieg (p = 0,006) i w 17 z 20 w kroku 4500, na końcu 11 z 20; wobec ramienia current
   13 z 20 (pole) i 8 z 20 (koniec). Udziału samej prognozy nie wykazano.
10. Tożsamość na parach (`maly_vit_tozsamosc_tla`, `fdf82918`): `tla.py` tworzy dla 180 sond z narzędziem
    zdjęcie z zamalowanym narzędziem (OpenCV Telea) i czarne; `wektor_z_tlami.py` liczy trzy wektory. Stary wektor
    odróżnia narzędzie od tego samego miejsca bez narzędzia tylko na d' 2,0; nowy na d' 16,6 (AUC na wycinku
    wyłączonym 0,997); cosinus stary-nowy 0,28. Prognoza siły nowego wektora nie lepsza od starego.
11. Zapis wektorów (`zapisz_wektory.py`, `887e48fb`): `WEKTORY.npz`; połowę wagi nowego wektora niesie ~78 z 768
    neuronów, 90 % ~312.
12. Obraz cząstkowy (`kierunki_tozsamosci.py`, `238851f7`): 1568 cząstek narzędzia na sieć kontra te same tokeny
    bez narzędzia. Kolejne kierunki rozdzielające (LDA z usuwaniem kierunku, AUC na wycinkach spoza wyznaczania):
    1-2 kierunki z AUC >= 0,9 (pierwszy 0,94-0,96), 5-8 z AUC >= 0,75, po ~15-20 AUC 0,6. Energia różnic: połowa
    w 2-3 kierunkach, 90 % w 29-48. Pierwsze uruchomienie w tle Bash zostało przerwane (bufor `grep`, limit 10
    minut), powtórzone jako proces odłączony.
13. Zapis kierunków cząstkowych (`zapisz_kierunki_czastek.py`, `8d50ae84`): 6 kierunków na sieć w
    `KIERUNKI_CZASTEK.npz`; pierwszy kierunek cząstkowy ma cosinus 0,40 z wektorem `nowy` (kąt ~66°), a oba
    rozdzielają cząstki (0,95 na wycinkach wyłączonych, 0,96). Tożsamość to wąska wiązka kierunków.

## Pliki zmienione / utworzone

### W projekcie (Grant/Projekt)

Brak zmian merytorycznych w `F:\Studia\MCHTR\Grant\Projekt`.

### Zewnętrzne (poza projektem)

Repozytorium ViT, `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\`:

- `experiments\maly_vit_od_zera\` - kod pętli, wersja Eden, wyniki 11 ziaren, analiza wsteczna, plan użytkownika.
- `experiments\maly_vit_wektor_tozsamosci\` - wersja 2, wyniki Eden 1817949, diagnozy wektora po 21 epokach.
- `experiments\fastgan_odwrotnosc_mlp\` - diagnoza FastGAN-a, `wzdluz_tozsamosci.py`, `sprawdz_prognoze_1500.py`,
  `odniesienie_bok.py`, PLAN.md „Krok 4”.
- `experiments\maly_vit_petla_sila_tozsamosci\` - `kod\sila.py`, `kod\petla.py`, `kod\wybor.py`,
  `kod\generuj.py`, `petla_eden.slurm`, `petla_eden_it2.slurm`, `analiza.py`, `diag_sila_ramion.py`,
  `ocena_sond.py`, `test_lda_lokalnie.py`, `ALGORYTM.md`, `PLAN.md`, `wyniki_eden\`, `wyniki_eden_it2\`.
- `experiments\maly_vit_tozsamosc_tla\` - `tla.py`, `wektor_z_tlami.py`, `zapisz_wektory.py`,
  `kierunki_tozsamosci.py`, `zapisz_kierunki_czastek.py`, `PLAN.md`, `tla\manifest.json`,
  `wyniki\WEKTOR_Z_TLAMI.json`, `wyniki\WEKTORY_SKUPIENIE.json`, `wyniki\KIERUNKI_TOZSAMOSCI.json`,
  `wyniki\KIERUNKI_CZASTEK.json`. Tylko lokalnie (poza gitem): `tla\images\`, `wyniki\WEKTORY.npz`,
  `wyniki\KIERUNKI_CZASTEK.npz`, logi.
- `docs\EDEN_SPRZATANIE_2026-09-28.md` - analiza i przebieg sprzątania Edenu.

Archiwum raportów z Edenu: `F:\Studia\PhD_projekt\VIT\eden_archiwum`.

Pamięć Claude (`C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\`):
`koopman-przewiduje-tylko-tozsamosc.md`, `feedback-nie-podmieniac-metody-uzytkownika.md`,
`eden-po-sprzataniu-2026-09-28.md`.

Tylko do odczytu, nieruszane: foldery równoległych sesji (`koopman_fastgan_petla`, `koopman_gesta_historia`,
`koopman_osobno_obraz_token`, `dobor_klatek_stan_sieci`, `dobor_cech_adaptacyjny`, `koopman_tozsamosc_kierunki`,
`fastgan_narzedzie`), `external\` i `F:\Studia\Articles\Zbiory\JEPA`. Commit `fb899bde` (plan dekodera latentu)
pochodzi z równoległej sesji.

## Otwarte problemy

- Zysk pętli pochodzi z wyszukiwania klatek podobnych do narzędzia, a nie z prognozy Koopmana i Langevina:
  ramię current (ten sam obraz FastGAN-a bez przesunięcia) jest prawie tak samo dobre.
- Przewaga nad losowymi klatkami jest w pierwszej części uczenia i znika pod koniec (11 z 20 ziaren).
- Nowy wektor tożsamości (z par z narzędziem i bez) izoluje narzędzie, ale jego siła nie jest lepiej przewidywalna
  od siły starego wektora; nie sprawdzono go w pętli.
- Tożsamość to wiązka kierunków: nie ma jednej kanonicznej strzałki, wybór zależy od sposobu liczenia.
  Wektory różnych sieci nie są porównywalne.
- Sondy syntetyczne (czarne i żółte tło, obroty) są poza rozkładem detektora, więc nie rozstrzygają pewności detekcji.

## Następne kroki

- Ewentualna kampania pętli z nowym albo łącznym wektorem tożsamości (lub z pierwszymi kierunkami cząstkowymi)
  wymaga osobnego planu i zgody użytkownika.
- Rozważyć sterowanie FastGAN-em w podprzestrzeni kilku kierunków tożsamości zamiast jednej strzałki.
- Sprawdzić, jak wiązka kierunków tożsamości obraca się w trakcie uczenia (wektor jest dziś stały od kroku 1000).
