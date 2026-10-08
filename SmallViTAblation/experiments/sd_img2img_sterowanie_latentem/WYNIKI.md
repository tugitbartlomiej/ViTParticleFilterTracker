# img2img SD K8000 sterowane tylko sztucznym latentem (2026-10-08)

Jeden obraz: klatka TRAIN 0055/4019 (TR_0055_S1_P03), na której ViT K8000 nie wykrywa cystotomu. Sztuczny latent z*
z `outputs/latent_cel_do_obrazow/k8000_trening/zb_j10389_minmax.npy` (korekta detekcji w zakresach kanałów,
||δ|| = 2,9 ||z0||, odległość od naturalnych tokenów 19,8 wobec 3,1 dla zdjęcia). Metoda jak `sd_prowadzone.py`
użytkownika: zdjęcie celu -> VAE -> zaszumienie do siły s -> DDIM 50 leading; prowadzenie energią średnia (z − z*)^2
po pełnym latencie (2 kroki gradientu po x̂0 na krok, eta 0,15 σ_t, ostatnie 5 kroków bez prowadzenia), 3 rundy.
Różnica wobec dawnego prowadzenia: warunkiem SD jest teraz z*, nie z0. Bez straty detekcji.

Kod `kod/img2img_sterowanie.py`, wyniki `outputs/sd_img2img_sterowanie_latentem/` (WYNIK.json, SIATKA.jpg, `obrazy/`,
zdjęcie celu ZDJECIE_CELU_0055_4019_*.png/jpg).

| wariant | siła | ułamek δ | ułamek widocznej dla fc2 | cos do δ | bliżej z* | poza | PSNR do zdjęcia |
|---|---|---|---|---|---|---|---|
| kontrola: warunek z0 | 0,3 / 0,5 / 0,7 | 0,3 / 0,0 / 0,0 % | 1 / 0 / −1 % | ≈ 0 | −1 do −2 % | 2,7-2,9 | 26 / 24 / 22 dB |
| warunek z*, bez prowadzenia | 0,3 / 0,5 / 0,7 | 4,4 / 5,5 / 5,2 % | 0 / 1 / 2 % | 0,12-0,15 | −1 do −2 % | 2,5-2,6 | 25 / 22 / 20 dB |
| warunek z* + prowadzenie, runda 1 | 0,3 / 0,5 / 0,7 | 9,2 / 10,7 / 11,8 % | 6,6 / 9,9 / 10,8 % | 0,24-0,29 | 1,6-3,6 % | 2,8-3,3 | 25 / 22 / 20 dB |
| warunek z* + prowadzenie, najlepsza runda | 0,3 r2 / 0,7 r3 | 12,7 / 12,4 % | 11,9 / 12,0 % | 0,30 | 3,5 / 4,1 % | 3,3 | 22 / 16 dB |

- img2img zachowuje scenę celu; sam warunek z* dokłada ok. 5 % przyrostu, prawie nic z części widocznej dla fc2.
- Prowadzenie odległością od z* podwaja to do ok. 12 %, a część widoczna dla fc2 rośnie do 12 %, przy tokenach nadal
  naturalnych (poza 3,3 wobec 3,1 zdjęcia). Najlepszy kompromis: siła 0,3, druga runda (PSNR 22 dB).
- Dalsze rundy i większa siła nie dokładają kierunku, tylko oddalają obraz od zdjęcia.
- Wizualnie przyrost pojawia się jako cienkie niebiesko-metaliczne kreski na źrenicy (ten sam motyw co obraz zastępczy
  SD z szumu), więcej przy prowadzeniu.
- Odległość od z* maleje tylko o 3-4 %: większa część δ jest dla obrazów nieosiągalna. Wcześniej z szumu bez img2img
  i bez prowadzenia: ok. 2 % netto; prowadzenie k1000 przy sztucznym celu: 0,5 %.

Jeden obraz i jedno ziarno na rundę, więc to pokaz metody, nie pomiar istotności.

## Pętla generowania i szukania z losowego szumu, 50 generacji (2026-10-08)

Kod `kod/petla_szum.py`, wyniki `outputs/sd_img2img_sterowanie_latentem/petla_szum/` (wersja 1) i `petla_szum_elita/`
(wersja 2): obrazy `obrazy/g*.png`, latenty wszystkich obrazów i latentów sterujących 600 x 768 w `LATENTY.npz`,
wykresy WYKRES_PETLA.png, WYKRES_LATENT_PLASZCZYZNA.png, WYKRES_KANALY.png, pasek PASEK.jpg.

Każda generacja: SD K8000 od nowego szumu z warunkiem c (latent sterujący), latent obrazu z, poprawka
c <- Π(c + 0,5 (z* − z)); Π rzutuje c do zakresu warunków z uczenia SD (reszta tokenu poza PCA-64 prawdziwych tokenów
<= percentyl 99, kanały w [min, max] prawdziwych tokenów). Najlepszy obraz = najmniejsza ||z − z*||.

- Wersja 1 (poprawki sumowane od ostatniego obrazu, jedno przycięcie): rzut nie trzymał (latent sterujący doszedł do
  odległości 8,9 od naturalnych tokenów przy progu 4,4) i przestrzelił cel (113 % przyrostu); obrazy od generacji 5
  to tekstura. Najlepszy g16: 4,9 % przyrostu, 1,3 % części widocznej dla fc2.
- Wersja 2 (rzut powtarzany do skutku, krok od najlepszego dotąd, krok maleje bez poprawy): latent sterujący trzyma
  się progu i stoi na ok. 51 % przyrostu; obrazy wracają do sceny oka. Najlepszy g48: 6,5 % przyrostu, 5,6 % części
  widocznej dla fc2, odległość od z* mniejsza o 1,3 %. Średnio generacje 40-49: 5,4 % (pierwsze 10: 3,6 %).
- Pętla z szumu daje mniej niż img2img ze sterowaniem gradientem (12 %): rozrzut między szumami (składowa prostopadła
  0,25-0,40 ||δ||) jest większy niż postęp wzdłuż δ.
- Dlaczego: sztuczny przyrost jest rozłożony równo na wszystkie 600 tokenów (norma przyrostu na token 26-31 przy
  normie tokenu zdjęcia 10, 50 największych tokenów to tylko 10 % energii przyrostu), a narzędzie zajmuje kilka
  tokenów. Przyrost podnosi ok. 300 kanałów o ok. 2 na każdym tokenie, także na tle; żaden prawdziwy obraz tak nie
  wygląda (WYKRES_KANALY.png). Wniosek: sztuczny latent trzeba liczyć tylko na tokenach narzędzia (i granicy),
  dopiero wtedy SD ma szansę go zrealizować.

## Gradientowy dobór wejścia sterującego SD (2026-10-08)

Kod `kod/optymalizacja_warunku.py`, wyniki `outputs/sd_img2img_sterowanie_latentem/optymalizacja_warunku/`
(WYKRES_OPTYMALIZACJA.png, `obrazy/g*.png`, `latenty_grafika/` z wierszem na każdy obraz, LATENTY.npz, WYNIK.json,
NAJLEPSZY_WARUNEK_*.png). Odpowiednik sterowania FastGAN-em użytkownika: parametrem jest warunek SD c (600 x 768),
zawsze rzutowany do zakresu warunków SD; każda iteracja to pełna generacja od stałego szumu (DDIM 20, różniczkowalnie
z checkpointem UNet) -> VAE -> ViT K8000; strata = −postęp latentu obrazu wzdłuż δ + 0,1 odejście w bok; Adam lr 0,05.
Stop po 15 iteracjach bez poprawy albo po 150.

- 150 iteracji bez zatrzymania (poprawa do końca, coraz wolniej): postęp latentu obrazu wzdłuż δ 2,8 % -> 11,3 %,
  część widoczna dla fc2 0,8 % -> 7,2 %, odległość od z* mniejsza o 1,4 %; obraz naturalny (poza 3,1, zdjęcie 3,1).
  Warunek SD przesunął się wzdłuż δ tylko o 6 %, ale doszedł do granicy naturalnego zakresu (4,43).
- Najlepszy warunek z tym samym szumem w 50 krokach DDIM: 8,4 % (5,1 % widocznej). Z 4 innymi szumami: 2,3-3,6 %,
  czyli tyle co kontrola z warunkiem z0 na tych szumach (1,2-4,2 %, średnio 2,7 %). Wynik dotyczy więc pary
  warunek + szum: optymalizacja znalazła jeden obraz, nie warunek, który przenosi kierunek dla dowolnego szumu.
- Dla kuracji danych to wystarcza (potrzebny jest obraz-zapytanie, którego latent idzie w stronę z*), ale żeby warunek
  był ogólny, trzeba optymalizować średnio po kilku szumach albo optymalizować też szum.
- Na grafikach latentu (latenty_grafika) różnica z* − z prawie się nie zmienia: przyrost na wszystkich 600 tokenach
  (kanały 0-300 o ok. 2) pozostaje nieosiągalny.

## Najbliższe klatki TRAIN dla najlepszego obrazu (g148) i grafika podobieństwa cech (2026-10-08)

Kod `kod/szukanie_najlepszego.py`, `kod/grafika_podobienstwa.py`; wyniki `optymalizacja_warunku/SZUKANIE_TRAIN_g148.jpg/.json`,
`PODOBIENSTWO_CECH_g148.png`, film `FILM_OPTYMALIZACJA_30fps.mp4` (150 obrazów optymalizacji).

- Najbliższe obrazowi SD g148 (DINO cos 0,65-0,67): 0166/2161, 0078/511, 0010/2879, 0078/1043, 0078/1044; żadna z filmu
  celu. Najbliższe samemu zdjęciu celu: sąsiednie klatki 0055 (cos 0,92-0,96).
- Obraz SD jest do zdjęcia celu podobny tylko w 0,44 (DINO) i 0,61 (latent ViT): generacja od szumu nie zachowuje sceny.
- Postęp latentu znalezionych klatek wzdłuż δ: 0078/511 15,1 % (11,6 % widocznej dla fc2), więcej niż sam obraz SD
  (11,3 / 7,1 %); pozostałe 4-6 % przy prawie zerowej części widocznej. Sąsiedzi zdjęcia celu: 0,7-7,3 % i część
  widoczna od −2,8 do 1,4 %.
- Zastrzeżenie: to położenie latentu innego obrazu wobec osi z0 -> z*, a nie dowód, że dodanie tej klatki do uczenia
  przesunie latent celu; to mierzy pchnięcie (etap 12 latent_cel_do_obrazow).
