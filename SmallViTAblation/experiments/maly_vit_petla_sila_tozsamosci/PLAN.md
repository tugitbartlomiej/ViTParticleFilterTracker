# Mały ViT, wersja 3: pętla sterowana siłą tożsamości narzędzia (plan do zgody)

Data: 2026-09-28. Zgoda użytkownika na plan z nazwanymi wyborami i ramieniem `current`: „Na edenie trenuj”.
Status: zgłoszone na Edenie — próba kodu 1818024 (ziarno 47, `proba`), kampania 1818025 (ziarna 47–56, `afterok`).

## Skąd ta wersja

- Prognoza pełnego latentu (768 liczb) nie bije „braku zmiany” (wersje 1 i 2).
- Prognoza **siły wektora tożsamości** wyznaczonego po 21 epokach bije „brak zmiany” o 6–10 % na 10 ziarnach
  (`../maly_vit_wektor_tozsamosci/koopman_na_wektorze.py`); na ziarnie 42 prognoza z kroku 1000 sprawdzona na kroku
  1500: Langevin zmniejsza błąd o 54 % na 12 kotwicach, znak 12/12
  (`../fastgan_odwrotnosc_mlp/sprawdz_prognoze_1500.py`).
- FastGAN sterowany tylko wzdłuż wektora tożsamości osiąga przewidzianą zmianę siły w całości, znak 12/12, 85 %
  przetrwa zaburzenia obrazu (`../fastgan_odwrotnosc_mlp/wzdluz_tozsamosci.py`, wariant Z_1x).
- Decyzja użytkownika (2026-09-28): Koopman i Langevin przewidują **tylko tożsamość**, nie cały latent.

## Pętla (schemat użytkownika)

1. ViT od zera na bazie 1500 klatek przez 2 obroty po 500 kroków = 1000 kroków = 21 epok (1 epoka = 47 kroków).
2. Tożsamość ze zdjęć sond po 21 epokach: wektor oddzielający narzędzie od tła (LDA ze ściąganiem kowariancji na
   latentach 180 sond narzędzia i ich tła, jak `wektor_tozsamosci_21ep.py`). Siła tożsamości sondy = rzut jej latentu
   narzędzia na ten wektor.
3. Na początku każdego obrotu Koopman i Langevin prognozują siłę tożsamości 180 sond jeden obrót naprzód (500 kroków
   ≈ 10,6 epoki), z historii siły od kroku 50 (bloki po 50 kroków). Model wybierany w każdym obrocie testem wstecznym
   na tej historii: ten, który ostatnio przewidywał lepiej.
4. FastGAN: dla 6 kotwic (oryginalne wycinki narzędzia) o największej przewidzianej zmianie siły — inwersja kotwicy,
   potem sterowanie wejściem z (300 kroków gradientu), aż siła tożsamości obrazu = siła startu + przewidziana zmiana;
   kara za zmianę reszty latentu.
5. Obraz FastGAN-a → wyszukanie podobnych prawdziwych klatek w puli (DINOv2, top-90, reguła ±30 klatek od bazy
   i dodanych, jak w wersji 2) → 134 klatki na obrót, na przemian z list 6 kotwic.
6. 500 kroków uczenia na powiększonym zbiorze, powrót do 3. Razem 7 obrotów w ramionach (938 klatek dodanych).

## Ramiona (ten sam stan po rozgrzewce, ta sama liczba klatek)

- `tozsamosc` — metoda (punkty 3–5).
- `losowe` — 134 losowe klatki z puli na obrót, reguła ±30.
- `kolejne` — **nowe, z listy użytkownika**: kolejne klatki tych samych filmów. W każdym obrocie 134 klatki puli
  najbliższe w czasie **po** klatkach bazy (i po już dodanych kolejnych), w tym samym filmie. Bez reguły ±30, bo
  sensem ramienia są klatki sąsiednie.
- `bez` — bez dodawania.
- `current` — **moja propozycja kontroli, spoza listy użytkownika**: to samo co `tozsamosc`, ale FastGAN sterowany do
  OBECNEJ siły (zmiana 0). Różnica `tozsamosc` − `current` pokazuje, ile daje sama prognoza, a ile „klatki podobne
  do kotwic”.

## Wybory, które nie wynikają wprost ze słów użytkownika (do potwierdzenia)

1. Wektor tożsamości wyznaczony raz, po 21 epokach, i stały do końca (tak był sprawdzany); nie wyznaczamy go od
   nowa w każdym obrocie.
2. Horyzont i długość obrotu: 500 kroków ≈ 10,6 epoki, jak w wersji 2 („kilka epok” — tu około dziesięciu).
3. FastGAN sterowany samym wejściem z (najczystszy obraz); warstwy głębsze dawały artefakty.
4. Rozgrzewka wzięta z wersji 2 (te same ziarna 47–56, ten sam kod, krok 1000) — kopia twardymi dowiązaniami, bez
   ponownego uczenia; pary z wersjami 1 i 2.
5. Na Edenie nie ma sklearn: LDA ze ściąganiem Ledoit-Wolf napisane w numpy, sprawdzone lokalnie na zgodność
   z wektorem sklearn (cosinus ≥ 0,999) przed zgłoszeniem.

## Pomiar w każdym obrocie i ramieniu

- prognoza siły z poprzedniego obrotu kontra to, co się stało: błąd / brak zmiany i zgodność znaku, dla 180 sond
  i dla kotwic, osobno Koopman i Langevin;
- FastGAN (`tozsamosc`, `current`): osiągnięta część zmiany siły, na czysto i na zaburzonych kopiach;
- wspólna część list wyszukiwania dla obrazu z prognozą i obrazu bez zmiany (top-90, top-22);
- mAP@0,5 co 100 kroków na walidacji (jak w wersjach 1–2).

## Kryteria (ustalone przed wynikiem)

- Główne: pole pod krzywą mAP@0,5 w obrotach 3–9: `tozsamosc` > `losowe`, > `kolejne` i > `bez` w ≥ 8 z 10 ziaren.
- Rola prognozy: `tozsamosc` > `current` w ≥ 8 z 10 ziaren; inaczej zysk nie pochodzi z prognozy.
- Prognoza w pętli działa, jeśli błąd siły / brak zmiany < 1 w ≥ 8 z 10 ziaren w ≥ 4 z 6 obrotów 4–9.
- FastGAN działa w pętli, jeśli mediana odpornej części zmiany ≥ 0,5 i znak ≥ 0,8 w każdym obrocie.

## Eden

`~/smallvit_runs/2026-09-28_maly_vit_sila/` (`kod/`, `logi/`, `wyniki/`); wejścia i `timm` z katalogu wersji 1,
rozgrzewki z wersji 2. `hopper,short`, 1 GPU na ziarno, limit 3 h, sprzątanie `/dev/shm` pułapką.
Najpierw próba kodu (krótkie obroty, 1 zadanie), potem kampania 10 ziaren z zależnością `afterok`.
Szacunek: ~1 h na ziarno (5 ramion, FastGAN w dwóch), razem ~10 GPU-godzin, ~1 h zegara.

## Sprawdzenia przed zgłoszeniem (2026-09-28)

- `test_lda_lokalnie.py` → `wyniki/TEST_LDA.json`: wektor z numpy (`kod/sila.py`) zgodny z sklearn, cosinus 1,000
  w kroku 1000 i w kroku 0; różnica siły < 3·10⁻⁶. Siła narzędzia 6,20 wobec negatywów 0,00 (sd 0,22).
- Próba lokalna (RTX 3070, ziarno 42, rozgrzewka z wersji 1 = te same kroki 0–1000, `--proba`: 2 obroty po 50
  kroków, 10 klatek, 2 kotwice): wszystkie 5 ramion przechodzi, 170 s. FastGAN w `tozsamosc`: część zmiany 1,00,
  odporna 0,96–1,00; wspólne top-90 z obrazem startowym 22–50 z 90, czyli prognoza zmienia wyszukane klatki.
  `kolejne` bierze klatki puli 5 klatek po klatkach bazy. Liczby prognozy w próbie nic nie znaczą (obrót 50 kroków).
- Jedyna zmiana kodu wspólnego: `kod/generuj.py` czyta ścieżkę wag FastGAN ze zmiennej `MALY_VIT_FASTGAN_CKPT`,
  gdy jest ustawiona (tylko próba lokalna; na Edenie zmienna nie jest ustawiana).
