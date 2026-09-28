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

## Wyniki kampanii (2026-09-28, Eden 1818025, ziarna 47–56, wszystkie COMPLETED, 15–25 min na ziarno)

Pliki: `wyniki_eden/` (JSON-y obrotów i ramion, logi; latenty, wagi, obrazy FastGAN i listy wyszukiwania
`zapytanie_*.json` tylko na Edenie albo lokalnie poza gitem), `wyniki_eden/ANALIZA.json` (`analiza.py`),
`wyniki_eden/DIAG_SILA.json` (`diag_sila_ramion.py`, diagnoza po wyniku, nie kryterium).

Pole pod krzywą mAP@0,5 w obrotach 3–9 (średnia 10 ziaren): `tozsamosc` 0,4432, `current` 0,4428, `losowe` 0,4327,
`kolejne` 0,4255, `bez` 0,4030. Klatek dodanych średnio: `tozsamosc` 930, `current` 929 (w ostatnich obrotach
wyczerpują się listy podobnych klatek, jak w wersji 2), `losowe` i `kolejne` 938.

- **Kryterium główne: niespełnione (7 z 10, wymagane 8).** `tozsamosc` − `losowe` +0,0105, lepsza w 7 z 10
  (Wilcoxon p = 0,11); − `kolejne` +0,0177, 10 z 10 (p = 0,002); − `bez` +0,0402, 10 z 10 (p = 0,002).
- **Rola prognozy: niespełniona.** `tozsamosc` − `current` +0,0004, lepsza w 5 z 10 (p = 0,92). Obraz z przewidzianą
  zmianą siły i obraz bez niej dają tę samą trafność, choć prowadzą do innych klatek (wspólne top-90 ze startem
  średnio 27–35 z 90).
- **Prognoza siły w pętli: spełniona** (5 z 6 obrotów z ≥ 8/10 ziaren poniżej „braku zmiany”); mediana błąd / brak
  zmiany 0,89–0,97 w ramieniu `tozsamosc`. Tak samo w ramionach kontrolnych (`bez` 0,86–0,98), więc to cecha przebiegu
  uczenia, nie doboru klatek. Model wybierany: Koopman 40, Langevin 30 razy na 70 obrotów.
- **FastGAN w pętli: spełnione.** Mediana części zmiany 1,00 w każdym obrocie; na zaburzonych kopiach 0,87–1,02;
  znak zgodny 0,90–0,97.
- Diagnoza siły tożsamości (`DIAG_SILA.json`): przewidziane zmiany kotwic są w połowie wzrostami, w połowie spadkami
  (średnio −0,004); siła na końcu biegu w `tozsamosc` wyżej niż w `current` tylko w 5 z 10 ziaren. Dobór klatek według
  prognozy nie przesuwa siły tożsamości sieci w żadną stałą stronę.
- Porównanie z wersją 2 (te same ziarna i rozgrzewka): `koopman` v2 0,4432 = `tozsamosc` v3 0,4432; `losowe`
  0,4341 → 0,4327; `bez` 0,3999 → 0,4030.

**Wniosek.** Wszystkie ogniwa działają technicznie: prognoza siły tożsamości bije „brak zmiany” o 3–11 %, FastGAN
realizuje przewidzianą zmianę, a wyszukane klatki są w dwóch trzecich inne niż bez prognozy. Mimo to prognoza nie
poprawia uczenia: ramię z prognozą i ramię kontrolne bez niej dają identyczną trafność. Przewaga nad losowaniem
(+0,01, nieistotna, ta sama co w wersji 2) pochodzi z wybierania klatek podobnych do wycinków narzędzia, nie z
prognozy. Kolejne klatki tych samych filmów są gorsze od losowych (−0,007), a każde dokładanie bije brak dokładania.

## Po wyniku: zliczenie wygranych i sondy syntetyczne (2026-09-29, prośba użytkownika; analizy dodatkowe)

**Wygrane `tozsamosc` w 10 ziarnach** (`wyniki_eden/ANALIZA.json`):
- pole pod krzywą mAP@0,5 (obroty 3–9): z `current` 5, z `losowe` 7, z `kolejne` 10, z `bez` 10;
- mAP@0,5 na końcu (krok 4500): z `current` 6 (+0,012), z `losowe` 8 (+0,023), z `kolejne` 9, z `bez` 9;
- `current` wygrywa z `losowe` w 9 z 10 (pole pod krzywą) — przewagę nad losowaniem daje wybór klatek podobnych do
  wycinków narzędzia, wspólny dla obu ramion; sama prognoza (`tozsamosc` − `current`) jest na poziomie rzutu monetą.

**Sondy syntetyczne** (`ocena_sond.py` → `wyniki_eden/OCENA_SOND.json`, `.png`; końcowe sieci 5 ramion i stan po
rozgrzewce, 10 ziaren, 12 wycinków × 15 wersji + `empty_black`):
- pewność trafnego wykrycia (IoU ≥ 0,5), średnio po wersjach: `tozsamosc` 0,133, `current` 0,139, `losowe` 0,128,
  `kolejne` 0,124, `bez` 0,132, po rozgrzewce 0,086; `tozsamosc` lepsza od `losowe` w 7 z 10 ziaren, od `current` w 2;
- najpewniejsze wykrycie trafia w narzędzie w ~20 % sond we wszystkich ramionach (0,17–0,21); żółte tło ~0,07;
- na `empty_black` (narzędzia nie ma) detektor zgłasza wykrycie z pewnością 0,24–0,39, wyższą niż na narzędziu;
- wersja, w której `tozsamosc` bije `losowe` w ≥ 8 z 10 ziaren: tylko `yellow`, gdzie wszystkie ramiona mają ~0,02.

**Wniosek.** Sondy są dla tego detektora za daleko od danych uczących (wycinek 256 px powiększony do 176 px, sztuczne
tło), więc detektor prawie ich nie wykrywa i różnice między ramionami giną w szumie. Do porównania ramion na obrotach
i tłach potrzebne są sondy w skali klatek uczących (narzędzie wklejone w pełną klatkę 480×320), albo pomiar na
prawdziwych klatkach walidacji pogrupowanych według kąta i tła.

## Iteracja 2: dłużej i więcej ziaren (plan do zgody, 2026-09-29; nic nie zgłoszono)

Cel: sprawdzić, czy przewaga metody nad losowaniem (7–8 z 10 ziaren, +1 do +2 punkty mAP@0,5) rośnie przy dłuższym
dokładaniu i czy utrzymuje się na 20 ziarnach. Próg użytkownika: metoda lepsza w ≥ 80 % ziaren.

Zmiany względem kampanii 1818025 (reszta bez zmian: ramiona, wektor tożsamości, prognoza, FastGAN, wyszukiwanie,
reguła ±30, ocena co 100 kroków na tych samych 3000 klatkach walidacji):
1. 14 obrotów zamiast 7: uczenie do kroku 8000, 1876 dodanych klatek zamiast 938. Przewaga nad kolejnymi klatkami
   rosła z obrotem (+1,4 → +2,7 punktu), więc dłuższe dokładanie powinno ją pokazać wyraźniej.
2. Lista podobnych klatek na kotwicę 180 zamiast 90, w `tozsamosc` i `current` jednakowo. Przy 90 metoda dostawała
   w ostatnich obrotach mniej klatek niż kontrole (średnio 930 wobec 938); przy 14 obrotach brak byłby duży.
   Pula mieści ~3300 klatek przy regule ±30 (sprawdzone losowaniem, 3 próby), więc 1876 się zmieści.
3. Ziarna 47–66 (20 zamiast 10): 47–56 z rozgrzewką z wersji 2 jak teraz, 57–66 z nową rozgrzewką liczoną w zadaniu
   (ten sam kod, 1000 kroków, ~3 min).
4. Wszystkie ramiona od nowa od kroku 1000, bo kampania 1818025 nie zapisała stanu optymalizatora, więc nie da się
   jej uczciwie przedłużyć.

Kryteria (ustalone przed wynikiem):
- główne: `tozsamosc` > `losowe` w ≥ 16 z 20 ziaren, osobno dla mAP@0,5 na końcu (krok 8000) i pola pod krzywą;
- rola prognozy: `tozsamosc` > `current` w ≥ 16 z 20 ziaren;
- raport także względem `kolejne` i `bez`, test znaków, oraz stan w kroku 4500 (porównanie z kampanią 1818025).

Koszt: ~30 min na ziarno (5 ramion × 14 obrotów), razem ~10 GPU-godzin, ~1–1,5 h zegara; zapis ~38 GB latentów
(kwota: 68 z 200 GB). Najpierw próba kodu (1 zadanie, krótkie obroty), potem kampania z zależnością `afterok`.
