# Mały ViT, wersja 2: wektor tożsamości na 768 neuronach, start po 21 epokach

Data: 2026-09-28. Zgoda użytkownika: „Dobra, to puść trening na Edenie i pomiar na Edenie i zobaczmy, jak to wyjdzie”
(po analizie wstecznej z `../maly_vit_od_zera/PLAN_NASTEPNE_SPRAWDZENIE.md`).

## Skąd ta wersja

Analiza wsteczna 11 ziaren wersji 1 (`../maly_vit_od_zera/wyniki/ANALIZA_WSTECZNA.*`):
- lista neuronów tożsamości wymienia się w połowie co ~5 epok, kurczy się (80 → 33 neuronów), rdzeń nie powstaje;
- prognoza Koopmana i Langevina na tej liście nie bije „braku zmiany” w żadnej chwili 1–96 epok;
- wektor tożsamości na wszystkich 768 neuronach (średnia na narzędziu minus średnia negatywów, w sd negatywów) jest
  stabilny od ~16 epok (cosinus z chwilą sprzed 5 epok 0,85–0,90) i gładko dryfuje do stanu końcowego;
- wykrywanie ma ~70 % poziomu końcowego po ~21 epokach.

## Co się zmienia względem wersji 1 (kod w `kod/`, różnice opisane w nagłówku `kod/petla.py`)

1. Stan prognozy: aktywacje narzędzia wszystkich 768 neuronów na 180 sondach, standaryzowane negatywami z chwili
   prognozy (zamiast listy neuronów pełnego latentu). Cel FastGAN też na 768 neuronach.
2. Rozgrzewka 2 obroty × 500 kroków = 1000 kroków = 21 epok bazy (było 32), potem 7 obrotów w ramionach; łącznie
   4500 kroków jak w wersji 1. Klatek dodanych: 7 × 134 = 938.
3. Horyzont prognozy: 1 obrót (500 kroków), bo klatki dodane teraz działają w najbliższym obrocie (było 3 obroty).
4. Pomiar w każdym ramieniu i obrocie (`pomiar()` w `kod/petla.py`):
   - stabilność wektora tożsamości (cosinus z poprzednim obrotem i z końcem rozgrzewki);
   - test wsteczny prognozy na 768 neuronach i na liście neuronów z wersji 1 (ten sam horyzont);
   - **sprawdzenie prognozy z poprzedniego obrotu na tym, co się naprawdę stało** (bez podglądania przyszłości):
     błąd dla sond i dla wektora (średnia po sondach) względem „braku zmiany”, zgodność znaku zmiany neuronów.
   - Na końcu ramienia ten sam pomiar trafia do `KONIEC.json` (`pomiar_koncowy`).

Ramiona bez zmian: `koopman`, `losowe`, `bez`. Ziarna 47–56 (pary z wersją 1). Nowe ramiona (`current` jako kontrola
prognozy, `kolejne z rzędu`) nie wchodzą — czekają na osobną zgodę; wspólna rozgrzewka zostaje na Edenie, więc można je
dołożyć później bez powtarzania rozgrzewki (`petla.py --ramiona ...`).

## Kryteria (ustalone przed wynikiem)

- Główne: pole pod krzywą mAP@0,5 w obrotach 3–9 (średnia ocen co 100 kroków po rozgrzewce): `koopman` > `losowe`
  i > `bez` w ≥ 8 z 10 ziaren (`analiza.py`).
- Prognoza wektora działa, jeśli w sprawdzeniu z poprzedniego obrotu mediana błędu wektora / brak zmiany < 1
  i < 1 w ≥ 8 z 10 ziaren, w co najmniej 4 z 6 obrotów, w których sprawdzenie jest możliwe (obroty 4–9).
- Porównanie z wersją 1 na tych samych ziarnach: średnia mAP@0,5 w oknie kroków 1600–4500 (wspólnym dla obu wersji)
  w ramieniu `koopman`, znak różnicy w 10 parach.

## Eden

`~/smallvit_runs/2026-09-28_maly_vit_wektor/` (`kod/`, `logi/`, `wyniki/`); wejścia i `timm` z katalogu wersji 1.
Skrypt: `petla_eden.slurm` (hopper,short, 1 GPU, 8 CPU, 64 GB, 2 h 30 min, sprzątanie `/dev/shm` pułapką).
Najpierw próba kodu (`--array=0-0 ... 1 proba`), potem kampania `--array=0-9 ... 47` z zależnością afterok.

## Sprawdzenie przed kampanią (offline, na latentach wersji 1, `test_wektor_offline.py`)

Prognoza na 768 neuronach z tymi samymi funkcjami co w wersji 2 (`kod/prognoza.py`: `prognoza_bloku`, `porownaj`),
ramię `bez` wersji 1, 11 ziaren, chwile 1000–4000 kroków co 250, horyzont 500 kroków, porównanie z prawdziwą
przyszłością (`wyniki/TEST_WEKTOR_OFFLINE_v1.json`):
- błąd wektora / brak zmiany: Koopman 1,00–1,04, Langevin 0,99–1,06 (poniżej 1 najwyżej w 7 z 11 ziaren);
- błąd sond / brak zmiany: Koopman 1,00–1,02, Langevin 1,03–1,06;
- zgodność znaku zmiany neuronów: 0,45–0,59 (rzut monetą 0,50).

Wniosek przed wynikiem kampanii: stabilny wektor nie znaczy przewidywalny. Zmiana wektora w ciągu jednego obrotu
jest zdominowana przez wahania uczenia, więc na tych danych prognoza nie powinna bić „braku zmiany” także w wersji 2.
Kampania i tak rusza (zgoda użytkownika) — mierzy to samo w pętli, z innym startem i z dokładaniem klatek.

## Wyniki kampanii (2026-09-28, Eden 1817949, ziarna 47–56, wszystkie COMPLETED ~20 min)

Pliki: `wyniki_eden/` (JSON i logi; obrazy FastGAN i latenty zostały na Edenie), `wyniki_eden/ANALIZA.json`
(`analiza.py wyniki_eden`), `wyniki_eden/POMIAR.json` (`analiza_pomiar.py`).

- **Kryterium główne: spełnione, ale słabo.** Pole pod krzywą mAP@0,5 w obrotach 3–9: `koopman` 0,4432, `losowe`
  0,4341, `bez` 0,3999. `koopman` > obu w 8 z 10 ziaren; nad losowym +0,009 (Wilcoxon p = 0,064, test znaków
  p = 0,11); nad `bez` +0,043 w 10 z 10 (p = 0,002). Dokładanie losowych klatek daje +0,034 (10 z 10).
- **Prognoza wektora nie działa (kryterium niespełnione, 0 z 6 obrotów).** Sprawdzenie prognozy z poprzedniego
  obrotu na tym, co się naprawdę stało, ramię `koopman`: mediana błędu wektora / brak zmiany dla Koopmana 0,989;
  1,018; 1,006; 1,004; 1,003; 0,999 (obroty 4–9), poniżej 1 w 3–7 z 10 ziaren; Langevin 1,003–1,037. Zgodność znaku
  zmiany neuronów 0,52–0,61. W ramionach `losowe` i `bez` to samo.
- Wektor tożsamości między obrotami: cosinus 0,84–0,86 (stabilny kierunek, zmiana w obrocie nieprzewidywalna).
- **Wersja 2 kontra wersja 1** (średnia mAP@0,5 w krokach 1600–4500, 10 par ziaren): `koopman` +0,014 (lepsza
  w 7 z 10), `losowe` +0,015 (7 z 10), `bez` +0,003 (6 z 10). Zysk pochodzi z wcześniejszego i większego dokładania
  (start po 21 epokach, 938 klatek zamiast 804), bo `losowe` zyskuje tyle samo co `koopman`; wektor tożsamości
  nie zmienił przewagi Koopmana nad losowym (+0,013 w wersji 1, +0,009 w wersji 2).

**Wniosek.** Ramię `koopman` znów dobiera klatki minimalnie lepiej niż losowanie, ale prognoza latentu, na której
ma się opierać, jest przez całą kampanię równa „brakowi zmiany”. Przewaga nie może więc pochodzić z prognozy;
najprostsze wyjaśnienie z wersji 1 (klatki szukane wokół obrazów narzędzia) pozostaje. Niesprawdzona hipoteza
błędu w liczeniu: PCA do 8 wymiarów liczona na poziomach latentu (różnice między sondami), a nie na jego zmianach
w czasie, więc model może nie widzieć ruchu (użytkownik przerwał sprawdzenie; czeka na decyzję).

## Tożsamość po 21 epokach metodą użytkownika (2026-09-28, `tozsamosc_21ep.py`)

Użytkownik: „po 21 epokach wyznacz tożsamość teraz”. Krok 1000 (koniec rozgrzewki wersji 2), średnia 5 ostatnich
zapisów, 192 sondy (12 wycinków × 16 wersji), definicje z `kod/tozsamosc.py`. Wyniki: `wyniki_eden/TOZSAMOSC_21EP.json`
(lista neuronów pełnego latentu z udziałem zapaleń w każdej wersji), `wyniki_eden/TOZSAMOSC_21EP.png`.
- Pełny latent: 45–87 neuronów (średnio 66). **Rdzeń: 0 w każdym z 10 ziaren.** Najlepszy neuron zapala się we
  wszystkich 15 wersjach na 25–50 % wycinków (próg rdzenia 75 %).
- Domknięcie wersji (średnio): rozjaśnione narzędzie 0,44, obrót 180° 0,42, przyciemnione 0,40, oryginał 0,39 …
  skala 0,8: 0,18, **czarne tło 0,03, żółte tło 0,00**. Po 21 epokach neurony tożsamości prawie w ogóle nie
  zapalają się na narzędziu na sztucznym tle — sieć rozpoznaje narzędzie razem z tłem operacyjnym.
- Bez czarnego i żółtego tła (13 wersji, odejście od definicji, tylko informacyjnie): ≥ 75 % wycinków w każdej
  wersji — 2 neurony w s52 i 1 w s55, w pozostałych 0; ≥ 50 % — 1–5 neuronów na ziarno.
- Sieć nieuczona (krok 0) przechodzi ten sam test lepiej: 231–329 neuronów „pełnego latentu”; 26–58 neuronów
  z listy po 21 epokach było na niej już w sieci nieuczonej.

## Wektor tożsamości po 21 epokach (2026-09-28, `wektor_tozsamosci_21ep.py`)

Użytkownik: „policz wektor tożsamości po 21 epokach teraz” (kierunek w latencie zamiast pojedynczych neuronów).
Wektor = liniowy rozdzielacz narzędzie/tło (LDA ze ściąganiem kowariancji) na średnich latentach sond, próg zapalenia
jak dla neuronów (średnia + 2 sd rzutów negatywów). Wyniki: `wyniki_eden/WEKTOR_21EP.json`, `WEKTOR_21EP.npz`
(wektory 768 na ziarno), `WEKTOR_21EP.png`.
- **Wersja wyłączona** (wektor uczony bez danej wersji, sprawdzany na niej): narzędzie zapala wektor w 100 % wycinków
  we wszystkich wersjach poza żółtym tłem (85 %); AUC 1,00 (żółte 0,93). Rdzeń wektorowy według progu użytkownika
  (≥ 75 % w każdej wersji wyłączonej): **8 z 10 ziaren** (s49 i s50 zawodzą na żółtym tle, 42–50 %).
- **Wycinek wyłączony** (nowe narzędzie, którego wektor nie widział): 100 % zapaleń, AUC 1,00 w każdym ziarnie.
- Tło fałszywie zapala wektor w wersjach wyłączonych: 0–41 % (najwięcej obroty i przesunięcia — rogi i krawędzie
  po przekształceniu), przy AUC 1,00, więc problemem jest próg, nie kierunek.
- Prosty wektor (średnia narzędzia − średnia tła): słabszy, najsłabsze czarne tło (77 %).
- **Sieć nieuczona (krok 0) daje to samo albo lepiej**: 100 % we wszystkich wersjach poza żółtym (97 %), mniej
  fałszywych zapaleń tła. Test na tych sondach nie odróżnia więc tożsamości wyuczonej od zwykłej różnicy wyglądu
  (jasny, cienki metal kontra tkanka), którą widać już w losowych wagach.

**Wniosek.** Wektor tożsamości po 21 epokach istnieje i przenosi się na nowe wersje i nowe narzędzia, ale ten sam
test przechodzi sieć nieuczona. Żeby zmierzyć tożsamość wyuczoną, potrzebne są trudniejsze negatywy (rzeczy podobne
do narzędzia, które nim nie są: odblaski, krawędzie tkanek, cienkie jasne struktury, klatki bez narzędzia) albo
próba przyczynowa (usunięcie wektora z latentu psuje wykrywanie).

## Koopman i Langevin na wektorze tożsamości z 21 epok (2026-09-28, `koopman_na_wektorze.py`, 10 ziaren)

Wektor i standaryzacja stałe z kroku 1000; stan = rzut latentu narzędzia każdej ze 180 sond na wektor („siła
tożsamości” sondy); chwile 1000–4250 co 250 kroków; prawdziwa przyszłość. Wyniki: `wyniki_eden/KOOPMAN_WEKTOR.json`,
`KOOPMAN_WEKTOR.png`. (Wstępny przebieg na 3 ziarnach dawał lepsze liczby, np. trafność kierunku 85 %; na 10
ziarnach jest niżej — raportowane są liczby z 10 ziaren.)
- **Pierwszy raz prognoza bije „brak zmiany”, powtarzalnie, ale niewiele.** Błąd dla sond / brak zmiany, horyzont
  500 kroków: ramię `bez` Koopman 0,94 (lepszy w 85 % przypadków), Langevin 0,90 (75 %); `losowe` 0,93 / 0,92;
  `koopman` 0,94 / 0,94. Horyzonty 50 i 250 kroków podobnie (0,90–0,95).
- Siła tożsamości (średnia po sondach): Langevin 0,86 × brak zmiany w `bez`, ~1,0 w ramionach z dokładaniem.
- Kierunek zmiany siły (wzrośnie czy spadnie): `bez` Koopman 64 %, Langevin 69 %; `losowe` 67 % / 55 %;
  `koopman` 54 % / 58 % (rzut monetą 50 %).
- Siła tożsamości wzdłuż wektora z 21 epok maleje z czasem: 6,13 → 5,30 (`bez`), 4,36 (`losowe` i `koopman`);
  Langevin widzi hamowanie (k ≈ −0,16 na blok): tożsamość obraca się od kierunku z 21 epok, szybciej przy dokładaniu klatek.
