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
