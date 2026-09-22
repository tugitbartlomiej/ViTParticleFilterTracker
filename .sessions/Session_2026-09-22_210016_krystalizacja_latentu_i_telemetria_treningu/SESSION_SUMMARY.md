# Krystalizacja latentu w trakcie uczenia i pelna telemetria treningu

## Cel sesji

Pytanie autora: jak zmienia sie latent detektora podczas uczenia, czy sie krystalizuje, czy
mozna z niego odczytac, ktore klatki albo grupy klatek niosly informacje lepiej poprawiajaca
model, i czy takie klatki da sie wskazac z analizy koopmanowskiej, bezposrednio albo posrednio.
W drugiej czesci: uzupelnic brakujace metryki treningu na Edenie, wlaczyc zapis straty
walidacyjnej i odpowiedziec, kiedy siec jest juz wytrenowana.

## Kontekst

- Repozytorium: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, galaz
  `codex/eden-nano-training-20260914`.
- Model: RF-DETR Nano, rozdzielczosc 384, zbior Cataract-LMM (tracking), pula uczaca 10 637
  klatek ze 118 filmow, walidacja 12 043 klatki z 25 filmow, splity rozlaczne po filmach.
- Material wejsciowy juz istniejacy: jedenascie eksportow latentu wzdluz zegara treningu
  (`results/koopman_langevin_epochs/latents_epochs`), dwadziescia siedem biegow etapu 1
  (dziewiec ramion doboru klatek po trzy ziarna), dwa nagrania czastek z checkpointu `ut_b0500_best`
  (pula uczaca 10 637 klatek i szesnascie klipow sondujacych 4 800 klatek).
- Poprzednia sesja w tym repozytorium:
  `.sessions\Session_2026-09-22_160617_koopman_tozsamosc_narzedzia`.
- Rownolegla sesja pracuje w tym samym repozytorium nad
  `experiments/dobor_klatek_stan_sieci` oraz dokumentem 37 i nie byla ruszana.

## Wykonane kroki

1. **Rozpoznanie materialu.** Ustalono, ze istnieje jedenascie eksportow latentu z biegu
   `epochs_uniform_time_b1500_s42` (checkpoint COCO plus epoki 2, 5, ..., 29) na tych samych
   4 800 klatkach z szesnastu klipow, oraz krzywa jakosci walidacyjnej per epoka
   (`epoch_val_map.json`, szczyt w epoce 6, wartosc 0,6869).

2. **Zalozenie eksperymentu** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu`
   z katalogami `dane/` i `wyniki/`.

3. **`krystalizacja.py` - zegar treningu.** Dla kazdego checkpointu policzono: liczbe faktycznie
   uzywanych kierunkow, udzial zmiennosci przypadajacy na roznice miedzy filmami, liczbe modow
   operatora kroku klatki o czesci rzeczywistej powyzej progu, srednia czesc rzeczywista osmiu
   najwolniejszych modow ("trwalosc"), oraz odczyt polozenia, wielkosci i wydluzenia narzedzia
   z latentu (uczenie i ocena na roznych filmach). Miedzy sasiednimi checkpointami policzono
   przesuniecie klatki po najlepszym obrocie, CKA i srednia korelacje kanoniczna.

   Wyniki (blok `box_pool`): przesuniecie spada z 1,05 do 0,26, CKA trzyma sie okolo 0,86 i nigdy
   nie dochodzi do jednosci. W bloku `query` liczba uzywanych kierunkow rosnie z 8,57 w epoce 2
   do 16,42 w epoce 29, czyli latent sie rozlewa. Trwalosc spada z 0,952 do 0,933.
   Korelacja trwalosci z jakoscia walidacyjna: od 0,74 do 0,89 zaleznie od bloku.

4. **Odpornosc sygnalu.** Przeliczono trwalosc dla czterech blokow latentu, trzech wymiarow
   redukcji i trzech liczb modow. W 32 z 36 ustawien szczyt trwalosci wypada w epoce 5, czyli
   tam, gdzie szczyt jakosci. Pozostale cztery ustawienia wskazuja epoke 2.

5. **`wklad_klatek.py` - ktore klatki model traci.** Policzono pokrycie ramki przewidzianej
   z prawdziwa (IoU) w kazdym checkpointcie dla 4 698 klatek z adnotacja. Po szczycie jakosci
   12,0 procent klatek traci ponad 0,05 IoU, a 8,2 procent zyskuje. Proba przewidzenia tej zmiany
   z latentu (uczenie i ocena na roznych filmach) dala wynik negatywny: R2 nie przekracza
   7 procent w zadnym opisie, wiekszosc wartosci jest ujemna.

6. **Pewnosc detekcji.** Srednia pewnosc rosnie z 0,83 w epoce 5 do 0,93 w epoce 29, mimo
   spadajacej jakosci. Udzial klatek o niskiej pewnosci pozostaje ponizej 2,5 procenta.

7. **`pokrycie_a_bledy.py` - pokrycie filmu a pudla.** Opis walidacyjny przepisano do bazy puli
   (zgodnosc baz: srednia 0,934). Dla czterech ramion (B1, B2, B3, K4) i dwunastu wspolnych
   filmow walidacyjnych policzono mediane odleglosci klatki do najblizszej wybranej klatki
   uczacej, w czterech przestrzeniach. Surowa korelacja z liczba pudel w przestrzeni wolnych
   modow wynosi 0,60, ale korelacja rangowa wynosi zero: caly efekt robi jeden film odstajacy
   (odleglosc 1,44, pudla 36,8 procent). Po zdjeciu sredniej filmu i sredniej ramienia znaki
   korelacji sa niespojne. Wynik negatywny.

8. **`co_przewiduje_wynik.py` - czy po zestawie widac wynik.** Dla dziewieciu ramion policzono
   luke pokrycia puli, rozpietosc chmury wybranych klatek, liczbe filmow i udzial szybkich
   klatek. Z wszystkimi ramionami luka pokrycia koreluje z jakoscia na poziomie od -0,67 do
   -0,84, ale bez ramienia B0 (kolejne klatki filmu) spada do -0,40 az -0,58, przy czym
   kierunki losowe wypadaja najlepiej. Wniosek: widac tylko ramie wyraznie zle, a przestrzen
   Koopmana nie ma przewagi nad kierunkami losowymi.

9. **Telemetria kampanii z Edenu.** Wczytano srodowisko `~/venvs/rfdetr_nano_clean_1_10_1`
   (RF-DETR 1.10.1, tensorboard 2.21.0) i zrzucono wszystkie skalary TensorBoard z 28 biegow
   do `TELEMETRIA_stage1.json` (697 kB), pobrane lokalnie.

10. **`telemetria_treningu.py` - kiedy siec jest wytrenowana.** Dla 28 biegow: szczyt jakosci
    wypada w epoce 8 (mediana, zakres od 4 do 16), a granica pol procenta od szczytu juz
    w epoce 6. Trening trwa 30 epok, jakosc spada od szczytu do konca srednio o 0,03.
    Ramie K4 (pokrycie w modach Koopmana) dochodzi do szczytu najszybciej, w epoce 5, a ramie
    B0 najpozniej, w epoce 16. Strata na zbiorze uczacym spada okolo piec razy.

11. **Brak straty walidacyjnej.** Ustalono, ze zaden bieg kampanii jej nie zapisywal:
    RF-DETR 1.10 ma `compute_val_loss: "auto"`, co oznacza pominiecie, gdy zadne narzedzie
    ani harmonogram jej nie pilnuje. Dopisano przelacznik `--val-loss` do
    `pipeline/train/train_rfdetr.py` i `pipeline/train/run_with_validation.py`. Przelacznik
    nie dotyka przepisu optymalizacji: `compute_train_metrics` swiadomie zostawiono wylaczone,
    bo zmienia sciezke normalizacji straty w kroku uczenia.

12. **Nowe biegi na Edenie.** Napisano `eden/run_telemetria_epok.slurm`: ten sam wariant
    `matrix_B2_uniform_time_b1500_sel42`, trzy ziarna, wagi co dwie epoki, latent kazdego
    zachowanego checkpointu, zrzut skalarow do `TELEMETRIA.json`. Trzy biegi (1809553, 1809554,
    1809555) wykonaly sie na dgx-1 i dgx-4 w okolo 70 minut kazdy. Telemetria ma teraz 40 tagow
    zamiast 26, w tym `val/loss` z pelnym rozbiciem na czlony.

13. **Pulapka zgloszen SLURM.** Pierwsze trzy zgloszenia z `--export=ALL,ZMIENNA=...` scheduler
    anulowal po sekundzie (`CANCELLED by 0`), bez zadnego logu. Minimalne zadanie testowe z tymi
    samymi zasobami przeszlo bez problemu. Rozwiazanie: wyeksportowac zmienne w powloce przed
    `sbatch`. Usunieto tez `#SBATCH --nodelist=hopper`, zeby mozna bylo wybrac partycje `short`,
    bo wezel hopper zajmowala rownolegla sesja.

14. **Wynik: kiedy siec jest wytrenowana.** Strata na walidacji ma dolek w epoce 6, 6 i 7
    (ziarna 42, 1337, 2024), a jakosc szczyt w epoce 6, 8 i 7. Strata na klatkach uczacych
    spada do konca z 6,2 do 1,07, a walidacyjna zawraca w epoce szostej i rosnie z 3,22 do 3,65.
    Trening do trzydziestej epoki kosztuje srednio 0,030 mAP.

15. **Wynik: sygnal bez etykiet.** Krzywe trwalosci dla trzech nowych biegow policzono na
    klastrze (`TRWALOSC_telemetria.json`), bo latenty waza okolo gigabajta. W blokach `query`
    i `features` szczyt trwalosci wypada w epoce 5, 5 i 7, a pojscie za nim kosztuje 0,0064,
    0,0035 i 0,0011 mAP, srednio 0,004 zamiast 0,030. Blok `global_pool` jest zawodny: dwa razy
    wskazuje epoke 3 i traci wtedy 0,015 oraz 0,009.

16. **Zapisy.** Dwa commity w repozytorium ViT: `7bb3484` (eksperyment, telemetria kampanii,
    przelacznik `--val-loss`, skrypt SLURM) i `7c3ddec` (wyniki trzech biegow z strata
    walidacyjna i porownanie sygnalow).

## Pliki zmienione / utworzone

### W projekcie ViT (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`)

Utworzone:

- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\README.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\krystalizacja.py`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wklad_klatek.py`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\pokrycie_a_bledy.py`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\co_przewiduje_wynik.py`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\telemetria_treningu.py`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\kiedy_przestac.py`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\dane\TELEMETRIA_stage1.json`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\dane\TRWALOSC_telemetria.json`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\dane\biegi\telemetria_uniform_time_b1500_s42\TELEMETRIA.json`
  (analogicznie dla ziaren 1337 i 2024)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_krystalizacja_box_pool.json`
  oraz warianty `query`, `features`, `global_pool`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_wklad_klatek.json`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_pokrycie_a_bledy.json`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_co_przewiduje_wynik.json`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_telemetria.json`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\wyniki\REPORT_kiedy_przestac_query.json`
  oraz warianty `box_pool` i `features`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\eden\run_telemetria_epok.slurm`

Zmienione:

- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\train\train_rfdetr.py`
  (nowy przelacznik `--val-loss`)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\train\run_with_validation.py`
  (przekazanie przelacznika)

### Zewnetrzne (poza repozytorium)

Na klastrze Eden, katalog `~/smallvit_runs/2026-09-21_koopman_langevin`:

- `TELEMETRIA_stage1.json` - zrzut skalarow z 28 biegow kampanii
- `TRWALOSC_telemetria.json` - krzywe trwalosci dla trzech nowych biegow
- `results_telemetria/telemetria_uniform_time_b1500_s{42,1337,2024}/` - pelne biegi z wagami
  co dwie epoki, `TELEMETRIA.json`, raportami walidacji i archiwami predykcji
- `latents_telemetria/telemetria_uniform_time_b1500_s{42,1337,2024}/` - po szesnascie eksportow
  latentu na bieg
- `SmallViTAblation/pipeline/train/train_rfdetr.py` i `run_with_validation.py` - kopie zmienionych
  plikow
- `SmallViTAblation/eden/run_telemetria_epok.slurm`

Nie ruszano: `experiments/dobor_klatek_stan_sieci`, dokumentu 37 ani zadan `features_kg_`
rownoleglej sesji.

## Otwarte problemy

1. Rozdzielczosc sygnalu bez etykiet wynosi dwie epoki, bo tyle wynosi odstep miedzy zapisanymi
   wagami. Nie da sie z niego odczytac dokladniejszego momentu zatrzymania.
2. Sygnal sprawdzono na jednym wariancie doboru klatek (B2) i jednym zbiorze. Nie wiadomo,
   czy trzyma sie przy innych ramionach, budzetach i modelach.
3. Blok `global_pool` daje falszywy wskaz w dwoch na trzy ziarna. Nie wiadomo, czemu akurat on.
4. Pytanie o wklad pojedynczych klatek pozostaje bez odpowiedzi. Z dostepnych czterech ramion
   nie da sie oszacowac wplywu zestawu na wynik, a roznice miedzy zdrowymi ramionami (od 0,682
   do 0,689) sa rzedu szumu ziarna (0,003).
5. Nie wiadomo, gdzie dokladnie siedzi spadek jakosci po szczycie. Na klipach sondujacych
   pokrycie ramki prawie sie nie zmienia, wiec strata mAP bierze sie z czegos innego, zapewne
   z falszywych wykryc i kalibracji pewnosci.

## Nastepne kroki

1. Powtorzyc bieg z zapisem wag co epoke, zeby sygnal bez etykiet mial rozdzielczosc jednej
   epoki, i sprawdzic, czy wtedy trafia dokladnie w dolek straty walidacyjnej.
2. Sprawdzic sygnal na innym ramieniu doboru klatek (na przyklad K4) i na innym budzecie,
   zeby wykluczyc, ze dziala tylko dla rownomiernego doboru w czasie.
3. Policzyc rozklad falszywych wykryc i kalibracje pewnosci w funkcji epoki, zeby ustalic,
   skad bierze sie spadek jakosci po szczycie.
4. Wlaczyc `--val-loss` do przepisu kampanii jako stale, bo nic nie kosztuje poza jednym
   przebiegiem po walidacji, a daje standardowy sygnal przeuczenia.
5. Zaplanowac etap 2 doboru klatek. Nadal czeka na zatwierdzenie: ramiona B2, B3, K4
   z poprawionym doborem, K5, K5c plus ramie odniesienia na skladowych glownych, trzy ziarna,
   budzet 1500. Nowa przeslanka z tej sesji: dwanascie epok w przepisie etapu 2 wystarczy,
   bo szczyt wypada okolo szostej.
