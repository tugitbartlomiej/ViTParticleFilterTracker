# Koordynacja odbiorcy temporalnego dla main/Hubble

Aktualizacja 2026-10-09. Ownership tylko `vit_k1000_wzorce_pydmd_deeptime_20261009`.

- Smoke bibliotek 1825145 COMPLETED: PyDMD 2025.8.1, deeptime 0.4.5, NumPy 2.2.6, scipy 1.15.3, sklearn 1.6.1. 10 testów PASS, 128 trajektorii/64000 przejść. Prywatny venv: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv`.
- Smoke uzupełniający 1825150: usuwa sztuczne unit singular values VAMP wywołane cechami `[pozycja,poprzedni przyrost]`; nowe cechy są wyłącznie przyrostem. Operatorów DMD/DMDc i splitu nie retunowano po test. Stary wynik i snapshot zachowane.
- Adapter czyta rzeczywiste pola producenta; panel1024 i dawny128 pozostają podzbiorami, niezależnymi od wizualizacji900k. PCA3 nie trafia do fit.
- Gotowa docelowa komenda oczekuje dependency **na scaleniu** i manifestu `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/results/merged/TEMPORAL_PANEL_MANIFEST.json`. Potrzebny potwierdzony merge_job_id, potem odbiorca zgłasza swój CPU job `afterok:ID`.
- Odczytano kod Hubble: ścieżki plików względem merged, `analysis_split` i role bitowe, status manifestu complete. Nic w producerze nie zmieniono.

## Rekomendacja rozszerzenia całej populacji

Kontrakt szczegółowy: [KONTRAKT_DANYCH.md](../../experiments/vit_k1000_wzorce_pydmd_deeptime_20261009/KONTRAKT_DANYCH.md).

Stała analityczna PCA32 train-only, wszystkie1500×600×101: 11.6352 GB surowych projekcji stanów. Stany101 i trzy kontrole g/m/v dla100 początków interwałów: 46.1952 GB łącznie przed kompresją. Przy zgłoszonych133 GB może się zmieścić z obecnymi wynikami, ale producent powinien sprawdzić realny aktualny zapas i budżet niedokończonych plików. Panel pełne768 co1 zachować. Nie zapisujemy TB.

Warunek krytyczny: PCA32 musi być stała przed projekcjami. Baza panelowa train-only jest dopuszczalna jako wstępna baza populacyjnych projekcji, lecz nie można nazwać jej fit całej populacji. Streaming fit na wszystkich filmach w tym test byłby leakage. Jeżeli nie ma gotowej bazy i czasu na drugi przebieg, bezpieczne rozszerzenie pierwszego replay to pełne768 normy/dot/cos/kierunkowa zmiana i fazy/role wszystkich tokenów oraz train-only strumieniowe momenty kierunków. To audyt populacyjnej geometrii, nie populacyjny DMDc.

Samo g/m/v co5 nie wystarcza do odtworzenia pięciu kroków Adama. Co więcej `P(m/sqrt(v)) != P(m)/sqrt(P(v))`; projected m2 może być ujemne. Nie należy stosować Adama do zredukowanych momentów. KnownB kontrola musi powstać z działania Adama i clamp w pełnych768 przed projekcją. Suma pięciu analitycznych kroków jest niemal tożsama z docelowym przemieszczeniem, więc jest wyłącznie teacher-forced kontrolą integracji, nie odkryciem predykcyjnego prawa. Do niebanalnego DMDc przy lag5 potrzebne są jasno nazwane wejścia początkowe/sekwecyjne i ich dostępność w prognozie; przy jednym wejściu z t nie udawać pełnego sterowania interwałem.

Regimy panelu nie dowodzą reżimów900k. Można testować transfer zamrożonych panelowych baz i reguł na projekcjach populacji; należy osobno zmierzyć zachowanie energii przyrostów i błędy poza bazą na filmach test, nie dopasowując jej ponownie.

Kanał natywnych agentów nie jest dostępny w narzędziach tego wykonawcy; wcześniejsza próba przekazania do main przez narzędzie wątków została odrzucona jako wiadomość do natywnego ancestor. Ten plik i bieżące commentary są komunikatem koordynacyjnym; nie użyto obejść ani zmiany cudzych plików.

## Gotowe wykonanie po scaleniu — finalny kontrakt producenta

Odbiornik potwierdza finalny format: `analysis_positions_pca32`, `analysis_state_residual_norm`, `analysis_increment_residual_norm`; manifesty `TEMPORAL_PANEL_MANIFEST.json`, `GLOBAL_MOTION_DESCRIPTOR_MANIFEST.json`, `ANALYTICAL_PCA32_MANIFEST.json` i `REPLAY_INDEX.jsonl` pod `results/merged/`. Kontroluje wzajemną zgodność list/hashy, `basis_path`, hash kontenera, ortonormalność768×32, brak filmów validation/test w `fit_image_indices`, basis hash każdego NPZ, status walidacji replay i niezmienność wag.

**Jedna zalecana komenda na Edenie**, gdy `submission/PRODUCTION_SUBMITTED.json` Hubble ma status submitted:

```bash
/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/.venv/bin/python \
  /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/code/submit_after_merge.py
```

Opcjonalnie `--merge-job NUMER` sprawdza zgodność z markerem Hubble. Skrypt odczytuje rzeczywisty ID, uruchamia gate testów, zamraża snapshot kodu i zgłasza **oba** poniższe zadania. Powtórne wywołanie po sukcesie wyłącznie drukuje istniejące ID. Częściowy lub nieznany wynik zatrzymuje ponowienie, aby nie tworzyć duplikatu. Marker odbiorcy:

`/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/results/PRODUCTION_SUBMITTED.json`

Dokładne komendy sbatch generowane przez skrypt, do odczytu (nie uruchamiać równolegle z wrapperem):

```bash
sbatch --parsable --dependency=afterok:MERGE_ID --kill-on-invalid-dep=yes \
  /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/code_frozen_production/panel.slurm \
  /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009/results/merged/TEMPORAL_PANEL_MANIFEST.json
sbatch --parsable --dependency=afterok:MERGE_ID --kill-on-invalid-dep=yes \
  /home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/code_frozen_production/population.slurm
```

Oba joby CPU4/28GB, panel do3h, population do4h. Nie żądają GPU. Zamrożone źródła i hashe są w markerze; następne lokalne edycje nie zmienią oczekujących jobów. `--kill-on-invalid-dep` dotyczy wyłącznie naszych dwóch konsumentów, nie jobów Hubble.

Docelowe rooty wyników Eden:

- panel1024: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/results/panel64/` (64obrazy×16tokenów).
- population900k: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/results/population900k/`.
- logi: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/logs/panel_JOBID.out` oraz `population_JOBID.out`.
- docelowe lokalne kopie: `outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/` i `population900k/`.

Przed zgłoszeniem: smoke1825150 COMPLETED,13testów aktualnego pipeline PASS. Nowy learnedB nie był retunowany na teście. Regimy populacyjne to rzeczywisty fit train/przypisanie wszystkich punktów, nie rzutowanie reżimów starego panelu. Populacyjny deeptime DMD przewiduje przyrost co5; h20 ma dokładne błędy32D i uczciwe granice błędu768, ponieważ kąty residuów między odległymi stanami nie są zapisane.

Ostatnia korekta przed freeze: manifest panelu jest przekazywany jako argument `$1` skryptu SLURM; wrapper nie używa listy `--export`. Estymatory i hiperparametry pozostały bez zmian. Zgodnie z dyspozycją main uruchamia wrapper jeden raz; ten wykonawca nie zgłasza równolegle konsumentów. Immutable copy powstaje wewnątrz wrappera po przejściu gate testów; nie należy jej tworzyć ręcznie wcześniej.

## Trzeci konsument — zgłoszony suplement PyDMD DMDc całej populacji

2026-10-09 02:10:31 UTC: **job1825187**, dependency **afterok1825166**, dokładnie jedno zgłoszenie na osobne zlecenie użytkownika. Istniejące1825171/1825172 i ich snapshot `code_frozen_production` niezmienione; checksum wszystkich plików i markera sprawdzono przed oraz po zgłoszeniu suplementu.

Collector: dodać trzeci folder **`population900k_dmdc_supplement`**:

`/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/results/population900k_dmdc_supplement/`

Snapshot:

`/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/code_frozen_population_dmdc_supplement/`

Log: `logs/population_dmdc_supp_1825187.out`. Marker: [SUPPLEMENT_DMDC_SUBMITTED.json](SUPPLEMENT_DMDC_SUBMITTED.json). Gate: [TESTS_READY_POPULATION_DMDC_SUPPLEMENT.txt](TESTS_READY_POPULATION_DMDC_SUPPLEMENT.txt). Lokalna docelowa kopia: `outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k_dmdc_supplement/`.

Suplement dopasowuje jedną wspólną A/B przez faktyczne PyDMD.DMDc.fit na train-only próbce stanówPCA32. U=[norma g start, norma m1 start, norma m2 start, LR(t+1), t/500, bias]. Bez mean/last gradientów, przyszłych delt lub stanów w U. Normy są amplitudami, nie pełnymi wektorami sterowania. Każde900k tokenów ma wszystkie100 prognoz h5 i97 okien h20. H20 obejmuje frozenStartU oraz jawne futureU_oracle; baseline DMD autonomiczne na tych samych stanach i persistence. Stratyfikacja faz/roli/statusu, OOD, ranga, stabilność, retained energy i znany błąd768 h5; h20 tylko bounds. Ten suplement rozszerza poprzedni zakres; nie przedstawia wcześniejszego joba1825172 jako kontrolowanego DMDc.


## Zakończenie analizy i syntezy, 2026-10-09 02:54 UTC

Wszystkie trzy joby **1825171/1825172/1825187 COMPLETED0:0**, bez awarii i rerunów. Collector PID51580 zakończył pobranie panel64/population900k/population900k_dmdc_supplement. Ponowny hash1539 plików PASS; wszystkie18 plików frozen snapshots zgodne ze zgłoszeniem. Nie zmieniono estimatorów ani hiperparametrów po teście.

**Raport dla main/użytkownika:** [WNIOSKI_WZORCE.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/WNIOSKI_WZORCE.md). Diagnozy obliczone wyłącznie z verified outputs: [SYNTEZA_F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/SYNTEZA_DIAGNOSTYKA.jsonNOSTYKA.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/SYNTEZA_DIAGNOSTYKA.json). Potwierdzenie końcowe: [F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/VALIDATION_SYNTHEZY.jsonATION_SYNTHEZY.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/VALIDATION_SYNTHEZY.json).

Najważniejsze: wspólna dynamika istnieje, lecz nie oznacza gotowej poprawki bez GT. Populacyjny DMD prędkości h5: RMSE względem persistence niższy o~54%, względem last-increment o3,61/5,65% w fit/min_norm. Populacyjny scalar DMDc: ogólny iloraz RMSE do persistence0,99223, bootstrap filmów[0,97830;1,01213]; pomaga fit, pogarsza min_norm i targety. PCA32 zachowuje tylko31–33% energii ruchu targetu, podczas gdy innych78–81%. Nieudane targety mają24,96% odwróceń co5; udane-fit0,86%. Brak testu aplikacji correctora, zatem brak nowego wyniku IoU/mAP.

Podział70/23/25filmów,914/293/293obrazów; fit/skale/PCA train-only. Wszystkie900k mają100 interwałów; VAMP oznaczył89,3814mln oraz618600mixed jako255. Suplement ocenił90mln h5 i87,3mln h20 okien per metoda. Raport rozdziela full768 panel co1, PCA32+geometrię768 populacji co5, startU i futureUoracle. Model gpt-6-astra, effort xhigh. Zadanie syntezy zakończone.
