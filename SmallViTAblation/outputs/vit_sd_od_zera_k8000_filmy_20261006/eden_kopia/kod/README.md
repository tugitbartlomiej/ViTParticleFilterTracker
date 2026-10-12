# Dyfuzja od losowych wag, filmy online, teacher K8000

VAE, UNet i adapter zaczynają z losowych wag. Jedynymi wczytanymi wagami
początkowymi są zamrożone wagi teachera ViT K8000. Wznowienie wczytuje własny
checkpoint tego eksperymentu. Nie korzystamy z pretrained SD, VAE ani adaptera.
Filmy są dekodowane online; obrazy i latenty pozostają w RAM. Zapisywane są
checkpointy, metryki i logi, bez obrazów ani cache aktywacji na dysku.

## Ścieżki i pochodzenie

- Nowy root Eden: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy`.
- Kod: `kod/`; wejścia: `wejscia/`; trening: `wyniki/trening/`; logi: `logi/`.
- Smoke: osobny `wyniki/smoke_JOBID/`, nigdy checkpoint produkcyjny.
- Manifest: dokładna kopia `2026-10-06_vit_sd_pelny_latent_filmy/wejscia/FILMY.json`.
  Deployment nie przebudowuje splitu ani nie kopiuje filmów.
- Teacher lokalny: `assets/MalyViT_K8000_s47_losowe/MALY_VIT_K8000_s47_losowe.pth`.
- Teacher na Eden, źródło: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_petla43_sd_prowadzenie/wyniki/petla43b_s47/losowe/wagi_koncowe.pth`.
- SHA256 teachera: `0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b`.

Pozostaje dotychczasowy Python
`/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python` oraz
dotychczasowe katalogi `deps` z poprzedniego eksperymentu i
`2026-09-30_dyfuzja_sd_filmy`. Offline HF jest wymuszone. Deployment nie instaluje
bibliotek ani nie przenosi starych wag generatywnych.

## Jawne polecenia operatora

Uruchamiaj z katalogu eksperymentu. `--help` oraz import niczego nie wysyłają.
Bez flag skrypt odmawia działania. Domyślny alias SSH to `eden-cluster`;
opcjonalnie użyj `--host ALIAS`.

```powershell
python -B deploy.py --help
# Tylko transfer kodu i wejść, bez sbatch:
python -B deploy.py --deploy
# Jawnie zgłasza zadanie smoke na czterech GPU:
python -B deploy.py --smoke
# Po sprawdzeniu logów i checkpointów smoke, jawny start/wznowienie treningu:
python -B deploy.py --submit
# Można połączyć transfer z jednym wybranym rodzajem zgłoszenia:
python -B deploy.py --deploy --smoke
python -B deploy.py --deploy --submit
```

`--smoke` i `--submit` są wzajemnie wykluczające. Obie flagi wykonują `sbatch`;
nie należy ich używać do samej kontroli składni. Smoke nie zgłasza treningu.
Deployment wymaga gotowych plików pozostałych agentów (`run.py`, `model.py`,
`model_snapshot.py`, `dane.py`). Weryfikuje składnię, hashe transferu i teachera,
zdalne `bash -n` oraz `run.py --help`. Rozbieżny istniejący manifest lub teacher
powoduje błąd zamiast nadpisania. Nie usuwa checkpointów i wyników.

## SLURM i CLI treningu

Nazwa zadania: `sd0k8000`. Konto `transformers_vsc`, partycja `hopper`, cztery GPU,
24 CPU, 160 GB RAM, czas `23:30:00`, sygnał USR1 dziesięć minut przed końcem.
`--smoke` nadpisuje czas zgłoszenia przez `sbatch --time=00:30:00`.
Launcher DDP wykonuje odpowiednik:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 run.py \
  --manifest "$MY/wejscia/FILMY.json" \
  --vit "$MY/wejscia/MALY_VIT_K8000_s47_losowe.pth" \
  --output "$MY/wyniki/trening" \
  --vae-steps 30000 --sd-steps 100000 --eval-every 500 \
  --batch-size 4 --workers 2 --resume auto
```

Smoke ma osobny output i parametry `--smoke --vae-steps 2 --sd-steps 2
--eval-every 1 --batch-size 1 --workers 0 --resume auto`. Batch i workers są
argumentami per rank. Skrypt SLURM faktycznie używa `deploy.py launch-ddp`, aby
zachować specjalny kod 75, który zwykły torchrun zamienia na ogólny błąd.

## Wznowienie i ograniczona kontynuacja

`latest_checkpoint.pt` pozostaje w `wyniki/trening/`. Każdy segment używa
`--resume auto`; zmiana sygnatury wejść/konfiguracji może zablokować wznowienie.
Sprawdź `LATEST.json`, `STATUS.json`, `EVAL.json`, `PROVENANCE.json` oraz log SLURM.

USR1 batch shella tworzy `STOP_TIME`. `run.py` rozpoznaje ten plik, synchronizuje
stop ranków i publikuje checkpoint przed wyjściem 75. Wewnętrzny launcher
zachowuje ten kod z torchrun. Manualny plik `STOP` jest rozpoznawany przez trening,
ale nie uprawnia do automatycznej kontynuacji. STOP nie jest usuwany automatycznie.

Kolejny segment zostaje zgłoszony wyłącznie, gdy równocześnie:

1. To trening produkcyjny, a batch shell otrzymał sygnał czasu USR1.
2. Workerzy zakończyli się kodem 75; mieszany lub inny błąd jest odrzucany.
3. Istnieje `latest_checkpoint.pt` nowszy niż żądanie `STOP_TIME`.
4. Nie ma innego zadania o dokładnej nazwie `sd0k8000`.
5. Nie osiągnięto limitu segmentów, domyślnie i maksymalnie dziesięciu.

Nowe zadanie ma zależność `afterany:CURRENT_JOBID`, więc rusza po zakończeniu
obecnego. Zależność nie służy rozpoznaniu przyczyny stopu; tę kontrolują powyższe
warunki przed `sbatch`. Sukces, plateau, manualny STOP, cancel, błędy i smoke nie
zgłaszają kolejnego zadania. Przy limicie segmentów pozostaje kod 75 i checkpoint
do ręcznego wznowienia. Nowe ręczne `--submit` rozpoczyna nowy budżet segmentów.

`.submission.lock` serializuje sprawdzenie pełnej nazwy w `squeue` i zgłoszenie,
także przy kontynuacji. `.training.lock` blokuje równoległe wykonanie. Deployment
odmawia aktualizacji kodu, gdy zadanie jest w kolejce lub działa. Zgłoszenia są
dopisywane do `logi/submissions.log`. Używaj `deploy.py` zamiast bezpośredniego
`sbatch`, aby zachować blokadę ręcznych zgłoszeń.

## Podział odpowiedzialności i weryfikacja

Zweryfikowano lokalnie ścisłe wczytanie teachera i pełny warunek 600x768.
VAE ma 20945575 parametrów, a UNet z adapterem 115547524. Podana dostępna
przestrzeń zdalna wynosiła 44 GB w chwili kontroli; nie stanowi rezerwacji miejsca.

## Bramki jakości i zakres wniosków

Smoke wykonuje dwa kroki VAE i dwa kroki SD, omijając bramki jakości.
W treningu produkcyjnym bramki jakości i plateau wymagają pokrycia co najmniej
90 procent filmów TRAIN. Bramka VAE wymaga PSNR co najmniej 28 i SSIM co najmniej
0.85. VAE kończący etap z PSNR poniżej 24 lub SSIM poniżej 0.75 zatrzymuje cały
bieg jako wymagający review, bez rozpoczęcia SD.

Bramka SD wymaga względnego MSE latentu najwyżej 0.1, cosinusa co najmniej 0.9,
odchylenia standardowego obrazu powyżej 0.06 oraz błędu przy złym warunku
co najmniej o 10 procent większego. Plateau: VAE po co najmniej 5000 kroków
i 20 ocenach bez poprawy; SD po co najmniej 10000 kroków i 12 ocenach bez poprawy.

Checkpoint `best` jest wybierany według wyniku walidacji. Sam ten wybór nie
dowodzi generalizacji; wnioski dotyczą ocenionego splitu VAL. Kryteria oraz
raportowanie pokrycia implementuje `run.py` należący do agentów treningu/review.

Ten zakres obejmuje wyłącznie `deploy.py`, `trening.slurm`, `README.md`, `PLAN.md`
i `__init__.py`. Model, dane i trening należą do pozostałych agentów. Samo
przygotowanie tych plików nie oznacza wdrożenia, wykonania smoke ani zgłoszenia
zadania. Test integracyjny na czterech GPU wymaga osobnego jawnego uruchomienia
przez operatora.

## Próbkowanie od 2026-10-06
Na polecenie użytkownika trening kontynuuje z zachowaniem wag i optymalizatora,
przy --frame-stride 600: klatki 1, 601, 1201 itd. w każdym filmie TRAIN.
Walidacyjne klatki 33% i 67% pozostają niezmienione. Stride jest zapisany osobno
w provenance i checkpointach: to jawna zmiana polityki danych, nie architektury.
Dotychczasowe best/stale i pokrycie filmów są zachowane. Ponowne dekodowanie
pozostałych klatek MP4 nadal może być konieczne, ale RGB, tensory i obliczenia
sieci wykonujemy tylko dla wybranych klatek. Brak ekstrakcji na dysk.
