# Plan: losowy VAE i dyfuzja z zamrożonym teacherem K8000

## Cel i granice

Sprawdzić uczenie rekonstrukcji obrazów z pełnego warunku ViT na filmach online,
gdy VAE, UNet i adapter zaczynają z losowych wag. Zamrożony teacher to zweryfikowany
K8000 s47; jego wagi stanowią jedyny wyuczony punkt startowy. Nie dziedziczymy
wag SD/VAE poprzedniego eksperymentu. Wznawiamy wyłącznie własne checkpointy.

Kod modelu i treningu powstaje równolegle u pozostałych agentów. Ten plan oraz
deployment nie zastępują ich implementacji. Zakres własności: `deploy.py`,
`trening.slurm`, `README.md`, `PLAN.md`, `__init__.py`.

## Dane i kontrola pochodzenia

Zachować bajtowo manifest `wejscia/FILMY.json` poprzedniego eksperymentu
`2026-10-06_vit_sd_pelny_latent_filmy`, a więc ten sam split filmów TRAIN/VAL.
Nie tworzyć cache klatek ani latentów na dysku. Nie kopiować filmów do nowego
roota. Teacher lokalny i zdalny muszą mieć SHA256
`0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b`.
Zachować manifest, sygnaturę konfiguracji, checkpointy, metryki i logi.

## Kolejność uruchomienia

1. Dokończyć kod wszystkich agentów; sprawdzić składnię i kontrakt CLI/sygnałów.
2. Operator wykonuje `deploy.py --deploy` do nowego roota
   `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_od_zera_k8000_filmy`.
3. Operator wykonuje `--smoke`: cztery GPU, dwa kroki VAE i dwa kroki SD,
   ocena co krok, batch jeden, workers zero, osobny output.
4. Sprawdzić skończone straty, ocenę, checkpoint i brak zapisanych obrazów/latentów;
   sprawdzić, że teacher jest zamrożony i modele generatywne losowo inicjalizowane.
5. Dopiero po poprawnym smoke operator wykonuje `--submit`: 30000 kroków VAE,
   100000 kroków SD, ocena co 500, batch cztery, workers dwa, `--resume auto`.

Przygotowanie plików nie uruchamia żadnego z tych zadań. Zasoby: cztery Hopper,
konto `transformers_vsc`, 160 GB RAM, 24 CPU, 23 godziny 30 minut na segment.
Środowisko i zależności pozostają jak w starym eksperymencie.
Smoke ma osobny limit 30 minut ustawiany podczas zgłoszenia.

## Kryteria operacyjne i stop

Trening publikuje `latest_checkpoint.pt` atomowo i synchronizuje stop ranków.
Żądanie `STOP_TIME` jest rozpoznawane przez `run.py`; przerwanie czasowe zwraca 75.
Ukończenie SD lub plateau prowadzące do zakończenia zwraca zero. Manualny STOP
nie jest przyczyną kontynuacji. Wrapper zachowuje rozróżnienie kodów torchrun.

Kontynuacja wymaga sygnału czasu, kodu 75 i świeżego checkpointu. Zgłoszenie jest
chronione wspólnym flock i sprawdzeniem dokładnej nazwy `sd0k8000` w `squeue`.
Następny segment ma zależność od obecnego; łańcuch ma najwyżej dziesięć segmentów.
Plateau, sukces, manualny stop, cancel lub błąd nie uruchamiają kontynuacji.
Checkpointy pozostają także po wyczerpaniu budżetu.

## Ocena i ograniczenia

W produkcji decyzje jakościowe i plateau wymagają pokrycia co najmniej
90 procent filmów TRAIN. Smoke wykonuje 2+2 aktualizacje bez tych bramek.
VAE osiąga bramkę przy PSNR co najmniej 28 i SSIM co najmniej 0.85. Gdy końcowa
ocena VAE ma PSNR poniżej 24 lub SSIM poniżej 0.75, bieg wymaga review i nie
rozpoczyna SD. Plateau VAE: 20 ocen bez poprawy po co najmniej 5000 kroków.

Bramka SD: względne MSE latentu najwyżej 0.1, cosine co najmniej 0.9,
odchylenie standardowe obrazu powyżej 0.06 i błąd złego warunku co najmniej
o 10 procent większy. Plateau SD: 12 ocen bez poprawy po co najmniej 10000 kroków.
Wybór `best` jest wyborem według walidacji i nie stanowi dowodu generalizacji
poza oceniony split VAL. Implementacja tych reguł należy do treningu/review.

Lokalne wczytanie teachera `strict=True` i kształt 600x768 zostały zweryfikowane.
VAE ma 20945575 parametrów; SD z adapterem 115547524. Kontrola miejsca wskazała
44 GB wolnej przestrzeni; checkpointy wymagają pozostawienia miejsca na atomowy
zapis także podczas wznowienia.

Porównanie z poprzednim SD wymaga tego samego held-out splitu i zgodnych miar.
Rozdzielić jakość rekonstrukcji VAE od zgodności obrazu z warunkiem ViT. Niska
strata odszumiania sama nie dowodzi jakości rekonstrukcji ani użycia warunku.
Sprawdzić kontrolę błędnego warunku i niezerową różnorodność obrazu, według
metryk udostępnionych przez `run.py`.

Losowa inicjalizacja nie gwarantuje zbieżności w tym budżecie. Za niepowodzenie
operacyjne uznać niefinitywne straty, awarie ranków, brak poprawnego checkpointu
lub niezgodne pochodzenie wag. Wyniki jakościowe wymagają wykonania smoke i
treningu na Eden; lokalna kontrola deploymentu ich nie zastępuje.
