AGENT 2 — trening i kolejka na klastrze
=======================================

Przeczytaj najpierw `README.md` w tym katalogu. Ten plik zakłada, że go znasz.

**Twój katalog:** `pipeline/train/` i `eden/`
**Nie dotykasz:** niczego poza nimi.

---

## Co budujesz i po co

Dwie rzeczy. Pierwsza: skrypt, który trenuje **jeden** bieg dokładnie według
protokołu. Druga: mechanizm, który przepycha przez klaster **około stu
siedemdziesięciu takich biegów** bez ręcznego doglądania.

Klaster nazywa się Eden, ma węzeł z ośmioma kartami H100 i twarde ograniczenia
opisane w `Eden/EDEN_CONSTRAINTS.md` w korzeniu projektu. **Przeczytaj ten plik
przed pisaniem czegokolwiek pod SLURM.**

## Zadanie 1 — `pipeline/train/train_rfdetr.py`

Trening jednego biegu. Model: RF-DETR, biblioteka `rfdetr` od Roboflow,
warianty Nano/Small/Medium/Large na licencji Apache 2.0.

Wejście z wiersza poleceń: `--run-id`, `--variant-file`, `--protocol`,
`--data-root`, `--out`.

Wszystkie hiperparametry **czytane z `protocol.yaml`**, sekcja `training`.
Zero wartości zaszytych w kodzie.

Wymagania twarde, każde wynika z zarzutu recenzenta:

1. **Stała liczba epok. Zero wcześniejszego zatrzymywania.** Nawet gdy krzywa
   walidacyjna stoi. Wcześniej każdy wariant kończył w innej epoce, więc
   dostawał inny budżet obliczeń i porównanie mierzyło dwie rzeczy naraz.
2. **Start zawsze z tych samych wag COCO.** Nigdy z checkpointu innego biegu.
3. **Zero augmentacji w trakcie treningu**, chyba że `protocol.yaml` mówi
   inaczej. Pula bywa już powiększona wcześniej.
4. **Zapis dwóch checkpointów**: ostatnia epoka oraz najlepszy na walidacji wg
   jednego, zapisanego kryterium (`checkpoint_selection.best_val_criterion`).
   Obie kolumny idą potem do tabel i nigdy się ich nie miesza.
5. **Ziarno losowe steruje** inicjalizacją głowy, kolejnością danych i losowością
   w treningu. Ustaw je we wszystkich bibliotekach, także w `numpy` i `random`.
6. **Na koniec zapisz `run_manifest.json`** w formacie z `README.md`, sekcja 4.4:
   hash commita, suma kontrolna `protocol.yaml`, suma kontrolna pliku wariantu,
   wersje bibliotek, sprzęt, czas. To jest wymóg, nie ozdoba — poprzednio nie
   dało się ustalić, czym był jeden z wariantów, i sprawa wisiała miesiącami.
7. **Po biegu zostaw tylko** `last.pth`, `best.pth`, `run_manifest.json` i logi.
   Checkpointy pośrednie skasuj; przy stu siedemdziesięciu biegach to terabajty.

Skrypt **nie liczy metryk** i **nie dotyka zbioru zewnętrznego** — to należy do
Agenta 3. Twój skrypt kończy się na wytrenowanym modelu.

## Zadanie 2 — `eden/queue.py`

Kolejka biegów oparta na pliku, bo osiem procesów na jednym węźle musi z niej
brać zadania bez deptania sobie po palcach.

- stany: `pending`, `running`, `done`, `failed`,
- pobranie zadania **atomowe**, pod blokadą pliku,
- bieg w stanie `running` bez żywego procesu przez ustalony czas wraca do
  `pending` (zadanie ubite limitem czasu SLURM),
- **wznawialność**: ponowne uruchomienie podejmuje kolejkę tam, gdzie była,
- błąd jednego biegu **nie zatrzymuje reszty** — zapisz `failed` z logiem i idź
  dalej.

## Zadanie 3 — `eden/runner.py`

Pracownik. W pętli: weź bieg z kolejki, uruchom trening, oznacz wynik, weź
następny. Jeden proces na jedną kartę.

Argumenty: `--queue`, `--data`, `--worker`, `--protocol`.

## Zadanie 4 — `eden/run_node.slurm`

Jedno zadanie SLURM zajmuje **cały węzeł** i uruchamia **osiem pracowników**,
po jednym na kartę, wszystkich z tej samej kolejki.

Punkty, na których łatwo polec, wszystkie opisane w `EDEN_CONSTRAINTS.md`:

- partycja `hopper` (osiem kart H100, limit pięciu dni). Partycja `debug` jest
  **zablokowana przez administratora** — nie używać.
- `/tmp` na węźle jest **za mały**. Dane rozpakowuj na `/raid` (czternaście
  terabajtów) albo `/dev/shm`. Wzorzec wyboru katalogu jest w tamtym pliku.
- **rozpakowanie danych raz na węzeł**, nie raz na bieg. To najczęstsza strata
  czasu na klastrze.
- `trap cleanup EXIT INT TERM`, żeby po sobie posprzątać.

## Zadanie 5 — `eden/submit_chain.sh`

Łańcuch zadań przez `--dependency=afterany`. Potrzebny, bo przy pesymistycznym
wariancie całość nie zmieści się w limicie pięciu dni na jedno zadanie.

## Test końcowy

1. jeden bieg na małym modelu (RF-DETR-Nano przy 384), 2 500 obrazów, 5 epok —
   kończy się i zapisuje komplet plików,
2. kolejka z ośmioma biegami i dwoma pracownikami — żaden bieg nie wykonany
   dwa razy, żaden pominięty,
3. **ubicie pracownika w połowie** — po restarcie bieg wraca do kolejki i się
   dokańcza,
4. bieg, który celowo się wywala — oznaczony `failed`, reszta idzie dalej.

Punkt trzeci jest najważniejszy. Zadanie ubite limitem czasu to normalna
sytuacja, nie awaria.

## Kontekst liczbowy

- około **174 biegi**, głównie RF-DETR-Small przy wejściu 512,
- pojedynczy bieg przy budżecie pięciu tysięcy obrazów i sześćdziesięciu epokach,
- szacunek całości: od 250 do 1000 godzin karty, czyli od półtora do pięciu dób
  na ośmiu kartach; dokładna liczba po biegu kalibracyjnym,
- dane wjeżdżają na klaster jako jedna paczka **około siedmiu gigabajtów**:
  suma wszystkich podzbiorów plus walidacja i test. W domu na klastrze jest
  **sześćdziesiąt pięć gigabajtów wolnego**, więc mieści się z zapasem.

Narzędzie `tools/build_run_manifest.py` w korzeniu projektu generuje kolejkę
i już działa. Nie pisz go od nowa.
