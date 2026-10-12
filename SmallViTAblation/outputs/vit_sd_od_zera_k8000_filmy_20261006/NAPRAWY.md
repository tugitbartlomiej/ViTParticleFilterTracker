# Naprawy uruchomienia

- 2026-10-06: stdin Windows w trybie tekstowym zamieniał LF na CRLF; SSH uruchamia teraz bash ze stdin UTF-8 w trybie binarnym.
- Kontrola --help korzysta z tych samych katalogów bibliotek i ustawień offline co właściwy trening.
- Próba SLURM 1823263 została zatrzymana przez system przed startem skryptu: user_env_retrieval_failed_requeued_held. Anulowano tylko to zadanie. Ustawienia smoke/train i numer segmentu są teraz argumentami skryptu zamiast listy zmiennych --export. Nie wykonano żadnych aktualizacji modelu w tej próbie.
- Dokumentacja mechanizmu środowiska SLURM: https://slurm.schedmd.com/sbatch.html

## 2026-10-06: blokada produkcji 1823267
Bieg RUNNING przez około godzinę nie wykonał zapisanej aktualizacji. W folderze produkcyjnym była wyłącznie PROVENANCE.json, bez checkpointów. Smoke z workers=0 działał; produkcja z workers=2 uruchamiała domyślne fork po CUDA/NCCL. Proces rank0 czekał na futex, pracownicy odczytu danych nie wykazywali postępu. Hipoteza: niebezpieczne fork z NCCL (zgodne z ostrzeżeniem PyTorch DDP).
Poprawka: multiprocessing_context=spawn dla wieloprocesowego DataLoader; smoke również używa workers=2. Anulowano wyłącznie własny job 1823267, potwierdzono zniknięcie z kolejki. STOP powstały przez TERM podczas naprawy zachowano jako STOP_REPAIR_1823267. Nie istniał checkpoint do resume. Log zachowano lokalnie w DEADLOCK_1823267.log. Weryfikacja poprawki i nowy job zostaną dopisane po pomiarze.
Potwierdzenie: smoke 1823296 COMPLETED 0:0, 2 kroki VAE i 2 SD, workers=2 na każdym z 4 GPU, źródła zgodne SHA256. Produkcja zgłoszona jako 1823302. Smoke nie służy ocenie jakości i jego wag nie wczytano do produkcji.
Produkcja 1823302: potwierdzono STATUS stage=vae step=150 loss=0.17682336270809174 state=running. Poprawiony bieg wykonuje aktualizacje. Walidacja jakości jeszcze przed pierwszą oceną co 500 kroków.
Zmiana użytkownika: stride TRAIN=600, VAL bez zmian. Kontrolowany zapis przez STOP_TIME bez sygnału SLURM (bez automatycznej drugiej kopii); zapisany recovery SD step=18487 finished=false. Test samplera potwierdził klatki 1/601/1201 dla 1300 klatek i odrzucenie stride=0. Polityka danych ma osobne pole w checkpoint/provenance; parametry architektury, optymalizator, best i stale są zachowane.
Wdrożenie stride600 zakończyło weryfikację źródeł i CLI na Edenie. Zgłoszono pojedynczy job 1823430 z resume auto ze sprawnego checkpointa SD18487. Walidacja i kryteria early stop bez zmian. Poprzedni exit75 był kontrolowanym zapisem dla zmiany próbkowania, nie awarią.
