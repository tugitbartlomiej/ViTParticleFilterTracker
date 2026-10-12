# Poprawka infrastruktury po jobie1819498

JobFAILED na dgx-4,15:19:15–15:19:16,0kroków NN. Błąd: mkdir pod /raid: Permission denied. Przyczyna: poprzedni launcher sprawdził pojemność, ale nie rzeczywiste prawa utworzenia własnego katalogu. Stary launcher, hashe, status i logi zachowane w failure_1819498/.

Nowy launcher sprawdza dozwolone SLURM_TMPDIR/TMPDIR, istniejące katalogi użytkownika na /raid, /raid i /dev/shm. Odrzuca home/sharednetwork; tworzy wyłącznie nowy własny katalog (mkdir bez -p), sprawdza zapis i marker właściciela. Nie zmienia praw ani katalogów innych użytkowników. W cleanup wymaga resolved exactpath, basename zawierającego jobID i zgodnego markeru.

Próg full101 >350000000KB, rolling13 >50000000KB. Tmpfs zawsze rolling13, z minimum64GB przydzielonego RAM; job ma128GB. TypFS, wolna pojemność, RAM i wybrany path trafiają do logu.

Testy selektora na Edenie PASS: odtworzone niepowodzenie mkdir we własnym readonlyparent, następnie poprawny mkdir/writefallback; tmpfs128GB nie wybiera full101; tmpfs32GB odrzucony. Progi wolnej pojemności i typFS w jednostkowym teście były stubowane; rzeczywisty job sprawdza df/stat i zapis na przydzielonym węźle. Bash -n PASS.

Wszystkie pierwotne hashe poza launcherem pozostają identyczne; NN/config/dane/wybór bez zmian. Nowa integralność: FROZEN_INFRA_RETRY.sha256.
