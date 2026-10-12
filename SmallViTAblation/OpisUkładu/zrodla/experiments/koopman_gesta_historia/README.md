# Gęstsza historia: osiem epok

Uruchomiony na zlecenie użytkownika wariant poprzedniego pilota.
Konfiguracja: `config.json`; metodologia i ograniczenia: `PLAN.md`.
Eden: zadanie 1810662, partycja short, początkowy węzeł dgx-4.
Dokładna lokalizacja zdalnego wyniku: `EDEN_RUN.json`.

`run_eden.slurm` uruchamia tylko pretrening przez `pretrain.py`.
Kod treningu, danych, cząstek, deterministyki i granic epok skopiowano bez zmian
z `../koopman_fastgan_petla`; ich sumy SHA256 zapisuje INPUTS.json każdego biegu.
Początek to wagi COCO. Zapis co pięć aktualizacji, osiem epok po sto aktualizacji.
Walidacja co epokę; brak osobnego testu w dostarczonej puli.
Koopman, FastGAN i wyszukiwanie pozostają lokalne i nie uruchamiają się w tym zadaniu.

`archive_old.py` pobiera wyłącznie pth i ckpt poprzedniego pretrain 1810448,
sprawdza sumy lokalne i zdalne, następnie usuwa tylko te zdalne pliki.
Manifest, potwierdzenie kopii i potwierdzenie usunięcia trafiają do `backup_1810448/`.
Żadne inne zadania ani modele nie są zatrzymywane ani usuwane.


Potwierdzono: krok 10, zrzuty co 5 aktualizacji; test powtarzalności latentu: błąd 0.0. Pierwsza próba 1810641 przerwała się przed aktualizacją z powodu brakującego modułu sonda.py; pakiet poprawiono. Testy danych, granic i historii: 8 zaliczonych. Usunięto 29 starych zdalnych plików wag po sprawdzeniu lokalnych SHA256.
