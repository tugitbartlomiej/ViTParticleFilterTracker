// Konfiguracja odtwarzacza punktów (odtwarzacz.html).
// Czas: "m:ss.cc" (minuty:sekundy.setne), np. "1:05.40". Działa też "65.4" i "0:01:05.40".
// "film" to ścieżka względem odtwarzacz.html.
// Plik można edytować ręcznie albo zapisać z edytora w odtwarzaczu (Ctrl+S).
// Punkty startowe wybrane z nieruchomych fragmentów filmu, po zakończeniu każdej animacji.
window.KONFIG_PUNKTOW = {
  "film": "final.mp4",
  "punkty": [
    { "czas": "0:05.62", "opis": "Krok 1: sonda, obraz SD, prawdziwa klatka" },
    { "czas": "0:11.82", "opis": "Krok 1: pytanie badawcze" },
    { "czas": "0:22.72", "opis": "Krok 2: siatka 600 tokenów" },
    { "czas": "0:31.89", "opis": "Krok 2: sieć i miejsce pomiaru latentu" },
    { "czas": "0:38.79", "opis": "Krok 2: tokeny jako cząstki" },
    { "czas": "0:49.62", "opis": "Krok 3: zdjęcie przed korektą" },
    { "czas": "0:54.62", "opis": "Krok 3: korekta 128 kanałów" },
    { "czas": "1:00.49", "opis": "Krok 3: po korekcie, IoU 99,93 %" },
    { "czas": "1:05.92", "opis": "Krok 3: AP50 0,44 na 0,88" },
    { "czas": "1:18.95", "opis": "Krok 4: pętla generatora SD" },
    { "czas": "1:35.12", "opis": "Krok 4: cel 51,35 poza zasięgiem 10,25" },
    { "czas": "1:40.49", "opis": "Krok 5: obraz SD jako zapytanie" },
    { "czas": "1:57.65", "opis": "Krok 5: najbliższa prawdziwa klatka" },
    { "czas": "2:05.29", "opis": "Krok 6: paczka 32 zdjęć i krok AdamW" },
    { "czas": "2:19.02", "opis": "Krok 6: pęd powtarza pchnięcie" },
    { "czas": "2:21.52", "opis": "Krok 6: szczyt 3,2 w 5. kroku" },
    { "czas": "2:23.82", "opis": "Krok 6: powrót do ~0,4" },
    { "czas": "2:27.55", "opis": "Krok 6: jedna paczka pcha mocno, ale krótko" },
    { "czas": "2:36.89", "opis": "Krok 7: pętla zamknięta" },
    { "czas": "2:48.59", "opis": "Krok 7: test rozstrzygający" }
  ]
};
