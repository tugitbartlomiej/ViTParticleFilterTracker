// Konfiguracja odtwarzacza punktów (odtwarzacz.html?punkty=punkty_modele.js).
// Czas: "m:ss.cc" (minuty:sekundy.setne), np. "1:05.40". Działa też "65.4" i "0:01:05.40".
// "film" to ścieżka względem odtwarzacz.html.
// Punkty startowe: nieruchome ujęcia po każdym etapie animacji (podane przez sceny, sprawdzone freezedetect).
window.KONFIG_PUNKTOW = {
  "film": "final_modele.mp4",
  "punkty": [
    { "czas": "0:14.10", "opis": "Model 1: sieć: ramka i strata po przejściu w przód" },
    { "czas": "0:18.37", "opis": "Model 1: gradient wstecz zmienił wagi" },
    { "czas": "0:21.53", "opis": "Model 1: latent po pierwszym kroku" },
    { "czas": "0:32.33", "opis": "Model 1: po 4 krokach: dobra ramka, rozdzielone chmury" },
    { "czas": "0:40.95", "opis": "Model 2: chmury narzędzia i tła, pierwsza prosta" },
    { "czas": "0:50.32", "opis": "Model 2: kierunek Fishera, duże d′" },
    { "czas": "1:02.19", "opis": "Model 2: krok 2000: kierunek stały słabnie" },
    { "czas": "1:10.69", "opis": "Model 2: krok 4500: cos 0,12, odświeżany trzyma ~7" },
    { "czas": "1:19.35", "opis": "Model 3: szum, cel poza obszarem osiągalnym" },
    { "czas": "1:27.88", "opis": "Model 3: obraz odszumiony, punkt na brzegu" },
    { "czas": "1:40.15", "opis": "Model 3: cztery szumy, ten sam punkt brzegu" },
    { "czas": "1:45.65", "opis": "Model 3: luka: granica 10,25 wobec celu 51,35" },
    { "czas": "1:54.14", "opis": "Model 4: sfera cech i wektor zapytania" },
    { "czas": "2:01.74", "opis": "Model 4: szeroki stożek, wiele trafień" },
    { "czas": "2:07.94", "opis": "Model 4: wąski stożek, pięć punktów" },
    { "czas": "2:13.34", "opis": "Model 4: znaleziona klatka, cos 0,875" },
    { "czas": "2:23.19", "opis": "Model 5: latent w dolinie przed paczką" },
    { "czas": "2:27.66", "opis": "Model 5: uderzenie paczki: pchnięcie P" },
    { "czas": "2:35.33", "opis": "Model 5: szczyt 3,2 w 5. kroku" },
    { "czas": "2:43.46", "opis": "Model 5: powrót: ~0,4 po 500 krokach" },
    { "czas": "2:51.79", "opis": "Model 5: druga paczka: nakładanie odpowiedzi" },
    { "czas": "3:00.85", "opis": "Model 6: trajektoria wag do kroku 200" },
    { "czas": "3:07.75", "opis": "Model 6: pełna trajektoria do kroku 1000" },
    { "czas": "3:12.45", "opis": "Model 6: średnia wag z okna 100 kroków" },
    { "czas": "3:23.85", "opis": "Model 6: mAP: średnia wag 0,416 wobec 0,337" },
    { "czas": "3:30.55", "opis": "Model 6: średnia jedzie spokojnym torem" }
  ]
};
