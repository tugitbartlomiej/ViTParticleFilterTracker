# Dolina odpowiedzi cząsteczki 226

Wizualizacja dotyczy **zbioru położeń dających określoną odpowiedź zamrożonej głowy**, a nie ogniska laserów MLP. Badany jest stan tokenu po ostatnim bloku, przed końcową LayerNorm: 192 współrzędne. Ukryta warstwa MLP ma tutaj osobne 768 aktywacji. Wagi: bieg s47, checkpoint 65640, zdjęcie z historii `eafb5a973be3d490`.

## Co oznacza 187

Głowa zwraca pięć liczb: logit środka narzędzia i cztery logarytmy odległości do krawędzi ramki. Lokalny jakobian ma rząd pięć. Zatem lokalnie zbiór tej samej skończonej pewności i tej samej ramki ma wymiar 192−5=187. To 187 niezależnych kierunków stycznych, z których można tworzyć dowolne kombinacje. Nie jest to liczba odpowiedzi detektora ani 187 konkretnych osi oryginalnego układu. Wektory są kombinacjami jego współrzędnych.

Proste w tych kierunkach zachowują odpowiedź tylko w pierwszym przybliżeniu. Żeby wykonać większy ruch po zbiorze identycznej odpowiedzi, trzeba uwzględniać jego krzywiznę. Układ kierunków zależy od położenia i wag głowy. W tym przykładzie dwa prawie płaskie kierunki wiążą się z niezmienniczościami LayerNorm; do filmu wybrano dwa kierunki o największej bezwzględnej krzywiźnie lokalnej.

## Dwie różne geometrie na ekranie

1. **Położenia w latencie:** wysokość jest rzutem położenia na wybraną normalną n. Błękitna zakrzywiona powierzchnia to rzeczywiście wyznaczone położenia w 192D. Wszystkie mają pewność 0,995 i ramkę zgodną z oznaczeniem. Geometria jest rzutem dwuwymiarowego wycinka 187-wymiarowego zbioru, nie obrazem całego latentu.
2. **Krajobraz niezgodności odpowiedzi:** wysokość oznacza E, jawną miarę niezgodności. Wzdłuż dna odpowiedź się nie zmienia, w poprzek pojawia się błąd. E to suma kwadratu błędu logitu i kwadratów względnych błędów odległości ltrb. Nie jest to oryginalna strata treningowa ani potencjał pola laserów.

Nie wolno odczytywać kształtu 2D wycinka jako globalnej geometrii całego zbioru. Nie wyznaczano wszystkich jego składowych ani globalnej topologii.

## Identyczna i wystarczająco dobra odpowiedź

**Identyczna odpowiedź** oznacza dokładnie określoną pewność oraz cztery odległości ramki. **Wystarczająco dobra odpowiedź** oznacza nierówności, tutaj przykładowo p≥0,99 i IoU≥0,90. Dopuszczenie tolerancji daje obszar o grubości również w kierunkach odczytu. Nie należy całego takiego obszaru nazywać 187-wymiarowym.

Złota chmura to tylko próbki rodziny x(u,v,δ)=x(u,v)+δn, ograniczonej zakresem badania. Brak złotej próbki nie dowodzi braku rozwiązania poza badaną rodziną lub zakresem. W HTML można zmieniać oba progi. Wysoka pewność sama nie gwarantuje dobrej ramki: przy dodatnim odsunięciu o 12 pewność wynosi około 99,84%, ale IoU spada do 0,8173.

Pewność jest wynikiem sigmoidu logitu mapy środka narzędzia. Nie należy utożsamiać jej z udowodnionym, skalibrowanym prawdopodobieństwem poprawności detekcji lub rozpoznania typu narzędzia.

## Jak skonstruowano powierzchnię

- Punktem startu jest zapisany stan tokenu 226 z ostatniego checkpointu. Jego odpowiedź: p≈0,94945, IoU≈0,88580. Historyczny zapis stanów był kwantowany do float16; kontrola porównuje go z ponownym przejściem rzeczywistego modelu.
- Znaleziono **przykładowe** położenie x* o p=0,995 oraz ramce GT. Odległość od stanu startowego: 21,17893, czyli 37,44% jego normy. Nie dowodzono minimalności przesunięcia ani osiągalności x* przez pełny ViT dla tego obrazu.
- Ze SVD jakobianu 5×192 otrzymano pięć kierunków normalnych Q i 187 wektorów bazy jądra. Dwa kierunki t1,t2 wybrano według krzywizny mierzonej hesjanem. Ten wybór ma pokazać krzywiznę; nie jest losową reprezentatywną próbką wszystkich kierunków.
- Rozwiązano pięć równań head(x)=cel dla x=x*+u t1+v t2+Qc(u,v), aktualizując pięć współrzędnych c. Zakres u,v wynosi ±3,5. Większy zakres ±5 nie zbiegał we wszystkich punktach, dlatego nie jest pokazywany. To ograniczenie lokalnej parametryzacji/algorytmu, a nie dowód nieistnienia równoważnych położeń dalej.
- Zapisano 1089 stanów powierzchni i 361 stanów ścieżki w pełnych 192 współrzędnych. Maksymalna reszta pięciu równań na powierzchni: około 1,8×10⁻¹⁴; na ścieżce około 8,0×10⁻¹⁴, w arytmetyce float64. Rzeczywisty trening używa innej precyzji; te liczby opisują dokładność rozwiązania zamrożonej funkcji, a nie dokładność modelu na zbiorze danych.

## Pliki

- `dolina_odpowiedzi_t226.mp4`: 50 sekund, 1920×1080, 30 klatek/s, bez lektora. Fazy: 0–6 s powierzchnia; 6–18 s ruch po niej; 18–29 s ruch poprzeczny; 29–38 s obszar jakości; 38–50 s dolina niezgodności E.
- `dolina_interaktywna.html`: samodzielny plik offline. Obracanie bryły, ruch cząsteczki, progi jakości, wybór wszystkich 187 kierunków i ich składowe w 192D, krajobraz E. Chmura w HTML jest zmniejszona do 9009 rzeczywiście policzonych próbek. Filtracja używa ich zapisanych miar, a nie przybliżenia z samego rysunku.
- `01_...png` do `05_...png`: plansze z kolejnych faz.
- `geometria.npz`: pełne stany powierzchni i ścieżki, baza N (192×187), Q, jakobian J, normalna, wartości wyjścia, siatki próbek i metryki.
- `POMIAR.json`: parametry, wyniki, suma SHA256 checkpointu i reszty numeryczne.
- `WERYFIKACJA.json`: niezależna kontrola na oryginalnych modułach modelu i pełne dekodowanie MP4.

Źródła i komendy: `experiments/dolina_odpowiedzi_t226_20261015/`. Stare filmy i aplikacja pozostają punktem odniesienia; nowy film zapisano osobno.
