# Wynik pomiaru ruchu tokenów narzędzia

Data: 2026-09-30. Analiza lokalna istniejących wag. Nie uruchamiano treningu.

## Pytanie i aparat

Czy maska od jednego piksela i osobne trajektorie tokenów ujawniają kierunki ruchu odróżniające narzędzie od tła?

Odczytano 2 stanów sieci, kroki: [60, 120]. Pełny zapis ma kształt [2, 12, 600, 768]: czas, obraz, token, cecha.

600 tokenów na pełną klatkę 480 × 320, 768 cech na token. Nie uśredniano tokenów ani punktów czasu.

Dopasowanie: 8 filmów. Ocena: 4 innych filmów; te same przedziały uczenia, bez testu przyszłego czasu.

## Maska i kierunki

Próg choć jednego piksela: 7–31 tokenów na obraz, mediana 18.5.

Próg 25 procent: 2–27 tokenów, mediana 9.5.

Do 90 procent energii skorygowanego ruchu na obrazach uczących potrzeba 16 osi SVD. Nie jest to wymiar tożsamości narzędzia.

Zapisano pełną bazę 768 osi w wyniki/KIERUNKI_RUCHU.npz. Kolumny directions są kierunkami w surowym układzie 768 aktywacji tej sieci. Zapis mean_direction zawiera kierunek średniego ruchu względem tła.

Z ruchu odejmowano wspólny dryf. Na testowych obrazach referencja tła pochodzi wyłącznie z obrazów uczących, bez masek testowych w obliczaniu wyników tokenów.

## Ocena poza dopasowaniem i podłogi

ROC AUC mierzy rozdzielenie ruchów tokenów maski i odległego tła. Wartość 0,5 odpowiada brakowi uporządkowania. Wyniki są średnią po przedziałach i obrazach, nie odsetkiem poprawnie wykrytych narzędzi.

Norma całego ruchu: AUC 0.5063.

Norma po odjęciu referencyjnego dryfu: AUC 0.5665.

Podpisany rzut na średni kierunek: AUC 0.6689.

1 kierunków ruchu: AUC 0.4744; losowa podprzestrzeń tej samej wielkości: 0.6016.

2 kierunków ruchu: AUC 0.5466; losowa podprzestrzeń tej samej wielkości: 0.5489.

4 kierunków ruchu: AUC 0.5673; losowa podprzestrzeń tej samej wielkości: 0.5391.

8 kierunków ruchu: AUC 0.6071; losowa podprzestrzeń tej samej wielkości: 0.5606.

16 kierunków ruchu: AUC 0.6106; losowa podprzestrzeń tej samej wielkości: 0.6082.

32 kierunków ruchu: AUC 0.6064; losowa podprzestrzeń tej samej wielkości: 0.5730.

## Zastrzeżenia i czego wynik nie dowodzi

Nie wybieramy najlepszego wymiaru na teście jako potwierdzonego hiperparametru. Pokazujemy wszystkie zaplanowane wymiary.

To jeden przebieg sieci i niewielki zestaw filmów. Tokeny tego samego obrazu i kolejne przedziały nie są niezależnymi próbami.

Próg jednego piksela obejmuje tokeny z bardzo małym udziałem narzędzia; nie czyni ich czystymi reprezentacjami narzędzia. Tło kontrolne jest oddalone od maski, więc zadanie nie obejmuje wszystkich trudnych tokenów brzegowych.

SVD znajduje kierunki energii zmian. Wynik nie wykazuje przyczynowego wykorzystania ich przez detektor, odróżniania cystotomu od innych narzędzi ani poprawy uczenia.

Nie odtworzono brakujących checkpointów i nie rekonstruowano cząstek ze średnich. Źródła wag i odstępy należy odczytywać z wyniki/dane/metadata.json.

Prognoza Koopmana i Langevina nie była przedmiotem tego biegu: kierunki ruchu nie są jeszcze modelem przyszłej dynamiki. Dostępność gęstej historii rozstrzyga manifest pomiaru.

## Weryfikacja

Sprawdzono kształty danych, rosnące kroki, rozdział filmów, rozłączność maski i tła, ortonormalność kierunków. Syntetyczna kontrola potwierdza dzielenie przez rzeczywiste odstępy kroków i usuwanie wspólnego dryfu bez mieszania tokenów.

Sumy SHA256 źródłowych danych znajdują się w wyniki/REPORT.json. Wykres: wyniki/RUCH_TOKENOW.png.