# Przegląd protokołu i implementacji

Data: 2026-10-01. Przegląd wykonał agent Astra, tylko do odczytu, przed oceną danych testowych. To audyt matematyczny i kodu, nie formalny dowód ani niezależna reprodukcja wyniku.

## Protokół

- Walidacja z początkiem 500 i horyzontem 100 kończy się na 600. Pierwszy test zaczyna się od znanego stanu 600 i ocenia cele od 610. Ten podział jest przyczynowo poprawny, jeżeli hiperparametry są zamrożone przed testem.
- Z 41 stanów model wykorzystujący poprzedni przyrost otrzymuje 39 wyrównanych par. Model bez pamięci używa tych samych par.
- Rank ograniczony liczbą próbek dotyczy korekty B, a nie całego operatora I+B.
- Baza PCA, centrowanie i operator pozostają stałe podczas rolloutu. Horyzonty są przeliczone na 1/2/5/10 obserwacji.
- Rekonstrukcja `z_t+U(xhat−x_t)` zachowuje ortogonalną część stanu początkowego. Podłoga nieodtworzonego ruchu poza U jest oceniana osobno.
- PCA i metryki mają zgodne wagi: obraz 1/12, pokrycie maską w obrębie obrazu. Tokeny nie są niezależnymi replikacjami.

## Implementacja

W read-only przeglądzie `prognoza.py`, `dane.py` i `pilot_zgodnosci.py` sprawdzono indeksy, dual ridge, trace-normalizację kary, zakres historii PCA, autonomiczny rollout i rekonstrukcję.

Znaleziono jedną rozbieżność redakcyjną: plan początkowo mówił o średniej norm, podczas gdy kod i wybór ustawień używały RMS norm. **Przed oceną testu** zapisano RMS jako jawną główną metrykę w PLAN i CONFIG, a zwykłą średnią norm dodano jako wynik dodatkowy. Nie wybierano tej zmiany na podstawie wyników danych.

Nie wykryto krytycznego błędu w pozostałym audytowanym zakresie. Brak jawnego odejmowania średniej przed projekcją w rollout nie zmienia tutaj wyniku: regresja afiniczna centruje wejścia, a rekonstrukcja wykorzystuje różnicę współrzędnych.

## Wykonane małe kontrole kodu

Funkcja `validate_math()` sprawdza niezależne własności:

1. Rekonstrukcja zachowuje składową ortogonalną, a kwadrat błędu rozkłada się zgodnie z twierdzeniem Pitagorasa.
2. Stały ruch liniowy pełnych wektorów jest odtwarzany przez intercept modelu z pamięcią i bez pamięci po 10 krokach autonomicznych. To kontrola wyrównania celu i aktualizacji rolloutu.

Obie kontrole przeszły lokalnie przed pobraniem całego pliku. Nie używały rzeczywistych przyszłych tokenów do dostrajania modelu.

Ograniczenia pozostają: jeden przebieg uczenia, mało przejść względem 768 wymiarów, odrębne predyktory tokenów, wspólna baza bez jawnego modelu oddziaływań, brak zapisanego sterowania minibatchami tego biegu. Brak przewagi predykcyjnej jest pełnoprawnym wynikiem pilota.
