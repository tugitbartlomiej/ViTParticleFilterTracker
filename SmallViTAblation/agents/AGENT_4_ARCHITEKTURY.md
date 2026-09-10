AGENT 4 — przeniesienie na inne architektury
============================================

Przeczytaj najpierw `README.md` w tym katalogu. Ten plik zakłada, że go znasz.

**Twój katalog:** `pipeline/transfer/`
**Nie dotykasz:** niczego poza nim.

---

## Co budujesz i po co

Teza pracy brzmi: **dobór danych treningowych potrafi znaczyć więcej niż wybór
modelu**. Jeśli tak, to efekt musi być widoczny na różnych modelach, a nie tylko
na jednym. Recenzenci wytknęli to dwa razy, niezależnie.

Twoje zadanie: powtórzyć najważniejsze konfiguracje na **dwóch dodatkowych
architekturach**, przy dokładnie tym samym protokole.

## Architektury

| Model | Czym jest | Skąd |
|---|---|---|
| RF-DETR-Small | model główny, robi go Agent 2 | odniesienie |
| **D-FINE-S** | detektor transformerowy z ulepszoną regresją ramki; recenzent wskazał go z nazwy, bo w tym zadaniu liczy się dokładna lokalizacja małego obiektu | `https://github.com/Peterande/D-FINE`, ICLR 2025 |
| **bieżąca YOLO** | rodzina konwolucyjna; ma być wersja aktualna, nie YOLOv8 | wybierz wersję i **uzasadnij wybór w `README.md`** |

Recenzent napisał wprost: praca zgłoszona w 2026 roku porównuje YOLOv8
i oryginalny DETR, a od tego czasu powstały wyraźnie lepsze modele. YOLOv8
nie reprezentuje już swojej rodziny.

## Zadanie 1 — `pipeline/transfer/train_transfer.py`

Trening na D-FINE-S i na wybranej YOLO, przy **identycznym protokole** co model
główny.

**Iso-resource jest tu obowiązkowe** i to jest sedno zadania. Poprzednia wersja
pracy porównywała DETR chodzący na obrazach 800 na 1333 z YOLOv8 na 640 na 640.
Recenzent słusznie napisał, że takie porównanie mierzy **dwie recepty
treningowe, a nie dwie architektury**.

Co musi być identyczne:

1. **rozdzielczość wejścia** — ta sama liczba pikseli dla wszystkich trzech
   modeli, wartość z `protocol.yaml`,
2. **zbiór treningowy** — ten sam plik wariantu, co do klatki,
3. **budżet epok** — ta sama liczba, bez wcześniejszego zatrzymywania,
4. **podział danych** — ten sam plik,
5. **ziarno losowe**,
6. **ścieżka ewaluacji** — te same metryki, ten sam sposób wybierania progu.

Czego **nie** da się zrównać i co trzeba **wypisać w artykule wprost**: liczba
parametrów, optymalizator narzucony przez bibliotekę modelu, sposób
przypisywania obiektów do predykcji. Nie udawaj, że tych różnic nie ma —
opisz je w tabeli.

## Zadanie 2 — ustawienia NMS dla YOLO

Modele rodziny YOLO usuwają nadmiarowe wykrycia procedurą NMS (non-maximum
suppression, czyli tłumienie wykryć, które nakładają się na mocniejsze).
DETR i pochodne tego nie potrzebują.

Recenzent zauważył, że praca podkreśla tę różnicę, ale **nigdzie nie podaje
ustawień NMS ani nie sprawdza, jak wpływają na wynik**.

Zrób trzy rzeczy:

1. wypisz użyte ustawienia — próg IoU i próg ufności — do `run_manifest.json`,
2. policz wrażliwość: kilka wartości progu IoU, wpływ na precyzję i czułość,
3. przy zadaniu z jedną klasą i zwykle jednym narzędziem w kadrze to krótka
   analiza; **jeśli wychodzi, że NMS nie ma znaczenia, napisz to i pokaż
   liczby**, zamiast pomijać temat.

## Zadanie 3 — spójny zapis wyników

Wyniki z obu architektur muszą lądować w **dokładnie tym samym formacie**, co
wyniki modelu głównego, czyli `metrics.json` z `README.md`, sekcja 4.5.
Agent 5 składa z tego jedną tabelę i nie będzie tłumaczył formatów.

Pole `arch` w `run_manifest.json` rozróżnia architektury.

## Test końcowy

1. jeden bieg D-FINE-S na małym zbiorze — kończy się, zapisuje komplet plików,
2. jeden bieg YOLO na tym samym zbiorze i tej samej rozdzielczości,
3. porównanie `run_manifest.json` z obu i z modelu głównego: **rozdzielczość,
   liczba epok, ziarno i plik wariantu identyczne**,
4. analiza NMS na kilku progach → tabela.

## Skala

Dziesięć biegów: pięć konfiguracji razy dwie architektury, po jednym ziarnie.
To dokładka do głównego eksperymentu, więc nie potrzebuje wielu powtórzeń —
pokazuje kierunek, nie precyzyjną wartość.

Konfiguracje wskaże wcześniejszy etap. Zawsze będą wśród nich: pełny pipeline,
losowy wybór i losowy wybór z klatek źródłowych.
