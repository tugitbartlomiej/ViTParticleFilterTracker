# Widoki przestrzeni cząsteczek

Wizualizacja: http://127.0.0.1:8774/

Dodano przełącznik osi z nazwami i liczbową podziałką oraz sześć widoków:

1. Wspólna projekcja PCA 3D.
2. Płaszczyzna PC1–PC2.
3. Płaszczyzna PC1–PC3.
4. Płaszczyzna PC2–PC3.
5. Kolumna patcha, wiersz patcha i norma ruchu w pełnych 768 wymiarach. Ten widok nie używa PCA. Osie mają niezależne skale; mapa nie przedstawia odległości między latentami.
6. Trzy dowolne oryginalne kanały 0–767 wybranego obrazu. Pokazuje wszystkie jego 600 cząsteczek. Profil wybranego patcha przedstawia wszystkie 768 wartości: start, aktualny stan i różnicę.

PCA i mapa ruchu korzystają z gęstego zapisu 0–500 co 5 kroków. Pełne oryginalne latenty mają tylko stany 0, 100, 500; widok surowych kanałów nie dorabia pośrednich stanów. Granice osi dla wybranego obrazu pozostają stałe między tymi trzema stanami. Nie stanowią globalnej skali wszystkich obrazów.

Pełne dane są czytane na żądanie z Edena, sprawdzane według SHA256 i przechowywane tylko w RAM. Cache obejmuje maksymalnie dwa źródła. Obrazy źródłowe są czytane z istniejących plików. Nie uruchomiono nowego treningu.

Weryfikacja: 106 testów backendu, sprawdzenie składni JavaScript i działanie w przeglądarce. Widok GPU renderował 900000 punktów bez błędu WebGL. Sprawdzono surowe kanały 100, 400, 767 dla obrazu 1469, patcha 9 i stanu 500; profil zawierał 768 kanałów. SHA256 źródła: 5fb3747725cfe49523d8a02dd8c581b045a4199afc98ee3d1413838a6b68e013.

Odtwarzanie zmian HTML: upgrade_space.py instaluje rozszerzenie jednorazowo; sync_space.py odświeża je po zmianach space-inline.js. Po użyciu sync_selection.py należy również wykonać sync_space.py.
