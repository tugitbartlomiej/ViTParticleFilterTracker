# Zdjęcia i cząsteczki w podglądzie

Podgląd: http://127.0.0.1:8774/ . Zmiana z 2026-10-09.

- Przyciski „Długość ruchu”, „Według zdjęcia” i „Według patcha” zmieniają kolorowanie. Kolory identyfikatorów są stałe między krokami. Przy 1500 zdjęciach bliskie kolory mogą wyglądać podobnie; o tożsamości rozstrzyga numer zdjęcia, nie sam kolor.
- Zdjęcie wybiera się z listy, wyszukuje po numerze/filmie/nazwie albo przegląda przyciskami „Poprzednie” i „Następne”. „Pokaż tylko to zdjęcie” przełącza na jego 600 cząsteczek bez resetowania wybranego zapisanego kroku.
- W całej chmurze wyróżnione jest 600 cząsteczek wybranego zdjęcia. Opcję wyróżnienia można wyłączyć.
- Patch wybiera się kliknięciem zdjęcia lub wpisaniem numeru 0–599. Kliknięcie punktu w chmurze wybiera odpowiadające mu zdjęcie i patch. Biała ramka oraz biały punkt oznaczają tę samą parę.
- Siatka ma 20 wierszy i 30 kolumn. Patch `p` odpowiada wierszowi `p // 30` i kolumnie `p % 30`; globalny indeks to `image * 600 + p`.
- Zdjęcie jest oryginalnym JPEG-em odczytanym z istniejącego zbioru po sprawdzeniu SHA256. Nie tworzy się dodatkowych kopii zdjęć. Rysowana jest siatka i zielona ramka etykiety, nie wynik nowej predykcji.
- Osobny wykres przedstawia zapisane confidence i IoU wybranego zdjęcia. Dotychczasowy wykres średniej pozostaje osobny. Obraz źródłowy nie zmienia pikseli podczas odtwarzania sztucznej optymalizacji.
- Widok pojedynczego zdjęcia obsługuje rzeczywiste gęste stany 0–500 co 5 kroków, wycięte z tej samej zweryfikowanej chmury. Zachowuje wspólną bazę PCA i zakres osi.
- W panelu kontrolnym zapisanym co krok podgląd pełnej siatki używa najbliższego wcześniejszego zapisu wszystkich 600 patchy i jawnie podaje jego krok. Nie generuje brakujących latentów.

Weryfikacja: 66 testów backendu; rzeczywiste zdjęcia dostępne dla wszystkich 1500 rekordów. W przeglądarce na RTX 3070 sprawdzono shader dla 900000 punktów, wszystkie tryby koloru, ostatnią parę (1499,599) = indeks 899999, wybór siatki oraz rzeczywisty stan 500 pojedynczego zdjęcia.

Kod: `serve.py`, `cloud-webgl.js`, `selection-inline.js`, `upgrade_selection.py` i `sync_selection.py` w `experiments/vit_k1000_taniec_czasteczek_20261009/`. Generator pierwotnej strony należy uruchamiać przed `upgrade_selection.py`; po zmianie samego inspektora można użyć `sync_selection.py`.
