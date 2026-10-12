# Koopman osobno per obraz i token

Pytanie, podział i matematyka: PLAN.md. Dane źródłowe pozostają w koopman_gesta_historia. Tutaj są wyłącznie kod i wyniki nowej analizy.

Uruchomienie: `python test_operator.py`, `python analiza.py --device cuda --batch 32`. Modele każdej pary obraz/token mają własne parametry. Obliczenia float64, bez PCA. Stan: zakończony. Wyniki: WYNIKI_TEST.md, DIAGNOZA.md i wyniki/REPORT.json.

Sprawdzenie operacyjne: `python test_operacyjne.py`. Odtworzenie kontrolnych wyników: `python weryfikacja_wynikow.py`. Eksport z kontrolą SHA256: `python eksport.py --image 66 --token 299 --stage final`.

Kod faktycznie użyty w biegu jest w wyniki/source_snapshot/. Po uruchomieniu dodano zabezpieczenia wznawiania, eksportu i statusu; nie zmieniono wzorów ani podziału. Ukończony raport jest chroniony przed nadpisaniem. Przepis odtworzenia modeli oceny używa pierwszych 48 par, a końcowe czynniki wszystkich 72 par. To różne dopasowania.
