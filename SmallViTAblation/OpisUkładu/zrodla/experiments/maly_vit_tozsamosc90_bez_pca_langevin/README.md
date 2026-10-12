# Natywne kanały obecności i GLE bez PCA

[Wyniki](WYNIKI.md), [plan](PLAN.md), [audyt](AUDYT_ASTRA.md), [podsumowanie](PODSUMOWANIE.md). 128 kanałów wybranych tylko na train/val; test rozpoznania ma 92.83% BA. Prognoza tych samych 128 współrzędnych jest osobnym zadaniem i nie osiąga 90% trafności. Wszystkie kanały zachowane; żadnych PCA/POD/SVD/rotacji/autoenkodera stanu.

Odtwarzanie: F:/Python/python.exe i kolejno wybor_kanalow.py --stage validation, audyt, wybor_kanalow.py --stage test, przygotuj_dynamike.py, dynamika.py --stage validation, audyt, dynamika.py --stage test, reszty.py, raport.py. Testy mają guard istniejących wyników. Bez treningu/pobierania/commitów.
