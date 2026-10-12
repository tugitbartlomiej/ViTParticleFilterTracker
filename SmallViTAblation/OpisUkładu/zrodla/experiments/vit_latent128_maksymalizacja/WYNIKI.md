# Wynik: mocna optymalizacja J128

2026-10-04, lokalna RTX3070. Checkpoint1000,400 obrazów. Pilot12 obrazów wybrał lr0.3; potem800 kroków Adam na obraz. Wszystkie wagi i bufory ViT pozostały zamrożone. Optymalizowano wyłącznie128 wskazanych współrzędnych każdego z600 tokenów, z zachowaniem zakresu GELU.

| Wariant | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|
| Bez interwencji | 0.437555 | 0.021881 | 0.120881 |
| Główna optymalizacja | 0.884748 | 0.774186 | 0.770525 |
| Kontrola znaków z modułami oracle | 0.873660 | 0.752233 | 0.756869 |
| Dodatkowy cel na głowie,64 obrazy | 0.850983 | 0.751623 | 0.736021 |

Duża poprawa na optymalizowanej puli jest potwierdzona. Cel AP50>=0.95 nie został osiągnięty. Dodatkowy wariant był gorszy, pozostaje zapisany osobno. Kontrola znaków dziedziczy moduły zmian z GT i ograniczenie GELU wymusza wiele dodatnich znaków; nie wolno wnioskować, że przypadkowe wartości bez etykiet dają taki sam zysk.

Średnia IoU najpewniejszej ramki wzrosła z0.471680 do0.872713. Średnia strata spadła z2.384983 do0.208414. Tylko0.5% obrazów ma jednocześnie confidence>=0.9 i IoU>=0.5, więc wysoki AP nie oznacza pewności bliskiej100%. Największa zmiana współrzędnej201.51149; cele mają poprawny zakres GELU, ale ich osiągalność przez rzeczywiste zdjęcia pozostaje nieznana.

Wszystkie etykiety użyte do optymalizacji odpowiednich latentów. To manipulacja reprezentacją przy zamrożonym detektorze, nie trening ViT ani wynik generalizacji.128 kanałów wybrano wcześniej przy checkpointcie500 i zachowano ich indeksy przy1000. Wyniki nie dowodzą, że są jedynymi kanałami związanymi z tożsamością narzędzia.

Kontrole:4 testy CPU, zerowa różnica bazowego tail, zmodyfikowany pełny forward na obrazach0/199/399 zgadza się do1.7e-6. Pozostałe640 kanałów bez zmian, hash wag/buforów identyczny. Wszystkie400×600×128 cele skończone i zapisane; target=base+delta, minimum około-0.16997123.

Obliczenia główne448s. Błąd serializacji przy końcowym dopisaniu do rejestru naprawiono bez powtarzania obliczeń; RESULT.json i NPZ już istniały. Główny wynik zarejestrowany w BIEG_full400_k1000.json, dodatkowy w BIEG_refine64_k1000.json.

Artefakty w `outputs/vit_latent128_maksymalizacja/`: RAPORT.md, POROWNANIE.png, PRZYKLADY.png, VERIFICATION.json, KANALY128.csv, TARGETS_FULL400.npz, DETECTIONS_FULL400.npz. Pliki REFINED dotyczą odrzuconej próby dalszej poprawy. Źródła wykonania zachowane w EXECUTED_SOURCE.
