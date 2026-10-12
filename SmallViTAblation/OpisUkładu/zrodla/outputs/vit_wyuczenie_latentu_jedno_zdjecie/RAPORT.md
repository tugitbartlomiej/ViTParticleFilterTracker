# Naturalne wyuczenie latentu jednego obrazu

Pytanie: czy uczenie oryginalnych wag przed GELU osiągnie zapisany sztuczny target after bez podmiany aktywacji?

Wynik kryterium łącznego: NIEOSIĄGNIĘTY.

| Pomiar | Oryginalny naturalny | Najlepszy naturalny | Końcowy naturalny |
|---|---:|---:|---:|
| MSE128 | 85.9211807 | 10.9162397 | 10.9162397 |
| MSE640 | 0 | 4.18178752e-06 | 4.18178752e-06 |
| MSE768 | 14.3201971 | 1.81937671 | 1.81937671 |
| agreement128 | 0.335600708 | 64.4756232 | 64.4756232 |
| cosine128 | 0.106738292 | 0.943290353 | 0.943290353 |
| confidence | 0.867869496 | 0.771777928 | 0.771777928 |
| iou | 0.856594683 | 0.999952275 | 0.999952275 |
| GTloss | 0.490047604 | 0.154576004 | 0.154576004 |

Agreement128 = 100 max(0, 1-sqrt(MSE128/mean(target128²))). MSE640 mierzy zmianę wobec niezmienionych kanałów before. Cosine jest osobno i nie zastępuje agreement.

Czas treningu wraz z przygotowaniem i eksportem: 148.779 s. Weryfikacja i odczyt: 1.171 s.

## Aparat i biegi

{"seed": 20261005, "steps": 1000, "lr": 0.0003, "min_lr": 3e-05, "retention": 10.0, "detection_weight": 0.02, "pause_seconds": 0.03, "agreement_threshold": 90.0, "iou_threshold": 0.9, "confidence_threshold": 0.9, "plateau_window": 250, "plateau_gain_pp": 0.2, "precision": "float32", "autocast": false, "tf32": false}

T1_norm2_fc1: 1000 aktualizacji, best krok 1000, stop budget, 148608 uczonych parametrów, czas 143.640 s. Końcowy przyrost najlepszego agreement przez 250 kroków: 1.832262 pp.

T2 nie został uruchomiony: końcowa poprawa T1 nie spełniła zadeklarowanej definicji plateau. Limit T1 został wyczerpany.

Pierwszy start przerwał błąd typu w przygotowaniu GT. Następny start wykonał 96 aktualizacji i został przerwany przez Windows PermissionError podczas atomowego zapisu progress, nie przez OOM. Nie było zapisanych wag po aktualizacjach, więc zakończony bieg rozpoczął od oryginału. Zachowano STARTUP_ERROR i LOCK_FAILURE. Writer ma unikatowe pliki PID/czas, ponawianie OSError i niekrytyczny progress; każde poprawione best jest zapisywane przed progress.

## Podłoga i audyt

Oryginalny checkpoint i before odtworzone: max błąd GELU 0, rtol/atol 2e-5. Ten sam RGB480x320 INTER_AREA, ImageNet wewnątrz modelu. Target128 energia 86.5008011; target640 energia 0.154263243.

Niezależny proces utworzył nowe modele, załadował before/best/final, wykonał forward bez hooków i sprawdził detekcje. Odczytujący hook nie zmienił mapy ani o jeden bit. Audyt obejmuje każdy tensor state_dict; wszystkie nieuczone parametry i bufory BN pozostały identyczne. Źródłowe checkpoint, NPZ i obrazy zachowały SHA256. FRESH_RELOAD.json zawiera pełny audyt i błędy odczytu.

## Zastrzeżenia i zakres wyniku

To optymalizacja jednego uprzednio korzystnie wybranego obrazu oraz jego GT. Target był wcześniej optymalizowany z użyciem GT. Uczenie ma bezpośredni dostęp do targetu, a inference używa tylko obrazu i nowych wag. Wynik nie dowodzi generalizacji, poprawy na nowych zdjęciach ani możliwości uzyskania targetu przy oryginalnych zamrożonych wagach. T2, jeśli uruchomiony, ma większą pojemność i może zmienić wejście residual; oceniamy rzeczywistą głowę. Nie dodano lookup, biasfield ani nowych warstw; nie zmieniono lub przycięto targetu.

Przy nieosiągnięciu kryterium zapisany najlepszy residual jest wynikiem negatywnym w zadanym budżecie; nie dowodzi niemożliwości dopasowania innym treningiem.

## Artefakty

MODEL_before/best/final.pt, oryginalne checkpointy etapów z Adamem, LATENT_NATURAL_*.npz (600x768), VEKTORY_J128_*.csv (600x128), HISTORY.csv/json, OVERLAY_*.png i POROWNANIE_PRZED_CEL_NATURAL.png. RUN_CONFIG.json identyfikuje architekturę, wersje, źródło, kod, PID i stałe energie. Deklaracja rodziny pozostaje lokalna w dozwolonym zakresie.
