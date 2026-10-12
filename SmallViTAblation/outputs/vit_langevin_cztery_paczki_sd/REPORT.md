# Cztery paczki SD: prognoza i100 aktualizacji

Punkt wyjścia to oryginalny checkpoint SD k1000 i jego autentyczny stan AdamW. Sonda29: niezmienione zdjęcie TR_0154/1926. Cel: after z oryginalnego LATENT_PELNY, nie naturalny latent obrazów SD. Wszystkie32 zdjęcia każdego ramienia mają prawdziwe GT z ZIP.

Prognozy zapisane i zahashowane przed aktualizacjami. Główna suma S_nowa+S_ped, dodatkowo historyczne współczynniki ze źródła. Transfer do600x768 oceniony na kroku1. Horyzont1/5/10/25/50/100: stała siła początkowa, brak przyszłych gradientów. Jeden krok powtarza paczkę32;100 nie oznacza100 epok pełnego zbioru.

| Ramię | prognoza gain128 na100 | rzeczywisty gain128 | prognoza gain768 | rzeczywisty gain768 | IoU sondy | pewność sondy |
|---|---:|---:|---:|---:|---:|---:|
| 2026100511 | -9.2306% | 0.5222% | -34.8204% | 0.2742% | 0.6016 | 0.5651 |
| 2026100523 | -14.7352% | 0.7818% | -58.7136% | 0.4496% | 0.5737 | 0.5072 |
| 2026100537 | -13.1607% | 0.0481% | -60.8608% | -0.2521% | 0.6579 | 0.7417 |
| 2026100549 | -17.1148% | 0.4438% | -76.4192% | 0.0470% | 0.1725 | 0.3887 |
| random32 | -8.6383% | 0.3001% | -40.5220% | 0.1621% | 0.8064 | 0.9158 |

Gain dodatni oznacza redukcję MSE, ujemny pogorszenie. Nochange gain=0. Wyniki ujemne są zachowane. Progi5/10/25/85/90 procent redukcji błędu: null oznacza brak przekroczenia do100, bez wymyślonych czasów ani epok. Obserwowane progi IoU/pewności są oddzielne od prognozy latentu.

Pełne wyniki progów i kierunku: RESULT.json, FORECAST_VS_ACTUAL.csv. Dokładne trajektorie naturalnych latentów: TRAJECTORY_*.npy; checkpointy mają również optymalizator. Głowa detektora zmieniała wagi w każdym ramieniu. Wirtualny krok na parametrach daje naturalną odpowiedź sondy tylko dla kroku1; ekstrapolowany latent nie wyznacza przyszłej głowy, więc dalszej pewności z niego nie wymyślamy.

Koszt: {"initial_gradient_batches": 5, "functional_probe_forwards_total": 526, "actual_updates": 500, "actual_training_images": 16000, "actual_probe_forwards": 500}; 110.2 sekund.500 rzeczywistych aktualizacji po32 zdjęcia. Kontrola nochange ma0 aktualizacji. Żadne cudze zadanie nie zostało zatrzymane.

Poprzednia, inna paczka32 pogorszyła MSE768 z14.3202 do14.3605 po50. To kontekst, nie założenie obecnego wyniku. Test jednej sondy nie dowodzi generalizacji ani globalnej nieosiągalności celu po dalszym uczeniu. Brak augmentacji zadeklarowano; zachowano zwykły BN pełnej paczki32.

## Wniosek i kontrola prognozy

Prognoza kroku1 prawidłowo wskazała oddalanie od celu wszystkich czterech paczek. Skill przewidywania przyrostu pełnego latentu wyniósł88.80-98.60 procent z rzeczywistym BN, a nie taki procent zbliżenia do celu. Po100 krokach trzy paczki nieznacznie zmniejszyły MSE768, jedna pogorszyła. Najlepsza końcowa redukcja0.4496 procent jest mniejsza od pierwszego progu5 procent. Ekstrapolacja stałej początkowej siły na100 zdecydowanie nie opisuje rzeczywistego ruchu. Nie podajemy na jej podstawie liczby dalszych aktualizacji do celu.

Dwie paczki,2026100511 i2026100523, uzyskały lepszą końcową redukcję MSE768 niż random32. Pozostałe nie. Wszystkie cztery pogorszyły końcowe IoU oraz confidence sondy względem niezmienionego modelu. Random32 również pogorszył IoU, choć zwiększył confidence. Zbliżenie latentu do sztucznego celu nie oznacza poprawy detektora.

Zweryfikowano wszystkie101 stanów pięciu trajektorii z zapisanymi metrykami, checkpointy i optymalizatory k1100, oryginalny before i seal prognozy. Odczyt naturalnego latentu z każdego końcowego checkpointu daje dokładnie zapisany ostatni latent: maksymalny błąd0. Kontrola głowy potwierdza jej aktualizację. Pierwszy błąd kodowania JSON przed treningiem zachowano w ATTEMPT01_ENCODING_FAILURE.json; odczyt naprawiono na UTF-8.

Dodatkowy wykres rzeczywistych odległości w powiększonej skali: DISTANCE_ACTUAL_DETAIL.png. Weryfikacja: FINAL_VERIFICATION.json.
