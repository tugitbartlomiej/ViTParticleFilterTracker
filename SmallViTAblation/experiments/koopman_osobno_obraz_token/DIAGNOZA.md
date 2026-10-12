# Osobna macierz dla każdego obrazu i wektora: wykonanie i wniosek

Data: 2026-09-25. Wykonano lokalnie na RTX 3070, w float64.

## Co policzono

72 obrazy × 576 pozycji tokenów = **41472 niezależne macierze**, każda działająca na pełnym wektorze 384 liczb. Każda ma własne średnie, regularyzację i współczynniki. Nie połączono obrazów ani pozycji w regresji, nie użyto PCA i nie uśredniano stanu.

Historia każdego wektora ma 73 dostępne checkpointy od kroku 5 do 365. Daje to 72 kolejne pary na macierz, łącznie 2985984 pary we wszystkich osobnych zadaniach. Nie daje to milionów przykładów jednej wspólnej macierzy. Pozostałe etapy treningu nie są tutaj pełnymi historiami wszystkich obrazów.

Końcowe macierze dopasowano do wszystkich 72 par własnej historii. Są zapisane dokładnie jako K=I+Xc.T*C oraz b=mu_delta−mu_x*(K−I). Czynniki C, średnie i parametry zajmują około 9.5 GB. Xc odtwarzane jest z tensorów źródłowych o sprawdzonym SHA256. Eksportowano również jawną macierz 384×384 dla obrazu 66, tokenu 299.

## Rozdzielenie dopasowania i prognozy

Ridge każdej jednostki wybrano z {0.001,0.1} na jej wcześniejszej walidacji, bez błędów innych obrazów. 197 jednostek wybrało 0.001, a 41275 wybrało 0.1.

Do testu dopasowano oddzielną macierz na pierwszych 48 parach, kończących się w kroku 245. Zamrożono ją i oceniono na kolejnych 24 przejściach, do kroku 365. Macierz końcowa wykorzystująca całą historię została dopasowana osobno; nie przypisujemy jej wyników uzyskanych przez macierz wcześniejszą. Jej prognoza po 365 jest niezweryfikowana.

| Miara RMSE | Osobny Koopman | Ostatni przyrost | Brak zmian | Osobny model alpha |
|---|---:|---:|---:|---:|
| Jeden checkpoint | 0,0103307 | 0,0038589 | 0,0082256 | **0,0037180** |
| 100 aktualizacji | 0,2321807 | 0,1169686 | 0,1362229 | **0,0710764** |
| Jeden checkpoint, maska >=0,5 | 0,0111571 | 0,0045265 | 0,0091912 | **0,0043701** |

Każdy model alpha też ma własny współczynnik dla danego obrazu/tokenu, dopasowany tylko do przyrostów w znanej przeszłości. Nie korzysta ze wspólnych parametrów ani przyszłych stanów. Długi horyzont ma trzy początki, krótki 24; porównanie metod w każdym wierszu ma identyczne warunki.

**Zero z 41472 macierzy spełniło pełną ustaloną bramkę**, liczoną po 24 testowych przejściach: przewaga RMSE co najmniej 5 procent nad ostatnim przyrostem, co najmniej 20 procent nad brakiem zmian i cosinus >=0,5. To nie znaczy, że każda pojedyncza prognoza była błędna albo że żaden wektor nie jest przewidywalny. Nie zmieniono bramki po wyniku.

## Co dokładnie się pogarsza

Bezpośrednio zaobserwowano:

- RMSE macierzy rośnie od 0,0052897 na pierwszym testowym przejściu do 0,0144163 na ostatnim. W tym samym czasie RMSE braku zmian maleje z 0,0118773 do 0,0059724.
- Średni cosinus przewidywanego i rzeczywistego przyrostu wynosi 0,6193, wobec 0,8643 dla ostatniego przyrostu.
- Mediana stosunku długości prognozowanej zmiany do rzeczywistej wynosi 1,6143. Dla ostatniego przyrostu: 1,0247; dla alpha: 0,9433. Macierz często przewiduje zbyt duży ruch.
- Błąd dopasowania 48 par wynosi 0,0044499; po dopasowaniu wszystkich 72 par 0,0044847. Są to błędy odtworzenia innych zestawów danych, nie porównanie dwóch prognoz.
- Rząd wycentrowanych końcowych danych wynosi 71 we wszystkich jednostkach. Nadal nie określa to dowolnej mapy 384×384. Maksymalny względny residual rozwiązania układu liniowego to około 1,05e-12.

Wniosek: wspólna macierz różnych obrazów nie była jedynym problemem wcześniejszej konstrukcji. Nawet po usunięciu tego założenia stała mapa dopasowana do wcześniejszej części historii słabo działa na jej późniejszej części. To zgodne z niedostatecznym opisem zmieniającej się dynamiki i ekstrapolacją poza dobrze opisany zakres. Nie dowodzi jednej wyłącznej przyczyny ani błędu samej teorii Koopmana.

Ten test nie rozstrzyga, czy osobne macierze aktualizowane w krótkim przesuwanym oknie byłyby lepsze. Poprzedni test jednego tokenu aktualizował model, a ten zamraża macierz do oceny; dodatkowo zmieniono zbiór jednostek. Nie wolno przypisywać całej różnicy względem poprzedniego wyniku wyłącznie rozdzieleniu obrazów.

## Gdzie są macierze i błędy

- wyniki/PER_VECTOR.csv: wynik każdej pary obraz/token, włącznie z maską, ridge, błędem dopasowania, prognozy i kontrolami.
- wyniki/MSE_ONE_STEP.npy: kształt 72×576×24×4; osie: obraz, token, przejście testowe, metoda. Kolejność metod jest w PROTOCOL.json.
- wyniki/COSINE_ONE_STEP.npy i INCREMENT_RATIO.npy: kierunek i długość zmian na każdym przejściu.
- wyniki/C_FINAL.npy, MEAN_X_FINAL.npy, MEAN_DELTA_FINAL.npy: wszystkie końcowe operatory w postaci dualnej, bez obcięcia rzędu.
- wyniki/MATRIX_final_066_299.npz: jawny końcowy operator przykładowego wektora.
- wyniki/EXAMPLE_EVALUATION_MATRIX.npz: inny, wcześniejszy operator użyty w teście tego samego wektora.
- [Wykres błędów](wyniki/WYNIKI.png).

Przykładowy wektor wybrano z góry zgodnie z poprzednim eksperymentem, nie na podstawie najlepszego testu. Ma RMSE 0,0084629, wobec 0,0048289 ostatniego przyrostu.

## Weryfikacja

Pięć testów matematycznych obejmuje zgodność primal/dual, prognozę wielokrokową, niezależność jednostek w batchu, stały dryf, wybór ridge i zgodność CPU/GPU. Trzy testy operacyjne obejmują odmowę eksportu nieukończonych danych, odmowę wznowienia z brakującymi czynnikami oraz ochronę statusu ukończonego biegu. Wszystkie przeszły.

Sprawdzono SHA256 73 tensorów wejściowych i zapisano SHA256 wszystkich wynikowych tablic. Eksport jawnej macierzy ponownie sprawdził używane wejścia i czynniki. Niezależne odtworzenie na CPU sprawdziło cztery z góry ustalone jednostki, ich współczynniki i błędy testowe, a także jawne macierze oceny i końcową. Dowód: wyniki/VERIFICATION.json.

Po starcie dodano operacyjne zabezpieczenia eksportu/wznawiania/statusu. Nie zmieniono obliczeń ani podziałów. Wersja faktycznie użyta w biegu jest w wyniki/source_snapshot/, a protokół zachowuje jej hashe. Ukończony bieg nie korzystał z problematycznych ścieżek wznowienia i eksportu częściowych danych.

Ograniczenia: znana wcześniej pula, zależne augmentacje, wielokrotne eksploracyjne porównania, brak testu przyczynowej tożsamości narzędzia. Maska nie zapewnia, że token nie opisuje kontekstu. Brak PCA nie usuwa niedookreślenia ani założenia o pomijaniu sprzężeń między tokenami.
