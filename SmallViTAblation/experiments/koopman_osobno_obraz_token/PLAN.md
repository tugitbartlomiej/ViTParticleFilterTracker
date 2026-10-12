# Osobny operator dla każdego wektora każdego obrazu

Data: 2026-09-25. Jawna zmiana zlecona przez użytkownika: nie łączyć różnych obrazów ani tokenów w regresji. To nowy eksperyment, bez nadpisywania poprzednich wyników.

## Pytanie

Czy osobna afiniczna macierz dla historii jednego pełnego wektora lepiej przewiduje jego następne stany niż proste kontrole? Wynik negatywny pozostaje wynikiem. Niski błąd na całej historii nie jest dowodem trafnej prognozy.

## Jednostka i dane

Jednostka dopasowania to para (image_index, token_index). Dane: koopman_gesta_historia/wyniki/macierze_wiele_obrazow, 72 obrazy, 576 tokenów, po 384 współrzędne, 73 checkpointy kroków 5..365 co 5 aktualizacji. Powstanie 41472 oddzielnych modeli. Każdy końcowy model dostaje wszystkie 72 kolejne pary WŁASNEJ historii. Nie tworzymy przejść między obrazami lub pozycjami. Nie ma PCA, obcinania rzędu, średniej reprezentacji ani kompresującego enkodera.

Różne obrazy, również warianty tego samego źródła, dostają oddzielne parametry. Wspólna implementacja i obliczenia partiami nie oznaczają wspólnej macierzy. Maska służy tylko do osobnych podsumowań foreground/background. Przy stałych pikselach i siatce token ma tę samą pozycję między checkpointami, choć jego pole recepcyjne może zawierać kontekst.

## Matematyka

Osobno dla każdej jednostki: X ma wiersze z_t, Y wiersze z_(t+1). Uczymy przyrost:

```text
dx = Y-X
mu_x = mean(X); mu_dx = mean(dx)
Xc = X-mu_x; Dc = dx-mu_dx
lambda = relative_ridge * trace(Xc*Xc.T)/N
C = solve(Xc*Xc.T + lambda*I_N, Dc)
K = I_384 + Xc.T*C
b = mu_dx - mu_x*(K-I)
z_next = z*K+b                 # konwencja wierszowa
```

Parametr dodatni wybiera jedno z niedookreślonych rozszerzeń operatora. Nawet 72 pary nie wyznaczają jednoznacznie dowolnych 384×384 współczynników. Postać dualna jest dokładnym sposobem liczenia wybranej regresji, nie PCA. Brak sprzężeń między tokenami jest założeniem tej wersji, nie dowodem ich niezależności.

## Podział czasowy zamrożony przed wynikiem

1. Kalibracja każdej jednostki na jej własnych danych: 40 par (stan 0..40, kroki 5..205). Dwie kandydatury relative_ridge: 0.001 i 0.1. Stała macierz przewiduje jeden krok dla początków 40..47, z celami do stanu 48 (krok 245). Wybór minimalnego MSE; remis wybiera silniejszą regularyzację. Nie łączymy błędów różnych jednostek do wyboru.
2. Ocena: dopasować macierz do 48 pierwszych par (do kroku 245) z wybraną regularyzacją i zamrozić. Testować 24 przejścia o początkach 48..71, celach 49..72 (kroki 250..365). Każda prognoza jednego kroku otrzymuje rzeczywisty stan na początku, nie przyszły cel.
3. Dodatkowe prognozy 20 checkpointów (100 aktualizacji) od stanów 48,50,52: ta sama zamrożona macierz, bez prawdziwych stanów pośrednich. Te błędy nie wybierają modelu.
4. Dopiero po zapisaniu błędów testu: końcowa macierz z wszystkich 72 par własnej historii, ta sama regularyzacja. Zapis zastrzeżenia, że prognoza poza ostatni checkpoint jest niezweryfikowana. Nie przypisujemy błędów testowych tej końcowej, ponownie dopasowanej macierzy.

## Kontrole, miary i bramka

Kontrole: brak zmian, ostatni przyrost oraz alpha*v (alpha dopasowane tylko do przyrostów znanych w pierwszych 48 parach danej jednostki). Te same początki i cele. Raportować błąd dopasowania, testowy RMSE każdej jednostki i każdego przejścia, stosunek do rzeczywistej zmiany, cosinus i długość przyrostu. Wszystkie obrazy są własnymi trajektoriami; nie jest to test przenoszenia jednej macierzy na nowe obrazy.

Przedwynikowa flaga kandydata: RMSE <=0.95 RMSE ostatniego przyrostu, <=0.8 RMSE braku zmian i średni cosinus >=0.5. Osobno podać przewagę nad dodatkową kontrolą alpha oraz wyniki masek >=0.5. Liczba spełniających kryteria wśród 41472 testów jest eksploracyjna, nie certyfikatem statystycznym ani dowodem semantycznej tożsamości. Nie wybieramy przykładu z najmniejszym błędem po obejrzeniu testu; ilustracja z poprzedniego eksperymentu jest ustalona z góry: obraz 66, token 299.

Historyczna pula była wcześniej oglądana. Podział jest chronologiczny i niewykorzystywany do doboru bieżącej konfiguracji, lecz nie stanowi nowego niezależnego testu całego programu badań.

## Zasoby i zapis

RAM: pełne odczyty około 4.65 GB float32 (oryginalna dokładność zapisu), obliczenia regresji float64. Wolne RAM sprawdzone przed startem: około 16 GB. Wolny dysk: około 107 GB. Jawne 41472 macierze float64 zajęłyby około 49 GB; równoważne czynniki C, średnie i parametry około 9.5 GB, a Xc odtwarzamy z wejść o sprawdzonym SHA256. Wybrany operator można eksportować jawnie jako 384×384. Zapisać również modele oceny lub ich pełny przepis odtworzenia, podziały, błędy per przejście, sumy kontrolne i wersję kodu.

Obliczenia partiami, bez zmiany wag RF-DETR/FastGAN, bez nowego inference i bez zatrzymywania innych procesów. Najpierw test algebraicznej zgodności, niezależności jednostek, podziału czasowego oraz dokładności CPU/GPU; mały pomiar czasu rozstrzyga urządzenie. Nie zmieniać progów po ocenie. Wyniki zapisać wraz z ograniczeniami natychmiast po biegu.
