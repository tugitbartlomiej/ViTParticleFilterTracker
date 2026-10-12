---
date: 2026-10-02
type: mixed
tags: [pykoopman, nndmd, encoder-decoder, edmd, tozsamosc, no-pca]
status: in-progress
importance: high
---
# Postscriptum: użytkownik zlecił rzeczywisty NNDMD z encoderem i decoderem

**Stan odczytany:** 2026-10-02, 12:25:52 +02:00, Europe/Warsaw. Ten plik uzupełnia [podsumowanie sesji](SESSION_SUMMARY.md) bez przepisywania jego wcześniejszego stanu.

**Aktualna intencja po korekcie użytkownika około 12:30:** wejście NNDMD to **pełne 768 oryginalnych współrzędnych cząsteczki**, nie J128. Planowany decoder ma odtwarzać pełne 768; J128 pozostaje podzbiorem do osobnej oceny tożsamości. Ekstrakcja pełnych danych została zlecona, a wynik NN nadal nie jest znany. Poniżej zachowano chronologię wcześniejszej wersji, którą ta korekta zastępuje.

Użytkownik najpierw jawnie zatwierdził **nową gałąź PyKoopman NNDMD z uczonym encoderem, decoderem i operatorem**, wraz z rzeczywistym treningiem i obliczeniami. Wcześniejsze ograniczenie „bez autoenkodera” pozostaje opisem historycznego wariantu EDMD i sposobu identyfikacji kanałów; nie blokuje nowej, wyraźnie zleconej architektury. W pierwszej wersji zlecenia zakładano wejście i odtworzone wyjście J128. **To historyczne założenie zostało następnie skorygowane do 768 i nie opisuje obecnego wejścia NN.**

To odrębna konstrukcja od wcześniejszego afinicznego operatora 128×128 wprost na kanałach. W NNDMD operator działa w reprezentacji kodera, a decoder zgodnie z aktualnym planem wraca do pełnej reprezentacji 768D. Zaproponowano ukrytą reprezentację uczoną o wymiarze 128 i operator w tym latencie, ale **wymiar ukryty nie jest tu przedstawiany jako ostatecznie ustalony przez użytkownika**. Natywne J128 i wyuczone 128 współrzędnych kodera byłyby różnymi obiektami. Wymiar ukryty, funkcję celu, podział danych, wybór ustawień i regułę zamrożenia musi określić nowy protokół. Nadal obowiązuje brak PCA oraz zakaz przyszłych pomiarów w identyfikacji i autonomicznej prognozie.

Pełna ekstrakcja została zlecona agentowi plan_poprawek: docelowo `[101,34,600,768]`. Istniejący `[101,34,600,128]` pozostaje potwierdzonym wcześniejszym artefaktem, lecz nie jest pełnym wejściem nowej wersji NN. Zlecenie ekstrakcji 768D nie stanowi jeszcze dowodu jej ukończenia; nowy manifest musi wykazać kształt, zgodność źródłowych sond, skończoność i komplet checkpointów.

Za wykonanie odpowiada agent `/root/pykoopman_obliczenia`. **Wynik NNDMD jest nieznany**; zlecenie, utworzenie kodu albo rozpoczęcie treningu nie stanowi dowodu osiągnięcia celu 85 procent. Nie przepisujemy wyników wcześniejszego EDMD jako wyników sieci neuronowej. Przyszłe pomiary po originie nie mogą korygować autonomicznej prognozy ani brać udziału w dopasowaniu ocenianego modelu.

## Nowe artefakty dostępne po pierwszym zapisie

Poniższe informacje pochodzą z odczytanych plików. Autor zapisu sesji nie powtarzał fitu ani niezależnego pełnego audytu nowego EDMD.

- [RUNTIME.json](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/RUNTIME.json) potwierdza izolowane środowisko, PyKoopman 1.2.1 i setuptools 80.9.0. Poprawka setuptools rozwiązywała brak `pkg_resources` wymagany przez tę wersję pakietu.
- [WYNIKI.json](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/WYNIKI.json) dotyczy **statycznego EDMD, nie NNDMD**. Ma `status=complete`, zamrożenie modelu 12:12:39 i zakończenie 12:13:17. Zawiera A 128×128, b 128, operator rozszerzony 129×129, 30 obrazów fit, 4 holdouty, 1500 par obraz–czas oraz 25100 rzeczywistych par tokenów do kroku 500. Wybrano lambda=0,01.
- Raportowany wynik tego EDMD na H500 w czterech holdoutach wynosi **57,3690 procent znaku** i **−51,6026 procent MSEskill**. MSE jest około 1,516 razy większy od persystencji. Lista horyzontów osiągających 85 procent jest pusta. Są to liczby z raportu wykonawcy; nie stanowią wyniku nowej gałęzi NN ani zamiennika wcześniejszych metryk benchmarku 99 konfiguracji.
- Odczytana główna metryka EDMD miała próg numerycznego zera 10⁻⁸. Na raportowanych H10/H100/H200/H500 masa aktywnych współrzędnych jest praktycznie 1; osobną diagnostykę 10⁻³ zapisano z jej coverage. Wymóg raportowania dokładnego znaku wszystkich współrzędnych zgłoszono koordynatorowi. Nie należy ukrywać różnic definicji metryki między eksperymentami.
- [PIPELINE_TOZSAMOSCI_NARZEDZIA.md](../../SmallViTAblation/PIPELINE_TOZSAMOSCI_NARZEDZIA.md) już istnieje; sprawdzono jego odczyt. Porządkuje identyfikację J128, pomiar na 34 obrazach oraz dalszą prognozę. Wyjaśnia też późniejsze, jawne dopuszczenie NNDMD. W pierwszym zapisie sesji dokument był jeszcze tylko zlecony; to postscriptum aktualizuje ten stan.

## Koordynacja zasobów i zapis

GPU RTX 3070 ma priorytet dla NNDMD. Próbę użycia GPU do indeksowania sesji zatrzymano natychmiast po otrzymaniu nowego zlecenia; zakończono wyłącznie własne procesy indeksowania. Kontynuacja indeksu ma działać na CPU. Nie zmieniano danych, kodu ani parametrów eksperymentu z poziomu zapisu sesji.

Ostateczny techniczny stan zapisu i indeksu jest w [INDEX_STATUS.json](INDEX_STATUS.json). Podsumowanie oraz to postscriptum pozostają zapisami dostępnych ustaleń i artefaktów, a nie dosłownymi transkryptami rozmowy.
