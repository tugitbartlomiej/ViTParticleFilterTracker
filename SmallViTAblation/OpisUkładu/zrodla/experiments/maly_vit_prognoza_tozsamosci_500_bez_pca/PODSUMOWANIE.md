# 99 modeli na 128 oryginalnych kanałach: wynik poszukiwania do 500 kroków

2026-10-02. **Nie osiągnięto celu ponad 85 procent.** Sprawdzono 99 konfiguracji na wcześniejszej walidacji i następnie 16 zamrożonych modeli na horyzontach1–500. Żaden z tych 16 modeli nie przekroczył 85 procent trafności znaku średnio ani na obrazach dopasowania, ani na transferze, w żadnym z 500 ocenionych horyzontów.

Stan zachowuje wszystkie 128 wcześniej wskazanych oryginalnych kanałów. Nie użyto PCA, rotacji ani autoenkodera. Prognozowane są współrzędne cząsteczek narzędzia:218 tokenów z 12 stałych obrazów. Wspólne osie pozostają stałe. Nie utożsamiamy wcześniejszego rozpoznawania oryginału versus LaMa z prognozowaniem dynamiki.

## Co udało się uzyskać

Przed testem zachowano dwa różne wybory. Główne kryterium maksymalizowało najgorszą trafność znaku wśród horyzontów po przejściu bramki błędu. Dodatkowy, także zamrożony przed testem wybór maksymalizował średnią redukcję MSE. Poniżej **transfer na cztery obrazy niewykorzystane do dopasowania równań**:

| Model | Znak H1 | Znak H100 | Znak H500 | Redukcja MSE H500 |
|---|---:|---:|---:|---:|
| Główny wybór: powrót do średniej, okno 200 i czas 200 |50.56%|62.36%|56.52%|−7.52%|
| Wybór według MSE: powrót do średniej, okno 50 i czas 50 |49.02%|66.20%|62.19%|+9.82%|
| Langevin, wykładnicza pamięć, krok modelu10 |46.26%|patrz pełna krzywa|57.46%|−148.23%|
| Koopman, krok modelu10 |43.90%|patrz pełna krzywa|56.02%|−170.40%|
| Obserwator Kalmana |69.28%|patrz pełna krzywa|43.28%|−279377.46%|

Model wybrany według MSE dawał przy H100 redukcję błędu25.32 procent, a przy H500 redukcję9.82 procent. Jednak norma prognozowanego ruchu przy H500 wynosiła średnio około 0.464 normy rzeczywistej. Trafiał część tendencji, ale zaniżał wielkość przemieszczeń. Główny wybór przy H500 na transferze był gorszy od braku ruchu. Nie zmieniamy go po obejrzeniu testu na lepiej wyglądający wariant.

Na jedną aktualizację prosty ostatni przyrost dawał75.73 procent znaku i45.72 procent redukcji MSE. Przedłużanie tego przyrostu na 500 aktualizacji powodowało ogromny błąd. Trafna krótka prognoza nie oznacza użytecznej długiej prognozy.

## Najbardziej użyteczne równanie w tym porównaniu

Dodatkowy wariant wybrany według MSE opisuje powrót każdej współrzędnej cząsteczki do poziomu wyznaczonego z jej ostatnich 50 zapisów:

```text
mu_i(t) = (1/50) sum_{k=0..49} z_i,J(t−k)
zhat_i(t+H) = mu_i(t) + [z_i,J(t) − mu_i(t)] exp(−H/50)
```

J oznacza128 niezmienionych kanałów, a i konkretną cząsteczkę. W chwili startu wyznaczamy mu wyłącznie z przeszłości i zamrażamy je na cały przebieg prognozy. Nie wstawiamy późniejszej prawdziwej średniej. To postać dryfu powrotu do równowagi, podobna do deterministycznej części modelu Ornsteina–Uhlenbecka. Nie dopasowaliśmy ani nie potwierdziliśmy tu pełnego procesu szumu tego modelu.

To wynik porównania dostępnych wariantów, nie dowód, że ta prosta postać jest najlepszym możliwym równaniem rzeczywistej sieci. Szczególnie nie dowodzi zamknięcia stanu w 128 kanałach ani tego, że aktualizacje AdamW stały się nieistotne.

## Co mówi wynik o bardziej złożonych modelach

W tym teście pamięć i sprzężenie nie wystarczyły do zbudowania dobrej prognozy500 aktualizacji. Modele mogą odtwarzać lokalny trend, ale przy autonomicznym rozwijaniu zwiększać błąd lub nietrafnie przewidywać jego dalszy kierunek. Obserwator nie otrzymuje pomiarów po starcie, więc nie ma późniejszych korekt Kalmana. Przypadek z niegasnącą prędkością ilustruje nadmierne przedłużanie krótkiego trendu: norma prognozy była około 51 razy większa od normy rzeczywistego ruchu.

Wynik **nie rozstrzyga**, czy główną przyczyną jest brak stanu optymalizatora, niestacjonarność uczenia, za mało niezależnych trajektorii, forma regresji czy połączenie tych czynników. Nie ma podstaw do stwierdzenia, że85 procent jest matematycznie niemożliwe. Jest podstawa do odrzucenia twierdzenia, że osiągnęliśmy ten wynik w tej próbie.

## Jak chroniono ocenę

Kanały określono na kroku500. Dobór parametrów korzystał z wcześniejszych fragmentów, z centrami i skalami obliczanymi tylko na właściwym prefiksie; nie odejmowano przyszłego stanu500 we wcześniejszych foldach. Modele dopasowano ponownie do 500, a potem oceniono autonomiczne prognozy500–1000. Reprezentacje przyszłych obrazów, paczki, gradienty i późniejsze stany tła nie były podawane modelom.

99 konfiguracji obejmowało natywne operatory Koopmana, przycięcie widma, pamięć skończoną i wykładniczą GLE, krok modelu1 lub10, obserwatory Kalmana, wspólny ruch narzędzia/tła oraz kontrolne trendy i relaksację. Modele co10 wypełniają pośrednie kroki interpolacją samych prognoz, bez użycia przyszłych stanów rzeczywistych. Szczegółowa reguła wyboru, jawna historia doprecyzowania jednostki rho oraz hash plików są zapisane obok wyników.

Dla H500 istnieje jeden moment startu po ustaleniu kanałów. Wszystkie obrazy obserwują jeden trening; nie są niezależnymi seedami sieci. Dodatkowo tę historię wcześniej analizowano. Wynik pozostaje eksploracyjny, a liczba 218 cząsteczek nie zamienia go w 218 niezależnych eksperymentów.

## Materiały

- [Pełne wyniki i tabela wszystkich rodzin](WYNIKI.md).
- [Wykres procentowej trafności na transferze](ZNAK_TRANSFER.png).
- [Wykres błędu i amplitudy](BLAD_I_AMPLITUDA.png).
- [Rzeczywiste i prognozowane trajektorie](TRAJEKTORIE.png).
- [Literatura, Undermind i audyt matematyczny](LITERATURA_I_AUDYT.md).
- [Metryki wszystkich horyzontów](POROWNANIE.csv).
- [Definicja modeli](CO_OPISUJE_MODEL.md).

Następna niezależna próba powinna dostarczyć dłuższą historię po ustaleniu kanałów oraz kilka rzeczywiście oddzielnych biegów. Dopiero to umożliwi dobór i potwierdzenie500-krokowej prognozy bez polegania na jednym punkcie końcowym. Możliwym rozszerzeniem pozostaje nieliniowy model stanu lub uczenie bezpośrednio na wielokrokowej stracie, przy zachowaniu natywnych kanałów i zakazu przyszłych wejść; nie są one wynikiem obecnego pilota.
