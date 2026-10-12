# Rozpoznanie narzędzia i Langevin na oryginalnych kanałach, bez PCA

Data: 2026-10-02. Użytkownik określił próg jako ponad 90 procent wspólnej trafności rozpoznania, a nie procent energii latentu. Cała analiza zachowuje oryginalne osie wybranej reprezentacji tokenu. Nie używa PCA, obrotu bazy, autoenkodera ani dawnej podprzestrzeni U8.

## Rozpoznanie: próg punktowy osiągnięty

Wybrano **128 oryginalnych kanałów spośród 768**. Jest to najmniejsza liczba z przebadanej siatki, która przekroczyła 90 procent na walidacji. Kanały uszeregowano wyłącznie na zbiorze treningowym według standaryzowanego efektu obecności narzędzia; wspólnie odczytuje je klasyfikator logistyczny. Zastosowano C = 0.01 i stały próg decyzji 0.5.

| Pomiar | Wynik |
|---|---:|
| Trafność walidacyjna, filmy ważone jednakowo | 90.67% |
| Trafność testowa, filmy ważone jednakowo | **92.83%** |
| Zwykła trafność testowa, obrazy ważone jednakowo | 91.76% |
| Czułość testowa | 91.97% |
| Swoistość testowa | 93.68% |
| AUROC testowe | 0.9741 |
| 95-procentowy przedział bootstrap po całych filmach | 88.17–96.99% |

Test obejmuje 85 par obrazów z 24 filmów, rozłącznych z filmami treningowymi i walidacyjnymi. Cały zbiór zawiera 400 par. Kanały i klasyfikator zamrożono przed tym testem; nie zmieniano ich na podstawie jego wyniku. Wynik punktowy przekracza 90 procent, ale dolna granica przedziału niepewności tego progu nie przekracza.

To rozpoznanie obecności narzędzia w znanym obszarze jego maski: oryginalny obraz versus wersja z narzędziem usuniętym przez LaMa. Reprezentację obrazu stanowi średnia cząsteczek ważona pokryciem maski, zmierzona na checkpointcie 500. Wynik nie oznacza, że każda cząsteczka sama rozpoznaje narzędzie z tą trafnością, ani że wykryto wszystkie przyczynowo odpowiedzialne osie. Nie jest też miarą detekcji mAP ani rozróżniania różnych typów narzędzi. Ten sam test był oglądany we wcześniejszych badaniach, więc wynik ma status eksploracyjny. Syntetyczne usunięcie może pozostawiać artefakty. Nie sprawdzono utrzymania trafności ponad 90 procent na każdym późniejszym checkpointcie.

Pełna lista osi, w obu numeracjach, znajduje się w `WYBOR_WALIDACJA.json` oraz `KANALY_MODEL.npz`. `RANKING_KANALOW.csv` zawiera ranking wszystkich oryginalnych kanałów. Podstawą liczb są `OCENA_OBECNOSCI_TEST.json` i zamrożony model o SHA-256 `5212a7bfff03d4d5a0756b37eae958ddc15a4f7fbd3935a9f8e1388b68b587a6`.

## Dynamika tych samych 128 kanałów

Stan każdej cząsteczki zachowuje wszystkie wybrane współrzędne w oryginalnej bazie. Odejmujemy jedynie jej stan z checkpointu 500. Makrowektory, czyli osie bazowe, pozostają stałe; prognozujemy zmiany współrzędnych cząsteczek wzdłuż nich.

```math
x_{i,t+\Delta t} = x_{i,t} + b + A x_{i,t}
  + \sum_{\ell=0}^{L-1} K_\ell \Delta x_{i,t-\ell\Delta t}
  + \varepsilon_{i,t+\Delta t}.
```

To lokalny wariant równania. Wariant sprzężony uwzględnia dodatkowo średni ruch cząsteczek narzędzia oraz kontekstu w tych samych kanałach. Część chwilowa jest liniowym przybliżeniem dryfu, a macierze przy historii przyrostów opisują skończoną pamięć. W prognozie wielokrokowej model korzysta z własnych przewidywań, bez prawdziwych przyszłych stanów czy obrazów.

Obliczenia obejmują 31 cząsteczek narzędzia na pierwszym obrazie oraz transfer na 7 cząsteczek drugiego obrazu, z jednego przebiegu uczenia. Sprawdzono po 32 konfiguracje dla kroku co jedną i co dziesięć aktualizacji. Wszystkie porównania dynamiki dotyczą tych samych 128 kanałów, więc nie porównujemy ich procentów bezpośrednio z dawnymi prognozami ośmiowymiarowymi.

Walidacja wybrała lokalny model z pamięcią dwóch przyrostów: L = 2, ridge = 1 dla dt = 1 oraz L = 2, ridge = 10 dla dt = 10. Ten sam zestaw wygrał również osobne kryterium najkrótszego horyzontu. Dodanie wspólnego ruchu narzędzia i kontekstu nie wygrało zadeklarowanego kryterium walidacyjnego. Nie zmienialiśmy wyboru po teście.

### Wynik na późniejszym odcinku tego samego obrazu

Prognozy co jedną aktualizację, starty 800–950. Znak oceniono dla wszystkich 128 współrzędnych wszystkich 31 cząsteczek, z wagami pokrycia narzędziem, bez wybierania tylko największych zmian.

| Horyzont w aktualizacjach | Trafność znaku zmiany | Redukcja MSE względem braku zmiany |
|---|---:|---:|
| 1 | **71.68%** | **47.75%** |
| 2 | 67.38% | 33.45% |
| 5 | 60.60% | 7.53% |
| 10 | 58.73% | −4.15% |
| 20 | 59.70% | −7.15% |
| 50 | 57.18% | −45.30% |

Redukcja błędu to `100 × (1 − MSE_model / MSE_brak_zmiany)`. Wartość ujemna oznacza gorszą prognozę niż utrzymanie bieżącego stanu. Na przykład −45.30 procent oznacza MSE o 45.30 procent większy od tego punktu odniesienia. Trafność znaku jest inną miarą i sama nie mówi, czy wielkość przesunięcia została trafnie przewidziana.

Na jeden krok **proste przedłużenie ostatniego przyrostu jest lepsze**: 78.67 procent trafności znaku i 55.92 procent redukcji MSE. Langevin pomaga względem braku ruchu, ale nie pokonuje tego prostego modelu na H = 1. Sam człon chwilowy L = 0 daje wtedy −0.43 procent redukcji MSE i 50.65 procent trafności znaku, więc pamięć poprawia ten wariant. Nie jest jednak wystarczająca do najlepszego przewidywania krótkiego ruchu.

W osobnym modelu uczonym na zapisie co dziesięć aktualizacji redukcja MSE na horyzontach 10, 20 i 50 wynosi odpowiednio 4.30, 9.40 i 4.15 procent. Dla H = 50 przedział bootstrap redukcji MSE wynosi −0.91 do 7.89 procent i obejmuje zero. Te wyniki opierają się na mniejszej liczbie startów, co dziesięć kroków. Bezpośrednie porównanie rozdzielczości należy czytać z zapisanej kontroli wspólnych startów, a nie odejmować liczby z różnych zbiorów startów.

### Transfer na drugi obraz

Bez ponownego dopasowania modelu ani kanałów, wybrany model dt = 1 daje na H = 1, 10 i 50 redukcję MSE odpowiednio −1.41, −262.36 i −265.27 procent. Dla H = 50 jest to około 3.65 razy większy MSE niż brak zmiany. Model dt = 10 również ma ujemną redukcję MSE: −23.01 procent na H = 10 i −70.63 procent na H = 50. Nie uzyskano dobrego przenoszenia dynamiki na drugą klatkę.

Wynik rozpoznania jest dodatni, natomiast **ta próba Langevina bez PCA nie potwierdza poprawy jakości prognozy nad wszystkimi prostymi modelami odniesienia ani uogólnienia na drugi obraz**. Nie można na jej podstawie obiecać 90 procent trafności przyszłego ruchu ani ustalić, że większa liczba kanałów zawsze pomaga. Porównanie z poprzednim celem ośmiowymiarowym wymagałoby odrębnego, wspólnego celu oceny.

### Zakres kontroli

Niezależny audyt potwierdził niezmienione kanały i modele po zamrożeniu oraz 48 z 48 kontroli niezależności prognozy od prawdziwej przyszłości. Wszystkie 128 kanałów pozostaje celem obliczeń. Równoważność obliczeń na danych skalowanych i w oryginalnych jednostkach sprawdzono numerycznie. Pliki `MACIERZE_DT1.npz` i `MACIERZE_DT10.npz` zawierają dopasowane równania, a `PROGNOZY_DT*.npz` i `METRYKI_DT*.json` prognozy i wyniki.

Promienie spektralne wybranych operatorów wynoszą około 1.00148 i 1.00570. Skończone prognozy na ocenionych horyzontach nie dowodzą zatem asymptotycznej stabilności ani poprawności dowolnie długiego przewidywania. Duże macierze są regularyzowanymi predyktorami; ta ilość danych nie uzasadnia fizycznej interpretacji każdego współczynnika. Przedziały dynamiki są opisowe, z jednego przebiegu i małej liczby bloków czasowych, nie zastępują powtórzeń na niezależnych treningach.

W resztach jednoetapowych pozostają wyraźne zależności czasowe. Dla dt = 1 diagnostyka Ljunga–Boxa daje surowe p < 0.05 w 4084 z 4096 badanych szeregów współrzędnych. To diagnostyka bez korekty wielokrotnych testów, a nie dowód dokładnej postaci szumu. Nie wykazano białych reszt ani domknięcia modelu. Normalność rozkładu i białość w czasie badano oddzielnie; jedna własność nie oznacza drugiej.

## Pliki do obejrzenia

- `ROZPOZNANIE_VS_K.png`: dobór liczby kanałów i pojedynczy wynik testowy rozpoznania.
- `ZNAK_PROGNOZY_VS_H.png`: procent poprawnie przewidzianych znaków zmian w funkcji horyzontu.
- `SKILL_PROGNOZY_VS_H.png`: osobno redukcja błędu, z zachowaniem wyników ujemnych.
- `TRAJEKTORIE_NATYWNE.png`: przykładowy ruch cząsteczek w oryginalnych kanałach.
- `WYNIKI.md`: pełny opis, wszystkie numery osi, równania i porównania.
- `AUDYT_ASTRA.md`: niezależna kontrola doboru, braku przyszłych informacji i zakresu wniosków.

Trafność rozpoznania oraz trafność prognozowania ruchu to dwa różne wyniki. Samo przekroczenie 90 procent w rozpoznaniu nie gwarantuje takiej skuteczności równania Langevina.
