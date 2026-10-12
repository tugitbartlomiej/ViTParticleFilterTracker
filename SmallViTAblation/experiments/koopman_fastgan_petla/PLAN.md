# Koopman, FastGAN i rzeczywiste klatki: pilot petli

Użytkownik zatwierdził implementację, małe testy i wykonanie kolejnych etapów na Edenie 2026-09-24.
Źródło metody: `../fastgan_narzedzie/PIPELINE_LATENT_KOOPMAN_FASTGAN.md`.

## Ustalony zakres

RF-DETR Nano z wag COCO, 1500 istniejących klatek uniform_time, cztery epoki wstępnego dostrojenia.
Tempo aktualizacji obniżone trzykrotnie, stałe w całym pilocie, backbone trenowany.
Jeden seed, maksymalnie trzy rundy i jedna prawdziwa klatka na rundę. Brak dowodu ogólnej przewagi przy jednym seedzie.
Jeden worker GPU na Edenie; lokalnie testy CPU. Generator już wytrenowany, jego wag nie uczymy.

Epoka ma budżet 1600 prezentacji (1500 różnych klatek i deterministyczne dopełnienie), 100 aktualizacji
przy paczce efektywnej 16. Co 20 aktualizacji zapis pomiaru i wag. Pełny stan wznawiania na końcu epoki.
Wybór tej siatki zachowuje równe odstępy czasowe potrzebne macierzy przejścia.

## Etapy i podłogi

1. Sonda z rzeczywistych masek narzędzia; podział operacjami na fit/holdout.
   Stałe obroty, skala, długość, przesunięcia, światło, perspektywa 2D, tła i kontrola bez narzędzia. Przechowujemy raw post-LN tokeny,
   a do prognozy agregaty i nadzorowane kierunki. Nie używamy niezależnie dopasowanej PCA każdego checkpointu.
2. Oddzielny DINOv2 i dokładny FAISS na treningowej puli wycinków.
   Walidacja/test nie są kandydatami do wyszukiwania.
3. Wstępny trening; powtórzenie sondy bez zmiany wag sprawdza błąd numeryczny.
4. Kierunki i Koopman dopasowane tylko przed granicą czasową; późniejsze kroki i odłożone operacje do oceny.
   Kontrole: stała odpowiedź i trend. Dobra prognoza jest bramką przed generowaniem pod przyszły cel.
5. Zamrożony FastGAN; gradient wyłącznie po jego wejściu. Cele: Koopman, obecny opis, losowy przyrost o równej normie.
   Te same starty i liczba kroków. Maska generatora nie jest etykietą prawdy.
6. Wyszukiwanie wizualne FAISS; bez kryterium zgodności z latentem RF-DETR. Ocena wygenerowanego obrazu i rankingu.
7. Kontynuacje z tego samego stanu: bez dodania, wybrana klatka i losowa klatka kontrolna.
   Równa liczba aktualizacji i prezentacji nowej klatki. Sparowany strumień poza zastąpionym slotem.
8. Przyrosty na tych samych obrazach oraz detekcja na walidacji. Na końcu test obu gałęzi, bez strojenia testem.

Jeżeli bramka prognozy lub jakość obrazów zawiedzie, zapisujemy wynik negatywny i kończymy zależny etap.
Nie wymuszamy trzech rund wbrew niespełnionym warunkom. Nie nazywamy podobieństwa obrazu użytecznością treningową.

## Decyzje implementacyjne

- Pełny zbiór wektorów tożsamości narzędzia jest nazwą celu; sonda daje kandydacką podprzestrzeń.
- Równania nie zawierają Langevina ani filtra Kalmana w pierwszej wersji.
- Pilota nie mieszamy z projektami innych sesji. Nie usuwamy ich artefaktów i procesów.
- Wznawiamy od granicy epoki: gęste checkpointy wewnątrz epoki służą obserwacji,
  nie deklarujemy dokładnego odtworzenia iteracji dataloadera z ich środka.
- Jeden pokaz dodatkowej klatki może dać efekt mniejszy od szumu. Raportujemy to, nie zwiększamy po cichu wagi próbki.
- Pełne schematy decyzji, progi i parametry znajdują się w config.json i manifeście wykonania.

## Dziennik

2026-09-24: utworzono osobny katalog, kod w trakcie implementacji. Brak nowych treningów.

2026-09-24: testy CPU i przeglądy modułów; skorygowano serializację RNG, ścisłe
wznawianie, porównanie strumieni, niezależny gradient cech oraz walidację pochodzenia sondy.
Na Edenie uruchomiono osobny test rzeczywistych modeli; trening naukowy czeka na jego wynik.
Przy każdej rundzie bramka prognozy dotyczy całej epoki, baza jest zamrożona po dwóch pierwszych snapshotach.

Test wznowienia wykrył pozostawiony przez Lightning kursor zakończonej paczki.
`granica.py` zapisuje pełną granicę przed następną epoką, zachowując sumaryczny
licznik aktualizacji, momenty AdamW, harmonogram i RNG. Surowy stan pętli zostaje
w checkpoincie do audytu. Kolejna próba wykonała wszystkie gałęzie techniczne;
dokończenie kontroli nie powtarza ich treningu.

Kontrola wag wykryła następnie niedeterministyczność GPU: przy identycznym RNG
i strumieniu danych maksymalna różnica wag dwóch kontynuacji wyniosła 0,0001084.
Nie zwiększono tolerancji zaliczenia. Nowy wariant wymusza math SDPA, ścisłą
deterministyczność i CPU grid_sample z autograd; wszystkie gałęzie są sprawdzane
od nowa. To dodatkowy koszt diagnostyczny, nie wynik porównania szybkości uczenia.

Test 1810438 na A100 zakończył się powodzeniem: różnica wag identycznych
kontynuacji i continuous–resume równa dokładnie zero; zgodne momenty optymalizatora,
harmonogram i RNG. Rzeczywisty gradient przez FastGAN i detektor jest skończony
i niezerowy. Działa wyszukiwanie FAISS oraz końcowa ocena testowa. Czas 4 min 35 s.
Przed właściwym uruchomieniem dodatkowo porównujemy kod tej samej kotwicy
z zapisu treningowego i po załadowaniu detektora do sterowania generatorem.

Kontrola 1810447 zakończyła cały przepływ poprawnie, także po włączeniu ścisłej
deterministyczności FastGAN-a. Zgodność kotwicy: skalowany RMSE 4,96e-6 przy
progu 1e-5. Adaptive pooling w gradientach generatora również liczymy na CPU.
Lokalne zestawy: 72 testy danych/dynamiki/wyszukiwania i 33 testy odczytu/generatora.
Właściwe zadanie 1810448 zgłoszone na partycję short, kod code_07.tar.
Obejmuje cztery epoki i maksymalnie trzy rundy zgodnie z powyższymi bramkami.

Pierwsza epoka właściwego pretreningu: mAP walidacji 0,497096. Zadanie kontynuuje
drugą epokę; jest to pomiar bazowego dostrajania, nie efekt doboru klatek.
Automatyzacja `wynik-eksperymentu-koopman-fastgan` sprawdza dalszy przebieg,
pobierze końcowe artefakty z kontrolą SHA256 i zgłosi wynik lub konkretny błąd.

### Zmiana miejsca wykonania na prośbę użytkownika

Pretrening zakończony: 400 aktualizacji, cztery epoki, walidacja około 0,6574 mAP.
To nadal wynik bazowy, przed doborem nowych klatek. Artefakty przejęto lokalnie
do `wyniki/local_1810448/import`; zweryfikowano SHA256 wszystkich 448 plików.
Stary orkiestrator Edena celowo zakończono po przekazaniu; checkpoint zachowano.

Koopman, FastGAN i wyszukiwanie DINOv2–FAISS działają odtąd lokalnie, przez
`hybryda_lokalna.py`. Podczas uruchomienia potwierdzono obliczenia indeksu na
RTX 3070. Dalsze sparowane treningi i ocena RF-DETR są zlecane na Edena przez
`hybryda_zdalne.py`. Zamrożony FastGAN nie jest ponownie uczony.
Wszystkie dotychczasowe kryteria i limit rund pozostają obowiązujące.
81 testów CPU dotyczących przekazania, przenoszenia stanu, transportu, pętli
i oceny zakończyło się poprawnie. Nie jest to jeszcze wynik rundy naukowej.

Przy przenoszeniu stwierdzono brak obrazów `test` w rzeczywistym manifeście puli.
Ocena pustej partycji zwraca jawnie `unavailable`, a nie pozornie poprawną metrykę.
Nie zmieniamy podziału danych po obejrzeniu walidacji. Ewentualne wnioski o poprawie
pozostają ograniczone do tej walidacji, dopóki nie ma osobnego testu.
