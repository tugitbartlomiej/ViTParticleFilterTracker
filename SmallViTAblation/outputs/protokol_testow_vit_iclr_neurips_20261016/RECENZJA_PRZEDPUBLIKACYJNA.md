# Druga recenzja planu przed publikacją — 16 października 2026

**Werdykt: sensowny program badawczy, ale sam dodatni wynik T09 nie wystarczy do tezy „wyjaśniamy mechanizm ViT”.** Doprecyzowano sześć kwestii w protokole, katalogu testów i kontrakcie kampanii. Nie wykonano żadnego testu modelu ani nowego treningu. To ocena projektu badań, nie ocena gotowości wyników do akceptacji.

Przegląd objął matematykę, przyczynowość, statystykę i koszt Edena; trzy niezależne przeglądy pomocnicze porównano z kodem model.py i literaturą. Poprzednie poprawki dotyczące LN, normy sham, osiągalności GT przez kotwicę oraz eksportu wyników pozostają w mocy.

## 1. Najważniejsze poprawki

| Problem | Dlaczego ma znaczenie | Zmiana |
|---|---|---|
| Zmiana ramki może wynikać wyłącznie z kierunkowej czułości | Dwie poprawki tej samej normy mogą inaczej ustawić się względem pochodnej głowy | P1 nazywamy testem kierunkowej podatności; mocniejszy claim wymaga prospektywnej prognozy i związanej z nią ścieżki T11/T15 |
| Brak jawnego tokenu interwencji | W ostatnim MLP zmiana innego tokenu niż odczytywany daje strukturalne zero w jego współrzędnych | Podstawowy T09: j=i0; inne tokeny w oddzielnym teście, z kontrolą zerową ostatniego MLP |
| Stały kandydat może zmieniać się między checkpointami | P2 może mieszać ewolucję reprezentacji i zmianę wyboru kotwicy | Stałość w obrębie seed×checkpoint×obiekt; osobna kontrola wspólnej kotwicy GT dla claimu o uczeniu |
| Rescue może być tautologią | Przywrócenie całego stanu albo dokładne cofnięcie poprawki przywraca obliczenie konstrukcyjnie | Takie operacje są kontrolą dodatnią; dowód wymaga ograniczonego komponentu i donorów rozdzielających informację od kontekstu |
| Wybór mediatorów i donorów nie był w pełni zamrożony | Oglądanie filmów confirmation może prowadzić do wyboru najładniejszej historii | Jawne hashe reguł wyboru, budżetów, warstwy odłożonej i agregacji; bez strojenia po odczycie confirmation |
| Pełny atlas i null-treningi powiększają budżet | Pięć nowych treningów podstawowych nie obejmuje dodatkowych kontroli zadania | Zawężony rdzeń confirmation oraz osobna wycena random-label/loss ablations |

Dodatkowo zamrożono interpretację S/A w natywnej euklidesowej metryce residualu na ilorazie symetrii LN. Zmiana dowolnej metryki może zmienić rozkład; nie deklarujemy niezależnego od parametryzacji prawa fizycznego. Nie potrzeba z tego powodu dodawać całej kampanii whitening.

## 2. Co rzeczywiście pokazuje T09

Dla poprawki q=−αUBUᵀδ oraz pochodnej dalszej części modelu C lokalna odpowiedź wynosi w pierwszym rzędzie Δy≈Cq. Samo dopasowanie ||q|| nie dopasowuje Cq. Ukierunkowana poprawka może więc silniej zmieniać bbox nawet przy całkowicie liniowym odczycie.

To jest wartościowy pomiar podatności konkretnego obliczenia, lecz znaczenie „sieć używa tej organizacji do tłumienia zakłóceń i zachowania lokalizacji” wymaga dodatkowego przewidywania. Przed confirmation trzeba zapisać:

- konkretną postulowaną informację/operację oraz jej zakres;
- komponent i zamrożony sposób jego wskazania w nowym modelu;
- podpisaną prognozę zmiany wybranych krawędzi/odczytów, zależność od dawki i warunek małego/brakującego efektu;
- kontrast, w którym konkurencyjne wyjaśnienie daje inną prognozę albo równie trafne wyjaśnienie wymaga większej liczby komponentów;
- minimalny praktyczny efekt i regułę uznania wyniku za wspierający, obalający daną wielkość albo nierozstrzygający.

Prognozy zapisujemy przed wyliczeniem odpowiedzi na daną interwencję. M0 obejmuje czułość, pełny lokalny JVP i rzeczywisty dekoder. Nie żądamy „nowej informacji ponad pełny Jacobian”: rozkład jest jego funkcją. Potencjalna wartość M1 polega na oszczędnym, sprawdzalnym wyjaśnieniu o jawnym zakresie, popartym naturalnymi kontrastami i mediacją.

Ryzyko utożsamienia skutecznego sterowania z lokalizacją normalnie używanej informacji jest opisane przez Makelova, Langego i Nandę (2023). Autorzy pokazują m.in. możliwość uruchomienia wcześniej nieaktywnej ścieżki. Przenosimy to jako kontrolę interpretacji, nie twierdzenie o wyniku naszego ViT. [Źródło pierwotne](https://arxiv.org/html/2311.17030v2).

## 3. Token, odczyt i zmiana podczas uczenia

Kod po ostatnim bloku wykonuje per-token LayerNorm i per-token głowę. Dlatego dla j≠i0 interwencja w ostatnim MLP nie zmienia bezpośrednio współrzędnych ustalonego kandydata i0. Może natomiast zmienić ranking lub maskę lokalnych maksimów, a przez to końcowy wybór detekcji. Te dwa odczyty muszą pozostać osobne.

Podstawowy T09 dotyczy j=i0, z i0 wybranym na clean forward danego seed×checkpoint×obrazu i zamrożonym dla wszystkich jego ramion. Ta sama reguła dotyczy wcześniejszych bloków, ale nie oznacza, że zawartość tokenu reprezentuje wyłącznie ten patch. Efekty j≠i0 we wcześniejszych blokach bada się oddzielnie przez downstream attention.

Dla P2 oprócz reguły clean-winner wykonujemy z góry zadeklarowaną kontrolę tej samej kotwicy obiektu i tej samej populacji obrazów przez porównywane stany. Kotwicę wybieramy deterministycznie na podstawie GT, ze stałą regułą geometrycznej osiągalności i raportem pokrycia; jest to pomiar mechanistyczny z etykietami, nie samodzielny detektor bez GT. Bez takiej kontroli wynik P2 opisuje łącznie zmianę modelu i reguły wyboru kandydata.

## 4. Nietautologiczna ścieżka i rescue

Za kontrolę dodatnią uznajemy cofnięcie dokładnie własnej poprawki oraz przywrócenie całego późnego stanu. Nie liczymy ich jako głównego dowodu małego mechanizmu.

T11/T15 muszą odnosić się do tego samego postulowanego procesu co T09. Z góry wybieramy ograniczony komponent/ścieżkę, wielkość podmiany i metodę dopasowania do nowego seeda. Resztę modelu rzeczywiście przeliczamy. Donorzy powinni, w zwalidowanych parach, rozdzielać informację docelową od kontekstu: zachować postulowaną zmienną przy innym kontekście lub zmienić ją przy możliwie podobnym kontekście. Hipoteza przewiduje odzysk określonej odpowiedzi i brak niepożądanej zmiany pozostałych aspektów. Zasięg wniosku ogranicza jakość tych par i zgodność interwencji z naturalnymi stanami.

Miller, Chughtai i Saunders (2024) pokazują wrażliwość wyników faithfulness na sposób ablacji. Dlatego zapisujemy granularity, component, replacement value, token positions, direction i set, a nie tylko nazwę „patching”. [Źródło pierwotne](https://arxiv.org/html/2407.08734v1).

## 5. Rdzeń do policzenia na Edenie

Po E0 proponowany rdzeń obejmuje jeden komponent wybrany na discovery, jawny kontrast zadaniowy, dawkę i sham, rzeczywisty dekoder, ograniczoną mediację/rescue, kontrolę inicjalizacji oraz replikację na nowych filmach i seedach.

Mały pilot T09 może liczyć J tylko dla i0: 24 obrazy × 3 checkpointy × 3 bloki = **216 lokalnych Jacobianów**, zamiast wykonywania całego atlasu 12 096 J na tym etapie. To wyłącznie liczba Jacobianów źródłowych, bez walidacji, sham, full suffix, attention, kontrolnych kotwic i baseline'ów. Nie jest estymacją całego czasu GPU.

Po discovery confirmation może ograniczyć kosztowny pomiar do zamrożonego operatora, wybranych bloków oraz init/late. Nie zmniejszamy w zamian niezależności seedów/filmów ani właściwych kontroli. Pełny atlas jest nadal potrzebny, jeśli artykuł ma twierdzić coś o przebiegu całej głębokości i czasu. Gęsty cache mikroskopu pozostaje na małej uprzednio wybranej próbce.

Osobno wyceniamy:
1. dwa nowe treningi discovery oraz początkowo trzy confirmation;
2. warunkowe treningi random-label i loss ablations, z własną liczbą seedów i długością okna;
3. eksport, transfer, retencję i odczyt lokalny.

Jeśli null-trening trwa tylko krótko, porównujemy go z odpowiadającym krótkim oknem treningu właściwego. Nie przedstawiamy takiego wyniku jako kontroli pełnego końcowego mechanizmu. Bez odpowiedniej kontroli zadania claim ograniczamy do zmiany względem inicjalizacji.

## 6. Ostateczna bramka przed wysłaniem artykułu

Ponowna recenzja **rzeczywistych pobranych wyników** jest częścią T29. Lista do wykonania:

- Czy główna teza jest identyczna z zamrożoną kartą, a wszystkie odstępstwa są opisane?
- Czy prognozy zapisane przed interwencją zgadzają się z wynikami na odłożonych filmach i seedach?
- Czy efekt praktyczny ma odpowiednią niepewność, a strukturalne zera, braki, nieudane kontrole i wyniki ujemne są policzone?
- Czy dokładnie ten sam proces wspierają T09, ścieżka i nietautologiczny rescue?
- Czy prostsze wyjaśnienie czułości/dekodera nie wystarcza do wszystkich deklarowanych wniosków?
- Czy powstała nowość jest porównana z najbliższymi pracami, w tym coupling i korektą bloków?
- Czy tabele, figury i film dają się odtworzyć z pobranego pakietu, a zakres filmu odpowiada faktycznie zapisanym stanom?
- Czy zakres tytułu i abstraktu odpowiada liczbie modeli, zadań i interwencji rzeczywiście zbadanych?

Dodatni wynik pojedynczego testu nie jest automatycznym zaliczeniem tej bramki. Protokół nie obiecuje akceptacji ICLR/NeurIPS.

## 7. Źródła i zakres ponownego wyszukiwania

W tej rundzie SciSpace zwrócił 10 kandydatów; nie wykorzystano ich abstraktów jako nowych dowodów dla ViT. Scite potwierdził metadane dwóch prac poniżej, ale nie zwrócił pól fulltextExcerpts ani editorialNotices, również po odczycie DOI. Brak tych pól oznacza brak informacji, nie potwierdzenie braku korekt/retrakcji. Wnioski metodologiczne sprawdzono w pełnych tekstach autorów na arXiv. Nie użyto liczby cytowań jako kryterium poprawności.

References:

- Makelov, A., Lange, G., & Nanda, N. (2023). *Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching*. [arXiv v2](https://arxiv.org/html/2311.17030v2).
- Miller, J., Chughtai, B., & Saunders, W. (2024). *Transformer Circuit Faithfulness Metrics Are Not Robust*. COLM. [Pełny tekst](https://arxiv.org/html/2407.08734v1).

Pozostałe podstawy i status odczytu są w [LITERATURA.md](../../experiments/protokol_testow_vit_iclr_neurips_20261016/LITERATURA.md). Ten raport jest własną recenzją projektu; proponowane warunki i decyzje nie są oficjalnymi wymaganiami konferencji.

