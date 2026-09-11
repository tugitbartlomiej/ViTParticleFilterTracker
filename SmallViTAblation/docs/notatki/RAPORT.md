# Audyt selekcji danych i ewaluacji RT-DETR / RF-DETR

> Archiwum audytu sprzed refaktoru kodu. Aktualne poprawki i testy: [21_STAN_REFAKTORU.md](../21_STAN_REFAKTORU.md). Numery linii i wyniki reprodukcji odnoszą się do zapisanej wersji źródeł.

Data: 2026-09-11. Zakres: `SmallViTAblation/pipeline`, oryginalny `AdvancedDatasetSelection`, `DETR_IEEE_FINAL.tex` i `DETR_IEEE_FINAL.pdf`. Badanie obejmuje analizę kodu, kontrolę lokalnych manifestów, testy syntetyczne CPU i literaturę. Nie uruchomiono treningów GPU, nie przeliczono historycznych checkpointów i nie zmieniono kodu produkcyjnego.

**Werdykt: pomysł selekcji ma odpowiedniki naukowe, ale obecna implementacja oraz opis eksperymentu nie wystarczają do wiarygodnego porównania metod. Najpierw trzeba poprawić ewaluację, kontrakt danych i adapter modelu.** Dopiero potem warto uruchamiać kosztowną macierz ablacji.

## Jak korzystać z wyników audytu

- [MAPA_UWAG_RECENZENTOW.md](MAPA_UWAG_RECENZENTOW.md): uzupełnienie po lekturze obu recenzji; 30 punktów, wymagane dowody, niepotwierdzone deklaracje w projekcie odpowiedzi i plan eksperymentów RT-DETR. [Dodatkowe źródła o kalibracji i Soft-NMS](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/linki/kalibracja-detekcji-i-soft-nms-2026-09-11.txt).
- [PROTOKOL_RT_DETR.md](PROTOKOL_RT_DETR.md): kompletny proponowany sposób selekcji, treningu, pomiaru metryk i niepewności.
- [AUDYT_EWALUACJI_TRENINGU.md](AUDYT_EWALUACJI_TRENINGU.md): miejsca w kodzie, reprodukcje i ograniczenia ustaleń dotyczących modeli i metryk.
- [AUDYT_SELEKCJI.md](AUDYT_SELEKCJI.md): ekstrakcja cech, EL2N, cache, reprezentanci, baseline'y i materializacja.
- [ZRODLA.md](ZRODLA.md): 19 publikacji i 8 źródeł implementacyjnych, z zakresem lektury i ograniczeniami przenoszenia wyników.
- [linki_artykuly.txt](/F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/linki/artykuly-rt-detr-selekcja-i-ewaluacja-2026-09-11.txt), [sources.csv](../pochodne/sources.csv), [sources.json](../pochodne/sources.json): bibliografia do dalszego wykorzystania.
- `pdf_all/`: PDF-y (w tym 10 przeniesionych z `papers/`); `papers/`: historyczne ekstrakcje tekstu; `Ocr/`: skrypty i wyniki programu MinerU użytkownika; `search_logs/`: rzeczywiste wyniki poleceń paper-search; `evidence/`: testy, wyniki, hashe i fragmenty implementacji referencyjnych.

Nie należy traktować surowych trafień wyszukiwarki jako bibliografii. Wyszukiwanie wielowyrazowe w paper-search zwróciło również nietrafne wyniki; ostateczny dobór został ręcznie zweryfikowany. Semantic Scholar częściowo ograniczał żądania kodem 429. Wykorzystano arXiv, Crossref, oficjalne strony publikacji oraz repozytoria autorów. Nie twierdzę, że wszystkie znalezione prace przeczytano w całości.

## 1. Najważniejsze potwierdzone błędy nowego pipeline'u

| Problem | Dowód / miejsce | Konsekwencja |
|---|---|---|
| Znikanie obrazów bez predykcji z mAP | `pipeline/eval/common.py:245`, `imgIds = gt_ids & dt_ids` | Pominięte obiekty mogą przestać obniżać recall; wynik zostaje zawyżony |
| Odtwarzanie zbioru ocenionych obrazów z samych predykcji | `pipeline/eval/evaluate.py:177` | Przy braku pełnego manifestu brak detekcji może być pomylony z brakiem ewaluacji obrazu |
| Dobieranie progu na teście przy braku pliku walidacyjnego | `pipeline/eval/evaluate.py:285` | Optymistyczne F1 na teście; ostrzeżenie nie blokuje niedozwolonej ścieżki |
| Nieprawidłowy matching nakładających się GT w transferze | `pipeline/transfer/metrics.py:76` | Kod sprawdza zajęty najlepszy GT zamiast kolejnego wolnego; zaniża TP i zawyża FP/FN |
| Ignorowanie/przepisywanie klas | `pipeline/transfer/metrics.py:144`, `pipeline/eval/common.py:211` | Niepoprawna klasa może zostać uznana za poprawną detekcję |
| Nieprawidłowe rozpoznawanie ostatniej epoki | `pipeline/train/train_rfdetr.py:394`, `:410` | Sortowanie nazw może wskazać `checkpoint_best_total.pth` jako `last.pth` |
| Brak rozdzielenia artefaktów checkpointów | `pipeline/eval/evaluate.py:105`, `:329` | Możliwe nadpisanie predykcji/progu, a ponowna agregacja może odczytać własny zbiorczy plik jako blok |
| Niespójna integracja RF-DETR | Szczegóły i wersje w audycie treningu | Argumenty API, format `sv.Detections`, konfiguracja i ładowanie wag nie są wiarygodnie weryfikowane |

**Reprodukcja mAP:** dwa obrazy, po jednym GT, idealna predykcja tylko pierwszego. Obecna funkcja zwraca AP50 około 1 i AR=1; COCOeval obejmujący oba obrazy daje AP50 około **0,50495** i AR=**0,5**. AP50 nie wynosi dokładnie 0,5 z powodu 101-punktowej interpolacji COCO. To test matematyki evaluatora, nie pomiar skuteczności modelu.

Ten błąd dotyczy nowego `SmallViTAblation`. **Nie jest dowodem, że stary artykuł używał tego samego wadliwego kodu.** Również nie każdy rzeczywisty bieg musi ujawnić błąd: jeśli każdy obraz ma choć jedną predykcję, przecięcie identyfikatorów nie usuwa obrazów. Nie można jednak uzależniać poprawności evaluatora od tego warunku.

Odtworzenie: [reproduce_eval_bugs.py](../pochodne/evidence/reproduce_eval_bugs.py), [eval_repro_results.json](../pochodne/evidence/eval_repro_results.json). Zgodność bibliotecznego API sprawdzano m.in. wobec oficjalnego RF-DETR 1.2.1, nie potwierdzono wersji uruchamianej na Edenie. Brak smoke testu z prawdziwą biblioteką pozostaje ograniczeniem.

## 2. Nowa selekcja nie odtwarza automatycznie starej metody

**Są też bezpośrednie blokady wykonania:** konfiguracja ekstraktora jest w niewłaściwym miejscu względem odczytu, `DinoExtractor` nie udostępnia oczekiwanego wymiaru `dim/DIM`, a usuwanie cyfr z nazwy rodziny kieruje `dinov3` do repozytorium DINOv2. Testy odtworzyły `KeyError: dino_model`, brak `DIM` i błędny wybór repozytorium. Działający tryb `--fake-features` nie potwierdza działania ekstrakcji produkcyjnej.

Dodatkowo nowy wariant nazwany k-center ponownie wybiera reprezentantów po klastrowaniu i może zmienić punkty wybrane przez algorytm greedy. Test jednowymiarowy pogorszył promień pokrycia z 13 do 15. W oryginalnym pipeline odczyt cache może zastąpić żądaną listę obrazów listą z cache, a nowy selector akceptuje metadane fake. Cache musi być związany z manifestem, splitami, ekstraktorem i jego parametrami; sam istniejący plik nie potwierdza zgodności danych.

Artykuł opisuje DINOv3 1024 -> PCA 32, Fourier **9** wymiarów, FastSAM **1** wymiar i trudność **1** wymiar. Nowa konfiguracja przypisuje Fourierowi **1**, a FastSAM **9** wymiarów. Oprócz innego ekstraktora/sygnału zmienia się udział bloków w odległości.

Artykuł definiuje skalowanie `sqrt(w_g/d_g)`, natomiast `select_subset.py` stosuje `z-score -> PCA -> mnożenie przez w_g`. Zmienia to geometrię i znaczenie zadanych wag. Nie wolno opisać wyników nowej macierzy jako wiernej reprodukcji starego DESF bez odtworzenia reprezentacji i kolejności przekształceń.

`el2n_from_proxy.py` używa `1-confidence` dla wykrycia dopasowanego po IoU i domyślnego wyniku 1 dla obrazu bez GT. To sygnał detekcyjny, nie bezpośrednia implementacja EL2N z cytowanej publikacji. Traktowanie poprawnie rozpoznanego tła jako maksymalnie trudnego zaciera rozróżnienie między trudną negatywną klatką a łatwym tłem. Przy klastrowaniu taki score zmienia geometrię; nie oznacza sam przez się monotonicznego preferowania tła.

W badanej kopii lokalnych anotacji walidacja V1 ma 10 780 obrazów i 10 921 adnotacji, test 9240 obrazów i 9333 adnotacje. Założenie jednej ramki na obraz nie jest więc bezpiecznym ogólnym kontraktem, nawet przy jednej klasie.

Szczegóły oraz odtwarzalne przykłady: [AUDYT_SELEKCJI.md](AUDYT_SELEKCJI.md), [selection_repro_results.json](../pochodne/evidence/selection_repro_results.json).

Weryfikacja końcowa: niezależnie powtórzono 5 reprodukcji ewaluatora oraz 13 kontroli selekcji (12 odtworzeń problemów/rozbieżności i jeden poprawnie działający warunek train-only). Wszystkie oczekiwane zachowania odtworzono, bez nieoczekiwanych błędów i bez zmiany plików źródłowych. **To potwierdzenie diagnozy, nie naprawa.** Powtórzenie selekcji zapisano jako [selection_repro_recheck.json](../pochodne/evidence/selection_repro_recheck.json).

## 3. Co potwierdza sprawdzenie artykułu i manifestów

PDF i TEX zawierają te same istotne informacje o podziale, progach i ablacji. Zweryfikowano tekst oraz wizualnie strony 7 i 11 PDF. Wersje źródeł utrwalono w [file_hashes.json](../pochodne/evidence/file_hashes.json).

**Niekontrolowane porównanie.** TEX w linii 545 przypisuje różnice strategii selekcji, natomiast linia 595 przyznaje, że pełny pipeline miał inne partycjonowanie danych i regułę stopping niż ablacje. Różnica 26,04% vs 0,23% dotyczy zatem odmiennych recept selekcji i treningu. To nie pozwala wyznaczyć czystego efektu selekcji. Nie da się także z samej wielkości ilorazu dowieść błędu: przy baseline bliskim zera ilorazy są niestabilne. W nowej publikacji lepiej podawać różnicę w punktach procentowych z niepewnością.

**Progi i kalibracja.** TEX linie 341-343 opisują predykcje z progiem 0,3, a linia 631 wiąże słabe AP z kalibracją i utratą słabo ocenionych trafień. To wymaga oddzielenia AP z zachowanych standardowych predykcji od F1 przy punkcie pracy. Własny test COCO: idealny bbox z confidence 0,2 daje AP=1 przy cutoff 0,001 i AP=0 po odrzuceniu progiem 0,3. Zapis: [coco_threshold_demo.json](../pochodne/evidence/coco_threshold_demo.json). To pokazuje mechanizm, nie odtwarza wyników artykułu. Lokalizacja dowolnego zapytania ignorująca confidence nie jest wdrożeniowym recall ani diagnozą kalibracji.

**Split po klatce.** TEX linia 302 jawnie opisuje random frame-level split. Nie mierzy on tej samej generalizacji co split po nowych pacjentach/operacjach. Aktualny plik 35/8/7 jest dobrym kandydatem na wspólny split, ale jego istnienie nie dowodzi, że historyczny checkpoint był na nim trenowany.

**Przeciek zewnętrzny w produkcyjnym pliku anotacji.** Odczyt `AdvancedDatasetSelection/output/selected_dataset1/merged_20k_annotations_fixed.json` dał:

| Właściwość | Wynik |
|---|---:|
| Obrazy | 20 000 |
| Różne nazwy źródeł po usunięciu suffixu augmentacji | 15 565 |
| Obrazy augmentowane | 14 259, czyli 71,295% |
| Nadmiarowe kopie tych samych źródeł | 4435 |
| Nazwy zgodne z listą przecieków zewnętrznych w projekcie | 48 |
| Przynależność do aktualnego podziału: train / val / test / poza mapą | 15 648 / 2322 / 1982 / 48 |

Wynik **71,3% i 15 565 zgadza się z TEX**, a nie z wcześniejszą notatką o 75,3% dla innego pliku wyboru. Nie należy mieszać tych artefaktów. Dowód: [production_pool_audit.json](../pochodne/evidence/production_pool_audit.json). Są to kontrole nazw i bieżących manifestów; nie pełny audyt hashy obrazów ani dowód, że każda wymieniona klatka wpłynęła na konkretny historyczny checkpoint. Do tego potrzebne są manifesty rzeczywistych biegów.

**Deklaracja zero-look wymaga ograniczenia.** Artykuł prezentuje trajektorie wielu checkpointów na zewnętrznym benchmarku i audyt wszystkich zapytań. Mogą to być analizy post hoc, jeśli nie wpłynęły na wybór metody, ale bez zapisu chronologii nie dowodzą ścisłego zamrożenia wyboru. Bieżąca lista 48 przecieków dodatkowo wymaga wyjaśnienia. Usunięcie 48 obrazów daje jedną analizę czułości; nie dowodzi rozłączności operacji i nie cofa wcześniejszego dostępu do wyników.

## 4. Najbliższe odpowiedniki w literaturze

| Metoda | Co jest podobne | Czego nie wolno zakładać |
|---|---|---|
| [CSOD, Lee et al., 2024](https://arxiv.org/abs/2404.09161) | Selekcja już opisanych obrazów na podstawie cech obiektów i pokrycia | Badano przede wszystkim Faster R-CNN; nie jest to gotowy selector RT-DETR |
| [OFDS, Popp et al., v2 2025](https://arxiv.org/abs/2412.10032) | Cechy obiektowe i klastrowanie; dodatkowo eksperyment z Deformable DETR | Oryginał wybiera dane do etykietowania. Użycie cropów GT to nasza adaptacja |
| [COWAL, Wu et al., WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Wu_Correlation-Aware_Active_Learning_for_Surgery_Video_Segmentation_WACV_2024_paper.html) | Wideo chirurgiczne, redundancja, klastry i niepewność | Segmentacja i aktywne uczenie; nie wynik dla detekcji końcówki |
| [AdaDeDup, Kang et al., 2025](https://arxiv.org/abs/2507.00049) | Dedup, klastry, informacja zwrotna z proxy i różny stopień redukcji | Praca nie ustanawia uniwersalnego przepisu na redukcję do 22% oryginalnej puli |
| [Foundation-Assisted AL, Zhang et al., 2026](https://arxiv.org/abs/2607.16671) | RTDETRV2, obiektowe DINOv2, SAM2 i łączenie niepewności z różnorodnością | Zdjęcia lotnicze. Hungarian łączy propozycje detektora i modelu foundation, nie GT jak w proponowanym scorerze |

Ostatnia praca jest **preprintem**, używa także UPN i pretrainingu FMOW. W sekcji IV-A podano split FAIR1M 40%/20%/20%, który nie sumuje się do 100%; w opisie DSBS występuje niejednoznaczność Recall@100/IoU75 vs mAP. Jest bliską inspiracją techniczną, lecz nie wzorcem, który należy kopiować bez weryfikacji. Priorytet dla podstaw metryk mają oficjalny evaluator i [Metrics Reloaded](https://doi.org/10.1038/s41592-023-02151-z).

## 5. Korekty wcześniejszych zaleceń

1. **Train-on-train scoring nie oznacza automatycznie wycieku.** Oryginalny EL2N ocenia przykłady treningowe wcześnie w treningu. Należy kontrolować ekspozycję próbek, ziarna i brak kontaktu z testem, nie narzucać OOF jako definicji metody. Wersja arXiv v2 zawiera również korektę wcześniejszych wyników GraNd na inicjalizacji. [Paul et al.](https://arxiv.org/abs/2107.07075).
2. **Duży wynik względem random nie jest sam w sobie sprzeczny z literaturą.** Problem tutaj to udokumentowane różnice protokołu i błędy implementacji. Random pozostaje koniecznym baseline'em; przewagę trzeba dopiero zmierzyć. [DeepCore](https://arxiv.org/abs/2204.08499).
3. **K-means + centroid jest poprawną metodą referencyjną.** Nie zapewnia jednak równych rozmiarów klastrów ani optymalnego pokrycia rzadkich zdarzeń. Nowość i skuteczność połączenia czterech sygnałów wymagają testu, nie wynikają z samego istnienia podobnych publikacji.
4. **Same seedy nie mierzą całej niepewności.** SD między treningami jest czym innym niż CI generalizacji na nowych operacjach. Bootstrap po klatkach może zawęzić przedział przez nieuwzględnienie korelacji. [Bouthillier et al.](https://arxiv.org/abs/2103.03098).
5. **RF-DETR to nie RT-DETR.** Migracja wymaga nowego adaptera modelu, sprawdzenia loss, identyfikatorów klas i konwersji boxów. Zmiana nazwy w YAML nie wystarczy.

## 6. Zalecana kolejność dalszej pracy

1. Poprawić wspólny evaluator i import predykcji, zabronić wyboru progu na teście, rozdzielić checkpointy w artefaktach. Potwierdzić wszystkie przypadki syntetyczne wobec COCOeval.
2. Przypiąć rzeczywistą wersję modelu i bibliotek. Naprawić inicjalizację, format predykcji, layout danych i checkpointy; sprawdzić konfigurację efektywnie używaną, nie tylko zapisaną w YAML.
3. Zbudować train/val/test z oryginałów po pacjentach/operacjach. Wyjaśnić 48 zewnętrznych nazw i historię checkpointów. Ustalić wykonalny budżet unikalnych źródeł.
4. Ustalić, czy celem jest wierna reprodukcja DESF, czy nowa metoda obiektowa. Przy reprodukcji odtworzyć Fourier/SAM/wagi/PCA; przy nowej metodzie oznaczyć te zmiany jako osobne eksperymenty.
5. Uruchomić mały pilotaż random i selekcji obiektowej. Dopiero po poprawnej kontroli całego łańcucha rozszerzyć macierz i przeprowadzić zamrożoną ocenę końcową.

Pełna specyfikacja metryk, budżetów, proxy i adaptera: [PROTOKOL_RT_DETR.md](PROTOKOL_RT_DETR.md). Wszystkie zalecenia zmieniające metodę są propozycjami do sprawdzenia; audyt nie wykazał jeszcze, która metoda daje najlepszy model.
