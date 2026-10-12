# Identyfikacja podprzestrzeni narzędzia: 400 par

Utworzono 2026-10-01 Europe/Warsaw, przed pomiarem. Badanie lokalne, eksploracyjne, bez treningu ViT. Rodzina badawcza: maly_vit_tozsamosc_czasteczki_pary_400. Lokalizacja jawnie wskazana przez użytkownika; nie rejestrujemy artefaktów w osobnym repozytorium grantowym.

## Co mierzymy i co obali interpretację

Cząsteczka to pełny token z_i w R^768 po GELU ostatniej MLP; wspólne makrowektory e_j są osiami tej samej bazy. 600 tokenów na pełny obraz 480×320. Delta z_i = oryginał minus LaMa na tych samych indeksach maski ORYGINAŁU. To reakcja na zmianę wejścia, nie ruch podczas uczenia. Poolingi służą klasyfikatorowi i opisowi osi; nie zastępują zapisanych pełnych wektorów tokenów.

Checkpoint k1000.pt z ukończonego źródłowego biegu: 1000 aktualizacji, nie numer epoki. Model identyczny z walidowanym modelem pomiaru pełnych tokenów. Eval, zamrożone BN, no_grad, fp32, brak autocast/TF32; RGB /255, wewnętrzna normalizacja ImageNet. Pełny obraz resize AREA480×320 bez cropu, ten sam preprocessing obu członów. Maski i skale źródła z audytu400par. Model, dane i kod mają SHA256.

Negatyw to syntetyczne usunięcie anotowanego cystotome. Nie jest prawdziwym tłem ani negatywem innych instrumentów. Jeżeli sygnał wystąpi tak samo w odległym tle lub sham edits, nie wolno przypisać go swoiście narzędziu. Jeżeli znaleziony kierunek nie rozdziela par na innych filmach, brak generalizacji. Sam największy ruch ani spadek detekcji po dowolnej perturbacji nie jest dowodem semantyki.

## Podział i estymacja

116 identyfikatorów filmu: deterministyczne tasowanie seed20261001, 69 train /23 validation /24 test. Wszystkie klatki filmu i oba człony pary w jednej części. Nie potwierdzono pacjentów; nie jest to zewnętrzny heldout detektora (obrazy mają membership train). Test nie uczestniczy w fit ani wyborze parametrów. Ostateczna interpretacja opisowa dla jednego checkpointu i jednego modelu inpaintingu.

Wagi fit: każdy film ma równą masę, obrazy filmu równą masę; tokeny narzędzia proporcjonalnie do area coverage oryginalnej maski. SVD kandydatów różnic przez niescentrowany drugi moment delta: zachowuje wspólny średni kierunek. Rank1/2/4/8/16, logistic regression C0.01/0.1/1/10 na ważonych średnich tokenów w wybranej przestrzeni. Standardyzacja tylko train. Wybór val AUROC, tie mniejszy rank i większa regularizacja. Porównanie regularizowanej LDA na768D: pooled train within-class covariance + ridge0.01/0.1/1 razy mean diagonal; wybór val. Kierunki SVD i LDA orientujemy dodatnio według średniej train delty; nie według testu.

Osi768 nie utożsamiać z konceptem: raport ranków mean paired delta /std po obrazach, znak, 0/1-based channel IDs, zgodność znaku między filmami oraz wagi mieszanych kierunków. r50/r90 to energia delta drugiego momentu train, nie liczba semantycznych cech.

Metryki obecności: AUROC, balanced accuracy przy progu wybranym wyłącznie validation, zwykła accuracy. Główne miary ważone jednakowo po filmach, dodatkowo image-weighted; bootstrap95% po filmach, nie po tokenach. Labelpositive=oryginał, negative=LaMa, ROI maska oryginału znana także przy klasyfikacji negatywu. Podział filmowy zachowuje pary; to zadanie znanego ROI, nie detektor pełnego obrazu. Raport osobno12binów kątów. Dodatkowa leave-angle-bin-out diagnostyka: refit U i standardyzacji oraz probe na train bez danego binu, ale rank i C ustalone na podstawowym val wszystkich kątów; ocena na test klatkach danego binu gdy >=3 klatki. Jest częściowa ekspozycja hiperparametrów na dany kąt przez podstawowy val, więc nie jest to rygorystyczny unseen-angle benchmark ani sposób wyboru metody na teście.

## Kontrole wpływu na detekcję

Na wszystkich obrazach z filmów test: nohook, shamnoop, identity removal z_o - U U^T(z_o-z_n) tylko na tokenach oryginalnej maski, pełna podmiana tych tokenów z_n, trzy losowe ortonormalne podprzestrzenie tego samego ranku (seeds11/22/33). Losowy projektor działa na tę samą delta; normę perturbacji dopasować PER TOKEN do identity normy, co zachowuje też normę per image. Jawna diagnostyka zerowych mianowników. Nie używać przyszłych wag ani refitować kierunków na tych danych. Użycie z_n w interwencji jest znanym członem pary eksperymentalnej, nie prognozą przyszłości.

Interwencja hook dokładnie na last mlp.act, forward dokończony przez projekcję MLP/residual/norm/głowę. Po tej GELU nie ma kolejnej attention; zmiana wpływa bezpośrednio na wskazane komórki detektora. Zapisać udział GT-center i peaków poza maską ROI. Macierz fc2 ma768→192: zapisać także normę perturbacji po fc2; równe normy768D nie gwarantują równej siły funkcjonalnej. DaneGT i oryginalne obrazy identyczne dla wszystkich ramion. Zachowanie poza wskazanymi tokenami i noop sprawdzić numerycznie. Metryki: dokładny source strata, confidence×IoU (source iou), AP50 (source 101punktów), max confidence, strata heat/box. Loss/q ważone po filmach; AP50 to standardowa pooled source miara po obrazach, bootstrap klastrowy po filmach. Source funkcje dekoduj/strata/ap50 zachowane w lokalnym snapshot. Bootstrap różnic po filmach; AP bootstrap po filmach ograniczony200 powtórzeń. Wynik mniejszego spadku niż dopasowane random nie potwierdza selektywnego wpływu. Bez innych instrumentów i niezależnych sham kontroli brak pełnego dowodu tożsamości.

Root przygotowuje24sham edycje w controls_sham (jego ownership). Postfit diagnostyka na zamrożonym U i klasyfikatorze: delta L2/projekcja w sham ROI oraz oryginalnym tool ROI, reakcja klasyfikatora. Te same narzędzia pozostają. Inny obszar/treść nie usuwa całkiem confoundingu artefaktów.

## Zapis i budżet

Disk F przed pomiarem około1.352GB; cały nowy katalog <=250MB i minimum1GB po zapisie. Nie zapisywać pełnych800×600×768. Zachować losslessfloat32 wszystkie pełne tokeny maski narzędzia obu członów i8deterministycznych far-background tokenów na obraz; pokrycia i indeksy obowiązkowe. Analiza full600delta energy/projection maps w RAM lub małym zapisie; inferencebatch2. Przed forward sprawdzić GPU i RAM; brak nowych dependencies/cache. Zamrożone źródłowe obrazy/wagi, brak commitów/Eden/treningu. Skrypty odtwarzają niezapisane tło.

Raport pełnych liczb: model/data hash, czasy, degeneracja/brak tokenów/noNaN, walidacyjny wybór i negatywy, ranking tokenów każdego obrazu,4–8 heatmap test, topaxis plot oraz tabela ramion detekcji. Późniejsze prognozowanie ruchu między checkpointami to osobny etap.
