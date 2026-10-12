# Audyt lokalnej prognozy przyrostów NNDMD na Hopper

2026-10-02, Europe/Warsaw. Audytor: Astra. **Status na 21:02: CODE AUDIT PASS oraz EXECUTION PASS dla pełnego syntetycznego modelu 1vs11 na GPU. Launcher i poprawiony reader: PASS. Kontrole rzeczywistych danych pozostają obowiązkowymi guardami pełnego zadania przed fit.**

Końcowy zakres kodowy i dokładne hashe są zapisane na końcu raportu. Starsze akapity statusowe poniżej dokumentują przebieg kontroli, a nie aktualny werdykt. Nie ma jeszcze wyniku treningu nowego modelu.

Własność audytora obejmuje tylko ten raport. Nie modyfikuję kodu, danych ani launcherów wykonawców. Nie używam GPU i nie zgłaszam zadań. Wcześniejsze eksperymenty oraz audyty pozostają zachowane. Użytkownik autoryzował nową metodę i anulowanie poprzedniego treningu; root potwierdził CANCELLED dla1819574.

## Zatwierdzona konstrukcja

Pełny natywny z ma768 współrzędnych; jedna cząsteczka oznacza jeden z600 tokenów obrazu. Rozszerzone wejście nadal obejmuje z, dwie historyczne różnice oraz59 charakterystyk historii wag/BN/AdamW z maskami i wiekiem. Encoder tworzy phi0 w R768, a learned propagator A ma rozmiar768×768. Decoder jest nieliniowy.

Przy natywnej skali s prognoza ma postać:

`zhat_h = z0 + s · [D_z(A^h phi0) − D_z(phi0)]`.

Native block decoderu jest w jednostkach z/s. Auxiliary są dekodowane absolutnie jak wcześniej, bez odejmowania wartości originu. Prawdziwe przyszłe auxiliary nie są wejściami rollout. Encoder pracuje jeden raz na początku prognozy.

Ta konstrukcja jest prawidłową prognozą skumulowanych przyrostów i daje H0=z0. W implementacji należy najpierw obliczyć różnicę decoderów, a potem dodać z0; zapis `(z0 + decoded_h) − decoded_0` może naruszyć bitową tożsamość przez zaokrąglenie. W treningu **obie gałęzie decoderu muszą pozostać w autograd**. Detach bazowego D_z(phi0) zmieniałby gradient zdefiniowanej straty.

## Co oznacza stan autonomiczny

Do odzyskania natywnego wyjścia potrzeba nie tylko phi_h, ale również stałego dla danego originu offsetu `c = z0 − s·D_z(phi0)`. Można zapisać:

`phi_(h+1) = A phi_h`, `c_(h+1) = c_h`, `zhat_h = c + s·D_z(phi_h)`.

Zatem768D dotyczy wyuczonego stanu dynamicznego i macierzy A. Pełny stan potrzebny do tego zapisu wyjścia ma dodatkowy stały768D offset; równoważny rozszerzony propagator ma bloki diag(A,I). Inny równoważny zapis zachowuje parę (phi_h,zhat_h) i dodaje różnicę kolejnych decoderów do poprzedniego zhat. Nie należy nazywać samego phi768 kompletną autonomiczną reprezentacją wyjścia ani dowodzić tym domknięcia całej dynamiki natywnego z768.

H0 identity nie oznacza odwracalności E/D ani poprawnej rekonstrukcji. Jeżeli e_h jest błędem absolutnego decoderu w chwili h, błąd prognozy zakotwiczonej wynosi e_h−e_0. Błędy mogą się znosić lub wzmacniać. Trzeba osobno raportować rzeczywistą rekonstrukcję `s·D_z(phi0)` i błąd zakotwiczonego H0; ten drugi jest zerem z konstrukcji, nie sukcesem uczenia.

## Inicjalizacja i sygnał gradientowy

Dla A bliskiego I różnica decoderów jest mała. Gradient prediction loss względem encodera/decodera częściowo znosi się między dwiema gałęziami i skaluje z A^h−I. Gradient względem G przy A=exp(G) nie musi znikać. Native reconstruction z zachowaną wagą1 oraz aux mogą dostarczać dodatkowego sygnału uczenia E/D. Nie należy z tego wyprowadzać gwarancji poprawy.

Dwa kandydaty lrG=1e−4 i1e−3 wystarczają do zleconego porównania. Architektura, seed, obrazy, kolejność paczek, native reconstruction i pozostałe wagi mają być identyczne. Przy pierwszych krokach i w ustalonych dalszych punktach potrzebne są: norma gradientu G **przed clipping**, norma G, norma A−I, względna zmiana A po optimizer.step i amplituda przewidzianego ruchu. Sam większy learning rate nie dowodzi, że A efektywnie się zmienia. Kara A−I pozostaje regularizacją bliskości identyczności, nie gwarancją stabilności.

## Granice czasu i informacja statystyczna

Inner fit obejmuje cp50..80:31 stanów,30 unikalnych przejść i21 okien H10, origins50..70. Walidacja jest autonomiczna80→90. Final refit obejmuje cp50..90:41 stanów,40 przejść i31 okien H10, origins50..80. Test90→100 jest otwierany dopiero po wyborze i freeze. Historia potrzebna do origin50 może legalnie obejmować cp48 i49, bo jest wcześniejsza. Skalery nie mogą korzystać z przyszłych celów walidacji/testu; zakres ich estymacji trzeba zapisać jawnie.

Nakładające się okna powodują, że ten sam fizyczny krok może występować w kilku parach origin–horyzont. Równa waga okien i horyzontów nie oznacza równej częstości każdego checkpointu. Wszystkie obrazy obserwują jeden wspólny bieg aktualizacji wag;900 tysięcy tokenów nie oznacza900 tysięcy niezależnych trajektorii optymalizatora.

Krótka historia ogranicza rozróżnialność mechanizmów i informację czasową, zwłaszcza dla globalnych aux. Nie wynika jednak z niej automatycznie ranga≤30 całej macierzy operatora: macierz uczonych cech zestawiona z wielu różnych tokenów może mieć pełną rangę768. To rozróżnienie między rangą algebraiczną projektu regresji a niezależną informacją o dynamice.

Wybór learning rate i epoki ma korzystać wyłącznie z80→90. Cele91..100 nie służą zmianie architektury ani strojenia. Pozostaje to badanie eksploracyjne znanej wcześniej trajektorii, nie nietknięty test potwierdzający na nowym biegu.

## Urządzenia i niezmienna waga paczki

W trakcie przygotowania zmieniały się możliwości przydziału. Aktualna dyspozycja root przekazuje najnowsze żądanie użytkownika: najpierw sprawdzić dostępność11 urządzeń na Hopper-2, obejmujących4 pełne H200 i7 wolnych instancji MIG. Test infrastruktury Slurm/NCCL jest odrębny od dowodu poprawności modelu. Nie deklaruję, że taka mieszana konfiguracja już działa; nie wolno samoczynnie zamienić jej na trening4GPU, jeśli próba11 nie przejdzie.

Globalna paczka pozostaje32 obrazami, a ostatnia28. Dla11 procesów podział `array_split` daje odpowiednio dziesięć części po3 i jedną po2 oraz sześć części po3 i pięć po2. Każda lokalna suma błędów musi być dzielona przez globalne B×600; gradienty są następnie sumowane, kara doliczana raz i clipping wykonywany po SUM. Średnia lokalnych średnich byłaby błędna przy nierównych częściach. Potrzebny jest test CPU1vs11 oraz, po sukcesie infrastruktury, test rzeczywistego końcowego modelu1vs11 na GPU przed fit. Dotychczasowe dowody1vs2/4 nie dowodzą wykonania nowej konfiguracji11.

## Pozostałe punkty końcowego audytu

- Zweryfikować końcowy kod różnicy decoderów, brak detach i brak ponownego anchorowania prawdziwą przyszłością.
- Sprawdzić fit/scalery48/50..80, wybór dwóch lrG, epoki, fresh refit50..90 i freeze przed91..100.
- Sprawdzić faktyczny reset modeli po kontrolach numerycznych oraz osobne liczniki preflight i fit.
- Sprawdzić wspólny mianownik przy nierównych fragmentach, kolejność kolektywów, reader i zapis rank0.
- Odczytać testy końcowych hashy, a potem osobno rzeczywiste metryki i zapisane prognozy. Nie deklarować85 procent przed wynikiem.

Na tym etapie **PASS dotyczy designu**, a nie jeszcze konkretnego immutable execution ani uruchomionego nowego treningu.

## Aktualizacja: pierwszy odczyt kodu i niezależne gradienty

Odczytano pierwszą wersję nowych model.py, train.py, state.py, CONFIG.json i PLAN_MODELU.md. W loss i rollout różnica decoderów ma właściwy nawias; decoded_origin pozostaje w grafie. Auxiliary są nadal absolutne, a przyszłe inputy reconstruction są ograniczone do fit boundary. Operator i encoder nie otrzymują przyszłej prawdy podczas rollout.

Wyznaczanie native scalar pozostaje według zatwierdzonego planu przy cp0 wszystkich1500 obrazów. Aux średnie i odchylenia używają dostępnego prefiksu0..80 lub0..90, mimo że same okna gradientowe zaczynają się w50. To jest przyczynowe; nie należy opisywać normalizacji jako oszacowanej wyłącznie na późnym oknie50..80. Pierwsza wersja tekstu raportu błędnie zachowała90 przejść/900 aktualizacji; wykonawca poprawił ją na40 unikalnych przejść/400 aktualizacji i31 finalnych okien.

Zbudowałem osobno pełną stratę Torch bez wywoływania `_loss_piece` ani `direct_loss`. Ręczny wzorzec obejmował obie gałęzie różnicy decoderów, native reconstruction, maskowane etykiety aux i karę A−I. Parametry kontroli: D4, learned5, hidden7, H10, origin50, scale2,3, wagi rec0,7/aux0,2/identity0,03. Nie wykonano optimizer.step ani obliczeń GPU.

| Paczka | Mikrofragment | Różnica loss | Maksymalna różnica wszystkich gradientów |
|---|---:|---:|---:|
| B3, P5 | 6 z niepełnym ogonem | 3,6999225e−8 | 5,5879354e−9 |
| B28, P2 | 17 z niepełnym ogonem | 5,2407742e−8 | 3,7252903e−9 |

Wszystkie gradienty zgadzały się przy rtol3e−5, atol1e−6; algebraiczny H0 z nawiasem był bitowo równy z0. To potwierdza nową stratę i jej akumulację w małej niezależnej kontroli, nie skuteczność fitu produkcyjnego.

## Status infrastruktury11 i ograniczenie testu

Root zgłosił, że rzeczywista próba infrastruktury11, job1819903, zakończyła się FAILED przed NCCL: środowisko deklarowało urządzenia0..10, ale Torch raportował8 widocznych urządzeń. To nie jest wynik treningu NN. Wykonawca infrastruktury sprawdza jawne przydzielenie UUID osobno procesom; nie zgłoszono jeszcze pozytywnego dowodu wykonywania pełnego modelu na11 urządzeniach.

Ewentualny Gloo z transferem gradientów przez CPU jest zmianą transportu. Matematyczna równoważność wymaga nadal SUM globalnie znormalizowanych gradientów przed clipping i wspólnego Adam. Implementacja musi uwzględniać także pozostałe kolektywy CUDA tensorów, w tym licznik obrazów i statystyki, oraz mapowanie pojedynczego widocznego UUID na cuda:0. Sama próba sumy małej macierzy ani testCPU nie zastępują końcowego guardu pełnego768/all600/global32 i28 dla rzeczywiście użytego transportu.

Końcowy przedział cp90..100 był już oceniany w poprzednich wariantach, w tym na dawnym zbiorze30 obrazów i w późniejszej kontroli jednego obrazu. Nowy FREEZE chroni przepływ informacji wewnątrz tego przebiegu, lecz nie przywraca globalnie nietkniętego holdoutu. Raport końcowy musi nazywać ocenę eksploracyjną; nie wolno retuningować nowego modelu na jego końcowych wynikach.

## Końcowy audyt kodu, 20:56

**PASS dla rdzenia matematycznego, drivera i testów wskazanych w [CODE_READINESS.json](CODE_READINESS.json).** Niezależnie porównałem wszystkie dziewięć hashy z rzeczywistymi plikami. Odczytałem końcowy [log 19 testów CPU](TEST_CPU_FINAL.log): wszystkie przeszły w 28,050 sekundy. Nie uruchamiałem GPU ani nowego fitu.

| Plik | SHA256 odczytanej wersji |
|---|---|
| model.py | d47cc45f8872d99a273fcb52d8cad7e0e68cddb6c1bcb363be3f040dbb9ceca6 |
| state.py | f9b7289da4f225f99dce21b6021922f5405aec0d3c16f0f122c290a83e7726b2 |
| train.py | 0272af7aed10a1dccf6b29e9bc2297a635061aefc944f9b6f8c634d1990c1b4d |
| gpu_smoke.py | 97633c83b6c3d128252d7918680665f57dffc4e088109387e2d776ff07424026 |
| CONFIG.json | d0df5a71d6caa6a41eb10c0156b518b0abf709ceb48fcb5090e3353ae259311c |
| PLAN_MODELU.md | 4e77503e44531e20fe13e27c2cf3d75bf7b0329753dba344e62800539388f326 |
| test_expanded.py | db14a018f27d774bef54f0a64084e2d0e03941ddc7757e12b5b9554981cd6dc4 |
| test_driver.py | 6fb14d7de253943ce4278182f64f0ffa276f5a726f7b52a493aae223d1b0ead6 |
| test_distributed.py | 28b305473ca8d6afb4d09f9fd7bfb4ea69a171538f1c7299499500a537286974 |

### Zweryfikowane własności

1. `model.py:54` definiuje zakotwiczoną stratę z obiema gałęziami decoderu w grafie. `model.py:178` koduje origin raz i wykonuje autonomiczne potęgi operatora. Przyszłe prawdziwe z i auxiliary są etykietami straty lub oceny, a nie wejściami prognozy. Dodatkowa rekonstrukcja przyszłych stanów podczas treningu pozostaje legalnie ograniczona do danego prefiksu fitu.
2. `train.py:339` porównuje dokładnie dwa ustalone lrG. Modele startują z tego samego seedu i porządku obrazów. Wybór epoki oraz kandydata korzysta z pełnego natywnego RMSE dla originu 80 i horyzontów 1–10. Tie-break to mniejszy lrG, następnie wcześniejsza epoka. J128 nie steruje wyborem.
3. `train.py:369` tworzy świeży finalny model i normalizer prefiksu do 90. Wznowienie odtwarza model, Adam, licznik oraz ostatni kompletny origin i sprawdza config, fazę, granicę normalizacji i lrG. Wznowienie nie zastępuje finalnego modelu najlepszym checkpointem inner fitu. Przerwane kroki po ostatnim zapisanym originie mogą być wykonane ponownie z tego checkpointu; licznik oznacza kontynuowany bieg modelu, a nie wszystkie fizycznie wykonane operacje porzuconej próby.
4. `train.py:378` zapisuje końcowy model, macierz A i freeze przed `ensure_window(90,10)` w wierszu 389. Dodatkowy guard w `evaluate` sprawdza granice i SHA modelu. Cele 91–100 nie uczestniczą w wyborze. Globalnie ten przedział był już oglądany w innych wariantach, co ogranicza wniosek do oceny eksploracyjnej.
5. `model.py:127` używa globalnego mianownika B×600, sumowania gradientów, jednej kary na ranku 0 i clippingu po SUM. Dla 11 ranków obie nierówne paczki 32 i 28 są poprawnie ważone. Nie ma dodatkowego dzielenia przez liczbę ranków. Każdy rank wykonuje ten sam krok Adam.
6. `model.py:118` obsługuje opcjonalny Gloo: CUDA tensor jest kopiowany do CPU, sumowany i kopiowany z powrotem. Dotyczy to licznika obrazów, wszystkich gradientów oraz skalarnych, predykcyjnych i auxiliary statystyk. Dtype jest zachowany. Kolejność kolektywów pozostaje wspólna; błędy lokalnej akumulacji są rozgłaszane przed redukcją gradientów. To zmiana transportu, bez zmiany zdefiniowanej funkcji celu. Test CPU z atrapą transportu nie dowodzi wydajności ani działania rzeczywistego CUDA/Gloo na Hopperze.
7. `train.py:78` porównuje dwie rzeczywiste paczki danych, pełne 32 obrazy i ogon 28, z 600 tokenami każda. Modele kontrolne są usuwane, a fit tworzy model ponownie z ustalonego seedu. Liczniki `actual_fit_steps` i kroków kontroli są rozdzielone.
8. Metryka główna jest pierwiastkiem ze wspólnego średniego kwadratu błędu po obrazach, tokenach, natywnych współrzędnych i horyzontach. Nie jest średnią lokalnych RMSE. Znaki są oceniane dla wszystkich współrzędnych skumulowanego przemieszczenia, bez odsiewania małych zmian. H0 zakotwiczone i rzeczywista rekonstrukcja pozostają różnymi pomiarami.
9. Generator G jest bezpośrednią macierzą propagatora PyKoopman, A=exp(G), dt=1 oznacza jeden krok checkpointowy, czyli 10 aktualizacji źródłowego ViT. Normy gradientu G są zbierane po SUM i przed clippingiem. Diagnostyka A−I używa rzeczywistego wymiaru macierzy, także w małej fixture. Kara bliskości I nadal nie gwarantuje stabilności spektralnej.

Odczytane dowody [CPU 1vs11](PROOF_CPU_GLOO_11.json) i [CPU 1vs4](PROOF_CPU_GLOO_4.json) obejmują globalne paczki 32 i 28. Maksymalny błąd gradientu wyniósł odpowiednio 5,5879354e−9 i 3,7252903e−9; maksymalny błąd wag po Adam wyniósł 1,4901161e−8 w obu kontrolach. Osobny, opisany wcześniej ręczny wzorzec straty potwierdził gradient nowej różnicy decoderów niezależnie od wspólnego helpera implementacji.

### Oddzielny warunek wykonania na 11 urządzeniach

Na moment tego zapisu nie ma jeszcze pozytywnego dowodu pełnego modelu na docelowych 11 urządzeniach. Według raportu właścicieli per-UUID job 1819908 rozwiązał widoczność pojedynczego urządzenia, ale NCCL 2.27.5 zgłosił powielony fizyczny GPU dla dwóch różnych instancji MIG. Izolowany NCCL 2.32.3 jest przygotowywany do sprawdzenia. Sama instalacja nowej biblioteki nie potwierdza, że proces ją załadował ani że kolektywy modelu działają.

Znalazłem konkretny problem integracyjny w `reader_preflight.py:10`: odczytana wersja nadal używa `LOCAL_RANK` jako indeksu CUDA i na sztywno wybiera NCCL. Przy pojedynczym UUID widocznym w każdym procesie musi używać lokalnego indeksu 0. Musi też respektować ten sam backend co model i launcher. Uwagę przekazałem właścicielowi danych przed startem; nie zmieniałem jego pliku.

**CODE PASS nie jest EXECUTION PASS.** Ten drugi wymaga końcowego, identycznego runtime i snapshotu źródeł, poprawnego per-rank przypisania UUID, rzeczywiście załadowanej wersji transportu oraz pełnego GPU guardu 1vs11: native 768, learned 768, auxiliary 59, wszystkie 600 tokenów, globalne paczki 32 i 28, zgodne gradienty, Adam i rollout. Launcher ma następnie przejść kontrolę readera i rzeczywistych danych przed pierwszym krokiem fitu. Jeśli próba 11 nie przejdzie, nie jest to upoważnienie do samoczynnej zamiany na trening na czterech urządzeniach.

Nie wymagam dalszych zmian rdzenia modelu ani dodatkowych wariantów badawczych. Wynik skuteczności nowego treningu pozostaje nieznany.

## Zamknięcie dowodu wykonania, 21:02

**EXECUTION PASS dla syntetycznej kontroli pełnego modelu na 11 urządzeniach.** Odczytałem rzeczywisty [RESULT joba 1819912](gpumodel11_1819912/model/RESULT.json), zapis kroków oraz [manifest końcowych źródeł](SOURCE_HASHES.json). Zapis dotyczy pełnych wymiarów native 768, learned 768, auxiliary 59, wejścia 2544 i decoderu 827. Obie globalne paczki, 32 i 28 obrazów, miały po 600 tokenów na obraz i H10. Były to rzeczywiste operacje GPU na syntetycznych danych, a nie fit 1500 obrazów.

| Globalna paczka | Maks. różnica gradientu | Maks. różnica wag po Adam | Maks. różnica decoded rollout |
|---|---:|---:|---:|
| 32 obrazy | 6,5192580e−9 | 3,4868717e−6 | 7,1525574e−7 |
| 28 obrazów | 1,0244548e−8 | 3,4868717e−6 | 1,3113022e−6 |

Sprawdzono loss, wszystkie gradienty i wagi, prognozę, telemetrię oraz zapis/odczyt checkpointu. Tolerancje zadeklarowano w teście przed jego wykonaniem: rtol 3e−4, atol gradientu i prognozy 3e−5, atol wag 2e−5. Kontrola przyszłych auxiliary zatruwa ich sufiks po originie. Licznik `actual_fit_steps` wynosi 0; dwa kroki optimizer dotyczą wyłącznie modelu kontrolnego.

Odczytałem wszystkie 11 plików `rank_00.json` do `rank_10.json` z [infrastruktury 1819911](infra11nccl232_1819911/rank_00.json). Każdy ma PASS, jeden widoczny CUDA device, efektywny numer biblioteki NCCL 23203 i mapę załadowanego pliku z izolowanego `runtime_nccl232`. Profile obejmują dokładnie 4 pełne H200, 4 instancje MIG 4g.71gb oraz 3 instancje MIG 3g.71gb. Pole kompilacyjne Torch `nccl=[2,27,5]` nie jest tutaj numerem efektywnie podmienionej biblioteki; dowodem jej wersji jest osobny odczyt `actual_nccl_version` oraz loaded maps. Zapisany SHA256 biblioteki: `a23d2e3e47ad7f8987ac73eb04a373d6d40cfbb3a7a5bf41a9e0e8434817559b`.

Porównałem lokalnie wszystkie **22/22** pliki z [FINAL_SOURCE.sha256](FINAL_SOURCE.sha256) i SOURCE_HASHES.json. Nie ma rozbieżności. Dziewięć wspólnych plików kodu/danych snapshotu GPU jest bitowo identycznych z aktualnymi odpowiednikami; dotyczy to modelu, state, drivera, gpu_smoke, configu, readera, modelu źródłowego, auxiliary i przygotowania źródła. Dziewięć wcześniejszych hashy CODE_READINESS pozostaje zgodnych. Właściciel infrastruktury dodatkowo zgłosił zgodność 22/22 plików na Edenie.

Wcześniejszy problem `reader_preflight.py` został usunięty: pojedynczy widoczny UUID jest mapowany na `cuda:0`, a backend pochodzi z tego samego środowiska co driver. `uuid_entry.py` ustawia UUID przed importem Torch. Launcher żąda dokładnie 11 urządzeń i ładuje tę samą izolowaną bibliotekę NCCL. `NCCL_MNNVL_ENABLE=0` jest jawne. Cache pełny jest wybierany tylko przy wystarczającej pojemności; dla tmpfs wymaga także przydziału co najmniej 384 GiB RAM. W przeciwnym razie wybierany jest rolling13. Pełny tryb cache oznacza zachowywanie już wyliczonych stanów, nie przedwczesne wyliczenie przyszłego testu. Cleanup jest ograniczony do własnego katalogu joba z kontrolą ścieżki i znacznika właściciela.

Przed pierwszym właściwym krokiem treningu launcher nadal musi przejść source preflight, a driver kontrolę rzeczywistych paczek 32 i 28 ze wszystkich 600 tokenów, po czym odrzucić modele kontrolne i odtworzyć seed. Ten warunek jest zaimplementowany; niniejszy zapis nie twierdzi, że rzeczywiste dane nowego pełnego zadania zostały już przetestowane. Po przejściu zapowiedzianej suite w docelowym runtime nie widzę blokera do pojedynczego autoryzowanego pełnego joba z tymi guardami. Nie potrzeba kolejnego syntetycznego testu niezmienionego rdzenia ani zmiany modelu.

Wynik prognozowania nowego modelu nadal nie jest znany. PASS dotyczy poprawności konstrukcji i sprawdzonego wykonania, a nie uzyskania celu 85 procent lub poprawy względem persistence.
