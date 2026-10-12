# Prognoza Langevina czterech paczek SD

Pytanie: czy trening pełnej sieci na czterech dokładnych paczkach po32 realne zdjęcia train kieruje naturalny latent niezmienionej sondy29 ku sztucznemu after? Obala korzystny kierunek brak redukcji błędu ponad nochange i losowe32. Obala poprawność aparatu niezgodny checkpoint/before, sztuczne GT, zmiana BN w prognozie lub przeciek przyszłych sił.

Punkt wyjścia: wyłącznie oryginalny MODEL_VIT_K1000.pt z outputs/vit_latent128_jedno_zdjecie_90. Pełny autentyczny stan AdamW pobrany wcześniej z Edenu jest czytany z ukończonego vit_langevin_paczka32_cel_sztuczny, kopia i dawny zdalny SHA zachowane. Wszystkie tensory muszą być identyczne z oryginalnym modelem SD. Nowy model dopasowany Lorentzem do pojedynczego zdjęcia nie uczestniczy.

Dokładnie cztery TOP32.json z vit_sd_cztery_obrazy_paczki32,128 różnych pikselowo zdjęć. Wybrano je z deterministycznego prefiksu88469, nie pełnego train. Realne adnotacje narzędzi z annotation.csv, wykluczone cornea/pupil, skala480x320, zachowane SHA/CRC i surowe GT. Kontrola losowa32 bez powtórzeń z pełnego znanego train310257, ustalona seed20261005 przed prognozą; nochange zachowuje wszystko i ma0 aktualizacji.

Oryginalna sonda29 TR_0154/1926: identyczne piksele źródła RGB480x320 INTER_AREA, nie zdjęcia paczki ani obrazy SD. Before i after600x768 oraz J128 z oryginalnego LATENT_PELNY. Przed każdym ramieniem niezależny model i deepcopy stanu optymalizatora z k1000.

Siły z rownanie_original: gradient aktualnej znanej paczki32, normalny train BN na kopiach buforów, bf16, clipping normy1. Części AdamW według oryginalnych beta/epsilon i lr5e-4. S_nowa=z(theta+d_nowa)-z(theta), S_ped analogicznie, odczyty functional_call eval float32 ze starymi buforami. Parametry i bufory nie mogą się zmienić. Momentum wykorzystuje mianownik z gradientem tej samej bieżącej paczki, jawna prognoza warunkowa. Zanik wag osobno, w głównej sumie pominięty zgodnie z dawnym równaniem; dokładny połączony wirtualny krok zawiera wd i daje naturalny output detektora.

Główne równanie S_nowa+S_ped, współczynniki1/1. Osobno0.9838/0.9844 sprawdzone ze źródłowym TOZSAMOSC_JEDEN_KROK, bez dopasowania do nowego testu. Transfer średniej tokenów narzędzia na pełne600x768 jest przybliżeniem ocenianym krokiem1: skill, cosinus przyrostu i zgodność znaków, ze starym oraz rzeczywistym BN.

Prognozy origin0 horyzont1/5/10/25/50/100, zapisane i zahashowane PRZED realnymi krokami. Znana paczka jest powtarzana; stała początkowa siła to założenie ekstrapolacyjne, nie przewidywanie przyszłego gradientu. Nie wykonujemy rolling reforecast. MSE J128/full768, zysk odległości, podpisany iloczyn z kierunkiem target-before, cosinus, zgodność współrzędnych. Przekroczenia5/10/25/85/90 procent redukcji MSE wyłącznie w granicy100; null oznacza brak trafienia.100 aktualizacji to100 powtórzeń32, nie100 epok pełnego datasetu.

Walidacja:5 niezależnych ramion po100 aktualizacji pełnej sieci, prawdziwa strata detektora, pełna paczka32, normalny BN, bez augmentacji jak w poprzednim protokole. Nochange nie uczy. Wszystkie wagi, również głowa, aktualizowane. Naturalna odpowiedź sondy to confidence i top1 IoU oraz maxIoUtop10. Dalszej odpowiedzi głowy nie wyliczamy przez podmianę prognozowanego latentu. Latent bliżej celu nie musi dawać lepszej detekcji.

Koszt docelowy500 realnych aktualizacji,16000 ekspozycji treningowych,5 gradientów początkowych dodatkowo. Próg GPU2816MiB z wcześniejszego smoke tego samego modelu/full32; bieżący smoke mierzy nową paczkę. Bez mikrobatchy ani cichej zmiany BN, bez kończenia cudzych procesów. Oczekiwanie na pamięć ograniczone90minut. Zapis atomic retry Windows.

Poprzednia inna paczka32 po50 pogorszyła MSE76814.3202 ->14.3605. To kontekst z RESULT_SUMMARY.md, nie założenie kierunku nowych paczek. Jeden probe nie mierzy generalizacji. Wyniki ujemne, kontroli, opóźnienia i progi nieosiągnięte zachować w raporcie. Wszystkie nowe pliki w tej rodzinie, bez commitów.

Kontrola wykonania: nowy smoke pełnej32, peak allocated1329.08MiB i reserved1640MiB. Brak zmian BN/mikrobatchy. Pierwszy odczyt JSON zatrzymał się przed gradientami z powodu domyślnego kodowania Windows; jawne UTF-8 usunęło błąd. Przygotowane GT ponownie zweryfikowano przez SHA i użyto bez zmian. Główny bieg ukończył500 aktualizacji. Końcowa walidacja checkpointów i każdej trajektorii przeszła.
