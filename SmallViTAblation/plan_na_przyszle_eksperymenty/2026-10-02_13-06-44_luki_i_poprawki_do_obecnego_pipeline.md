# Luki i poprawki do obecnego pipeline tożsamości narzędzia

Utworzono: 2026-10-02T13:06:44+02:00 (Europe/Warsaw). Status: osobna notatka hipotez, kontroli i priorytetów; **nie nowy krok 8, nie implementacja i nie polecenie uruchomienia treningu**. Obowiązująca specyfikacja pozostaje w PIPELINE_TOZSAMOSCI_NARZEDZIA.md. Nie deklarujemy nowości naukowej ani skuteczności niewykonanych wariantów.

## 1. Artefakty LaMa mogą współtworzyć sygnał obecności

**Dowód dostępny:** klasyfikator natywnych J128 uzyskał 92,8274 procent filmowo ważonej balanced accuracy na oryginał vs LaMa w znanym ROI. To wynik rozpoznania tej pary warunków. Nie dowodzi wyłącznej semantyki narzędzia. Kanały wyznaczono ze średniej coverage-ważonej ROI, nie oddzielnej identyfikacji każdego tokenu.

**Hipoteza ryzyka:** część decyzji może pochodzić z artefaktów inpaintingu, treści tła lub pozycji, a nie samej obecności. Usunięcie nie odzyskuje prawdziwego zasłoniętego tła. Self-attention miesza informacje między tokenami, więc nominalne maskowane tokeny mogą zawierać kontekst.

**Minimum:** zachować kontrolę sham edycji tła przy pozostawionym narzędziu, te same maski/indexy obu stron, matched pozycję/kąt i jawne scores. Istniejący eksperyment podprzestrzeni ma 24 strict-test sham oraz norm-matched random interventions; nie utożsamiać ich ze sprawdzeniem przyczynowości każdego z natywnych 128 kanałów. Sham reagował na poziomie probability mimo braku zmian decyzji binarnej, a 13/24 edit-ROI dzieliło token z tool-ROI.

**Rozszerzenia:** naturalne obrazy bez narzędzia z kontrolą domeny; inny model inpaintingu; różne regiony/maski edycji; inne instrumenty; testy matched kąt/położenie. Interwencje kanałowe porównywać z losowymi kierunkami o dopasowanej normie i, gdy właściwe, normie po fc2.

**Co podważa interpretację:** jeśli rozdzielenie znika dla naturalnego negatywu lub zmiany metody inpaintingu, a sham daje podobny efekt do usunięcia, hipoteza wyłącznej tożsamości słabnie. Utrzymanie efektu na różnych kontrolach osłabia proste wyjaśnienie artefaktowe, lecz nie jest pełnym dowodem semantycznej wyłączności.

## 2. Przyszły latent i bieżący ViT mogą tworzyć niedopasowany cel

**Hipoteza, nie wykazany fatalny błąd:** prognozujemy współrzędne pod przyszłym stanem treningu, lecz prowadzenie SD ocenia obraz przez **bieżący zamrożony ViT**. Po uczeniu te same obrazy mogą mieć inną reprezentację; nie wynika automatycznie, że cel z przyszłego układu jest tak samo osiągalny lub semantycznie równoważny przy current model.

**Minimum:** jawnie oznaczyć checkpoint/model stanowiący układ celu i model mierzący SD. Na stałych obrazach porównać reprezentacje przed/po, centrowanie, normy i zgodność kanałów; sprawdzić dopasowanie SD do current target jako kontrolę bez prognozy. Nie kalibrować tego rzeczywistymi przyszłymi stanami podczas prospective generacji.

**Rozszerzenia:** historycznie dopasowana mapa kalibracji/decoder między układami, teacher-student lub osobny readout. Wszystko wyłącznie z zakończonej historii i z testem przeniesienia; użycie przyszłego teacher checkpointu może być tylko retrospektywnym oracle.

**Wynik negatywny:** dobry forecast-error przy poor attainability SD lub poprawne current-target SD, ale błędny forecast-target SD, wskazuje na lukę interfejsu. Brak takiej różnicy osłabia tę hipotezę. Ani uwzględnienie wszystkich 768, ani warmup nie dowodzą domknięcia: brakuje pełnych wag, BatchNorm, AdamW i wejść. Model wspólny dla tokenów/obrazów jest założeniem empirycznym, nie prawdą o jednorodności ich dynamiki; zachować image-ID i pozycje.

## 3. Podobieństwo retrieval może ograniczać różnorodność

**Minimum obowiązującego kroku 5:** pełny obraz DINO → eligible niewykorzystane train → cosine>0,85 → najwyżej 100, bez quota. Zero qualifying oznacza 0. Score 100×cosine jest umowną skalą cech, nie prawdopodobieństwem ani procentem zgodnych pikseli. FAISS, precompute, jakość, geometria, orientacja i SSIM pozostają opcjonalne.

**Hipoteza ryzyka:** podobne kadry mogą pochodzić z jednego filmu i pobliskiego czasu, dając nadmiar duplikatów i mały zysk danych. Duży similarity-score nie musi oznaczać dużego przyrostu jakości detektora.

**Minimum kontroli:** frame-ID/hash deduplikacja między rundami, used/added/sampled rozdzielone, histogram filmów, odstępów czasu, kątów i wielkości. Porównać progressive random z dokładnie dopasowaną liczbą zaakceptowanych obrazów. Nie dodawać val/test.

**Rozszerzenia:** predeklarowany limit per film, odstęp czasowy, warstwowanie kątów lub reranking różnorodności; kontrole uncertainty/hard examples z takim samym budżetem. Niska jakość nie jest automatyczną karą: trudne klatki mogą być informatywne.

**Co podważa sens similarity:** brak przewagi nad matched random/direct real retrieval lub przewaga znikająca po wyrównaniu filmów i prezentacji. Wyższy score bez poprawy stałego validation nie uzasadnia sam w sobie metody.

## 4. Rozpoznanie obecności i ruch tokenów nie są poprawą uczenia

**Dowód dostępny:** presence classifier działa w opisanym zadaniu. W odrębnym eksperymencie paired interwencje podprzestrzeni obniżały confidence×IoU bardziej niż dopasowane random, ale CI zmiany AP50 i loss obejmowały 0. Nie przenosimy tego wprost na aktualne natywne J128 ani na przyszły efekt uczenia.

**Hipoteza ryzyka:** duża zmiana/projekcja tokenu może nie mieć wpływu na detekcję. Przykład source 8: największa reakcja w tokenie 280, największy spadek q po interwencji w 248. Maska ROI i pooling nie dowodzą indywidualnej odpowiedzialności cząsteczek.

**Minimum:** oddzielić classifier BA, forecast sign/RMSE, latent delta i detector mAP/loss. Testować związek zmian z metryką detekcji i norm-matched interwencje, z kontrolą sham/noop oraz GT-center vs tool-token. Korelacja ruchu z loss nie jest przyczynowością.

**Rozszerzenia:** natywne channel ablations/restoration i controls innych klas; nie wybierać osi po efektach testowych. Gdy latent-response rośnie, ale mAP/precision/recall lub wynik interwencji nie zmienia się, zachować negatyw zamiast nazywać ruch semantycznym wpływem.

## 5. Prognoza jest aktualnym wąskim gardłem

**Dowód dostępny:** wcześniejsze 99 konfiguracji walidacji i 16 zamrożonych wariantów testu nie osiągnęły 85 procent znaków. To nie oznacza testowania wszystkich 99 na final. Cztery holdouty dynamiki były wcześniej oglądane: wyniki pozostają eksploracyjne. Wynik trwającego NNDMD nie jest jeszcze końcowo zatwierdzony.

**Minimum:** wielohoryzontowy błąd całych 768 i osobno wszystkich J128 po decode, RMSE/MSEskill względem persistence, velocity i reversion, cosine/norm-ratio oraz znaki skumulowanego ruchu bez ukrywania małych zmian. Nie mylić presence 90 z forecast 85; nie wybierać modelu na test.

Prefix/poison obejmuje surowe wejścia i pomocnicze przetwarzanie. Inicjalizacja, skale, filtry i kalibracje używają tylko dostępnego prefiksu; rollout nie otrzymuje przyszłych pomiarów, obrazów, batchy lub gradientów. Native state 768 i learned state 768 mają różne znaczenie mimo tej samej liczebności; ocena dotyczy decoded native.

**Wynik negatywny:** trafne znaki przy zaniku amplitudy lub MSE gorszym niż podłogi nie oznaczają użytecznej prognozy. Brak przewagi na transfer/świeżym biegu podważa uogólnienie. Nowe ziarna i świeży test są rozszerzeniem potwierdzającym, nie już wykonanym dowodem.

## 6. SD z czystego szumu może nie osiągnąć arbitralnego celu

**Hipoteza:** natywne współrzędne prognozy mogą być trudne do realizacji przez SD, szczególnie przy określonych cząsteczkach, geometrii i ograniczeniu prior generatora. Target nie powstaje z szumu: pochodzi z historii konkretnego obrazu. Potrzebna jest mapa tokenów, maska/warunek przestrzenny i zgodny preprocessing.

**Minimum:** rozpocząć od czystego losowego latentu VAE, bez img2img; frozen current ViT bez zmian wag/BN; jawna energia full768/J128, step/restart budget oraz log failure. Końcowe acceptance mierzyć po decode finalnegoRGB, kwantyzacji i ścisłym ViT encode, nie na pośredniej estymacie czystego obrazu.

Cel ponad 90 procent pokrycia wymaga zdefiniowania tolerancji/normalized-error/RMSE i osobno cosine, z ochroną małych norm. Nie uznać cosine 0,9 za 90 procent zgodności. Overshoot i nie-monotoniczny denoising są możliwe; reguła stop nie gwarantuje zachowania progu po kolejnych krokach.

**Co podważa ten etap:** wysoka liczba rejection, tylko nierealistyczne obrazy spełniające latent loss lub rozjazd final vs intermediate. Minimum porównawcze to current-target SD bez prognozy z wyborem kotwic niezależnym od forecastu. Różne schedulery/prowadzenia i lepsza geometria są opcjonalne; nie zmieniać ich po final test tylko dla ratowania wyniku.

## 7. DINO jest roboczym rdzeniem, dodatki nie są obowiązkiem

Aktualna intencja użytkownika jest prosta: pełny obraz DINO i cosine>0,85, adaptacyjnie 0–100 bez dopełniania. Baza FAISS i wszystkie rozszerzenia struktury/jakości są opcjonalne z uwagi na koszt. Notatka luk nie zmienia tej decyzji i nie uruchamia budowy bazy.

**Minimum:** spójny full-image preprocessing, zapis scores/threshold/version/IDs, deterministic ranking/ties, właściwa train eligibility, deduplikacja. Oceniać ranking i jego późniejszy zysk uczenia, nie udawać skalibrowanych 85 procent podobieństwa fizycznego obrazu.

**Opcjonalnie:** pose/size/orientation względem środka, blur/global layout, DINO-version comparison, SSIM reranking i FAISS. Żaden dodatek nie ma gwarantowanej poprawy. Jeśli zwiększa koszt bez przewagi w matched comparison, zachować negatyw i nie włączać go do minimalnej metody.

## 8. Końcowa poprawa ViT wymaga niezależnych i uczciwych ramion

**Minimum:** plan A/B/F z tym samym stanem startowym/warmup, architekturą, AdamW/lr/augmentacjami i optimizer-step budget; rzeczywista liczba nowych klatek per round dopasowana w B do F. A to fixed random conventional baseline; B kontroluje progresywną dostępność. Separate runs/checkpoints, nie kolejne zabiegi na tej samej sieci.

Dokładny provenance: kiedy frame dodano i kiedy naprawdę sampled, sourcefilm/frame/hash/split, augmentacje, RNG, optimizer-step i checkpoint, old/new prezentacje. Nie wykorzystać val/test do learning lub retrieval. Fixed validation przed/po rundzie; final test raz po zamrożeniu protokołu.

**Ocena:** paired learning-mAP AUC i CI po całych seed-run, final mAP/loss/precision/recall/IoU, strata kąty/blur/tło. 10 seedów zaplanowane, nie wykonane. Koszt end-to-end obejmuje NN/SD/DINO/index/ViT; krzywe po updates, unikalnych obrazach, całkowitych prezentacjach i GPU-hours. Pretrained/no-add/all-available-train są kontekstem, a odmienny budżet etykiet nie jest primary fair comparison.

**Rozszerzenia etapowe:** D direct real retrieval bez SD/forecast, E SD current target bez prognozy i forecast-anchor selection, C sequential wcześniej ustalonych filmów. Uncertainty/hard examples mogą stanowić dalsze kontrole. Brak przewagi F nad B/D/E przy wyrównanych budżetach podważa dodatkową korzyść prognozy lub SD, nie automatycznie całą istniejącą metodę retrieval. Nie ogłaszamy ustalonej nowości przed takim testem.

## Priorytety i reguła decyzji

Najpierw sprawdzalne cele/metody/telemetria oraz prognoza przeciw podłogom, następnie minimalny A/B/F i current-target/sham kontrole. D/E są kluczowymi rozszerzeniami rozdzielającymi wkład SD i forecastu; C, naturalne negatywy/inny inpainting oraz interwencje natywnych osi uzupełniają dowód. Geometria/FAISS/reranking nie są warunkiem minimalnego wykonania.

W każdej kontroli z góry określić metrykę, budżet i czego wynik negatywny nie pozwoli twierdzić. Nie poprawiać modelu według final test ani pomijać failure. Ten dokument rejestruje potencjalne niedomknięcia i plany ich sprawdzenia; nie zastępuje obowiązującej specyfikacji i nie deklaruje, że wszystkie hipotezy są prawdziwe.

## Sprawdzalne źródła lokalne

- [Obowiązujący pipeline](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/PIPELINE_TOZSAMOSCI_NARZEDZIA.md).
- [Rozpoznanie J128 i jego ograniczenia](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_tozsamosc90_bez_pca_langevin/WYNIKI.md).
- [Interwencje, sham, confidence×IoU i nierozstrzygnięte AP50](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_tozsamosc_czasteczki_pary_400/WYNIKI.md).
- [Wcześniejsze wyniki prognozy 500 i negatywy](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_prognoza_tozsamosci_500_bez_pca/WYNIKI.md).
- [Zakończone pełne dane 768](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DANE_768_WERYFIKACJA.json).
- [Krok 4/5: SD, ranking i opcjonalne dodatki](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_12-49-00_sd_z_czystego_szumu_prowadzenie_prognozowanym_latentem_768.md).
- [Telemetria aktualnego użycia danych](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_12-59-31_telemetria_uzytych_klatek_uczenie_i_walidacja_rund.md).
- [Plan porównania zwykłej sieci i pipeline](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_13-02-43_test_standardowego_vit_i_pipeline_doboru_klatek.md).
