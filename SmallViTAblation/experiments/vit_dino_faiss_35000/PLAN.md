# Pilotaż DINO i FAISS, 35000 obrazów TRAIN

Pytanie: jakie rzeczywiste oznaczone obrazy TRAIN są najbliższe zapytaniom SD według jednego globalnego CLS DINOv2? To indeks wyszukiwania, bez treningu i oceny generalizacji. Obalają poprawność: niewłaściwy split, niepoprawne GT, niezgodna liczba wierszy, zmiana modelu lub preprocessingu, niefinitywne cechy, niezgodność FAISS z bezpośrednim cosinusem.

Źródło: lokalne ZIP Cataract-LMM/3_Object_Tracking/annotations. Członkostwo TRAIN wyłącznie według eden_static36/data/tracking_static36/split_manifest.json. GT narzędzi z annotation.csv; ramki cornea i pupil zachowane pomocniczo, nie wystarczają samodzielnie do kwalifikacji. Kwoty równomierne per film z redystrybucją niedoborów, równomierne docelowe numery klatek, zastąpienie defektu najbliższą dostępną klatką. Pełne dekodowanie i kontrola GT przed ekstrakcją. Manifest z row, ID, ZIP, member, film, frame, GT i hashami.

Kontrola aparatu: identyczny model i preprocessing, FP32 kontra AMP na klatkach z całego przekroju filmów; norma L2, self-query z uwzględnieniem remisów, świeży reload FAISS i brute-force cosine top32; świeży reload modelu i ponowne zapytania. Nie jest potrzebna podłoga detektora, bo nie mierzymy jego skuteczności.

Limit alokatora GPU 2 GiB; batch zmniejszany przy OOM. Sprawdzanie wolnego VRAM, backoff bez ingerencji w inne procesy. CPU dekoduje ZIP w wątkach. Chunk 1000 float32, atomowy zapis i hash konfiguracji pozwalają wznowić. CPU IndexFlatIP. Wszystkie zapisy w rodzinie vit_dino_faiss_35000. Bez globalnego registry, commitów i push.

Ograniczenia: 35000 to pilotaż, a nie cały TRAIN. CLS może opisywać tło. Podobieństwo DINO nie dowodzi realizmu obrazu SD, zgodności pełnego latentu 600x768 ani korzyści z uczenia.
