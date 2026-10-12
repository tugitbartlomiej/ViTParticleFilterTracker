# Ponowna optymalizacja sztucznego latentu

Użyto MODEL_best.pt z eksperymentu vit_wyuczenie_latentu_jedno_zdjecie_do85, po 6402 aktualizacjach. To najlepszy zapis dopasowania latentu jednego obrazu, nie model o potwierdzonej najlepszej jakości na niezależnym zbiorze. Obraz: TR_0154_S1_P03, klatka 1926, ten sam co w poprzednim eksperymencie. Etykieta: ramka [212,130,356,170].

Zamrożono wagi ViT. Optymalizowano aktywacje po GELU w ostatnim bloku, dla tych samych 128 kanałów i wszystkich 600 tokenów. Pozostałe 640 kanałów pozostały dokładnie niezmienione. Wykonano 4500 aktualizacji Adam z learning rate 0.01, 0.03 i 0.1. Funkcja celu uwzględnia odpowiedź detektora, regresję ramki, GIoU, tłumienie pozostałych odpowiedzi i niewielką karę za zmianę aktywacji. Wybór preferował IoU co najmniej 0.99, a następnie najwyższą pewność.

| Pomiar | Przed | Po interwencji |
|---|---:|---:|
| Pewność detektora | 74.4268% | 93.3576% |
| IoU | 99.9947% | 99.8930% |
| Największa wartość w J128 | 44.7612 | 57.0202 |

IoU nie poprawiło się: było już niemal idealne i minimalnie spadło. Wzrosła pewność, o 18.9308 punktu procentowego. To odpowiedź modelu, nie procent poprawnych rozpoznań ani mAP zbioru.

Każdy modyfikowany kanał ograniczono według jego indywidualnej granicy wynikającej z aktualnych wag LayerNorm i fc1 oraz funkcji GELU. Największa górna granica J128 wynosi 57.0202469. Zgodność z ograniczeniami sprawdzono z tolerancją numeryczną float32. Te granice dotyczą bieżącego checkpointa; nie należy przenosić granic starego checkpointa na ten model. Granice pojedynczych współrzędnych są warunkiem koniecznym, ale nie dowodzą, że wszystkie współrzędne tego sztucznego latentu można jednocześnie wywołać rzeczywistym obrazem.

Końcowy wynik zweryfikowano pełnym przejściem przez ViT z interwencją aktywacji. Maksymalna różnica względem szybszej pętli wyniosła 9.54e-7. Wagi pozostały niezmienione. To nie jest naturalny forward wytwarzający nowy latent bez interwencji, ani trening nowego checkpointa. To dopasowanie z etykietą do jednego obrazu użytego także wcześniej do uczenia tego modelu, bez wniosku o generalizacji lub przyspieszeniu treningu.

Artefakty: LATENT_PELNY.npz zawiera before, after, delta (600 × 768), channels0 oraz indywidualne granice. LATENT128.npz i KANALY128.csv wydzielają zmieniane kanały. RESULT.json zawiera metryki, hashe i historię. DETEKCJA_PRZED_PO.png pokazuje ramki na niezmienionym zdjęciu. Program: experiments/vit_latent128_nowy_checkpoint_granice_20261006/run.py.

Adapter SD wytrenowany wcześniej na pierwotnym MODEL_VIT_K1000.pt nie ma automatycznie potwierdzonej zgodności z przestrzenią reprezentacji tego nowszego ViT. Ewentualna generacja z tego nowego celu wymaga uwzględnienia tej zmiany checkpointa.
