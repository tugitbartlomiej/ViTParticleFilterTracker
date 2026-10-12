# Cztery zapytania, cztery paczki po 32 zdjęcia train

Pytanie: jakie rzeczywiste zdjęcia train mają najbliższe naturalne latenty do czterech ukończonych obrazów SD?

Zapytania: OBRAZ_2026100511_prowadzony.png, OBRAZ_2026100523_prowadzony.png, OBRAZ_2026100537_prowadzony.png i OBRAZ_2026100549_prowadzony.png z outputs/vit_sd_cztery_szumy. Bez sztucznego celu. Zamrożony MODEL_VIT_K1000.pt i kanały channels0 z outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz.

Pełny zbiór: wszystkie 310257 JPG/JPEG/PNG z archiwów, dla których split_manifest groups[archive.stem] to train. Val/test/unknown są wykluczone. Jeden forward każdego kandydata i cztery porównania naturalnych latentów 600x768. Ranking raw MSE; pomocniczo MSE128/640, cosinus768/128/640 i pewność detektora. Batch 4, eval, float32, TF32 wyłączone. Wagi nie są uczone.

Kontrola aparatu: każde zapytanie porównane z własnym naturalnym latentem daje MSE zero, cosinus jeden. To podłoga odległości zachowująca obraz, preprocessing i model. Cztery porównania widzą ten sam pełny zbiór train i ten sam checkpoint. Nie jest to ocena poza próbą ani test generalizacji.

Strumieniowo zachowujemy top64 różnych pełnych pikseli RGB na zapytanie. Wspólna unia par archive/member jest przeliczana pojedynczo, każda para jeden raz, i porównywana ponownie z czterema zapytaniami. Po refinowaniu wybieramy top32 różnych pikseli i par na zapytanie. Dokładny hash RGB uwzględnia rozmiary; zapisujemy wszystkie aliasy archive/member wybranych pikseli. Duplikaty między zapytaniami są dopuszczone; raport pokazuje macierz nakładania i liczbę obrazów unikalnych globalnie.

Wynik obala niepełne pokrycie, błędny split, niezgodność kontrolnego forwardu, zmiana wag lub duplikat w paczce. Duże odległości przeczą bliskiej realizacji lokalnej; bez ustalonego progu raportujemy wartości i nie deklarujemy sukcesu. Podobieństwo latentu nie gwarantuje realizmu ani podobieństwa wizualnego.

Proces vit_sd_zgodnosc90 ma priorytet GPU. Jeśli jest aktywny, wyszukiwanie kolejkuje się lub oddaje GPU. Nie czekamy na nieistniejący proces; aktywna kolejka ma limit 90 minut i uczciwy FAILURE. Pozostałe procesy pozostają nietknięte.

Artefakty: cztery pełne CSV, oryginalne zdjęcia w ZDJECIA_32, stykówki 4x8, TOP32 CSV/JSON, naturalne query NPZ i wybrane latenty NPZ, TRAIN_MEMBERSHIP, COVERAGE, dedup provenance, PROGRESS, INTERIM_TOP64, RESULT, raport i HASHES bez aktywnych logów. Znane brudne drzewo Git jest oznaczone; bez commitów.

## Zmiana zakresu zatwierdzona przez użytkownika 2026-10-05T15:56:52.835940+00:00

Użytkownik polecił zakończyć pełny skan przy bieżącej liczbie około 90 tysięcy i wyeksportować cztery paczki z już obejrzanego podzbioru. Nie wznawiamy skanu. Ponieważ pierwotny proces nie obsługiwał flagi stop, wysłano Ctrl+C wyłącznie do jego zweryfikowanej prywatnej konsoli; przerwanie nastąpiło przed następną paczką, a CSV zamknięto. Migawki INTERIM_TOP64 i PROGRESS zachowano. Tryb --finalize-existing odtwarza top64 z kompletnych CSV, sprawdza identyczny porządek i liczbę rekordów we wszystkich czterech zapytaniach, a następnie wykonuje tylko pojedyncze forwardy finalistów i eksport. Raport jest PARTIAL SCAN; mianownik rankingu to dokładny zapisany podzbiór, nie cały train. Podzbiór jest deterministycznym prefiksem uporządkowanych ZIP i plików, nie losową próbką.
