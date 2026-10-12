# Optymalizacja 128 kanałów zamrożonego ViT

2026-10-04. Zlecenie: przygotować i wykonać eksperyment na Edenie. Rodzina badawcza.

Pytanie: jakie wartości 128 wybranych kanałów po ostatnim GELU poprawiają poprawną detekcję, i czy wspólna korekta działa na innych filmach? Nie jest to trening ViT, prognoza Langevina ani dowód istnienia obrazu realizującego zmodyfikowany latent.

Model: identyczny MalyViT z checkpointu k0500 biegu seed47. Kanały i skale z KANALY_MODEL.npz wybrane wcześniej na tym checkpointcie. Stan pełny 600×768; zmienne tylko 600×128. Pozostałe 640 kanałów, wagi, bufory i residual sprzed końcowego MLP pozostają identyczne. Używamy wszystkich 600 tokenów, bez maski GT do umiejscawiania interwencji. Źródło modelu i metryk skopiowane z eksperymentu maly_vit_tozsamosc_czasteczki_pary_400. Cache końcówki sieci musi odtwarzać pełny forward.

Dane: 400 oryginalnych klatek z istniejącego manifestu, split po filmach 240/75/85. Są to klatki z wcześniejszych analiz i zbioru treningowego ViT; odłożony split dotyczy uczenia nowej korekty, nie jest niezależnym nowym testem ViT. Etykieta to ramka cystotomu z manifestu, nie gwarancja kompletności etykiet wszystkich narzędzi.

Trzy równoległe ramiona na osobnych GPU: maksymalna bezwzględna zmiana każdej współrzędnej 0.25/0.5/1.0 odchylenia standardowego kanału z wcześniejszych train pooled features. Ta skala jest umowną skalą interwencji, nie SD pojedynczych tokenów. Optymalizacja Adam lr0.05; strata detektora CenterNet + 2 GIoU i kara 0.01 mean(delta²). Ograniczenie box, nie gwarancja osiągalności przez naturalny obraz. Maksimum globalne nie jest gwarantowane.

T1 oracle: dla każdego obrazu osobno optymalizować 80 kroków jego 600×128 przyrostów, znając GT; zachować najlepszy stan według własnej straty z karą, włącznie ze stanem zero. Ten pomiar pokazuje podatność głowy na interwencję i cele per obraz; poprawa na obrazie optymalizowanym nie jest generalizacją. Zapisać stan bazowy J, delta, docelowe wartości J, kierunki/normy i wyniki przed/po. Pełny latent odtwarza się z bazowego forward oraz tych 128 wartości.

T2 transfer: jeden wspólny wektor 128 przyrostów, stosowany do wszystkich tokenów wszystkich obrazów bez GT przy inferencji. Uczenie 20 epok na train; najlepsza epoka według film-weighted straty val, włącznie z zerem. Test po zamrożeniu. Wszystkie promienie raportowane, główny promień wybrany wyłącznie według val, przed analizą testów. Dobór kanałów pochodzi z wcześniejszego train/val; uczciwie nazwać analizą eksploracyjną.

Kontrole: brak zmiany; losowy znak zmian oracle przy zachowanej bezwzględnej zmianie każdej współrzędnej i identycznym ograniczeniu; losowy znak wspólnego wektora. Losowe kontrole trzy ziarna 11/22/33. Nie porównywać wielkości oracle z transferem jako modeli o równym dostępie do informacji.

Metryki: AP50 na opisanym zbiorze ramek, film-weighted strata, confidence×IoU, confidence, FP z confidence>=0.5 i IoU<0.5 do oznaczonej ramki (proxy, nie pełne FP przy niekompletnych adnotacjach). Raportować delta i normy. Dla transferu sparowany bootstrap całych filmów 1000 powtórzeń dla straty i confidence×IoU. AP opisowe; jednostką replikacji nie są tokeny. Nie ogłaszać poprawy ogólnej detekcji z jednego checkpointu i adaptacji latentów.

Kryterium: T1 musi poprawić własną stratę poza losową kontrolą; jeśli AP lub IoU nie rosną, rozbieżność raportować. T2 uznajemy za wstępnie użyteczny, jeśli na odłożonych filmach spada strata z CI różnicy poniżej zera, rośnie AP50 i wynik confidence×IoU. Ujemny transfer obala hipotezę jednego wspólnego przesunięcia, nie wszystkich reguł korekty.

Kontrole implementacji: nietknięte 640 kanałów; gradient w zmiennych; zgodność cache-tail i forward/hook; frozen state SHA; skończoność; ograniczenie promienia; rozdzielenie filmów. Najpierw smoke na GPU, potem trzy pełne ramiona zależne od jego sukcesu. Zasoby: 1 GPU i 16GB RAM na ramię, limit 2h, maksymalnie 3 równolegle. Żadnych zmian współdzielonego środowiska.

Wyniki nie określają jeszcze, jakie obrazy wywołają cele. Dalszy etap: inversion/retrieval i test osiągalności oraz wpływu rzeczywistego treningu. Ten etap nie wchodzi do obecnego uruchomienia.
