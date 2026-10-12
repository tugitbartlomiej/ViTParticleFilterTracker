# ViT K8000: podmiana 1 z 256 obrazów

Stan: zakończony na Edenie 8 października 2026. Wszystkie 9 gałęzi i agregacja zakończyły się kodem `0:0`.

Wyniki i analiza znajdują się w `../../outputs/vit_k8000_podmiana_1z256_dino_m0011_eden_20261008/results/`. Raport `ANALIZA.md` stosuje różnicę przyrostów względem kontroli, ponieważ sam dalszy trening odpowiada za większość poprawy, a wykonanie BF16 na różnych GPU dało niewielkie różnice już w kroku zero.

Eksperyment porównuje kontrolę z ośmioma unikalnymi naturalnymi obrazami top-1 DINO odpowiadającymi 11 obrazom SD z interpolacji latentu m0011.

Kod uruchamiany jest na Edenie jako tablica dziewięciu niezależnych wariantów. Każdy wariant zaczyna od tych samych wag K8000 i wykonuje 500 aktualizacji na puli 256 obrazów.

Źródła wyników wyszukiwania:

`outputs/latent_cel_do_obrazow/sd_k8000_m0011_dino_train_retrieval/WYNIK.json`

Docelowe wyniki:

`outputs/vit_k8000_podmiana_1z256_dino_m0011_eden_20261008`
