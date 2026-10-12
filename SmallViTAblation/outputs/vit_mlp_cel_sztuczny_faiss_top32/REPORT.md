# Sztuczny latent → MLP → DINO → FAISS

Zweryfikowano wagi, preprocessing,35000 wierszy, każdy z35 chunków, świeże wczytanie MLP i zgodność FAISS z brute force. Indeks IndexFlatIP zbudowany w RAM w osobnym procesie, ponieważ finalizacja istniejącego procesu indeksowania zatrzymała się na konflikcie OpenMP. Nie liczymy ponownie biblioteki DINO i nie zapisujemy tablic latentów.

Najbliższy obraz: TR_0082_S1_P03/934. Cosine DINO: 0.825064. Naturalny forward oryginalnego ViT względem sztucznego celu: MSE128 85.185991, cosine128 0.215531, umowna zgodność wartości128 0.7629 procent. To kandydat według przybliżonego translatora, nie dowód realizacji celu przez prawdziwy obraz.

Cosine między przetłumaczonym celem a niezmienionym latentem: 0.694056; wspólne zdjęcia w top32: 0. Wyłącznie skalary i metadane zapisane w RESULT.json; kopie32 zdjęć w zdjecia/, galeria i przegląd TOP32.jpg.
