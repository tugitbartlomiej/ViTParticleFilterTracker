# Plan eksperymentu

## Pytanie

Które naturalne klatki z treningowego zbioru 35 000 obrazów są najbliższe każdemu z 11 obrazów wygenerowanych podczas interpolacji latentu m0011?

## Metoda

- Zapytania: 11 zapisanych obrazów `k00`–`k10` oraz obraz referencyjny m0011 jako kontrola.
- Reprezentacja: DINOv2 ViT-S/14, znormalizowany wektor CLS o 384 wymiarach.
- Przetwarzanie: krótszy bok 256, centralne wycięcie 224, normalizacja ImageNet.
- Biblioteka: istniejący indeks 35 000 klatek, wyłącznie split `TRAIN`.
- Miara: podobieństwo cosinusowe, dokładne wyszukiwanie po całej macierzy cech.
- Wynik: pięć najbliższych naturalnych klatek dla każdego zapytania oraz zbiorczy pasek wyników top-1.

## Kryteria poprawności

- manifest ma dokładnie 35 000 rekordów i wszystkie należą do `TRAIN`;
- macierz cech ma kształt 35 000 x 384;
- każde zapytanie zwraca pięć prawidłowo odczytanych klatek z archiwów;
- wyniki zawierają identyfikator filmu, numer klatki i cosine.

## Ograniczenie interpretacyjne

Najbliższy obraz w DINO oznacza podobieństwo wizualno-semantyczne w przestrzeni DINO. Nie dowodzi zgodności pełnego latentu ViT ani identycznego wpływu obrazu na trening detektora.
