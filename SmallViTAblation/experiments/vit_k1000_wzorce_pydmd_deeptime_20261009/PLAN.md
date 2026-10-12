# Plan wykonawczy

1. Kontrakt danych i izolacja rodziny. Nie modyfikować innych eksperymentów.
2. Prywatne środowisko na Edenie; rzeczywiste PyDMD i deeptime, zapis wersji, pip freeze i hash źródeł bibliotek.
3. Testy przed implementacją: train-only scaling, split filmów, brak par przekraczających token/film/fazę, znany układ sterowany przez publiczne DMDc.fit, known-B kontra learned-B, odrzucenie PCA3/braku telemetrii. Testy ograniczone do tej rodziny zgodnie z ownership.
4. Smoke na istniejącym panelu: 16 obrazów/128 trajektorii, bez targetów, nie raportować go jako nowego pełnego panelu. Prawdziwy fit i predykcja bibliotek.
5. Bogaty panel od Hubble'a: adapter jego gotowego kontraktu, walidacja hashy, podział filmów, fit train, wyniki 1/5/20 osobno po fazach/rolach/sukcesie.
6. Submit CPU z dependency po potwierdzonym ID producenta; zapisać ID, config i provenance. Bez nadpisania jego plików i bez ponowienia oracle przez tę rodzinę.

Pytanie: czy operator warunkowany dostępnym przed aktualizacją wejściem przewiduje ruch lepiej od kompetentnych baseline i czy cechy temporalne mają powtarzalną strukturę w nowych filmach? Nie jest to jeszcze trening korektora detekcji bez GT.

Podłogi: persistence zachowuje aktualny stan; stały przyrost fit train zachowuje średni drift; skurcz zachowuje znaną tendencję powrotu do z0. Adam known-B ma uprzywilejowane wejścia i jest oznaczony jako kontrola integracji, nie zwycięski predictor bez GT. Prognozy wielokrokowe learned-B mają prawdziwe wejścia oracle, lecz rekurencyjny przewidywany stan. Raport uwzględnia niestabilne rollouty bez ukrywania ich clippingiem wyników.
