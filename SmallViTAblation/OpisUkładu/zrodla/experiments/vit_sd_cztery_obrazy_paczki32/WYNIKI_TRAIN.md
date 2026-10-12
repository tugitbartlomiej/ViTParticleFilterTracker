# PARTIAL SCAN: cztery paczki po 32 zdjęcia train

Zakres rankingu: 88469 przetworzonych zdjęć train z 23 rozpoczętych ZIP; cały dostępny train liczy 310257. Ranking najbliższych wyłącznie w tym podzbiorze. Kolejność to posortowane ścieżki ZIP i nazwy plików wewnątrz; deterministyczny początek, nie losowa próbka. Jeden forward na kandydata; 255 dodatkowych forwardów finalistów. Naturalne zapytania z czterech obrazów SD, bez sztucznego celu. Zamrożone wagi, RGB 480x320 INTER_AREA, ranking surowe MSE 600x768.

Każda paczka ma 32 różne hashe pełnych pikseli RGB i różne pary archive/member. Łącznie 128 wyborów, ale 128 różnych obrazów. Dopuszczamy powtórzenia pomiędzy zapytaniami. Macierz wspólnych pikseli (kolejność [2026100511, 2026100523, 2026100537, 2026100549]):

[[32, 0, 0, 0], [0, 32, 0, 0], [0, 0, 32, 0], [0, 0, 0, 32]]

Ranking nie gwarantuje podobieństwa wizualnego. To retrieval ze zbioru train, bez pomiaru generalizacji. Pewność detektora nie jest dokładnością. Znane niezacommitowane zmiany repozytorium. TOP32, pełne CSV, latenty, oryginalne zdjęcia i stykówki są w katalogach zapytań.
