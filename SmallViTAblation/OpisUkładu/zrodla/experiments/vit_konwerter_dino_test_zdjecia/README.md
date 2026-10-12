# Test konwertera ViT -> DINO na kilku zdjęciach

Uruchomienie: `F:/Python/python.exe -X utf8 experiments/vit_konwerter_dino_test_zdjecia/test.py`.
Wyniki: `outputs/vit_konwerter_dino_test_zdjecia` (`WYNIKI.json`, `MACIERZ_ZESTAWU.npy`, `macierz_zestawu.png`,
`zdjecie_NN_<film>.png`).

Konwerter: `outputs/vit_konwerter_dino_filmy_eden/v2/KONWERTER.pt`. Kod obrazów, ViT i DINO importowany z
`experiments/vit_konwerter_dino_filmy_eden` (te same funkcje co w uczeniu; SHA256 plików w `WYNIKI.json`).

Zdjęcia: 8 klatek JPEG z archiwów adnotacji, po jednej ze środka 8 filmów VAL rozłożonych równo po liście.
Konwerter nie widział tych filmów, a żaden z nich nie jest w bazie. JPEG to domena bazy 35000, a uczenie szło na MP4.
Baza: `outputs/vit_dino_faiss_35000/features.npy` (35000 klatek TRAIN, CLS DINO), przeszukiwanie dokładne.

## Wynik 2026-10-06

| film VAL | cos CLS | po odjęciu średniej | cos tokenów | trafienie w zestawie 8 | prawdziwe top1 w kolejności predykcji | film top1: prawdziwe / predykcja |
|---|---|---|---|---|---|---|
| TR_0002 | 0,743 | 0,559 | 0,744 | tak | 17 | TR_0003 / TR_0065 |
| TR_0040 | 0,772 | 0,551 | 0,774 | tak | 382 | TR_0169 / TR_0047 |
| TR_0060 | 0,609 | 0,292 | 0,764 | nie | 10 432 | TR_0045 / TR_0166 |
| TR_0099 | 0,831 | 0,634 | 0,809 | tak | 34 | TR_0095 / TR_0095 |
| TR_0118 | 0,762 | 0,488 | 0,797 | tak | 1201 | TR_0122 / TR_0126 |
| TR_0129 | 0,637 | 0,451 | 0,746 | tak | 187 | TR_0029 / TR_0081 |
| TR_0159 | 0,739 | 0,452 | 0,745 | nie | 172 | TR_0158 / TR_0091 |
| TR_0163 | 0,845 | 0,567 | 0,800 | tak | 10 | TR_0158 / TR_0158 |

Średnio: cos CLS 0,742, po odjęciu średniej 0,499, tokeny 0,772; trafienia w zestawie 6/8 (losowo 1/8).
Wyszukiwanie w bazie: wspólne top5 z prawdziwym DINO średnio 0,125 z 5; mediana pozycji prawdziwego top1
w kolejności z predykcji 172 z 35000. Zdjęcie znalezione predykcją ma do zapytania cosinus 0,809, najlepsze
w bazie 0,870, losowe 0,681: predykcja znajduje podobne sceny, ale nie te same co prawdziwe DINO.

Wnioski: w małym zestawie konwerter rozróżnia zdjęcia (6/8). W bazie 35000 trafia film w 2 przypadkach na 8
(TR_0095, TR_0158), w pozostałych znajduje inną operację o podobnym wyglądzie. Najgorszy jest TR_0060
(jasne, prześwietlone ujęcie): cos po odjęciu średniej 0,29, prawdziwe top1 na pozycji 10 432.
8 zdjęć to pokaz, nie pomiar; liczby z pełnego VAL są w `vit_konwerter_dino_filmy_eden/README.md`.
