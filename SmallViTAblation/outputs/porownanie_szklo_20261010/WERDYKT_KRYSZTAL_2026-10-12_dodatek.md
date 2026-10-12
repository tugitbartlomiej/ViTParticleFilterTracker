# Dodatek 2026-10-12: czy model „sfera 768 laserów + 600 koloidów” dobrze opisuje MalyViT?

Kod: `eden/test_sfery_laserow.py`, wynik `eden/wyniki/TEST_SFERY.json`; rozkład ruchu z `eden/wyniki/wiazki_koloidy/WIAZKI.json` (Eden, zadanie 1826881, 100 zdjęć).
Latent z_j = GELU(W u_j + b): u_j ∈ R^192 = wejście fc1 ostatniego bloku po LayerNorm („położenie cząsteczki”), wiersze W = 768 „laserów”.

| test | wynik | dla metafory |
|---|---|---|
| Triangulacja: punkt u_j dopasowany do 80 % natężeń przewiduje pozostałe 20 % | R² 0,99997 (mediana) w kroku 0, epoce 1, 21 i 60 | ognisko = przecięcie wiązek: DOKŁADNIE, ale w 192D, nie w 3D |
| Ile laserów świeci (z > 0) na cząsteczkę | 374 → 289 → 185 → 156 (krok 0, ep. 1, 21, 60) | późno mniej jasnych wiązek niż wymiarów (192); same jasne dają R² 0,76, z ciemnymi 0,99997 — położenie koduje też to, które lasery są CIEMNE |
| Wymiar efektywny chmury 600 cząsteczek w 768D | 12 → 4,3 → 6,3 → 6,9 | 768 laserów to nadmiar: chmura żyje w kilku wymiarach, stopni swobody na cząsteczkę jest 192 |
| Udział zmiany laserów (W) w energii ruchu, krótkie opóźnienie | 2 % (kroki 100–1000), 11–16 % (epoki 1–60) | szybkie drgania robi ruch samych cząsteczek u (sieć przed fc1), nie migotanie laserów |
| To samo, opóźnienie 40 checkpointów | 39–68 % | przy wolnym dryfie lasery się liczą |
| Cała droga epoka 1 → 60 | 75 % zmiany z laserów; lasery obrócone (cos 0,24) | dryf w skali epok = przestawianie laserów |
| Korelacja sąsiadów w ruchu: tylko lasery / tylko cząsteczki | 0,93 / 0,88 | wspólny „wiatr” powstaje obiema drogami |

Wniosek: sfera laserów to dobry model ODCZYTU (detektory wspólne dla wszystkich tokenów, triangulacja dokładna, wolny dryf przez obrót laserów), ale zły model SIŁ:
lasery niczego nie trzymają (brak sprzężenia zwrotnego z na u w tym samym przejściu), sieć adresów pochodzi z kodowania pozycji w u, sprzężenia między cząsteczkami
(uwaga ~2 % globalnie, nakładanie pól lodygi lokalnie) w modelu nie występują, a szybkie drgania płyną z generatora wprost do cząsteczek (84–98 % energii), nie przez lasery.
Film 5 (`lasery_5_drgania.mp4`) rysuje drogę generator → lasery → cząsteczki; dla szybkich drgań główna droga to generator → cząsteczki (warstwy przed fc1).
