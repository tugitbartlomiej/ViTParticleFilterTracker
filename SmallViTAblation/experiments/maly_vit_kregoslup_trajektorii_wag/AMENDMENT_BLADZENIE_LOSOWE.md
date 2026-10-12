# Poprawka do planu: próg K1 nie odróżnia kręgosłupa od błądzenia losowego

Data: 2026-10-05, po odczytaniu pierwszego wyniku (`wyniki/KREGOSLUP.json`). PLAN.md zostaje bez zmian.

## Co było nie tak z kryterium

K1 wymagało, żeby pierwszy kierunek główny niósł co najmniej 0,70 kwadratu przesunięcia wag. Tyle i więcej daje
zwykłe błądzenie losowe w wielu wymiarach. Antognini i Sohl-Dickstein 2018 („PCA of high dimensional random walks
with comparison to neural network training”) pokazali, że składowe główne błądzenia losowego niosą udziały malejące
jak `1/k²` (po centrowaniu pierwsza `6/π² ≈ 0,61`), a rzuty wyglądają jak gładkie krzywe. Dla miary użytej tutaj
(przesunięcia względem punktu startu, bez centrowania) ten sam rachunek daje dla pierwszego kierunku `8/π² ≈ 0,81`;
potwierdza to obliczenie w `wyniki/POROWNANIE_Z_BLADZENIEM.json` (0,811), a współrzędna wzdłuż tego kierunku rośnie
monotonicznie z samej konstrukcji. K1 jest więc spełnione dla każdej trajektorii, która dyfunduje, i nie rozstrzyga
pytania. To samo dotyczy liczby 78-81 procent z pracy Xu i in. 2026,
jeśli czytać ją w oderwaniu od ich drugiej miary (stałości kierunku).

K2 (cosinus sąsiednich okien 100 kroków co najmniej 0,5) było postawione na surowych przesunięciach okien. Xu i in.
podają 0,8 dla stałości kierunku kręgosłupa między sąsiednimi oknami oraz 0,15-0,34 dla zgodności skumulowanego kroku
z 200 kroków z kierunkiem kręgosłupa. Definicji okna z pierwszej liczby nie mam z pełnego tekstu, więc porównywalna
jest tylko druga.

## Co dodano (opisowo, bez kryterium z góry)

1. `kod/porownanie_z_bladzeniem.py` → `wyniki/POROWNANIE_Z_BLADZENIEM.json`: te same miary dla błądzenia losowego
   (bez pamięci i z korelacją sąsiednich przyrostów 0,3, jak daje pęd) oraz dopasowanie „stały dryf plus błądzenie”.
2. `kod/dryf_okna.py` → `wyniki/DRYF_OKNA.json`: cosinus przesunięć w oknach, które nie sąsiadują (odstęp co najmniej
   długości okna). Dla błądzenia losowego i dla drgań wokół stałego środka wynosi zero, więc wartość dodatnia mierzy
   wprost udział trwałego dryfu w energii przesunięcia okna. Liczone dla wag i dla latentu sond, także po wygładzeniu
   latentu średnią z 50 kroków.
