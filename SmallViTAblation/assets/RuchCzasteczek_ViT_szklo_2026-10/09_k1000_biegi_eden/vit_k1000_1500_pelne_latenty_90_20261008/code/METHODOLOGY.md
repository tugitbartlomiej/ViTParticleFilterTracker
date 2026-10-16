# Metodologia dynamiki i reżimów

## DMD i DMDc

DMD opisuje pary kolejnych stanów przez `z_(t+1) = A z_t`. DMDc rozszerza ten
model do `z_(t+1) = A z_t + B u_t`, gdzie `u_t` musi być rzeczywistym wejściem
dostępnym przed przejściem. Podstawa metodologiczna:

- Proctor, Brunton, Kutz, *Dynamic Mode Decomposition with Control*, SIAM
  Journal on Applied Dynamical Systems 15(1), 2016,
  https://doi.org/10.1137/15M1013857
- oficjalny tutorial PyDMD DMDc:
  https://pydmd.github.io/PyDMD/tutorial7dmdc.html
- dokumentacja DMD deeptime:
  https://deeptime-ml.github.io/latest/notebooks/dmd.html

Środowisko Eden użyte przez ten eksperyment nie zawiera PyDMD, deeptime ani
scikit-learn. Skrypt `analiza_dmdc.py` wykonuje więc bezpośrednio równanie
najmniejszych kwadratów z artykułu w NumPy. Jest to zadanie CPU. Redukcja stanu
i wejścia używa deterministycznego randomized SVD.

`u_t` zawiera surowy gradient oraz pierwszą i drugą chwilę Adama zapisane przed
`optimizer.step`. Ruch `z_(t+1)-z_t` nie jest wejściem służącym do przewidywania
samego siebie. Pary nie przekraczają granic obrazów ani tokenów. Ocena jest na
odłożonych całych trajektoriach.

DMDc jest wyłącznie analizą opisową procesu oracle. Nie jest używany do
przypisania reżimu ani do wspólnej korekty, ponieważ gradient i stan Adama nie
są dostępne przy zwykłym wnioskowaniu ze świeżego obrazu.

## Osiągalność celu i interpretacja oracle

Historyczny lub sztucznie zadany latent po GELU może leżeć poza manifoldem
osiągalnym przez naturalny obraz i wcześniejsze warstwy. Dla przyszłych nowych
celów trzeba osobno badać ich realizowalność oraz składowe niewidoczne dla
`fc2`, w tym ruch w jej nullspace. Obecna optymalizacja oracle działa jednak
bezpośrednio w punkcie interwencji i wyznacza górną granicę możliwości
zamrożonego ogona oraz głowy. Nie wymaga założenia, że znaleziony stan da się
odtworzyć z pikseli. Wyniku oracle nie wolno przedstawiać jako dowodu
osiągalności przez naturalny obraz.

## Reżimy i brak wycieku celu

Reżimy odkrywa się na `z0`, kierunku oracle delta i logarytmie magnitudy oracle
delta. To etap opisujący przyszłe trajektorie. Następnie dla każdego odkrytego
reżimu oblicza się centroid cech dostępnych przed korektą: projekcji naturalnego
`z0`, pozycji tokenu oraz średniego kontekstu `z0` obrazu. Podczas inferencji
reżim wybiera wyłącznie najbliższy taki centroid. Oracle delta nie jest
argumentem warstwy wykonawczej.

Ocena in-sample jest oznaczona jako dopasowanie na tych samych 1500 obrazach.
Ocena wewnętrzna używa trzech rozłącznych foldów grupowanych po filmie; cały
model reżimów i tablice korekt danego foldu powstają bez filmów testowych.
