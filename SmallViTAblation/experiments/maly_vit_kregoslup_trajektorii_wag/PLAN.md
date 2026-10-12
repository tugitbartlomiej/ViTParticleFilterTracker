# Kręgosłup trajektorii wag małego ViT: czy wolny dryf jest niskowymiarowy i trwały

Data: 2026-10-05. Plan zapisany przed uruchomieniem kodu. Pomiar bez treningu i bez Edenu, na zapisanych wagach i
latentach biegu gęstej historii (Eden 1818732, ziarno 47).

## Po co

Pomiar z 3 października (`../maly_vit_sufit_przewidywalnosci_100`) pokazał, że w latencie nie ma trwałego dryfu:
kierunek przesunięcia gładkiej ścieżki w kolejnych 100 krokach nie zgadza się z poprzednimi 100 krokami (cosinus od
-0,01 do -0,34). Praca Xu i in. 2026 („Optimizer-Induced Low-Dimensional Drift and Transverse Dynamics in Transformer
Training”) pokazuje coś innego dla wag: w transformerze uczonym AdamW przesunięcie wag jest prawie jednowymiarowe
(pierwszy kierunek główny niesie 78-81 procent kwadratu przesunięcia) i lokalnie stabilne (cosinus sąsiednich okien
około 0,8), a drgania leżą w poprzek tego kierunku. Jeśli nasze wagi mają taki „kręgosłup”, to wolną część ruchu
warto opisywać jedną współrzędną postępu, a nie 768 liczbami latentu. Jeśli nie mają, trzeba to wiedzieć, zanim
ktokolwiek zacznie prognozować wagi.

## Pytania

1. Jaka część kwadratu przesunięcia wag leży w pierwszym kierunku głównym? Osobno dla całej sieci i dla grup wag
   (łodyga splotowa, bloki 0-5, głowa), od kroku 0 i od kroku 300 (koniec rozgrzewki współczynnika uczenia).
2. Czy kierunek przesunięcia w sąsiednich oknach 50 i 100 kroków jest trwały?
3. Jaka część drogi to ruch tam i z powrotem (przesunięcie netto podzielone przez długość drogi liczonej co 10 kroków)?
4. Czy współrzędna postępu wzdłuż pierwszego kierunku rośnie monotonicznie?
5. Te same cztery wielkości dla latentu 180 sond z narzędziem (średnia po tokenach narzędzia) i dla ich tła, z tego
   samego biegu i tych samych kroków. To porównanie „wagi kontra latent” na jednej skali.

## Dane

- Wagi: `../maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi/k0000.pt ... k1000.pt`, 101 stanów co 10 kroków,
  2,93 mln parametrów. Bufory BatchNorm (średnie i wariancje bieżące) pomijam, bo nie są wagami uczonymi gradientem.
- Latent sond: `../maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane/LATENT_SOND.npz`, (101, 192, 2, 768).

## Aparat

Wszystko liczę z macierzy iloczynów skalarnych przesunięć `K[a,b] = <θ_a − θ_0, θ_b − θ_0>` (101 × 101, podwójna
precyzja), osobno dla każdej grupy; cała sieć to suma grup.

- **Udział kierunków głównych:** wartości własne macierzy przesunięć względem stanu odniesienia (krok 0 albo 300),
  bez centrowania, jak w pracy Xu i in.
- **Trwałość dryfu:** cosinus między przesunięciem w oknie `[t, t+w]` i w oknie `[t−w, t]`, `w` = 50 i 100 kroków,
  wszystkie położenia okna od kroku odniesienia; mediana i kwartyle.
- **Netto do drogi:** `|θ_koniec − θ_start|` podzielone przez sumę długości kroków co 10.
- **Autokorelacja przyrostów co 10 kroków** dla opóźnień 1, 2, 3, 5, 10.
- **Współrzędna postępu:** rzut na pierwszy kierunek; udział dodatnich przyrostów.

## Kryteria ustalone przed wynikiem

- **K1, kręgosłup istnieje:** pierwszy kierunek niesie co najmniej 0,70 kwadratu przesunięcia całej sieci od kroku 300
  do 1000.
- **K2, dryf trwały:** mediana cosinusa sąsiednich okien 100 kroków (od kroku 300) co najmniej 0,5 dla całej sieci.
- **K3, opisowo:** netto do drogi; udział dodatnich przyrostów współrzędnej postępu; te same liczby dla latentu.

Jeśli K1 i K2 są spełnione dla wag, a dla latentu nie, to wolny ruch jest uporządkowany w wagach, a latent sond
widzi go słabo na tle drgań. Jeśli nie są spełnione także dla wag, to w tym biegu nie ma wolnego kierunku do
prognozowania w żadnej z tych dwóch przestrzeni.

## Czego ten pomiar nie dowodzi

- Jedno ziarno, pierwsze 1000 kroków (z tego 300 to rozgrzewka), baza 1500 klatek bez dokładania.
- Zapis co 10 kroków: drgania krótsze niż 10 kroków są uśrednione i niewidoczne.
- Istnienie kręgosłupa w wagach nie znaczy, że da się go przedłużać w przyszłość ani że latent za nim podąża liniowo.
