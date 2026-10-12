# Czy mAP drga razem z wagami i czy uśrednienie zapisanych wag je wygładza i podnosi

Data: 2026-10-05. Plan zapisany przed uruchomieniem kodu. Bez treningu i bez Edenu: sama ocena zapisanych wag biegu
gęstej historii (`../maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi`, Eden 1818732, ziarno 47) na lokalnej
karcie. Żadna waga nie jest uczona ani zapisywana z powrotem do tamtego katalogu.

## Po co

Trzy pomiary z 5 października dają jeden obraz: latent sond drga wokół wolno płynącego środka, pchnięcie paczki w
większości wraca, a mAP w zapisach kampanii v2 zmienia się między ocenami co 100 kroków średnio o 0,04, czyli o tyle,
ile wynosi rozrzut między ziarnami i dwa razy więcej niż zysk z doboru klatek (0,016 pola pod krzywą). Jeśli te
wahania mAP to te same drgania, to (1) pojedyncza ocena mAP jest złą miarą skutku decyzji o danych, (2) wagi
uśrednione po kilkudziesięciu krokach mogą dawać wyższe mAP przy tej samej liczbie kroków, bez żadnego doboru.
Literatura (Morales-Brotons i in. 2024, Izmailov i in. 2018) mówi, że uśrednianie wag zmniejsza rozchwianie
przewidywań i poprawia jakość; dla tego detektora nikt tego nie sprawdził.

## Pytania

1. Jak bardzo mAP@0,5 waha się między zapisami wag odległymi o 10 kroków?
2. Czy wagi uśrednione z ostatnich 50, 100 i 200 kroków dają wyższe mAP niż wagi z końca okna i niż średnie mAP w
   tym oknie?
3. Czy ma znaczenie, jak potraktować statystyki BatchNorm w łodydze (uśrednić je jak wagi albo przeliczyć na
   klatkach uczących)?

## Dane i pomiar

- Wagi surowe: zapisy `k0500.pt, k0510.pt, ..., k1000.pt` (51 stanów). mAP@0,5 każdego stanu na tej samej próbce
  3000 klatek walidacji co w biegu (`dane.walidacja()`, ziarno próbki 0), tym samym kodem (`trening.wykryj`,
  `trening.ap50`). Kontrola zgodności: mAP w krokach 500, 600, ..., 1000 musi odtworzyć `OCENY.json` biegu.
- Wagi uśrednione: średnia arytmetyczna parametrów z `K` ostatnich zapisów (`K` = 5, 10, 20, czyli okna 50, 100, 200
  kroków), liczona w krokach 600, 700, 800, 900, 1000.
  Statystyki BatchNorm w dwóch wariantach: (a) uśrednione jak wagi, (b) przeliczone od zera na 512 klatkach bazy
  (16 paczek po 32, bez augmentacji, średnia skumulowana).
- Dodatkowo średnia wykładnicza po zapisach (waga nowego zapisu 0,2 i 0,1, od kroku 300), BatchNorm jak w (b).

## Kryteria ustalone przed wynikiem

- **K0 (zgodność):** mAP surowych wag w krokach 500-1000 co 100 różni się od `OCENY.json` o mniej niż 0,005.
- **K1 (opisowo):** średniokwadratowa zmiana mAP między sąsiednimi zapisami co 10 kroków po odjęciu trendu liniowego
  (kroki 500-1000) oraz rozstęp mAP w oknach 100 kroków.
- **K2 (uśrednianie pomaga):** mAP wag uśrednionych z okna 100 kroków (wariant (b)) jest wyższe od średniego mAP
  surowych wag z tego samego okna o co najmniej 0,01 w co najmniej 4 z 5 chwil ORAZ wyższe od mAP surowych wag z
  końca okna w co najmniej 4 z 5 chwil.
- **K3 (opisowo):** zależność od długości okna i od wariantu BatchNorm.

Jeśli K2 jest spełnione, to w każdej następnej kampanii mAP należy liczyć także na wagach uśrednionych, a skutki
decyzji o danych oceniać na nich. Jeśli nie, wahania mAP nie biorą się z drgań, które da się uśrednić w wagach, i
trzeba je traktować jak szum pomiaru (średnia z kilku ocen).

## Czego ten pomiar nie dowodzi

- Jedno ziarno, kroki 500-1000 (wczesne uczenie, mAP 0,24-0,31), baza 1500 klatek bez dokładania.
- Uśrednianie po fakcie, z zapisów co 10 kroków; średnia prowadzona w trakcie uczenia co krok może dać inny wynik.
- Walidacja, nie zbiór testowy. Pięć chwil pomiaru to mała próba; okna zachodzą na siebie dla `K` = 20.
- Nie mówi nic o tym, czy uczenie OD wag uśrednionych jest lepsze (literatura: nie jest).
