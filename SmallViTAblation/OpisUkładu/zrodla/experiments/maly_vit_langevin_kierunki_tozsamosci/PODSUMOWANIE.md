# Langevin dla ruchu cząsteczek w kierunkach tożsamości narzędzia

2026-10-02, Europe/Warsaw. Obliczenia lokalne na zapisach jednego biegu Edena, seed 47. Bez nowego treningu ViT.

**Dało się dopasować użyteczny model na jedną aktualizację: zmniejsza MSE o 60,24% względem braku ruchu i trafia znak zmiany współrzędnych w 78,61% przypadków.** Wynik 81,59% dotyczy wyłącznie zmian przekraczających zadany próg wielkości. Nie oznacza 81,59% dokładności całego wektora ani pewności prognozy. Korzyść na dłuższym horyzoncie jest znacznie mniejsza i zależy od odstępu pomiaru.

## Co dokładnie śledziliśmy

Stały obraz `cataract_lmm/tracking/TR_0057_S1_P03/00001431`, sonda o indeksie 2 (numeracja od zera). Ma 600 cząsteczek po 768 współrzędnych. Anotacja narzędzia przecina 31 patchy; wszystkie 31 śledzono oddzielnie, z wagami równymi pokryciu narzędziem. Osiem odległych patchy tła stanowi kontrolę. Obraz jest z treningu, ma źródłową rolę `fit`; ocenę prognozy odłożono w CZASIE, nie na nowy obraz.

Mamy 1001 stanów z aktualizacji 0..1000. Pełne wagi sieci są co 10, ale zapisane cząsteczki są co 1. Forward lokalnego checkpointu 500 na dokładnie tym obrazie zgadza się z zapisanym latentem: względny błąd L2 `4,30e-7`, maksymalny błąd składowej `1,79e-6`. Ruch obejmuje zmiany wag oraz bieżących statystyk BatchNorm w sieci.

Kierunki tożsamości wyznaczono ponownie w checkpointcie 500 na 240 parach „narzędzie obecne–usunięte przez LaMa” z 69 filmów. Walidacyjne i testowe pary nie wyznaczały tych kierunków. Zamrożono osiem kierunków U500. Diagnostyczny klasyfikator z ustalonym progiem 0,5 osiągnął na 85 testowych parach AUROC 95,53% i balanced accuracy 90,00%; są to miary po obrazach. To operacyjna identyfikacja obecności narzędzia w tych parach, nie dowód pełnej semantycznej tożsamości.

Najsilniejsze pojedyncze osie bazowe według standaryzowanego kontrastu train w checkpointcie 500: **359, 594, 472, 512, 138** (numeracja od 1). W checkpointcie 1000 wcześniejszy ranking wynosił 500, 472, 735, 512, 647. Osiem kierunków U500 jest kombinacjami wielu takich osi, a nie po prostu wyborem ośmiu pojedynczych kanałów.

[Obraz i numery śledzonych cząsteczek](CZASTECZKI_NA_STALYM_OBRAZIE.png) · [Zmiana składowych na pięciu osiach](RUCH_OSIE_K500.png) · [Trajektorie w kierunkach U500](RUCH_KIERUNKI_K500.png).

## Wyznaczone równanie

Dla każdej cząsteczki i:

```math
q_{i,t}=U_{500}^{T}z_{i,t},\qquad x_{i,t}=q_{i,t}-q_{i,500},
```

```math
x_{i,t+1}=x_{i,t}+b+A x_{i,t}
  +\sum_{\ell=0}^{3}K_\ell\bigl(x_{i,t-\ell}-x_{i,t-\ell-1}\bigr)
  +\varepsilon_{i,t+1}.
```

Stan x ma 8 współrzędnych. Wyraz b ma 8 liczb, A ma rozmiar 8×8, a jądro pamięci składa się z czterech macierzy 8×8. Macierze są wspólne dla 31 cząsteczek, lecz każda ma własny stan i własną historię ruchu. Macierze dryfu i pamięci dopasowano jednocześnie. Prognoza punktowa przyjmuje epsilon równe zeru; nie oznacza to, że zmierzone reszty testowe mają dokładnie zerową średnią. Reszty rzeczywistych prognoz analizujemy osobno.

Dryf stanowi skończoną, afiniczną aproksymację typu Koopmana/EDMD. Całe równanie jest empirycznym dyskretnym modelem GLE, algebraicznie odpowiadającym modelowi autoregresyjnemu z pamięcią. Nie wyprowadziliśmy dokładnego prawa Mori–Zwanziga dla całej sieci. W tym samym biegu nie zapisano faktycznej historii paczek, augmentacji i gradientów, więc jawna macierz sterowania obrazami nie została wyestymowana. Wpływ tych nieobserwowanych wejść pozostaje w resztach.

Pełne współczynniki w oryginalnych jednostkach są w [MACIERZE_DT1.npz](MACIERZE_DT1.npz), pola `selected_b`, `selected_A`, `selected_K`. U500 jest w [IDENTYFIKACJA_K500.npz](IDENTYFIKACJA_K500.npz). Przewidywany ruch części tożsamości w 768D można odtworzyć jako `U500 @ (q_hat - q_now)`. Pozostałych 760 ortogonalnych kierunków ten model nie prognozuje.

## Wynik sprawdzianu czasowego

Pierwszy fit: cele 501..700. Walidacja wybierała pamięć i regularyzację, korzystając wyłącznie z celów do 799. Potem refit do 799 i ocena od originu 800. Dla horyzontów 1..50 te same151 originów 800..950. Prognozy wielokrokowe są autonomiczne: bez rzeczywistych przyszłych stanów pomiędzy początkiem i końcem prognozy. Usunięcie albo zatrucie przyszłego sufiksu danych nie zmienia przewidywań.

Główny model: krok pomiaru 1, pamięć 4 przyrostów, ridge 0,01. MSE liczony w oryginalnej skali współrzędnych q, ważony pokryciem narzędziem.

| Horyzont, aktualizacje | Redukcja MSE wobec braku ruchu | Trafność znaku wszystkich zmian | Kosinus wektorów przemieszczenia |
|---:|---:|---:|---:|
| 1 | 60,24% | 78,61% | 0,740 |
| 2 | 44,28% | 73,83% | 0,623 |
| 5 | 12,34% | 63,87% | 0,383 |
| 10 | 9,11% | 65,34% | 0,367 |
| 20 | 9,40% | 66,62% | 0,413 |
| 50 | −5,03% | 63,84% | 0,393 |

Ujemna redukcja oznacza większy błąd niż prognoza braku ruchu. MSE, trafność znaku i kosinus są trzema odrębnymi miarami; żadna nie jest procentem poprawnie przewidzianego całego latentu.

Na jeden krok sama ostatnia prędkość redukuje MSE o 47,77%, a dryf bez pamięci daje −0,31%. Model z pamięcią ma więc dodatkową korzyść wobec obu kontroli w tym sprawdzianie. Oddzielny wariant dobrany na walidacji wyłącznie do H1 daje 60,59% redukcji MSE, ale 78,52% trafności znaku; nie zapewnia 80% dla wszystkich zmian.

Trafność 81,59% głównego modelu jest obliczona po pominięciu rzeczywistych zmian mniejszych niż 0,1 odchylenia standardowego przyrostów fit, osobno dla każdej współrzędnej. Próg pochodzi z przeszłości, lecz ocena tego, czy przyszła zmiana go przekroczyła, jest wykonywana po poznaniu wyniku. To analiza podzbioru zmian, nie prognozowana pewność ani reguła wyboru dostępna z góry.

Osobny model, którego krok wynosi 10 aktualizacji, wybrał pamięć jednego przyrostu i ridge 1. Dla H10/20/50 redukcja MSE wyniosła odpowiednio 16,12% / 20,12% / 13,28%. Na dokładnie tych samych 16 originach co 10 model gęsty uzyskał 12,27% / 5,90% / −4,25%. Odstęp 10 może więc być przydatniejszy do dłuższego ruchu, ale ma mniej obserwacji. Opisowy 95% przedział redukcji MSE na H50 obejmuje zero: −7,96% do 29,94%.

H100 zapisano dodatkowo jako diagnostykę na krótszej liście originów: redukcja MSE 13,05% dla dt=1 na 101 originach i 26,81% dla dt=10 na 11 originach. Nie porównujemy tych liczb bezpośrednio między oboma dt ani z H50 jako poprawy przy dłuższym horyzoncie; zmienia się lista ocenianych początków prognozy i zostaje około jeden blok czasowy. H500 nie oceniono w tym protokole.

![Prognozowane i rzeczywiste zmiany dwóch cząsteczek na horyzontach 1, 10 i 50 aktualizacji](PROGNOZY_H1_H10_H50.png)

Kolumna lewa: cząsteczka 324 na granicy narzędzia; prawa: cząsteczka 294 niemal całkowicie pokryta narzędziem. Czarna linia to rzeczywista przyszła zmiana pierwszej współrzędnej q, czerwona to prognoza. Każdy punkt na osi poziomej oznacza osobny początek prognozy z historią dostępną do tej chwili, a nie jeden nieprzerwany rollout od kroku 800.

## Co mówi to o cząsteczkach i makrowektorach

Makrowektory bazowe e_j są stałe. Zmieniają się składowe z_ij e_j cząsteczek. Zamrożone U500 pozwala mierzyć ten ruch w jednym układzie. Osobno porównaliśmy kierunki tożsamości wyznaczone ponownie na obu checkpointach: kąty główne pomiędzy podprzestrzeniami 500 i 1000 wynoszą 27,69..83,95 stopnia. **Oszacowana podprzestrzeń związana z narzędziem również się zmienia.** Nie należy utożsamiać tego z obrotem bazowych e_j. Szczegóły: [porównanie podprzestrzeni](porownanie_podprzestrzeni/WYNIKI.md).

Ten sam operator przeniesiony na osiem cząsteczek tła daje na H1 redukcję MSE 57,60% i trafność znaku 77,55%. Część przewidywalnego ruchu jest zatem wspólna także dla tła. Sam dobry forecast nie dowodzi, że odkryliśmy dynamikę występującą wyłącznie w cząsteczkach narzędzia. Zapis nie rozstrzyga, ile wnosi semantyka, a ile globalne zmiany modelu i BatchNorm.

Reszty różnych cząsteczek są mocno współzależne: średnie korelacje między parami cząsteczek w poszczególnych współrzędnych wynoszą około 0,49..0,89. Nie wolno traktować ich jako niezależnych realizacji białego szumu. Gaussowskość i brak korelacji czasowej są sprawdzane osobno w [SZUM_DT1.json](SZUM_DT1.json).

To eksploracyjny wynik jednego obrazu i jednego biegu. Cała historia była już wcześniej oglądana, chociaż sam fit i rollout korzystają tylko z dozwolonego prefiksu. Bootstrap blokowy ma około 3 efektywne bloki dla H1..50 i daje wyłącznie opisową niepewność. Wynik krótkoterminowy uzasadnia dalsze sprawdzanie modelu, ale nie ustanawia gwarancji 80% ani poprawy uczenia detektora przez użycie tej prognozy.

Pełny raport i wykresy: [WYNIKI.md](WYNIKI.md). Kontrola metody i wykonania: [AUDYT_ASTRA.md](AUDYT_ASTRA.md). Reprodukcja: [README.md](README.md).

Kontrola końcowa: audyt Astry ma status FINAL, bez błędu blokującego. Zgodne są hashe macierzy i danych, 36 kontroli prefiksu/przyszłego sufiksu oraz numeryczna zgodność skalowanej i oryginalnej postaci równania. Root sprawdził kompilację skryptów, integralność plików PNG, odnośniki lokalne i wizualną czytelność głównych wykresów. Nowe artefakty zajmują około 22 MB; pliku źródłowego 22 GB nie kopiowano.
