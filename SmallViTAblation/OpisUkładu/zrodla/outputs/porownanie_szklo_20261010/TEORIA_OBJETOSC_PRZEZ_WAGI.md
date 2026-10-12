# Jak wprowadzić „objętość”, „gęstość” i „ośrodek” do modelu ruchu cząsteczek ViT

Data: 2026-10-10. Odpowiedź na pytanie: w szkle koloidalnym cząstki mają objętość wykluczoną i pływają w cieczy o zadanej
gęstości; czy u nas da się to przetłumaczyć, skoro cząsteczki (tokeny) oddziałują tylko przez wspólne wagi?

## 1. Równanie ruchu tokenu: oddziaływanie idzie przez jądro tangensowe

Cząsteczka i to wektor tokenu `z_i(t) = f_θ(t)(x)` (stałe zdjęcie, zmieniające się wagi θ). Jej ruch w jednym kroku:

```
Δz_i = J_i Δθ,              J_i = ∂z_i/∂θ            (768 × P, P = 2 926 832 dla małego ViT)
Δθ   = −η Σ_{j∈paczka} J_j^T g_j,   g_j = ∂ℓ_j/∂z_j   (SGD; Adam dodaje skalowanie po współrzędnych i pęd)
⇒ Δz_i = −η Σ_{j∈paczka} K_ij g_j,  K_ij = J_i J_j^T  (blok 768 × 768 jądra tangensowego, NTK)
```

Czyli siła działająca na cząsteczkę i to suma „pchnięć” od obrazów j w paczce, każde przekazane przez K_ij.
To jest dokładny odpowiednik Σ_j F(r_i − r_j) w cieczy, tylko że zasięg i znak oddziaływania zadaje K_ij, a nie
odległość w przestrzeni. Trzy elementy modelu cząstek w cieczy mają więc swoje odpowiedniki:

| Ciecz / szkło koloidalne | Tokeny ViT | Skąd to wiemy |
|---|---|---|
| siła od sąsiadów (twarde kule) | K_ij g_j od obrazów w paczce | równanie wyżej |
| szum rozpuszczalnika, temperatura T | losowy skład paczki; T_eff ∝ η/B | Mandt i in. 2017 (SGD jako Langevin) |
| tarcie / bezwładność | pęd Adama β1 = 0,9 → pamięć prędkości ~10 kroków | VACF +0,73 po 1 kroku, reżim balistyczny α ≈ 1,6–1,8 dla τ ≤ 3 |
| gęstość (ułamek objętości φ) | liczba więzów na parametr N·wyjścia/P (zacieranie) | Franz–Parisi 2016, Spigler i in. 2019, Geiger i in. 2019 |
| klatka = sąsiedzi blokują ruch | frustracja: siły od różnych obrazów się znoszą | do pomiaru (p. 2B) |

Koloidy są przetłumione (brak reżimu balistycznego, Hunter i Weeks 2012), więc przez pęd Adama nasz układ jest bliższy
szkłu molekularnemu lub polimerowemu z symulacji MD (Bennemann i in. 1999) niż zawiesinie Weeksa. Trening zwykłym SGD bez
pędu (β1 = 0) powinien dać dynamikę przetłumioną jak w koloidzie: krótki test, kilkaset kroków, ten sam pomiar α(τ).

## 2. Cztery sposoby wprowadzenia „objętości”, od najtańszego

### A. Objętość efektywna z samych trajektorii (do aplikacji, bez treningu)

- Promień cząsteczki: `ℓ_i` = rms przesunięcia względem klatki (minus średnie przesunięcie 6 sąsiadów) w plateau
  τ = 35–80 kroków. To odpowiednik rozmiaru klatki `r_msd` z tabeli I Weeksa i Weitza (2002).
- Gęstość lokalna: `ℓ_i / d_nn,i` (odwrotne kryterium Lindemanna; w kryształach i szkłach ≈ 0,1–0,15, w cieczy > 0,2)
  albo liczba sąsiadów w promieniu ℓ_i. Kolorowanie tą wielkością pokazuje, gdzie chmura jest „zatłoczona” (szkło),
  a gdzie „rzadka” (ciecz).
- Frakcja upakowania: udział cząsteczek, których sfery klatki nachodzą na sfery najbliższych sąsiadów
  (`ℓ_i + ℓ_j > d_ij`). Rysować w PCA3 sfery o promieniu ℓ_i zamiast punktów.
- Ośrodek: pole kolektywne (średnie przesunięcie populacji, zdjęcia, obszaru) to „przepływ rozpuszczalnika”; reszta to
  ruch termiczny. Hierarchia pól jest już policzona (udział pola globalnego 0,28–0,47; cos sąsiadów 0,87).

### B. Zasięg oddziaływania z wag (mechanistycznie, mamy wszystko: checkpointy co 10 kroków i skład paczek)

Dla checkpointu k i paczki `PACZKI.npz[idx][k]`:
1. dla każdego obrazu j w paczce jeden backward: `v_j = J_j^T g_j` (gradient wag po tym obrazie);
2. dla każdego v_j jeden JVP przez sieć na zdjęciach sond: `Δz_i^(j) = J_i v_j` dla wszystkich 60 000 tokenów;
3. test równania ruchu: `Σ_j Δz_i^(j)` (ze skalowaniem Adama) ≈ zmierzone `Δz_i` z gęstego zapisu;
4. rozkład siły według źródeł: macierz `F_ij = |Δz_i^(j)|` (tokeny × obrazy paczki). Z niej:
   - zasięg oddziaływania = odległość w latencie, przy której `|K_ij|` spada do połowy (to jest „rozmiar cząstki”),
   - sąsiedzi mechanistyczni = obrazy/tokeny o dużym `|K_ij|`, zamiast najbliższych w latencie,
   - stopień wygaszania `|Σ_j F_ij| / Σ_j |F_ij|`: blisko 0 = klatka (siły się znoszą, drganie), blisko 1 = skok.
Koszt: 32 backward + 32 JVP na krok, kilkanaście kroków wystarczy (rozgrzewka, plateau, późne epoki).

### C. Gęstość jako liczba więzów na parametr (analogia zacierania)

| Bieg | N zdjęć | więzy ≈ N × (600 heat + 4 box) | P | więzów na parametr |
|---|---|---|---|---|
| K1000 | 1 500 | 0,9 M | 2,93 M | 0,31 (faza niezatarta, „ciecz”) |
| 35 000 | 35 000 | 21 M | 2,93 M | 7,2 (faza zatarta, „szkło”) |

Pokrętło gęstości już istnieje: liczba zdjęć N albo szerokość modelu P. Przewidywanie z teorii zacierania: większa
gęstość → dłuższe plateau MSD, wolniejsze Fs, silniejsze starzenie; mniejsza → szybsza ucieczka z klatki. Pierwszy test
bez nowego treningu: te same odciski (`odciski_klatkowe.py`) na historii K1000 i porównanie z biegiem 35 000.
Uwaga: cele heatmapy są silnie skorelowane, więc liczba więzów jest górnym oszacowaniem; rząd wielkości jest jednak jasny.

### D. Model efektywny: symulacja „szkła” dopasowana do pomiarów

Dynamika Langevina z pędem w PCA-k dla N cząsteczek:

```
m dv_i = −γ v_i dt − Σ_j ∇U(z_i − z_j) dt − ∇V_i(z_i) dt + sqrt(2 γ T(t)) dW_i
```

- `m`, `γ` z VACF (0,73 przy opóźnieniu 1, minimum −0,13 przy 5–6 i 12–13 kroków),
- `T(t)` z energii drgań: E1 ∝ t^−0,79 (kroki 300–2188), potem ∝ t^−0,30 (lr stały, więc to własne stygnięcie),
- zasięg i siła `U` z poziomu B albo przez dopasowanie sił (regresja Δz_i na względnych położeniach sąsiadów),
- `V_i` z gradientu straty głowy po tokenie (narzędzie/tło: różne znaki → odpychanie między klasami, przyciąganie w klasie).
Weryfikacja: ten sam zestaw odcisków (MSD, α2, Fs, c(τ), trwałość sąsiedztwa, μ starzenia) dla symulacji i pomiaru.
To jest dosłowne „przetłumaczenie ViT na szkło” i można je puścić w aplikacji obok prawdziwych trajektorii.

## 3. Czego nie da się przenieść (i jak to nazywać uczciwie)

- Brak zachowania energii: gradient flow jest dyssypatywny; „energia” to miara ruchu, nie hamiltonian.
- Temperatura nie jest zadana termostatem: szum gradientu maleje razem ze stratą, więc układ stygnie sam. Najbliższy
  odpowiednik fizyczny to gaz granularny (chłodzenie Haffa), nie kąpiel cieplna. Dlatego u nas krótkoczasowe drgania
  maleją z wiekiem (MSD(τ=1) spada 4× między krokiem 300 a 2000), a w szkle koloidalnym ruch w klatce nie zależy od wieku
  próbki, wydłuża się tylko plateau (Hunter i Weeks 2012, rys. 15).
- Wymiar 32/768 zamiast 3: liczba sąsiadów nie jest ograniczona geometrią, więc „ułamek objętości” trzeba zastępować
  miarami A lub C.
- Tokeny jednego zdjęcia są związane ciągłością obrazu (76 % sąsiadów w latencie to sąsiedzi na siatce patchy):
  to raczej stop polimerowy (łańcuchy) niż zawiesina niezależnych kul. Trwałość sąsiedztwa 57 % po 22 000 kroków
  jest po części strukturalna, nie dynamiczna.

## 4. Literatura (zweryfikowana w Undermind)

Jacot i in. 2018 (NTK) arXiv:1806.07572 · Mandt, Hoffman, Blei 2017 arXiv:1704.04289 · Franz, Parisi 2016
doi:10.1088/1751-8113/49/14/145001 · Spigler i in. 2019 doi:10.1088/1751-8121/ab4c8b · Geiger i in. 2019
doi:10.1103/PhysRevE.100.012115 · Baity-Jesi i in. 2019 doi:10.1088/1742-5468/ab3281 · Kerr Winter, Janssen 2025
doi:10.1103/physrevresearch.7.023010 · Weeks, Weitz 2002 doi:10.1103/PhysRevLett.89.095704 i
doi:10.1016/S0301-0104(02)00667-5 · Hunter, Weeks 2012 doi:10.1088/0034-4885/75/6/066501 · Vivek i in. 2017
doi:10.1073/pnas.1607226113 (przesunięcia względem klatki) · Chaudhuri, Berthier, Kob 2007
doi:10.1103/PhysRevLett.99.060604 (wykładnicze ogony) · Bennemann i in. 1999 doi:10.1016/S1089-3156(99)00008-2.
