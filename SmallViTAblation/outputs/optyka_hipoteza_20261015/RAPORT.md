# Próba analogii światła na zapisanych checkpointach SmallViT

Data: 15 października 2026. Wyniki nowych obliczeń na istniejącym biegu s47. Aplikacja i zapisane wagi nie były modyfikowane; nie uruchamiano treningu ViT.

**Wniosek:** opis superpozycji pozwala mierzyć rzeczywiste wzmacnianie i kompensację wkładów neuronów. Nie uzyskano jednak stabilnej przewagi prognostycznej cech „optycznych” nad kontrolami. Pamięć dwóch poprzednich zmian pomaga prognozować wyjście, lecz zapis tłumionego oscylatora nie ma wyraźnej przewagi nad zwykłym AR(2). Nie jest to dowód fazy optycznej ani mechanizmu światła w modelu.

## 1. Co faktycznie policzono

- 806 checkpointów: kroki 0–65 640; trzy różne obrazy z istniejących historii aplikacji, łącznie 2418 przejść obraz–checkpoint.
- 24 stałe tokeny na obraz, razem 58 032 obserwacje token–checkpoint. Po 8 przy środku anotacji, 8 przy brzegu, 8 w tle. To grupy geometryczne, nie maski segmentacji ani wyłącznie dodatnie komórki straty.
- Pełny forward na RTX 3070, float32, eval. Odtworzono także dokładne pochodne końcowej głowy. Wagi i bufory BatchNorm z każdego checkpointu wczytywane strict=True.
- Analiza dotyczy ostatniego MLP: z ma 768 współczynników, a u_c=z_c v_c działa w 192-wymiarowym strumieniu residual. Ruch między checkpointami mierzono w tym rzeczywistym strumieniu; nie w rzutowanym filmie 3D.
- Dla prognoz użyto 646 regularnych zapisów od 1100 do 65 600 co 100 aktualizacji. Dodatkowych końców epok nie interpolowano ani nie mieszano z odstępem 100.
- Obraz 3 nie uczestniczył w dopasowaniu ani wyborze hiperparametrów modeli prognostycznych. Nie ustalano, czy pochodzi z innego filmu. Wszystkie trzy historie dotyczą tego samego treningu s47, więc nie są trzema niezależnymi biegami.

![Obrazy i wybór tokenów](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/obrazy_i_tokeny.png)

## 2. Rzeczywista superpozycja i „poziom bez współdziałania”

\[
U=\sum_c u_c,\qquad P_{\rm diag}=\sum_c\|u_c\|^2,\qquad
P_{\rm net}=\|U\|^2,\qquad
\eta=P_{\rm net}/P_{\rm diag}.
\]
\[
P_{\rm net}-P_{\rm diag}=2\sum_{c<d}u_c^\top u_d.
\]

η>1 oznacza dodatnią sumę wyrazów mieszanych, η<1 — ujemną. P_diag jest również oczekiwaną normą kwadratową sumy przy hipotetycznych niezależnych, jednostajnie losowanych fazach w zespolonym rozszerzeniu tych wektorów. Nie oznacza to, że takie fazy istnieją w sieci. Wielkości nie są mocą w watach ani energią. Bias MLP wyłączono z sumy neuronów i zachowano osobno przy odtwarzaniu rzeczywistego kroku.

Nie mylić η z q=||sum u||/sum ||u||: q≤1 również wtedy, gdy η>1. Mały stosunek do sumy długości nie dowodzi wygaszania względem odniesienia P_diag.

| Obraz | Mediana η przy środku, inicjalizacja | Mediana η przy środku, kroki ≥55 000 |
|---|---:|---:|
| 1 | 1.066 | 2.708 |
| 2 | 1.049 | 2.785 |
| 3 | 1.048 | 2.047 |

W późnym oknie wypadkowa norma kwadratowa jest więc przeciętnie większa od sumy norm kwadratowych pojedynczych wkładów. Wzrost nie jest monotoniczny: krzywe najpierw rosną, później częściowo maleją. To pomiar organizacji wkładów; bez interwencji nie dowodzi, że ona powoduje lepszą detekcję.

Dodatkowo policzono odpowiednik podatności: J głowy dla logitu i czterech krawędzi znormalizowanych do wymiarów obrazu. Te same ilorazy zastosowano do J u_c. Wynik zależy od kierunku odczytu: znoszenie dla logitu i dla geometrii ramki nie musi być takie samo. Jest to liniowa odpowiedź przy rzeczywistym stanie po MLP, nie efekt skończonego usunięcia neuronu.

## 3. Czy te cechy pomagają przewidzieć następny checkpoint?

Dopasowanie obejmuje wyłącznie obrazy 1–2 i cele przed krokiem 33 400; walidacja zaczyna się od 33 400, test od 43 000. Jednokrokowy horyzont wynosi 100 aktualizacji. W teście za każdym razem dostępny jest rzeczywisty obecny stan; nie jest to swobodna symulacja całej przyszłej trajektorii.

Liczby wierszy: dopasowanie 15408, walidacja 4560, test 16272. W części testowej jest 226 momentów czasu na obraz. PCA, standaryzacje i parametry dopasowano bez obrazu 3 i bez przyszłych checkpointów.

Baseline ma 41 cech: czas, normy stanu/aktywacji/wkładów, P_net i P_diag, obecne wyjście, poprzednią zmianę, zgodność wcześniejszego ruchu z U, położenie tokenu, grupę oraz po 8 składowych PCA stanu i U. Dodatek ma 9 cech: ilorazy superpozycji, efektywną liczbę kanałów i miary odpowiedzi głowy. Zysk samego ilorazu η byłby efektem użytecznej parametryzacji, ponieważ jego składniki są już w baseline.

Porównano ridge (alpha wybierane na walidacji) i nieliniowy histogram gradient boosting o ustalonej konfiguracji. Kontrola tasuje dodatkowe cechy między tokenami wewnątrz tego samego obrazu, czasu i grupy, zachowując ogólny trend. Trening tych małych modeli analitycznych nie zmienia ViT.

**Trzeci obraz, późniejsze checkpointy. Dodatnie wartości to zmniejszenie MSE; ujemne to pogorszenie.**

| Cel | Ridge + cechy optyczne | Nieliniowy + cechy optyczne | Przedział 95% dla nieliniowego | Nieliniowy + tasowane cechy |
|---|---:|---:|---:|---:|
| Długość ruchu latentu, 24 tokeny | +4.54% | -4.52% | [-6.95; -2.18]% | -0.85% |
| Zmiana logitu tokenu środkowego | -14.31% | -5.18% | [-7.45; -3.14]% | -9.19% |
| Zmiana czterech krawędzi tokenu środkowego | -2.87% | +2.90% | [+1.72; +4.22]% | +0.57% |

Dla ramki trzeciego obrazu RMSE prognozy krawędzi spadło z 9,59 do 9,45 piksela. Jednak na obrazach 1 i 2 ten sam dodatek pogorszył prognozę ramki odpowiednio o 4,82% i 2,22% MSE. Nie ma stabilnego zysku między obrazami i rodzinami modeli. Cechy mogą być użyteczne lokalnie, ale wynik nie uzasadnia ogólnej tezy o lepszym modelu dynamiki.

**Te liczby dotyczą prognozowania zmian wyjścia zamrożonych checkpointów. Nie są poprawą IoU, mAP ani jakości oryginalnego detektora.** Bbox jest odczytem ustalonego tokenu najbliższego środkowi anotacji: 226, 317 i 369. Nie jest to końcowy wybór detekcji po argmax/NMS.

![Superpozycja i test prognozy](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/superpozycja_i_prognoza.png)

## 4. Próba fazy: oscylator kontra zwykła pamięć

Dla zmian odczytu d_n=y_n−y_(n−1) dopasowano:

\[
d_{n+1}=c+a d_n+b d_{n-1}.
\]

W modelu oscylacyjnym narzucono a=2r cos(ω), b=−r², 0<r<1. Zwykły AR(2) ma swobodne a,b. Parametry są wspólne dla dwóch obrazów dopasowania, osobne dla pięciu odczytów i trzech grup tokenów: 15 modeli. Stabilny oscylator przeszukuje z góry zadaną siatkę tłumienia i okresów; nie używa transformaty Hilberta ani przyszłości do wyznaczania fazy.


| Trzeci obraz | Zysk oscylatora względem AR(1) | Zysk względem AR(2) | Przedział 95% względem AR(2) |
|---|---:|---:|---:|
| logit | +12.53% | +0.38% | [-1.38; +1.98]% |
| box | +8.32% | +0.96% | [-0.83; +2.99]% |

Pamięć dwóch zmian daje więc rzeczywisty zysk względem krótszej pamięci w tej prognozie. Specyficzny zapis oscylacyjny daje względem zwykłego AR(2) mniej niż 1% na trzecim obrazie, a wszystkie sześć przedziałów dla porównania oscylator–AR(2), na trzech obrazach i dwóch odczytach, obejmują zero.

**Kontrola pozornej fazy.** Wszystkie 15 modeli AR(2) zmian miały zespolone bieguny, odpowiadające okresom około 271–309 kroków. Jednak w 200/200 symulacji całkowicie niezależnego szumu położeń jego różnice również dały zespolone bieguny. Powód można wyprowadzić bez fizyki światła: dla y_n=ε_n z niezależnym szumem różnice mają ujemną autokorelację dla opóźnienia jeden. Najlepszy liniowy AR(2) ma wtedy a=−2/3, b=−1/3, co daje pozorny okres około 287 kroków przy próbkowaniu co 100. Sama obecność zespolonych biegunów nie identyfikuje oscylatora fizycznego.

Kontrola na surowych odczytach, bez różnicowania, dała 0/15 modeli z zespolonymi biegunami. Wymuszenie oscylatora pogorszyło prognozę względem AR(2). Modele poziomów miały także problem przeniesienia średniego bbox między obrazami; dlatego ich porażka nie dowodzi braku jakichkolwiek oscylacji w sieci. Wniosek dotyczy braku wystarczających dowodów na fazę w tej konkretnej próbie.

![Porównanie przy tej samej pamięci](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/faza_vs_taka_sama_pamiec.png)

## 5. Dosłowny zapis „amplituda + faza” pojedynczego lasera

Wykonano dodatkową próbę reprezentacji. W bazie dwóch kierunków e1,e2 przypisano każdemu neuronowi:

\[
E_c=(u_c\cdot e_1)+i(u_c\cdot e_2),\quad A_c=|E_c|,\quad \phi_c=\arg E_c.
\]

Sumowanie zespolone dokładnie odtwarza rzut sumy U na tę płaszczyznę. Baza została wyznaczona na kroku 33 300 z obrazów 1–2 i zamrożona; nie obracano jej w celu dopasowania przyszłych wyników. Kierunki wybrano z macierzy drugich momentów neuronowych wkładów. Nie jest to faza elektromagnetyczna, lecz kąt w określonym układzie współrzędnych.

Rekonstruowano ostatni przyrost MLP za pomocą 2, 8, 32, 96 i 192 kierunków, pozostawiając pełny stan wejściowy, bias i rzeczywistą głowę. Pomiar obejmuje 18 wybranych checkpointów, w tym 13 od kroku 43 000. To kontrolowana rekonstrukcja odczytu, nie retrening ani zmiana zapisanej sieci.

| Liczba rzeczywistych kierunków | Mediana błędu względnego U, środek obrazu 3, późne zapisy | Mediana zniekształcenia krawędzi [px] |
|---|---:|---:|
| 2 | 91.71% | 1.786 |
| 8 | 80.23% | 1.676 |
| 32 | 53.55% | 0.581 |
| 96 | 23.15% | 0.188 |
| 192 | 0.00% | 0.000 |

Pojedyncza zespolona amplituda na neuron w tej wspólnej płaszczyźnie nie wystarcza do odtworzenia pełnego ruchu. Nie dowodzi to niemożliwości każdego modelu optycznego: wielomodowe pole może mieć więcej stopni swobody. Dokładny zapis wszystkich 192 współrzędnych jako 96 liczb zespolonych jest możliwy, lecz sam w sobie tylko zmienia notację i nie dodaje mocy wyjaśniającej.

Wyraźnie mniejszy błąd odczytu ramki niż błąd wektora U pokazuje, dlaczego trzeba oddzielać ruch reprezentacji od skutku dla zadania. Nie można oceniać poprawności całego modelu ruchu wyłącznie po względnie podobnej ramce.

## 6. Weryfikacja rachunku i ograniczenia

- Maksymalny względny błąd odtworzenia rzeczywistego przyrostu MLP w sprawdzonych checkpointach: 3.92e-07.
- Błąd tożsamości wyrazów mieszanych, kontrola float64: maksymalnie 1.25e-15.
- Niezależna kontrola pochodnej głowy centralnymi różnicami, float64: maksymalnie 8e-09.
- Epsilon końcowego LayerNorm: 1e-05; epsilon LayerNorm MLP: 1e-06. Użyto modułów modelu, a nie historycznej ręcznej kopii głowy z innym epsilon.
- Rozbieżność aktywacji względem wcześniejszego cache float16 wyniosła do około 0,025% RMS. Nowe miary policzono ze świeżego forward float32, a nie z zaokrąglonego cache.
- Nie znaleziono NaN/Inf. Wybory tokenów są stałe; obrazy mają różne hashe. Maski dopasowania, walidacji i testu są rozłączne.
- Przedziały pochodzą z 1000 replik bootstrapu bloków po 20 checkpointów. Najpierw uśredniano błąd w obrębie checkpointu. Nie traktowano 24 tokenów jako niezależnych powtórzeń. Przedziały są warunkowe dla tych trajektorii, eksploracyjne, bez korekty wielu porównań i bez niepewności między seedami.
- Jakość prognozy mierzono dla konkretnego próbkowania co 100 kroków. Ruch szybszy od tego zegara nie jest rozdzielony. Test nie obejmuje wszystkich możliwych zjawisk falowych ani wszystkich warstw ViT.
- Nie dopisano dowolnych długości fali, polaryzowalności, równań Maxwella, temperatury ani faz uczonych przez nowy ViT. Takie zmiany wymagałyby osobnego modelu i eksperymentu.

## 7. Co ten wynik daje dla mikroskopu

Warto traktować η i jego odpowiedniki dla głowy jako diagnostykę współdziałania wkładów: pozwalają odróżnić duże indywidualne działania od ich wypadkowego skutku. Jest również uzasadnienie dla opisu pamięci i naprzemiennych korekt podczas uczenia. Na tej próbie nie ma podstaw, aby przypisać neuronom fazę lasera jako dodatkową przyczynę ruchu lub uznać model falowy za lepszy ogólny predyktor.

Najciekawszy dalszy kierunek publikacyjny pozostaje związany z funkcją: czy grupy neuronów wskazane przez te miary odpowiadają za konkretne korekty ramki w kontrolowanych interwencjach i na niezależnych biegach. Obecne wyniki są pilotażem i kontrolą analogii, nie potwierdzeniem takiego mechanizmu.

## 8. Pliki i odtwarzanie

[Protokół przed obliczeniami](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/optyka_hipoteza_20261015/PROTOKOL.md). Rozwinięcie kontroli surowych poziomów i dosłownego zapisu zespolonego wykonano po pierwszym wyniku, aby sprawdzić alternatywne wyjaśnienia; nie używano tego do dostrajania głównego testu prognostycznego.
- [Pełne pomiary float32](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/pomiar.npz).
- [Kontrole ekstrakcji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/ekstrakcja.json).
- [Modele i wyniki prognostyczne](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/prognozy.json).
- [Cele, maski i wszystkie prognozy](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/prognozy.npz).
- [Modele zmian i kontrola szumu](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/faza.json).
- [Kontrola surowych poziomów](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/faza_poziomy.json).
- [Amplitudy, fazy i błędy rekonstrukcji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/zapis_zespolony.json).
- [Pochodzenie danych i hashe](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/optyka_hipoteza_20261015/MANIFEST.json).

Polecenia z katalogu SmallViTAblation, w istniejącym środowisku Python z PyTorch/timm/NumPy/SciPy/scikit-learn/Matplotlib:
```powershell
python -X utf8 experiments/optyka_hipoteza_20261015/ekstrakcja.py
python -X utf8 experiments/optyka_hipoteza_20261015/analiza.py
python -X utf8 experiments/optyka_hipoteza_20261015/zapis_zespolony.py
python -X utf8 experiments/optyka_hipoteza_20261015/raport.py
```

Odtwarzanie zapisuje pliki wyłącznie w katalogu wyników tej próby. Dla nowego wariantu należy nadać oddzielny katalog, aby zachować ten wynik.

Punkt odniesienia literaturowy: [raport wzorów i pomiarów](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_wzory_pulapki_optyczne_20261015/ATLAS_WZOROW_I_POMIAROW.md). Wyniki tej próby są własnymi obliczeniami, a nie wnioskami zaczerpniętymi z tamtych artykułów.
