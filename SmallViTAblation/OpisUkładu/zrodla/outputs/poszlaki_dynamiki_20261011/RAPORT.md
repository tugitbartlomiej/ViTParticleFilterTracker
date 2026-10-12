# Nowe poszlaki: uspokojenie kroków i utrzymująca się przebudowa reprezentacji

Stan analizy: 11 października 2026. Przejrzano nowe wyniki, kod pomiarów i literaturę przez Undermind. Niezależnie przeliczono porównania z zapisanych JSON-ów oraz wykonano nowy test na pełnych aktywacjach 768D i checkpointach istniejącego treningu 60 epok. Nie uruchamiano ponownie treningów.

**Wniosek:** materiał jest znacznie mocniejszy niż pojedyncza analogia do szkła. Najlepsza poszlaka mechanistyczna to wspólne aktualizacje wag, które wywołują różny, przestrzennie uporządkowany ruch reprezentacji. Nowe wyniki potwierdzają rozdzielenie niektórych miar uspokojenia i przebudowy. Nie potwierdzają jeszcze przejścia szklistego, przyczynowej roli rodzaju nadzoru ani przewidywania przyszłej dynamiki.

## 1. Co rzeczywiście jest nowe

W `outputs/porownanie_szklo_20261010/eden/wyniki/pokretla` są 67 plików ANALIZA, obejmujących 49 nazw biegów. To nie jest 67 niezależnych replik: część plików opisuje różne okna i warstwy tego samego treningu; biegi współdzielą również ziarna i protokoły. Doszły trzy ziarna dla wybranych learning rates, CNN, SGD z momentum, zmiany batch size, sześć warstw, klasyfikacja Imagenette, długie okna oraz interwencje na checkpointach.

Większość nowych szczegółowych trajektorii jest w stałym losowym rzucie **RP128** reprezentacji GELU768. Osobno zapisano normy kroków w 768D. To istotna poprawa względem PCA32 dopasowanego na początku uczenia, ale nie pełny zapis 768 współrzędnych dla wszystkich nowych warunków. Nowy test z punktu 4 używa pełnych FP32 aktywacji ośmiu pierwotnych obrazów.

## 2. Najważniejsza korekta: mniejszy wskaźnik względny nie oznacza mniejszego ruchu

Porównałem wczesne i późne okna **tego samego biegu**, zawsze po rozgrzewce do kroku 300. Detekcja: kroki 300–2188 oraz 12034–13128. Klasyfikacja: 300–2188 oraz 10746–11840. Oba biegi: ViT, AdamW, learning rate 0.0005.

„Lokalne MSD” poniżej oznacza średni kwadrat przemieszczenia po odjęciu średniego przemieszczenia sześciu sąsiadów wybranych na początku danego okna (S1), w RP128. To inna miara niż samo usunięcie przesunięcia środka obrazu.

| Pomiar | Detekcja: wcześnie → późno | Klasyfikacja: wcześnie → późno |
|---|---:|---:|
| Średnia energia pojedynczego kroku w 768D, z zapisanej próbki co 10 kroków | 0.29674 → 0.15720 (−47.0%) | 0.69452 → 0.51189 (−26.3%) |
| Lokalne MSD, odstęp 1 krok | 0.02788 → 0.02576 (−7.6%) | 0.06986 → 0.14688 (+110.2%) |
| Lokalne MSD, odstęp 10 kroków | 0.64161 → 0.75689 (+18.0%) | 1.48932 → 3.77111 (+153.2%) |
| Lokalne MSD, odstęp 105 kroków | 1.51584 → 2.76329 (+82.3%) | 2.68319 → 8.63776 (+221.9%) |
| Skala sąsiedztwa `d_shell` | 2.31081 → 3.25843 (+41.0%) | 2.07933 → 2.58829 (+24.5%) |
| Wskaźnik nazwany Lindemannem | 0.49041 → 0.44372 | 0.73209 → 1.04761 |
| Próg relaksacji `tau_1e` z zapisanej siatki | 176 → 136 | 105 → 17 |
| Jakość przy końcu okna lub ostatnim pomiarze przed nim | mAP50: 0.4571 → 0.6151 | top-1: 0.7575 → 0.7837 |

W detekcji licznik wskaźnika Lindemanna również rośnie: RMS ruchu dla odstępów 35–80 kroków wynosi 1.13325 → 1.44582. Mianownik rośnie szybciej, więc iloraz maleje. Interpretacja „detekcja lokalnie stygnie” na podstawie samego ilorazu jest nieuzasadniona. W tym konkretnym porównaniu `tau_1e` również się nie wydłuża.

W klasyfikacji maleje energia całego pojedynczego kroku, a lokalny ruch rośnie już przy odstępie jednego kroku. To szczególnie czytelna poszlaka rozdzielenia skal ruchu. W detekcji kierunek zmiany zależy od odstępu czasowego: ruch jednokrokowy nieco maleje, lecz skumulowane lokalne przemieszczenie na średnich odstępach rośnie. Może to wiązać się zarówno ze skalą ruchu, jak i jego korelacjami w czasie.

Ograniczenia: okna mają różną długość, a sąsiedztwa są ponownie wyznaczane na początku każdego z nich. Wyniki są opisowe, bez przedziałów ufności i bez ponownego przeliczenia surowych trajektorii w identycznych oknach i na wspólnym grafie. Energia kroku nie jest tożsama z energią dryfu środka populacji. Nie używam jej do twierdzenia, że wykazano spadek każdej składowej ruchu zbiorowego.

## 3. Najmocniejszy trop mechanistyczny: odpowiedź na wspólne zmiany wag

Nowe eksperymenty `jadro_pola.py` przywracają checkpoint i stan AdamW, wykonują pojedyncze aktualizacje dla 40 paczek, a następnie mierzą ruch tokenów. To interwencja na przyczynie ruchu. Porównania obejmują także 100 losowych kierunków zmian wag.

| Checkpoint | Spearman mapy ruchliwości: reszta po odpowiedzi na średni gradient | Izotropowe zmiany wag | Pełna odpowiedź wag na paczki |
|---|---:|---:|---:|
| LR 0.000125, epoka 2 | 0.9565 | 0.8010 | 0.8220 |
| LR 0.0005, epoka 2 | 0.8493 | 0.6143 | 0.7224 |
| LR 0.0005, epoka 12 | 0.6867 | 0.4618 | 0.3401 |

Mapy są normalizowane średnią ruchliwością każdego obrazu, a Spearman liczony łącznie po tokenach. Wysokie zgodności wskazują, że wrażliwość reprezentacji i struktura aktualizacji wag wyjaśniają istotną część tego, *które tokeny są bardziej ruchliwe*. Sama zgodność map nie podaje procentu wyjaśnionej wariancji dynamiki.

Naturalny model roboczy:

`Δz_i ≈ J_i Δθ`, a dla wycentrowanych aktualizacji `Cov(Δz_i, Δz_j) ≈ J_i Σ_θ J_jᵀ`.

Ta zależność wynika z liniowego przybliżenia sieci. Wspólne wagi mogą powodować skorelowany ruch cząsteczek bez postulowania sił między nimi w przestrzeni latentu. Nowe dane są zgodne z takim mechanizmem, ale kod mierzy skończone perturbacje, nie wylicza jawnie tej kowariancji ani błędu linearyzacji.

Co ogranicza siłę wyniku:

- Checkpoint z końca okna porównuje się z **poprzednimi** 200 krokami. To dopasowanie retrospektywne, jeszcze nie prognoza przyszłości.
- „Szum” to odpowiedź na paczki minus odpowiedź na średni gradient. AdamW jest nieliniowy względem bieżącego gradientu, więc nie jest to dokładnie odejmowanie empirycznej średniej odpowiedzi.
- Odpowiedzi powstają na niezależnych resetach tego samego checkpointu. Ich sumowanie w pseudotrajektorie nie odtwarza czasowego przebiegu uczenia.
- Przestrzenne korelacje amplitud są liczone z innym filtrem czasowym dla perturbacji i dla treningu (`W=0` versus `W=5`).
- Normy przewidywane są w 768D, obserwowane mapy w RP128. W tabeli zbiorczej pomiar `E_128` bywa prezentowany w tej samej kolumnie co `E_768`.

Energia odpowiedzi wynikającej wyłącznie ze zmiany statystyk BatchNorm wynosi 0.86–2.72% energii odpowiedzi na paczki w trzech checkpointach. To stosunek energii dwóch interwencji, nie rozłączny udział przyczynowy. Wspiera mały wpływ BN na te pojedyncze kroki. Nie unieważnia wcześniejszej obserwacji dużego wpływu BN na porównania oddalone o wiele epok.

## 4. Nowy test pełnych 768 wymiarów: czy ruch chowa się w kierunkach odrzucanych przez sieć?

Zmierzona reprezentacja to 768 aktywacji GELU ostatniego MLP. Następne `fc2` mapuje je do 192 wymiarów. W sprawdzonych checkpointach rząd macierzy wynosi 192; jej jądro ma zatem 576 wymiarów. Perturbacja aktywacji należąca do tego jądra nie zmienia wyniku `fc2`, przy ustalonych wagach i pozostałym wejściu bloku.

Na istniejących aktywacjach FP32 ośmiu obrazów policzyłem ortogonalny podział prawdziwego przemieszczenia na jądro i przestrzeń wierszy `fc2`. Usunięto średnie przemieszczenie tokenów każdego obrazu. Sprawdzono obie macierze z końców każdej pary oraz aktywacje z zamrożonym BN.

| Para kroków | Energia w jądrze `fc2` z początku pary, zwykłe FP32 | Ta sama miara, zamrożony BN |
|---|---:|---:|
| 300 → 310 | 67.43% | 67.67% |
| 990 → 1000 | 58.88% | 60.81% |
| 1094 → 2188 | 64.61% | 63.58% |
| 5470 → 10940, epoki 5 → 10 | 59.73% | 62.32% |
| 27350 → 32820, epoki 25 → 30 | 60.55% | 65.64% |
| 54700 → 60170, epoki 50 → 55 | 61.98% | 66.57% |

**To nie potwierdza prostego wyjaśnienia „ruch odbywa się wyłącznie w odrzucanych kierunkach”.** Około 33–41% energii sprawdzonych przemieszczeń trafia do przestrzeni przekazywanej przez `fc2`. Dla izotropowego ruchu udział w 576-wymiarowym jądrze wynosiłby średnio 75%, a tu wszystkie powyższe wartości są mniejsze. Wynik zachowuje ten kierunek dla macierzy z końca pary i zamrożonego BN. Izotropia jest tylko punktem odniesienia geometrycznego, nie dopasowanym modelem zerowym treningu.

To nadal nie dowodzi zmiany decyzji sieci. Dalsza normalizacja i głowa mogą odrzucać kolejne kierunki; podczas treningu zmieniają się też same wagi i gałąź residual. Test dotyczy warunkowej odpowiedzi jednej projekcji. Nie wolno z niego wnioskować, że pozostałe 33–41% ruchu jest w całości potrzebne zadaniu. Błąd numeryczny zerowania odpowiedzi w jądrze był poniżej 2.1e-15 względnej normy.

## 5. Co już się powtarza i co wymaga ostrożniejszego opisu

Przy LR 0.0005 i trzech ziarnach 47, 48, 49 korelacja amplitud lokalnego ruchu dla stałego odstępu 10 kroków wynosi 0.522, 0.510, 0.528. Względny ranking ruchliwości między pierwszym a ostatnim podoknem wynosi około 0.823–0.829. To dobra replikacja uporządkowania, z zastrzeżeniem zależności tokenów tego samego obrazu.

Efekt występuje także w CNN i SGD z momentum. Dla stałego odstępu 10 kroków korelacje wynoszą odpowiednio 0.626–0.701 oraz 0.376–0.570 w zbadanych ustawieniach. Nie jest więc specyficzny wyłącznie dla attention ani dla AdamW. Eksperyment CNN zmienia również pozycjonowanie i liczbę parametrów, więc nie izoluje roli attention w wielkości pola obrazu. SGD ma momentum 0.9: nadal brakuje rozstrzygającego testu bez pamięci optymalizatora.

W sześciu warstwach korelacja między obrazami dla odstępu 10 kroków maleje od około 0.255 do 0.108. Jednocześnie względny wskaźnik ruchliwości rośnie, ale bezwzględne MSD nie rośnie monotonicznie. To następny argument za analizą osobno ruchu i skali geometrii. Pomiar obejmuje cztery obrazy i jeden bieg.

Klasyfikacja i detekcja różnią się zbiorem danych, stratą, głową, rozdzielczością, liczbą tokenów i historią uczenia. Wyniki uzasadniają hipotezę o roli nadzoru; nie stanowią jeszcze jej przyczynowego testu. Zerowa suma zmian końcowych cech przed global average pooling nie zmienia ich średniej. Nie wolno automatycznie przenosić tej niezmienniczości na wcześniejsze GELU768, ponieważ po nim występuje m.in. LayerNorm.

## 6. Dwie korekty metryk przed artykułem

**Energia:** składnik „średnia sąsiadów” i reszta nie są ortogonalne. Zapisane cztery udziały sumują się w analizowanych wynikach do 94.28–112.51%. Wymagany jest człon mieszany `2〈l,r〉`, równy po normalizacji `1 − suma udziałów`. Nie należy mówić o rozłącznym procentowym podziale na cztery rodzaje energii ani automatycznie renormalizować tych liczb do 100%. Oddzielnie trzeba zapisać średnie energie składników; iloczyn średniej energii i średniego udziału nie musi być ich średnią energią.

**Szkło:** wielkość nazwana `l_klatka` jest RMS dla z góry wybranych odstępów 35–80 kroków. Nie sprawdza automatycznie, czy w tym zakresie wystąpiło plateau. `tau_1e` zależy od skali `d_shell`, która zmienia się między oknami, i wybiera pierwszy punkt z logarytmicznej siatki. Korelacja `C1` dotyczy amplitud ruchu po normalizacji, a nie zgodności kierunków wektorów. Te definicje są użyteczne, lecz nie są same w sobie dowodem fizycznego szkła, mechanicznego sprzężenia czy temperatury termodynamicznej.

## 7. Literatura i rzeczywista szansa na wkład publikacyjny

Undermind znalazł bardzo bliską pracę: Farhad Pashakhanloo, **Contribution of task-irrelevant stimuli to drift of neural representations**, NeurIPS 2025. Autor pokazuje, że składniki wejścia nieistotne dla zadania mogą poprzez szum uczenia wywoływać dryf reprezentacji istotnych dla zadania. Badane są modele prostsze od ViT; praca nie dowodzi mechanizmu w naszych tokenach ani roli gęstego nadzoru. Samo „wynik stabilny, reprezentacja się zmienia” nie wystarczy jako nowość. [Pas25](https://arxiv.org/abs/2510.21588), [oficjalne proceedings NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/vol38-main-conference).

Aray Karjauv, **Rethinking Global Average Pooling: Your Classifier Is Secretly a Multi-Instance Learner** (2026), opisuje przestrzenny odczyt klasyfikatora i tożsamość uśredniania logitów przy liniowej głowie. Nie bada dryfu podczas uczenia ani kontrolowanego wpływu nadzoru gęstego. To wsparcie dla precyzyjnego sformułowania hipotezy, nie gotowe potwierdzenie. [Kar26h](https://arxiv.org/abs/2606.14555).

Potencjalny wkład tej pracy: **ilościowe przewidywanie przestrzennej i czasowej przebudowy tokenów z wrażliwości sieci, statystyki aktualizacji i ograniczeń zadania, wraz z wykazaniem konsekwencji dla działania modelu**. Nowe testy checkpointów zbliżają się do pierwszej części. Obecne dane nie rozstrzygają jeszcze przewidywania poza dopasowanym oknem ani korzyści funkcjonalnej.

Najbardziej rozstrzygające następne doświadczenia:

1. W checkpointcie oszacować wycentrowaną kowariancję rzeczywistych aktualizacji, sprawdzić zakres linearyzacji, a następnie przewidzieć ruchliwość i korelacje w **kolejnych** 200–1000 krokach. Ten sam filtr, projekcja, graf i odstępy; niezależne ziarna i obrazy. Porównania: izotropowe perturbacje oraz wcześniejsza mapa ruchliwości jako prosta prognoza.
2. Na tych samych obrazach i tym samym backbone porównać gęsty nadzór z nadzorem po uśrednianiu, z kontrolą straty, historii uczenia i skali aktualizacji. Mierzyć pełną odpowiedź od GELU do predykcji, nie samą średnią tokenów.
3. Selektywnie ograniczyć przewidywaną składową ruchu i zmierzyć błąd, generalizację lub odporność przy kontrolowanej długości aktualizacji. To rozdzieli ruch obojętny, pomocny i szkodliwy.

## 8. Pliki i odtwarzalność

- `inventory.json`: 67 analiz z identyfikacją biegów, okien i projekcji; brakujące starsze metadane pozostają jawnie niepełne.
- `same_run_comparisons.json`: porównania identycznych biegów, bez łączenia różnych treningów.
- `kernel_audit.json`: trzy poprawione testy checkpointów, bez historycznych wariantów `_bezBN` i `_tf32`.
- `nonorthogonal_energy_audit.json`: sumy udziałów i brakujące człony mieszane.
- `readout_nullspace.json`: 48 konfiguracji pomiaru: sześć par, dwa checkpointy odniesienia, dwa ustawienia BN i dwa sposoby centrowania.
- `sources.json`: ścieżki i SHA256 70 źródłowych plików wyników.
- `poszlaki.png`, `poszlaki.pdf`: wykresy porównań.

Kod: `experiments/poszlaki_dynamiki_20261011/analyze_evidence.py` i `readout_nullspace.py`. Raport nie zmienia historycznych README ani źródłowych wyników. Nowe wartości pochodzą z niezależnej agregacji istniejących podsumowań oraz rzeczywiście wykonanego testu aktywacji 768D; nie przedstawiono tego jako ponownego przeliczenia wszystkich surowych trajektorii.
