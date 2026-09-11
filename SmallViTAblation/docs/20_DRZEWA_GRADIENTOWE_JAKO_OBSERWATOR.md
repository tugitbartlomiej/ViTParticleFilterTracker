# 20 — Czy drzewo gradientowe (LightGBM) może być „obserwatorem" ważności danych obrazowych

Data: 2026-09-11. Powód: pytanie, czy LightGBM da się użyć jako tani sędzia, który
ocenia, które obrazy z puli warto podać sieci neuronowej.

Podstawa: deep search Undermind z 2026-09-11 (171 prac, workspace ViT), sześć prac
przeczytanych w całości ([Sch25], [Kat18], [Kil21], [Oka23], [Jia21], [Jia23]) plus dwie
przeczytane pod kątem audytu ([Kuz24], [An23]) i dokumentacja LightGBM.

---

## 1. Najpierw sprostowanie pojęć

Termin, którego brakowało w pytaniu, to najpewniej **funkcje bazowe**, nie „naturalne".
Drzewo w gradient boostingu jest funkcją **schodkową**: dzieli przestrzeń cech na prostokątne
kawałki i na każdym zwraca stałą. Model to suma setek takich schodków, więc razem dają
dowolnie pokręconą, nieliniową powierzchnię. Stąd wrażenie „nieliniowego składania sygnałów".

Drugie sprostowanie, ważniejsze. LightGBM natywnie mierzy **ważność cech** (kolumn), a nie
**ważność próbek** (wierszy). To są dwie różne rzeczy i tylko druga odpowiada na pytanie
„które obrazy są istotne".

| | co mierzy | narzędzie |
|---|---|---|
| ważność **cechy** | ile dana kolumna daje zysku w podziałach | `feature_importance`, SHAP |
| ważność **próbki** | ile dany wiersz wnosi do wyniku modelu | GOSS, data valuation, EL2N |

---

## 2. Co w LightGBM faktycznie jest „obserwatorem"

Cztery mechanizmy odróżniają LightGBM od zwykłego boostingu:

1. **GOSS** (*Gradient-based One-Side Sampling*) — sortuje przykłady po wartości bezwzględnej
   gradientu, zachowuje górne `a%` i losuje `b%` z reszty. Uzasadnienie: mały gradient znaczy,
   że model już się tego przykładu nauczył. **To jest dokładnie ten „obserwator"** z pytania,
   tylko działa wewnątrz treningu drzew, na wierszach tabeli.
2. **Histogramy** — cechy ciągłe wrzucone do kubełków, koszt spada z `O(dane × cechy)`
   do `O(kubełki × cechy)`.
3. **EFB** (*Exclusive Feature Bundling*) — scala kolumny, które rzadko są niezerowe naraz.
4. **Wzrost liściowy** — dzieli ten liść, który najbardziej zmniejszy stratę, zamiast
   rozbudowywać całe piętro.

Interpretowalność, o którą chodzi przy „istotności", daje SHAP: rozkłada predykcję na wkłady
poszczególnych cech, także dla pojedynczej próbki.

---

## 3. Cztery pytania i co na nie mówi literatura

### P1. Czy pomysł GOSS przeniesiono na sieci głębokie?

**Tak, i to dawno — ale jako ważenie próbek w trakcie treningu, nie jako trwałe usuwanie.**

- [Kat18] („Not All Samples Are Created Equal") losuje próbki z prawdopodobieństwem
  proporcjonalnym do **górnego ograniczenia normy gradientu**, liczonego z gradientu po
  przedaktywacjach ostatniej warstwy — czyli jednym przejściem w przód, bez pełnej
  propagacji wstecznej. Zyski: błąd testowy niżej o 5–17% względnie, strata treningowa
  nawet dziesięciokrotnie niżej przy tym samym czasie.
- **Dwa ostrzeżenia z tej samej pracy, oba istotne dla nas.** Po pierwsze, **sama wartość
  straty jest złym zamiennikiem gradientu**: błąd dopasowania prawdopodobieństw do prawdziwych
  norm gradientu wyniósł 0,017 dla straty i 0,002 dla ograniczenia. Po drugie, autorzy
  **wprost odradzają trwałe wyrzucanie danych**: ważność próbki zmienia się z każdą epoką,
  więc rozkład policzony raz nie obowiązuje później.
- [Kil21] (GRAD-MATCH) robi wersję „koresetową": dobiera podzbiór z wagami tak, żeby jego
  gradient przybliżał gradient całego zbioru, i **przelicza podzbiór co 20 epok**. CIFAR-100
  przy 10% danych: 69,58% kontra 65,95% dla losowania; ImageNet przy 30%: 69,06% kontra
  55,12%. Tylko klasyfikacja, tylko gradienty ostatniej warstwy.
- [Hu21] pokazuje, kiedy priorytet po stracie zawodzi (szum, błędne etykiety), a [Kir23]
  że wcześniejsze wyniki GraNd na inicjalizacji wymagały weryfikacji; nie jest
  to twierdzenie o każdym modelu i zbiorze. EL2N należy czytać także w poprawionej wersji pracy.

### P2. Czy ktoś uczy taniego modelu tabelarycznego, żeby przewidywał wartość próbki?

**Pomysł jest opublikowany i działa, ale w wersji z sąsiadami lub grafem, nie z drzewami.**

[Sch25] („Effective Data Pruning through Score Extrapolation") robi dokładnie to, o co pytasz:

- liczy kosztowny wynik trudności (Dynamic Uncertainty, TDDS) tylko na **10–25% puli**,
- ekstrapoluje go na resztę **ważonym kNN albo trzywarstwową siecią grafową na embeddingach**,
- korelacja rangowa Spearmana z prawdziwym wynikiem: **0,53–0,70** (syntetyczny CIFAR-100),
  ale tylko **0,16–0,36** na Places-365 i **0,20–0,35** na ImageNet,
- strata dokładności wobec liczenia wyniku na wszystkim: **0,19–1,71 pp**,
- oszczędność całkowitego czasu: **45–80%**,
- żadnego eksperymentu z detekcją.

Model drzewiasty **nie pojawia się** w tej roli. Najbliżej jest **Data-OOB** z benchmarku
[Jia23] — wycena z próbek out-of-bag lasu losowego. I tu ważny wynik negatywny: Data-OOB
wygrywa w wykrywaniu błędnych etykiet i w dokładaniu danych, ale w **usuwaniu danych wypada
na poziomie losowania albo gorzej**. Czyli akurat w naszym zadaniu (przyciąć pulę) tree-based
valuation zawodzi.

### P3. Wycena danych na obrazach — koszt i skuteczność

- **KNN-Shapley** [Jia21]: liczony w formie zamkniętej na osadzeniach z modelu wstępnie
  wytrenowanego, złożoność `O(Nd + N log N)`, **bez trenowania czegokolwiek**. Sekundy tam,
  gdzie Data Shapley potrzebuje godzin (10⁰–10¹ s kontra 10³–10⁴ s przy tysiącu punktów).
  Tiny ImageNet, usunięte 40% danych: **49% kontra 40,5% dla losowania**. Wykrywanie błędnych
  etykiet: 78% złapanych przy obejrzeniu 20% danych, wobec 20% dla losowania.
- **Data Shapley** [Gho19] i warianty Monte Carlo: wymagają wielokrotnego trenowania,
  dla sieci głębokich nie kończą się w rozsądnym czasie.
- **LAVA** [Jus23] (transport optymalny): tanie, ale jego przewaga dotyczy wykrywania
  zepsutych cech, nie rankingu wkładu próbki.
- **Wszystko powyżej to klasyfikacja.** Ani jedna z tych metod nie ma eksperymentu
  detekcyjnego — potwierdzają to i [Jia21], i [Jia23].

### P4. SHAP jako audyt zbioru

**To jedyne zastosowanie drzew, które ma w detekcji konkretny precedens.** [Kuz24] uczy las
losowy na metadanych sceny i obiektu (odległość, rozmiar kątowy, zasłonięcie, pogoda, pora dnia,
prędkość, postawa), a przewiduje, czy dany obiekt zostanie wykryty, i czy predykcja nie jest
duchem. Ranking ważności liczony przez SHAP wskazał odległość, rozmiar kątowy i zasłonięcie
jako dominujące dla kamer. Zastrzeżenie: praca **audytuje błędy, nie wybiera danych**, a
skuteczności samego lasu nie podaje liczbowo. Podobnie [Kon19b] przewiduje jakość detekcji
z cech obrazu.

---

## 4. Werdykt

**Jako gotowa metoda wyboru obrazów: nie ma na to dowodu.** W korpusie 171 prac nie ma ani
jednego pełnego łańcucha „drzewo gradientowe na cechach obrazu → wartość próbki → podzbiór
treningowy detektora". Są trzy kawałki tej drogi osobno: gradient jako sygnał ważności
[Kat18, Kil21], ekstrapolacja wyniku tanim modelem [Sch25] i drzewo jako audytor [Kuz24].

Do tego trzy rzeczy, które trzeba mieć z tyłu głowy:

1. **Drzewo nie widzi obrazu.** Potrzebuje wektora cech, czyli i tak najpierw liczymy
   osadzenia (DINOv3) albo deskryptory. Koszt tego przejścia jest większy niż koszt samego
   drzewa, więc „tani obserwator" jest tani dopiero po zapłaceniu za cechy.
2. **Drzewo potrzebuje celu.** Żeby przewidywać ważność, musi się jej najpierw nauczyć od
   modelu kosztownego — czyli i tak trzeba policzyć trudność na 10–25% puli [Sch25].
   To nie zastępuje modelu pomocniczego, tylko go skraca.
3. **Koszt selekcji liczy się do wyniku.** [Oka23] pokazuje, że gdy doliczyć czas wyboru,
   zwykłe **losowanie nowego podzbioru w każdej epoce (RS2)** bije metody koresetowe:
   ImageNet przy 10% danych 66% kontra 55% dla najlepszej metody koresetowej, a najszybszy
   baseline przycinania potrzebował 250 minut na sam model pomocniczy. To jest baseline,
   którego nasz protokół jeszcze nie ma.

---

## 5. Co z tego warto wziąć do naszego pipeline'u

### Zastosowanie A: ekstrapolacja trudności — hipoteza do porównania

Zamiast liczyć VPS/DUAL na wszystkich 17,8 tys. kadrów źródłowych, policzyć je z proxy
RF-DETR-S na **20% puli**, a resztę uzupełnić modelem tanim: ważony kNN na osadzeniach
DINOv3 (wersja z [Sch25]) **albo** LightGBM na tych samych osadzeniach po PCA plus metadane
(czas względny, rozmiar ramki GT, ostrość). Identyfikator pacjenta/filmu służy
do grupowania podziału, nie jako ciągła cecha liczbowa. Cel kosztownego scorer'a
obliczamy na części puli train i oceniamy ekstrapolację na innych grupach train.
Rozróżniamy ograniczenie liczby punktów scoringu od ograniczenia treningu proxy
do tych punktów: to inne oszczędności i inne źródła biasu.

- Nie ma uniwersalnego progu Spearmana 0,5 oddzielającego użyteczny ranking.
  Raportować korelację z niepewnością, overlap wybranych zbiorów i downstream
  AP względem pełnego scorera oraz random przy tym samym K. Kryterium
  akceptowalnej utraty AP i oszczędności czasu ustalić przed eksperymentem.
- Potencjalny zysk: mniej czasu GPU na scoring; trzeba zmierzyć cały koszt
  ekstrakcji, proxy, modelu zastępczego i końcowego treningu.
- Ryzyko: jeśli korelacja wyjdzie niska, wynik trzeba zaraportować jako negatywny i liczyć
  trudność na całej puli.

### Zastosowanie B: SHAP jako audyt, co pipeline naprawdę wybiera (to polecam bardziej)

Uczymy LightGBM na kadrach treningowych: wejście to **mierzalne własności kadru** (rozmiar
ramki narzędzia w pikselach, ostrość z widma Fouriera, liczba segmentów, liczba narzędzi,
czas w filmie), wyjście to **czy proxy wykrył narzędzie** albo jego błąd
lokalizacji. Podział obserwatora jest grupowy po pacjencie/operacji wewnątrz
train; identyfikator grupy nie trafia jako arbitralna liczba do predyktorów.
SHAP wyjaśnia predykcje modelu zastępczego, nie dowodzi przyczyn porażki
detektora. Trzeba sprawdzić jakość samego obserwatora na odłożonych grupach.

To odpowiada na dwa pytania recenzentów naraz: czym naprawdę różnią się kadry trudne,
i czy nasz wybór nie preferuje po cichu jednej cechy (na przykład dużych, ostrych narzędzi).
Precedens: [Kuz24]. Koszt: minuty na CPU, bo cechy i tak mamy policzone.

### Czego nie robić

- Nie nazywać tego nową metodą selekcji — bez porównania z losowaniem przy identycznym
  budżecie to będzie pierwsza rzecz, którą recenzent podważy.
- Nie uzasadniać trwałego usuwania obrazów samym istnieniem GOSS: to mechanizm
  przyspieszania treningu drzew. Ważność gradientowa zmienia się z modelem;
  przydatność statycznego podzbioru wymaga osobnego testu.
- Data Shapley i influence functions traktować jako osobne propozycje z kosztem
  do zmierzenia. Brak wyniku detekcyjnego w zacytowanych pracach nie dowodzi
  braku całej literatury detekcyjnej ani niewykonalności każdej aproksymacji.

---

## 6. Odnośniki

- [Ke17] Ke i in., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 2017 —
  https://www.semanticscholar.org/paper/497e4b08279d69513e4d2313a7fd9a55dfb73273
- [Kat18] Katharopoulos, Fleuret, *Not All Samples Are Created Equal: Deep Learning with
  Importance Sampling*, ICML 2018 — https://arxiv.org/abs/1803.00942
- [Kil21] Killamsetty i in., *GRAD-MATCH*, ICML 2021 — https://arxiv.org/abs/2103.00123
- [Sch25] Schoch i in., *Effective Data Pruning through Score Extrapolation*, 2025 —
  https://doi.org/10.48550/arXiv.2506.09010
- [Oka23] Okanovic i in., *Repeated Random Sampling for Minimizing the Time-to-Accuracy of
  Learning*, 2023 — https://doi.org/10.48550/arXiv.2305.18424
- [Jia21] Jia i in., *Scalability vs. Utility*, CVPR 2021 —
  https://doi.org/10.1109/CVPR46437.2021.00814
- [Jia23] Jiang i in., *OpenDataVal: a Unified Benchmark for Data Valuation*, NeurIPS 2023 —
  https://doi.org/10.48550/arXiv.2306.10577
- [Gho19] Ghorbani, Zou, *Data Shapley*, ICML 2019 — https://arxiv.org/abs/1904.02868
- [Jus23] Just i in., *LAVA: Data Valuation without Pre-Specified Learning Algorithms*,
  ICLR 2023 — https://doi.org/10.48550/arXiv.2305.00054
- [Kuz24] Kuzin i in., *Methodology for an Analysis of Influencing Factors on 3D Object
  Detection Performance*, 2024 — https://doi.org/10.1109/SWC65939.2025.00226
- [Kon19b] Kong i in., *Blind Image Quality Prediction for Object Detection*, MIPR 2019 —
  https://doi.org/10.1109/MIPR.2019.00046
- [An23] An i in., *Active Learning Guided by Efficient Surrogate Learners*, AAAI 2024 —
  https://doi.org/10.1609/aaai.v38i10.28961
- [Hu21] Hu i in., *When does loss-based prioritization fail?*, 2021 —
  https://arxiv.org/abs/2107.07741
- [Kir23] Kirsch, *Does „Deep Learning on a Data Diet" reproduce?*, 2023 —
  https://doi.org/10.48550/arXiv.2303.14753
- [Yoo19] Yoon i in., *Data Valuation using Reinforcement Learning*, ICML 2020 —
  https://arxiv.org/abs/1909.11671

Deep search (171 prac): https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893

---

## Historia zmian

- **2026-09-11, refaktor** — usunięto arbitralny próg Spearmana, rozdzielono
  scoring od ekspozycji proxy, SHAP od przyczynowości i ID pacjenta od cechy.
- **2026-09-11** — pierwsza wersja: mechanizm LightGBM, rozdzielenie ważności cech od
  ważności próbek, cztery pytania, werdykt i dwa zastosowania dla pipeline'u.
