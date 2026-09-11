# 21 — Ważność pojedynczej próbki w zadaniach obrazowych: katalog metod

Data: 2026-09-11. Ciąg dalszy `20_DRZEWA_GRADIENTOWE_JAKO_OBSERWATOR.md`, tym razem bez
ograniczenia do drzew: **wszystkie rodziny metod, które mierzą, ile wart jest pojedynczy
obraz dla trenowanej sieci.**

Przeczytane dziś w całości: [Sun25], [Eva24], [Cal25], [Jin18], [Hua25c] (plus wcześniej
[Kat18], [Kil21], [Sch25], [Jia21], [Jia23], [Oka23]). Reszta z metadanych i abstraktów —
oznaczona kursywą w kolumnie „dowód".

---

## 1. Sześć rodzin — skrót

| Rodzina | Co uznaje za „ważne" | Koszt | Detekcja? |
|---|---|---|---|
| A. gradient | próbka, która mocno ciągnie wagi | jeden przebieg w przód (ostatnia warstwa) | częściowo |
| B. strata / trudność | próbka, której model jeszcze nie umie | trening proxy, kilka epok | tak ([Yag25]) |
| C. wkład do metryki | próbka, która podnosi AP całego zbioru | analitycznie, bez retreningu | **tak** ([Sun25]) |
| D. uczony oceniacz | model uczy się przewidywać wartość próbki | meta-trening albo ekstrapolacja | nie |
| E. uczenie się względem odniesienia | próbka, którą model referencyjny umie, a nasz nie | dwa przebiegi w przód | nie |
| F. bez etykiet, z wideo | detekcja migocząca w czasie | koszt inferencji | **tak** ([Jin18]) |

---

## 2. A. Sygnał z gradientu — prosty potomek GOSS

| Metoda | Definicja | Koszt | Dowód |
|---|---|---|---|
| **Importance sampling** [Kat18] | prawdopodobieństwo losowania proporcjonalne do ograniczenia normy gradientu, liczonego z przedaktywacji ostatniej warstwy | jeden przebieg w przód na dużej partii | błąd testowy niżej o 5–17% względnie; **autorzy odradzają trwałe usuwanie danych** |
| **GRAD-MATCH** [Kil21] | podzbiór z wagami, którego gradient przybliża gradient całości; przeliczany co 20 epok | pomijalny (gradienty ostatniej warstwy) | CIFAR-100 przy 10%: 69,58% kontra 65,95% dla losowania; ImageNet przy 30%: 69,06 kontra 55,12 |
| **Mimic Score** [Hua25c] | rzut gradientu próbki na kierunek do wag modelu referencyjnego: `<-g_i, θ_ref - θ_t> / ||θ_ref - θ_t||` | tylko ostatnia warstwa, 2,6x szybciej niż metody wpływu | CIFAR-100, odcięte 20%: ViT-B/16 77,6% kontra 76,7% losowania; wykrywanie złych etykiet F1 powyżej 0,95 |
| **VoG** [Aga20b] | wariancja gradientu **po wejściu** (po pikselach) między checkpointami | wymaga zapisanych checkpointów | *wysoki VoG = nietypowe, zapamiętane, błędnie opisane* |

Wniosek dla nas: gradient jest dobrym sygnałem **w trakcie** treningu, a złym jako trwały
wyrok. Wartość próbki zmienia się z epoki na epokę [Kat18].

## 3. B. Strata i trudność

| Metoda | Definicja | Uwaga |
|---|---|---|
| **OHEM** [Shr16] | w detekcji: propaguj tylko regiony o największej stracie | klasyk (2 778 cytowań), ale to ważenie regionów wewnątrz obrazu, nie wybór obrazów |
| **EL2N** [Pau21] | norma błędu po kilku epokach, średnia z 10 inicjalizacji | patrz `19_...md`, punkt R1 |
| **VPS** [Yag25] | odchylenie IoU dopasowanej predykcji po epokach; wybiera trudność **pośrednią** | jedyna praca o przycinaniu zbioru napisana pod detekcję |
| **DUAL** [Cho25] | `(1 − średnie p) × odchylenie p` w oknie epok | tanie, bo 15–33% pełnego treningu |
| **Second-split forgetting** [Mai22] | kiedy przykład zostaje zapomniany po zmianie zbioru | rozdziela szum od rzadkości |
| **C-score** [Jia20b] | jak często model wytrenowany bez tego przykładu i tak go trafia | kosztowne, ale to wzorzec „prawdy" o trudności |
| **Kiedy zawodzi** [Hu21] | priorytet po stracie psuje się przy szumie i błędnych etykietach | dlatego CCS odcina górę rozkładu |

## 4. C. Wkład do metryki — to jest najbliżej „istotności próbki"

**DetGain** [Sun25] to jedyna znaleziona praca, która liczy wartość obrazu **wprost jako
zmianę mAP całego zbioru**, i robi to dla detekcji.

- Wartość obrazu = przyrost mAP, jaki dają jego detekcje, policzony **analitycznie**
  z krzywej precyzja–czułość, w czasie stałym na detekcję, bez trenowania czegokolwiek.
- Wynik to różnica nauczyciel–uczeń: `s(x) = δmAP(x; nauczyciel) − δmAP(x; uczeń)`.
- Działa **online**: w każdym kroku ocenia super-partię 64–80 obrazów i wybiera z niej
  najlepsze 20% do propagacji wstecznej.
- Wyniki na COCO: Faster R-CNN **40,0 mAP** kontra **37,3** przy losowaniu, FCOS 40,9
  kontra 38,2, ATSS 41,6 kontra 39,4. Średnio **około +2 mAP** na sześciu architekturach,
  a przy zaszumionych etykietach nawet +6,9.
- Koszt: **+200% czasu na iterację**, ale zbieżność około dwa razy szybsza, więc w całości
  wychodzi na zero albo lepiej.
- **Obrazy bez obiektu obsługuje naturalnie**: wszystkie detekcje liczą się jako fałszywe
  alarmy i dają ujemny wkład. To jest dokładnie to, czego brakuje naszemu kodowi (`19_...md`, R2).
- Czego nie robi: nie modeluje redundancji klatek wideo ani duplikatów.

Pozostałe w tej rodzinie: **KNN-Shapley** [Jia21] (Tiny ImageNet, usunięte 40%: 49% kontra
40,5% losowania; koszt `O(Nd + N log N)`), **MoSo** [Tan23b] (wartość = zmiana ryzyka po
usunięciu próbki), benchmark **OpenDataVal** [Jia23] — z którego płynie ostrzeżenie, że żadna
metoda nie wygrywa we wszystkich zastosowaniach, a tree-based Data-OOB przy **usuwaniu** danych
wypada jak losowanie albo gorzej.

## 5. D. Uczony oceniacz — „obserwator" w wersji dosłownej

| Metoda | Czym jest oceniacz | Wynik |
|---|---|---|
| **DataRater** [Cal25] | transformer 50M uczony meta-gradientem przez wewnętrzną pętlę treningu; ocenia próbkę skalarem | do **46,6% oszczędności obliczeń**; meta-trening kosztuje 58% jednego pełnego biegu 1B; **przenosi się między skalami modelu**; wyłącznie język |
| **Score extrapolation** [Sch25] | kNN albo sieć grafowa na osadzeniach przewiduje wynik trudności policzony na 10–25% puli | Spearman 0,16–0,70; strata 0,19–1,71 pp; czas mniejszy o 45–80% |
| **DVRL** [Yoo19] | sieć wyceniająca uczona uczeniem ze wzmocnieniem | dobra w wykrywaniu szumu, słabsza w usuwaniu danych [Jia23] |
| **Surrogate GP** [An23] | rzadki proces Gaussa na cechach sieci, przewiduje resztę błędu | 0,35 s na zapytanie kontra ~300 s dla proxy sieciowego |

## 6. E. Uczenie się względem modelu odniesienia (learnability)

**JEST** [Eva24] liczy `strata(partia | nasz model) − strata(partia | model referencyjny)`.
Wysoki wynik znaczy „to jest do nauczenia, a my jeszcze tego nie umiemy" — odsiewa naraz
przykłady zbyt łatwe i śmieciowe.

- **Wybiera całe partie, nie pojedyncze próbki**, bo o jakości partii decyduje jej skład.
  Przy odsiewie 90% wybór łączny daje do **+6%** wobec wyboru po jednej próbce, który w tym
  reżimie **się psuje**. To bezpośrednie ostrzeżenie przed naiwnym progowaniem wyników.
- Koszt oceny spada do 10% kroku, gdy super-partię ocenia się w niskiej rozdzielczości
  (Flexi-JEST, FlexiViT z łatami 32 zamiast 16).
- Efekt: ten sam wynik przy **13x mniejszej liczbie iteracji** i 10x mniejszej liczbie operacji.
- Tylko uczenie kontrastywne obraz-tekst. **Dla detekcji nie zrobił tego nikt** — a przeniesienie
  jest proste, bo modelem odniesienia może być RF-DETR wytrenowany na COCO.

Pokrewne: [Eva23] (aktywne uczenie w dużej skali), [Uda24] (aktywna kuracja jako destylacja).

## 7. F. Sygnał bez etykiet, prosto z wideo — najtańszy dla nas

**Jin18** wydobywa trudne przykłady z **niespójności czasowej detekcji**, bez żadnych etykiet:

- detekcja o pewności co najmniej 0,8 jest śledzona dopasowaniem wzorca w oknie ±5 klatek;
  jeśli największe IoU ze śladem w sąsiednich klatkach spada poniżej 0,2, detekcja jest
  **migotnięciem**, czyli fałszywym alarmem (trudny negatyw);
- odwrotnie, jednoklatkowa dziura w ciągłym śladzie to **trudny pozytyw**;
- dołożenie takich klatek do treningu: Caltech Pedestrians, wskaźnik pominięć **23,83% → 18,72%**;
  WIDER Face AP z 0,907 na 0,921;
- koszt: sama inferencja, zero pracy człowieka.

To jest sygnał, którego nasza pula dostarcza za darmo — Cataract-LMM to ciągłe wideo, a my
mamy zapisane checkpointy i kolejność klatek. Pokrewne: [Yoo23] (redundancja klatek przy
anotacji), COWAL, Wu i in. 2024 (klastry plus niepewność w wideo chirurgicznym).

## 8. G. Bez modelu: same cechy i gęstość

**DBP** [Abb24] (złożoność klastrów pojęciowych), SemDeDup, **CLIP-powered selection** [Yan24c].
Tania podstawa, ale mierzy reprezentatywność, nie wartość dla uczenia.

---

## 9. Co z tego bierzemy — ranking dla naszego pipeline'u

1. **Klatki bez narzędzia — naprawić natychmiast wzorem z DetGain** [Sun25]: obraz bez obiektu
   nie dostaje maksymalnej trudności, tylko ujemny wkład za fałszywe alarmy. To domyka punkt R2
   z `19_...md` i ma za sobą wynik +2 mAP.
2. **Migotanie czasowe jako darmowy sygnał** [Jin18]: policzyć na naszym wideo z proxy RF-DETR.
   Daje i baseline („trudne klatki bez etykiet"), i dodatkową cechę do selekcji, i rysunek
   do artykułu. Koszt to jedna inferencja po puli.
3. **Learnability względem modelu COCO** [Eva24]: `strata(proxy) − strata(RF-DETR COCO)`.
   Dwa przebiegi w przód, żadnego treningu. Dla detekcji nieopisane, więc to realna nowość,
   a nie powtórka.
4. **Mimic Score** [Hua25c] jako tani wariant gradientowy — rzut gradientu ostatniej warstwy
   na kierunek do wag modelu referencyjnego.
5. **Ekstrapolacja wyniku** [Sch25], jeśli liczenie trudności na całej puli okaże się za drogie.

## 10. Trzy ostrzeżenia, które trzeba zapisać w protokole

1. **Wybór po jednej próbce psuje się przy silnym odsiewie.** [Eva24] mierzy to wprost:
   przy 90% odsiewu wybór łączny bije wybór niezależny o 6%, a ten drugi degraduje.
   Nasza selekcja jest niezależna per klatka — stąd znaczenie budżetu per klaster.
2. **Metody online zmieniają trening.** DetGain i JEST wybierają dane w każdym kroku, więc
   **łamią regułę P5** („identyczny harmonogram dla wszystkich wariantów"). Jeśli je bierzemy,
   to jako **osobny eksperyment**, nie jako kolejny wariant macierzy.
3. **Koszt selekcji wlicza się do wyniku** [Oka23]. Każdą metodę z tej listy raportujemy
   z czasem: ile GPU-godzin poszło na scoring, zanim zaczął się trening.

---

## 11. Odnośniki

- [Sun25] Sun i in., *Online Data Curation for Object Detection via Marginal Contributions to
  Dataset-level Average Precision* (DetGain), 2025 — https://doi.org/10.48550/arXiv.2511.14197
- [Eva24] Evans i in., *Data curation via joint example selection further accelerates multimodal
  learning* (JEST), 2024 — https://doi.org/10.48550/arXiv.2406.17711
- [Eva23] Evans i in., *Bad Students Make Great Teachers*, 2023 —
  https://doi.org/10.48550/arXiv.2312.05328
- [Uda24] Udandarao i in., *Active Data Curation Effectively Distills Large-Scale Multimodal
  Models*, CVPR 2025 — https://doi.org/10.1109/CVPR52734.2025.01345
- [Cal25] Calian i in., *DataRater: Meta-Learned Dataset Curation*, 2025 —
  https://doi.org/10.48550/arXiv.2505.17895
- [Jin18] Jin i in., *Unsupervised Hard Example Mining from Videos for Improved Object
  Detection*, ECCV 2018 — https://doi.org/10.1007/978-3-030-01261-8_19
- [Hua25c] Huang i in., *Evaluating Sample Utility for Efficient Data Selection by Mimicking
  Model Weights*, 2025 — https://arxiv.org/abs/2501.06708
- [Aga20b] Agarwal i in., *Estimating Example Difficulty using Variance of Gradients*,
  CVPR 2022 — https://doi.org/10.1109/CVPR52688.2022.01012
- [Shr16] Shrivastava i in., *Training Region-Based Object Detectors with Online Hard Example
  Mining*, CVPR 2016 — https://doi.org/10.1109/CVPR.2016.89
- [Jia20b] Jiang i in., *Characterizing Structural Regularities of Labeled Data in
  Overparameterized Models* (C-score), ICML 2021 — https://arxiv.org/abs/2002.03206
- [Mai22] Maini i in., *Characterizing Datapoints via Second-Split Forgetting*, NeurIPS 2022 —
  https://doi.org/10.48550/arXiv.2210.15031
- [Tan23b] Tan i in., *Data Pruning via Moving-one-Sample-out*, NeurIPS 2023 —
  https://doi.org/10.48550/arXiv.2310.14664
- [Kao18] Kao i in., *Localization-Aware Active Learning for Object Detection*, ACCV 2018 —
  https://doi.org/10.1007/978-3-030-20876-9_32
- [Abb24] Abbas i in., *Effective pruning of web-scale datasets based on complexity of concept
  clusters*, ICLR 2024 — https://doi.org/10.48550/arXiv.2401.04578
- [Yan24c] Yang i in., *A CLIP-Powered Framework for Robust and Generalizable Data Selection*,
  ICLR 2025 — https://doi.org/10.48550/arXiv.2410.11215
- [Yoo23] Yoon i in., *Exploring Video Frame Redundancies for Efficient Data Sampling and
  Annotation in Instance Segmentation*, CVPRW 2023 — https://doi.org/10.1109/CVPRW59228.2023.00333
- pozostałe ([Kat18], [Kil21], [Sch25], [Jia21], [Jia23], [Oka23], [Yoo19], [An23], [Hu21])
  — spis w `20_DRZEWA_GRADIENTOWE_JAKO_OBSERWATOR.md`, sekcja 6.

---

## Historia zmian

- **2026-09-11** — pierwsza wersja: sześć rodzin metod ważności próbki dla obrazów,
  ranking zastosowań dla pipeline'u, trzy ostrzeżenia do protokołu.
