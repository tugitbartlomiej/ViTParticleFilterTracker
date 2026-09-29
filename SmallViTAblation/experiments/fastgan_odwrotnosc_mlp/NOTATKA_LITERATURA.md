# Przegląd literatury: jak przejść z latentu przewidzianego przez Koopmana do obrazu (2026-09-28)

Źródła: Undermind (przestrzeń „ViT: detekcja końcówki narzędzia w chirurgii zaćmy”), pełne teksty przeczytane przez
`read_pdfs` dla [Rou26] i [Vat25]; pozostałe na podstawie streszczeń i wcześniejszego głębokiego wyszukiwania
„generative model synthesizes target image then retrieves real unlabeled samples for training” (2026-09-25).

## Praca wskazana przez użytkownika

**Rousselot, Wendebourg, Köthe 2026, „Show Me What You Don't Know: Efficient Sampling from Invariant Sets for Model
Validation”** — https://doi.org/10.48550/arXiv.2603.21782
- Pojęcie: zbiór niezmienniczy („włókno”) φ⁻¹(h) = {x : φ(x) = h} — wszystkie obrazy, które sieć φ widzi tak samo;
  próbkują z niego obrazy realistyczne, p(x|h) ∝ p(x) na włóknie.
- Dwa sposoby: (A) warunkowy generator (przepływy, dyfuzja) uczony z dodatkową „stratą włókna”
  ‖φ(x) − φ(x̃)‖² — dokładny, ale drogi (~14 000 min na A40 dla ImageNet); (B) bez uczenia: nieuwarunkowany model
  dyfuzyjny sterowany w trakcie odszumiania (NDTM, Pandey i in. 2025), z poprawką u_t optymalizowaną w każdym kroku
  i z **dynamicznym celem**: porównują cechy estymaty obrazu z cechami tak samo zaszumionej i odszumionej referencji,
  żeby porównywać obrazy o tym samym poziomie szczegółu. ~1,8 min na próbkę (A40).
- Wynik: próbki znacznie bliżej włókna niż najbliższy sąsiad w zbiorze (DINOv2 na ImageNet: 273 wobec 1225) przy FID
  bliskim modelu bazowego.
- **Kluczowe dla nas (dodatek A):** dopasowanie reprezentacji gradientem wprost w pikselach daje szum w stylu ataku,
  „nieznaczące semantycznie zmiany pojedynczych pikseli”; poprawki trzeba robić w przestrzeni semantycznej generatora.
  Dokładnie to zobaczyliśmy: piksele dały 61 % drogi przy niewidocznej zmianie obrazu.
- Czego nie robią: nie celują w reprezentację przesuniętą o prognozę (my: h + krok Koopmana), tylko w h istniejącego
  obrazu; nie używają GAN-ów.

## Prace pokrewne, które zmieniają projekt

- **Väth i in. 2025, „Diffusion Classifier Guidance for Non-robust Classifiers”** — https://doi.org/10.48550/arXiv.2507.00687.
  Sterowanie dyfuzją siecią NIEODPORNĄ (jak nasz mały ViT) bez poprawek nie działa wcale; działa po: (1) liczeniu sieci
  na jednokrokowej estymacie czystego obrazu, (2) wygładzaniu gradientu średnią wykładniczą między krokami (β = 0,99,
  bez korekty obciążenia; wariant w stylu Adama pogarszał). CelebA: trafność klasy 56 % → 99,99 %.
- **Nguyen i in. 2016, „Synthesizing the preferred inputs for neurons in neural networks via deep generator networks”**
  — https://arxiv.org/abs/1605.09304. Optymalizacja wejścia generatora jako priorytetu obrazów zamiast pikseli: obrazy
  realistyczne i interpretowalne. To jest nasze obecne podejście z FastGAN-em.
- **Santurkar i in. 2019, „Image Synthesis with a Single (Robust) Classifier”** — gradienty sieci odpornych są
  „percepcyjnie wyrównane” (zmiana latentu = widoczna zmiana obrazu), sieci zwykłych — nie. Wyjaśnia, dlaczego krok
  Koopmana w latencie naszego ViT jest w dużej części niewidoczny.
- **Bansal i in. 2023, „Universal Guidance for Diffusion Models”** — https://doi.org/10.1109/CVPRW59228.2023.00091.
  Sterowanie dowolną siecią (cechy, detekcja, segmentacja) bez przeuczania modelu dyfuzyjnego.
- Generowanie zapytania, a potem wyszukanie prawdziwych próbek zamiast uczenia na syntetycznych: **Mayer i Timofte,
  „Adversarial Sampling for Active Learning”** — https://doi.org/10.1109/WACV45572.2020.9093556; prawdziwe obrazy
  wyszukane lepsze od syntetycznych: **Geng i in. 2024** — https://doi.org/10.48550/arXiv.2406.05184, **Burg i in. 2023**
  — https://arxiv.org/abs/2304.10253. Nasz schemat „FastGAN → podobne prawdziwe klatki” ma więc oparcie w literaturze.

## Wnioski dla naszego przejścia „latent Koopmana → obraz FastGAN”

1. **Cel względny, nie bezwzględny** (zrobione): obraz startowy + krok Koopmana; odpowiednik dynamicznego celu z [Rou26].
2. **Sterowanie w przestrzeni generatora, nie w pikselach** ([Rou26] dod. A, [Ngu16]); wejście z nie wystarcza (4 %),
   potrzebne warstwy głębsze (36 %).
3. **Ochrona przed rozwiązaniami niewidocznymi**, bo ViT jest nieodporny ([San19], [Vat25]): latent liczony jako
   średnia z zaburzonych kopii obrazu (jasność, szum, rozmycie) — zmiana musi przetrwać zaburzenia; test: `sterowanie3.py`.
4. **Realizm**: kara dyskryminatora FastGAN-a (uwaga: dyskryminator ocenia czysty szum wyżej niż obrazy z generatora,
   więc jest wiarygodny tylko blisko rozkładu danych — mała waga).
5. **Wariant docelowy z literatury** (większa zmiana): mały nieuwarunkowany model dyfuzyjny na wycinkach narzędzia
   i sterowanie jak w [Rou26] + stabilizacja z [Vat25]; koszt: uczenie modelu dyfuzyjnego (godziny na Edenie) i ~2 min
   na obraz. FastGAN zostaje jako szybki wariant.

## Uzupełnienie 2026-09-29: Stable Diffusion, które trafia w zadany latent

Źródło: wyszukiwanie `/paper-search` w równoległej sesji (Undermind, szybkie i pogłębione, 133 prace; Scite bez
wycofań), wklejone przez użytkownika. Liczby poniżej pochodzą z tamtego podsumowania; pełnych tekstów tu nie czytano.
Pełna lista: projekt Undermind `40294fa6-c9a3-4469-9a61-d3eddeaee893`, ścieżka „Stable Diffusion generating images
whose representation in an external frozen network matches a target latent”.

Trzy sposoby:
1. Sterowanie w trakcie odszumiania stratą z zewnętrznej sieci, bez douczania: Bansal i in. 2023 (Universal
   Guidance, https://doi.org/10.1109/CVPRW59228.2023.00091), Yu i in. 2023 (FreeDoM,
   https://doi.org/10.1109/ICCV51070.2023.02118), Shen i in. 2024 (https://doi.org/10.52202/079017-3460),
   Song i in. 2023 (Loss-Guided Diffusion). **Shirahama 2025, „Feature Space Analysis by Guided Diffusion Model”**:
   cosinus z celem > 0,99, ale ~2,5 h na obraz (RTX 3090) i >= 15 GB pamięci karty (RTX 3070 ma 8 GB); kod publiczny.
2. Optymalizacja szumu startowego (odpowiednik naszej inwersji FastGAN-a): Wallace i in. 2023 (DOODL,
   https://doi.org/10.1109/ICCV51070.2023.00669), Eyring i in. 2024 (ReNO, https://doi.org/10.48550/arXiv.2406.04312),
   Tang i in. 2024 (https://arxiv.org/abs/2405.18881). **Zhang i in. 2024, „Unlocking Visual Secrets: Inverting
   Features with Diffusion Priors”** (https://doi.org/10.48550/arXiv.2412.10448): odtwarzanie cech ViT i YOLO,
   ~1500 kroków na obraz; najbliżej naszej pętli.
3. Douczony dodatek biorący latent na wejściu: Bordes, Balestriero, Vincent 2021 (RCDM,
   https://arxiv.org/abs/2112.09164), Ye i in. 2023 (IP-Adapter, https://doi.org/10.48550/arXiv.2308.06721),
   Neukirch i in. 2025 (FeatInv, https://doi.org/10.48550/arXiv.2505.21032; cosinus z celem tylko 0,5-0,6),
   Karthikeyan 2026 (Stable Diffusion douczone na latencie DINOv2; obrazy dały klasyfikator lepszy o 10,8 punktu
   niż obrazy z samej nazwy klasy).

Pokrewne: kontrfakty (Farid i in. 2023, https://doi.org/10.48550/arXiv.2310.06668, ~39 s na obraz; Augustin i in.
2022, https://doi.org/10.48550/arXiv.2210.11841; Sobieski, Biecek 2024, Global Counterfactual Directions,
https://doi.org/10.1007/978-3-031-73036-8_5), kierunki błędów w latencie (Jain i in. 2022,
https://doi.org/10.48550/arXiv.2206.14754), zaćma: Frisch i in. 2023 (MICCAI, https://doi.org/10.48550/arXiv.2308.02587,
dyfuzja sterowana etykietą narzędzia, rzadkie przypadki) i Frisch i in. 2025 (GAUDA, WACV,
https://doi.org/10.1109/WACV61041.2025.00370).

Wnioski dla nas:
- Wyszukiwanie ponownie wskazuje, że prawdziwe wyszukane zdjęcia często uczą lepiej niż syntetyczne ([Bur23], [Gen24]);
  nasz schemat „generator tylko do zapytania, uczenie na prawdziwych klatkach” zostaje.
- Wersja 3 pętli pokazała, że zysk daje samo wyszukiwanie podobnych klatek (ramię current ≈ ramię z prognozą).
  Lepszy generator poprawi realizm zapytania, ale sam nie doda udziału prognozy.
- Tożsamość to wąska wiązka kierunków (`maly_vit_tozsamosc_tla`, 2026-09-29), więc wiele różnych obrazów ma tę samą
  siłę tożsamości; to ten sam „zbiór niezmienniczy” co w [Rou26].
- Na RTX 3070 wykonalne są tylko warianty szybkie (sterowanie stratą, optymalizacja szumu w mniejszej rozdzielczości);
  [Shi25] tylko na Edenie i za wolno: pętla potrzebuje 6 obrazów na obrót, czyli 84 na ziarno w iteracji 2
  (~210 h na ziarno przy 2,5 h na obraz).
