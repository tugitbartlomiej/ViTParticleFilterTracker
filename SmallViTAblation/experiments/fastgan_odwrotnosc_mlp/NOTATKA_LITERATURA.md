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
   ~2,5 h na obraz (RTX 3090) i ~15 GB pamięci karty (RTX 3070 ma 8 GB); kod publiczny. (Liczba „cosinus > 0,99”
   z podsumowania dotyczy tylko jednego testu, patrz sekcja z pełnymi tekstami niżej.)
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

## Pełne teksty przeczytane 2026-09-29 (Undermind `read_pdfs`)

Surowe odpowiedzi podagentów: `literatura/pelne_teksty_sd_2026-09-29.md`. Przeczytane w całości: [Shi25f], [Zha24p],
[Kar26], [Far23b], [Neu25], [Yu23], [Wal23b], [Eyr24], [Tan24d], [Fri23b]. [Ban23c] (Universal Guidance): PDF
nieczytelny dla Undermind. [Rou26] i [Vat25] przeczytane wcześniej (sekcje wyżej).

- **[Shi25f] Shirahama 2025.** Stable Diffusion 1.4 bez tekstu, 512 × 512, 1000 kroków odszumiania; na każdym kroku
  strata L2 między cechą sieci na estymacie czystego obrazu a celem, gradient przeskalowany do normy szumu i obcięty
  (2-3 sigma), 1000 powtórzeń na pierwszych 5 krokach i 8 na pozostałych. Sieci: CLIP ResNet-50, ResNet-50,
  ViT-H/14; małych ani nieodpornych sieci nie badano. Trafienie: odległość do celu o rząd mniejsza niż między
  podobnymi prawdziwymi zdjęciami. „Cosinus > 0,99” dotyczy tylko celów będących przeskalowanymi cechami prawdziwych
  zdjęć (× 0,8 i × 1,2). Czas: 2,5 h i ~15 GB (CLIP, ResNet-50), 4,7 h i 19,6 GB (ViT-H) na RTX 3090. Wniosek autorów:
  mocno zniekształcone obrazy mogą mieć prawie tę samą cechę co naturalne; przy ViT-H pojawiały się duże czarne plamy.
- **[Zha24p] Zhang 2024.** Odwracanie cech ResNet, ViT-base (bloki 3-8), YOLO i CLIP. Optymalizowany jest szum
  startowy modelu o architekturze SD 2.1 (4 × 64 × 64), gradient przez wszystkie 20 kroków próbkowania, 1500 iteracji
  Adam, lr 0,1, strata L2 cech plus wygładzenie obrazu (TV). Wyniki tylko w PSNR, SSIM i IS (np. ViT-base blok 5:
  PSNR 32,5 dB). Czasu i pamięci nie podano; model wewnętrzny, kod niewydany. Prosty opis tła w tekście („snow”)
  bardzo pomaga w głębokich warstwach, zły opis szkodzi.
- **[Kar26] Karthikeyan 2026.** Model dyfuzyjny w latencie uwarunkowany tokenem CLS DINOv2 ViT-B/14 (768 liczb),
  ImageNet-100, 256 × 256, 125 epok. +10,76 punktu to najlepszy wynik DINOv2 (79,26 % przy 4 × danych syntetycznych)
  minus najlepszy wynik modelu warunkowanego klasą (68,5 %). Zgodności latentu obrazu z celem nie mierzono, porównania
  z wyszukiwaniem prawdziwych zdjęć nie ma, domen medycznych nie ma, kodu nie podano.
- **[Far23b] Farid 2023 (LDCE).** Kontrfakt: prawdziwe zdjęcie kodowane do latentu Stable Diffusion, zaszumione tylko
  do kroku 191-250 z 1000 (układ sceny zostaje), potem odszumianie z gradientem zamrożonego klasyfikatora na estymacie
  czystego obrazu i z karą L1 za oddalenie od oryginału. **Zgoda kierunków:** gradient klasyfikatora jest zerowany
  w miejscach, gdzie odbiega o więcej niż 45-55° od kierunku, który sam model dyfuzyjny uważa za zmianę klasy; to
  usuwa zmiany w stylu ataku u sieci nieodpornych bez pomocniczej sieci odpornej. 39 s na obraz na RTX 3090 (24 GB);
  kod MIT `lmb-freiburg/ldce`. Autorzy ostrzegają przed gorszym działaniem poza domeną modelu (np. medycyna).
- **[Neu25] FeatInv 2025.** ControlNet na MiniSD 256 × 256 uwarunkowany przestrzenną mapą cech (ResNet-50, ConvNeXt,
  SwinV2); uczenie 45-60 h na 2 × L40. Cosinus cech obrazu z celem 0,46-0,61 dla ostatniej mapy cech, 0,96-0,98 dla
  wczesnych warstw ConvNeXt. Sterowanie pojęciem działa przez **osłabianie** jego składowej (× 0,25); wzmacnianie
  wypycha cechy poza rozkład uczenia. Kod `AI4HealthUOL/FeatInv`.
- **[Yu23] FreeDoM 2023.** Gradient straty zamrożonej sieci na estymacie czystego obrazu dodawany w każdym kroku;
  cofanie w czasie (ponowne zaszumienie i powtórzenie kroku) tylko w fazie 800-500 z 1000, gdzie powstaje semantyka.
  ~84 s na obraz 512 × 512 na RTX 3090, wobec ~40 min dla Universal Guidance. Działa ze Stable Diffusion przez dekoder
  VAE. Kod `vvictoryuki/FreeDoM`.
- **[Wal23b] DOODL 2023.** Optymalizacja szumu startowego przez cały łańcuch 50 kroków SD 1.4, pamięć stała dzięki
  odwracalnemu próbnikowi EDICT; renormalizacja szumu do normy startowej. Według [Eyr24] ~20 min i ~40 GB na obraz.
  Kod `salesforce/DOODL`.
- **[Eyr24] ReNO 2024.** Optymalizacja szumu startowego jednokrokowego generatora (SD-Turbo i inne), 50 iteracji,
  regularyzacja normy szumu, obcinanie normy gradientu 0,1, zwracany najlepszy obraz z trajektorii. SD-Turbo 512 × 512:
  20 s i 15 GB na A100 (fp16). Dowolna różniczkowalna strata (także dopasowanie cech) da się podpiąć. Kod
  `ExplainableML/ReNO`.
- **[Tan24d] DNO 2024.** Optymalizacja szumu startowego i wszystkich szumów wstrzykiwanych w próbniku SDE (DDIM
  η = 1), gradient przez próbnik z checkpointingiem, regularyzacja prawdopodobieństwa szumu przeciw „oszukiwaniu
  nagrody”. Poniżej 1 min przy 15 krokach; poniżej 15-20 GB (A800).
- **[Fri23b] Frisch 2023 (zaćma).** Własny model dyfuzyjny DDIM na CATARACTS, 128 × 128, warunkowany fazą i zestawem
  narzędzi (classifier-free guidance, w = 2); uczenie 500 epok na 2 × A40. 30 000 obrazów syntetycznych: F1
  klasyfikatora narzędzi 0,897 → 0,916, w rzadkich fazach do +0,10 F1; sam syntetyk daje tylko F1 0,299. Klinicyści
  mylili obrazy syntetyczne z prawdziwymi w 61 % par. Tylko obecność narzędzia w klatce, bez detekcji i bez końcówki.
  Kod `MECLabTUDA/CataSynth`.

### Wnioski z pełnych tekstów dla pętli małego ViT

1. **Pamięć:** każdy wariant ze Stable Diffusion 512 × 512 potrzebuje >= 15 GB; lokalny RTX 3070 (8 GB) odpada,
   zostaje Eden.
2. **Najbliższy naszej pętli jest schemat LDCE [Far23b]:** start od prawdziwej kotwicy zamiast od czystego szumu,
   częściowe zaszumienie, gradient siły tożsamości małego ViT (cel s0 + Δs) plus kara za oddalenie od kotwicy.
   Zgoda kierunków jest gotowym lekarstwem na nieodporność małego ViT, które u nas dziś pełni uśrednianie po
   zaburzonych kopiach. Koszt ~39 s na obraz na RTX 3090, czyli ~1 h na ziarno przy 84 obrazach.
3. **Domena:** Stable Diffusion nie widziało obrazów z mikroskopu operacyjnego; [Far23b] ostrzega przed spadkiem
   jakości poza domeną, a [Fri23b] uczyło własny model na CATARACTS. FastGAN jest uczony na naszych wycinkach, więc ma
   tu przewagę.
4. **Wzmacnianie kierunku jest trudniejsze niż osłabianie** ([Neu25]); nasza pętla prosi o wzrost siły tożsamości,
   więc obraz może wyjść poza rozkład danych.
5. **Sieć nieodporna przyjmie obraz zniekształcony** ([Shi25f]): trafienie w cechę nie gwarantuje realistycznego
   obrazu, dlatego obraz nadal powinien służyć tylko do wyszukania prawdziwych klatek.
6. Żadna z prac nie zmienia wyniku wersji 3: zysk daje samo wyszukiwanie podobnych klatek, a lepszy generator nie
   doda udziału prognozy.
