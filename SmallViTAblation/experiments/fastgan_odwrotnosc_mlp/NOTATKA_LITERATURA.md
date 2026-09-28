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
