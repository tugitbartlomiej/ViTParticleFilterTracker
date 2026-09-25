# Teoria doboru klatek przez przewidziany latent: model Langevina, twierdzenie o stabilności wyboru

Data: 2026-09-25/26, noc. Cel: zamknąć wyniki sekcji 5–7 w jednym modelu, który wyjaśnia, dlaczego dobór klatek
przez przewidziany latent (Koopman → FastGAN → DINOv2) daje w praktyce te same klatki co sama kotwica, oraz mówi,
kiedy mógłby dawać inne. Teoria nie zmienia wyniku pomiaru; daje przewidywania do sprawdzenia.

## 1. Model: Langevin z liniowym dryfem (proces Ornsteina–Uhlenbecka w czasie dyskretnym)

Latent tożsamości narzędzia dla obrazu sondy (stały obraz przepuszczany przez detektor na kolejnych punktach
kontrolnych) po aktualizacji t:

    z_{t+1} − z* = A (z_t − z*) + σ ξ_t,   ξ_t ~ N(0, I)

A to część deterministyczna (to, co przybliża operator Koopmana), σ ξ_t to szum mini-batchy SGD, z* to stan, do
którego latent dąży. Sekcja 5e: dryf ogólny hamuje wykładniczo, α ≈ 0,89 na 5 aktualizacji, i gaśnie około kroku 500.

Prognoza na horyzont h i jej przesunięcie względem stanu obecnego:

    E[z_{t+h}] = z* + A^h (z_t − z*),   d_h = E[z_{t+h}] − z_t = −(I − A^h)(z_t − z*)

Pomiar (sekcja 5c/6): od kroku 245 ‖d_100‖ to 6 procent rozrzutu między obrazami, do końca uczenia 13 procent.
Szum wokół prognozy ma kowariancję σ² Σ_{i<h} A^i A^{iT}; prognoza niesie informację tylko wtedy, gdy ‖d_h‖ jest
duże względem tego szumu i względem rozdzielczości wyszukiwarki (niżej).

## 2. Od latentu do wyboru klatek

Generator daje obraz G(z), wyszukiwarka liczy wektor q = e(G(z)) (DINOv2, znormalizowany) i ocenia każdą klatkę
puli x_i przez s_i(q) = ⟨q, x_i⟩ (kosinus). Wybór to top-k według s_i. Obraz przewidziany daje q' = e(G(z_t + d_h)),
obecny q = e(G(z_t)).

**Twierdzenie 1 (stabilność wyboru).** Niech s_(k) i s_(k+1) będą k-tą i (k+1)-szą największą oceną dla q oraz
δ = ‖q' − q‖. Jeśli δ < (s_(k) − s_(k+1)) / 2, to top-k dla q' jest tym samym zbiorem co top-k dla q.

Dowód. Dla wektorów jednostkowych x_i: |s_i(q') − s_i(q)| = |⟨q' − q, x_i⟩| ≤ ‖q' − q‖ ‖x_i‖ = δ (Cauchy–Schwarz).
Dla i z top-k(q) i j spoza niego: s_i(q') ≥ s_i(q) − δ ≥ s_(k) − δ > s_(k+1) + δ ≥ s_j(q) + δ ≥ s_j(q'). Każdy element
top-k(q) wyprzedza więc przy q' każdy element spoza niego, a obu zbiorów jest po k. ∎

**Wniosek 1 (ile klatek może się zmienić).** Bez założenia o przerwie: klatka może wejść do wyboru lub z niego wypaść
tylko wtedy, gdy jej ocena dla q leży w pasie [s_(k) − 2δ, s_(k+1) + 2δ]. Liczba zamian jest ograniczona liczbą
klatek w tym pasie. Przy małym δ i gęstej puli (9 136 klatek, oceny co 0,0005–0,02) zamian jest niewiele.

**Wniosek 2 (co faktycznie mierzy ramię „dobrane”).** Gdy δ jest małe, ramię koopman_fastgan jest (z dokładnością
do zamian z Wniosku 1) ramieniem „sąsiedzi obrazu obecnego latentu kotwicy”. Jego efekt w treningu nie może więc
zależeć od prognozy Koopmana bardziej, niż pozwala na to kilka zamienionych klatek.

## 3. Dlaczego trudne klatki uczą, a sąsiedzi kotwicy nie (ten sam obraz Langevina, strona wag)

SGD na wagach θ to też równanie Langevina: θ_{t+1} = θ_t − η ∇L_B(θ_t) + szum. Dodanie zbioru S zmienia średni dryf
o −η ∇L_S. Zmiana straty na obszarze docelowym R (np. top-134 val) w pierwszym rzędzie:

    ΔL_R ≈ −η ⟨∇L_R, ∇L_S⟩

Klatki, które detektor już dobrze rozpoznaje, mają małe ∇L_S, więc prawie nie przesuwają wag. Klatki trudne mają
duże ∇L_S; pomagają, gdy ich błędy są podobne do błędów w R. To przybliżenie pierwszego rzędu, nie dowód dla całego
uczenia; daje kierunek i znak, nie wielkość efektu.

## 4. Konfrontacja z danymi, które już mamy

| Przewidywanie | Pomiar | Zgodne? |
|---|---|---|
| Małe d_h po kroku 245 ⇒ obrazy G prawie równe | różnica 0,1–0,9 procent jasności (sekcja 6) | tak |
| Małe δ ⇒ prawie ten sam wybór (Tw. 1, Wn. 1) | top-20: wspólne 19/20, 15/20, 20/20 klatek (54/60) dla trzech kotwic | tak |
| Sąsiedzi kotwicy mają małe ∇L_S (łatwe) ⇒ mały efekt | AP na dodanych 0,857 na starcie, około 0,90 bez ich oglądania (7e); 1 epoka: +0,0007 (7c) | tak |
| Trudne klatki (duże ∇L_S) ⇒ efekt dodatni tam, gdzie ich błędy są podobne do błędów w R | 5 epok (7f): dolna połowa val +0,009, przy kotwicach −0,018 wobec losowych | tak: znak zależy od ⟨∇L_R, ∇L_S⟩, nie od samej trudności |
| Dobrane ≈ sąsiedzi kotwicy ⇒ efekt jak z łatwych klatek | 5 epok (7f): dobrane −0,0005 globalnie, lokalnie wobec losowych +0,006, −0,005, −0,013 | tak |

## 5. Przewidywania do sprawdzenia (falsyfikowalne)

P1. Ablacja „wyszukaj po samym obrazie obecnym kotwicy” da zbiór klatek pokrywający się z ramieniem dobranym w co
najmniej 90 procentach i efekt treningowy w granicach szumu ziaren od ramienia dobranego.

P2. Prognoza zacznie zmieniać wybór dopiero tam, gdzie ‖d_h‖ jest duże: na wczesnych punktach kontrolnych (przed
krokiem około 200, gdy dryf jeszcze trwa) albo dla celu o dużej amplitudzie (kierunek ku błędom detektora, różnice
między obrazami). Miernik bez treningu: δ i liczba zamian w top-k (Wn. 1) jako funkcja kroku startowego.

P3. Cel zbudowany z kierunku straty (klatki o dużym ∇L_S podobne do błędów w R) pobije klatki losowe na top-134 val;
cel z prognozy dryfu po kroku 245 nie pobije.

## 6. Koopman czy Langevin

Koopman to tylko część deterministyczna A. Langevin dodaje szum σ i daje właściwą miarę: stosunek ‖d_h‖ do szumu
prognozy i do rozdzielczości wyszukiwarki. Porażka nie wynika z tego, że operator liniowy źle przewiduje (sekcja 5:
przewiduje lepiej niż brak zmian), tylko z tego, że po kroku 245 przewidywana zmiana jest za mała, by zmienić wybór
klatek. Twierdzenie 1 da się sformalizować w Lean (tylko nierówność Cauchy'ego–Schwarza i porządek na liczbach).
