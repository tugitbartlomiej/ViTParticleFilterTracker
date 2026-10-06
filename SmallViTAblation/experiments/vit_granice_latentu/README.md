# Dokładne granice wartości latentu małego ViT

Latent to aktywacje ukrytej warstwy MLP ostatniego bloku: h = GELU(W1 · LN(x) + c), W1: 768 x 192.

## Wyprowadzenie

LayerNorm: u = (x - średnia(x)) / sqrt(wariancja(x) + eps). Dla każdego wejścia x:
suma(u) = 0 oraz ||u|| = sqrt(d) · sqrt(var / (var + eps)) <= sqrt(d), d = 192.
Wyjście LN: g ⊙ u + b. Pobudzenie neuronu j: a_j = (W1_j ⊙ g) · u + (W1_j · b + c_j).

Maksimum iloczynu skalarnego po u z tej sfery w hiperpłaszczyźnie suma(u) = 0 to sqrt(d) · ||P(W1_j ⊙ g)||,
gdzie P odejmuje średnią składowych. Stąd dokładny przedział pobudzenia:

a_j ∈ [m_j - r_j, m_j + r_j], m_j = W1_j · b + c_j, r_j = sqrt(d) · ||P(W1_j ⊙ g)||.

GELU rośnie poza jedynym minimum -0,16997 w a = -0,7518, więc:
h_j max = GELU(m_j + r_j), h_j min = -0,16997 (o ile przedział zawiera -0,7518, co zachodzi dla wszystkich 768 kanałów).

Granica jest ścisła: dla dowolnego u z tej sfery istnieje x dające to u, więc wartość h_j max jest osiągalna
dla pewnego wektora wejściowego bloku. Nie musi być osiągalna dla prawdziwego zdjęcia (to ograniczenie
od góry, nie zakres naturalny). Kanały nie są niezależne: wszystkie 768 wartości jednego tokenu pochodzą
z tego samego u, więc nie da się jednocześnie ustawić wszystkich na maksimum.

Granice zależą tylko od wag g, b (norm2) i W1, c (fc1) ostatniego bloku. W trakcie uczenia zmieniają się
razem z wagami, więc liczy się je osobno dla każdego checkpointu: `python granice.py <checkpoint.pt>`.

## Wynik dla MODEL_VIT_K1000.pt (2026-10-06)

`outputs/vit_granice_latentu/GRANICE_MODEL_VIT_K1000.{json,npz}` (h_min, h_max, a_min, a_max dla 768 kanałów).

- h max: od 7,02 do 10,24 zależnie od kanału, mediana 7,94; h min: -0,17 dla każdego kanału.
- Naturalnie (400 zdjęć) największa wartość to 4,43, czyli zdjęcia wykorzystują około połowy zakresu.
- Sztuczny latent J128: 15 661 wartości (3,4 procent) w 85 kanałach przekracza granicę, najwięcej o 42,9
  (wartość 51,4 przy granicy około 8,5). Takiego latentu nie da żadne wejście do ostatniego bloku, nie tylko
  żadne zdjęcie.
