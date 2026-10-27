# Predykcje zapisane przed pomiarem (2026-10-26, 21:25, Europe/Warsaw)

Zapisane przed uruchomieniem jakiegokolwiek zadania tego eksperymentu na Edenie. Progi nie będą zmieniane po pomiarze; uwagi po
fakcie będą oznaczone. Pytanie: **co z nienormalności pola MLP wynika dla działania sieci?** Rodzice (`mosty_fizyki_24seedy_ablacje`,
`kompensacja_lodyga_i_cifar`) wykazały, że ‖N‖/‖J‖ rośnie z 0,76 do 0,92 we wszystkich trzech układach; nie wykazały, czy to ma
skutek funkcjonalny. Miary zdefiniowane w `Kod_eksperymentu/wzmocnienie_eden.py` (docstring); ocena w
`Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py`.

Biegi: detekcja seedy 3047, 4047, 5047 (checkpointy 0, 3000, 65 640), MałyViT CIFAR s47 i DeiT-Tiny CIFAR s47 (0, 3000, 93 780);
razem 5 biegów, test znaku 5/5. „Koniec” = ostatni checkpoint, „k0” = wagi losowe. Blok środkowy b* = NB // 2 + 1 (4 w MałymViT,
7 w DeiT). Mediany po tokenach (detekcja: narzędzie i tło razem) i klatkach / obrazach. A = I + J, ρ = max|1 + λ|.

## W: widmo i normy (liniowe)

| nr | predykcja | warunek liczbowy (mediana tokenów bloku b*) | próg |
|---|---|---|---|
| W1 | trening zwiększa wzmocnienie przejściowe jednego półkroku | G₁ = ‖A‖₂/ρ: koniec > k0 **i** koniec > 1,5 | 5/5 |
| W2 | wzrost przejściowy jest istotny, nie marginalny | max_k G_k (k ≤ 6) na końcu ≥ 2 | 5/5 |
| W3 | stała Kreissa rośnie w treningu | K: koniec > k0 | 5/5 |
| W4 | luka abscysy numerycznej rośnie | (ω − max Re λ): koniec > k0 | 5/5 |
| W5 | złożenie bloków wzmacnia bardziej niż pojedynczy blok | G_P = ‖A_NB⋯A_1‖₂ / Π ρ_b (mediana tokenów) > mediana G₁ po blokach, na końcu | 5/5 |
| W6 | zero kontrolne: przy k0 nienormalność jest losowa, więc G₁(k0) ≈ wartość dla macierzy Ginibre'a | G₁(k0) ∈ [1,1; 1,6] | 5/5 |

Uwaga do W6: dla losowego J (‖N‖/‖J‖ 0,71) lokalny test dał G₁ = 1,33, K = 1,75, G₆ = 2,4 — nienormalność losowa też daje wzrost
przejściowy, więc W1–W3 pytają o **przyrost** w treningu, a nie o samo istnienie wzrostu.

## F: zaburzenia (nieliniowe, mierzone na sieci)

| nr | predykcja | warunek liczbowy (koniec, blok b*) | próg |
|---|---|---|---|
| F1 | wzmocnienie losowego zaburzenia zgadza się z przewidywaniem nienormalnym, nie normalnym | δ = 0,01, kierunek losowy: \|amp/pred_nn − 1\| < 0,10 **i** pred_nn / pred_norm > 1,15 | 5/5 |
| F2 | linearyzacja jest wierna w najgorszym kierunku | δ = 0,01, najgorszy kierunek: \|amp/σ_max − 1\| < 0,20 | 5/5 |
| F3 | najgorszy kierunek szkodzi wyjściu bardziej niż losowy | δ = 0,03: mediana \|Δwyjścia\| najgorszy > 2 × mediana \|Δwyjścia\| losowy | 5/5 |
| F4 | blok 1 jest najwrażliwszy (krytyczność bloku 1 z rodziców) | δ = 0,1, losowy: argmax po blokach mediany \|Δwyjścia\| = 1 | 5/5 |
| F5 | wrażliwość wyjścia rośnie w treningu razem z nienormalnością | δ = 0,03, najgorszy, blok b*: mediana \|Δwyjścia\| koniec > k3000 | 5/5 |

Δwyjścia: detekcja = ΔIoU ramki szczytu; CIFAR = Δmargines logitów (klasa prawdziwa − najlepsza inna). W F5 porównanie z k3000, nie
z k0, bo przy k0 wyjście jest bez znaczenia (AP = 0, dokładność 0,1) i Δ nie ma sensu.

## Co uznam za odpowiedź

- „Nienormalność ma skutek”: W1–W3 i F1, F3. Wtedy zdanie do artykułu: trening zwiększa wzmocnienie przejściowe pola MLP, zaburzenia
  tokenów są wzmacniane bardziej, niż mówi widmo, a najgorsze kierunki mają mierzalny skutek dla wyjścia.
- „Nienormalność jest tylko statystyką”: W1 lub F1 nie zachodzi. Wtedy ‖N‖/‖J‖ zostaje w artykule jako opis, bez interpretacji
  dynamicznej, a mosty do pseudospektrów idą do dyskusji jako niepotwierdzone.
- F2 to kontrola linearyzacji; jej niezajście unieważnia F1 i F3 jako test (nie tezę).
