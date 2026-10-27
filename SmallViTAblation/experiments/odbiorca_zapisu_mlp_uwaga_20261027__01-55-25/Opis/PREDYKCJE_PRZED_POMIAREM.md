# Predykcje zapisane przed pomiarem (2026-10-27, 02:05, Europe/Warsaw)

Zapisane przed uruchomieniem jakiegokolwiek zadania tego eksperymentu na Edenie i przed testem lokalnym kodu. Progi nie będą zmieniane
po pomiarze; uwagi po fakcie będą oznaczone „(po fakcie)”. Miary zdefiniowane w `Kod_eksperymentu/odbiorca_eden.py` (docstring); ocena
w `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py` (napisana przed odczytem wyników).

## Pytanie

Rodzic `balanced_amplification_schur_odczyt_20261027__00-31-41` (wersja 2): kierunek, w którym MLP **pisze** (u₁, lewy wektor osobliwy
jakobianu pola J = W₂·diag(GELU′)·W₁·J_LN2), nie leży w podprzestrzeni odczytu głowy (detekcja: energia 0,02–0,04 przy losowej 0,021)
i nie jest czytany przez MLP następnego bloku (cos²(u₁ᵇ, v₁ᵇ⁺¹) 0,008–0,02 przy losowym 0,0052). **Kto czyta to, co pisze wzmacniany
mod MLP?**

- **H1 (główna):** wejście uwagi kolejnych bloków (q/k/v po LN1), a w szczególności tor routingu (q, k), nie tor treści (v).
- **H2:** uwaga czyta u₁ przez v (treść), nie przez q/k.
- **H3:** nikt nie czyta — u₁ rozprasza się w kierunkach o małej wrażliwości reszty sieci (zapis „do kosza”).
- (alternatywa poza H1–H3, sprawdzana tą samą miarą) MLP bloku b+1 czyta u₁ w swojej pełnej macierzy odczytu W₁·J_LN2, choć nie
  w jego pojedynczym kierunku v₁ᵇ⁺¹.

## Biegi, statystyka, próg

5 biegów: detekcja s3047 / s4047 / s5047 (k0, 3000, 65 640), MałyViT CIFAR s47 i DeiT CIFAR s47 (k0, 3000, 93 780). Tokeny jak
w rodzicu (detekcja: wszystkie narzędzia + 20 tła na klatkę diagnostyczną, 50 klatek; CIFAR: 16 tokenów obrazu, 50 obrazów); pomiary
nieliniowe (B), odczyt głowy i JVP (C) na 8 tokenach na klatkę.

Statystyka M[·] (o ile nie napisano inaczej): **mediana po blokach b = 0…NB−2 z median po tokenach i klatkach w bloku b**, dla
odbiorcy c = b+1, na ostatnim checkpoincie (koniec). Kierunki d: u₁ᵇ (zapis bloku b), v₁ᵇ (odczyt bloku b), losowy (jednostkowy,
nowy dla każdego tokenu i bloku), v₁ᵇ⁺¹ (odczyt MLP bloku b+1, odniesienie „czytane na pewno przez MLP b+1”).

Względny zysk odczytu g_X(d) = ‖R_X d‖² / mediana ‖R_X r‖² po 64 losowych kierunkach jednostkowych r (ten sam token; kierunek losowy
daje g ≈ 1). Energia E_X(d) = ‖P₁₆ d‖², P₁₆ = rzut na 16 wiodących prawych wektorów osobliwych R_X (losowe 16/192 = 0,083).
R_q, R_k, R_v = W_q/k/v·J_LN1(x_c) (192 × 192), R_att = [R_q; R_k; R_v] (576 × 192), R_m = W₁·J_LN2(x_c po uwadze) (768 × 192).
g_qk(d) = max(M[g_q(d)], M[g_k(d)]).

**Próg biegów:** PASS, gdy warunek zachodzi w ≥ 4/5 biegów (jednostronny test znaku: 5/5 → p = 0,031, oznaczane „silnie”;
4/5 → p = 0,19, oznaczane „słabo”). Kontrola K1 wymaga 5/5.

## Predykcje

| nr | predykcja | warunek liczbowy (koniec, M[·], c = b+1) | próg |
|---|---|---|---|
| O1 | uwaga czyta zapis (q lub k) | g_qk(u₁) > 3 × g_qk(los) **i** g_qk(u₁) na końcu > g_qk(u₁) przy k0 | ≥ 4/5 |
| O2 | czyta go tor routingu, nie treści | g_qk(u₁) > M[g_v(u₁)] | ≥ 4/5 |
| O3 | uwaga czyta u₁ bardziej niż MLP następnego bloku (ta sama miara, k = 16) | M[E_att(u₁)] > M[E_m(u₁)] | ≥ 4/5 |
| O4 | nieliniowo: zapis u₁ zmienia wyjście uwagi bloku b+1 **innych tokenów** (sygnatura toru k/v) | mediana_b ( med ‖Δattn_inne(u₁)‖ / med ‖Δattn_inne(los)‖ ) ≥ 2 przy δ = 0,03 | ≥ 4/5 |
| O5 | kontr-test H3: zapis u₁ nie trafia „do kosza” | mediana_b ( med ‖J_reszty u₁‖ / med ‖J_reszty los‖ ) ≥ 1 (JVP wyjścia całej sieci od stanu po bloku b) | ≥ 4/5 |
| O6 | zapis jest czytany przez uwagę także dwa bloki dalej (c = b+2) | g_qk(u₁) dla c = b+2 > 3 × g_qk(los) dla c = b+2 | ≥ 4/5 |
| O7 | nieliniowo: u₁ przesuwa równowagę uwagi ku routingowi | [max(med ‖Δattn_q‖, med ‖Δattn_k‖) / med ‖Δattn_v‖](u₁) > to samo dla kierunku losowego, δ = 0,03 (Δattn_X = zmiana wyjścia uwagi, gdy zaburzenie wchodzi tylko przez X; wszystkie tokeny) | ≥ 4/5 |
| O8 | (mechanizm H3) zapis częściowo radialny / w średniej — ślepy dla LayerNorm każdego odbiorcy | M[E_null(u₁)] > 3 × 2/192, E_null = energia u₁ w jądrze J_LN1(x_c) (span{1, x̂_c}) | ≥ 4/5 (opisowo) |
| K1 | kontrola czułości miary: MLP b+1 czyta swój kierunek v₁ᵇ⁺¹ | M[E_m(v₁ᵇ⁺¹)] > 2 × 0,083 | 5/5 |

Obok każdej wartości zostanie podana wartość dla kierunku losowego i dla v₁ᵇ⁺¹, k0 i checkpoint wczesny (3000), osobno detekcja i CIFAR.
Odczyt głowy (energia u₁ w podprzestrzeni gradientów wszystkich wyjść, jak w rodzicu) i g_m(u₁) podane opisowo.

## Reguła rozstrzygania (zapisana z góry)

- **H1 potwierdzona:** O1 ∧ O2 ∧ O4 PASS (O3 i O7 wspierają). Zdanie do artykułu: MLP pisze wzmocniony kierunek do współrzędnych,
  które uwaga następnego bloku czyta przez q/k (routing) — struktura „MLP → routing uwagi”, nie łańcuch MLP → MLP.
- **H2:** M[g_v(u₁)] > 3 × M[g_v(los)] ∧ O2 FAIL (g_v ≥ g_qk).
- **H3:** O5 FAIL (wrażliwość wzdłuż u₁ < losowej) ∧ O1 FAIL ∧ M[g_v(u₁)] ≤ 3 × losowe ∧ M[g_m(u₁)] ≤ 3 × losowe. Silna wersja:
  stosunek w O5 < 0,5. Jeżeli przy tym O8 PASS → „kosz” to kierunki ślepe dla LayerNorm.
- **MLP b+1 czyta (alternatywa):** M[g_m(u₁)] > 3 ∧ O3 FAIL.
- **Odbiorca dalej / pośrednio:** O5 PASS przy O1, O3 FAIL → zapis wpływa na wyjście, ale nie przez blok b+1 (sprawdzić O6 i odczyt
  głowy).
- Wynik może się różnić między detekcją a CIFAR (rodzic: w CIFAR u₁ leży w odczycie logitów); wtedy wniosek osobno dla każdego typu
  odczytu, bez uogólniania.

## Uwagi

- g_X jest miarą względną do losowego kierunku w tym samym tokenie, więc porównywalną między q, k, v i MLP mimo różnych norm wag.
  Energia w top-16 jest drugą, niezależną od skali miarą; dla R_att (576 wierszy) i R_m (768 wierszy) k = 16 jest tym samym progiem.
- J_LN ma jądro span{1, x̂}: kierunek radialny jest niewidoczny dla każdego odbiorcy po LayerNorm, ale zmienia normę tokenu (a przez
  nią skalę wszystkich kolejnych LN). O8 mierzy, jaka część u₁ tam leży.
- Zaburzenie nieliniowe przykładane do stanu tuż po MLP bloku b (wejście bloku b+1), δ ∈ {0,03; 0,1} × mediana normy tokenu w tej
  klatce; jeden token na kopię, więc zmiana innych tokenów pochodzi wyłącznie z uwagi (k/v tego tokenu).
