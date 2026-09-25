# Równanie Langevina dla latentu detektora (dryf Koopmana, pamięć, wymuszenie, sterowanie, szum)

Data: 2026-09-26, noc. Zlecenie użytkownika: „Koopman przewiduje latent i tyle, ma to robić dobrze. Bierz moje
równanie Langevina z projektu (dryf, pamięć, wymuszenie, sterowanie, szum) i dostosuj. Dryf można liczyć Koopmanem.”
Źródło równania: `F:\Studia\MCHTR\Grant\Projekt\Research\syntheses\schemat-diagnostyki-latentu-koopman-langevin-2026-09-18.md`
(część A3 i C, zegar uczenia), cztery rozdzielne, wyłączalne człony plus sterowanie jako osobny człon.

## Równanie (zegar uczenia: cząstka = skalar tożsamości narzędzia jednego obrazu sondy, krok = 5 aktualizacji)

    v̂_{τ+1|τ} = D(a_τ, v_τ) + JP(v_{τ-1}, …, v_{τ-L+1}) + W(f_τ) + S(u_τ) + N(r_τ, …, r_{τ-q+1})
    a_{τ+1} = a_τ + v_{τ+1},   v_{τ+1} = v̂_{τ+1|τ} + e_{τ+1}

- `a_τ` skalar tożsamości (sekcja 1 `koopman_tozsamosc_kierunki`), `v_τ = a_τ − a_{τ−1}`.
- D, dryf: Koopman z wejściem na podniesionym stanie [v_τ, a_τ − ā_τ, 1] (ā_τ = średnia po obrazach fit w τ).
- JP, jądro pamięci: starsze przyrosty, L − 1 opóźnień (Mori-Zwanzig).
- W, wymuszenie: znany harmonogram uczenia: granica epoki (przetasowanie danych co 100 aktualizacji), faza
  epoki (sin, cos), 1/krok. Tempo uczenia w biegu 1810662 stałe.
- S, sterowanie: skład danych uczących (co dodano). W biegu 1810662 stały, więc nieidentyfikowalny; etap 2 na
  12 biegach kampanii 5 epok (te same sondy, ten sam start, różne dodane klatki).
- N, szum kolorowy: AR(q) na dawnych resztach jednokrokowych; e_{τ+1} biała innowacja (część nieusuwalna, A4).

Trzy fazy (reguła projektu): uczenie parametrów na przejściach ≤ S na 48 obrazach fit; przewidywanie z informacji
w τ; obserwacja: reszta z τ+1 aktualizuje bufor N, nigdy nie poprawia wstecz prognozy z τ.

## Protokół (jak w sekcji 2 i 5 `koopman_tozsamosc_kierunki`, bez zmian)

K = 160 (kroki 5..800), S = 48 (krok 245) i S = 96 (krok 485). Wybór L, grzbietu, q i członów na kalibracji:
ostatnie 20 przejść okna fit na obrazach fit. Test na 24 obrazach holdout: RMSE 1 kroku (przejścia S..K−2)
i rollout od S (h = 20 i do końca) względem braku zmian. Odniesienie: zapisany operator L4
(`OPERATORY_K160_S{S}.npz`). Ablacje: wyłączenie każdego członu osobno. Białość: autokorelacja reszt holdout
(opóźnienia 1–5) i korelacja z W.

## Bramka (ustalona przed wynikiem)

Równanie jest lepsze od L4, gdy na holdout RMSE rollout h = 20 ≤ 0,95 × L4 w obu oknach (S = 48 i S = 96) i RMSE
1 kroku nie gorsze od L4 o więcej niż 2 procent. Inaczej ogłaszamy L4 jako najlepszy dryf z pamięcią.

## Etap 2 (osobno, po zgodzie): sterowanie S(u)

Tokeny 72 sond z wag końca epok 12 biegów kampanii 5 epok (`koopman_tozsamosc_kierunki/wyniki/dobor_klatek/trening5ep/wagi`,
60 plików, ekstrakcja lokalna kilka minut GPU), różnice sparowane ramię − bez_dodania przy tym samym ziarnie jako
odpowiedź na u. Pozwala przewidzieć latent pod inny wybór danych, zanim się go trenuje.
