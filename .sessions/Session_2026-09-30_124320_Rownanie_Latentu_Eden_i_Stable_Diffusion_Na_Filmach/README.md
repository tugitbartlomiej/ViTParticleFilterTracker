# Równanie latentu małego ViT na Edenie i Stable Diffusion uczone na całych filmach

Data: 2026-09-29 (14:52) - 2026-09-30 (12:43). Kontynuacja sesji
`Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek`.

## Cel sesji

1. Przerobić równanie Langevina z TFT na równanie ruchu latentu małego ViT w trakcie uczenia, ze sterowaniem
   danymi, i dojść do co najmniej 80 procent trafności na jeden krok naprzód; potwierdzić to na 10 ziarnach Edenu.
2. Sprawdzić, czy uczone duże macierze jak w TS (jądro pamięci, wymuszenie, dryf, szum) poprawiają prognozę,
   przy regule użytkownika: do prognozy nie wchodzą przyszłe klatki.
3. Posprzątać Eden, wysłać tam całe filmy Cataract-LMM z obrysami i uczyć na nich Stable Diffusion 1.5 na wielu
   kartach.

## Typ

experiment

## Workerzy użyci

Brak - sesja solo.

## Co zrobiono (skrót)

- Horyzont prognozy siły tożsamości: Koopman i Langevin biją „bez zmiany” do 1000 kroków, ale tylko o 10-25 procent
  błędu kwadratowego (`maly_vit_horyzont_prognozy`).
- Literatura o dyfuzji trafiającej w zadany latent: 10 prac przeczytanych w całości przez Undermind.
- Równanie latentu w trybie wyjaśnienia `Δz(t+1) = 0,83 Δz(t) + 0,98 S(t)`: mediana 85 procent na 10 ziarnach Edenu,
  każde powyżej 80, placebo spada do poziomu bez sterowania. Siła S(t) pochodzi z bieżącej paczki, co użytkownik
  uznał za dozwolone wyjaśnienie, ale nie prognozę.
- Prognoza bez przyszłych klatek: 64-67 procent (siła z samego pędu optymalizatora), 57 procent (sama bezwładność).
  Uczone macierze jak w TS (11 mln parametrów) dają około 40 procent, bo uczą się okna na pamięć.
- Metoda zapisana w `SmallViTAblation\docs\23_METODA_PRZEWIDYWANIA_LATENTU.md`.
- Eden: sprzątanie 117 -> 57 GiB za zgodą użytkownika, wysyłka całego `3_Object_Tracking` Cataract-LMM (49 GB,
  sumy SHA256 zgodne).
- Stable Diffusion 1.5 z maskami narzędzia, rogówki i źrenicy jako warunkiem (`dyfuzja_sd_filmy`): latenty ze
  wszystkich klatek 118 filmów treningowych, uczenie 1818496 na 4 x H100 (209 obrazów/s), w chwili zapisu krok
  5050 z 20 000, strata 0,107, podgląd po 4000 krokach realistyczny.

Pełny opis: [SESSION_SUMMARY.md](SESSION_SUMMARY.md).
