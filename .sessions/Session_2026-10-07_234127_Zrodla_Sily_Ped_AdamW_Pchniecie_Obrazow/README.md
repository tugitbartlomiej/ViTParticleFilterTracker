# Sesja 2026-10-07 23:41: źródła siły, pęd AdamW i pchnięcie obrazów

## Cel sesji
Dalsza rozbudowa aplikacji „Edytor latentu” (zakładka „Uczenie ViT”) i zrozumienie, skąd biorą się siły działające
na cząstki (tokeny) małego ViT. Trzy kierunki: wygoda i czytelność interfejsu; ścisły rozkład wektora patcha na
przepływ sygnału przez sieć; równanie ruchu cząstek w uczeniu (pęd AdamW, pchnięcie pojedynczych obrazów) połączone
z wcześniejszym łańcuchem użytkownika „sztuczny latent -> SD -> prawdziwe obrazy (MSE, DINO, konwerter)”.

## Typ
experiment

## Workerzy użyci
- Jeden podagent Explore (przegląd lokalnych eksperymentów SD, DINO, konwertera, odpowiedzi na paczkę; tylko odczyt).
- Dwa przeglądy literatury w Undermind (pęd optymalizatora i rozkład pchnięcia na obrazy; obserwatory z nieznanym
  wejściem i wspólne mody).

## Co zrobiono (skrót)
- Uczenie ViT: zaznaczanie wielu patchy i śledzenie tylko ich, zapis policzonej sekwencji i autostart, ramki paneli
  (odpinanie, chowanie), równanie krok po kroku (uczciwe dopasowanie kroczące), czytelniejsze ścieżki Langevina,
  ustawienia jako strony „Opcje”, środek makro „zero latentu”, domyślne ustawienia makro, naprawa maksymalizacji okna.
- Okno „Źródła siły”: ścisły rozkład wektora patcha na łodygę, położenie, MLP bloków, głowy atencji (animacja przepływu).
- Bezwładność cząstek to pęd AdamW (siła pędu = 0,83-0,85 poprzedniej prędkości); paczki odtwarzane z ziarna.
- Zakładka „Pchnięcie obrazów”: dokładne pchnięcie jednego obrazu na cząstki sondy (krok 98 %, suma obrazów 99 %),
  kandydaci SD, DINO, konwerter; pierwszy pomiar w kroku 500.
Szczegóły: `SESSION_SUMMARY.md`.
