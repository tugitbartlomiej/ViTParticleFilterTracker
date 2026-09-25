# Dobór klatek do obrazów przewidzianego latentu i kontynuacja treningu (134 klatki)

Data: 2026-09-25, wieczór. Eksperyment `SmallViTAblation/experiments/koopman_tozsamosc_kierunki`, sekcja 7 w `WYNIKI.md`.

Pytanie użytkownika: czy klatki dobrane do obrazów FastGAN przewidzianego latentu (Koopman od kroku 245) poprawiają
detekcję narzędzia po dołożeniu ich do zbioru 1500 klatek i jednej epoce treningu.

Odpowiedź: nie pogarszają i nie poprawiają ponad losowe. 134 dobrane klatki dają +0,0007 mAP@0,5, 134 losowe +0,0009,
oba w rozrzucie ziaren (SD około 0,001). Szczegóły: [SESSION_SUMMARY.md](SESSION_SUMMARY.md).
