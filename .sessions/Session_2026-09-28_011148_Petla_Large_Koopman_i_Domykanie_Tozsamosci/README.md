# Pętla Koopman + FastGAN na RF-DETR LargeDeprecated i definicja domknięcia latentu tożsamości

Data: 2026-09-27 (15:00) – 2026-09-28 (01:15). Eksperymenty: `SmallViTAblation/experiments/petla_od_poczatku_large`
(wyniki: `WYNIKI.md`) i `SmallViTAblation/experiments/domykanie_tozsamosci_large` (projekt: `PLAN.md`, czeka na zgodę).

Pytanie użytkownika: czy pętla Koopmana z FastGAN, uruchomiona od początku dostrajania na większym modelu
(DINOv2-B), przyspiesza uczenie detektora.

Odpowiedź: nie. Kryterium z planu niespełnione w obu częściach; losowe klatki uczą lepiej niż wybrane pętlą
(koopman − losowe −0,0030 mAP@0,5, ujemne w 15 z 18 par). Latent tożsamości staje po dwóch obrotach, ale
niedomknięty: obrót narzędzia i kolor tła zmieniają kod bardziej niż różnica dwóch obrazów (R ≈ 0,78). Na tej
podstawie powstała definicja domknięcia i plan nowego doboru klatek. Szczegóły: [SESSION_SUMMARY.md](SESSION_SUMMARY.md).
