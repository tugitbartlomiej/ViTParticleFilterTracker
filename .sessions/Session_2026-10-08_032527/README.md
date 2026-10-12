# Session: Session_2026-10-08_032527

**Created:** 2026-10-08 03:25:27 Europe/Warsaw
**Type:** Mixed — Analysis / Benchmark
**Status:** Completed

## Quick Summary

Badanie semantyki ruchu tokenów SmallViT: kierunki związane z pozycją, kolorem i proporcjami narzędzia, ich zmiana podczas uczenia oraz lokalne interwencje w ostatniej gałęzi MLP. Wykonano dwa eksperymenty; drugi obejmuje 176 kontrolowanych obrazów, 11 kontroli koloru/jasności, cztery rzeczywiście wygenerowane próbki FastGAN i sprawdzenie sond na naturalnych klatkach.

## Key Results

- Pierwsze sondy miały silnie nakładające się podprzestrzenie i słaby związek ruchu z przyrostem IoU.
- Kontrolowane obrazy poprawiły rozróżnienie koloru i pozycji. Proporcje nadal nie są czystą sondą kształtu.
- Bardziej selektywne kierunki kontrastowe obejmują niewielką energię odpowiedzi/ruchu i mają mały średni efekt interwencji na IoU.
- FastGAN EMA po 50 tysiącach kroków działa. SD nie uruchomiono ze względu na zajęte zasoby; drugą analizę wykonano na CPU.
- Kontrole obliczeń obu eksperymentów przeszły. Jeden przebieg treningu i mała próba nie uzasadniają jeszcze twierdzeń publikacyjnych.

## Files

- `SESSION_SUMMARY.md` — kontekst, dokładne wyniki, ścieżki i sposób kontynuacji.
- `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_semantyka_ruchu_20261008/pilot_v1/REPORT.md`
- `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_kontrolowane_obrazy_20261008/pilot_v1/REPORT.md`
