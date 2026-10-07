# Sesja 2026-10-07 04:32: zakładka „Uczenie ViT” i liniowy ruch cząstek

## Cel sesji
Animacja kroków uczenia ViT w aplikacji, przyczyny ruchu cząsteczek (wagi, gradient, atencja), potem liniowy
opis ruchu każdej cząstki w uczeniu (Koopman, Langevin, obserwator) i bieg sił na Edenie.

## Typ
experiment

## Workerzy użyci
Cztery podagenty ogólne (przegląd literatury o wizualizacji uczenia, inwentaryzacja danych historii uczenia,
przegląd wiedzy projektu o dynamice latentu, przegląd literatury o liniowej dynamice uczenia).

## Co zrobiono (skrót)
- Makro: cząstki w kolorach najsilniejszego mikrowektora (jak w 3D), commit cb867d0f.
- Plan `app/PLAN_ANIMACJA_UCZENIA.md` i zakładka „Uczenie ViT: ruch cząsteczek po krokach” (f9bb44df).
- Przyczyny ruchu: 13 grup wag, gradient straty, atencja, strzałki w makro (cbab2c0c).
- Eksperyment `experiments/maly_vit_liniowy_ruch_czastek`: etap 1 lokalnie, bieg sił na Edenie (69c1eee4).

Szczegóły: `SESSION_SUMMARY.md`.
