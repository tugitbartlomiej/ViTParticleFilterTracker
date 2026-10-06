# Sesja 2026-10-06 12:41:41: konwerter latentu ViT na cechy DINO, uczony na filmach na Edenie

## Cel sesji

Przygotować i puścić na Edenie (DGX, kilka GPU) uczenie konwertera, który z latentu małego ViT z checkpointu
k1000 przewiduje cechy DINOv2 tego samego zdjęcia. Klatki liczone w locie z filmów MP4, bez zapisu klatek.
Monitorować bieg i poprawiać.

## Typ

experiment

## Workerzy użyci

Brak - sesja solo.

## Co zrobiono (skrót)

- Nowy eksperyment `SmallViTAblation/experiments/vit_konwerter_dino_filmy_eden`: strumień MP4 z ZIP do RAM,
  zamrożony ViT i DINOv2 na GPU, konwerter transformer na 600 tokenach, DDP na 4 GPU.
- Lokalny test dymny na RTX 3070 i test dymny na Edenie w tym samym zadaniu przed uczeniem.
- v1 (1823170) przeuczyła się od kroku 6000, zatrzymana plikiem STOP po 10 120 krokach.
- v2 (1823174) z augmentacjami natywnej klatki i regularyzacją: 20 000 kroków, mediana pozycji 101 z 4435
  wobec 190 w kontroli liniowej, film trafiony w 81 procentach.
- Wyniki i `KONWERTER.pt` pobrane do `SmallViTAblation/outputs/vit_konwerter_dino_filmy_eden`.

Szczegóły: `SESSION_SUMMARY.md`.
