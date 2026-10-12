# Plan: wpływ pojedynczego naturalnego obrazu na dalszy trening ViT K8000

## Pytanie

Czy zastąpienie jednego elementu stałej puli treningowej naturalną klatką wyszukaną dla kolejnych obrazów SD zmienia detekcję narzędzia na obrazie m0011?

## Układ kontrolowany

- Punkt startowy każdego wariantu: ten sam checkpoint Małego ViT K8000, seed 47.
- Pula każdego wariantu: 255 identycznych obrazów wspólnych oraz jeden wymieniany obraz.
- Kontrola: w pozycji 256 pozostaje stały obraz kontrolny.
- Interwencje: w pozycji 256 umieszczany jest jeden z naturalnych odpowiedników top-1 DINO.
- Jedenaście kroków interpolacji dało osiem unikalnych obrazów, dlatego liczone jest osiem interwencji; wyniki duplikatów są mapowane do wszystkich odpowiadających im kroków.
- Każdy wariant dostaje 500 aktualizacji przy tej samej kolejności indeksów, augmentacjach i seedzie.
- Optymalizator jest świeży, ponieważ K8000 zachowano jako `state_dict` bez stanu AdamW. Używany jest mały LR 2e-5.

## Pomiar

Na m0011 co 25 aktualizacji:

- confidence propozycji o najwyższej pewności;
- IoU tej propozycji z GT;
- najlepsze IoU wśród dziesięciu propozycji;
- confidence propozycji o najlepszym IoU;
- iloczyn confidence i IoU.

Przed i po treningu liczony jest również AP50 na stałym zbiorze kontrolnym 512 obrazów z filmów niewykorzystywanych przy budowie wspólnej puli.

## Hipoteza

Jeżeli znalezione zdjęcie niesie użyteczną informację dla m0011, jego wariant powinien poprawić trajektorię IoU lub confidence względem kontroli, bez wyraźnego spadku AP50 na zbiorze kontrolnym.

## Ograniczenia

To test wpływu jednego obrazu przy konkretnym checkpoincie, seedzie i budżecie 500 kroków. Nie jest jeszcze dowodem generalizacji strategii wyboru danych.
