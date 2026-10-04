# Sesja 2026-10-02 .. 2026-10-05: sufit przewidywalności latentu, równanie z pędem, paczki 128 i 320, raport 27

## Cel sesji

Znaleźć równanie przewidujące latent małego ViT w trakcie uczenia z wynikiem powyżej 85 procent: najpierw na
100 kroków, potem (po decyzji użytkownika) na 10 kroków, czyli dwa checkpointy. Zrozumieć, dlaczego NNDMD na
checkpointach 90 do 100 przegrał z prognozą bez zmian.

## Typ

experiment

## Workerzy użyci

- Agent Explore: przegląd w repo, jak wcześniej liczono kierunek LDA i prognozę rzutu (docs/23, 24, 26, WYNIKI).
- Undermind: nowe głębokie wyszukiwanie literatury (workspace grantowy, folder „02 Mori Zwanzig i Langevin”).

## Co zrobiono (skrót)

- Sufit przewidywalności na 100 kroków z gęstej historii (1001 kroków, 12 obrazów): wyrocznia 0,84–0,89,
  prognoza z historii 0,16–0,35. Ruch krąży wokół wolnego poziomu i zawraca.
- Prognoza od checkpointu 90 do 100 krok po kroku z wykresem: równanie OU 0,20–0,30, NNDMD ujemny na 10 checkpointach.
- Równanie Langevina na 1..10 kroków: historia + pęd AdamW 0,68 na 1 krok, 0,22 na 10. Z pełną siłą kroku 0,985.
- Dwa nowe biegi na Edenie (paczka 128 i 320): drgania bez zmian, prognoza na 10 kroków 0,24. Hipoteza
  „większa paczka uspokoi ruch” obalona; drgania tworzy AdamW.
- Raport PDF z architekturą, schematami i 8 rysunkami: `SmallViTAblation\docs\27_PRZEWIDYWANIE_LATENTU_RAPORT.pdf`.

Szczegóły: [SESSION_SUMMARY.md](SESSION_SUMMARY.md).
