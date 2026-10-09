# Sesja 2026-10-08/09: regresja szumu startowego SD, 150 klatek, podmiana w treningu K1000

## Cel sesji

Znaleźć szum startowy Stable Diffusion, z którego SD rysuje obraz o latencie ViT idącym w stronę sztucznego latentu
(zdjęcie celu 0055/4019), użyć tego obrazu do wyszukania prawdziwych klatek TRAIN (DINO) i sprawdzić treningiem ViT,
czy podmiana 150 klatek bazy przesuwa latent celu w stronę sztucznego.

## Typ

experiment

## Workerzy uzyci

- agent tła: regresja z 5 krokami DDIM (przerwany przez użytkownika po 9 krokach);
- agent tła: Eden, kontynuacja K1000, ramiona C0 i A150 (zadanie 1824868);
- agent tła: Eden, kontrola R150 i 3 ziarna (zadanie 1824901, w toku przy zapisie);
- agent tła: przygotowanie 1500 startów regresji w K1000 (zatrzymany na warunku wstępnym, bez generacji).

## Co zrobiono (skrót)

- Regresja szumu startowego (bez gradientu przez SD): 10,2 do 17,2 procent postępu w 20 krokach, ale w 50 krokach DDIM tylko 6,0.
- 10 losowych startów po 5 kroków z większym krokiem: średnio 4,3 do 11,7 procent, najlepszy 16,9 (13,9 w 50 krokach).
- 150 klatek TRAIN najbliższych w DINO do klatki 0088/1661 (143 z jednego filmu) zapisane do folderu.
- Eden, kontynuacja od K1000 z AdamW, 5 epok: podmiana 150 klatek mocno zmienia latent celu, ale prawie nie w stronę z* (kosinus od -0,06 do 0,11).
- Naturalna detekcja K1000 na 1500 obrazach bazy: mediana pewności 0,77, 74 procent trafień; na zdjęciu celu pewność 0,51 przy IoU 0.

Szczegóły: [SESSION_SUMMARY.md](SESSION_SUMMARY.md).
