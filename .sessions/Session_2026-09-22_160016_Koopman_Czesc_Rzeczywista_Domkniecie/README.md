# Session 2026-09-22 16:00 - Czesc rzeczywista modow Koopmana, niezmienniki po czasie i teoria domkniecia stanu

## Cel sesji
Runda trzecia tego samego dnia. Trzy zadania od autora: policzyc niezmienniki Koopmana na WYCINKU z ramki
narzedzia klatka po klatce i porownac je z niezmiennikami liczonymi po przesunieciu; napisac analize TEORETYCZNA
tego, ile brakuje do domkniecia modelu narzedzia i czy da sie tym sterowac doborem klatek; sprawdzic hipoteze,
ze tozsamosc narzedzia siedzi w czesci rzeczywistej modow, w tym mod po modzie i z podzialem na czesc dodatnia
i ujemna. Praca merytoryczna zyje w repozytorium ViT; tutaj tylko zapis procesu.

## Typ
experiment

## Workerzy uzyci
Brak - sesja solo. (Podagent przegladowy dzialal w rundzie poprzedniej, opisanej w
`Session_2026-09-22_150300_kod_tozsamosci_narzedzia_widmo_i_slad`.)

## Co zrobiono (skrot)
- Niezmienniki liczone PO CZASIE na wycinku ramki: najtrwalsza wielkosc ma modul 0,989, ale przy potasowanym
  czasie wychodzi 0,966, wiec prawie cala trwalosc bierze sie z bezwladu latentu.
- Dwie drogi trafiaja w te same wektory latentu: kosinusy glowne 0,95, 0,90, 0,87 i 0,83 dla czterech pierwszych
  kierunkow; dalsze sie rozchodza.
- Dokument 42: analiza teoretyczna domkniecia przestrzeni stanow, trzy nierownowazne miary deficytu i wniosek,
  ze struktura niezmiennicza jest mocniejsza jako czlon straty niz jako regula doboru klatek.
- Z rachunku wyszla zlokalizowana dziura: zwrot narzedzia ginie w ESTYMATORZE (srednia po tokenach kasuje czesc
  nieparzysta), nie w latencie; poprawka to moment pierwszy wazony polozeniem wzdluz osi ramki.
- Hipoteza autora o czesci rzeczywistej: potwierdzona w wersji praktycznej (same czesci rzeczywiste, 15 liczb,
  daja 0,887 wobec 0,902 dla pelnych 20 liczb), ale OBALONA jako regula szeregowania modow (korelacja czesci
  rzeczywistej z odroznieniem narzedzia od tla wynosi -0,04).
- Rozdzielone dwie rozne wlasnosci, dotad mieszane: modul wartosci wlasnej mierzy TRWALOSC pod ruchem i idzie
  w parze z odczytem ksztaltu, a nie z odroznianiem metalu od tkanki.
- Obalone wlasne przewidywanie z dokumentu 42, ze postaci kwadratowe pobija liniowe; poprawka zapisana w miejscu.

Pelny opis: `SESSION_SUMMARY.md`.
