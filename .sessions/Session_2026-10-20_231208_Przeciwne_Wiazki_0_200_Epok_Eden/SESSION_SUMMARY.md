# Sesja: przeciwne wiązki w treningu ViT, epoki 0–200

**Data:** 2026-10-20/21 · **Branch:** codex/eden-nano-training-20260914 · **Seed:** 3047

## Zrobione
1. Przegląd stanu Edenu: żadne zadanie nie liczyło, trening do 200 epok (job 1907130) zakończony 20:43. Katalog domowy 184,7 GB z limitem miękkim 200 GB, katalog grupy 37,7 GB z 512 GB.
2. Pobranie 14 checkpointów epok 70–200 i logów (agent Sonnet), SHA256 zgodne z logiem Edenu.
3. Eksperyment `pary_przeciwne_do_200epok_20261020__23-21-30`:
   - liczba par ≥150/160/170/175/179° na 15 checkpointach epok 60–200 (job 1911518, 134 s), kontrola kroku 65640 bitowo identyczna z rodzicem;
   - para 370–391 (blok 6): historia wag, tokeny, wyłączenie na tle 300 par kontrolnych; detekcja IoU i pewność na 500 VAL + 500 TEST dla 26 checkpointów (job 1911919, 247 s);
   - wykresy 01–09, raport HTML/PDF, krótki wniosek (na zamówienie użytkownika).
4. Eksperyment `vit_kontynuacja_do_400epok_20261020__23-50-57`: wznowienie z epoki 200 (pełny stan optimizera, samplera, RNG), job 1911922; PREFLIGHT zaliczony, trening był przy epoce ok. 203, kontynuacja nieweryfikowana po restarcie sesji.
5. Przegląd literatury (Undermind, Scite) rozpoczęty przez 4 agentów badawczych; **nie ukończony** (agenci przerwani wraz z poprzednią sesją), notatki w `SmallViTAblation/research_notes/` mogą być niepełne.

## Główne wyniki
- Par ≥150°: 24 (epoka 60) → szczyt 30 (epoki 80–110) → 15 (epoka 200). Pierwsza para ≥170° w epoce 170, w epoce 200 kąt 173,0° (neurony 370 i 391, blok 6). Progu 175° nie osiąga żadna para.
- Żadna para nie ma przeciwnych wag W2 (maks. ok. 142°). Przeciwność powstaje z prawie równoległych W2 i przeciwnych znaków GELU.
- Para 370–391 to kopie: cos W1 0,31 → 0,96, kąt W2 72° → 7° między epokami 100 i 200; sumują się na 94% tokenów, znoszą tylko na 6% (κ 0,19, 0,3–0,6% wyjścia MLP). Wyłączenie pary: ΔIoU ok. 10⁻⁵, CI obejmuje zero; efekty na logity się sumują (iloraz 1,66).
- IoU stoi od ok. epoki 40 (VAL 0,762 → 0,765), pewność rośnie (0,863 → 0,899), także na klatkach bez narzędzia (0,629 → 0,675, n=34).
- Pilot kanału Mar25l (3 seedy): H0, brak kanałów; zamiast nich kopie.

## Do zrobienia
- Sprawdzić job 1911922 na Edenie; po zakończeniu policzyć pary, parę 370–391, IoU i pewność dla epok 210–400.
- Dokończyć syntezę literatury i ogólny wniosek do artykułu (nowe zlecenie agentów).
- Dodatkowe seedy dla wniosków o parach (obecnie jeden seed, 50 klatek dla par).

## Uwagi techniczne
- Python 3.11: `W:\Python\3.11\python.exe`. Claude Code config: `W:\Chaty\Claude\CodeCLI\config`.
- Walidator palety wymaga `node`, którego nie ma na tym komputerze; użyto palety referencyjnej.
