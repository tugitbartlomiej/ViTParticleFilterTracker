# Session: Odbiorca zapisu MLP, kanał normy i rodzina efektów balanced amplification

**Created:** 2026-10-27 03:57
**Type:** Mixed (Analysis + SSH Eden + Literature)
**Status:** Active (pełny bieg znaku normy 1933192 w toku; start artykułu TMLR zaplanowany na 04:38)

## Quick Summary
Kontynuacja sesji `Session_2026-10-27_014253`. Domknięto balanced amplification v2 (BA3c/BA3d), przeszukano literaturę o rodzinie efektów
pokrewnych balanced amplification (Undermind, 134 prace), a agent Opus 5.5 wykonał eksperyment `odbiorca_zapisu_mlp_uwaga` (kto czyta
zapis wzmacnianego modu MLP). Na jego podstawie założono i wysłano na dgx eksperyment `znak_kanalu_normy_mlp` (norma czy średnia,
znak zmiany normy, stabilizator czy zatrzask).

## Key Results
- Balanced v2: wyjście u₁ nie leży w odczycie ramki (BA3c 0/3) ani w wejściu MLP następnego bloku (BA3d 3/5); kolejne bloki czytają
  wspólny kierunek v₁ (cos² 0,09–0,21, 17–40× losowe) — struktura równoległa, nie łańcuchowa. W CIFAR u₁ w logitach (BA3c′ 2/2).
- Odbiorca (5/5 biegów): zapis u₁ nie ma wyróżnionego odbiorcy — uwaga, MLP i głowa czytają go 1,3–4,5× losowe; O1 FAIL 2/5, O4 FAIL 1/5,
  O5 PASS 5/5 (nie „kosz”), O8 PASS 4/5: 20–26 % energii u₁ w jądrze LayerNorm w detekcji (CIFAR MałyViT 8 %, DeiT 0,7 % = losowe).
- Komunikacja MLP–uwaga idzie przez wspólną współrzędną odczytu, nie przez zapis MLP; w detekcji zapis czyta najmocniej projekcja k
  (po fakcie), w DeiT v.
- Literatura: balanced amplification to korowa realizacja szerszej rodziny „nienormalnego wzmocnienia przejściowego”; nowe cytowania
  Pattadkal 2023, Symon 2017, Ramos 2020, Saito 2026; luka (MLP transformera) potwierdzona drugi raz.

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
