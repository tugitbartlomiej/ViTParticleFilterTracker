# 05 — Pytania otwarte i rzeczy do sprawdzenia przed startem

Punkty `O1`–`O3` są **zamknięte** (2026-09-08, bez GPU). Otwarte zostają
`O4`–`O7` — to są decyzje, nie pomiary.

---

## O1. Podział po filmie = podział po pacjencie — ROZSTRZYGNIĘTE

**Rozstrzygnięte 2026-09-08 przez autora: jedno nagranie = jeden pacjent.**

Gotowy podział: `Eden/Checkpoints/Ablations/ablation/video_split.json` — ziarno 42,
**50 filmów, czyli 50 pacjentów**, 35 / 8 / 7. Nazwy `test01`–`test25`
i `train01`–`train25` to etykiety plików, nie dwa różne zbiory.

Konsekwencje:

- podział po filmie **jest** podziałem po pacjencie, którego zażądał recenzent 1
  w zarzucie 1 — nie trzeba go przebudowywać,
- w artykule piszemy wprost: *„patient-level split; one recording corresponds to
  one patient, 50 patients partitioned 35/8/7"* — jedno zdanie zamyka zarzut,
- zbiór testowy to **7 pacjentów**, a nie 589 losowych klatek, co odpowiada też
  na zarzut 2 recenzenta 2 o rozmiarze i konstrukcji testu,
- w tabeli podziału podajemy **liczbę pacjentów obok liczby klatek** — recenzent
  patrzy na tę pierwszą.

Zostaje jedno ograniczenie, które trzeba napisać uczciwie: **50 pacjentów
z jednej instytucji** to mała i jednorodna próba. Recenzent 2, zarzut 9,
wytknął jedną instytucję. Tego nie da się naprawić eksperymentem — tylko
opisem w sekcji ograniczeń.

**Inwentarz Edenu zrobiony** - pelny wynik w `07_EDEN_INWENTARZ.md`. Skrot:
są cztery biegi pod tym podziałem (DETR kurowany 20k od zera do ep240, DETR pełny
trening ep50-90, YOLO pełny trening 190 epok kompletny, wariant jednozapytaniowy
anulowany na ep6), brakuje YOLO na kurowanym 20k, a ablacja V1-V7 startowała
z checkpointu ep170 wytrenowanego na losowym podziale klatek.

## O2. Klatki źródłowe w treningu — POLICZONE

**Policzone 2026-09-08** z `Eden/EdenBackup/DETR/fullpool_videosplit/annotations_*.json`:

| Split | Obrazy | Klatki źródłowe | Wariantów na klatkę |
|---|---:|---:|---|
| train (35 pacjentów) | 71 316 | **17 832** | 17 820 × 4, 12 × 3 |
| val (8 pacjentów) | 10 780 | 2 695 | wszystkie × 4 |
| test (7 pacjentów) | 9 240 | 2 310 | wszystkie × 4 |
| **razem** | **91 336** | **22 837** | |

**Baseline `B-RANDSRC` jest wykonalny na obu budżetach.** Definicja: losujemy
N/4 klatek źródłowych i bierzemy wszystkie cztery ich warianty.

- budżet 5 000 → 1 250 klatek źródłowych z 17 832 dostępnych,
- budżet 20 000 → 5 000 klatek źródłowych z 17 832 dostępnych.

Nie trzeba obniżać budżetu poziomu B ani przedefiniowywać baseline'u.

## O3. Skąd 91 336, skoro 22 837 × 4 = 91 348? — ROZWIĄZANE

**Różnica to dokładnie 12 obrazów.** Dwanaście klatek źródłowych ma **trzy**
warianty augmentacji zamiast czterech; pozostałe 22 825 mają po cztery.
Wszystkie dwanaście leżą w części treningowej.

```
22 837 x 4 = 91 348
   minus 12 brakujących wariantów
         = 91 336
```

Do artykułu wchodzi jedno zdanie: *„22,837 source frames yield 91,336 augmented
images; twelve frames produced three variants instead of four."* Recenzent 2,
zarzut 8, zamknięty.

## O4. Czy bronimy miary złożoności FastSAM w obecnej postaci?

Recenzent 2 zarzut 1: nie chodzi o wagi, tylko o to, **dlaczego akurat te pięć
wielkości** opisuje złożoność sceny. To jest pytanie merytoryczne, nie
statystyczne.

Dwie drogi: (a) uzasadnić każdą składową odniesieniem do problemu — odbicia
lustrzane, przesłonięcia, zmienne oświetlenie mikroskopu; (b) uprościć miarę do
dwóch–trzech wielkości, które da się obronić, i pokazać w `S-*`, że wynik się
nie zmienia. Droga (b) jest uczciwsza i tańsza w obronie.

**Decyzja użytkownika.**

## O5. Czy poziom B ma zostać przy 20 000?

Budżet 20 000 wiąże nowy artykuł z narracją starego. Jeśli sweep pokaże, że
krzywa wypłaszcza się wcześniej, sensowniej opisać **punkt nasycenia**
wyznaczony z danych niż trzymać się okrągłej liczby wybranej wcześniej.

**Decyzja po sweepie**, nie teraz.

## O6. Gdzie publikujemy?

Recenzent 2 odradził ponowne zgłoszenie do IEEE Access. Wybór miejsca zmienia
limit stron i to, ile eksperymentów wchodzi do tekstu głównego, a ile do
materiału uzupełniającego.

**Decyzja użytkownika**, ale warto ją podjąć przed pisaniem, nie po.

## O7. Czy trzymamy DETR-R50 jako punkt odniesienia w nowym artykule?

Za: ciągłość z dotychczasową pracą, mamy gotowe wyniki. Przeciw: recenzent uznał
tę architekturę za przestarzałą jak na rok zgłoszenia, a jej trening jest
najdroższą pozycją w całym planie.

Propozycja: **nie trenować DETR-R50 od nowa**, ale pokazać stare wyniki jako
„skąd przyszliśmy", z jawnym zaznaczeniem, że powstały w innym protokole.
