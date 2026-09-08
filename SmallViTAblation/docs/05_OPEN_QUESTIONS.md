# 05 — Pytania otwarte i rzeczy do sprawdzenia przed startem

Punkty `O1`–`O3` są **darmowe** (bez GPU) i rozstrzygają kształt reszty planu.
Robimy je pierwsze.

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

**Powiązane, do sprawdzenia na Edenie:** czy istnieją już wytrenowane modele DETR
i YOLO na tym podziale. Wiem, że jest katalog
`Yolo\20k_finetune\train_out_scratch_90k_vs\` opisany jako trening video-split
(job 1729466, wznowiony jako 1733562, sesja 2026-07-12) oraz skrypty
`Eden/Scripts/Train90kFromScratch_VideoSplit_2026-06-24/`. Jeśli te biegi są
kompletne, dają **darmowy punkt odniesienia** dla nowego protokołu i nie trzeba
ich powtarzać.

## O2. Ile jest klatek źródłowych w filmach treningowych? [BLOKUJE BUDŻET B]

Cała pula: 22 837 klatek źródłowych, po augmentacji 91 336 (albo 91 348 — patrz
O3). Filmy treningowe to 35 z 50, więc klatek źródłowych w treningu jest rzędu
16 tys., ale **dokładna liczba zależy od długości filmów**, nie od ich liczby.

To rozstrzyga, czy baseline `B-RANDSRC` da się zrobić przy budżecie 20 000.
Jeśli nie — obniżamy budżet poziomu B albo przedefiniowujemy ten baseline.

**Do zrobienia:** policzyć z `augmented_coco_*.json` i `video_split.json`.
Skrypt liczy dwie liczby: klatki źródłowe w treningu, klatki augmentowane
w treningu.

## O3. Skąd 91 336, skoro 22 837 × 4 = 91 348?

Recenzent 2 znalazł niezgodność dwunastu obrazów. Możliwe wyjaśnienia: część
klatek nie dostała kompletu trzech augmentacji, część odpadła przy filtrowaniu,
albo jedna z liczb w artykule jest przepisana z innego etapu.

**Do zrobienia:** policzyć rozkład liczby wariantów na klatkę źródłową. To jedno
zapytanie do pliku anotacji, a odpowiedź musi trafić do artykułu jednym zdaniem.

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
