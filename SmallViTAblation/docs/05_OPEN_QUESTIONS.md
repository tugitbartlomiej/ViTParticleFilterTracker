# 05 — Pytania otwarte i rzeczy do sprawdzenia przed startem

Punkty `O1`–`O3` są **darmowe** (bez GPU) i rozstrzygają kształt reszty planu.
Robimy je pierwsze.

---

## O1. Czy podział po filmie to podział po pacjencie? [BLOKUJE WSZYSTKO]

Mamy gotowy podział po filmach: `Eden/Checkpoints/Ablations/ablation/video_split.json`
— ziarno 42, **50 filmów**, 35 / 8 / 7.

Nazwy filmów: `test01`–`test25` i `train01`–`train25`. Recenzent 1 napisał
„video/procedure or **patient** level". Jeżeli któryś pacjent ma dwa nagrania —
na przykład operację obu oczu — to podział po filmach **nie jest** podziałem
po pacjencie i klatki tej samej osoby siedzą po obu stronach.

**Do zrobienia:** ustalić mapowanie film → pacjent. Jeśli nie ma go w metadanych,
napisać w artykule wprost, że jeden film odpowiada jednej procedurze, a mapowania
na pacjenta zbiór nie udostępnia. To jest uczciwa odpowiedź; udawanie, że
sprawdziliśmy, nie jest.

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
