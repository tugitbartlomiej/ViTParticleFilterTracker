# Wyniki: pętla Koopman + FastGAN od początku dostrajania na RF-DETR LargeDeprecated (DINOv2-B)

Data: 2026-09-27/28. Plan i kryterium: `PLAN.md` (ustalone przed wynikiem). Analiza: `analiza_large.py` →
`wyniki/ANALIZA.json`, `wyniki/ANALIZA.png`. Hasze plików binarnych (wagi, latent, obrazy): `wyniki/SHA256.json`.

**Stan: 53 z 54 obrotów pętli policzone.** Brakuje `bez` k06 ziarna 43: zadanie 1817673_2 padło w 0 s na dgx-3
(`cp -a` do `/dev/shm`: „Numerical result out of range”, usterka węzła); zgłoszone ponownie jako 1817712 z
`--exclude=dgx-3`, `STAN.json` s43 wskazuje nowe zadanie. Po jego końcu: `dyrygent_large.py --s 43` (wznawia i
pobiera), potem ponownie `analiza_large.py`. Werdykt kryterium nie zależy od tego punktu (patrz niżej).

## Werdykt

**Kryterium niespełnione w obu częściach.** Pętla koopman nie przyspiesza uczenia modelu Large; losowe klatki
uczą szybciej niż klatki wybrane pętlą.

1. Pole pod krzywą mAP@0,5 (średnia po obrotach 1–6): losowe najwyżej w każdym ziarnie
   (s42 0,9053 / s43 0,9075 / s44 0,9013), koopman 0,9018 / 0,9033 / 0,9001, bez 0,9008 / 0,9032 (k1–5) / 0,9004.
2. Obroty 1–3, koopman − bez: średnio −0,0023 (wymagane ≥ +0,005), dodatnie 3 z 9.

## Różnice sparowane mAP@0,5 (to samo ziarno i obrót)

| Porównanie | Obroty 1–3 | Obroty 4–6 | Obroty 1–6 |
|---|---|---|---|
| koopman − bez | −0,0023 (sd 0,0032; 3/9 > 0) | +0,0033 (sd 0,0055; 6/8 > 0) | +0,0004 (sd 0,0052; 9/17 > 0) |
| koopman − losowe | −0,0032 (sd 0,0036; 1/9 > 0) | −0,0027 (sd 0,0031; 2/9 > 0) | −0,0030 (sd 0,0033; 3/18 > 0) |
| losowe − bez | +0,0009 (sd 0,0055; 5/9 > 0) | +0,0060 (sd 0,0057; 7/8 > 0) | +0,0033 (sd 0,0060; 12/17 > 0) |

Koopman − losowe jest ujemne w 15 z 18 par (średnia −0,0030 przy sd 0,0033): to nie szum, klatki wybrane pętlą
uczą gorzej niż losowe. Dokładanie klatek zaczyna pomagać od k04 (losowe − bez +0,006), gdy dodanych jest ponad 500.
Tempo (pierwszy obrót z mAP@0,5 ≥ 0,90): koopman k2/k2/k4, losowe k2/k1/k3, bez k4/k1/k3.

Przebieg mAP@0,5 (k00 = wspólna rozgrzewka; obroty 1–6):

| Ziarno | k00 | koopman | losowe | bez |
|---|---|---|---|---|
| 42 | 0,8622 | 0,8953 0,9002 0,8948 0,9057 0,9048 0,9098 | 0,8976 0,9029 0,8982 0,9078 0,9092 0,9164 | 0,8979 0,8994 0,8998 0,9050 0,9018 0,9007 |
| 43 | 0,8610 | 0,8989 0,9012 0,9062 0,9071 0,9048 0,9018 | 0,9039 0,9045 0,9148 0,9115 0,9049 0,9051 | 0,9001 0,9028 0,9040 0,9018 0,9073 — |
| 44 | 0,8591 | 0,8932 0,8938 0,8972 0,9008 0,9087 0,9067 | 0,8936 0,8900 0,9043 0,9071 0,9066 0,9061 | 0,8931 0,8999 0,9044 0,9065 0,9016 0,8972 |

## Latent i prognoza

- Model Large rusza latentem znacznie mocniej niż Nano: w rozgrzewce skalar tożsamości 16,8 → 19,1 (0,38 rozrzutu
  sond w 160 aktualizacjach); przewidziana droga kotwic na 3 obroty naprzód 1,2–1,7 rozrzutu w k01 (pilot Nano 0,12),
  spada do 0,45–0,8 w k02. Dryf gaśnie po około dwóch obrotach (ruch skalara 0,013 → 0,002 rozrzutu na krok).
- Obraz celu FastGAN różnił się od obrazu obecnego wyraźniej niż w pilocie (L1 do 0,046 wobec 0,002), a pokrycie
  top-45 cel/obecny spadło do ~80 % (pilot 89 %). Wybór był więc inny niż „to, co trening i tak osiąga”, ale nie lepszy.
- Pętla koopman przesuwa skalar tożsamości trochę dalej niż pozostałe (k01: +0,03…+0,10 względem bez we wszystkich
  ziarnach), bez zysku w mAP. Niezmienniczość tożsamości (R = rozrzut między wariantami obrazu / rozrzut między
  obrazami) stoi na ~0,78 od k02 we wszystkich pętlach; obrót narzędzia (−3,4), żółte tło (+3,3) i jasne narzędzie
  (+3,7) przesuwają kod bardziej niż różnica dwóch obrazów. Szczegóły i definicja domknięcia:
  `../domykanie_tozsamosci_large/PLAN.md`, `DIAGNOZA.json`.

## Czego to nie dowodzi

- Nie dowodzi, że prognoza latentu jest bezużyteczna — tylko że ten sposób zamiany prognozy na klatki (FastGAN →
  podobieństwo DINOv2 → „do nauczenia” wg Nano) nie bije losowania na tym zbiorze i modelu.
- Losowe klatki są różnorodne; wybór wokół trzech kotwic skupia 134 klatki na wąskim fragmencie danych. Porównanie
  z wyborem różnorodnym nie było robione.
- Trzy ziarna, jeden zbiór (cataract_lmm), jeden model; val ma naturalny rozkład, bez osobnego testu trudnych klatek.

## Przebieg techniczny

Rozgrzewka 49 min/zadanie, obrót ~55 min/zadanie (hopper, dgx), cała kampania ~9 h. Poprawki w trakcie
(`PLAN.md`, „Uwagi z przebiegu”): KMP_DUPLICATE_LIB_OK w wyborze (faiss + torch), kontrola kwoty Edenu przed
obrotem, od k02 bez zapisów .ckpt i FINAL.pth (chwilowo 30 GB przy dziewięciu zadaniach). Commity: 7190b27, e5fecc1,
e9055b2 i ten opis.
