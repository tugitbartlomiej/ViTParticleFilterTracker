# Diagnoza: dlaczego pętla Koopman + FastGAN nie przyspiesza uczenia

Data: 2026-09-28. Pomiary: `diagnoza_porazki.py` → `wyniki/DIAGNOZA_PORAZKI.json` (na danych kampanii z `WYNIKI.md`,
bez nowego GPU). Literatura: dwa przeglądy (Undermind, Scite; wycofań brak dla sprawdzonych DOI).

## Receptura uczenia na Edenie (co dokładnie liczyliśmy)

RF-DETR LargeDeprecated od wag COCO; 1 obrót = 1 epoka = 2560 próbek = 160 kroków (batch 4 × akumulacja 4);
AdamW, stały LR bez rozgrzewki, **lr = 1,67e-5 (dekoder, projektor, głowice), lr_encoder = 1,67e-6 (DINOv2)** —
6× i ~90× mniej niż domyślne w rfdetr 1.10.1 (1e-4 i 1,5e-4); bez EMA, bez augmentacji (`aug_config={}`), bez
multi-scale. Każdy obrót startuje z samych wag poprzedniego (stan optymalizatora od zera). Sampler: każda klatka bazy
(1500) i każda dodana raz na epokę, reszta losowe powtórzenia bazy, więc dodane klatki to 5 % próbek w k01 i 31 %
w k06. Łącznie 7 epok; mAP@0,5 0,86 po pierwszej, 0,90–0,91 po siódmej.

## Co jest zepsute — pomiary

1. **Prognoza Koopmana jest gorsza niż „latent się nie ruszy”.** Na 3 obroty naprzód błąd Koopmana średnio 0,196
   rozrzutu sond, brak zmiany 0,078; Koopman gorszy w 12 z 12 przypadków (3 ziarna × start k1–k4). Kierunek ruchu
   jest trafiony (korelacja przewidzianego i prawdziwego ruchu skalara po sondach 0,81–0,94), ale wielkość jest
   przestrzelona kilkukrotnie: operator dopasowany do szybkiego ruchu z rozgrzewki (ruch netto skalara 2,30 w k00)
   nie widzi hamowania (0,99 w k01, 0,10 w k02). Cele FastGAN wskazywały więc stany, do których sieć nie dochodzi.
2. **Latent siedzi w części sieci, która prawie się nie uczy.** Względna zmiana wag k00 → k06: kręgosłup DINOv2
   0,026 %, projektor 1,6 %, dekoder 1,3 % (COCO → k00: 0,013 % wobec 0,74–0,78 %). Postęp detekcji robią projektor
   i dekoder; latent sond to cechy kręgosłupa, zamrożonego w praktyce przez lr_encoder.
3. **Latent nie wiąże się z wynikiem detekcji.** Między pętlami w tym samym (ziarno, obrót): skalar tożsamości vs
   mAP@0,5 ρ = 0,07 (p = 0,61, n = 53); niezmienniczość R vs mAP ρ = −0,20 (p = 0,16). Latent staje po k02,
   a mAP rośnie dalej.
4. **Wybór skupia klatki.** Dodane przez koopman są do siebie bardziej podobne (średni cos DINOv2 0,66–0,68 wobec
   0,62 losowych), mają ~2× więcej sąsiadów z tego samego filmu w odstępie ≤ 30 klatek (138–140 wobec 62–83 na 804)
   i gorzej pokrywają pulę (0,893 wobec 0,895). Filtr „do nauczenia” (RHO) bierze 134 z ~190–224 kandydatów, czyli
   prawie nie filtruje; prognoza zmienia ~20 % kandydatów (pokrycie top-45 cel/obecny ~80 %).
5. **Dźwignia jest słaba, a szum duży.** Efekty ~0,003 mAP przy szumie ziaren 0,003–0,006 i 3 ziarnach; literatura
   wymaga 10–29 sparowanych biegów, by wykryć taką różnicę.

Nie jest przyczyną: rozmiar modelu (Nano i Large dają to samo; Large rusza latentem ~10× mocniej, bez zysku) ani
liczba epok (latent kręgosłupa stoi od k02 z powodu LR; więcej epok go nie ruszy).

## Co mówi literatura

- Koopman w uczeniu sieci prognozuje wagi, działa w basenie atraktora, krótki horyzont; nikt nie używał go do doboru
  danych; faza wczesna jest odsprzęgnięta od późniejszej (Dogra, Redman 2020, arXiv:2006.02361; Tano i in. 2020,
  arXiv:2006.14371; Redman i in., NeurIPS 2023, 10.52202/079017-0744).
- Szybka zmiana reprezentacji trwa 2–3 epoki (Fort i in. 2020, arXiv:2010.15110); niższe warstwy przy dostrajaniu
  prawie stoją (Neyshabur i in. 2020, arXiv:2008.11687); ruch cech nie jest miarą postępu (Kumar i in. 2022,
  arXiv:2202.10054); zamrożony kręgosłup z pojemną szyją dorównuje dostrajaniu w detekcji (Vasconcelos i in.,
  CVPR 2022, 10.1109/CVPR52688.2022.01326).
- Top-k bez różnorodności przegrywa z losowym nawet na cechach DINOv2 (Gupte i in. 2024, arXiv:2401.14555; Kirsch
  i in. 2021, arXiv:2106.12059); w wideo sąsiednie klatki są redundantne, a losowy z wykluczeniem sąsiadów bije
  zwykły losowy (Bengar i in. 2019, 10.1109/ICCVW.2019.00120); w detekcji dobór po stracie obniża mAP, pomaga miara
  zgodna z AP (Sun i in. 2025, arXiv:2511.14197).
- RHO-LOSS działa przy doborze online co paczkę i ~10 % kandydatów (Mindermann i in. 2022, arXiv:2206.07137); model
  odniesienia musi być dobry (Evans i in. 2023, arXiv:2312.05328).
- Odwracanie generatora przez cechy detektora nie gwarantuje wyglądu, a wyszukiwanie DINOv2 działa po wyglądzie
  (Sabour i in. 2016, arXiv:1511.05122) — łańcuch FastGAN → DINOv2 rwie związek z latentem.
- Różnice w aktywnym uczeniu często mieszczą się w szumie (Munjal i in., CVPR 2022, 10.1109/CVPR52688.2022.00032;
  Bouthillier i in. 2021, arXiv:2103.03098); losowe podzbiory zmieniane co epokę są mocnym punktem odniesienia
  (Okanovic i in. 2023, arXiv:2305.18424).

## Co zmienić, żeby metoda miała szansę (w kolejności)

1. **Bramka przed doborem:** latent musi przewidywać wynik zadania. Zanim użyjemy go do wyboru klatek, sprawdzić
   na historii punktów kontrolnych, że jego zmiana koreluje z mAP lub stratą walidacyjną; obecny nie przechodzi.
2. **Latent tam, gdzie jest uczenie:** czytać sondy po projektorze (wejście dekodera) albo podnieść lr_encoder do
   domyślnego z zanikiem po warstwach, jeśli kręgosłup ma się uczyć.
3. **Prognoza z hamowaniem i porównaniem z „brak zmiany”:** dopasowywać od epoki 0, z członem dryfu do punktu stałego
   (jak dryf w równaniu Langevina z projektu); używać tylko, gdy bije brak zmiany na walidacji w przód.
4. **Wybór zgodny z zadaniem i różnorodny:** ocena klatki po wpływie gradientowym na walidacyjną stratę detekcji
   (TracIn/LESS) albo mierze zgodnej z AP; jedna klatka na skupisko, wykluczenie sąsiadów ±30 klatek; punkt odniesienia
   „losowe z wykluczeniem sąsiadów”.
5. **Moc:** więcej ziaren (≥ 10 par) lub dobór online co paczkę; większy udział dodanych klatek na początku.
