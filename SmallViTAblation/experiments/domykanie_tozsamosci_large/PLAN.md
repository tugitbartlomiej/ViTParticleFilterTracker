# Domykanie latentu tożsamości: klatki, które najszybciej uczą, że to samo narzędzie ma ten sam kod

Data: 2026-09-27. Status: **projekt, czeka na zatwierdzenie** (nic nie jest zgłoszone na Edenie).

Polecenie użytkownika: „To można szukać jakby tych obrazów, które najszybciej domykają latent tożsamości. Tylko musisz
dobrze określić warunki domykania i szybkiego uczenia latentu tożsamości latentu!”

Kontekst: pętla `../petla_od_poczatku_large` (Koopman + FastGAN + wybór „do nauczenia”, RF-DETR LargeDeprecated,
DINOv2-B) po trzech obrotach nie przyspiesza mAP@0,5: koopman − bez średnio −0,0023, koopman − losowe −0,0032
(9 par ziarno × obrót), kryterium z jej PLAN.md niespełnione.

## Co już wiemy z danych (bez nowego GPU)

`diagnoza_domkniecia.py` → `wyniki/DIAGNOZA.json`, latent sond z obecnej kampanii (3 ziarna × 3 pętle × obroty 0–3).
Sondy: 12 obrazów narzędzia × 6 wariantów (oryginał, obrót o 90°, czarne tło, żółte tło, narzędzie skrócone do 0,8,
narzędzie rozjaśnione), 8 obrazów fit, 4 holdout. Latent tożsamości sondy: skalar a = w·(narzędzie − tło) oraz 5 liczb
(w, S_4).

| Warunek | Miara | Start (wagi COCO) | Koniec k00 | Koniec k03 | Wniosek |
|---|---|---|---|---|---|
| W1 zatrzymanie | średni ruch a na 5 aktualizacji / rozrzut sond | 0,007–0,010 | 0,013 | 0,002 | latent stoi od k02 |
| W2 niezmienniczość | R = rozrzut a między wariantami obrazu / rozrzut średnich obrazów | 1,30 | 0,91 | 0,79 | plateau od k02 |
| W2 niezmienniczość | rozpoznanie: sonda najbliżej środka własnego obrazu (przypadek 0,08) | 0,29 | 0,33 | 0,31 | brak postępu |
| W2 niezmienniczość | przesunięcie wariantu względem oryginału (rozrzut sond 6,1) | | | obrót −3,4; żółte tło +3,5; jasne narzędzie +3,9; czarne tło −0,9; skrócone +0,2 | systematyczne |
| W3 rozdzielenie | min a po sondach (> 0: narzędzie odróżnione od tła) | 3,0 | 4,4 | 5,5 | spełniony |

Wszystkie trzy pętle (koopman, losowe, bez) są tu nierozróżnialne (różnice R ≤ 0,02). Latent przestaje się ruszać
po dwóch obrotach, ale **zatrzymuje się w stanie niedomkniętym**: obrót narzędzia albo kolor tła zmienia kod bardziej,
niż różnią się dwa różne obrazy. Uczenie ma wyłączone przekształcenia (`aug_config={}`, bez multi-scale), więc sieć
nie dostaje przykładów „to samo narzędzie, inny obrót lub światło”, poza tym, co zdarza się w klatkach naturalnie.

## Definicja domknięcia (ustalona przed nowym wynikiem)

Latent tożsamości jest **domknięty**, gdy jednocześnie:

- **W1 zatrzymanie:** średni ruch a < 0,005 rozrzutu sond na 5 aktualizacji (dalsze uczenie go już nie przesuwa);
- **W2 niezmienniczość:** R < 0,5, rozpoznanie ≥ 0,8 oraz |przesunięcie| < 0,5 (8 % rozrzutu) dla obrotu, czarnego
  tła, żółtego tła i jasnego narzędzia (wariant skrócony zmienia kształt, więc tylko raportowany);
- **W3 rozdzielenie:** min a > 0.

Uzasadnienie progów: R < 0,5 znaczy, że warianty jednego obrazu leżą dwa razy bliżej siebie niż różne obrazy;
rozpoznanie 0,8 to cztery na pięć wariantów przypisane do własnego obrazu; 0,5 w skali a to 8 % rozrzutu sond,
około siedem razy mniej niż obecne przesunięcia obrotu i koloru (3,4–3,9). Sam W1 nie wystarcza: obecnie spełniony,
a tożsamość niedomknięta.

**Tempo domykania** (miara główna): pole pod krzywą R po obrotach 1–3 (mniejsze = szybciej) i liczba aktualizacji do
spełnienia W2; osobno R i rozpoznanie na 4 obrazach holdout (nieużywanych przy wyborze klatek).

## Metoda: wybór klatek po zmierzonym wpływie na domknięcie

Obecna pętla wybierała klatki podobne (DINOv2) do obrazu FastGAN — to pośrednia miara. Nowa ocenia wprost, jak krok
uczenia na klatce x zmienia niedomkniętą część latentu:

- cel domknięcia J(θ) = średnia po obrazach fit wariancji a między wariantami / wariancja średnich obrazów, z wagami
  przesunięć, których **prognoza Koopmana nie zamyka sama** (operator z obecnej pętli dopasowany do historii R
  i przesunięć; tu wszystkie, bo R stoi od k02);
- wynik klatki: s(x) = ⟨∇θ L_det(x), ∇θ J⟩ — mały krok uczenia na x obniża J o η·s(x) (przybliżenie pierwszego
  rzędu); gradient J przez różniczkowalny odczyt sond (`czastki.DifferentiableReader`), gradienty ograniczone do
  ostatnich bloków kręgosłupa i projektora (pamięć);
- kandydaci: 1200 losowych klatek puli na obrót (warstwowo po filmach, bez bazy i wcześniej dodanych); 134 najwyższe
  s(x), najwyżej 10 z jednego filmu;
- FastGAN nie jest potrzebny w etapie 1; rozszerzenie na później: generować narzędzie w obrotach i oświetleniu,
  których nie pokrywa żadna prawdziwa klatka.

## Etapy

**Etap 0 — czy wynik s(x) coś przewiduje** (1 zadanie hopper, ~30 min): wagi k01 ziarna 42, 24 kandydatów; dla każdego
prawdziwy krok optymalizatora na tej klatce i pomiar zmiany J (z przywróceniem wag). Warunek przejścia: korelacja
rang Spearmana przewidzianej i zmierzonej zmiany J ≥ 0,5. Jeśli nie — stop i raport, bez etapu 1.

**Etap 1 — kampania** (3 ziarna × obroty 1–3, od istniejących wag k00 każdego ziarna):

- `domykanie`: +134 klatki na obrót wg s(x);
- `augmentacja`: bez dodawania klatek, uczenie z obrotami o 90°/180°/270° i zmianą jasności/koloru (bezpośredni,
  tani sposób uczenia niezmienniczości; punkt odniesienia, który metoda musi pobić);
- `losowe` i `bez`: **wzięte z obecnej kampanii** (te same wagi k00, ten sam sampler, to samo ziarno losowania,
  ta sama receptura; od k02 różni się tylko brak zapisu plików .ckpt, co nie zmienia uczenia).

Do sprawdzenia przed etapem 1: czy środowisko Edena ma albumentations (inaczej przekształcenia w naszym `PairedData`).

## Kryterium (ustalone przed wynikiem)

1. `domykanie` ma mniejsze pole pod krzywą R (wszystkie sondy) niż `losowe` i `bez` w każdym z 3 ziaren, oraz
   wyższe rozpoznanie na holdout średnio po ziarnach;
2. mAP@0,5 nie niższe niż `bez` o więcej niż 0,005 średnio po obrotach 1–3 (szum par: odchylenie 0,003–0,006).

Raportowane osobno: `domykanie` vs `augmentacja` (jeśli augmentacja domyka tak samo lub lepiej, wybór klatek nic nie
dodaje) oraz czy szybsze domknięcie przekłada się na mAP (pytanie użytkownika o tempo uczenia sieci). Zbiór val ma
naturalny rozkład obrotów i światła, więc domknięcie może nie podnieść mAP — to też jest wynik.

## Koszt i miejsce

Etap 0: ~0,5 h GPU. Etap 1: 18 zadań uczenia × ~55 min ≈ 16,5 h GPU + 9 zadań oceny kandydatów × ~20 min ≈ 3 h GPU.
Kwota Edenu: 18 × 0,59 GB ≈ 10,6 GB. Stan o 21:10: wolne 24,1 GB; w obrotach k04–k06 obecna kampania zajmie jeszcze
~15,9 GB, zostanie ~8 GB — **obie się nie zmieszczą**. Do wyboru (decyzja użytkownika): zatrzymać obecną kampanię
po k04 (kryterium i tak rozstrzygnięte po k03; oszczędza ~10,6 GB i ~16 h GPU) albo zgoda na usunięcie starych
katalogów na Edenie (`2026-09-26_petla_uczenia` 6,9 GB — pilot Nano, wyniki lokalnie i w git 12d3f02;
`2026-09-25_kontynuacja_dobor` 2,4 GB; `2026-09-25_kontynuacja_5epok`, `2026-09-25_kontynuacja_dobor150`).
Kontrola kwoty przed każdym obrotem jak w pętli Large (stop poniżej 8 GB).
