# Odpowiedź latentu na pchnięcie jednej paczki: funkcja odpowiedzi z istniejących nagrań

Data: 2026-10-05. Plan zapisany przed uruchomieniem kodu. Pomiar bez treningu i bez Edenu, na nagraniach, które już
leżą na dysku.

## Po co

Przegląd z 5 października proponuje wybierać paczki zdjęć według przewidzianego skutku na latencie (siła paczki
z dokumentu `docs/23`). Ta propozycja ma sens tylko wtedy, gdy skutek jednej paczki nie znika po kilku krokach.
Dokument `docs/25` wyprowadził z równania `Δz(t+1) = 0,83·Δz(t) + 0,98·S(t)`, że skutek paczki narasta przez pęd
do około 5,8 pierwszego kroku. To wniosek z dopasowania jednego kroku. Pomiar z 3 października pokazał natomiast,
że przyrosty latentu po 3-5 krokach są ujemnie skorelowane, czyli ruch częściowo wraca. Te dwa obrazy trzeba
rozstrzygnąć pomiarem wprost.

## Pytania

1. Ile z jednorazowego pchnięcia paczki zostaje w latencie po 1, 2, 5, 10, 20 i 50 krokach: narasta, utrzymuje
   się, czy wraca?
2. Czy składanie pchnięć kolejnych paczek z dopasowaną funkcją odpowiedzi odtwarza ruch latentu na 10-50 kroków?
3. Jaką część ruchu na `h` kroków wyznacza sama paczka bieżąca, czyli to, czym steruje jedna decyzja?

## Dane

`../maly_vit_rownanie_latentu/nagranie_s47_sily` i `nagranie_s48_sily` (opis w `docs/23`): latent 180 sond z
narzędziem po każdym kroku (średnia po tokenach narzędzia, 768 liczb) oraz siły w stanie `t` przepuszczone przez
sieć: `S_nowa` (gradient paczki `t`), `S_ped` (pęd AdamW sprzed kroku), `S_zanik` (zanik wag). Ziarno 47: siły od
kroku 900 do 2999. Ziarno 48: od 2000 do 2999.

Dlaczego to wystarcza bez nowych przebiegów: paczki są losowane niezależnie od stanu sieci, więc pchnięcie paczki
jest losowym wstrząsem. Regresja przyszłego przesunięcia latentu na pchnięcia z kolejnych kroków rozdziela wkład
każdego kroku z osobna (w ekonometrii: projekcje lokalne i model z rozłożonymi opóźnieniami).

## Aparat

Dla horyzontu `h` i startu `t` (współczynniki wspólne dla 768 współrzędnych i 180 sond, jak w `docs/23`):

    z(t+h) − z(t) = Σ_{k=0..h−1} c(h,k)·S_nowa(t+k) + a(h)·S_ped(t) + w(h)·S_zanik(t) + g(h)·(z(t) − μ(t)) + reszta

`μ(t)` to średnia z ostatnich 101 stanów. Trzy modele:

- **A, granica:** wszystkie pchnięcia w oknie znane, każde policzone w swoim prawdziwym stanie. To nie jest
  prognoza (łamie regułę o przyszłych klatkach); mówi, czy liniowe składanie pchnięć wystarcza.
- **B, uczciwa prognoza:** tylko `S_ped(t)`, `S_zanik(t)` i odchylenie od średniej.
- **C, znana paczka bieżąca:** B plus `S_nowa(t)`. To sytuacja wyboru: znamy paczkę, którą zaraz podamy.

**Funkcja odpowiedzi** `R(n)` to współczynnik modelu A przy pchnięciu sprzed `n` kroków: `R(n) = c(h, h−n)`.
Odczytuję ją z `h = 50` i porównuję z `h = 10` i `h = 20` (dla układu liniowego niezmiennego w czasie muszą się
zgadzać).

Osobno liczę to samo dla **ruchu wspólnego** (średnia po 180 sondach) i **ruchu własnego** (odchylenie sondy od
tej średniej), bo drgania są w dużej części wspólne.

Dopasowanie: starty 1000-1949, ziarno 47 (cele do kroku 1999). Test: starty 2000-2940 ziarna 47 oraz ziarna 48,
bez ponownego dopasowania. Miara: `1 − SSE / SSE prognozy „bez zmian”`. Niepewność `R(n)`: losowanie bloków po 50
startów, 500 powtórzeń.

## Kryteria ustalone przed wynikiem

- **K1 (zgodność z docs/23):** `R(1)` między 0,9 a 1,1.
- **K2 (kształt odpowiedzi):** „narasta”, gdy `R(10) ≥ 2·R(1)`; „przejściowa”, gdy `R(20) ≤ 0,5·R(1)`;
  w pozostałych przypadkach „trwała bez wzmocnienia”.
- **K3 (składanie liniowe):** model A na `h = 10` osiąga co najmniej 0,85 na teście ziarna 47 i na ziarnie 48.
  Dla porównania: równanie jednego kroku iterowane z liniowym pędem dało 0,21 (`WYNIKI_ROWNANIE_Z_PEDEM.md`).
- **K4 (opisowo):** przyrost umiejętności modelu C ponad B na `h` = 1, 5, 10. To miara, ile daje wiedza o jednej
  paczce.

## Czego ten pomiar nie dowodzi

- To nie jest pomiar interwencyjny: nie ma par przebiegów różniących się jedną paczką.
- W modelu A przyszłe siły są policzone w prawdziwych przyszłych stanach sieci, więc A to granica, nie prognoza.
- Dopasowanie na jednym ziarnie, przeniesienie na jedno; kroki 1000-3000; paczka 32; średnia po tokenach narzędzia,
  nie pojedyncze tokeny.
- Wynik dotyczy latentu, nie straty detekcji ani mAP.
