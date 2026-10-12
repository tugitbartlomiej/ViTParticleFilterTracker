# Wynik: rozmiar paczki nie zmniejsza drgań latentu i nie poprawia prognozy na 10 kroków

Data: 2026-10-03. Plan: [PLAN.md](PLAN.md). Biegi Eden: 1820332 (próba), 1820333 (paczka 320), 1820334 (paczka 128),
H100, ziarno 47, 1000 kroków, ta sama recepta co bieg 1818732 (paczka 32). Dane: `wyniki/pelny_p{128,320}/dane`
(LATENT_GESTY.npy (1001, 12, 60, 768), SHA256 zgodne ze zdalnymi). Liczby: `wyniki/porownanie/POROWNANIE.json`
(32 i 320), `wyniki/porownanie_p128/POROWNANIE.json` (128). Tabela: `py -3.11 kod/tabela.py <oba pliki>`.

## Werdykt

Teza z PLAN.md jest obalona. Dziesięciokrotnie większa paczka nie zmniejsza drgań latentu (średni względny krok
0,078 → 0,082) i nie poprawia prognozy z historii na 10 kroków (0,20 → 0,24). Drgania nie pochodzą z losowania
obrazów do paczki. Większa paczka czyni je nawet mniej uporządkowanymi: autokorelacja kolejnych przyrostów spada
z 0,71 do 0,31, a prognoza na 1 krok z 0,64 do 0,38.

## Tabela (wszystkie 458 analizowanych jednostek: tokeny narzędzia plus 20 tła na obraz; miara 1 − SSE/SSE bez zmian)

| paczka | równanie h1 | h5 | h10 | h100 | OU h10 | sufit h10 | sufit h100 | podłoga h10 | D²/σ² h10 | acf 1 | acf 2 | acf 5 | krok względny | mAP koniec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0,64 | 0,19 | 0,20 | 0,24 | 0,18 | 0,79 | 0,91 | 0,64 | 1,18 | 0,71 | 0,28 | −0,20 | 0,078 | 0,31 |
| 128 | 0,44 | 0,17 | 0,24 | 0,21 | 0,23 | 0,76 | 0,88 | 0,64 | 0,81 | 0,50 | −0,05 | −0,04 | 0,083 | 0,32 |
| 320 | 0,38 | 0,19 | 0,24 | 0,15 | 0,22 | 0,72 | 0,87 | 0,62 | 0,62 | 0,31 | −0,28 | −0,06 | 0,082 | 0,34 |

Grupa samych tokenów narzędzia daje to samo (różnice do 0,04). Równanie: jądro pamięci J = 5 plus ściąganie
do średniej z 51 stanów, dopasowane wprost na horyzont, obrazy 0–5 / 6–11 na krzyż, test starty 700–890.

## Co z tego wynika

1. **Źródłem drgań jest optymalizator, nie dane.** AdamW normalizuje krok do stałej długości niezależnie od
   pewności gradientu. Blisko dna gradient zmienia kierunek z kroku na krok, a krok i tak jest pełny, więc sieć
   odbija się po dnie. Większa paczka celniej wskazuje dno (strata 0,9 → 0,27), ale odbijania nie uspokaja.
   Przewidywanie z teorii dyfuzji SGD (wariancja pchnięć ∝ 1/B) nie przenosi się na AdamW.
2. **Przy większej paczce ruch jest mniej gładki.** Spadek autokorelacji przyrostów (0,71 → 0,31) i ujemna
   autokorelacja przy opóźnieniu 2 (−0,28 przy 320) to obraz odbijania o okresie kilku kroków. Przy paczce 32
   szum gradientu uśredniany pędem (β1 = 0,9) nadaje ruchowi bezwładność, która pomaga na 1 krok.
3. **Prognoza na 10 kroków jest taka sama dla wszystkich paczek** (0,20–0,24): przesunięcie w 10 krokach zależy
   od 10 kolejnych kroków optymalizatora, których kierunków nie da się odczytać ze stanu w chwili t.
4. **Sufit wyroczni maleje z paczką** (0,79 → 0,72 na h=10), bo udział powolnego dryfu w ruchu spada
   (D²/σ² 1,18 → 0,62): sieć przy 320 jest dalej w uczeniu (213 epok), więc ma mniej systematycznego ruchu.

## Zastrzeżenia

- Porównanie w tym samym kroku to różne etapy uczenia (21, 85, 213 epok). Nie zmienia to wniosku o drganiach,
  bo ich wielkość i widmo zależą od paczki, nie od epoki, a prognoza na 10 kroków jest taka sama.
- Jedno ziarno na paczkę. Lr bez skalowania z paczką (celowo). BatchNorm i augmentacja bez zmian.

## Czego to NIE dowodzi

- Nie dowodzi, że przy SGD bez normalizacji kroku paczka też nic nie da; tam teoria dyfuzji powinna działać.
- Nie mierzy wpływu β1 (pamięci pędu) ani zaniku lr; to następne, osobne ramię (kod gotowy: `--beta1`).
