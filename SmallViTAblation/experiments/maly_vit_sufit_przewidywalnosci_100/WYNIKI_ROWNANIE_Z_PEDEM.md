# Wynik: równanie Langevina na 1..10 kroków z pędem optymalizatora i granice z przyszłymi paczkami

Data: 2026-10-03. Kod: `kod/rownanie_z_pedem_10.py`, liczby: `wyniki/rownanie_z_pedem/ROWNANIE_Z_PEDEM.json`
(pierwszy bieg z błędną aktualizacją pędu w granicy: `..._v1_bez_aktualizacji_pedu.json`).
Dane: nagranie `maly_vit_rownanie_latentu/nagranie_s47_sily` i `_s48_sily` (docs/23): latent 180 sond
z narzędziem co krok, siły w stanie t przepuszczone przez sieć (pęd, zanik wag, nowa paczka, pełny krok).
Dopasowanie: starty 1000–1999 ziarna 47 (cele do 2000). Test: starty 2000–2990 ziarna 47 oraz ziarna 48
(te same współczynniki, bez ponownego dopasowania). Miara: 1 − SSE / SSE prognozy bez zmian.

## Wynik (ziarno 47; ziarno 48 w nawiasie), horyzont 1 / 5 / 10 kroków

| model | co zna w chwili t | h=1 | h=5 | h=10 |
|---|---|---:|---:|---:|
| historia latentu (J=5, ściąganie do średniej), wprost | stany do t | 0,60 (0,60) | 0,25 (0,24) | 0,21 (0,22) |
| sam pęd i zanik wag, wprost | stan AdamW | 0,66 (0,64) | 0,18 (0,15) | 0,04 (0,04) |
| historia + pęd + zanik, wprost | stany do t, stan AdamW | **0,68 (0,67)** | 0,28 (0,27) | **0,22 (0,23)** |
| historia + pęd + zanik, równanie iterowane 10 razy | jak wyżej | 0,68 (0,67) | 0,16 (0,10) | −0,17 (−0,20) |
| granica: + prawdziwa siła każdej przyszłej paczki, pęd liczony liniowo | przyszłe paczki | 0,95 (0,96) | 0,75 (0,78) | 0,21 (0,24) |
| granica: + prawdziwa PEŁNA siła każdego przyszłego kroku | przyszłe paczki i stan sieci w każdym kroku | 0,96 (0,96) | 0,99 (0,99) | **0,985 (0,983)** |

Współczynniki jednego kroku (wspólne dla 768 współrzędnych i 180 sond): x−μ −0,0003; Δx_t 0,31; Δx_{t−1}
−0,34; Δx_{t−2} 0,02; Δx_{t−3} 0,00; Δx_{t−4} 0,03; S_pęd 0,69; S_zanik 1,1. Z pełną siłą: S_suma 1,02.
Przenoszą się na ziarno 48 bez straty (różnice do 0,04).

## Co z tego wynika

1. **Pęd optymalizatora pomaga tylko na 1 krok** (0,60 → 0,68, zgodnie z docs/23: 64–67 procent samym pędem).
   Na 10 kroków daje 0,22, tyle co sama historia. Pęd AdamW gaśnie w około 10 kroków, a nowe gradienty
   10 kolejnych paczek dominują przesunięcie.
2. **Równanie liniowe jest wystarczające.** Z prawdziwą pełną siłą każdego kroku iterowane równanie daje
   0,985 na 10 kroków. Cały brak przewidywalności to brak siły z przyszłych paczek, nie forma równania.
3. **Siła paczki bez przeliczania sieci w nowym stanie nie wystarcza.** Znając przyszłe paczki, ale licząc
   pęd liniowo z sił w stanie t, równanie daje 0,75 na 5 kroków i 0,21 na 10. Siła tej samej paczki zależy od
   stanu sieci; po kilku krokach stan odpływa i liniowe składanie sił się psuje.
4. Iterowanie równania jednego kroku bez przyszłych sił jest gorsze niż dopasowanie wprost na horyzont
   (−0,17 wobec 0,22 na 10 kroków): błędy iteracji się składają, a prognoza wprost uczy się ściągać do średniej.

## Czego to NIE dowodzi

- Nie dotyczy pojedynczych tokenów (tu średnie po tokenach narzędzia 180 sond); dla tokenów historia daje
  tyle samo (0,22, `WYNIKI_CO_JESZCZE.md`).
- Granice z przyszłymi paczkami łamią regułę prognozy z docs/23; są diagnozą, nie metodą.
- Dwa ziarna, kroki 1000–3000, paczka 32, jedna architektura.
