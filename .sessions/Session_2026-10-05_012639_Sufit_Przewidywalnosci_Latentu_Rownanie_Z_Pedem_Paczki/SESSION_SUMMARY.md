# Podsumowanie sesji 2026-10-02 .. 2026-10-05

Rozmowa prowadzona z katalogu projektu grantowego (`F:\Studia\MCHTR\Grant\Projekt`), ale cała praca merytoryczna
dotyczyła repozytorium ViT, więc zapis jest tutaj. Gałąź repo ViT w chwili zapisu: `codex/eden-nano-training-20260914`.

## Cel sesji

Użytkownik chciał prognozy latentu małego ViT w trakcie uczenia lepszej o ponad 85 procent od prognozy „bez zmian”,
najpierw na 100 kroków, potem na 10 kroków (dwa checkpointy). Punkt wyjścia: raport NNDMD
(`SmallViTAblation\experiments\maly_vit_nndmd_lokalne_przyrosty_hopper\run1819918\RAPORT_KONCOWY.md`), w którym
model przegrał z prognozą bez zmian o 14,8 procent na checkpointach 90 do 100.

## Kontekst

- Dane: gęsta historia tokenów (bieg Eden 1818732, ziarno 47, kroki 0..1000 co 1, 12 obrazów, 600 tokenów,
  768 liczb; 22 GB lokalnie), nagranie sił z docs/23 (`maly_vit_rownanie_latentu\nagranie_s47_sily`, `_s48_sily`),
  latent sond 10 ziaren co 10 kroków (`maly_vit_od_zera\wyniki_eden\petla_s47..s56`).
- Wcześniejsze sesje: `Session_2026-10-02_145632` (NNDMD hopper), sesje z 1 października (gęsta historia, animacje).
- Przy okazji na początku: zaplanowanie wyłączenia komputera (`shutdown /s /t 14400`) i podsumowanie animacji
  ManimGL z poprzedniej sesji (`SmallViTAblation\docs\animacje_tozsamosc_pary400`).

## Wykonane kroki

1. Diagnoza NNDMD: na walidacji przegrywał już po 1 epoce; kosinus przewidzianej i prawdziwej zmiany 0,03–0,15,
   znak trafiony w około 50 procentach, promień widmowy macierzy 1,05–1,08 (wzmacnia ruch). Średnia z 3 ostatnich
   stanów biła prognozę bez zmian, przedłużenie ostatniej zmiany było 7 razy gorsze: ruch zawraca.
2. Agent Explore sprawdził w repo, jak liczono kierunek LDA i prognozę rzutu: LDA w kroku 1000; prognoza rzutu
   działała na 1 krok, a odszumiało ją uśrednianie po tokenach, nie sam kierunek (losowe kierunki dawały to samo).
3. Literatura: Undermind deep search „wyznaczanie dryfu i szumu ruchu latentu sieci podczas uczenia SGD
   z zaszumionych checkpointow” (242 prace); ważne: Stochastic Force Inference (Frishman, Ronceray 2018),
   Langevin regression (Callaham i in. 2020), DCA (Clark i in. 2019), dryf reprezentacji pod SGD
   (Pashakhanloo, Koulakov 2023), granica SGD (Kunin i in. 2021), PCA błądzenia losowego (Antognini, Sohl-Dickstein 2018).
4. Nowy eksperyment `maly_vit_sufit_przewidywalnosci_100`: sufit wyroczni (gładka ścieżka przyszłości), podłoga
   (samo odszumienie startu), autokorelacja przyrostów, model liniowy z historii dopasowany krzyżowo między
   połowami obrazów. Wynik na 100 kroków dla tokenów narzędzia: sufit 0,87–0,89 (wygładzanie 21), 0,75–0,77 (51);
   osiągnięte 0,16–0,31. Trwałość dryfu ujemna, przedłużenie trendu ujemne. LDA nie lepszy od losowego kierunku.
   Bieg przerwany na chwilę przez brak RAM (zatrzymany był tylko proces czekający), zakończony w 1580 s.
5. Prognoza od kroku 900 do 1000 krok po kroku (`prognoza_900.py`, `wykres_900.py`, `PROGNOZA_900.png`):
   równanie OU (ściąganie do średniej z 51 lub 101 kroków, κ 0,03–0,055 na krok, połowa odchylenia znika w 13–23
   krokach) daje 0,22–0,27 dla tokenów; NNDMD −0,09 do −0,52 na checkpointach 91..100 (ten sam bieg wag, inne obrazy).
6. Test bogatszych modeli (`co_jeszcze_z_historii.py`): 768 osobnych współczynników, cechy wspólnego ruchu, OU
   wspólny i własny: wszystko 0,22–0,27 na 10–100 kroków. Wspólny ruch wszystkich tokenów to 20 procent błędu.
   Pierwszy bieg miał błąd znaku w dopasowaniu OU (κ = 0); poprawiony i przeliczony, stary plik zachowany jako
   `CO_JESZCZE_blad_znaku_ou.json`.
7. Autokorelacja przyrostów do 150 kroków: 0,69 przy 1, minimum −0,24 przy 4, zero od około 10. Wczesna faza na
   10 ziarnach (`wczesna_faza_10_ziaren.py`): prognoza na 100 kroków 0,11.
8. Jądro pamięci J = 5..40 na 1..10 kroków (`jadro_pamieci_10.py`): 0,63 na 1 krok, 0,22 na 10, długość jądra
   bez znaczenia. Bieg zatrzymany ręcznie po wariantach J = 5..40 fold 0 (wynik jasny, zwalniał RAM).
9. Równanie z pędem optymalizatora (`rownanie_z_pedem_10.py`, nagranie sił, ziarno 47, przeniesienie na 48):
   historia + pęd + zanik 0,68 / 0,28 / 0,22 na 1 / 5 / 10 kroków; iterowane −0,17 na 10. Granice: znane przyszłe
   paczki z pędem liniowym 0,75 na 5 i 0,21 na 10; znana pełna siła każdego kroku 0,985 na 10 kroków.
   Wniosek: forma liniowego równania wystarcza, brakuje siły z przyszłych paczek.
10. Eden: kwota 167/200 GB. Usunięto zweryfikowaną kopię `LATENT_GESTY.npy` biegu 1818732 (22 GB; lokalna
    kopia z SHA256 zgodnym) za zgodą użytkownika („możesz na edenie usuwać stare”).
11. Nowy eksperyment `maly_vit_gesta_historia_paczka`: kod uczenia skopiowany z gęstej historii, dodane `--paczka`,
    `--tokeny-na-obraz` (60 tokenów na obraz, 2,2 GB zamiast 22 GB), później `--beta1`. Biegi Eden: próba 1820332,
    paczka 320 (1820333), paczka 128 (1820334), H100, po 3–5 minut. Dane pobrane, SHA256 zgodne.
    Wynik: prognoza na 10 kroków 0,20 / 0,24 / 0,24 dla 32 / 128 / 320, względny krok bez zmian, autokorelacja
    po 1 kroku 0,71 / 0,50 / 0,31. Hipoteza z planu obalona: drgania pochodzą z AdamW, nie z losowania paczki.
12. Przygotowano (bez uruchomienia) ramię z β1 = 0,99 (`sbatch bieg_paczka.slurm pelny 32 0.99`); czeka na zgodę.
13. Raport PDF 27 (10 stron, 8 rysunków, architektura, potok pomiaru, wzory, tabele z JSON):
    `SmallViTAblation\docs\27_PRZEWIDYWANIE_LATENTU_RAPORT.pdf`, wysłany też na drugie urządzenie użytkownika.

## Pliki zmienione / utworzone

### W repozytorium ViT (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`)

- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_sufit_przewidywalnosci_100\`
  (PLAN.md, README.md, WYNIKI.json, WYNIKI.md, WYNIKI_PROGNOZA_900.md, WYNIKI_CO_JESZCZE.md, WYNIKI_ROWNANIE_Z_PEDEM.md,
  kod: dane.py, sufit.py, tabela.py, prognoza_900.py, wykres_900.py, co_jeszcze_z_historii.py, jadro_pamieci_10.py,
  autokorelacja_przyrostow.py, wczesna_faza_10_ziaren.py, rownanie_z_pedem_10.py; wyniki: pelny, proba, prognoza_900,
  co_jeszcze, jadro_pamieci, autokorelacja, wczesna_faza, rownanie_z_pedem)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_gesta_historia_paczka\`
  (PLAN.md, README.md, WYNIKI.json, WYNIKI.md, bieg_paczka.slurm, kod: bieg.py, trening.py, dane.py, model.py,
  porownaj_paczki.py, tabela.py; wyniki: pelny_p128, pelny_p320 z `dane\LATENT_GESTY.npy` po 2,2 GB, porownanie,
  porownanie_p128). Duże pliki .npy nie powinny trafić do gita.
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\27_PRZEWIDYWANIE_LATENTU_RAPORT.pdf`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\27_raport_przewidywanie_latentu\`
  (wykresy.py, zbuduj_pdf.py, rys\*.png)

### Zewnętrzne

- Eden `~/smallvit_runs/2026-10-03_maly_vit_gesta_paczka/` (kod, slurm, wyniki proba_p320, pelny_p320, pelny_p128).
- Eden: usunięty `~/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane/LATENT_GESTY.npy`.
- Undermind, workspace grantowy `bb70909d-8a94-4765-867e-83324d7e9b7b`: nowe deep search w folderze
  „02 Mori Zwanzig i Langevin”.
- W projekcie grantowym nic nie zmieniono.

## Otwarte problemy

- Cel 85 procent na 10 kroków nieosiągnięty prognozą bez przyszłych paczek (najlepiej 0,22–0,24).
- Strona 2 raportu PDF ma puste miejsce (schemat potoku przeszedł na stronę 3).
- Bieg `jadro_pamieci_10.py` nie zapisał JSON (zatrzymany); wyniki są w jego LOG.txt.
- Wyniki opierają się na jednym ziarnie dla gęstej historii i na średnich po tokenach narzędzia sond dla równania z pędem.

## Następne kroki

- Decyzja użytkownika: bieg z β1 = 0,99 (paczka 32), który powinien przenieść część przyszłej siły do pędu
  znanego w chwili prognozy. Kod gotowy, około 5 minut GPU, 2 GB.
- Jeśli β1 0,99 pomoże: powtórzyć na 2–3 ziarnach i sprawdzić mAP.
- Rozważyć prognozę „z planem paczek” w pętli doboru klatek (sieć przeliczana na próbę), skoro forma równania
  z pełną siłą daje 0,985.
