# Na ile kroków naprzód Koopman i Langevin przewidują siłę tożsamości (2026-09-29)

Pytanie użytkownika: „jak można wykorzystać [Koopmana i Langevina] do przewidywania latentu w kolejnych krokach do
przodu i na ile kroków do przodu jest w stanie Koopman i Langevin przewidywać; Koopman osobno, Langevin osobno”.

Pomiar lokalny, bez uczenia sieci: `horyzont.py` na zapisanych historiach siły tożsamości (`SILA.npy`, 180 sond,
zapis co 10 kroków) z pętli `../maly_vit_petla_sila_tozsamosci` — wersja 3 (10 ziaren, do kroku 4500) i iteracja 2
(20 ziaren, do kroku 8000), wszystkie ramiona. Bloki po 50 kroków, start prognozy na początku każdego obrotu,
dopasowanie tylko na historii do startu. Modele z `prognoza.py` (ten sam kod co na Edenie), każdy osobno:

- `koopman`, `langevin` — rekurencyjnie, blok po bloku, jak w pętli;
- `koopman_wprost`, `langevin_wprost` — te same cechy, ale osobne dopasowanie dla każdego horyzontu H
  (Y_{t+H} − Y_t wprost, bez składania kroków);
- odniesienia: brak zmiany oraz średnia zmiana z ostatnich 10 bloków przedłużona liniowo.

Miara: umiejętność = 100 · (1 − SSE_model / SSE_brak_zmiany); > 0 znaczy lepiej niż brak zmiany. 15 odpowiada błędowi
RMSE około 0,92 błędu braku zmiany. Wynik: `wyniki/HORYZONT.json`, log `wyniki/horyzont.log`.

## Wyniki (iteracja 2, 20 ziaren, mediana po ziarnach)

| ramię | model | 50 kroków | 100 | 250 | 500 | 750 | 1000 |
|---|---|---|---|---|---|---|---|
| bez | koopman | 12,2 | 16,9 | 15,5 | 14,1 | 12,7 | 13,3 |
| bez | langevin | 12,4 | 17,8 | 18,3 | 15,9 | 7,4 | 8,3 |
| bez | koopman_wprost | 12,2 | 17,6 | 17,0 | 15,4 | 18,6 | 18,2 |
| bez | langevin_wprost | 12,4 | 18,3 | 19,6 | 17,3 | 20,2 | 18,5 |
| losowe | koopman | 16,2 | 19,1 | 17,5 | 14,5 | 13,9 | 12,8 |
| losowe | langevin | 18,7 | 24,6 | 24,5 | 14,7 | 10,7 | 7,5 |
| losowe | langevin_wprost | 18,7 | 25,1 | 24,5 | 15,4 | 22,3 | 20,6 |
| tozsamosc | koopman | 15,0 | 17,8 | 16,6 | 14,9 | 13,3 | 11,3 |
| tozsamosc | langevin | 18,5 | 22,6 | 23,4 | 17,9 | 14,8 | 8,8 |
| tozsamosc | langevin_wprost | 18,5 | 22,4 | 24,8 | 20,6 | 24,9 | 23,6 |
| każde | średnia zmiana | −10 do −13 | −23 do −26 | −66 do −73 | −163 do −171 | −256 do −289 | −382 do −417 |

Odsetek ziaren lepszych od braku zmiany (iteracja 2): Koopman rekurencyjny i oba „wprost” 100 % przy każdym H
(„wprost” Langevin 90–95 % przy 1000 krokach); Langevin rekurencyjny 90–95 % przy 500 i 70–75 % przy 1000 krokach.
Wersja 3 daje ten sam obraz (tylko 6–7 punktów startu na ziarno, więc szum większy).

## Wnioski

1. Nie ma ostrej granicy horyzontu. Obie metody są lepsze od braku zmiany aż do 1000 kroków (20 bloków), ale tylko
   o 10–25 % błędu kwadratowego. To słaby, stały sygnał, a nie prognoza, która jest dobra blisko i psuje się dalej.
2. Siła tożsamości nie ma trendu: przedłużanie ostatniej średniej zmiany jest katastrofalne (−160 przy 500 krokach).
   Obie metody wygrywają, bo przewidują powrót do poziomu sondy i wygaszanie wahań.
3. Langevin rekurencyjny jest najlepszy na 100–250 krokach (18–25), ale składanie kroków psuje go dalej: przy 1000
   krokach spada do 2–9. Koopman rekurencyjny jest płaski (12–17, przy 1000 krokach 9–13).
4. Dopasowanie osobno dla każdego horyzontu („wprost”) trzyma 15–25 aż do 1000 kroków. To zgodne z Lin i in. 2021
   (składanie wielu małych kroków Koopmana mnoży błąd; model dopasowany na docelowy krok jest lepszy).
5. Na horyzoncie pętli (500 kroków): Langevin wprost 15–21, Koopman wprost 15–18, Langevin rekurencyjny 15–18,
   Koopman rekurencyjny 13–15.

## Czego to nie dowodzi

- Dotyczy jednej liczby na sondę (siły tożsamości), nie całego latentu 768.
- Start prognozy tylko na początkach obrotów (14 na ziarno w iteracji 2); przy H > 10 prognoza przechodzi przez
  kolejne dołożenie klatek.
- Warianty „wprost” używają tych samych cech co pętla; nie sprawdzono pełnego równania Langevina z projektu TFT
  (dryf od prędkości i przyspieszenia, jądro pamięci z tłumionymi obrotami, człon wymuszenia, człon zdarzeń, szum barwny).

## Literatura (Undermind, workspace ViT, deep search „multi-step forecasting of a network latent with Koopman and generalized Langevin equations and the predictability horizon”, 205 prac)

- Lin i in. 2021, SIAM J. Appl. Dyn. Syst., doi:10.1137/21m1401759 — dyskretne równanie GLE z Mori-Zwanziga;
  prognoza w pętli otwartej na własnych przewidywaniach, szum ustawiony na zero; przy 1,3 czasu Lapunowa błąd:
  Koopman składany z małych kroków ≈ 105–165, Koopman na docelowym kroku ≈ 73, z pamięcią ≈ 48.
- Lei, Baker, Li 2016, PNAS, doi:10.1073/pnas.1609587113 — jądro pamięci jako suma wykładników zamienione na zmienne
  pomocnicze; szum barwny generowany białym szumem w tych zmiennych (zgodnie z twierdzeniem fluktuacyjno-dyssypacyjnym).
- Lien i in. 2024, Phys. Rev. Research, doi:10.1103/PhysRevResearch.7.023042 — LIM z szumem barwnym; poprawia
  kalibrację rozrzutu zespołu prognoz, nie wydłuża horyzontu średniej prognozy.
- Aceituno i in. 2025, TMLR, doi:10.48550/arXiv.2506.03889 — horyzont uczenia jako hiperparametr (krzywa U); uczenie tylko na
  jednym kroku rzadko jest optymalne; szum ogranicza horyzont logarytmicznie.
- Pervez, Locatello 2026, doi:10.48550/arXiv.2605.08856 — przejściowe wzmocnienie operatorów nienormalnych psuje
  długie prognozy mimo stabilnych wartości własnych; kara za nienormalność i niekomutowanie.
- Mamakoukas i in. 2020, doi:10.1109/TRO.2022.3228130 — stabilny Koopman (promień spektralny ≤ 1).
- Dogra, Redman 2020, arXiv:2006.02361 — Koopman przewiduje wagi sieci 2500 kroków naprzód, ale późno w uczeniu, blisko
  minimum; zawodzi wcześnie w uczeniu. Tano i in. 2020, arXiv:2006.14371 — DMD z 14 zapisów najlepiej na ~41 kroków naprzód.
- Kunin i in. 2021, Neural Comput., doi:10.1162/neco_a_01626 — SGD z momentem jako niedotłumione równanie Langevina,
  czas korelacji ~ η(1+β)/(1−β) kroków, obroty w przestrzeni fazowej.
- Pashakhanloo, Koulakov 2023, doi:10.48550/arXiv.2302.02563 — szum SGD daje błądzenie przypadkowe reprezentacji wzdłuż
  kierunków, które nie zmieniają straty; odczyt zadania pozostaje stabilny.
