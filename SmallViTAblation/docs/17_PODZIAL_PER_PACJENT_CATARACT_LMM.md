# Czy jeden klip Cataract-LMM to jeden pacjent

Data: 2026-09-11. Narzedzia: `scripts/pobieranie/audyt_pacjentow.py`,
`scripts/pobieranie/stemple_czasowe.py`. Wyniki: `Dataset/Cataract-LMM/_audyt_pacjentow.json`.

## Odpowiedz krotko

**Prawie.** 170 plikow w `3_Object_Tracking` to **167 roznych operacji**, nie 170.
Podzial per pacjent jest wykonalny, ale nie wolno go robic po nazwie pliku.

## Co mowi dokumentacja

Nazwa `TR_0001_S1_P03` rozklada sie tak:
`TR` podzbior sledzenia, `0001` kolejny numer klipu **w tym podzbiorze** (0001-0170),
`S1` szpital Farabi, `P03` **faza operacji** (kapsuloreksja) — nie pacjent.

To jest pierwszy problem. Segmentacja ma w nazwie **Global ID**
(`SE_0001_0003_S1`, gdzie `0003` wskazuje nagranie w korpusie 3000 procedur).
Sledzenie tego identyfikatora **nie ma**. Z samej nazwy nie da sie wiec
powiedziec ani czy dwa klipy sledzenia to ten sam pacjent, ani czy klip
sledzenia i klatki segmentacji pochodza z tej samej operacji.

## Co znalazlem w danych

Trzy niezalezne testy, wszystkie zgodne.

**1. Skrot dokladny** (trzy klatki na klip). Dwie pary plikow sa identyczne:

| pliki | klatek | werdykt |
|---|---:|---|
| TR_0070 i TR_0138 | 1 364 | ten sam material pod dwiema nazwami |
| TR_0088 i TR_0144 | 1 760 | ten sam material pod dwiema nazwami |

**2. Odcisk percepcyjny** (male szare miniatury 12 klatek, porownanie korelacja).
Powyzej progu 0,9 wychodza tylko te dwie pary (podobienstwo 1,000) oraz trzecia:

| pliki | podobienstwo | klatek |
|---|---:|---:|
| TR_0045 i TR_0049 | 0,924 | 1 680 i 1 629 |

Ta trzecia para nie jest kopia bajtowa — zadna klatka nie ma tego samego skrotu.
Ale klatka 800 z TR_0045 i klatka 750 z TR_0049 roznia sie srednio o **4,2 na 255**
poziomu jasnosci. Rozne ujecia daja 20 do 60. To ten sam material, przesuniety
o 50 klatek i zapisany ponownie.

Pozostale 17 par miesci sie w przedziale 0,75 do 0,80 i to jest zwykle
podobienstwo miedzy operacjami: ta sama sala, ten sam mikroskop, podobne oko.
Stemple czasowe potwierdzaja, ze to rozne dni.

**3. Stempel czasowy wypalony przez mikroskop.** Kadr S1 zawiera nazwe sali oraz
date i godzine operacji. Odczytalem je wzrokowo ze **161 klipow** (w 9 napis
wychodzi poza kadr). Zadne dwa klipy poza wyzej wymienionymi nie maja tej samej
pary data-godzina. Pary potwierdzajace:

- TR_0045 `1/30/2021 09:07:47` i TR_0049 `1/30/2021 09:07:43` — cztery sekundy roznicy
- TR_0070 i TR_0138 `2011/12/21 10:24:39` — identyczny
- TR_0088 i TR_0144 `4/13/2016 18:09:49` — identyczny

## Wniosek dla protokolu

1. **Grupuj po operacji, nie po nazwie pliku.** Gotowe grupy sa w
   `_audyt_pacjentow.json` pod kluczem `grupy_operacji`. Sledzenie daje
   **167 jednostek podzialu**, nie 170.
2. **Nie laczy sie sledzenia z segmentacja w jednym podziale**, dopoki autorzy
   nie udostepnia Global ID dla sledzenia. Segmentacja ma 151 nagran z jawnym
   identyfikatorem, sledzenie 167 bez niego. Nie da sie wykluczyc, ze niektore
   to te same operacje. Bezpieczna opcja: jeden podzbior do uczenia, drugi
   **wylacznie** jako zewnetrzny zbior testowy.
3. **Napisz do autorow po mapowanie.** Global ID dla `3_Object_Tracking`
   rozwiazalby oba problemy naraz. Przy okazji trzeba wyjasnic licencje,
   podana w trzech sprzecznych wersjach.

## Ograniczenia

Odczyt stempli jest wzrokowy, bo w srodowisku nie ma OCR. Dziewiec klipow ma
napis poza kadrem i dla nich zostaja tylko testy 1 i 2 — a te nie wskazaly
zadnego dopasowania. Prog 0,9 dla odcisku percepcyjnego dobralem po tym, jak
zobaczylem przerwe miedzy 0,924 a 0,797; przy innym zbiorze trzeba go dobrac
na nowo.
