# Jakość adnotacji końcówki narzędzia w Cataract-LMM (3_Object_Tracking)

Data: 2026-09-11. Próbka: `Dataset/Cataract-LMM/_probka_koncowek/`.

## Po co to zmierzyliśmy

Zbiór podaje współrzędne końcówki cystotomu na 88,5% klatek. Podgląd pokazał,
że część znaczników stoi w miejscu, gdzie nie widać narzędzia. Zanim uznamy te
adnotacje za wzorzec do oceny własnej propagacji ramek, trzeba wiedzieć, jaka
część z nich jest trafna.

## Jak losowaliśmy

`scripts/pobieranie/probka_koncowek_do_oceny.py`, ziarno 1606:
40 losowych klipów × 5 losowych klatek z podaną końcówką. Wyszło 196 próbek
(część klipów miała mniej niż 5 klatek z końcówką). Każda próbka to wycinek
160×160 px wokół punktu, powiększony dwukrotnie, ze znacznikiem. Arkusze po 25.

## Wynik

| werdykt | ile | udział |
|---|---:|---:|
| T — krzyżyk na czubku narzędzia | 180 | 91,8% |
| N — nie da się rozstrzygnąć (rozmycie, narzędzie nieczytelne) | 12 | 6,1% |
| P — pudło, znacznik w pustym miejscu | 4 | 2,0% |

Trafność ostro (tylko T): **91,8%**, przedział Wilsona 95%: **87,2–94,9%**.
Przy założeniu, że niepewne są trafne: 98,0% (94,9–99,2%).
Realna trafność leży więc gdzieś w przedziale 87–98%, zależnie od tego, jak
policzyć klatki, na których w ogóle nie widać narzędzia.

Pudła: #023 (TR_0023 kl.1698), #088 (TR_0069 kl.25), #157 (TR_0131 kl.25),
#173 (TR_0139 kl.628). Werdykty per próbka: `_probka_koncowek/werdykty.json`.

## Co z tego wynika

1. **Błędy nie są rozłożone równomiernie po klipach.** Osiem z szesnastu
   nie-trafień pochodzi z czterech klipów: TR_0024 (3), TR_0133 (3), TR_0006 (2).
   TR_0133 to klip z wypalonym napisem „Hospital OR 2-3" wchodzącym w kadr i
   bardzo silnym rozmyciem — tam ocena wzrokowa w ogóle nie rozstrzyga.
   Wniosek praktyczny: filtrowanie po jakości klipu usunie większość złych
   adnotacji taniej niż czyszczenie klatka po klatce.
2. **Dwie klatki z czterech pudeł to kl.25 — początek klipu.** Warto sprawdzić,
   czy pierwsze kilkadziesiąt klatek każdego klipu nie jest systematycznie gorsze
   (rozgrzewanie się trackera, narzędzie jeszcze poza polem widzenia).
3. **To nie jest złoty standard, tylko punkt odniesienia.** 92% trafień oznacza,
   że własna propagacja ramek nie musi być idealna, żeby dorównać temu zbiorowi.
   Nie oznacza, że można na tym liczyć błąd lokalizacji w pikselach.

## Ograniczenia tego pomiaru

- Ocena wzrokowa modelu na wycinkach 160 px, nie ocena okulisty. Kategoria „na
  czubku" oznacza „znacznik leży na końcu widocznego trzonu", bez tolerancji
  wyrażonej w pikselach.
- Warstwowanie 40 klipów × 5 klatek zawyża wagę krótkich klipów względem
  losowania po klatkach. Do porównania z własnym zbiorem wystarczy, do
  raportowania „trafność zbioru" — nie.
- Nie sprawdzaliśmy 11,5% klatek BEZ podanej końcówki. Tam może być druga,
  większa pula błędów: końcówka widoczna, a nieoznaczona.

---

# Jakość RAMEK narzędzia

Ta sama metoda, dwie osobne próbki po 100 ramek, to samo ziarno 1606.
Skrypt: `scripts/pobieranie/probka_ramek_do_oceny.py`.
Werdykty: `_probka_ramek_seg/werdykty.json`, `_probka_ramek_trk/werdykty.json`.

Kategorie: **T** ramka ciasno obejmuje narzędzie, **L** narzędzie jest, ale
ramka luźna, przesunięta albo ucina jego część, **P** w ramce nie ma narzędzia.

## 2_Instance_Segmentation — 100 ramek, 9 klas narzędzi

| werdykt | ile |
|---|---:|
| T ciasna | 91 |
| L luźna lub przesunięta | 9 |
| P pudło | 0 |

Ciasne: **91,0%**, Wilson 95%: **83,8–95,2%**. Ani jednego pudła.

Błąd nie rozkłada się równo po klasach. **Lens-Injector: 4 złe na 6**.
Injektor to przezroczysta plastikowa tuba i na wielu klatkach jego granica jest
nieczytelna — adnotator zaznaczał obszar, a nie kształt. Forceps 2/28,
Cap-Cystotome 2/12, Second-Instrument 1/10, reszta klas bez błędu.

## 3_Object_Tracking — 100 ramek cystotomu, 20 klipów po 5 klatek

| werdykt | ile |
|---|---:|
| T ciasna | 89 |
| L luźna lub przesunięta | 8 |
| P pudło | 3 |

Ciasne: **89,0%**, Wilson 95%: **81,4–93,7%**.

Znowu skupisko w tych samych klipach: TR_0133 daje 4 z 11 błędów, TR_0022 dwa.
TR_0133 i TR_0024 wypadły źle także przy końcówkach. To potwierdza, że filtr
jakości na poziomie klipu załatwia obie wady naraz.

## Zestawienie trzech pomiarów

| co | trafnych | przedział 95% |
|---|---:|---|
| końcówka (tracking, 196 próbek) | 91,8% | 87,2–94,9% |
| ramka (segmentacja, 100 próbek) | 91,0% | 83,8–95,2% |
| ramka (tracking, 100 próbek) | 89,0% | 81,4–93,7% |

Trzy niezależne pomiary dają ten sam rząd wielkości: **około 90% adnotacji jest
dobrych, około 10% wymaga poprawy albo odrzucenia**. Przedziały się nakładają,
więc nie ma podstaw, żeby twierdzić, że któreś źródło jest istotnie lepsze.

## Co z tego wynika dla treningu

1. Segmentacja jest czystsza tam, gdzie narzędzie jest metalowe i ostro odcięte
   od tła. Problem robią narzędzia przezroczyste. Przy jednej klasie
   `instrument` te 6 ramek injektora to margines, ale w statystykach per klasa
   byłby to poważny błąd — kolejny argument za jedną klasą.
2. Pudeł w segmentacji nie ma wcale, w trackingu są trzy na sto. Tracking jest
   generowany półautomatycznie i to widać.
3. Odrzucenie kilku najgorszych klipów podniesie jakość obu rodzajów adnotacji
   naraz, bo błędy siedzą w tych samych nagraniach.

## Ograniczenia

Ocena wzrokowa modelu, nie okulisty. „Ciasna" znaczy „krawędzie ramki leżą przy
krawędziach narzędzia", bez progu w pikselach ani liczonego [IoU] (aj-oł-ju).
Żeby dostać liczbę odporną na taką subiektywność, trzeba by przerysować ramki
ręcznie i policzyć pokrycie — to zadanie na osobną sesję.
