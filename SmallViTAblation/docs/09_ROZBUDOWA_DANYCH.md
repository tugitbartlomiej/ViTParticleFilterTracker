# 09 — Jak szybko poprawić zbiór, dokładając zdjęcia

Cztery drogi, uszeregowane według stosunku „ile zamyka zarzutów" do „ile
kosztuje". Pierwsza jest zdecydowanie najmocniejsza.

---

## 1. Druga instytucja: Cataract-1K [NAJWIĘKSZY ZYSK]

**Cataract-1K** — Ghamsarian i in., *Scientific Data* 2024,
DOI `10.1038/s41597-024-03193-4`, kod i dane:
https://github.com/Negin-Ghamsarian/Cataract-1K

- **tysiąc nagrań operacji zaćmy** z Klinikum Klagenfurt w Austrii, 2021–2023,
- adnotacje pikselowe (maski) dla **2 256 klatek z 30 nagrań**: trzy struktury
  anatomiczne i **dziewięć narzędzi**,
- adnotacje faz operacji dla 56 nagrań.

Dlaczego to jest najmocniejszy ruch:

- **Zamyka zarzut 9 recenzenta 2** — „single institution" przestaje być prawdą.
  Artykuł zmienia się z „jedna klinika" na pracę wielo­ośrodkową, a to jest inna
  kategoria wiarygodności.
- **Daje drugi niezależny zbiór zewnętrzny.** Recenzent 2 napisał wprost:
  „additional independent external datasets would considerably strengthen the
  paper". Teraz masz jeden benchmark z 2 333 obrazów; drugi z innego kraju
  i innego sprzętu to zupełnie inny poziom dowodu na generalizację.
- **Maski zamieniasz na ramki jednym skryptem** — obwiednia maski to ramka.
  Zero pracy adnotacyjnej.
- Pacjenci są nowi, więc podział po pacjencie zostaje trywialny.

Ograniczenie: to jest **dziewięć klas narzędzi**, a twoje zadanie to jedna klasa
(końcówka narzędzia). Trzeba zdecydować, czy scalasz wszystkie dziewięć w jedną
klasę, czy mapujesz tylko te, które odpowiadają twojej definicji. Ta decyzja
musi trafić do artykułu jednym zdaniem, bo wpływa na porównywalność.

**Koszt:** pobranie i konwersja masek na ramki — dzień pracy, zero GPU.

## 2. Klatki bez narzędzia — trudne negatywy [NAJTAŃSZY ZYSK NA JAKOŚCI]

W zbiorze **99,1% klatek ma dokładnie jedno narzędzie** (zmierzone przy okazji
treningu 90k video-split). To znaczy, że model prawie nigdy nie widział sceny,
w której narzędzia **nie ma** — a takich klatek w prawdziwej operacji jest dużo:
przed wprowadzeniem narzędzia, przy wymianie, przy przemywaniu.

Skutek widać w liczbach: model z artykułu ma **550 fałszywych wykryć przy 835
trafnych**, czyli precyzję około sześćdziesięciu procent. Model „widzi"
narzędzie tam, gdzie go nie ma, bo nigdy nie nauczył się, że może go nie być.

Masz już do tego gotowy pipeline z sesji 2025-08-09: wyciąganie klatek tła
z wideo i klasyfikacja przez zgodność DINO, YOLO i DETR. Trzy modele muszą się
zgodzić, że narzędzia nie ma — dopiero wtedy klatka wchodzi jako negatyw.

**Koszt:** pipeline istnieje, trzeba go puścić na 35 nagraniach treningowych.
**Uwaga:** negatywy dokładamy **tylko do pacjentów treningowych**, nigdy do
walidacji ani testu, bo inaczej zmieniasz podstawę porównania ze starymi
liczbami.

## 3. Gęstsze próbkowanie tych samych nagrań [ŚREDNI ZYSK, WYMAGA ADNOTACJI]

Masz **22 837 klatek źródłowych z 50 nagrań**, czyli około 457 klatek na
operację. Operacja zaćmy trwa kilkanaście minut, więc przy 25 klatkach na
sekundę nagranie ma dziesiątki tysięcy klatek. Wyciągnięto z nich mniej niż
dwa procent.

Więcej klatek z tych samych nagrań **nie doda nowych pacjentów**, więc nie
zamyka żadnego zarzutu recenzenta. Ale daje gęstsze pokrycie faz operacji,
w tym tych rzadkich, które teraz mogą być reprezentowane przez kilka klatek.

Sposób, który pasuje do narracji artykułu o kuracji danych: **pseudo-etykiety
plus weryfikacja tam, gdzie model jest niepewny**. Model oznacza nowe klatki,
człowiek ogląda tylko te o niskiej pewności. To jest uczenie aktywne i wpisuje
się w temat pracy zamiast być doklejką.

**Koszt:** kilka dni pracy adnotacyjnej, nawet przy weryfikacji wybiórczej.

## 4. Inne publiczne zbiory zaćmy [DO ROZWAŻENIA]

Rodzina zbiorów z tej dziedziny: CATARACTS (masz), Cataract-101, CaDIS,
LensID. Każdy dokłada pacjentów i sprzęt, ale każdy ma inną definicję klas
i inny format adnotacji — koszt scalania rośnie szybko.

Rekomendacja: **wziąć jeden dobrze, a nie trzy byle jak**. Cataract-1K jest
najlepszym pierwszym wyborem, bo ma świeżą publikację w Scientific Data,
adnotacje instrumentów i jasną licencję.

---

## Co bym zrobił w tej kolejności

| # | Ruch | Zamyka | Koszt |
|---|---|---|---|
| 1 | Cataract-1K jako **drugi zbiór zewnętrzny** (tylko ewaluacja) | zarzut 9 recenzenta 2, „single institution"; prośbę o dodatkowy niezależny zbiór | dzień, bez GPU |
| 2 | Klatki bez narzędzia do części treningowej | fałszywe wykrycia, precyzja | pipeline istnieje |
| 3 | Cataract-1K także **do treningu** (wersja wieloośrodkowa) | zmienia artykuł na wieloośrodkowy | trzeba przemyśleć mapowanie klas |
| 4 | Gęstsze próbkowanie z adnotacją półautomatyczną | nic z listy recenzentów, ale poprawia zbiór | kilka dni pracy ręcznej |

Ruch pierwszy jest tani i sam w sobie wart osobnego zdania w abstrakcie.
Ruch drugi jest tani i naprawia najbardziej widoczną słabość modelu.
Reszta może poczekać, aż nowy protokół pokaże, czego naprawdę brakuje.

**Ważne dla protokołu:** cokolwiek dokładasz do **treningu**, musi wejść przed
zbudowaniem cech i selekcją, inaczej cała macierz z `02_MATRIX.md` liczy się na
innym zbiorze niż tabele końcowe. Dokładanie do **ewaluacji** można zrobić
w dowolnym momencie, bo to osobny zbiór.

---

## Odnośniki

- Cataract-1K, *Scientific Data* 2024 — https://doi.org/10.1038/s41597-024-03193-4
- Repozytorium Cataract-1K — https://github.com/Negin-Ghamsarian/Cataract-1K
- Preprint — https://arxiv.org/abs/2312.06295
