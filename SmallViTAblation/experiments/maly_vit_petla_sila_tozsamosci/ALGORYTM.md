# Algorytm pętli, wersja 3 (tak, jak liczy się na Edenie, kampania 1818025)

Kod: `kod/petla.py` (przebieg), `kod/sila.py` (wektor, prognoza, FastGAN), `kod/wybor.py` (wybór klatek).

1. **Rozgrzewka.** Mały ViT uczy się od zera na bazie 1500 klatek przez 1000 kroków (21 epok). Rozgrzewka jest wzięta
   gotowa z drugiej kampanii, z tych samych ziaren 47–56.
2. **Zapis latentu.** Co 10 kroków sieć patrzy na 192 zdjęcia testowe (sondy: 12 wycinków narzędzia w 16 wersjach).
   Z każdego bierze 768 liczb na narzędziu i 768 na tle.
3. **Wektor tożsamości.** Z zapisów tuż przed krokiem 1000 liczę kierunek w tych 768 liczbach, który najlepiej
   oddziela narzędzie od tła (LDA). Ten kierunek zostaje stały do końca biegu.
4. **Siła tożsamości.** Dla każdej ze 180 sond z narzędziem liczę jedną liczbę: jak daleko jej latent leży wzdłuż
   wektora tożsamości. Historię siły uśredniam w bloki po 50 kroków.
5. **Początek obrotu** (7 obrotów po 500 kroków, około 10 epok każdy):
   1. sprawdzam prognozę z poprzedniego obrotu z tym, co się naprawdę stało;
   2. dopasowuję Koopmana i Langevina do historii bez ostatnich 500 kroków i sprawdzam, który lepiej przewidział te
      500 kroków;
   3. zwycięzca prognozuje siłę tożsamości 180 sond na 500 kroków naprzód.
6. **Ramię `tozsamosc`:**
   1. z 12 oryginalnych wycinków biorę 6 o największej przewidzianej zmianie siły;
   2. szukam wejścia FastGAN-a, dla którego generator odtwarza wycinek (inwersja);
   3. przez 300 kroków zmieniam to wejście, aż siła tożsamości obrazu = siła startu + przewidziana zmiana; reszta
      latentu ma się zmienić jak najmniej;
   4. obraz trafia do DINOv2, które wybiera 90 najbardziej podobnych prawdziwych klatek z puli (bez klatek bliżej niż
      30 klatek filmu od bazy i od już dodanych);
   5. 134 klatki biorę na przemian z 6 list.
7. **Uczenie.** 500 kroków na powiększonym zbiorze; mAP@0,5 co 100 kroków na 3000 klatkach walidacji.
8. **Powrót do punktu 5.** Razem 7 obrotów, 938 klatek dodanych, koniec na kroku 4500.

Ramiona porównawcze z tego samego stanu po rozgrzewce, z tą samą liczbą klatek:
- `current` — jak `tozsamosc`, ale do wyszukiwania idzie obraz startowy, bez zmiany siły (kontrola prognozy);
- `losowe` — losowe klatki z puli, ta sama reguła ±30;
- `kolejne` — klatki puli tuż po klatkach bazy w tym samym filmie (zwykle 5 klatek dalej);
- `bez` — bez dodawania.

Wynik główny: pole pod krzywą mAP@0,5 po rozgrzewce; kryteria w `PLAN.md` i `analiza.py`.
