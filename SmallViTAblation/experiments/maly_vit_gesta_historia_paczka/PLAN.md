# Gęsta historia tokenów przy większej paczce: czy przewidywalność latentu rośnie z rozmiarem paczki

Data: 2026-10-03. Polecenie użytkownika: „Możesz na edenie usuwać stare. Możesz użyć równania langwina
z project żeby coś się poprawiło. Licz to! Szukaj równania które chociaż na 10 kroków będzie dobrze przewidywało”.
Plan przedstawiony w rozmowie: ten sam przepis co bieg 1818732, paczka 320 zamiast 32, zapis tylko wybranych
tokenów. Dodane ramię 128 (różnica względem planu z rozmowy, nazwana tu wprost: daje trzy punkty 32, 128, 320
zamiast dwóch, kosztem jednego dodatkowego biegu około 2 GB).

## Pytanie

Czy losowe pchnięcia latentu maleją z rozmiarem paczki tak, że prognoza z samej historii latentu (równanie
Langevina z jądrem pamięci i ściąganiem do średniej) przekracza 85 procent na 10 krokach, a na 100 krokach
zbliża się do tej wartości? Przewidywanie z teorii dyfuzji SGD: wariancja pchnięcia maleje jak 1/B, dryf zostaje,
więc stosunek dryfu do szumu rośnie liniowo z B. Przy paczce 32 prognoza na 100 kroków daje 0,22–0,35
(`../maly_vit_sufit_przewidywalnosci_100/WYNIKI_CO_JESZCZE.md`).

## Co obala

Wynik prognozy na 10 kroków przy paczce 320 poniżej 0,5 obala tezę, że większa paczka wystarczy.
Wynik rosnący z B, ale wolniej niż liniowo, pokazuje, że część pchnięć nie pochodzi z losowania paczki
(np. z BatchNorm, z Adam-owej normalizacji kroku).

## Bieg

- Ziarno 47, ta sama baza 1500 klatek, AdamW 5e-4, rozgrzewka lr 300 kroków, bf16, odbicie i jasność, 1000 kroków.
  Zmienia się TYLKO `trening.PACZKA` (32 → 128 albo 320). Lr bez skalowania, celowo: AdamW normalizuje krok,
  a chodzi o izolację wpływu szumu paczki.
- Uwaga o epokach: paczka 320 przez 1000 kroków to 213 przejść przez 1500 klatek (paczka 32: 21). Późna faza
  biegu 320 jest więc dalej w uczeniu niż bieg 32 w tym samym kroku. Porównanie w tym samym kroku dotyczy szumu
  paczki, nie tego samego etapu uczenia; porównanie „w tej samej epoce” to krok 100 biegu 320 wobec kroku 1000
  biegu 32.
- Zapis co krok: 60 tokenów na obraz (wszystkie tokeny narzędzia, 7–31, plus losowe tło do 60; lista w
  `dane/TOKENY_WYBRANE.json`), 12 obrazów, 768 współrzędnych, float32: (1001, 12, 60, 768) = 2,2 GB na ramię.
  Wagi co 100 kroków, pełny stan z AdamW co 500.
- Kwota Edenu: 167/200 GB przed biegiem; usunięto zweryfikowaną kopię `LATENT_GESTY.npy` biegu 1818732 (22 GB,
  lokalna kopia z SHA256 zgodnym). Dwa ramiona zajmą około 5 GB.

## Pomiar po biegu

Te same skrypty co dla paczki 32: sufit wyroczni, równanie OU, jądro pamięci J = 5..40 na h = 1..10 i 100,
na tych samych 60 tokenach na obraz (dla paczki 32 pomiar na tym samym podzbiorze tokenów z istniejącej
gęstej historii). Podłoga: wyrocznia na samych drganiach. Kryterium: wynik na 10 kroków w funkcji B.
