# Literatura (Undermind, 2026-10-26 22:30): nienormalność pola MLP, wpływ na uczenie i odpowiedniki fizyczne

Dwa wyszukiwania w przestrzeni „ViT: detekcja końcówki narzędzia w chirurgii zaćmy” (40294fa6-…):
`/Non normal Jacobians and transient amplification in neural network training and learning` (97 prac) i
`/Physical chemical and biological systems where non normal operators cause transient amplification without eigenvalue instability` (190 prac).
Podsumowania Undermind oparte na streszczeniach, nie pełnych tekstach. Klucze cytowań = klucze przestrzeni roboczej.

## 1. Nienormalność a uczenie (sieci)

- Nienormalność jako zasób, nie defekt: w rekurencyjnych sieciach sekwencyjna (łańcuchowa) nienormalna dynamika poprawia pamięć
  i wyniki względem sieci ortogonalnych [Orh19]; nnRNN z parametryzacją Schura utrzymuje stabilne widmo przy większej ekspresji niż
  ograniczenie ortogonalne [Ker19]. Trening tworzy strukturę łańcuchową i zwiększa nienormalność wag nawet z inicjalizacji ortogonalnej
  [Orh19] — to jest najbliższy odpowiednik naszego pomiaru (0,76 → 0,92), ale dla RNN, nie dla MLP transformera.
- Ślady pamięci w układach dynamicznych: nienormalne (feedforward w ukryciu) macierze połączeń dają najdłuższą pamięć [Gan08b];
  kodowanie trajektoriami przejściowymi [Bon18]; reżimy i mechanizmy wzmocnienia przejściowego w sieciach abstrakcyjnych i biologicznych [Chr22].
- Odporność: pseudospektra i wzrost wartości osobliwych opisują wrażliwość, której stabilne widmo nie zdradza [Sen18b, Tre05];
  „non-normal spectral signatures of instability” w dynamice treningu [Gho26b]; kontrola wzmocnienia przejściowego poprawia długie
  rozwinięcia modeli dynamiki [Per26d].
- Inicjalizacja ortogonalna i izometria dynamiczna przyspieszają propagację gradientu [Pen17b], ale wymuszanie ortogonalności w całym
  treningu spowalnia zbieżność lub obniża wynik [Vor17]. Wniosek literatury: trening odchodzi od normalności, bo normalność ogranicza ekspresję.
- Transformer: geometria widmowa strumienia resztkowego [Fer26d] (porównanie dla nas: ich gradient widmowy po głębokości jest monotoniczny,
  u nas nie); sprzężenie bloków a generalizacja [Aub24]; „transient reserves” i porażka rozumowania wartościami własnymi w propagatorze uwagi [Li26ad].
- Brak w literaturze: pomiaru odchylenia od normalności (Henrici) albo stałej Kreissa jakobianów MLP wytrenowanych transformerów w funkcji
  checkpointu i zadania, z kontrolami (bez wd, losowe etykiety), i nieliniowej weryfikacji wzmocnienia na sieci. To jest luka, którą zamyka
  ten eksperyment i rodzice.

## 2. Odpowiedniki fizyczne, chemiczne, biologiczne

Wspólny mechanizm: stabilne widmo nie ogranicza odpowiedzi w skończonym czasie; nieortogonalne mody przekierowują zaburzenie w kierunki
silnie wzmacniane; o zachowaniu decydują wartości osobliwe propagatora, abscysa numeryczna (reaktywność), pseudospektrum, stała Kreissa
[Tre05, Sch07f, Mcc13]. Przegląd międzydziedzinowy: [Mcc13] (płyny → chemia i biologia), [Asl18] (sieci nienormalne).

| dziedzina | źródło nienormalności | skutek wzmocnienia | diagnostyka | prace |
|---|---|---|---|---|
| przepływy ścinające (Couette, Poiseuille) | ścinanie średniego przepływu: mechanizm Orra, lift-up | wzrost 10²–10³, przejście podkrytyczne do turbulencji | SVD propagatora, optymalne zaburzenia, pseudospektra | [Tre93b, Sch07f] |
| plazma, MHD, dynamo | ścinanie magnetyczne, sprzężenie kinetyczne, warstwy prądowe | turbulencja podkrytyczna, wzmocnienie energii magnetycznej | wzrost przejściowy | [Lan15, Fed03] |
| termoakustyka, flatter | sprzężenie modów akustycznych i płomienia / struktury i przepływu | wyzwolenie oscylacji nieliniowych przed progiem | eksperyment: inicjalizacja modu i śledzenie energii | [Rav16, Mar11c, Sch03e] |
| ekologia, sieci pokarmowe | asymetryczne (skierowane) oddziaływania | reaktywność: odejście od równowagi przed powrotem; wymarcia | reaktywność Neuberta–Caswella z szeregów czasowych | [Neu97, Asl17, Dua22] |
| sieci skierowane, epidemie, socjoekonomia | skierowane sprzężenia, samoorganizacja ku nienormalności | niestabilności strukturalne, efektywna komunikacja | miary nienormalności sieci | [Asl18, Dua22, Wan25cn, Bag19] |
| kora mózgowa | pobudzenie stabilizowane hamowaniem: ukryte ścieżki feedforward | „balanced amplification”: selektywne, szybkie wzmocnienie wzorów | odpowiedź na zaburzenie, kontrola optymalna ruchu | [Mur09, Hen12b, Hen14] |
| sieci biochemiczne i sygnałowe | asymetryczne kaskady reakcji | wzmocnienie szumu mimo stabilnej równowagi | wariancja stochastyczna | [Mcc13, Dix16c] |
| fotonika nie-hermitowska | zysk–strata, sprzężenie niewzajemne | anomalne wzmocnienie przejściowe, skin effect | eksperyment w pasywnej sieci akustycznej | [Mak14, Mid22, Gu22e] |

Przypadki, gdzie nienormalność jest **tworzona przez optymalizację lub napęd**, nie dana z góry: trening RNN [Orh19], kontrola optymalna
hamowania w korze [Hen14], samoorganizacja sieci wpływu [Wan25cn], projekt sieci skierowanych [Dua22]. Nasz pomiar (gradient czyni MLP
nienormalnym niezależnie od zadania, kształt z treningu, skala z sygnału) należy do tej klasy.

Zastrzeżenie z literatury: wzmocnienie przejściowe zależy od normy i rodzaju wymuszenia i samo w sobie nie dowodzi przejścia nieliniowego
[Mag13, Blu17] — zgodne z naszym F4: wyjście sieci pozostaje odporne.

## 3. Co z tego dla artykułu

- Most do fizyki jest mocniejszy niż „kropla w ścinaniu”: właściwy odpowiednik to **nienormalny operator z wzrostem przejściowym** (teoria
  niemodalna, Trefethen), a konkretny analog mechanizmu to **balanced amplification** w korze [Mur09]: uczony MLP, jak kora, wzmacnia
  wybrane kierunki (ukryte pary czytaj→pisz) bez niestabilności widmowej.
- Cytować: [Tre05, Tre93b, Sch07f, Mcc13] (teoria i przegląd), [Mur09, Hen12b, Gan08b, Orh19, Ker19] (sieci neuronowe), [Asl18, Bag19]
  (sieci skierowane), [Fer26d, Aub24, Gho26b, Per26d] (transformery i trening, 2024–2026).
- Luka, którą wypełniamy: pomiar nienormalności i stałej Kreissa jakobianów MLP w funkcji treningu, zadania i architektury, z kontrolami
  i nieliniową weryfikacją; w literaturze są RNN, kora i sieci skierowane, nie ma MLP transformera.
