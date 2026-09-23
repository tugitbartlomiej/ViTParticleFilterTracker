# Eksperyment 3. Włókno pomyłek: kierunki błędów w latencie tożsamości

Status: plan, zapisany 2026-09-23. Etap A (prawdziwa pula, bez generatora) rozwijamy teraz.
Etap B (model dyfuzyjny) odkładamy; wymaga osobnej zgody, bo to kilkudniowe zadanie na Edenie.

Przegląd wszystkich trzech eksperymentów: `docs/Analiza_ponownej_publikacji/46_PLAN_TRZECH_EKSPERYMENTOW_DOBORU_KLATEK.md`.

## Pytanie

Jakie obrazy detektor widzi tak samo jak cystotom, choć cystotomu na nich nie ma albo leży gdzie indziej?
Czy różnice między nimi a prawdziwym cystotomem dają w latencie kierunki błędów, którymi można wybrać z puli
klatki mocno poprawiające rozpoznawanie narzędzia?

## Podstawa: Rousselot, Wendebourg, Köthe 2026

„Show Me What You Don't Know: Efficient Sampling from Invariant Sets for Model Validation”, arXiv 2603.21782,
przeczytany w całości 2026-09-23.

- **Włókno** to zbiór wszystkich wejść, które sieć odwzorowuje na ten sam latent h: φ⁻¹(h) = {x : φ(x) = h}.
  Obrazy z jednego włókna są dla sieci nie do odróżnienia.
- **Strata włókna**: L = ‖φ(x) − φ(x̃)‖². Mierzy, jak blisko próbka x̃ leży włókna wzorca x.
  Autorzy porównują ją ze stratą najbliższego sąsiada w zbiorze danych; próbka ma sens, gdy jest wyraźnie niższa.
- **NDTM**: gotowy, bezwarunkowy model dyfuzyjny, bez douczania. W każdym kroku odszumiania dodają poprawkę
  x̂ₜ = xₜ + γₜ uₜ, gdzie uₜ minimalizuje L_terminal + κ ‖uₜ‖² + τ L_score w kilku krokach spadku gradientu.
  Stan końcowy szacują wzorem Tweediego, E[x₀|xₜ] = xₜ + σₜ² ∇ log p(xₜ).
- **Ich poprawka stabilności**: wzorca też nie biorą wprost, tylko zaszumiają go do tego samego poziomu t
  i przybliżają Tweediem, L = ‖φ(E[x₀|xₜ]) − φ(E[x₀|x′ₜ])‖². Bez tego obrazy są przesycone i przeteksturowane.
- Ustawienia: κ = 10⁻⁴, τ = 0, 4–8 kroków na poprawkę, krok uczenia od 2·10⁻³ liniowo do 0,
  najsilniejsze prowadzenie dla t w przedziale około [700, 400].
- Koszt: około 1,8 min na próbkę (A40, 100 kroków). Warunkowy model włókna to około 14 000 min uczenia i opłaca się
  dopiero od około 9 tysięcy próbek.
- Dla zdjęć rentgenowskich wyuczyli własny bezwarunkowy model dyfuzyjny: 18 mln parametrów, 384 × 384, 120 tys. kroków.
- Dodatek A: dopracowanie próbki spadkiem gradientu w latencie autokodera, nie w pikselach, bo w pikselach wychodzą
  zmiany podobne do ataków adwersarialnych.
- Zastosowania w artykule są wyłącznie diagnostyczne (ślepe plamy, porównanie klasyfikatorów). Wybór danych do
  uczenia na tej podstawie jest naszym rozszerzeniem.

## Etap A: włókno w prawdziwej puli (bez generatora, lokalnie)

Zamiast generować obrazy z włókna, szukamy ich w prawdziwej puli.

1. **Latent tożsamości.** Dla klatki: tokeny backbone detektora pod przewidzianą ramką i średnia ważona aktywacją
   wektora tożsamości (32 liczby profilu zredukowanego, jak w `koopman_zaskoczenie.py`), plus wersja przestrzenna
   (siatka tokenów w układzie narzędzia).
2. **Wzorce pomyłek.** Klatki walidacyjne, na których detektor pewnie wskazał złe miejsce
   (pewność ≥ 0,5, IoU < 0,5), oraz klatki bez cystotomu z fałszywym alarmem.
3. **Włókno w puli.** Dla każdego wzorca klatki puli (`CZASTKI_pool_b0500`, 9683 klatki) z najmniejszą stratą
   włókna. Próg: strata poniżej typowej straty między sąsiednimi klatkami tego samego filmu, na których detektor trafia.
4. **Kontrola z etykietami (tylko do oceny):** jaki odsetek klatek z włókna ma ramkę detektora poza cystotomem
   albo nie ma cystotomu. To mówi, czy włókno naprawdę zbiera pomyłki.
5. **Kierunki błędów.** Różnica latentów „włókno pomyłki” minus „prawdziwy cystotom” (średnia i LDA między grupami).
   Kilka kierunków, po jednym na skupisko pomyłek (np. pęseta, rękojeść, niebieskie tło).
6. **Ocena puli.** Rzut każdej klatki puli na kierunki błędów; wybór najwyżej ocenionych z pilnowaniem różnorodności
   (jedna klatka na grupę podobnych).
7. **Galeria.** Dla kilku wzorców: wzorzec, jego włókno w puli i klatki wybrane kierunkami błędów, z ramkami.

Miary etapu A:

- odsetek pomyłek w klatkach wybranych kierunkami błędów wobec losowych (z etykiet);
- czy wybrane klatki pokrywają rodzaje chybień widziane w galerii chybień;
- AUC kierunków błędów dla pewnych chybień na walidacji (porównanie z eksperymentem 1).

Trening po etapie A (po zgodzie): 250 klatek z kierunków błędów wobec 250 losowych, trzy ziarna, ta sama pula,
jak w eksperymencie 1. Wariant dodatkowy: klatki z pęsetą bez cystotomu dodane jako przykłady negatywne.

## Etap B: model dyfuzyjny (odłożony)

- Bezwarunkowy model dyfuzyjny na klatkach Cataract-LMM (ponad 150 tys. klatek), rozmiar jak u Rousselota
  (około 18 mln parametrów), rozdzielczość 384. Uczenie na Edenie, kilka dni. Wymaga zgody i planu zasobów.
- NDTM ze stratą włókna na latencie tożsamości, z poprawką Tweediego dla wzorca.
- Wygenerowane obrazy służą najpierw do nazwania brakujących czynników (czego detektor nie widzi) i jako zapytania
  do wyszukania prawdziwych klatek w puli. Nie trafiają wprost do uczenia, dopóki nie sprawdzimy, że nie wprowadzają
  artefaktów.
- Zaletą wobec etapu A jest pokazanie pomyłek, których w puli jest za mało, na przykład rzadkiego tła.

## Czego ten eksperyment nie dowodzi

- Włókno w prawdziwej puli jest ograniczone do tego, co w puli jest. Rzadkich pomyłek może nie znaleźć.
- Mała strata włókna w latencie zredukowanym do 32 liczb nie oznacza, że pełny latent sieci jest ten sam.
  Sprawdzamy też na pełnych tokenach pod ramką.
- Tak jak w eksperymencie 1, dopiero trening rozstrzyga, czy wskazane klatki poprawiają detektor.
