# Kryteria interpretacji — pełne teksty przeczytane przez Undermind

Data 2026-10-10. Workspace ViT: 40294fa6-c9a3-4469-9a61-d3eddeaee893. Zapytania dotyczyły kryteriów rozróżnienia szkła od ograniczonego ruchu optymalizatora i zwykłej zbieżności, a nie potwierdzenia z góry hipotezy.

## Winter i Janssen, Glassy dynamics in deep neural networks: A structural comparison

https://arxiv.org/abs/2405.13098 (Undermind Win24b; wersja opublikowana Phys. Rev. Research 7, 023010).

Sekcje III.B, III.D, rysunki 3 i 5: brak jednoznacznej dwustopniowej relaksacji, klatki z późniejszą ucieczką i rozbieżności czasu relaksacji przy niezerowej temperaturze. Równanie 8: MSD. Sekcja III.F, rysunek 7: starzenie w części reżimów, nie we wszystkich. Dodatek H: ograniczenie ruchu w kwadratowym dołku daje nasycenie MSD. Nie wolno utożsamiać samego nasycenia z klatką wielociałową. Wagi sieci i nadtłumiona dynamika Langevina; nie aktywacje tokenów ani AdamW. Niegaussowość nie jest wystarczającym dowodem, w 768D trzeba także uwzględnić anizotropię.

## Baity-Jesi i in., Comparing dynamics: deep neural networks versus glassy systems

https://doi.org/10.1088/1742-5468/ab3281 (Undermind Bai18).

Równanie 3, rysunek 3: dwuczasowe MSD i zależność czasu relaksacji od wieku. Późno może pojawić się dyfuzja w płaskich kierunkach z brakiem starzenia. Rysunek 4: przybliżone skalowanie przez natężenie szumu SGD. Stały nominalny learning rate nie gwarantuje stałości statystyk aktualizacji. Brak starzenia w skończonym oknie nie wyklucza wszystkich analogii do szkła.

## Herrera Segura i in., Subaging in underparametrized deep neural networks

https://doi.org/10.1088/2632-2153/ac8f1b (Undermind Seg22).

Równanie 2, rysunki 2, 3 i 6: ustalony próg MSD, czas osiągnięcia t*(tw), wzrost t* ~ tw^mu. Mu opisuje wzrost CZASU do progu, a nie spadek AMPLITUDY MSD przy stałym opóźnieniu. Tych dwóch wykładników nie wolno nazywać zamiennie. Progi są umowne; autorzy sprawdzają czułość. Ograniczone okno wymaga oznaczania progów nieosiągniętych. Zmniejszenie przyrostów samo w sobie może wynikać ze zwykłej zbieżności.

## Wnioski dla audytu

- Używamy faktycznych numerów kroków i rozgrzewki 0–300.
- Analiza jest opisowa dla dwóch konkretnych biegów, nie jest testem uniwersalnej fazy materii.
- Plateau, zawracanie, starzenie i ich kontrolne wyjaśnienia należy rozpatrywać łącznie.
- Kryterium plateau alpha <=0.2 przez dekadę jest jawnym progiem operacyjnym audytu, nie twierdzeniem z tych publikacji.
- Dopasowanie krzywej przez OU nie dowodzi, że token jest procesem OU. Pokazuje brak identyfikowalności mechanizmu na podstawie samej krzywej.
- Sąsiedztwo w przestrzeni cech nie jest potwierdzonym fizycznym mechanizmem klatki.
