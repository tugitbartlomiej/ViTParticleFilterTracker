# Jawna korekta definicji porównania zgodności znaku

Data: 2026-10-02, około12:14 Europe/Warsaw. Korekta została zlecona i zapisana PO otwarciu końcowego testu. WYBOR_MODELU.json, ZAMROZENIE_MODELU.json, MODEL_PREFIX_500.npz i pierwotne METRYKI_HOLDOUT.json/WYNIKI.json pozostają zapisami oryginalnego wykonania.

Przedtestowy PROTOKOL_MODELU.md określał główną metrykę znaku na składowych o |delta_true|>1e-8, a dodatkową na |delta_true|>1e-3. Audyt nadrzędnego agenta przypomniał, że wcześniejszy cel85procent obejmował WSZYSTKIE128współrzędnych, z dokładnym sign(0)=0, bez odsiewu. Definicja pierwotna nie gwarantowała bezpośredniej porównywalności.

Dlatego zapisujemy teraz oddzielny wynik SIGN_ALL_COMPARABILITY.json: coverage-weighted mean po wszystkich128składowych tokenu i równa waga każdego obrazu, mean(sign(delta_pred)==sign(delta_true)), gdzie sign(0)=0. Ta metryka służy sprawdzeniu historycznego celu85. Jest jawnie dodana po teście jako korekta definicji porównania. Nie zastępujemy ani nie usuwamy pierwotnych wyników i pokazujemy obie definicje oraz udział składowych w wersjach progowanych.

Nie wybieramy ponownie lambda, nie dopasowujemy kolejnego operatora do prognozy i nie zmieniamy horyzontu500. MSE było kryterium wyboru modelu i pozostaje bez zmian. Prognozę odtwarzamy deterministycznie z zamrożonego A,b i origin500, w float64, oraz sprawdzamy bitową zgodność float32 z wcześniej zapisaną PROGNOZA_HOLDOUT.npz. Przyszłe stany służą wyłącznie obliczeniu metryki porównywalności. Dodatkowa macierz na0..1000, jeżeli powstanie, pozostaje wyłącznie opisowa i nie jest używana w tym teście.
