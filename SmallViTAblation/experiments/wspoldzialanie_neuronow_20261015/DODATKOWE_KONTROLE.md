# Kontrole eksploracyjne po wynikach

Po ukończeniu pomiarów par i grup dodano dwie kontrole. Nie zmieniają uprzedniego doboru par ani wyników.

1. `kontrola_startu.py`: ten sam algorytm doboru par i ablacja na checkpointach 0, 1000, 33000 i 65640. Pytanie: czy samo istnienie silnie kompensujących par jest skutkiem treningu? Checkpoint 0 jest kontrolą startu tego biegu, a nie populacją losowych inicjalizacji. Pary dobierane na nowo w każdym stanie; nie są tymi samymi neuronami.
2. `kontrola_przelaczenia.py`: niezależny pełny forward dla największego przesunięcia ramki wśród późnych globalnych ablacji grup. Ten przykład jest jawnie wybrany po obejrzeniu wyników. Sprawdzamy ablacje każdej połowy i całości, oficjalny dekoder i ranking logitów z głowy float64. Pozwala to rozdzielić nieliniowość głowy od dyskretnej zmiany tokenu wygrywającego wybór ramki.
