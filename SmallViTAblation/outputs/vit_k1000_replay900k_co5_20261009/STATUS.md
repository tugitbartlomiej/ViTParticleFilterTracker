# Status

- Stan końcowy: `complete`, bez blokad producenta, merge, eksportu lub lokalnego API.
- Smoke 1825156: `COMPLETED 0:0`; trzy obrazy, zerowe błędy stanów i telemetrii.
- Produkcja 1825165: osiem shardów `COMPLETED 0:0`; 1500 z 1500 obrazów, zero awarii przetwarzania.
- Merge i eksport 1825166: `COMPLETED 0:0`; 101 rzeczywistych stanów, 900000 cząsteczek na stan, `interpolation=false`.
- Zgodność historyczna: wszystkie 1500 obrazów dokładnie zgodne w krokach 0, 100 i 500; maksymalny błąd stanów i telemetrii wynosi 0.
- Zamrożenie: hash wag i buforów przed oraz po replay identyczny.
- Wynik oracle: 1283 sukcesy; 217 przypadków uczciwie oznaczonych `budget_exhausted` po 500 aktualizacjach.
- Finalizer 54824 przerwał po błędach połączenia, zachowując 85 poprawnych plików. Po niezależnej kontroli ich hashy został wznowiony raz jako PID 53924 i zakończył transfer 101 z 101 plików.
- Lokalny manifest ma ten sam SHA-256 co zdalny: `dcf79fdf5cfa648088fc07b1023dc9e239f580d4788daace200757e6144a35fd`.
- API `http://127.0.0.1:8774/` przeszło kontrolę osi czasu oraz kroków 0, 5, 250 i 500. Nie wykonywano testu renderowania WebGL/GPU.
- Raport końcowy: `REPLAY_FINAL_REPORT.json`.
- Kontrola lokalnego eksportu: `DENSE_LOCAL_VERIFICATION.json`.
- Stare dane i wyniki pozostały nienaruszone; nie zgłoszono duplikatu producenta, merge ani transferu.
