# Odpowiedź latentu sond na pchnięcie jednej paczki (bez treningu, z nagrań)

Plan: [PLAN.md](PLAN.md). Poprawka po wyniku: [AMENDMENT_PROJEKCJA_LOKALNA.md](AMENDMENT_PROJEKCJA_LOKALNA.md).
Wynik: [WYNIKI.md](WYNIKI.md). Deklaracja: [WYNIKI.json](WYNIKI.json).

Kod jest nowy (nie kopiuje innego eksperymentu). Czyta nagrania `../maly_vit_rownanie_latentu/nagranie_s47_sily` i
`nagranie_s48_sily`; niczego w nich nie zmienia. Nie uczy sieci i nie używa GPU.

## Uruchomienie (lokalnie, Python 3.11)

```bash
cd kod
py -3.11 odpowiedz.py --wyniki ../wyniki/proba --proba     # próba kodu na 100 startach, to nie jest wynik
py -3.11 odpowiedz.py --wyniki ../wyniki                   # modele A, B, C; około 14 minut, 1,4 GB RAM
py -3.11 analiza_gramow.py --wyniki ../wyniki              # korelacje i projekcja lokalna z GRAMY.npz, sekundy
py -3.11 odpowiedz_dluga.py --wyniki ../wyniki/dluga       # projekcja lokalna do 500 kroków; około 8 minut
```

Czas zależy głównie od dysku talerzowego: pierwszy odczyt nagrania ziarna 47 trwa około 9 minut, kolejne około
minuty. Długie biegi uruchamiać jako proces odłączony z logiem (`wyniki/pelny.log`).

## Pliki wyników

- `wyniki/ODPOWIEDZ.json`: umiejętność modeli A, B, C i współczynniki modelu A (plan).
- `wyniki/GRAMY.npz`: macierze Grama cech i celów w blokach po 50 startów (dopasowanie, test ziarna 47, ziarno 48).
  Każdą dalszą regresję na tych cechach da się policzyć z nich bez czytania nagrań.
- `wyniki/ANALIZA_GRAMOW.json`: korelacje pchnięć ze stanem i projekcja lokalna na 7 horyzontach.
- `wyniki/dluga/ODPOWIEDZ_DLUGA.json`, `wyniki/dluga/GRAMY_DLUGA.npz`: projekcja lokalna na 22 horyzontach do 500 kroków.
- `wyniki/proba/`: próba kodu.
