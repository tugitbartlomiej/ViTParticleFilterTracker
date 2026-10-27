# Artykuł pod TMLR: nienormalne wzmocnienie przejściowe w MLP transformera

**Rozpoczęty:** 2026-10-27 04:38 (Europe/Warsaw), na polecenie użytkownika z 03:37. **Stan:** pierwszy pełny szkic tekstu (wszystkie sekcje),
bez wykresów i bez tabeli predykcji w dodatku. Nie kompilowany: brak silnika LaTeX lokalnie i na Edenie.

Roboczy tytuł: *Gradient training turns transformer MLP sublayers into non-normal transient amplifiers*.

## Pliki

- `main.tex` — tekst (angielski, wymóg TMLR). Gdy obok leży `tmlr.sty` z oficjalnego szablonu TMLR, używa go (wersja anonimowa
  do zgłoszenia); bez niego kompiluje się w zwykłym układzie `article` z `natbib`.
- `references.bib` — 42 pozycje; metadane 36 z przestrzeni Undermind (get_paper_info, DOI / arXiv), 6 klasycznych oznaczonych `% klasyka`.

## Kompilacja

Overleaf: nowy projekt z szablonu TMLR, wgrać `main.tex` i `references.bib` (zastąpić plik główny szablonu). Lokalnie: zainstalować
MiKTeX lub TeX Live, potem `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Źródła liczb (każda liczba w tekście pochodzi z tych plików)

| sekcja | eksperyment (`SmallViTAblation/experiments/`) | plik |
|---|---|---|
| 3 (dane detekcji, AP 24 seedów) | `mosty_fizyki_24seedy_ablacje_20261026__13-04-31` | `Wyniki/Raport/RAPORT.md` § 1–2 |
| 4.1 (Tabela 1, przebieg w treningu, wd, losowe etykiety, ‖J‖) | jw. + `kompensacja_lodyga_i_cifar_20261026__19-01-59` | RAPORT § 2, 3, 7; OPIS (Stan, C1, C1z, C2z, C3) |
| 4.2 (Tabela 2, blok 1, separacja) | jw. | RAPORT § 4, 6, 7; OPIS kompensacji (K1–K5, „blok 1 krytyczny”) |
| 4.3 (Kreiss, G₁, G₆, F2, F3, odporność, W5) | `wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11` | `Opis/OPIS_EKSPERYMENTU.md` (Stan), `Wyniki/Dane/OCENA_PREDYKCJI.txt` |
| 4.4 (Tabela 3, wagi, bramkowanie, reflektory) | `balanced_amplification_schur_odczyt_20261027__00-31-41` | `Wyniki/Dane/OCENA_PREDYKCJI.json` (BA1, BA2b, BA4a/b, BA5a/b, BA6), OPIS |
| 4.5 (odczyt v₁, podprzestrzeń głowy, między blokami) | jw. (wersja 2) | OCENA (BA3a/b, BA3c, BA3c′, BA3d), OPIS v2 |
| 4.5 (odbiorcy u₁) | `odbiorca_zapisu_mlp_uwaga_20261027__01-55-25` | `Wyniki/Dane/OCENA_PREDYKCJI.{txt,json}` (O1–O8, opis: g_m, g_glowa, gmax) |
| 4.5 (kanał normy) | `znak_kanalu_normy_mlp_20261027__02-47-19` | `Wyniki/Dane/OCENA_PREDYKCJI.{txt,json}` (R1–K3, opis) |
| 2, 5 (literatura, nowość) | balanced_amplification, wzmocnienie_przejsciowe | `Wyniki/Dane/{audyt_nowosci_undermind.md, literatura_rodzina_efektow_undermind.md}`, `literatura_undermind.md` |
| 3, dodatek (liczba predykcji) | wszystkie sześć | `Wyniki/Dane/OCENA_PREDYKCJI.txt`: 17 + 19 + 11 + 13 + 9 + 10 = 79; PASS 39, FAIL 40 |

## Sprawdzone przy pisaniu (rozbieżności z wcześniejszymi notatkami)

- Krótki wniosek `mosty_fizyki…/Wyniki/Krótki_Wniosek/WNIOSEK.md` mówi „8 z 18 PASS, 10 FAIL”; plik oceny ma 17 predykcji, 8 PASS, 9 FAIL.
  W artykule użyto pliku oceny. Wniosek nie był zmieniany (jest zacommitowany; do decyzji użytkownika).
- Opis balanced v2 podawał zakresy BA3c / BA3c′ odczytane z wydruku per blok; poprawione według pliku oceny (zapis w Stanie opisu).
- Wzajemność reflektorów po treningu: 0,10–0,16 (plik oceny), nie 0,10–0,24.
- Praca o geometrii wzmocnienia z szeregów czasowych: Saiprasad, Troude, Sornette 2026 (nie „Saito”).
- Rok i edycja ICLR pracy Aubry i in. do sprawdzenia (w bib zapisano 2024 za przestrzenią Undermind).

## Do zrobienia

- [ ] Wykresy (decyzja użytkownika): nienormalność w treningu (24 seedy), koncentracja Schura, wzmocnienie nieliniowe vs σ_max,
      schemat odczytu i zapisu.
- [ ] Dodatek A: pełna tabela 79 predykcji z progami i wynikami, generowana z plików OCENA_PREDYKCJI.
- [ ] Szczegóły treningu (optymalizator, harmonogram, augmentacje) z `Kod_eksperymentu/trening_ablacje.py` i skryptów CIFAR;
      dokładności CIFAR wszystkich biegów.
- [ ] Kompilacja z szablonem TMLR (Overleaf albo lokalny TeX), sprawdzenie długości i odnośników.
- [ ] Najważniejsza luka merytoryczna: pomiar na dużych, wstępnie trenowanych ViT (timm), jeśli celem ma być mocniejszy odbiór.
