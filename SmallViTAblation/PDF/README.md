# Literatura do SmallViTAblation: pułapki optyczne, pole sił w ognisku wiązek, koloidy (porównanie z polem wiązek ViT)

Zbior tematyczny obslugiwany przez wspolne narzedzia PdfDownloader.
Kod lezy w `F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader` - tutaj sa tylko dane i cienkie pliki `.bat`.

## Obieg pracy

1. **Zbierz linki** do listy `.txt` w `linki\`:

   ```
   zbierz_linki.bat --nowa "Augmentacja obrazow" --po-co "Po co powstal ten plik."
   zbierz_linki.bat --do augmentacja-obrazow-RRRR-MM-DD --sekcja "MIXUP I CUTMIX" ^
       --link "https://doi.org/10.48550/arXiv.1710.09412 # Zhang i in. 2018, mixup"
   ```

   Skrypt normalizuje adresy (gole DOI i `arXiv:` tez przyjmuje) i odrzuca
   powtorki wzgledem wszystkich list w `linki\`.

2. **Pobierz**:

   ```
   pobierz_pdfy.bat                       wszystkie listy
   pobierz_pdfy.bat linki\<nazwa>.txt     jedna lista
   pobierz_pdfy.bat --sucho               proba bez zapisu
   ```

   PDF-y ladują w `pulapki_optyczne\`, raport nieudanych w `nieudane\nieudane_linki.html`,
   pelny log w `pobieranie.log`.

3. **Ponow nieudane** po jakims czasie: `pobierz_nieudane.bat`.

4. **Pelny obieg** razem z kolejka Primo (dostep instytucjonalny):
   `odswiez_kolejke.bat`. Miedzy jego przebiegami pobierasz recznie przez Primo
   i wrzucasz pliki do `pulapki_optyczne\` - kolejka skurczy sie sama.

## Format listy linkow

Jeden adres na linie, komentarz po spacji i `#`. Linie zaczynajace sie od `#`
sa pomijane. Sekcje jako `# --- N. TYTUL ---`.

```
# --- 1. PRZEGLADY ---
https://doi.org/10.1109/TPAMI.2022.3152247  # Xu i in. 2023, przeglad augmentacji
https://arxiv.org/abs/1710.09412            # Zhang i in. 2018, mixup
```

Prace za paywallem tez dopisuj - skrypt kieruje je do raportu nieudanych
i do sciezki Primo, zamiast je gubic.

## Ustawienia

`config\config.json`:

- `folder_pdf` - gdzie ladują pobrane pliki
- `skip_dirs` - katalogi z juz posiadanymi artykulami (nie pobiera drugi raz)
- `watki`, `delay_s`, `timeout_s`, `retries` - tempo pobierania
- `email` - wymagany przez API Unpaywall, ktore rozwiazuje DOI na kopie open access

## Wywolanie bez plikow .bat

Kazde narzedzie przyjmuje `--zbior <sciezka>`; zamiast tego mozna ustawic
zmienna srodowiskowa `ZBIOR_PDF`.

```
py -3.11 "F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\pobierz_pdfy.py" --zbior "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\PDF" linki\<nazwa>.txt
py -3.11 "F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader\nowa_lista.py"   --zbior "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\PDF" --pokaz
```

Pliki `.bat` koncza sie `pause`, wiec do uruchomienia w tle wolaj `.py`, nie `.bat`.
