# Podsumowanie sesji: rozgałęzienie rozmowy przy liczeniu kierunków tożsamości narzędzia

Data: 2026-09-29 (rozgałęzienie o 14:30:35) - 2026-09-30 (12:43).
Repozytorium: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker` (gałąź `codex/eden-nano-training-20260914`).
Rozmowa prowadzona z katalogu projektu grantowego `F:\Studia\MCHTR\Grant\Projekt` (gałąź
`feature/koopman-tft-experiments`), ale opisywana praca dotyczy repozytorium ViT.

## Cel sesji

Ta rozmowa jest rozgałęzieniem (fork) sesji głównej, która dalej pracowała równolegle. Historia do chwili
rozgałęzienia jest wspólna i opisana w sesji głównej. Tutaj jedyne pytanie użytkownika brzmiało: co było liczone
w tym rozgałęzieniu. Sesja niczego trwałego nie zmieniła poza tym zapisem.

## Kontekst

- Sesja główna: `.sessions\Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek`
  (pętla Koopman/Langevin z FastGAN-em na małym ViT, tożsamość narzędzia na parach z narzędziem i bez).
- W chwili rozgałęzienia w tle działał `kierunki_tozsamosci.py` (zadanie powłoki `b37opvd2w`), odpowiedź na pytanie
  użytkownika: ile wektorów wielowymiarowej przestrzeni latentu koduje tożsamość narzędzia w jego modelu cząstkowym
  (każdy token to cząstka o 768 współrzędnych).
- Po wznowieniu rozgałęzienia harness oznaczył to zadanie jako zatrzymane, bez zapisu zakończenia.

## Wykonane kroki

1. Odczytano wyjście zadania `b37opvd2w`: policzone 5 z 10 sieci (ziarna 47-51), potem proces zniknął (limit
   Bash w tle albo koniec poprzedniego procesu Claude Code).
2. Sprawdzono katalog `SmallViTAblation\experiments\maly_vit_tozsamosc_tla\wyniki`: sesja główna uruchomiła
   skrypt ponownie jako osobny proces (`kierunki_tozsamosci.log`, 14:45) i policzyła wszystkie 10 sieci, a potem
   zapisała kierunki cząstkowe (`zapisz_kierunki_czastek.log`, 14:51).
3. Potwierdzono w gicie, że wynik jest zacommitowany: `238851f7` (skrypt, `PLAN.md`,
   `wyniki\KIERUNKI_TOZSAMOSCI.json`) i `8d50ae84` (`zapisz_kierunki_czastek.py`, `wyniki\KIERUNKI_CZASTEK.json`).
4. Odpowiedziano użytkownikowi w trybie prostym. Wynik średnio z 10 sieci (1568 cząstek narzędzia na sieć):
   - kolejne kierunki rozdzielające (LDA, po każdym kierunku usuwanym z przestrzeni, INLP): 1,3 kierunku
     z AUC >= 0,9 (zakres 1-2) i 6,5 z AUC >= 0,75 na wycinkach wyłączonych z wyznaczania;
   - energia różnic „cząstka z narzędziem minus ta sama bez narzędzia”: połowa w 2,5 kierunku, 90 procent
     w 38,4 kierunku, pierwszy kierunek niesie 38 procent;
   - pierwszy kierunek cząstkowy ma cosinus 0,40 z wektorem `nowy`; na cząstkach, na których go wyznaczono,
     rozdziela z AUC 0,999, a na wycinkach spoza wyznaczania około 0,95 (wektor `nowy` 0,961).

## Pliki zmienione / utworzone

### W projekcie (Grant/Projekt)

Brak.

### Zewnętrzne (poza projektem)

- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-09-30_124311_Fork_Kierunki_Tozsamosci_Sprawdzenie_Stanu\README.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-09-30_124311_Fork_Kierunki_Tozsamosci_Sprawdzenie_Stanu\SESSION_SUMMARY.md`

Tylko odczytane: `SmallViTAblation\experiments\maly_vit_tozsamosc_tla\kierunki_tozsamosci.py`, `PLAN.md`, `tla.py`,
`wyniki\kierunki_tozsamosci.log`, `wyniki\zapisz_kierunki_czastek.log`, `wyniki\KIERUNKI_CZASTEK.json` oraz
wyjście zadania `b37opvd2w` w katalogu tymczasowym Claude Code. Foldery równoległych sesji, `external\`
i `F:\Studia\Articles\Zbiory\JEPA` nieruszane.

## Otwarte problemy

- Częściowe wyjście rozgałęzienia (5 sieci) pokrywa się liczbowo z pełnym biegiem sesji głównej dla ziaren 47-51,
  więc nie ma rozbieżności do wyjaśnienia.
- Otwarte problemy merytoryczne (zysk z wyszukiwania, nie z prognozy; tożsamość jako wiązka kierunków) są opisane
  w sesji głównej.

## Następne kroki

- Rozgałęzienie nie ma własnych zadań. Dalsza praca idzie według sesji głównej: ewentualna kampania pętli z nowym
  wektorem albo kilkoma kierunkami cząstkowymi wymaga osobnego planu i zgody użytkownika.
