# Fork sesji: sprawdzenie, co policzono w rozgałęzieniu (kierunki tożsamości w przestrzeni cząstek)

Data: 2026-09-29 (rozgałęzienie o 14:30) - 2026-09-30 (12:43). Rozgałęzienie sesji głównej zapisanej jako
`Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek`.

## Cel sesji

Odpowiedzieć użytkownikowi, co było liczone w tym rozgałęzieniu rozmowy i czy trzeba coś dokończyć. Sesja czysto
sprawdzająca (Q&A): nie zmieniła kodu, wyników ani dokumentów eksperymentu, zapisuje tylko ten opis.

## Typ

maintenance

## Workerzy użyci

Brak - sesja solo.

## Co zrobiono (skrót)

- Ustalono, że rozgałęzienie liczyło `kierunki_tozsamosci.py`: ile kierunków 768-wymiarowej przestrzeni cząstek
  odróżnia token z narzędziem od tego samego tokenu na zdjęciu bez narzędzia (zamalowane oko, czarne tło).
- Bieg w tle rozgałęzienia urwał się po 5 z 10 sieci (ziarna 47-51); sesja główna policzyła wszystkie 10 osobnym
  procesem i zapisała wynik w commitach `238851f7` i `8d50ae84`. Niczego nie trzeba powtarzać.
- Wynik (10 sieci): 1-2 kierunki rozdzielają cząstki z AUC >= 0,9 na wycinkach spoza wyznaczania, 5-8 z AUC >= 0,75.

Pełny opis: [SESSION_SUMMARY.md](SESSION_SUMMARY.md).
