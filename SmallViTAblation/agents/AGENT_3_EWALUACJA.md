AGENT 3 — ewaluacja, metryki i tabele
=====================================

Przeczytaj najpierw `README.md` w tym katalogu. Ten plik zakłada, że go znasz.

**Twój katalog:** `pipeline/eval/`
**Nie dotykasz:** niczego poza nim.

---

## Co budujesz i po co

Liczby, które trafią do artykułu. Twoja część zamyka **cztery zarzuty
recenzentów naraz**, więc szczegóły protokołu nie są tu kosmetyką.

## Zadanie 1 — `pipeline/eval/evaluate.py`

Inferencja modelu na zadanym zbiorze plus metryki.

Wejście: checkpoint, plik anotacji COCO, katalog obrazów, `protocol.yaml`.
Wyjście: `metrics.json` w formacie z `README.md`, sekcja 4.5, oraz plik
predykcji w formacie COCO.

Wymagania twarde:

1. **Inferencja standardowa.** Zwykłe dekodowanie detektora, wszystkie
   predykcje ponad progiem. Poprzednia wersja pracy brała **jedno wybrane
   zapytanie modelu** i recenzent uznał to za niekonwencjonalne i
   nieuzasadnione. Nie odtwarzaj tego.
2. **Próg ufności do generowania predykcji: 0,001**, wartość z
   `evaluation.map_conf_threshold`. Nie 0,3. Poprzedni protokół obcinał krzywą
   progiem 0,3, co zaniża mAP i miesza dwie różne rzeczy: jakość modelu
   i kalibrację jego pewności.
3. **mAP przez `pycocotools`**, pełna krzywa precyzja–czułość. Raportuj
   `mAP@0.5`, `mAP@[0.5:0.95]`, `mAP@0.75`, `AR@100`.
4. **Zapisuj predykcje na dysk.** Każda kolejna metryka ma się liczyć z pliku,
   bez powtarzania inferencji. Przy stu siedemdziesięciu biegach i kilku
   analizach progu to różnica między godzinami a dniami.
5. **Ground truth przycinaj do obrazów faktycznie ocenionych**, inaczej metryki
   są zaniżone przez obrazy, których model nie widział.

## Zadanie 2 — `pipeline/eval/operating_point.py`

Recenzent napisał wprost, że jeden arbitralnie wybrany próg ufności to za mało.
Trzeba pokazać, jak model zachowuje się na całym zakresie.

1. **Wybór progu operacyjnego** — próg maksymalizujący F1 **na zbiorze
   walidacyjnym**. Nigdy na teście, nigdy na zbiorze zewnętrznym.
2. **Krzywa F1 w funkcji progu** — dane do rysunku, dla każdego zbioru osobno.
3. **Test przeniesienia progu** — czy próg wybrany na walidacji własnej działa
   na zbiorze zewnętrznym. To osobny rysunek i osobne zdanie w artykule.
   Recenzent zauważył, że przy zignorowaniu ufności model lokalizuje 83,2%
   narzędzi, ale 85,7% tych trafień leży **poniżej** progu 0,3. Czyli kalibracja
   pewności rozjeżdża się przy zmianie dziedziny i to trzeba zmierzyć.

Wszystko liczone z zapisanych predykcji, bez ponownej inferencji.

## Zadanie 3 — `pipeline/eval/aggregate.py`

Zbiera wyniki wszystkich biegów w tabele.

- **średnia i odchylenie standardowe po ziarnach losowych**, nigdy pojedyncza
  wartość; recenzent wytknął, że cała praca stała na jednym ziarnie,
- **surowe wartości per ziarno** w osobnym pliku, do załącznika artykułu,
- **dwie kolumny checkpointu obok siebie**: ostatnia epoka i najlepszy na
  walidacji. **Nigdy nie mieszaj ich w jednej kolumnie** i nie wybieraj per
  wariant tego, co lepiej wygląda,
- eksport do LaTeX, gotowy do wklejenia,
- przy trzech ziarnach **nie rób testów istotności udających większą próbę**.
  Podaj średnią, odchylenie i wartości surowe. Przedział ufności metodą
  bootstrap tylko tam, gdzie ziaren jest pięć albo więcej.

## Zadanie 4 — `pipeline/eval/external_once.py`

Zbiór zewnętrzny to **2 333 obrazy z innego szpitala**, w dwóch wariantach:
pełnym oraz oczyszczonym z 48 obrazów, które wyciekły do treningu.

Recenzent zarzucił, że poprzednio ten zbiór był oglądany wielokrotnie w trakcie
rozwoju metody, więc określenie „zbiór, na który nie patrzyliśmy" było
nieuprawnione.

Dlatego ten skrypt:

1. przyjmuje **listę checkpointów już wybranych na walidacji** — sam niczego nie
   wybiera,
2. liczy metryki **raz**,
3. **dopisuje wiersz do `external_access_ledger.jsonl`**: data, kto, który
   checkpoint, w jakim celu,
4. **odmawia działania**, gdy wywołanie wygląda na przeszukiwanie zbioru
   zewnętrznego w poszukiwaniu najlepszej epoki. Wykryj to po liczbie
   checkpointów na jeden bieg i przerwij z czytelnym komunikatem.

Punkt czwarty jest celowy. Rejestr ma pozwolić napisać w artykule prawdę
zamiast deklaracji.

## Test końcowy

1. `evaluate.py` na małym modelu i 200 obrazach → poprawny `metrics.json`
   i plik predykcji,
2. `operating_point.py` z tych predykcji → próg, krzywa, wynik testu
   przeniesienia,
3. `aggregate.py` na trzech sztucznych biegach → tabela ze średnią
   i odchyleniem,
4. `external_once.py` → wiersz w rejestrze; próba podania dziesięciu
   checkpointów jednego biegu → odmowa z komunikatem.

## Zestaw metryk — jeden dla wszystkich tabel

Recenzent wytknął, że w artykule różne tabele podawały różne metryki i nie dało
się ich porównać. Od teraz wszędzie ten sam komplet:

```
mAP@0.5, mAP@[0.5:0.95], mAP@0.75, AR@100,
TP, FP, FN, precision, recall, F1,
n_images, n_annotations
```

Do tego zawsze **liczba pacjentów** obok liczby obrazów. W obrazowaniu
medycznym to pierwsza liczba, na którą patrzy recenzent: zbiór testowy to
**7 pacjentów i 9 240 obrazów**, a nie „9 240 obrazów".
