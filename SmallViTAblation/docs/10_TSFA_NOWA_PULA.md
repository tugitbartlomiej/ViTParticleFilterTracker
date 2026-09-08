# 10 — TSFA i surowe wideo: zbudować pulę zamiast ją augmentować

Sprawdzone 2026-09-08. To jest najmocniejszy pomysł w całym planie i zmienia
wymowę artykułu, a nie tylko liczby.

---

## 1. Co naprawdę leży na dysku

`E:\Cataract\videos\micro` — **49 nagrań mikroskopowych**, 45 GB,
1920×1080, około 29 klatek na sekundę.

```
klatek razem w 49 nagraniach:   943 353
zaadnotowanych klatek zrodlowych: 22 837   (2,4%)
```

Czyli **wykorzystano dwa i pół procent tego, co masz**. Reszta leży nietknięta.

`E:\Cataract\videos\tray` — **50 nagrań z drugiej kamery**, 99 GB, ta sama
pięćdziesiątka pacjentów, widok tacy z narzędziami. Osobny temat, sekcja 6.

Brakuje pliku `test15.mp4` w `micro` (jest 49 z 50); w `tray` jest komplet.
Prawdopodobnie siedzi w jednym z archiwów `.tar` obok — do sprawdzenia.

## 2. Jak wygląda obecna adnotacja — to zmienia strategię

Wyciągnąłem numery klatek z nazw plików (`test01_frame_0000746.jpg`):

| Wielkość | Wartość |
|---|---|
| pacjentów | 50 |
| zaadnotowanych klatek źródłowych | 22 837 |
| **ciągłych odcinków** | **11 020** |
| średnia długość odcinka | **2,1 klatki** |
| pokrycie wewnątrz zaadnotowanego zakresu | 2,7% |
| klatek na pacjenta | od **6** do **2 016**, mediana 362 |

Adnotacja nie jest równomierna ani nie tworzy długich odcinków. To jest
**jedenaście tysięcy dwuklatkowych punkcików** rozsianych po nagraniach,
średnio co czterdzieści klatek.

Dwa wnioski:

1. **Te punkciki to gotowe ziarna do propagacji.** Odległość między nimi to
   około 41 klatek, więc wystarczy rozciągnąć każdy o ±20 klatek, żeby pokryć
   praktycznie całe nagranie — bez nakładania się.
2. **Pacjenci są skrajnie nierówni.** Jeden ma 2 016 klatek, inny 6. To ponad
   trzysta razy różnicy. Model uczy się głównie kilku operacji.

## 3. TSFA robi dokładnie to, czego tu trzeba

`F:\Samsung\Code\Gaussy\TSFA` — *Tracking, Segmentation & Focus Analysis*.
Trzy rzeczy, które składają się w jedną maszynę do etykietowania:

| Składnik TSFA | Do czego tu służy |
|---|---|
| śledzenie (VitTrack, DaSiamRPN, KCF, CSRT, MOSSE, MIL) | przenosi ramkę z klatki zaadnotowanej na sąsiednie |
| korekta ramki przez SAM (EdgeSAM, MobileSAM, FastSAM, MediaPipe) | poprawia ramkę do rzeczywistych granic narzędzia, zamiast pozwolić jej dryfować |
| analiza ostrości | odrzuca klatki rozmyte — **ta sama wielkość, którą pipeline liczy cechą Fouriera** |
| przetwarzanie wsadowe | 11 tysięcy ziaren wymaga automatu, nie klikania |

Te trzy elementy razem dają **etykietę plus jej wiarygodność**, a nie samą
etykietę. To jest różnica między pseudo-etykietami do wyrzucenia a pseudo-etykietami,
które da się obronić w artykule.

## 4. Proponowany bieg

```
22 837 ziaren  ->  propagacja +-20 klatek  ->  bramka jakosci  ->  wielka pula
                                                                      |
                                                          pipeline kuracji
                                                                      |
                                                                 20 000 klatek
```

**Krok 1 — ziarna.** Każda zaadnotowana klatka to punkt startowy: pacjent,
numer klatki, ramka. Wszystko jest w nazwie pliku i w anotacji.

**Krok 2 — propagacja.** Z każdego ziarna śledzenie w przód i w tył, po ±20
klatek. SAM poprawia ramkę na każdej klatce.

**Krok 3 — kiedy przerwać.** Propagacja kończy się, gdy: pewność śledzenia
spada, ramka ze śledzenia i maska z SAM przestają się pokrywać, wykryta zostaje
zmiana ujęcia, albo klatka jest zbyt rozmyta.

**Krok 4 — bramka jakości.** Puścić **dwa różne trackery** i zostawić tylko te
klatki, na których się zgadzają. Zgodność dwóch niezależnych metod to tania
i mocna kontrola — ten sam pomysł, co konsensus DINO, YOLO i DETR w pipeline
tła z sierpnia 2025.

**Krok 5 — zmierzyć, ile te etykiety są warte.** Odłożyć część klatek
zaadnotowanych ręcznie, dojechać do nich propagacją z odległego ziarna
i porównać. **Rozkład IoU między etykietą propagowaną a ręczną to liczba, która
idzie do artykułu.** Bez niej cała reszta jest nie do obronienia.

**Krok 6 — pula wchodzi do kuracji.** Dopiero stąd pipeline wybiera 20 tysięcy.

## 5. Dlaczego to zmienia wymowę artykułu

Dziś narracja brzmi: *22 837 klatek rozdmuchane augmentacją do 91 336, z tego
wybieramy 20 000*. Recenzent od razu zauważył, że **71,3% wybranych to kopie
augmentowane**, i zapytał, czy kuracja nie wybiera po prostu korzystnego
rozkładu przekształceń. To jest słaby punkt i trudno go obronić.

Po zmianie narracja brzmi: *z 943 tysięcy prawdziwych klatek wideo budujemy
pulę kandydatów prawie za darmo, a pipeline kuracji destyluje ją do 20 tysięcy*.

Co się przez to zmienia:

- **redukcja rośnie z 4,6× do około 47×** i dotyczy prawdziwych klatek,
  nie kopii — a to jest właśnie teza pracy o kuracji danych,
- **zarzut o 71,3% augmentacji znika**, bo augmentacja przestaje być potrzebna
  do napompowania puli,
- **redundancja czasowa staje się problemem, który metoda rozwiązuje**, zamiast
  być czymś, co się przemilcza; sąsiednie klatki są prawie identyczne i to jest
  dokładnie sytuacja, w której pokrycie i różnorodność mają sens,
- **nierówność pacjentów da się naprawić** — dziś jeden pacjent daje 2 016
  klatek, a inny 6; przy pełnym pokryciu pula z każdego pacjenta jest
  porównywalna, a to, ile ich wejdzie do zbioru, decyduje pipeline, nie
  przypadkowa historia adnotacji.

## 6. Druga kamera: tacka z narzędziami

`tray` to 50 nagrań tej samej pięćdziesiątki pacjentów, ale widok jest inny —
narzędzia leżą na tacy, nie pracują w oku. To **nie jest** to samo zadanie
i nie wolno tego wrzucić do jednej puli treningowej bez namysłu.

Trzy sensowne użycia, w kolejności wartości:

1. **Test odporności na zmianę dziedziny.** Ten sam pacjent, ta sama sala, inna
   kamera i inne tło. Model wytrenowany na widoku mikroskopowego powinien
   wykrywać narzędzia na tacy słabo — i **zmierzenie tego spadku jest wynikiem**,
   nie porażką.
2. **Rozpoznanie, jakie narzędzia w ogóle są w danej operacji.** Taca pokazuje
   komplet; to jest kontekst, którego widok mikroskopowy nie daje.
3. **Zapas na później.** Osobne zadanie, osobny artykuł.

Rekomendacja: **nie ruszać tray w tym artykule**, poza ewentualnym jednym
akapitem o odporności. Rozproszenie tematu jest większym ryzykiem niż zysk.

## 7. Twarde zasady, bez których to się rozsypie

1. **Propagacja nigdy nie przekracza granicy pacjenta.** Ziarno z pacjenta
   treningowego rozchodzi się tylko po jego nagraniu, a pacjent zostaje w swojej
   części podziału. Inaczej odtwarzamy dokładnie ten wyciek, który właśnie
   naprawiliśmy.
2. **Zbiory walidacyjny i testowy zostają w pełni ręczne.** Ocena na
   pseudo-etykietach nie ma wartości. Propagacja dotyczy **wyłącznie treningu**.
3. **Jakość etykiet musi być zmierzona i podana.** Rozkład IoU względem
   adnotacji ręcznej, na odłożonych klatkach. Jedna liczba, jeden rysunek.
4. **Etykieta propagowana jest oznaczona w danych.** Każda ramka niesie
   informację, czy pochodzi od człowieka, czy z propagacji, i jak daleko od
   ziarna. Dzięki temu można potem pokazać, jak wynik zależy od udziału etykiet
   automatycznych — to jest osobna, ładna ablacja.
5. **Pula powstaje przed liczeniem cech.** Cały plan z `02_MATRIX.md` liczy się
   na tej puli. Dokładanie danych później oznacza liczenie wszystkiego od nowa.

## 8. Koszt

| Krok | Zasób | Szacunek |
|---|---|---|
| dekodowanie klatek z 49 nagrań | dysk, CPU | 943 tys. klatek, kilka godzin i sporo miejsca |
| propagacja ±20 klatek z 22 837 ziaren | GPU, lekkie modele | rzędu doby na jednym GPU, do zmierzenia |
| bramka jakości (drugi tracker) | GPU | drugie tyle |
| pomiar jakości etykiet | ręczny przegląd próbki | pół dnia |

Miejsce na dysku to realne ograniczenie: 943 tysiące klatek w rozdzielczości
1920×1080 to setki gigabajtów. **Zapisywać od razu przeskalowane do 512
i tylko te, które przeszły bramkę** — to jest ten sam krok `prescale`, który
i tak jest w planie (`03_COMPUTE`, punkt 3.3).

## 9. Co zrobić najpierw

Jeden pacjent, jedno nagranie, pełna ścieżka od ziarna do zmierzonego IoU.
Dopiero gdy liczba jakości etykiet będzie znana, decydować o reszcie.
Jeśli propagacja daje IoU poniżej rozsądnego progu, cały pomysł upada i lepiej
się o tym dowiedzieć po jednym dniu niż po miesiącu.
