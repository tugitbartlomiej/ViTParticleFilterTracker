# 11 — Skąd wziąć więcej nagrań operacji

Szukane 2026-09-08. Kryterium: **liczba nagrań**, nie adnotacji — etykiety
robimy sami propagacją (`10_TSFA_NOWA_PULA.md`).

Razem dostępne publicznie: **ponad 6 300 nagrań, około 1 500 godzin**.
Dla porównania masz dziś 50 nagrań i około 9 godzin.

---

## 1. Zestawienie

| Zbiór | Nagrań | Godzin | Rozdzielczość | fps | Ośrodek | Licencja | Dostęp |
|---|---:|---:|---|---:|---|---|---|
| **Cataract-LMM** | **3 000** | **1 134** | 720×480 (2 930 szt.)<br>1920×1080 (70 szt.) | 30 / 60 | Farabi + Noor, Teheran, Iran | CC BY-**NC-ND** 4.0 | Hugging Face `mjahmadi/Cataract-LMM` |
| **OphNet** | 2 078 | 205 | zmienna | zmienna | YouTube, wiele źródeł | praca CC BY-NC-SA | GitHub `minghu0830/OphNet-benchmark` — **linki, nie pliki** |
| **Cataract-1K** | 1 000 | ~119 | **512×324** | 25 | Klagenfurt, Austria | **CC BY 4.0** | Synapse, 89,9 GB |
| **SICS-105** | 105 | 22,6 | **1920×1080** | 30 | Sankara, Bangalore, Indie | Zenodo (sprawdzić) | DOI `10.5281/zenodo.13847781` |
| **Cataract-101** | 101 | ~14 | brak danych | — | Klagenfurt, Austria | CC BY-NC 4.0 | ITEC AAU, 8,3 GB |
| CATARACTS (twój) | 50 | ~9 | 1920×1080 | ~29 | Brest, Francja | — | masz na dysku |

## 2. Rozdzielczość jest tu ważniejsza niż liczba nagrań

Twoje zadanie to wykrywanie **końcówki narzędzia**, czyli małego obiektu.
Przy zejściu z 1920×1080 na 512×324 końcówka zajmuje **około trzynaście razy
mniej pikseli**. Model uczony na takich klatkach uczy się innego problemu.

Dlatego kolejność wcale nie jest według liczby nagrań:

| Miejsce | Zbiór | Dlaczego |
|---|---|---|
| **1** | **SICS-105** | ta sama rozdzielczość co twoje nagrania, **inna technika operacyjna** (ręczne małe cięcie zamiast fakoemulsyfikacji) i inny kraj — najwięcej zmiany dziedziny na gigabajt; mały, więc tani na start |
| **2** | **Cataract-LMM** | siedemdziesiąt nagrań ze szpitala Noor jest w 1920×1080; pozostałe 2 930 w 720×480 to i tak **ponad tysiąc godzin** i drugi ośrodek |
| **3** | **Cataract-1K** | najlepsza licencja (CC BY 4.0, bez ograniczeń niekomercyjnych) i tysiąc nagrań, ale 512×324 |
| 4 | Cataract-101 | ten sam szpital co Cataract-1K, więc dokłada mało nowego |
| 5 | OphNet | jaskra i rogówka to **inne operacje**; wartość jest w tym, że model widzi zupełnie inne sceny, ale pobieranie z YouTube to osobna morda roboty i osobne ryzyko prawne |

## 3. Licencje — przeczytać przed pobraniem

- **Cataract-1K: CC BY 4.0** — najczystsza sytuacja. Wystarczy cytować.
- **Cataract-101 i OphNet: NonCommercial** — praca naukowa mieści się w tym
  bez problemu.
- **Cataract-LMM: CC BY-NC-ND 4.0** — człon **ND, czyli NoDerivatives**, jest
  tu istotny. Trening modelu i publikacja wyników mieszczą się w normalnej
  praktyce badawczej, ale **redystrybucja przetworzonych klatek albo
  wyciętych fragmentów już nie**. Praktycznie: nie publikuj wyekstrahowanych
  klatek z tego zbioru jako części swojego materiału uzupełniającego. Publikuj
  listę identyfikatorów i kod, który je odtwarza.
- **SICS-105** — licencji nie potwierdziłem, trzeba otworzyć rekord Zenodo.

## 4. Problem, który trzeba rozwiązać przy nowych nagraniach

Powiedziałeś, że adnotacje nie są potrzebne, i to jest słuszne — ale
**propagacja z TSFA potrzebuje pierwszej ramki**, od której zaczyna śledzić.
W twoich nagraniach ziarna są gotowe: jedenaście tysięcy zaadnotowanych
punktów. W nowych nagraniach nie ma ani jednego.

Rozwiązanie w trzech krokach, wszystkie bez pracy ręcznej:

1. **Twój obecny model proponuje.** DETR albo YOLO przechodzi po nagraniu
   i zaznacza, gdzie widzi końcówkę narzędzia.
2. **Drugi model potwierdza.** Zostają tylko klatki, na których oba modele
   zgadzają się co do położenia. Zgodność dwóch niezależnych metod odsiewa
   większość błędów — ten sam pomysł, co konsensus trzech modeli w twoim
   pipeline tła z sierpnia 2025.
3. **Te potwierdzone klatki stają się ziarnami** i dopiero od nich rusza
   propagacja TSFA w obie strony.

Ryzyko do zmierzenia, nie do zignorowania: model widział dotąd tylko nagrania
z Brestu. Na nagraniach z Teheranu albo Bangaluru może się mylić systematycznie,
a wtedy propagacja utrwali jego błędy. **Dlatego pierwsze ziarna z nowego
zbioru trzeba obejrzeć ręcznie** — sto klatek wystarczy, żeby zobaczyć, czy
model w ogóle trafia w tej dziedzinie.

## 5. Co bym pobrał najpierw

**SICS-105.** Dwadzieścia dwie godziny, pełne HD, inna technika operacyjna,
jeden plik z Zenodo. To jest najtańszy sposób, żeby sprawdzić dwie rzeczy
naraz: czy twój model w ogóle działa poza Brestem i czy propagacja daje
sensowne ramki na obcym materiale.

Jeśli to zadziała, wtedy **Cataract-LMM** — bo tysiąc sto godzin z dwóch
szpitali zmienia artykuł z jednoośrodkowego na wieloośrodkowy, a to jest
zarzut, którego inaczej nie da się zamknąć.

**Nie pobierać wszystkiego naraz.** Cataract-1K to 90 GB, Cataract-LMM
przy tysiącu godzin będzie znacznie więcej, a ty masz już 144 GB samych
własnych nagrań na dysku E.

## 6. Co już masz lokalnie, a o czym można zapomnieć

W `E:\CrossDatasetsForModelsEval` leżą cztery zbiory w formacie COCO
z Roboflow, w tym `Cataract_1K.v3i.coco` — czyli **klatki** z Cataract-1K,
nie nagrania. Do propagacji się nie nadają, bo propagacja potrzebuje ciągłego
wideo. Ale jako dodatkowe zbiory ewaluacyjne są gotowe od ręki.

---

## Odnośniki

- Cataract-LMM, *Scientific Data* 2026 — https://doi.org/10.1038/s41597-026-07464-0 · preprint https://arxiv.org/abs/2510.16371 · kod https://github.com/MJAHMADEE/Cataract-LMM
- OphNet — https://arxiv.org/abs/2406.07471 · https://github.com/minghu0830/OphNet-benchmark
- Cataract-1K, *Scientific Data* 2024 — https://doi.org/10.1038/s41597-024-03193-4 · https://github.com/Negin-Ghamsarian/Cataract-1K
- SICS-105, *Scientific Reports* 2025 — https://www.nature.com/articles/s41598-025-00303-z · dane https://doi.org/10.5281/zenodo.13847781
- Cataract-101, MMSys 2018 — http://ftp.itec.aau.at/datasets/ovid/cat-101/ · https://doi.org/10.5281/zenodo.1220951
