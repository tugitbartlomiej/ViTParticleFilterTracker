# 00 — Research: co mówi literatura i co z tego bierzemy

Data: 2026-09-08. Źródła sprawdzone przez Scite i wyszukiwanie sieciowe
(Undermind nie był podłączony w tej sesji — MCP niedostępny).

---

## 1. Czy wolno badać dobór danych na małym modelu?

**Tak, i to jest opisana metoda, nie skrót.**

- **Selection via Proxy (SVP)** — Coleman i in., ICLR 2020, arXiv:1906.11829.
  Mały model zastępczy („proxy") służy do wyboru danych dla dużego modelu
  docelowego. Proxy powstaje przez zdjęcie warstw, zmniejszenie wymiaru albo
  krótszy trening; jest o rząd wielkości szybszy. Ma wyraźnie wyższy błąd, ale
  **rankingi przydatności próbek daje skorelowane z rankingami dużego modelu**.
  Na CIFAR10 proxy ponad 10× tańszy pozwolił usunąć do połowy danych bez straty
  na dokładności końcowej.
  → To jest cytowanie, które legitymizuje cały nasz plan. Mały ViT nie jest
  „gorszym eksperymentem", tylko **sondą**, a osobny krok potwierdzający na
  większym modelu domyka argument.

- **DeepCore** — Guo i in. 2022, DOI `10.1007/978-3-031-12423-5_14`. Biblioteka
  i empiryczne porównanie metod coreset selection. Ustala, co jest standardem
  raportowania w tej rodzinie metod.

- **EL2N** — trudność próbki jako średnia norma L2 wektora błędu, liczona bardzo
  wcześnie w treningu. **W oryginale to jest sygnał z modelu trenowanego krótko**
  (kilka epok), a nie z w pełni wytrenowanego modelu na całej puli. Nasz kod
  liczy EL2N z DETR po 170 epokach na pełnej puli — to jest odejście od metody
  i źródło zarzutu o błędne koło. Nowy protokół wraca do wersji oryginalnej.

- **Coverage-centric Coreset Selection (CCS)** — arXiv:2210.15809 i **D2 Pruning**
  — arXiv:2310.07931. Obie prace pokazują to samo: przy dużym stopniu
  przycinania sama trudność przestaje działać, bo zbiór traci pokrycie
  przestrzeni. Trzeba równoważyć trudność i różnorodność.
  → Uzasadnia, dlaczego mieszamy cztery sygnały, ale też każe zmierzyć pokrycie
  wprost, a nie zakładać.

**Ile ziaren losowych.** W pracach z tej rodziny spotykana praktyka to
**5 niezależnych ziaren dla metod i 20 losowań dla baseline'u losowego** —
losowy wybór ma największą wariancję, więc potrzebuje więcej prób. Nie jest to
formalny standard, ale jest to poziom, który recenzent uzna. Nasz plan: 3 ziarna
dla całej macierzy (ograniczenie czasu), 10 losowań dla baseline'u losowego,
5 ziaren dla konfiguracji trafiających do głównej tabeli.

---

## 2. Który mały ViT

Wymagania: (a) prawdziwy Vision Transformer, (b) szybka zbieżność, (c) standardowa
inferencja bez sztuczek, (d) licencja pozwalająca na publikację, (e) recenzent
wskazał go z nazwy.

### Wybór główny: **RF-DETR-Small** (backbone DINOv2 ViT)

| Wariant | Parametry | Rozdzielczość | COCO AP@[.5:.95] |
|---|---|---|---|
| RF-DETR-Nano | 30,5 M | 384×384 | 48,4 |
| **RF-DETR-Small** | **32,1 M** | **512×512** | **53,0** |
| RF-DETR-Medium | 33,7 M | 576×576 | 54,7 |
| RF-DETR-Large | 33,9 M | 704×704 | 56,5 |

- Backbone to **DINOv2**, czyli dokładnie ta sama rodzina cech, której używamy do
  selekcji (DINOv3 w pipeline). Ładna spójność narracyjna: cechy do selekcji
  i backbone detektora pochodzą z tej samej rodziny samonadzorowanej.
- Warianty Nano–Large są na **Apache 2.0**. XL i 2XL wymagają `rfdetr[plus]`
  na innej licencji — **nie używamy ich**.
- Publikacja: „RF-DETR: Neural Architecture Search for Real-Time Detection
  Transformers", arXiv:2511.09554, ICLR 2026.
- Dokumentacja podaje jako punkt wyjścia `epochs=50, batch_size=4` przy
  fine-tuningu; oficjalnych domyślnych wartości `lr` nie publikują wprost.
  **Nasze wartości trzeba wyznaczyć biegiem kalibracyjnym**, nie przepisać.
- Fine-tuning mieści się w 8 GB VRAM przy zmniejszonym batchu (Nano i Small) —
  czyli **RTX 3070 wystarcza do smoke testów lokalnie**, a Eden do produkcji.

### Druga architektura (test przeniesienia): **D-FINE-S**

- „D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution
  Refinement", arXiv:2410.13842, ICLR 2025 Spotlight.
- Recenzent wskazał ją imiennie i uzasadnił: w detekcji końcówki narzędzia liczy
  się dokładna lokalizacja ramki, a D-FINE poprawia właśnie regresję ramki.
- Uwaga: backbone D-FINE **nie jest ViT-em**, więc to jest test „czy efekt doboru
  danych przenosi się na inną rodzinę", a nie drugi mały ViT.

### Trzecia: aktualna YOLO

Recenzent 1 zarzut 7 i recenzent 2 zarzut 3 mówią to samo — YOLOv8 w 2026 roku
nie reprezentuje rodziny. Bierzemy jedną bieżącą wersję YOLO w rozmiarze
zbliżonym parametrami do RF-DETR-S, żeby porównanie było **iso-resource**:
ta sama rozdzielczość wejścia, ten sam budżet epok, ten sam rozmiar zbioru.

---

## 3. Czego literatura NIE popiera

Uczciwa lista, żeby nie wpisać do artykułu tezy, której dane nie unoszą:

1. **„Różnorodność zbioru daje generalizację"** — nasza własna analiza odcisku
   palca selekcji (sesja 2026-06-28) to obaliła. Warianty V2 i V3 miały odcisk
   praktycznie identyczny z pełnym pipeline'em, a wypadły siedemdziesiąt razy
   gorzej na obcym zbiorze. Zmierzona różnorodność była **wewnętrzna** (rozrzut
   w przestrzeni DINO na własnym rozkładzie), a generalizacja wymaga pokrycia
   rozkładu **zewnętrznego**. Nowy protokół mierzy to drugie wprost.
2. **„Każdy z czterech składników jest konieczny"** — obecna [Tab. 5] tego nie
   dowodzi, bo pełny pipeline miał inny trening niż warianty.
3. **Inferencja z jednego zapytania jako metoda** — recenzent uznał to za
   niekonwencjonalne i słusznie. Znika razem z DETR-em.

---

## 4. Co z naszych danych zostaje ważne

Policzone z `AdvancedDatasetSelection/ablation/detr/local_eval_2026-06-11/results/`,
kolumna `external` to **standardowa inferencja**, `external_q81` to jedno zapytanie:

| Wariant | external std | external Q81 | same-dist test |
|---|---|---|---|
| V1 pełny pipeline | 21,40% | 26,04% | 89,08% |
| V2 bez Fouriera | 2,19% | 0,37% | 47,51% |
| V3 bez EL2N | 1,51% | 0,29% | 59,24% |
| V4 tylko jakość | 1,54% | 2,94% | 85,10% |
| V5 losowy | 1,37% | 0,23% | 42,99% |
| V6 tylko różnorodność | 1,80% | 3,67% | 97,94% |
| V7 tylko trudność | 3,63% | 4,32% | 97,96% |

Dwie obserwacje, które przenoszą się do nowego artykułu niezależnie od nowych
treningów:

- **Jedno zapytanie zawyża rozpiętość.** Przy standardowej inferencji przewaga
  pełnego pipeline'u to czternaście razy, a nie dziewięćdziesiąt. To jest
  uczciwsza liczba i sama w sobie warta pokazania.
- **Wysoki wynik na własnym rozkładzie nie mówi nic o obcym.** V6 i V7 mają
  po około 98% na własnym teście i poniżej 4% na obcym. To jest najmocniejszy
  argument metodologiczny, jaki mamy — i on nie zależy od confoundu treningu.

---

## Odnośniki

- Coleman i in., *Selection via Proxy: Efficient Data Selection for Deep Learning* — https://arxiv.org/abs/1906.11829
- Guo i in., *DeepCore: A Comprehensive Library for Coreset Selection* — https://doi.org/10.1007/978-3-031-12423-5_14
- *Coverage-centric Coreset Selection for High Pruning Rates* — https://arxiv.org/abs/2210.15809
- *D2 Pruning: Message Passing for Balancing Diversity and Difficulty* — https://arxiv.org/abs/2310.07931
- *RF-DETR: Neural Architecture Search for Real-Time Detection Transformers* — https://arxiv.org/abs/2511.09554
- *D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement* — https://arxiv.org/abs/2410.13842
- Dokumentacja RF-DETR — https://rfdetr.roboflow.com/latest/
