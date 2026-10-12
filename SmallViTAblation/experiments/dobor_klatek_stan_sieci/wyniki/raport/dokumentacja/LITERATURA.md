# Co uzasadnia eksperyment z cząstką jako jednym patchem

Stan źródeł: 2026-09-22. Zakres: dobór klatek do uczenia detektora na podstawie
reprezentacji patchy i zmian sieci podczas treningu. To uzasadnienie hipotezy,
nie deklaracja potwierdzonej skuteczności Langevina.

## Dwa różne zegary

Niech `z[k,t,p] = phi(theta[k], I[t])[p]`, gdzie `k` oznacza checkpoint,
`t` klatkę filmu, a `p` ustalone miejsce na siatce patchy.

| Co zmieniamy | Co pozostaje stałe | Co mierzymy |
|---|---|---|
| `t`: kolejne klatki | checkpoint `k` | Reakcję ustalonej sieci na zmiany obrazu. |
| `k`: kolejne checkpointy | dokładnie ta sama klatka `t` i patch `p` | Zmianę reprezentacji tej próbki podczas uczenia. |

Nie wolno traktować kolejnych klatek jako kolejnych etapów uczenia ani łączyć
obu osi przez spłaszczenie tablicy. Jeden checkpoint pozwala badać pierwszy
zegar. Dwa checkpointy dają jeden przyrost drugiego zegara, nie serię
pozwalającą dopasować ośmiokrokową pamięć. „Cząstka” oznacza tutaj token patcha,
nie fizyczny punkt narzędzia. Między checkpointami ten sam patch wymaga
identycznego obrazu, deterministycznego preprocessingu, siatki i warstwy.
Między klatkami ten sam indeks siatki może pokazywać inną zawartość obrazu.

To rozróżnienie jest definicją projektu. Publikacje o training dynamics poniżej
analizują zmiany podczas uczenia, a nie sam upływ czasu nagrania.

## Sześć źródeł pierwotnych

| Źródło | Co wspiera | Granica wniosku |
|---|---|---|
| **Toneva i in., ICLR 2019, An Empirical Study of Example Forgetting during Deep Neural Network Learning.** [Artykuł autorów](https://arxiv.org/abs/1812.05159), DOI arXiv: `10.48550/arXiv.1812.05159`. | Przejścia tej samej próbki od poprawnej do błędnej klasyfikacji podczas uczenia opisują różnice między przykładami; autorzy badają selekcję danych według zapominania. | Używa wyniku zadania i historii treningu. Sam ruch wektora patcha nie jest zdarzeniem zapominania. Wynik dla klasyfikacji nie gwarantuje poprawy AP detektora. |
| **Swayamdipta i in., EMNLP 2020, Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.** [Oficjalny tekst](https://aclanthology.org/2020.emnlp-main.746/), DOI: `10.18653/v1/2020.emnlp-main.746`. | Średnia pewność prawdziwej klasy i jej zmienność między epokami pozwalają diagnozować dane. Trudność, niejednoznaczność i przydatność nie są tym samym; część trudnych przykładów zawiera błędne etykiety. | To empiryczne wyniki NLP, nie twierdzenie, że największa zmienność zawsze wybiera najlepsze klatki. Nie utożsamiamy niepewności zadania z normą zmiany latentu. |
| **Paul, Ganguli, Dziugaite, NeurIPS 2021, Deep Learning on a Data Diet: Finding Important Examples Early in Training.** [Artykuł autorów](https://arxiv.org/abs/2107.07075), DOI arXiv: `10.48550/arXiv.2107.07075`. | GraNd i EL2N wykorzystują gradienty lub błąd zadaniowy wcześnie w uczeniu; autorzy badają dobór danych i stabilność podprzestrzeni reprezentacji. To mocny punkt odniesienia dla prostego score bez modelowania dynamiki. | Wyniki uśredniane po inicjalizacjach nie stają się gwarancją dla pojedynczego checkpointu. Duża norma gradientu lub błąd mogą występować również dla szumu; publikacja nie potwierdza naszego score tokenów. |
| **Killamsetty i in., AAAI 2021, GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning.** [Oficjalny tekst](https://ojs.aaai.org/index.php/AAAI/article/view/16988), DOI: `10.1609/aaai.v35i9.16988`. | Użyteczność podzbioru wiąże się z poprawą zadania na wydzielonym zbiorze; przybliżenie Taylora prowadzi do zgodności gradientu podzbioru z gradientem celu. To bezpośredniejszy związek z uczeniem niż niespodzianka latentu. | Wyniki teoretyczne mają warunki dotyczące straty, gradientów i kroku uczenia. Zbiór używany do wyboru staje się częścią procedury treningowej; nie może jednocześnie pełnić roli nietkniętego testu końcowego. |
| **Lin, Lu, Journal of Computational Physics 424, 2021, Data-driven model reduction, Wiener projections, and the Koopman-Mori-Zwanzig formalism.** [Pełny tekst autorów](https://arxiv.org/abs/1908.07725), DOI: `10.1016/j.jcp.2020.109864`. | Redukcja układu do części obserwacji może prowadzić do pamięci i reszty. Praca łączy Koopmana, Mori–Zwanziga, filtrowanie Wienera i NARMAX: uzasadnia porównanie modelu pamięci z prostą autoregresją. | Dokładna tożsamość projekcyjna nie oznacza, że nasz skończony model poprawnie opisuje tokeny. Przy checkpointach optymalizator, learning rate i kolejność minibatchy wpływają na ewolucję; mały latent zwykle nie opisuje całego stanu. Lepsza predykcja tokenu nie dowodzi lepszego doboru danych. |
| **Kornblith i in., ICML 2019, Similarity of Neural Network Representations Revisited.** [Oficjalny tekst](https://proceedings.mlr.press/v97/kornblith19a.html). | Porównania reprezentacji wymagają określonej miary i jej niezmienniczości; autorzy przedstawiają CKA do porównywania struktur reprezentacji. Pomaga odróżnić zmianę współrzędnych od zmiany relacji między przykładami. | CKA jest diagnostyką zbioru reprezentacji, nie score użyteczności pojedynczego patcha. Nie dowodzi, że wyrównanie cech zawsze jest właściwe. Ewentualne PCA/normalizacja/wyrównanie muszą być dopasowane bez ocenianej grupy i bez przyszłych checkpointów. |

## Hipoteza możliwa do obalenia

Szerszy cel badawczy obejmuje pytanie, czy **zmiany tokenów tej samej klatki
między checkpointami informują o jej użyteczności do dalszego uczenia**.
Publikacje o training dynamics uzasadniają badanie tego celu, ale pierwszy
uruchamiany pilot nie rozstrzyga jeszcze takiego mechanizmu doboru.

**Hipoteza pierwszego pilota:** przy ustalonym checkpointcie L0 historia
tokenów w czasie filmu daje zadaniowy score wyboru klatek, który poprawia AP
względem bieżącej straty i placebo. Koopman i rozszerzony Langevin pracują tutaj
na zegarze filmu. Wybór zamraża się przed pięcioma treningami. Równolegle
osobna, stała sonda sprawdza reprezentację tych samych obrazów i patchy
między zachowanymi checkpointami treningu; ta diagnostyka nie wpływa na wybór.
Są to dwa osobne wyniki. Ich nazw ani dopasowania trajektorii nie traktujemy
jako dowodu skuteczności selektora lub mechanizmu uczenia.

Prosty rachunek wskazuje, dlaczego potrzebny jest związek z zadaniem. Dla
małego kroku SGD na przykładzie `i` i gładkiej straty na wydzielonym zbiorze
`R`, przy `g_i = grad L_i(theta)`:

```text
theta_next = theta - eta * g_i
L_R(theta_next) - L_R(theta)
    = -eta * <grad L_R(theta), g_i> + O(eta^2 * ||g_i||^2).
```

Jest to lokalne przybliżenie, nie gwarancja dla całego treningu ani AP.
Duży ruch tokenu może pochodzić z kierunku, który nie pomaga zadaniu. Duży
błąd przewidywania tokenu może być także szumem lub zmianą reprezentacji.

Rozstrzygający eksperyment zachowuje wspólne L0, jednakowy budżet dodatkowych
klatek i identyczny trening detektora. Wymaga co najmniej losowania, prostego
score zadaniowego, badanego score dynamiki i placebo niszczącego właściwe
przypisanie historii do próbki. Odpowiednia kontrola placebo zależy od tego,
który zegar modelujemy: tasowanie czasu filmu nie testuje mechanizmu zmian
między checkpointami. Ostatecznym wynikiem jest różnica AP na odrębnych
operacjach po takim samym treningu, nie sama korelacja lub błąd predykcji
latentu. Wynik pojedynczego seedu pozostaje pilotem.

Przy selekcji przed nowym treningiem nie można użyć jego przyszłych checkpointów.
Checkpointy pochodzą z jawnie wskazanego biegu pomocniczego. Jeżeli jego sieć
uczyła się na etykietach całej puli, należy raportować selekcję na danych
wcześniej opisanych i koszt tego biegu; nie jest to eksperyment uczenia tylko
z L0 ani oszczędzania niepozyskanych jeszcze etykiet. Korzystanie z GT do
oceny patchy jest dopuszczalne w takim eksperymencie, ale GT nie może trafiać
do wejścia ekstraktora ani przeciekać z operacji testowych.

## Sposób sprawdzenia źródeł

- **Scite:** metadane publikacji oraz fragmenty pełnego tekstu GLISTER
  (przybliżenie gradientowe) i Lin–Lu (pamięć i warunki modelu). Liczby
  cytowań nie były traktowane jako dowód skuteczności.
- **Undermind:** wyszukiwanie semantyczne dotyczące selekcji przez dynamikę
  uczenia oraz weryfikacja rekordów Toneva i Paul. Nie tworzono folderów,
  zdalnych dokumentów ani kolekcji.
- **paper-search CLI:** CrossRef potwierdził tytuł i DOI Dataset Cartography.
  Szerokie wyniki arXiv niepasujące do zapytania odrzucono; Semantic Scholar
  zwrócił ograniczenie zapytań. Dlatego treść potwierdzono w źródłach autorów
  i oficjalnych proceedings wskazanych wyżej.

To przegląd celowany dla projektu eksperymentu, nie systematyczny przegląd
wszystkich metod selekcji danych ani dowód nowości proponowanej metody.
