# Organizacja eksperymentów - polecenie użytkownika z 2026-10-20

Każdy nowy eksperyment ma osobny folder:

`experiments/<nazwa_eksperymentu>_YYYYMMDD__HH-MM-SS/`

Data i godzina: Europe/Warsaw. Zachowaj dokładną nazwę folderu, jeżeli podał ją użytkownik.

Każdy agent musi od razu przy utworzeniu nowego eksperymentu dopisać go do `experiments_index.md` w tym katalogu. Wpis zawiera pełną nazwę folderu, odnośnik do folderu i krótki opis celu: po co ten eksperyment jest wykonywany. Kontynuacja istniejącego eksperymentu aktualizuje jego wpis zamiast dodawać duplikat. Zachowuj wpisy innych agentów; indeks ma być wspólną listą nowych eksperymentów.

Eksperymenty pochodne: jeden eksperyment może być podstawą kolejnego. Gdy użytkownik zleca nowy eksperyment oparty na poprzednim, utwórz nowy folder z własną nazwą, datą, godziną i standardowymi podfolderami. Nie mieszaj nowego biegu z wynikami rodzica. W `Opis/` i konfiguracji dziecka zapisz: nazwę i ścieżkę rodzica, wykorzystane wyniki i dane, dokładny checkpoint (krok, odpowiednik epoki, ścieżka i SHA256), motywację, ustawienia odziedziczone i zmiany względem rodzica. Jeśli dane treningu pochodzą z jeszcze innego biegu, zapisz również to źródło. Dla wznowienia treningu jawnie podaj, czy przywrócono optimizer, sampler i RNG, czy wykorzystano tylko wagi. W `experiments_index.md` dodaj odnośnik do rodzica we wpisie dziecka i do dziecka we wpisie rodzica. Historycznych plików rodzica nie zmieniaj.

Powiązania zapisz również w `Kod_eksperymentu/pochodzenie.json`. Zwykłe uzupełnienie analizy istniejącego pomiaru może pozostać w tym samym eksperymencie. Przy wznowieniu zweryfikuj wagi, optimizer, dane, sampler i RNG; opisz wszystkie wznowione i resetowane stany. Samego załadowania wag nie nazywaj pełnym wznowieniem. Epoki licz według rzeczywistej liczby przykładów i batcha, zachowując globalny krok.

W folderze eksperymentu muszą być:

- `Kod_eksperymentu/`: kod eksperymentu, skrypty uruchomieniowe i konfiguracja potrzebne do odtworzenia;
- `Opis/`: obowiązkowy opis eksperymentu (np. `OPIS_EKSPERYMENTU.txt`): cel, hipotezy, model i checkpoint, dane, procedura, metryki, ograniczenia i odtwarzanie;
- `Wyniki/`: wszystkie wyniki tego eksperymentu.

Zachowaj dokładne nazwy i wielkość liter. Nie zapisuj kodu, konfiguracji ani opisu luzem w katalogu eksperymentu.

Obowiązkowa struktura wyników:

```text
Wyniki/
├── Code_Interpretujacy_Wyniki/  # kod analizy, interpretacji i wizualizacji
├── Dane/                       # pomiary, tabele, logi, manifesty i surowe artefakty
├── Wykresy/                    # wykresy, galerie i atlasy wykresów
├── Raport/                     # szczegółowy raport wynikowy
└── Krótki_Wniosek/             # osobny krótki wniosek
```

Modele/checkpointy, obrazy, latenty i inne surowe wyniki można grupować w podfolderach `Dane/`. Archiwa materiałów zapisuj w odpowiednim podfolderze wyników. Kod wykonujący eksperyment należy do `Kod_eksperymentu/`; kod przetwarzający zapisane wyniki należy do `Wyniki/Code_Interpretujacy_Wyniki/`.

Przed przygotowaniem wykresów, raportu i krótkiego wniosku zapytaj użytkownika, które materiały chce otrzymać (tak/nie, osobno lub łącznie). Nie generuj tych materiałów automatycznie. Wyraźne zamówienie lub wcześniejsza zgoda w bieżącej sesji oznacza, że dany element jest już zatwierdzony; nie powtarzaj pytania. Bez odpowiedzi przygotuj tylko już zamówione elementy, a pozostałe pozostaw niewykonane. Nie blokuje to obliczeń, zapisu danych ani krótkiej odpowiedzi o statusie. Foldery mogą pozostać puste, jeśli użytkownik nie zamówił danych materiałów.

Nie rozdzielaj nowego eksperymentu między główne `experiments/` i `outputs/`. Ten sam układ obowiązuje na Edenie. Zachowaj stare wyniki i ich ścieżki pochodzenia; nie przenoś historycznych eksperymentów bez polecenia ani nie uruchamiaj ich ponownie tylko z powodu porządkowania.

Aktualny przykład użytkownika:
`experiments/vit_wiazkiprzeciwstawne_20261020__18-10-00/`
