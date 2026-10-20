# Miejsce zapisu eksperymentów ViT

Aktualna zasada, na wyraźne polecenie użytkownika z 2026-10-20:

Każdy nowy eksperyment umieszczaj w `experiments/<nazwa_eksperymentu>_YYYYMMDD__HH-MM-SS/`. Nazwa folderu musi zawierać nazwę eksperymentu, datę i godzinę (lokalny czas Europe/Warsaw).

Każdy eksperyment musi mieć dokładnie ten podział (zachowaj wielkość liter):

- `Kod_eksperymentu/` - kod eksperymentu, skrypty uruchomieniowe i konfiguracja;
- `Opis/` - obowiązkowy opis: cel, hipotezy, model/checkpoint, dane, procedura, metryki, ograniczenia i sposób odtworzenia;
- `Wyniki/` - wszystkie wyniki, dane pomiarowe, obrazy, latenty, modele, wykresy, raporty wynikowe, logi i archiwa.

Nie zapisuj kodu ani opisu luzem w katalogu eksperymentu. Wszystkie pliki wynikowe muszą znajdować się w `Wyniki/` lub jego podfolderach. Nie zapisuj wyników nowych eksperymentów w osobnym głównym `outputs/`.

W `Wyniki/` obowiązują podfoldery: `Code_Interpretujacy_Wyniki/` (kod analizy i wizualizacji), `Dane/` (dane, pomiary, logi, manifesty i inne surowe artefakty), `Wykresy/`, `Raport/`, `Krótki_Wniosek/`. Szczegółowe zasady są w `experiments/AGENTS.md`.

Przed przygotowaniem wykresów, raportu i krótkiego wniosku agent musi zapytać użytkownika, które z nich chce otrzymać. Nie twórz ich automatycznie. Wyraźne zamówienie lub wcześniejsza zgoda w tej samej sesji jest odpowiedzią - nie pytaj ponownie o już zamówione elementy. Sam pomiar i zapis danych nie wymagają dodatkowego potwierdzenia. Odmowa dotyczy przygotowania materiałów, nie podstawowej informacji o wykonaniu zadania.

Aktualna lokalna baza projektu to `W:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation`. Użytkownik wskazał ją 2026-10-20, gdy wcześniej wskazywane F: było niedostępne. Ten sam układ `Kod_eksperymentu/`, `Opis/`, `Wyniki/` stosuj na Edenie w bazie tego projektu.

Przykład wskazany przez użytkownika: `experiments/vit_wiazkiprzeciwstawne_20261020__18-10-00/` z podfolderami `Kod_eksperymentu/`, `Opis/`, `Wyniki/`.

Każdy agent, tworząc nowy eksperyment, musi od razu dopisać go do `experiments/experiments_index.md`: pełna nazwa folderu, odnośnik do folderu i krótki cel (po co eksperyment jest wykonywany). Nie usuwaj istniejących wpisów i nie duplikuj wpisu przy kontynuacji tego samego eksperymentu.

Eksperyment może wynikać z wcześniejszego eksperymentu. Jeżeli użytkownik zleca nowy eksperyment pochodny, utwórz dla niego nowy folder zgodny ze strukturą zamiast dopisywać nowy bieg do folderu rodzica. W `Opis/` i konfiguracji zapisz powiązanie z rodzicem: jego nazwę i ścieżkę, wykorzystane wyniki/dane, checkpoint z krokiem/epoką i SHA256, uzasadnienie nowego badania oraz zmieniane i dziedziczone ustawienia. Przy kontynuacji treningu podaj, czy odtworzono też optimizer, sampler i stany RNG. W indeksie połącz dziecko z rodzicem i odnotuj powiązanie również we wpisie rodzica; zachowaj jego wyniki bez zmian.

Powiązania zapisz również jako `Kod_eksperymentu/pochodzenie.json`. Przy wznowieniu zweryfikuj zgodność wag, optymalizatora, danych oraz liczenia kroków/epok i zachowaj globalny krok; jawnie opisz wszystkie wznowione i resetowane stany.

Ta zasada zastępuje wcześniejszy podział na `experiments/` i `outputs/` z 2026-10-05. Nie używaj katalogu projektu Grant jako docelowego miejsca zapisu tych eksperymentów.

Wyjątek od reguły `experiments/` na wyraźne polecenie użytkownika z 2026-10-06: aplikacja GUI do edycji latentu 600x768 i generowania obrazów z wytrenowanego SD (checkpoint `assets/StableDiffusionModel/best_checkpoint.pt`) jest w `app/` (`app.py`, `silnik.py`, `uruchom.bat`). Jej wyniki nadal zapisuje do `outputs/latent_editor_sd_20261006/`.

Siedem dotychczasowych rodzin zostało skopiowanych z projektu Grant wraz z wynikami. Historyczne JSON-y, migawki kodu i sumy kontrolne pozostawiają oryginalne ścieżki pochodzenia. Manifest kopii: `outputs/przeniesienie_latent_sd_20261005/MANIFEST.json`.

Nie uruchamiaj ponownie zakończonych eksperymentów tylko w celu zmiany ścieżek. Historycznych katalogów nie przenoś bez polecenia użytkownika. Nowe biegi zapisuj oddzielnie, aby zachować istniejące wyniki.
