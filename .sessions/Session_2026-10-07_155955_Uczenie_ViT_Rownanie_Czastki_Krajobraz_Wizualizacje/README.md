# Sesja 2026-10-07 15:59: Uczenie ViT, równanie cząstki, krajobraz i wizualizacje 768 wymiarów

## Cel sesji
Rozbudowa aplikacji „Edytor latentu” (zakładki filmu i „Uczenie ViT”) o narzędzia do zrozumienia ruchu cząstek
(tokenów) małego ViT: Langevin na filmie i w uczeniu, wizualizacje przestrzeni 768 wymiarów, równanie ruchu
pojedynczej cząstki z samej jej historii (Langevin, GLE, Koopman) oraz krajobraz, po którym biegają cząstki.

## Typ
explore

## Workerzy użyci
Cztery podagenty (opus, effort medium), tylko przeglądy literatury i odczyt, bez zmian w plikach:
wizualizacja przestrzeni 768 wymiarów; GLE i Koopman dla jednej cząstki; fizyczna analogia ruchu cząstek
(szkło, krajobraz energii, tunele) i opis „studni” TFT.

## Co zrobiono (skrót)
- Zakładka filmu: wycinanie obszarów na żywo, Langevin po klatkach.
- Uczenie ViT: ślad i prognoza Langevina (także średnia narzędzia), śledzenie czubka, „Animuj”, do 10 000 kroków/s.
- Widoki 3D i makro: mieszanka kolorów wektorów, zdjęcie zdeformowane do położeń cząstek, klik w cząstkę,
  zaznaczanie w obie strony, PCA jako RGB, podobieństwo, rzut ortonormalny, Grand Tour, atlas kanałów.
- Panel równania cząstek: Langevin, GLE z jądrem pamięci, Koopman (Hankel-DMD) z samej historii.
- Okno „Krajobraz i tunele”: teren −ln gęstość, wosk z tunelami, otoczka, test wspólnych kanałów.
Szczegóły: `SESSION_SUMMARY.md`.
