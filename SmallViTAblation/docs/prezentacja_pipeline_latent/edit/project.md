# Film: pipeline doboru zdjęć przez cel w latencie

## Session 1 — 2026-10-05

**Strategy:** Brak nagrań, więc film złożony z siedmiu animowanych scen (Manim CE 0.20.1, czysty, bez ManimML) na
podstawie prezentacji `../PIPELINE_LATENT_VIT_SD_LANGEVIN.pdf` i obrazów z `../materialy/`. Bez lektora i bez muzyki,
polski tekst na ekranie, 1920 × 1080, 30 kl./s, ciemne tło i kolory zbliżone do prezentacji.

**Decisions:**
- Sceny zatwierdzone przez użytkownika: pytanie, model, cel w latencie, Stable Diffusion, wyszukiwanie, paczka i
  równanie Langevina, pętla. Każda scena budowana przez osobnego pomocnika, równolegle (`animations/slot_1..7`).
- Scena 6 pokazuje zmierzoną funkcję odpowiedzi na paczkę (180 sond, 2 ziarna, kroki 1000-2500) z
  `experiments/maly_vit_odpowiedz_latentu_na_paczke/wyniki/dluga/ODPOWIEDZ_DLUGA.json`.
- Montaż: `zrodla/scena_n.mp4` (scena z cichą ścieżką, bo `render.py` łączy sceny razem z dźwiękiem) →
  `edl.json` → `render.py --no-loudnorm --fps 30` → `final.mp4`.

**Reasoning log:**
- ManimML pominięty na prośbę użytkownika („jeśli lepsze animacje, rób”): własne kształty dają pełną kontrolę.
- „łodyga” zamieniona na „stem” w scenie 2 na prośbę użytkownika.
- Długie myślniki usunięte z tekstu na ekranie (sceny 3, 6, 7); przerywana linia „~0,4” w scenie 6 skrócona, żeby nie
  wyglądała na ciąg dalszy podpisu.

**Outstanding:** lektor (OmniVoice albo ElevenLabs) i ewentualny podkład muzyczny, jeśli użytkownik zechce.

## Session 2 — 2026-10-05

**Odtwarzacz punktów:** `odtwarzacz.html` + `punkty.js`. Strzałka w prawo, spacja albo PageDown odtwarza film do
następnego punktu i zatrzymuje się na nim; strzałka w lewo wraca do poprzedniego; cyfry skaczą do punktu N.
Edytor punktów (czas m:ss.cc, opis) zapisuje `punkty.js` przez Ctrl+S. 20 punktów startowych wybranych z nieruchomych
fragmentów filmu (`ffmpeg freezedetect`), sprawdzonych na arkuszu `verify/punkty_arkusz.png`. Test w Chrome headless
z `file://`: każde zatrzymanie dokładnie na punkcie i film stoi dalej. Przez `python -m http.server` przewijanie nie
działa (serwer bez zakresów bajtów), więc odtwarzacz otwierać bezpośrednio z dysku.

## Session 3 — 2026-10-05

**Odtwarzacz dopracowany:** wysuwane menu z prawej (zwęża film), karty punktów z miniaturami, oś z rombami i dymkiem,
chowanie paska i kursora po 2 s, Ctrl+Z, wybór pokazu przez `?punkty=plik.js` (menu „…”: „Pipeline: kroki 1-7”,
„Modele w ruchu”).

**Drugi film „Modele w ruchu”:** `final_modele.mp4` (3:32, 6 scen Manim w `animations/slot_m1..m6`, brief
`animations/BRIEF_MODELE.md`, montaż `edl_modele.json`, punkty `punkty_modele.js`, 26 punktów). Na życzenie
użytkownika ruch modeli zamiast animowanych napisów. Prawdziwe dane z `dane_animacji.json` (trajektoria wag z Grama,
mAP średniej wag, R(h), d′ i cos kierunku tożsamości). Schematyczne: chmury punktów, dolina, obszar osiągalny SD.
Uwaga z danych: rzut trajektorii wag na 2 główne kierunki to gładki łuk (typowy dla błądzenia losowego), nie
poszarpana droga. Sprawdzone: arkusz punktów `verify/punkty_modele_arkusz.png`, przejścia
`verify/przejscia_modele.png`, odtwarzacz w Chrome headless.
