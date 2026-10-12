# 400 par train: oryginały i lokalne usunięcie cystotome

Gotowe **800 zdjęć: 400 prawdziwych train i 400 syntetyków LaMa**. 116 filmów, 12 przedziałów kąta osi po 33–34 zdjęcia. Oryginalne maski, maski edycji, bbox/tip, COCO, 600 tokenów, provenance i checksums są zapisane. Każda para zachowuje wszystkie piksele poza maską edycji.

[Raport](WYNIKI.md), [weryfikacja 400 par](WERYFIKACJA_PAR.json), [plan](PLAN.md), [manifest](manifest.json), [pary](PARY.json), [model i cache](model_lama/MODEL.json), [parametry inpaint](INPAINT_CONFIG.json), [plansza 12 par](PARY_LAMA_12.jpg), [plansza 48 oryginałów](PLANSZA_48.jpg).

Model w standardowym cache HF_HOME=F:/HuggingFace; istniejący ONNXruntime na CPU. Te same indeksy tokenów maski oryginału w obu obrazach. Tło pod narzędziem jest syntetyczną rekonstrukcją, inne instrumenty poza maską mogą pozostać. Jedna para dostała jawną poprawkę maski po zauważeniu ciemnego śladu; opis i zachowane wersje w raporcie. Manualny przegląd był próbkowy, bez gwarancji braku artefaktów wszystkich 400 syntetyków. To test reprezentacji na train, nie heldout detektora. Bez forwardów badanego ViT, treningu, Eden jobs i commitów.

Skrypty: przygotuj.py, audyt_dawnych_negatywow.py, pobierz_lama.py, lama_inpaint.py, sprawdz_pary.py, raport_inpaint.py. Inferencję całego zbioru wykonuje argument --all; bez niego uruchamia pilot sześciu par. Wznowienie pomija zapisane, sprawdzone pliki.
