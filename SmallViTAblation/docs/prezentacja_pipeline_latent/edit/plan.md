# Film: jak działa pipeline doboru zdjęć przez cel w latencie

Data: 2026-10-05. Źródło treści: `../PIPELINE_LATENT_VIT_SD_LANGEVIN.pdf` i obrazy w `../materialy/`.
Zatwierdzone przez użytkownika: siedem scen poniżej, 1920 × 1080, 30 kl./s, około 2,5-3 min.
Ustalenia domyślne (użytkownik nie wybrał): bez lektora i bez muzyki, polski tekst na ekranie.
Silnik: Manim Community 0.20.1 (czysty, bez ManimML), Python 3.11, LaTeX z MiKTeX.

## Styl wspólny

- Tło `#0E1A24`, panel `#16293A`, linie pomocnicze `#24425A`.
- Tekst `#EAF2F5`, przygaszony `#7C93A3`, główny kolor morski `#2BB3C0`, akcent pomarańczowy `#F08A2E`,
  zielony „dobrze” `#5BC27A`, czerwony „źle” `#E0565B`.
- Krój: `Segoe UI` dla tekstu, `Consolas` dla liczb i etykiet technicznych, `MathTex` dla wzorów.
- Każda scena zaczyna się od małego znacznika „KROK n / 7” w lewym górnym rogu i tytułu, a kończy krótkim
  wyciszeniem do tła (żeby sceny łączyły się bez skoku).
- Wygładzanie zawsze nieliniowe (`smooth`, `rate_functions.ease_out_cubic`); jedna nowa rzecz naraz.

## Sceny

| nr | tytuł | czas | treść |
|---:|---|---:|---|
| 1 | Pytanie | 18 s | czy da się tak dobrać zdjęcia, żeby detektor uczył się szybciej |
| 2 | Model | 28 s | siatka 600 tokenów, sieć, latent 600 × 768, token jako cząstka, która porusza się w uczeniu |
| 3 | Cel w latencie | 24 s | korekta 128 kanałów: pewność 86,79 → 93,17 %, IoU 85,66 → 99,93 % |
| 4 | Stable Diffusion | 28 s | szum → obraz; cel 51,35 wobec granicy 10,25: poza zasięgiem obrazów |
| 5 | Wyszukiwanie | 22 s | obraz SD jako zapytanie, 461 547 zdjęć, najbliższa prawdziwa klatka (cosinus 0,875) |
| 6 | Paczka i Langevin | 30 s | 32 zdjęcia → pchnięcie latentu; rośnie przez pęd do 3,2 w 5. kroku i wraca do ~0,4 |
| 7 | Pętla | 22 s | sześć kroków w obiegu zamkniętym, powrót do kroku 1 |

Animacje: `animations/slot_<n>/scena.py` → `animations/slot_<n>/render.mp4`. Montaż: `edl.json` + `render.py`.
