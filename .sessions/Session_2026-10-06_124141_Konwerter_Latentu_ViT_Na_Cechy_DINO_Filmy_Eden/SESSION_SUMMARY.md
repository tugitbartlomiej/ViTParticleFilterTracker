# Podsumowanie sesji 2026-10-06 12:41:41

## Cel sesji

Użytkownik poprosił o trening na Edenie, który nauczy się tłumaczyć latent ViT na cechy DINO: najpierw
opracować sposób, potem uczyć na filmach z checkpointem ViT, bez zapisywania klatek, z porównaniem latentu
i cech DINO danego zdjęcia liczonym w locie na GPU, na kilku GPU na DGX. W trakcie sesji doprecyzował:
uczyć tylko na filmach (MP4), wrzucić na Eden po przygotowaniu, monitorować bieg i poprawiać.

## Kontekst

- Repozytorium: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, gałąź `codex/eden-nano-training-20260914`.
- Wcześniejsze lokalne translatory ViT -> DINO (`vit_translator_latent_dino`, `vit_translator_mlp_pelny_latent`,
  `vit_translator_mlp_random10000`) nie biły samej średniej cech: cosinus 0,80 wobec 0,81, top1 1,4 procent w galerii 512.
- Wejścia wzięte z biegu SD `vit_sd_pelny_latent_filmy_eden_20261006` (zadanie 1823167 na `hopper`):
  `FILMY.json` (118 filmów TRAIN, 25 VAL) i `MODEL_VIT_K1000.pt` (SHA256 `3df42a6a...649e`, zgodny z lokalnym).
- Specyfikacja DINO z `vit_dino_faiss_35000/DINO_SPEC.json`: DINOv2 ViT-S/14, CLS 384 L2, PIL Resize 256
  bicubic + CenterCrop 224. Wagi i kod `vision_transformer.py` na Edenie mają te same SHA256 co w specyfikacji.

## Wykonane kroki

1. Przegląd istniejących klocków: strumień MP4 w RAM z biegu SD, loader ViT k1000, specyfikacja DINO, wyniki
   poprzednich translatorów. Sprawdzenie Edenu: `dgx-[1-4]` po 8 x A100 40 GB, na `dgx-2` 7 wolnych GPU
   i 20 wolnych rdzeni; kwota domowa 155 z 200 GB.
2. Projekt: na każdą klatkę trzy obrazy z tych samych pikseli: ViT 480x320 (INTER_AREA), DINO CLS 224 (DINO_SPEC),
   DINO tokeny 420x280, czyli siatka 20x30 jak tokeny ViT. Konwerter: stała standaryzacja kanałów latentu,
   Linear 768 -> 384, 6 bloków transformera, dwie głowy (CLS 384 i 600 tokenów 384), ok. 11 mln parametrów.
   Strata: (1 - cos CLS) + 0,5 (1 - cos tokenów) + 0,2 InfoNCE z negatywami ze wszystkich GPU, bez sąsiednich
   klatek tego samego filmu. Kontrola liniowa na uśrednionym latencie i średnia TRAIN w tym samym biegu.
3. Po uwadze użytkownika usunięta ścieżka JPEG: uczenie i ocena tylko z MP4.
4. Lokalny test dymny (RTX 3070, 2 filmy TRAIN, 2 VAL): `passed=true`, TF32 kontra FP32 cosinus CLS 0,999999,
   tokenów 0,99992; ok. 100 klatek/s na GPU, ok. 115 klatek/s na jeden proces dekodujący.
5. Wdrożenie `deploy.py --submit`: zadanie 1823170 na `dgx-2`, partycja `short`, 4 x A100. Test dymny na
   Edenie `passed=true`, ok. 500-1000 klatek/s.
6. Monitoring (narzędzie Monitor, odnawiane co 30 min). v1 na VAL: mediana pozycji 399 -> 278 -> 227 (krok 6000)
   -> 252 -> 289; cosinus TRAIN 0,92. Przeuczenie, a od kroku 10 000 kontrola liniowa lepsza.
7. Wersja 2 w osobnych katalogach (`kod_v2`, `wyniki/smoke_v2`, `wyniki/trening_v2`): augmentacje natywnej klatki
   (odbicie 0,5, wycinek 60-100 procent, jasność i kontrast 0,2) przed ViT i DINO, dropout 0,1, drop-path 0,1,
   maska 25 procent tokenów, weight decay 0,1. Lokalny test dymny `passed=true`. Zgłoszona jako 1823174
   z `--dependency=afterany:1823170`.
8. v1 zatrzymana łagodnie plikiem STOP po 10 120 krokach (1 h 06 min, COMPLETED 0:0), checkpoint zapisany.
   v2 ruszyła od razu; test dymny `passed=true`.
9. v2 na VAL: mediana pozycji 234 -> 168 -> 150 -> 139 -> 115 -> 110 -> 105 -> 101 -> 104 -> 101.
   Koniec po 20 000 krokach (2 h 16 min, COMPLETED 0:0): cosinus po odjęciu średniej 0,435, top1 1,5 procent,
   top5 6,3 procent, film trafiony 81 procent (losowo 6), cosinus tokenów 0,781. Kontrola liniowa: mediana
   pozycji 190, top5 3,1 procent, film 68 procent.
10. Pobranie wyników i `KONWERTER.pt` (SHA256 `5230af4e...7112`, zgodny z Edenem), test wczytania przez
    `KonwerterGotowy`. Opis wyników dopisany do README eksperymentu, wpis pamięci projektu.

## Pliki zmienione / utworzone

### W repozytorium ViT (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`)

- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\vit_konwerter_dino_filmy_eden\`:
  `PLAN.md`, `README.md`, `WYNIKI.json`, `dane.py`, `model.py`, `model_vit.py` (kopia `MalyViT`), `run.py`,
  `config.json`, `config_v2.json`, `trening.slurm`, `trening_v2.slurm`, `deploy.py`.
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\vit_konwerter_dino_filmy_eden\`:
  `SUBMISSION.json`, `SUBMISSION_v2.json`, `DEPLOYMENT_v2.json`, logi SLURM obu zadań, `v1\` i `v2\`
  (CONFIG, HISTORIA, OCENA, LATEST, SMOKE), `v2\KONWERTER.pt` (poza gitem, `*.pt` w `.gitignore`).
- Ta sesja: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-10-06_124141_Konwerter_Latentu_ViT_Na_Cechy_DINO_Filmy_Eden\`.

### Zewnętrzne (poza repozytorium)

- Eden: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_konwerter_dino_filmy/` (`kod`, `kod_v2`,
  `wejscia` z twardymi dowiązaniami do wejść biegu SD, `logi`, `wyniki/{smoke,trening,smoke_v2,trening_v2}`;
  po dwa checkpointy 141 MB na wersję).
- Pamięć: `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\konwerter-vit-dino-filmy-eden.md`
  i wpis w `MEMORY.md`.
- Projekt grantowy niezmieniony.

## Otwarte problemy

- Surowy cosinus CLS konwertera (0,71) jest niższy niż sama średnia TRAIN (0,83); składnik InfoNCE odpycha
  predykcje od wspólnego kierunku. Przed użyciem z indeksem FAISS trzeba to zmierzyć.
- Indeks FAISS 35000 zbudowano z JPEG-ów, a konwerter uczono na klatkach MP4; różnica domeny niezmierzona.
- Konfigurację v2 wybrano po obejrzeniu krzywej VAL v1, więc VAL nie jest w pełni niezależnym testem porównania
  wersji. Jedno ziarno, jeden checkpoint ViT (k1000).
- Na `dgx-2` leżą stare resztki wcześniejszych zadań: `/tmp/tftlat192_1805661`, `/dev/shm/bpiotrowski_1745442_gle`.
  Usunięcie czeka na zgodę użytkownika.
- W `wyniki/trening` v1 został plik `STOP`; przed wznowieniem v1 trzeba go usunąć.

## Następne kroki

1. Zmierzyć zgodność konwertera z indeksem FAISS 35000: predykcja z latentu zdjęć JPEG kontra cechy z indeksu.
2. Sprawdzić konwerter na sztucznym latencie (cel z `vit_mlp_cel_sztuczny_faiss_top32`).
3. Drugie ziarno v2 i odłożony test na filmach spoza TRAIN i VAL.
4. Ewentualnie mniejsza waga InfoNCE albo osobna głowa pod surowy cosinus, jeśli FAISS tego wymaga.
