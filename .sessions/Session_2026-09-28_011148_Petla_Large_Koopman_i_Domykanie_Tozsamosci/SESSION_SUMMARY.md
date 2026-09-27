# Sesja 2026-09-27/28: pętla Large od początku dostrajania i domknięcie latentu tożsamości

## Kontekst

Poprzednie kroki (sesja `Session_2026-09-25_203826_Dobor_Klatek_Kontynuacja_134` i pilot
`petla_uczenia_koopman_fastgan`, commit 12d3f02): na RF-DETR Nano pętla Koopman + FastGAN nie przyspieszała uczenia;
diagnoza użytkownika — pętla wybiera klatki, do których trening i tak dochodzi. Analiza literatury
(`petla_uczenia_koopman_fastgan/ANALIZA_WCZESNIE_I_SKALA.md`, 6682ee1) zaleciła: od początku dostrajania, większy
model, 3 ziarna, wybór „do nauczenia i jeszcze nienauczone”. Polecenie użytkownika: „To weź ten z DINOv2-B,
LargeDeprecated. Zrób”.

## Co zrobiono

1. **Kampania `petla_od_poczatku_large`** (plan i kod: 7190b27): RF-DETR LargeDeprecated (DINOv2-B, 560 px, siatka
   40 × 40 × 768), rozgrzewka k00 od wag COCO, potem 6 obrotów × 160 aktualizacji, trzy pętle (koopman, losowe, bez),
   ziarna 42/43/44, 134 nowe klatki na obrót. FastGAN i uczenie na Edenie (hopper, dgx), prognoza i wybór lokalnie.
   Dyrygenci `dyrygent_large.py` (stan w `wyniki/s*/STAN.json`).
2. **Naprawy w trakcie** (e5fecc1, e9055b2):
   - wybór: faiss-cpu i torch z F:\Python ładują dwa OpenMP (OMP Error #15) → `KMP_DUPLICATE_LIB_OK=TRUE`
     w etapie `wybor`; sprawdzone, że top-90 faiss = top-90 z iloczynu w numpy (20 zapytań, różnica 8e-7);
   - kwota Edenu: koniec każdego zadania zapisywał 3 × 1,57 GB .ckpt + FINAL.pth, przy 9 zadaniach wolne spadło
     z 38 do 4 GB; kontrola kwoty w `prognoza` zatrzymała pętlę; od k02 bez tych zapisów (`wagi_tylko_koniec`),
     następny obrót startuje z `model_0000160.pth`;
   - s43 `bez` k06 (1817673_2) padło w 0 s na dgx-3 (`cp -a` do /dev/shm, „Numerical result out of range”);
     ponowione jako **1817712** z `--exclude=dgx-3`, `STAN.json` s43 przepięty, dyrygent s43 uruchomiony ponownie
     (zostanie przerwany wyłączeniem komputera o 02:00).
3. **Wyniki** (`petla_od_poczatku_large/WYNIKI.md`, `WYNIKI.json`, `wyniki/ANALIZA.json`, `wyniki/SHA256.json`
   — 2198 plików binarnych, 15,4 GB): kryterium niespełnione. Pole pod krzywą mAP@0,5 najwyższe dla losowych
   we wszystkich ziarnach; koopman − bez w obrotach 1–3 −0,0023 (wymagane +0,005); koopman − losowe −0,0030
   (15/18 par ujemnych). Od k04 dokładanie klatek zaczyna pomagać (losowe − bez +0,006).
4. **Diagnoza domknięcia latentu tożsamości** (`domykanie_tozsamosci_large/diagnoza_domkniecia.py`,
   `wyniki/DIAGNOZA.json`) na żądanie użytkownika („szukać obrazów, które najszybciej domykają latent tożsamości;
   dobrze określić warunki domykania”): trzy warunki — W1 zatrzymanie (spełnione od k02), W2 niezmienniczość
   (R = rozrzut między wariantami obrazu / rozrzut między obrazami ≈ 0,78, rozpoznanie wariantu 0,31, przypadek 0,08;
   przesunięcia obrót −3,4, żółte tło +3,3, jasne narzędzie +3,7 przy rozrzucie 6,1 — niespełnione, bez postępu),
   W3 rozdzielenie (spełnione). Przyczyna: uczenie z `aug_config={}`, bez obrotów i zmian koloru.
5. **Projekt** `domykanie_tozsamosci_large/PLAN.md` (e9055b2), **czeka na „tak”**: wybór klatek po zmierzonym
   wpływie kroku uczenia na cel domknięcia (iloczyn gradientów detekcji i celu J), etap 0 (test przewidywania,
   Spearman ≥ 0,5, ~0,5 h GPU), etap 1 (domykanie i augmentacja × 3 ziarna × 3 obroty; losowe i bez z tej kampanii),
   kryterium (mniejsze pole pod krzywą R, mAP nie gorsze o > 0,005).

## Stan na koniec sesji

- Eden: zadanie 1817712 (s43 bez k06) w kolejce/biegu; 1810406 (równoległa sesja, wstrzymane) nietknięte.
  Kwota: ~12 GB wolnego przed k06, po kampanii ~7 GB.
- Komputer: wyłączenie zaplanowane na 02:00 (`shutdown /s /t 3600`, prośba użytkownika).

## Do zrobienia w następnej sesji

1. Po końcu 1817712: `<venv rfdetr>/python dyrygent_large.py --s 43` w `petla_od_poczatku_large` (wznawia z
   `STAN.json`, pobiera k06), potem `py -3.11 analiza_large.py`, uzupełnić `WYNIKI.md` (pole `bez` s43, kryterium
   z `analiza_large.py`), `diagnoza_domkniecia.py` ponownie, dopisać hasze nowych plików do `SHA256.json`.
2. Decyzje użytkownika do `domykanie_tozsamosci_large`: zgoda na etap 0/1; miejsce na Edenie (usunięcie starych
   katalogów `2026-09-26_petla_uczenia` 6,9 GB, `2026-09-25_kontynuacja_*`) — bez zgody nic nie kasować.
3. Sprawdzić, czy środowisko Edena ma albumentations (ramię `augmentacja`).

## Commity (repo ViTParticleFilterTracker)

7190b27 (kod i plan Large), e5fecc1 (poprawki uruchomienia, analiza), e9055b2 (diagnoza i projekt domykania),
commit z tą sesją i `WYNIKI.md`.
