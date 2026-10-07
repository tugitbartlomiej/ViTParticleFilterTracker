# Podsumowanie sesji 2026-10-07 04:32

## Cel sesji
1. Te same kolory wektorów w widoku 3D i w przestrzeni makrowektorów.
2. Zakładka do podglądania uczenia ViT krok po kroku i jego wpływu na ruch cząsteczek (deep research, plan, wykonanie).
3. Powiązania ruchu cząstek z atencją, gradientem i wagami.
4. „Musimy zrozumieć, z czego wynika ruch cząsteczki każdej jednej, i potem zamodelować jej ruch w sposób liniowy”;
   zgoda na trening na Edenie, pobieranie i usuwanie niepotrzebnych plików na Edenie.
5. Na później: ulepszenie uczenia sieci tym, co wiemy.

## Kontekst
Repozytorium `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, gałąź `codex/eden-nano-training-20260914`.
Poprzednie sesje: `Session_2026-10-06_220807_Aplikacja_PCA_Czubek_GUI_Ustawienia_Odpinanie`,
`Session_2026-10-07_002139_Male_ViT_Rozne_Latenty_5000_Eden`. Wiedza projektu o dynamice latentu: `SmallViTAblation/docs/23`,
`docs/28` (prognoza z samej historii utyka na 20-35 % umiejętności; równanie dz = a dz + b S na średniej 85 %).

## Wykonane kroki
1. `przestrzen_makro.py`: kolor cząstek domyślnie = kolor najsilniejszego mikrowektora, opcja mieszanki; EMA na
   siłach w jednej pętli; mikrowektory nieprzezroczyste. Odcisk 24 pozycji bez zmian. Commit cb867d0f.
2. Dwa podagenty: przegląd literatury o wizualizacji uczenia (dynamic t-SNE, M-PHATE, AlignedUMAP, CKA, Procrustes)
   i inwentaryzacja danych. Kluczowe: lokalna gęsta historia `experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny`
   (1001 kroków co 1, 12 klatek, 22 GB), na Edenie latent usunięty. Plan `SmallViTAblation/app/PLAN_ANIMACJA_UCZENIA.md`.
3. Zakładka „Uczenie ViT” (`app/zakladka_uczenia.py`, `app/analiza_uczenia.py`): suwak kroków, widoki 3D i makro z bazą
   PCA z puli kroków, środek makro (średnia / stan końcowy / krok 0), detekcja i atencja z wag co 10, wykresy (strata,
   mAP, prędkości, odległość od końca, rozdzielenie, rząd), mapa CKA. Cache klatek float16 w `app/cache_uczenia/`.
   Commit f9bb44df.
4. Przyczyny ruchu (`app/przyczyny_ruchu.py`): udział 13 grup wag (podmiana grupy), zmiana wag, cos(−dL/dz, dz),
   ruch z atencją z kontrolą. Klatka 3, krok 500: łodyga (z BatchNorm) 59 % ruchu; cos z gradientem tej klatki
   około 0; ruch z atencją 0,65 wobec kontroli 0,72. Commit cbab2c0c.
5. Dwa kolejne podagenty: wiedza projektu o dynamice latentu i literatura o liniowej dynamice uczenia (eNTK,
   Ren i Sutherland 2024, AdamW jako model stanowy, Kunin 2021 Langevin z momentem, Saxe 2013 mody).
6. Eksperyment `experiments/maly_vit_liniowy_ruch_czastek` (PLAN.md, WYNIKI.md):
   - mody czasowe: 5 wspólnych modów 73-75 % wariancji położeń; mody wspólne dla klatek (cos 0,99);
   - kolejne przyrosty skorelowane 0,69-0,74 (pęd AdamW), MSD nasyca się po 20-50 krokach;
   - oscylator na każdej cząstce (beta 0,72-0,77, kappa 0,04-0,05) najlepszy z samej historii: h = 1 usuwa
     58-61 % błędu persystencji, h = 50: 19-22 %; wspólny Koopman PCA-32 gorszy od persystencji.
   Commit 69c1eee4.
7. Eden: `~/smallvit_runs/2026-10-07_maly_vit_sily_czastek`, `bieg_sily.py` (siły nowej paczki, pędu, zaniku,
   BatchNorm na każdej cząstce 3 klatek w każdym kroku, 13 grup wag co 10). Próba 1823538: reszta liniowości
   mediana 0,009. Pełny bieg 1823539, dopasowanie 1823540 (próba, działa) i 1823541 (pełne, po biegu).
   Usunięto stary `/dev/shm/kont245-1814592-tII0K9` na węźle hopper (zgoda użytkownika).

8. Prognoza liniowa toru cząstki (oscylator) w zakładce „Uczenie ViT”, commit e5d96e38.
9. Wynik biegu sił (1823539, 1202 s) i dopasowania (1823541): suma czterech sił odtwarza ruch każdej cząstki
   w 98,9-99,7 %; jedno prawo ruchu z jednymi parametrami dla wszystkich cząstek
   v(t+1) = 0,87 v(t) + 0,99 S_paczki(t) (+ 0,5 S_BatchNorm, sprężyna 0,005) opisuje 96-98 % ruchu każdej cząstki
   na krokach testowych 700-1000. Pęd 64-68 %, paczka 31-35 %, BatchNorm 1 %, zanik 0 %. Bez siły paczki
   prognoza jednego kroku 54-60 %. Commit c59ffce6.

## Pliki zmienione / utworzone
### W projekcie ViT
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\app\przestrzen_makro.py`, `app.py`, `README.md`, `.gitignore`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\app\zakladka_uczenia.py`, `analiza_uczenia.py`, `przyczyny_ruchu.py`, `PLAN_ANIMACJA_UCZENIA.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_liniowy_ruch_czastek\` (PLAN.md, WYNIKI.md, kod\, kod_eden\, wyniki\)
### Zewnętrzne
- Pamięć: `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\eden-gesta-historia-tokenow-malego-vit.md` (aktualizacja), `feedback-agenci-limit-i-model.md` (nowa)
- Eden: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-07_maly_vit_sily_czastek/` (około 17 GB po biegu)

## Otwarte problemy
- Siła paczki jest wejściem z zewnątrz; jej przewidywanie z obrazów paczki (eNTK) niezmierzone.
- Drugi stary katalog `/dev/shm/plarge-1817335-6Xmkq4` nie był na węźle zadania; zostaje.
- Etap B planu animacji (nowy bieg z detekcją co krok) czeka na decyzję.

## Następne kroki
- Pokazać w aplikacji siły z biegu Eden (pobrać wybrane kroki, około 17 GB całość).
- Pokazać model liniowy w zakładce „Uczenie ViT” (prognoza obok rzeczywistego ruchu, strzałki sił).
- Etap 4: plan ulepszenia uczenia (EMA wag, BatchNorm, dobór paczek, lr), przed treningiem zgoda.
