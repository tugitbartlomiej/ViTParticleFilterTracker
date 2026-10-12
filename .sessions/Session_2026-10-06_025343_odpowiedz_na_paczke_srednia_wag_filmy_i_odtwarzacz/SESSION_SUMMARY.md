# Podsumowanie sesji 2026-10-05 .. 2026-10-06

Rozmowa prowadzona z katalogu projektu grantowego (`F:\Studia\MCHTR\Grant\Projekt`), ale cała praca dotyczyła
repozytorium ViT, więc zapis jest tutaj. Gałąź repo ViT w chwili zapisu: `codex/eden-nano-training-20260914`.
Niczego nie commitowano.

## Cel sesji

1. Dokończyć pomiary dynamiki uczenia małego ViT (2,93 mln parametrów, cystotom w Cataract-LMM): jak latent reaguje
   na jedną paczkę, czy wagi mają trwały kierunek ruchu, czy odświeżany kierunek tożsamości śledzi mAP, czy średnia
   wag jest lepsza od stanu chwilowego.
2. Zebrać to w dużym przeglądzie `docs/28` („jak to badać tutaj”), z przekładem aparatu z szeregów czasowych.
3. Odpowiedzieć, co z tego wynika dla pipeline'u, który wybiera zdjęcia uczące maksymalizujące domknięcie latentu.
4. Zainstalować video-use, zrobić film pokazujący pipeline, potem drugi film z animacjami modeli i odtwarzacz
   zatrzymujący film w zaznaczonych punktach.

## Kontekst

- Poprzednia sesja: `Session_2026-10-05_012639_Sufit_Przewidywalnosci_Latentu_Rownanie_Z_Pedem_Paczki`
  (sufit przewidywalności, równanie z pędem, raport 27).
- Dane: gęsta historia tokenów (`maly_vit_ruch_czastek_gesta_historia\wyniki\pelny`, ziarno 47, kroki 0..1000),
  latent sond 10 ziaren, nagrania sił z `maly_vit_rownanie_latentu`.
- Równolegle w `outputs\prezentacja_pipeline_latent` pracowała sesja Codex (prezentacja PDF); jej plików nie ruszano.

## Wykonane kroki

1. **Odpowiedź latentu na paczkę** (`experiments\maly_vit_odpowiedz_latentu_na_paczke`). Model A dawał
   współczynniki niosące sprzężenie zwrotne, nie odpowiedź (K1 nie przeszedł); poprawka w
   `AMENDMENT_PROJEKCJA_LOKALNA.md` i analiza projekcją lokalną (`kod\odpowiedz_dluga.py`, horyzonty do 500).
   R(h): 0,98, 1,97, 2,72, 3,13, 3,22 (h 1-5), 2,40 (10), 1,23 (20), 0,64 (50), 0,51 (100), ~0,39 (500).
   Zysk modelu C nad B: +0,29 na 1 krok, 0,06 na 10, 0,002 na 50. Jedna paczka liczy się przez ~10 kroków.
2. **Kręgosłup trajektorii wag** (`experiments\maly_vit_kregoslup_trajektorii_wag`). PC1 0,886 wobec 0,811 dla
   czystego błądzenia losowego (8/π², wersja bez centrowania; poprawiono przypisanie do Antogniniego, który podaje
   wersję centrowaną). Mediana cos sąsiednich okien 100 kroków 0,051 (K2 nie przeszedł), odległe okna 0,10 i 0,16.
   Latent: sąsiednie okna −0,36, trwały dryf 2-5 procent. Poprawka w `AMENDMENT_BLADZENIE_LOSOWE.md`.
3. **Odświeżany kierunek tożsamości** (`experiments\maly_vit_tozsamosc_odswiezana_a_map`). d′ odświeżany 10,7
   w kroku 50 (mAP 0), potem ~7. K1 4/10 (nie przeszedł), K3 r = 0,12 [0,002; 0,25]. Kierunek stały: 27 w punkcie
   dopasowania, 9 po 50 krokach; cos z kierunkiem z kroku 1000: 0,45 po 500, 0,10 po 3500 krokach. Między ziarnami
   kierunki prawie ortogonalne, CKA 0,78 → 0,39. Poprawiono liczbę próbek (384, nie 372).
4. **Średnia wag a mAP** (`experiments\maly_vit_srednia_wag_a_map`). Surowe mAP drga o 0,041 na 10 kroków.
   Średnia z okna 100 kroków: +0,062 do +0,093 nad średnią surowych z okna; w kroku 1000 0,310 → 0,419
   (BN przeliczone nie zmienia wyniku). Ziarno 48, kroki 1400-3000: 9/9, +0,054. K0 i K2 przeszły.
5. **docs/28** (`docs\28_PRZEGLAD_I_PROGRAM_BADAN_LATENTU.md`): napisane sekcje 1-8 (słownik, aparat z szeregów
   czasowych, przekład pojęć, mapa prac 4.1-4.8, co się powtórzyło, pomiary 6.1-6.8, odpowiedzi 7.1-7.4,
   literatura 8.1-8.7). Zostały znaczniki `DALEJ-0`, `DALEJ-3`, `DALEJ-9`.
6. **Odpowiedź o pipeline doboru zdjęć**: kluczowe jest tylko równanie Langevina na 1 krok jako miernik; domknięcie
   narzędzie/tło jest pełne od kroku 50, brakuje dokładnej ramki (AP75 0,02 → 0,77). Propozycja (niewdrożona):
   średnia wag, ograniczona wyrocznia jako mapa braków, wyszukiwanie prawdziwych klatek, test 1-krokowy plus 20-50
   kroków, dołożenie zestawu i 500 kroków, SD tylko gdy wyszukiwanie nic nie da.
7. **video-use**: repo `C:\Users\bartl\Developer\video-use` (commit b877063, `.venv` Python 3.10), skill
   zarejestrowany junctionami w `C:\Users\bartl\.claude\skills\video-use` i `C:\Users\bartl\.codex\skills\video-use`
   (pierwsza próba przez `ln -sfn` w PowerShell zrobiła kopie, zastąpione junctionami). Klucza ElevenLabs nie
   podano, `.env` nie istnieje. `render.py` wymaga `PYTHONUTF8=1` (zapis `master.srt`).
8. **Film 1** (`outputs\prezentacja_pipeline_latent\edit\final.mp4`, 2:52, 7 scen Manim CE 0.20.1, czysty Manim
   zamiast ManimML na życzenie użytkownika). „łodyga” zamieniona na „stem”, długie myślniki usunięte z ekranu,
   przerywana linia „~0,4” w scenie 6 skrócona. Bez lektora i muzyki.
9. **Odtwarzacz punktów** (`edit\odtwarzacz.html`, punkty w `edit\punkty.js`): strzałka w prawo, spacja lub
   PageDown odtwarza do następnego punktu i zatrzymuje się; strzałka w lewo wraca; cyfry skaczą do punktu N.
   Wysuwane menu z kartami punktów i miniaturami, edycja czasu `m:ss.cc`, Ctrl+S zapis, Ctrl+Z cofanie, oś z rombami
   i dymkiem, chowanie paska i kursora po 2 s, pełny ekran, wybór pokazu `?punkty=plik.js`. 20 punktów filmu 1
   wybranych z nieruchomych fragmentów (`ffmpeg freezedetect`). Testy w Chrome headless przez CDP (`file://`):
   każde zatrzymanie dokładnie na punkcie. `python -m http.server` nie nadaje się (brak zakresów bajtów, nie da się
   przewijać).
10. **Film 2 „Modele w ruchu”** (`edit\final_modele.mp4`, 3:32, 6 scen w `animations\slot_m1..m6`, brief
    `animations\BRIEF_MODELE.md`, dane `edit\dane_animacji.json`, montaż `edl_modele.json`, 26 punktów w
    `punkty_modele.js`). Użytkownik chciał ruchu modeli zamiast animowanych napisów. Sceny: uczenie sieci (przód i
    gradient wstecz), kierunek tożsamości (LDA, dryf o zmierzony kąt), SD prowadzone celem (luka 10,25 wobec
    51,35), wyszukiwanie na sferze cech (cos 0,875), paczka i pęd (R(h)), trajektoria wag i średnia (0,416 wobec
    0,337). Obserwacja z danych: rzut trajektorii wag na 2 główne kierunki to gładki łuk, typowy dla błądzenia
    losowego. Sprawdzono kadry w 26 punktach i 5 przejść.

## Pliki zmienione / utworzone

### W repo ViT (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`)

- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_odpowiedz_latentu_na_paczke\` (kod `analiza_gramow.py`, `odpowiedz_dluga.py`, `AMENDMENT_PROJEKCJA_LOKALNA.md`, `WYNIKI.md`, `README.md`, `WYNIKI.json`, `wyniki\ODPOWIEDZ.json`, `GRAMY.npz`, `ANALIZA_GRAMOW.json`, `wyniki\dluga\ODPOWIEDZ_DLUGA.json`, `GRAMY_DLUGA.npz`)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_kregoslup_trajektorii_wag\` (kod `kregoslup.py`, `porownanie_z_bladzeniem.py`, `dryf_okna.py`, `AMENDMENT_BLADZENIE_LOSOWE.md`, `WYNIKI.md`, `README.md`, `WYNIKI.json`, `wyniki\`)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_tozsamosc_odswiezana_a_map\` (`PLAN.md`, `kod\odswiezana.py`, `WYNIKI.md`, `README.md`, `WYNIKI.json`, `wyniki\POMIAR.npz`, `WYNIK.json`, `UZUPELNIENIE.json`)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_srednia_wag_a_map\` (`PLAN.md`, `kod\dane.py`, `model.py`, `trening.py` skopiowane z kontrolą SHA, `kod\srednia_wag.py`, `kod\pozna_faza_s48.py`, `WYNIKI.md`, `README.md`, `WYNIKI.json`, `wyniki\SREDNIA_WAG.json`, `POZNA_FAZA_S48.json`)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\28_PRZEGLAD_I_PROGRAM_BADAN_LATENTU.md` (częściowy)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\outputs\prezentacja_pipeline_latent\edit\` (nowy folder montażu):
  - `plan.md`, `project.md`, `klatki_puli.txt`, `edl.json`, `edl_modele.json`, `dane_animacji.json`
  - `animations\slot_1..7\scena.py` i `render.mp4`, `animations\slot_m1..m6\scena.py` i `render.mp4`, `animations\BRIEF_MODELE.md`
  - `zrodla\scena_1..7.mp4`, `zrodla\model_1..6.mp4` (sceny z cichą ścieżką dźwięku)
  - `final.mp4`, `final_modele.mp4`, `base.mp4`, `clips_graded\`
  - `odtwarzacz.html`, `punkty.js`, `punkty_modele.js`
  - `verify\` (arkusze kadrów punktów i przejść)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\Session_2026-10-06_025343_odpowiedz_na_paczke_srednia_wag_filmy_i_odtwarzacz\` (ten zapis)

Pliki sesji Codex w `outputs\prezentacja_pipeline_latent` (PDF, skrypt prezentacji, `materialy\`) tylko czytane.

### Zewnętrzne (poza repo ViT)

- `C:\Users\bartl\Developer\video-use\` (klon, `.venv` przez `uv sync`)
- `C:\Users\bartl\.claude\skills\video-use` i `C:\Users\bartl\.codex\skills\video-use` (junctiony do repo video-use)
- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\` bez zmian w tej części sesji.
- Projekt grantowy `F:\Studia\MCHTR\Grant\Projekt`: bez zmian (tylko katalog roboczy rozmowy).

## Otwarte problemy

- docs/28 niedokończony: sekcja 0 (werdykt), 9 (program badań z etapami, kryteriami i kosztem; najpierw: czy zysk
  z doboru przetrwa na wagach uśrednionych), 10 (tabela twierdzeń do publikacji), 11 (czego nie dowodzi),
  12 (bibliografia z linkami), 13 (pochodzenie); usunąć znaczniki DALEJ; zaktualizować `docs\README.md` i
  `docs\28_material_przeglad_2026-10-05\STAN_PRAC.md`.
- Klucz ElevenLabs nie podany; video-use gotowe poza tym. Nie transkrybować bez prośby użytkownika.
- Bieg Eden z β1 = 0,99 nadal czeka na plan i zgodę.
- Filmy bez lektora i muzyki (decyzja domyślna).

## Następne kroki

1. Opinia użytkownika o filmie 2 i odtwarzaczu; ewentualne przyspieszenie scen (jeden render na scenę).
2. Dokończenie docs/28 i notatka w pamięci o uśrednianiu wag i funkcji odpowiedzi.
3. Pierwszy test pipeline'u: czy dobór zdjęć daje zysk na wagach uśrednionych (plan przed uruchomieniem).
