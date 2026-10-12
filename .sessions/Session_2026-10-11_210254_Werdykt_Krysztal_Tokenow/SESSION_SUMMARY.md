# Session Summary: Werdykt — tokeny ViT tworzą pełzający kryształ na siatce patchy

## Metadata
- **Date:** 2026-10-11
- **Time:** popołudnie do 21:02
- **Status:** Completed
- **Type:** Analysis (Eden + lokalny render)

## Objective
Ustalić, jaką klasą dynamiki jest ruch tokenów ViT (600 tokenów na zdjęcie) podczas 60 epok uczenia: szkło, koloid w pułapkach, ciecz czy kryształ. Odpowiedzieć „czy to kryształ, w ilu procentach i jaki”. Pokazać to na filmach i zapisać jako dokument.

## Context
Wcześniejsze fale (sekcje 1–6d `README.md` w `outputs/porownanie_szklo_20261010`) traktowały ruch jako szkło. Poprzednia sesja: `Session_2026-10-11_173359_Szklo_Wiazki_Koloidy_60_Epok_Eden`. Dane: bieg 35 000 klatek, 806 checkpointów, eksport RP128 i 768D dla 100 zdjęć (`assets/RuchCzasteczek_ViT_szklo_2026-10/03_.../ruch60_rp128_50tr_50val`).

## Actions Taken
1. Na Edenie: eksport ruch60 (RP128 i 768D), test Markowa (piDMD, VAMP, jądro pamięci), analiza 60 epok, wspólny napęd (zdjęcia i Δθ), wiązki i kulki (W fc1 vs u), pozycja patchy, fale na siatce, Lindemann (Zahn–Maret), podprzestrzenie stałe i aktywne, podatność IoU i pewności.
2. Na Edenie: biegi kontrolne bez kodowania pozycji (flaga `--bez_pozycji` w `trening_pokretla.py`), CNN, lr 2e-3, ziarna 48 i 49, a po nich `siatka_gesty.py`.
3. Lokalnie (Python 3.13, matplotlib + ffmpeg): filmy `krysztal_tokenow`, `adres_tresc`, `krysztal_sily`, `szescian_0`, `komorki_0` i `iou_pewnosc` w `film_krysztal/`.
4. Dwie listy literatury (Undermind deep search) w `literatura_szklo_tokeny_20261011/linki/`.
5. Werdykt z tabelą procentową, sprawdzony z równoległym audytem `audyt_krysztalu_20261011/OCENA_KRYSZTALU.md`. Zapis do `WERDYKT_KRYSZTAL_2026-10-11.md`, odnośnik w README i w pamięci.

## Results

### Key Findings
- Sąsiedzi zgodni z siatką: 70–72% (ziarna 47, 48 i 49); bez kodowania pozycji 34%; CNN 57%; lr 2e-3 59%; tło losowe 1,3%.
- R² szablonu numeru patcha: 0,998 na starcie, 0,44–0,56 później; bez kodowania pozycji −0,03.
- Moc przyrostów w długich falach: 60–73% (po tasowaniu tokenów 3%). Wykładnik od −2,5 do −2,9. Zależność czasu relaksacji od długości fali jest widoczna tylko we wczesnym oknie.
- γ_L rośnie z wykładnikiem 0,27, bez plateau, czyli układ pełznie. Cos kierunku wiązania po 800 krokach wynosi 0,60–0,71.
- Węzły przesuwa uczenie: korelacja naporu na ściany z ich ruchem wynosi −0,01. Heterogeniczność jest zamrożona (0,87–0,89).
- IoU ustala głowica (68–93% gradientu ramki). Pewność ustala stem (51–65%) i końcowa norma.

### Issues Encountered
- Błędna interpretacja w rozmowie: „60% wiązań”. W rzeczywistości to średni cos kierunku wiązania. Poprawiono w dokumencie.
- Audyt wykrył niewyrównany czasowo przyrost wag w `eden/wspolny_naped_wagi.py` (`dTh[idx[:-1]]`). NAPED_WAGI trzeba policzyć ponownie.
- Heredoc w Bash nie przetworzył długiego markdownu. Dokument zapisano narzędziem Write.

## Files Generated/Modified
- `SmallViTAblation/outputs/porownanie_szklo_20261010/WERDYKT_KRYSZTAL_2026-10-11.md` (nowy, główny dokument)
- `.../porownanie_szklo_20261010/README.md` (odnośnik do werdyktu na górze)
- `.../porownanie_szklo_20261010/eden/*.py`, `*.slurm`, `eden/wyniki/*`
- `.../porownanie_szklo_20261010/film_krysztal/*.py`, `*.mp4`, `*.json`
- `experiments/maly_vit_szklo_pokretla_20261010/kod/trening_pokretla.py` (flaga `--bez_pozycji`)
- pamięć: `szklo-fala4-substancja-2026-10-11.md` (wpis o werdykcie i korektach), `assets-smallvit-przebiegi.md`

## Commands Used
- `ZOSTAW=1 ANALIZY=300:2188 sbatch --partition=hopper-2,hopper,short pokretla_pelny.slurm ...`
- `ssh eden-cluster 'tar cf - ...' | tar xf -` (pobieranie)
- `F:\Python\3.13\python.exe -X utf8 film_krysztal/film_*.py` (render)

## Next Steps
- [ ] Poprawić wyrównanie czasowe w `wspolny_naped_wagi.py` i powtórzyć test z prognozą poza oknem.
- [ ] Odjąć ruch sztywny (obrót, skala) i policzyć γ_L na reszcie.
- [ ] Porównać model odpowiedzi sprężystej z modelem samego skorelowanego wymuszenia.
- [ ] Dodać ziarna 2 i 3 kontroli bez kodowania pozycji.
- [ ] Opcjonalnie: `szescian_50.mp4` dla zdjęcia VALID i pobranie PDF-ów z list literatury.
