# Animacja kroków uczenia ViT w aplikacji: plan

Data: 2026-10-07. Stan: plan, czeka na decyzje użytkownika. Nic z tego nie jest jeszcze w kodzie.

## Cel

Nowa zakładka „Uczenie ViT”. Oś czasu to krok uczenia, a nie klatka filmu. Na tej samej klatce
wejściowej widać, jak 600 cząstek (patchy) i ich mikrowektory ruszają się z każdym krokiem
optymalizatora, od wag losowych do modelu końcowego.

## Co już jest (sprawdzone 2026-10-07)

Skrót `GH` = `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_ruch_czastek_gesta_historia\wyniki\pelny`.

| Zasób | Kształt | Gęstość |
|---|---|---|
| `GH\dane\LATENT_GESTY.npy`: latent `bloki[-1].mlp.act`, tryb eval | (1001, 12, 600, 768) float32, 22,1 GB | co 1 krok, 0..1000 |
| `GH\wagi\k0000.pt` .. `k1000.pt` | 101 checkpointów po 11,7 MB | co 10 kroków |
| `GH\dane\obrazy.npy`, `maski.npz` | 12 stałych klatek z 12 filmów, maski narzędzia i tła | stałe |
| `GH\dane\RUCH_KROKOW.json`, `LOG.json`, `OCENY.json` | ruch tokenów, strata, mAP@0,5 | 1, 50, 100 kroków |
| `experiments\maly_vit_rozne_latenty_5000\wyniki\pelny_*\dane\LATENT_KONTROLNY.npz` | 7 modeli, 12 klatek | co 500 kroków |

Bieg `GH`: zadanie 1818732, A100, ziarno 47, 1000 kroków, 1500 klatek, paczka 32, AdamW lr 5e-4,
rozgrzewka 300. Kopia na Edenie została usunięta przy sprzątaniu; lokalna jest pełna (SHA256 zgodne).

Na gałęzi `wizualizacja-uczenia-vit` (worktree `.claude\worktrees\wizualizacja-uczenia`) leży
specyfikacja etapu 1 z 2026-10-06 (`docs/superpowers/specs/2026-10-06-wizualizacja-uczenia-vit-design.md`):
płaszczyzna krok uczenia x etap w sieci, przeliczana ze 101 checkpointów co 10 kroków. Ta specyfikacja
nie korzysta z `LATENT_GESTY.npy`, więc nie daje każdego kroku. Ten plan ją uzupełnia, a nie zastępuje.

## Wnioski z literatury, które zmieniają implementację

1. **Stała baza, bez wyrównywania.** W jednym biegu kanał c w kroku t i t+1 to ten sam neuron.
   Kierunki mikrowektorów zostają stałe. Średnią, skalę i PCA liczymy raz z puli kroków, nie
   osobno na krok. PCA na krok daje przeskoki znaku i obroty, które wyglądają jak ruch, a nim
   nie są (Vernier i in. 2020, doi:10.1111/cgf.13977: globalna PCA najlepsza pod względem stabilności).
2. **Procrustes tylko jako przełącznik** „układ współruchomy”. Wyrównanie zjada prawdziwy obrót
   reprezentacji, który jest częścią uczenia (Cantareira, Paulovich 2020). Obok pokazujemy dwie liczby:
   wielkość obrotu i resztę po wyrównaniu (deformację).
3. **Mapa CKA krok x krok jako nawigator.** Bloki na mapie to fazy uczenia; w ViT są ostre
   (Shan i in. 2024, arXiv:2405.17377). Klik w mapę ustawia krok.
4. **Kontrola szumu.** Pętle trajektorii w PCA może dać też błądzenie losowe (Antognini,
   Sohl-Dickstein 2018, arXiv:1806.08805). Widmo wariancji porównujemy z 6/(π²k²).
5. **Normy globalnie.** Normy aktywacji rosną w uczeniu; normalizacja na krok by to ukryła.

## Etap A: zakładka na gotowej historii (bez treningu)

1. **Źródło.** `MANIFEST.json` wskazujący pliki w `GH` bez kopiowania. Odczyt przez mmap.
   Jedna klatka to 1001 x 600 x 768, około 0,9 GB w float16; trzymana w pamięci tylko wybrana.
2. **Sterowanie.** Wybór klatki (12), suwak kroku 0..1000 co 1, odtwarzanie (kroki na sekundę),
   pętla, krok w przód i w tył, skok do kroku z listy faz.
3. **Widoki.** Te same widoki 3D nad obrazem i przestrzeń makrowektorów, przez `ustaw_sekwencje`,
   z osią „krok” zamiast „klatka”. Te same kierunki i kolory. EMA po krokach.
4. **Odniesienie.** Do wyboru: środek z puli wszystkich kroków, kotwica w kroku końcowym
   (widać dochodzenie do stanu końcowego) albo kotwica w kroku 0.
5. **Wykresy pod osią kroku**, z kursorem bieżącego kroku:
   - strata (co 50) i mAP (co 100);
   - prędkość tokenów ‖Δz‖ (co 1), średnio i dla wybranego patcha;
   - odległość od kroku 0 i od kroku końcowego;
   - efektywny rząd tokenów, wariancja narzędzie / tło, rzut na kierunek LDA narzędzia.
6. **Detekcja i atencja** z najbliższego checkpointu wag (co 10). W GUI napis, z którego kroku
   pochodzą („atencja z wag kroku 990”), żeby nie udawać rozdzielczości co 1.
7. **Mapa CKA krok x krok** (liczona raz, zapisywana obok manifestu) i przełącznik Procrustesa.
8. **Eksport** kolejnych klatek animacji do PNG, opcjonalnie MP4 przez ffmpeg.

Koszt: odczyt i przeliczenie widoków to istniejące ścieżki kodu; CKA 1001 x 1001 to minuty na CPU.

## Etap B: nowy, lokalny bieg z pełnym zapisem (wymaga zgody przed uruchomieniem)

Braki w `GH`:
- mapy detekcji i strata nie są zapisane co krok;
- model jest słaby (mAP 0,31 po 1000 krokach);
- 12 klatek pochodzi z 12 różnych filmów, a nie z jednego filmu po kolei.

Propozycja: dopisać do `experiments\maly_vit_rozne_latenty_5000\kod\bieg.py` zapis przez `open_memmap`
(około 30 linii): latent co krok, mapy detekcji co krok, strata co krok, wagi co 50 kroków, float16.
Do decyzji:

| Wariant | Rozmiar |
|---|---|
| 8 klatek x 5000 kroków co 1 | 37 GB |
| 8 klatek, co 1 do kroku 300, dalej co 5 | około 9 GB |
| 32 klatki x 1000 kroków | 29,5 GB |

Na dysku F: jest około 84 GB wolnego. Czas na RTX 3070 to szacunkowo 2 minuty na 1000 kroków
plus zapis. Nie zmierzone. Lokalnie jest prościej niż na Edenie (kwota i wolny zapis na Lustre).

## Etap C (opcjonalnie, później)

- AlignedUMAP tokenów co 10 kroków, liczony offline i tylko odtwarzany (umap-learn).
- Widok kanałów w stylu M-PHATE (węzeł = kanał w danym kroku): które mikrowektory ruszają się razem.
- Lokalny współczynnik uczenia (LLC, devinterp) na checkpointach wag, na klastrze. Wymaga kalibracji.

## Pułapki do opisania w GUI

- Tryb eval używa biegnących statystyk BatchNorm łodygi; tokeny ruszają się też przy bardzo małym lr.
- Animacja dotyczy tylko stałych klatek próbnych, nie całego zbioru testowego.
- Rzuty nieliniowe (UMAP) zniekształcają odległości; domyślnie globalna PCA.
- Rozdzielenie narzędzia w latencie nie dowodzi, że głowica detekcji go używa.

## Decyzje do podjęcia

1. Czy zaczynamy od etapu A na `GH` (każdy krok, od razu, bez treningu)?
2. Czy etap A ma być lekką zakładką teraz, a oś „etap w sieci” ze specyfikacji etapu 1 dojdzie
   później, czy od razu łączymy oba projekty na jednej gałęzi?
3. Etap B: który model (tiny ma najlepsze mAP 0,654), ile kroków, jaka gęstość zapisu.
