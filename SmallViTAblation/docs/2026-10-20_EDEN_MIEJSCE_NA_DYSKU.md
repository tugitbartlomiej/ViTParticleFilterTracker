# Eden: miejsce na dysku (stan na 2026-10-20)

**Źródło liczb:** pomiary podane przez użytkownika w sesji 2026-10-20. W tym zadaniu nie wykonywano poleceń ssh, więc nie odtwarzałem ich na Edenie. Są to dane zgłoszone, nie sprawdzone przeze mnie. Porównania z plikami w repozytorium podano przy każdym punkcie.

Ścieżki względem `W:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\`.

---

## 1. Pomiary (zgłoszone przez użytkownika, 2026-10-20)

| Katalog / system plików | Limit | Zajęte | Wolne | Uwagi |
|---|---|---|---|---|
| `/mnt/evafs/faculty/home/bpiotrowski` (home użytkownika) | 200 G | 185 G | 16 G (93 % zajęte) | `smallvit_runs` ok. 134 G; tam są wagi źródłowe 24 seedów |
| `/mnt/evafs/groups/transformers_vsc` (projekt grupy) | 512 G | 36 G | 477 G | `bpiotrowski` w grupie ok. 36 G |
| `/mnt/evafs` (cały system plików, Lustre) | — | 1,6 P | 445 T | system współdzielony fizycznie; to nie jest limit użytkownika |

Dodatkowe informacje:

- `df` pokazuje limit katalogu (projektu), nie całego systemu.
- `lfs quota` dla użytkownika: 219,9 G łącznie; brak twardego limitu na poziomie uid.
- `lfs quota` dla grupy: 35,2 G.
- Dysk lokalny W: ma ok. 1899 GB wolnego.
- Jednostka „G” w zgłoszeniu nie jest doprecyzowana (GB czy GiB). Wnioski poniżej są prawdziwe w obu przypadkach.

## 2. Porównanie z wcześniejszymi odczytami w repozytorium

- **2026-09-30 (`lfs quota -p 1299 /mnt/evafs`):** zajęte 117 z 200 GiB i 168 499 z 200 000 plików; pliki były bliżej limitu niż gigabajty. Źródło: `SmallViTAblation\docs\24_OPIS_EKSPERYMENTU_KOOPMAN_LANGEVIN_OD_POCZATKU_DO_KONCA.md:3664`, które odsyła do `SmallViTAblation\docs\EDEN_SPRZATANIE_2026-09-30.md`. Ten odczyt dotyczył projektu 1299. Dzisiejsze liczby dotyczą katalogów home i grupy. Z plików nie wynika, czy to ten sam limit, więc nie łączę tych liczb.
- **Odczyt z raportu sfery (data w pliku nieznana):** „około 491 GiB wolne na Edenie” oraz „73,3 GiB lokalnie po pobraniu pilota”. Źródło: `SmallViTAblation\outputs\wiazki_sfera_abcd_20261019\RAPORT_PL.md`. Nie wiadomo z pliku, który katalog był mierzony.

## 3. Wnioski praktyczne (wyliczone z liczb z punktu 1)

1. **Nowe wyniki zapisywać w katalogu grupy**, np. `/mnt/evafs/groups/transformers_vsc/bpiotrowski/ViTParticleFilterTracker/SmallViTAblation/experiments/<nazwa>/`. Home (16 G wolne) służy tylko do odczytu wag źródłowych.
2. **Surowy format z raportu sfery nie mieści się w katalogu grupy.** Raport podaje 24 × 50 jako ok. 605,6 GB (564 GiB) i 24 × 500 jako ok. 6,057 TB (5641 GiB), źródło: `SmallViTAblation\outputs\wiazki_sfera_abcd_20261019\RAPORT_PL.md`. Wolne miejsce w grupie to 477 G. Format 24 × 50 przekracza je w obu jednostkach. 24 × 500 (6,06 TB) nie mieści się w katalogu grupy ani w home. Mieściłby się w całym systemie plików (445 T wolne), ale ten zasób jest współdzielony i nie jest limitem użytkownika.
3. **Format zwarty** (np. agregaty i snapshoty) zajmuje znacznie mniej. Przykład: compact trial zapisał dane o rozmiarze 31 755 318 B dla pełnego zakresu jednego obrazu. Źródło: `SmallViTAblation\experiments\wiazki_sfera_compact_trial_20261019\PLAN.md`.
4. **Przed dużymi zapisami** sprawdzać `df -h <katalog>` albo `lfs quota`.
5. **Rozmiar pojedynczego checkpointu nie został zmierzony** w tej sesji. Przed uruchomieniem eksperymentu wykorzystującego 288 wag trzeba go sprawdzić na Edenie (np. `ls -l` na katalogu wag).

## 4. Dostęp (zgłoszone przez użytkownika)

- Połączenie: `ssh eden` (ProxyJump przez `eden-jump` = `ssh.mini.pw.edu.pl`).
- Python środowiska kampanii przeciwstawnej: `/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/przeciwstawne_eden_20261017/environment/venv/bin/python` (torch 2.10.0+cu126, timm 1.0.24, numpy 2.2.6).

Dane o limitach GPU konta `transformers_vsc` (5 × H100 w pomiarze z 30.09) nie są częścią tego pliku. Znajdują się w `SmallViTAblation\docs\24_OPIS_EKSPERYMENTU_KOOPMAN_LANGEVIN_OD_POCZATKU_DO_KONCA.md:3679` i w `SmallViTAblation\docs\2026-10-20_PRACE_SESJI.md`, sekcja 5.
