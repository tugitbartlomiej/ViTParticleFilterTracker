# Identyfikacja Langevina i Koopmana

Zakończony lokalny pilot z 2026-10-01. Nie wykazał poprawy prognozy na 500 aktualizacji po dodaniu historycznych paczek. Sprawdzono 17 ramion, dwa istniejące nagrania, H=10,50,100,500; bez nowego treningu i bez Edenu. Drugi etap ma status eksploracyjny po obejrzeniu pierwszego testu.

- [PLAN.md](PLAN.md): protokół przed testami i jawne rozszerzenie po pierwszym pilotażu.
- [WYNIKI.md](WYNIKI.md): porównania poza dopasowaniem, ujemne wyniki i ograniczenia.
- [MODEL.md](MODEL.md): stan, wejście, wyjście, pamięć, estymacja i niepewność.
- [zrodla/LITERATURA.md](zrodla/LITERATURA.md), [zrodla/SOURCES.json](zrodla/SOURCES.json): 11 źródeł pierwotnych i zakres sprawdzenia.
- [matematyka/MATEMATYKA.md](matematyka/MATEMATYKA.md): wyprowadzenie, cztery twierdzenia Lean, dokładne kontrole SymPy.
- [CONFIG.json](CONFIG.json), `wyniki/AUDYT.json`, `wyniki/METRYKI.json`, `wyniki/PROGNOZY.npz`, `wyniki/MODELE.npz`: konfiguracja, dane maszynowe i małe artefakty; pierwszy etap zachowany w `wyniki/pilot_v1/`.

Powtórzenie analizy na dostępnych lokalnych nagraniach:

```powershell
& 'F:/Python/python.exe' 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_identyfikacja_langevin_koopman/kod/pilot.py'
```

Skrypt zapisuje własne wyniki i odczytuje stare nagrania przez memmap. Potrzebuje NumPy, SciPy i scikit-learn oraz obu oryginalnych małych cache s/ped/nowa. Kolejne uruchomienie zastępuje własne końcowe METRYKI/PROGNOZY/MODELE, zachowany pierwszy pilot pozostaje oddzielny. Nie uruchamia uczenia, pobierania ani zadań klastra. Repo nie otrzymało commita.

## Kolejny etap: bliższy zapis

[Wynik krótkiego pilota](wyniki/krotki_horyzont/WYNIKI.md): H1/3/5/10/20/50, przyczynowe historyczne wejścia, wybór na walidacji s47, eksploracyjny transfer s48. H1:80,98procent znaków wszystkich sond,64,96procent redukcjiMSE; H10 nie osiąga80. Sam Koopman, Koopman+GLE+wejścia, dawny Langevin i kontrole porównane osobno. Świeży odłożony przebieg potrzebny do potwierdzenia. Kod: `kod/krotki_horyzont.py`; jego wyniki są oddzielne odH500. Etap prognozy wielowymiarowej tożsamości tylko zaplanowany.

## EDMD, learned lifting i filtr reszt

[Wynik odrębnego pilota](wyniki/koopman_obserwator/WYNIKI.md): rzeczywiste finite-lagEDMD vs uczony encoder/decoder, cechy obrazów paczek, pamięć i skalarna ukryta resztaKF. Wybrane warianty nie poprawiają wcześniejszego direct modelu i nie osiągają80; zachowano negatywy. [PrzeglądAstry](PRZEGLAD_ASTRA.md), [literaturaobserwatorów](zrodla/LITERATURA_OBSERWATOR.md). Kod `kod/koopman_obserwator.py`, eksport/wykresy `kod/podsumowanie_koopman.py`; CPU, bez treninguViT.


Aktualizacja 2026-10-01 16:35:25 Europe/Warsaw: [etap joint controlled bezKF](wyniki/joint_controlled/WYNIKI.md), [Astra sterowanie](PRZEGLAD_ASTRA_STEROWANIE.md), [Lean/SymPy](matematyka/CONTROLLED_MEMORY.md), [pełne źródła i Undermind DeepSearch](zrodla/LITERATURA_JOINT.md). H1s48 znak80,92procent i redukcjaMSE64,12procent; brak przewagi nad dawnymdirect. s48eksploracyjne, samych obrazów nie utożsamiać z wpływamiN/P.
