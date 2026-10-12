# Cztery paczki po 32 zdjęcia train

Pełny retrieval czterech naturalnych latentów SD w jednym przejściu 310257 zdjęć train. Ranking raw MSE600x768, top64 z deduplikacją pikseli, pojedyncze forwardy finalistów i top32 na zapytanie. Zobacz PLAN.md.

Uruchomienie z katalogu SmallViTAblation:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
F:/Python/python.exe -u experiments/vit_sd_cztery_obrazy_paczki32/szukaj.py --smoke
F:/Python/python.exe -u experiments/vit_sd_cztery_obrazy_paczki32/szukaj.py
```

Stan bieżący: outputs/vit_sd_cztery_obrazy_paczki32/PROGRESS.json. PID i logi w tej samej rodzinie; status końcowy w RESULT.json, błędy w FAILURE.json. Każde zapytanie ma osobny podkatalog nazwany ziarnem. Brak automatycznego wznowienia; nie nadpisywać zakończonych wyników. GPU ustępuje aktywnemu vit_sd_zgodnosc90.

Użytkownik zakończył pełny skan. Dozwolona jest tylko finalizacja dotychczasowych rekordów: `F:/Python/python.exe -u experiments/vit_sd_cztery_obrazy_paczki32/szukaj.py --finalize-existing`. STOP_REQUEST blokuje przypadkowe ponowne skanowanie. Wynik końcowy opisuje wyłącznie przetworzony podzbiór.

Finalizacja zakończona i zweryfikowana: 88469 zapisanych zdjęć train, cztery paczki po 32, 128 różnych pikselowo zdjęć globalnie. Status PARTIAL SCAN. Oryginały, stykówki, rankingi i latenty w katalogach czterech ziaren pod outputs/vit_sd_cztery_obrazy_paczki32.
