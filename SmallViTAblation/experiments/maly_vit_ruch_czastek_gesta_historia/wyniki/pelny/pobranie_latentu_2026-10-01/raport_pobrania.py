"""Write the download report only from a completed checksum-verified receipt."""
from pathlib import Path
import json
from datetime import datetime

HERE = Path(__file__).resolve().parent
m = json.loads((HERE/'MANIFEST_LOKALNY_2026-10-01.json').read_text(encoding='utf-8'))
assert m['status']=='complete_verified' and m['sha256_match']
assert Path(m['local_path']).stat().st_size == m['bytes']
assert not Path(m['local_path']+'.part').exists()
start = datetime.fromisoformat(m['started_at_utc']).astimezone().isoformat()
end = datetime.fromisoformat(m['completed_at_utc']).astimezone().isoformat()
lines = ['# Pobranie pełnej historii tokenów z Edenu', '', 'Data: 2026-10-01. **Status: ukończone i zweryfikowane.**', '',
         f"Pobrano jeden istniejący plik `LATENT_GESTY.npy`: **{m['bytes']:,} bajtów** ({m['bytes']/1e9:.6f} GB; {m['bytes']/1024**3:.6f} GiB).".replace(',',' '), '',
         'Plik zawiera **1001 × 12 × 600 × 768 liczb float32**: pełne położenia tokenów dwunastu stałych obrazów przed uczeniem i po każdej z tysiąca aktualizacji. Nie uśredniono tokenów. Ruch wylicza się jako różnicę kolejnych stanów. [Dokładny opis danych](OPIS_DANYCH.md).', '',
         '- Źródło: istniejący alias SSH `eden-cluster`, plik w `~/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane/`.',
         '- Cel: [../dane/LATENT_GESTY.npy](../dane/LATENT_GESTY.npy).',
         f'- Początek lokalnego transferu: {start}.', f'- Zakończenie wraz z weryfikacją: {end}.',
         f"- Wolne miejsce po ukończeniu: {m['free_bytes_after']:,} bajtów ({m['free_bytes_after']/1e9:.3f} GB).".replace(',',' '), '',
         '## Weryfikacja', '',
         'Zdalnie odczytano nagłówek oraz obliczono świeży SHA256 całego pliku. Rozmiar i mtime nie zmieniły się podczas tej kontroli. Hash był również zgodny z metadanymi zapisanymi po eksperymencie. Po transferze obliczono SHA256 całego pliku lokalnego; wszystkie trzy wartości są identyczne:', '',
         f"`{m['local_sha256']}`", '',
         'Sprawdzono nagłówek NumPy, shape, float32, kolejność C oraz 24 małe próbki pełnych wektorów w krokach 0, 1, 500 i 1000. Próbki są skończone. Odczyt wykonano przez mmap, bez ładowania całej tablicy do RAM. Próbkowa kontrola wartości nie jest deklaracją ponownego pełnego skanowania wszystkich wartości.', '',
         'Transport: OpenSSH SFTP `reget` do pojedynczego `.part`. Co 10 sekund kontrolowano prognozowany końcowy zapas miejsca, z minimum 1 GiB. Po poprawnym hashu wykonano zmianę nazwy w tym samym katalogu. Nie utworzono drugiej pełnej kopii. Skrypt zachowuje nieukończony plik na wypadek potrzeby wznowienia.', '',
         'Małe obrazy i maski istniały już lokalnie. Ich sumy porównano ze źródłem Eden i umieszczono identyczne pliki obok historii, łącznie około 5,63 MB. Zachowano właściwe metadane biegu gęstego. Nie pobierano stanów AdamW i nie zmieniano ani usuwano niczego na Edenie.', '',
         'Samo pobranie nie uruchamia forwardów, treningu ani analizy. Odrębny, autoryzowany pilot pełnych tokenów ma własny katalog `experiments/maly_vit_dynamika_pelnych_czasteczek_checkpointy` i nie tworzy drugiego dużego cache.', '',
         '## Dowody i wznowienie', '',
         '- [Manifest zdalny](MANIFEST_ZDALNY_2026-10-01.json).',
         '- [Manifest lokalny i próbki](MANIFEST_LOKALNY_2026-10-01.json).',
         '- [Audyt miejsca przed transferem](AUDYT_MIEJSCA_2026-10-01.json).',
         '- [Sumy małych plików pomocniczych](METADANE_ZDALNE_2026-10-01.json).',
         '- [Skrypt pobrania i weryfikacji](pobierz_latent.py).', '',
         'Wznowienie w tym samym środowisku Python: `pobierz_latent.py download`. Poprawny finalny plik zostanie zachowany, nie nadpisany. Przed przenoszeniem lub kasowaniem czegokolwiek potrzebna byłaby oddzielna decyzja; ten skrypt nie usuwa danych.']
(HERE/'POBRANIE_LATENTU_2026-10-01.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('download_report_ready')
