"""Ścieżki aplikacji w jednym miejscu (moduły leżą w podpakietach, więc Path(__file__).parent nie wskazuje katalogu aplikacji).

- APP_DIR: katalog aplikacji (LaserLatentSpaceApp), w nim cache, ustawienia, zrzuty i logi;
- PROJECT_ROOT: katalog projektu SmallViTAblation (experiments/, assets/, outputs/);
- CACHE_DIR: dane liczone przez aplikację (historie, etykiety); nazwa katalogu bez zmian, bo wskazują na niego
  junction-y starego edytora (`../app/cache_uczenia/{lasery,etykiety}`) i ścieżki historii zapisane w ustawieniach."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
DATASET_ROOT = PROJECT_ROOT.parent / 'Dataset'
CACHE_DIR = APP_DIR / 'cache_uczenia'
SCREENSHOTS_DIR = APP_DIR / 'screenshots'
SAMPLE_IMAGES_DIR = APP_DIR / 'sample_images'            # zdjęcia testowe z filmów Cataract-LMM (lista „Zdjęcie”)
SETTINGS_FILE = APP_DIR / 'settings.json'
OLD_SETTINGS_FILE = APP_DIR / 'ustawienia.json'          # nazwa sprzed refaktoru: przenoszona przy pierwszym starcie
LOG_FILE = APP_DIR / 'run.log'
OUTPUT_LOG_FILE = APP_DIR / 'output.log'
