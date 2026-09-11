# Session: Session_2026-09-11_170251

**Created:** 2026-09-11 17:02:51
**Type:** Mixed (Tools + Debug)
**Status:** Completed

## Quick Summary
Sesja zaczęła się od alarmu: skrypt deduplikujący przeniósł trzy PDF-y do Kosza,
a komunikat wyglądał, jakby kasował trzy różne prace. Sprawdziłem tytuły w środku
plików — każdy usunięty plik miał w zbiorze bliźniaka co do bajtu, więc nic nie
zginęło. Naprawiłem prawdziwą przyczynę duplikatów w pobieraczu i dołożyłem
deduplikatorowi odczyt pierwszej strony, żeby mówił tytułami, a nie nazwami plików.

## Key Results
- Trzy usunięte pliki to Coreset Selection, RT-DETRv2 i AdaDeDup — każdy z bliźniakiem
  o identycznym skrócie SHA-256 i identycznym tytule. Fałszywy alarm.
- **Prawdziwy błąd był w pobieraczu**, nie w deduplikatorze: sprawdzał numer arXiv
  tylko w adresie z listy, więc link podany jako DOI wydawcy omijał odsiew.
  Po poprawce: 17 linków pominiętych jako posiadane, zero ponownych pobrań.
- Deduplikator czyta teraz pierwszą stronę PDF-a. Przy pliku idącym do Kosza podaje
  jego tytuł i nazwę bliźniaka, który zostaje.
- W grupach „ten sam numer, inne bajty" tytuł rozstrzyga: podobieństwo powyżej 0,85
  to dwie wersje jednej pracy, poniżej — dwie różne prace pod mylącą nazwą.
  Test kontrolny dał 0,02 i poprawny werdykt.

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
