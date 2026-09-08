---
name: "source-command-wczytaj-ostatnia-sesje"
description: "Migrated source command `wczytaj-ostatnia-sesje`"
---

# source-command-wczytaj-ostatnia-sesje

Use this skill when the user asks to run the migrated source command `wczytaj-ostatnia-sesje`.

## Command Template

# Wczytaj Ostatnią Sesję

Wczytaj i pokaż zawartość ostatniej sesji z folderu `.sessions/`.

## Instrukcje

1. **Znajdź najnowszą sesję:**
   - Przeszukaj folder `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\`
   - Znajdź folder z najnowszą datą w nazwie (format `Session_YYYY-MM-DD_HHMMSS`)
   - Alternatywnie użyj daty modyfikacji folderu

2. **Wczytaj pliki sesji:**
   - Przeczytaj `SESSION_SUMMARY.md` z najnowszego folderu sesji
   - Opcjonalnie przeczytaj też `README.md` jeśli SESSION_SUMMARY jest krótkie

3. **Pokaż użytkownikowi:**
   - Nazwa sesji i data utworzenia
   - Pełne podsumowanie sesji (SESSION_SUMMARY.md)
   - Kluczowe wyniki i następne kroki

4. **Format odpowiedzi:**
   ```
   ## Ostatnia sesja: [NAZWA_SESJI]
   **Data:** [DATA]
   **Typ:** [TYP]
   **Status:** [STATUS]

   [Zawartość SESSION_SUMMARY.md]
   ```

## Przykład użycia

```
/wczytaj-ostatnia-sesje
```

Wynik: Pokaże pełne podsumowanie ostatniej sesji pracy.
