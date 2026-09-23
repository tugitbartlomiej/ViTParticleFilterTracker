# Kontrola matematyczna doboru klatek

Dokument główny: [wyprowadzenia i dowody](../../docs/Analiza_ponownej_publikacji/44_FORMALIZACJA_DOBORU_KLATEK_KOOPMAN_LANGEVIN.md).

Skrypt sprawdza algebrę, dokładne kontrprzykłady oraz nierówności na jawnych układach spełniających założenia. Nie używa obrazów, GPU ani Edenu. Nie trenuje modelu. Obliczenia skończone nie są formalnym dowodem twierdzeń uniwersalnych; dowody z założeniami zapisano w dokumencie.

Uruchomienie w PowerShell, w dowolnym katalogu:

```powershell
& 'F:\Python\python.exe' 'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\analysis\formalizacja_doboru_klatek\sprawdz_wyprowadzenia.py'
```

Zależności użyte podczas weryfikacji: SymPy 1.13.1, NumPy 2.2.5. Skrypt zapisuje raport `WERYFIKACJA.json` obok siebie. `--out` pozwala wskazać inną ścieżkę. Raport zawiera wersje bibliotek, ziarna losowości, zakres każdej kontroli oraz SHA256 skryptu i dokumentu. W razie błędu wykonanie przerywa asercja, a nowy raport nie powstaje; stary raport należy zawsze sprawdzić względem hashy i daty.

Najważniejszy test złożony sprawdza kolejno ograniczenie błędu równania z pamięcią, odtworzenia cech, przybliżenia gradientu i przewidzianej zmiany straty. Ograniczenia są weryfikowane na układach liniowych z odczytem kwadratowym. Weryfikacja nie przenosi ich automatycznie na RF-DETR.

## Częściowa formalizacja w Lean

Źródło znajduje się w istniejącym środowisku matematycznym: `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\ViTFrameSelection.lean`.
Nie modyfikowano wspólnego pliku importów `SpmProofs.lean` ani konfiguracji tego środowiska.

```powershell
& 'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\analysis\formalizacja_doboru_klatek\sprawdz_lean.ps1'
```

Skrypt używa zainstalowanego Lean 4.28.0 i istniejących zależności Mathlib. Parametry `-MathRoot` i `-LeanExe` pozwalają zmienić ich ścieżki. Powstaje `LEAN_WERYFIKACJA.json` z hashem źródła, listą twierdzeń i wypisanymi aksjomatami. Dodatkowy lokalny plik `.log` powtarza wyjście kompilatora. Brak audytu któregokolwiek twierdzenia, placeholder lub błąd kompilacji oznacza nieudaną weryfikację.

Lean sprawdza warunkowy przedział poprawy, jego konsekwencje dla wyboru klatek oraz przenoszenie błędów osobnych cząstek na opis klatki i gradient. Ograniczenie reszty Taylora i wrażliwość mapy gradientu są jawnymi hipotezami. Nie sprawdzono w Lean całego dokumentu ani empirycznych założeń RF-DETR.
