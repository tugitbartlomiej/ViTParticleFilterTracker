# Audyt K1000: A/B i temporalne reżimy

Rodzina tylko odczytu istniejących źródeł i małych obliczeń CPU. Nie uruchamia modelu na obrazach, treningów, nowego fit DMD ani Slurma. Nie modyfikuje Hubble'a, historycznych źródeł, rejestrów rodzin ani AU.

Wynik: [RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/RAPORT.md).

Skrypty uruchamiać z korzenia SmallViTAblation:

```powershell
python -B experiments/vit_k1000_analiza_rezimow_astra_20261009/diagnostyka_lokalna.py
python -B experiments/vit_k1000_analiza_rezimow_astra_20261009/pobierz_diagnostyke.py
python -B experiments/vit_k1000_analiza_rezimow_astra_20261009/podsumuj.py
```

`diagnostyka_lokalna.py` czyta zapisane normy 768D, metadata, projekcję (wyłącznie confidence/IoU) i istniejące tablice A/B. Nie estymuje nowego modelu. Odtwarza indeksy 50 000 par PCA B tym samym RNG, aby policzyć udział małych delt.

`pobierz_diagnostyke.py` przekazuje tekst `diagnostyka_eden.py` przez SSH do istniejącego środowiska. Zdalny skrypt ma jeden wątek CPU, wyłączony zapis bytecode, odczytuje wyłącznie istniejące 16 NPZ, checkpoint i DMDC_MODEL. Drukuje JSON na stdout. Skrypt lokalny zapisuje go wyłącznie we własnym outputs i pobiera małe kopie źródeł/logów oraz sacct. Timeout 180 s, historyczny czas obliczeń 123 s. Nie jest zgłaszany job i nie powstaje plik zdalny. Pierwsza próba odczytu schematu checkpointa zakończyła się błędem klucza przed analizą; poprawiono odczyt istniejącego klucza `net`, finalny przebieg jest kompletny.

`podsumuj.py` zapisuje zwięzłe agregaty, hash wejść i kontrolę lokalnych linków. Nie zmienia historycznych plików. RAPORT.md jest dokumentem audytu, nie automatycznie generowanym opisem eksperymentu treningowego.

Ograniczenia: zero docelowych tokenów w panelu 128; nullspace policzony dla endpointów wszystkich 600 tokenów tylko 16 obrazów; liczby dla 1500 bazują na zachowanych normach pełnych 768D. Brak wniosku o generalizacji lub skutku niewykonanych ablacji.

Zastosowano zasady dowodowe `audit-project-correctness` oraz izolację rodziny `eksperyment`. Jawne ownership użytkownika wyklucza standardowe zapisy tych skilli do wspólnego registry/AU.
