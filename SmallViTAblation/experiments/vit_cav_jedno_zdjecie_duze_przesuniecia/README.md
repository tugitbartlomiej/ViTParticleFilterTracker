# Jeden stały CAV i duże przesunięcia

Uruchomienie: `F:/Python/python.exe -B -u experiments/vit_cav_jedno_zdjecie_duze_przesuniecia/run.py`. Pilot z `--smoke`. Oryginalny k1000, zdjęcie29 i jeden fixed full768 CAV. PLAN.md zamraża siatkę, primaryq i tiehandling przed biegiem. To optymalizacja oracle jednego obrazu, bez strojenia768 współczynników lub128kanałów i bez SD.

W outputs: CURVE.csv/json, RAW_MAPS.npz, COARSE_CORRELATIONS, CAV_COEFFICIENTS.csv, BASE_BEST_LATENTS.npz, BASE_BEST_POSITIVE_NEGATIVE.jpg, ALPHA_RESPONSE.png, RESULT, AUDIT, PROVENANCE, HASHES, PID i atomicPROGRESS z logami. Naturalbaseline i sztuczne najlepsze latenty zapisane osobno w jednymNPZ. J128 jest etykietą prezentacyjną, nie maską sterowania. Źródła pozostają read-only.
