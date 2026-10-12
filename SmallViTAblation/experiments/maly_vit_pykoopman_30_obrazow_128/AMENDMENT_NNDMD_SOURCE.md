# Jawny ślad źródła wykonania NNDMD

Zapisano 2026-10-02T13:14:12.737620+02:00, przed nowym końcowym testem. W czasie szybkiego procesu dopisano w funkcji evaluate dodatkowy plik decodedJ128 wycięty z tej samej prognozy full768 oraz poprawiono trzy nieaktualne docstringi. Kod załadowany na starcie pozostaje identyfikowany przez NNDMD_RUNTIME.json.

Odtworzony NNDMD_TRAIN_EXECUTED_SNAPSHOT.py ma dokładnie SHA256 d0e2115335c5d7b65718e8056ab5f2a5c6915acf79652ab04404c4392a7c2945. Źródło odtworzono przez odwrócenie zapisanych oryginalnych wywołań apply_patch, a nie zgadywanie treści. NNDMD_TRAIN_VS_EVALUATION.diff pokazuje całą różnicę. AST całego modułu po wyłączeniu funkcji evaluate i docstringów jest identyczny: trening, strata, wagi, normalizacja, walidacja, wybór i refit bez zmiany.

ZAMROZENIE_NNDMD.script_sha256 identyfikuje plik ewaluacji na dysku przy freeze. Nie jest przedstawiany jako hash wcześniej załadowanego kodu treningu. Oba źródła i oba hashe zostają zachowane. Nie zmieniono planu, kandydatów ani kryterium wyboru.
