# TCAV tożsamości narzędzia

Uruchomienie: `F:/Python/python.exe -B -u experiments/vit_tcav_tozsamosc_narzedzia/run.py`. Krótka próba: dopisz `--smoke`. Wymagane torch, timm, cv2, numpy, scipy, sklearn i matplotlib. PYTHONDONTWRITEBYTECODE=1 zabezpiecza inne katalogi.

PLAN.md zawiera hipotezy, podłogi i ograniczenia. dane.py jest jedynym loaderem. Kopia model_snapshot.py pochodzi z rodziny retrieval, bez zmian. Wyniki, CAV, gradienty, splits, finite differences, mapy, wykresy i raport są w outputs/vit_tcav_tozsamosc_narzedzia. PROGRESS.json jest atomowy, PID.json identyfikuje własny proces, stdout/stderr zachowują logi. RESULT.json i WYNIKI_TCAV.md oznaczają ukończenie; samo uruchomienie nie oznacza wyniku.
