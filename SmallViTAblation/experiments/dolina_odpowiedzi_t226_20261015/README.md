# Wizualizacja doliny odpowiedzi tokenu 226

Samodzielny eksperyment na zamrożonej głowie checkpointu s47/65640. Cel i metoda w `PLAN.md`, interpretacja i ograniczenia w `outputs/dolina_odpowiedzi_t226_20261015/OPIS.md`.

Uruchomienie z katalogu SmallViTAblation, PowerShell:

```powershell
& 'F:/Python/3.13/python.exe' -u -X utf8 `
  experiments/dolina_odpowiedzi_t226_20261015/policz.py
& 'F:/Python/3.13/python.exe' -u -X utf8 `
  experiments/dolina_odpowiedzi_t226_20261015/rysuj.py
& 'F:/Python/3.13/python.exe' -u -X utf8 `
  experiments/dolina_odpowiedzi_t226_20261015/interaktywny.py
& 'F:/Python/3.13/python.exe' -u -X utf8 `
  experiments/dolina_odpowiedzi_t226_20261015/sprawdz.py
```

`rysuj.py --stills` zapisuje tylko pięć plansz. Renderer zachowuje styl istniejących filmów i używa Pillow oraz ffmpeg/h264_nvenc. Wymaga NumPy, PyTorch, SciPy, Pillow, Plotly, timm (do niezależnej kontroli) oraz ffmpeg i ffprobe. Wykorzystuje istniejący lokalny runtime; nie instaluje zależności i nie uruchamia treningu.

HTML zawiera Plotly, zdjęcie i próbki; działa także po otwarciu bez serwera. Podgląd w Codex można uruchomić serwerem HTTP powiązanym wyłącznie z 127.0.0.1, wskazując katalog wyników. W tej sesji: port 8788.

Pełna współrzędna geometryczna punktu nie jest jego rzutem: rysujemy `(u,v,<x-x*,n>)`, lecz pięć równań sprawdzamy dla pełnego `x` o 192 składowych. Dwa kierunki o największej bezwzględnej krzywiźnie wybrano celowo. Baza jądra jest lokalna; nie twierdzimy, że 187 stałych prostych utrzymuje odpowiedź przy dowolnym przesunięciu.
