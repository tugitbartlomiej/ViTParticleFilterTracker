from pathlib import Path
import re
root=Path(__file__).resolve().parents[2]
out=root/'outputs/vit_k1000_taniec_czasteczek_20261009'
path=out/'WIZUALIZACJA.html';s=path.read_text(encoding='utf-8')
old="await Plotly.react('cloud',traces,layout,CONFIG);\n}"
new="""await Plotly.react('cloud',traces,layout,CONFIG);
if(!el('cloud').dataset.eventsAttached){el('cloud').on('plotly_relayout',event=>{if(event['scene.camera'])lastCamera=event['scene.camera'];});el('cloud').on('plotly_hover',event=>{const v=event.points[0].customdata;if(v)el('point-detail').textContent=`Obraz ${v[0]} · patch ${v[1]} · ruch 768D: ${FMT.format(v[2])}`;});el('cloud').dataset.eventsAttached='true';}
} """
assert old in s;s=s.replace(old,new,1)
# New views use one shared point-information handler; remove obsolete init-only handlers.
s=re.sub(r"if\(el\('cloud'\)\.on\)el\('cloud'\)\.on\('plotly_relayout'.*?\}\);if\(el\('cloud'\)\.on\)el\('cloud'\)\.on\('plotly_hover'.*?\}\);",'',s)
path.write_text(s,encoding='utf-8')
scripts=re.findall(r'<script>(.*?)</script>',s,re.S)
(Path(__file__).parent/'viewer_inline_check.js').write_text('\n'.join(scripts),encoding='utf-8')
readme=out/'README.md';r=readme.read_text(encoding='utf-8')
r=r.replace('**Cały zbiór:** te same 8000 cząsteczek wybrane deterministycznie z 900 000 w każdym zapisanym stanie. Dane lokalne zawierają wszystkie 900 000, ale przeglądarka pokazuje próbkę dla płynnego obrotu.', '**Cały zbiór:** wszystkie 900 000 cząsteczek, bez próbkowania. Bezpośredni renderer WebGL 2 rysuje je jednym wywołaniem `drawArrays(POINTS)`.')
r=r.replace('Wskazanie punktu pokazuje oba identyfikatory.', 'Kliknięcie punktu w chmurze GPU lub wskazanie punktu w pozostałych widokach pokazuje oba identyfikatory.')
r+='''\n## Renderer GPU\n\nDane docierają jako binarne Float32: x, y, z oraz norma zmiany w pełnych 768 wymiarach. Kolejność punktów to obraz × 600 + patch. Każdy z trzech rzeczywistych stanów zajmuje 14,4 MB, a komplet buforów wierzchołków 43,2 MB. To nie obejmuje pamięci przeglądarki, buforów obrazu ani odczytu przy wybieraniu punktu.\n\nShader odejmuje pozycję początkową tylko w trybie przesunięć, obraca kamerę i nadaje kolor. Nie oblicza nowych latentów i nie interpoluje czasu. Zachowana jest jedna skala długości dla wszystkich trzech osi; ramka PCA i skala pozostają stałe w czasie. Gęste punkty mogą zasłaniać się na ekranie. Licznik oznacza liczbę przesłanych wierzchołków, nie liczbę rozróżnialnych pikseli.\n\nPrzeciągnięcie obraca, kółko myszy przybliża, a przełącznik automatycznego obrotu pokazuje zmierzoną częstotliwość rysowania. Renderer zatrzymuje ciągłe odświeżanie w tle oraz w innych widokach. Suwak widoczności zmienia przezroczystość punktów. Przy braku WebGL 2 komunikat jawnie zgłasza ograniczenie; pozostałe widoki nadal używają Plotly.\n\n`WERYFIKACJA_SHADERA.json` potwierdza bitową zgodność pozycji i norm wszystkich 900 000 punktów we wszystkich trzech krokach z oryginalnym NPZ.\n'''
readme.write_text(r,encoding='utf-8')
print('Uaktualniono obsługę i opis GPU.')
