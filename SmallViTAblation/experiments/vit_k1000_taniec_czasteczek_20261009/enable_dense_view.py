"""Upgrade view contract for real dense states, without pretending they exist yet."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[2];path=root/'outputs/vit_k1000_taniec_czasteczek_20261009/WIZUALIZACJA.html'
s=path.read_text(encoding='utf-8')
old="const actualStep=()=>el('mode').value==='exact'?Number(el('step').value):[0,100,500][Number(el('step').value)];"
new="""const availableSteps=()=>el('mode').value==='global'?(META?.global_steps||[0,100,500]):[0,100,500];
const actualStep=()=>el('mode').value==='exact'?Number(el('step').value):availableSteps()[Number(el('step').value)];
function setupTimeline(){const exact=el('mode').value==='exact',dense=el('mode').value==='global'&&META?.dense_ready,steps=availableSteps();el('step').max=exact?500:steps.length-1;el('speed').disabled=!(exact||dense);const stride=dense?steps[1]-steps[0]:1;el('speed').replaceChildren(...(dense?[stride,2*stride,4*stride]:[1,5,10]).map(v=>{const o=document.createElement('option');o.value=v;o.textContent=v===1?'Każdy krok':`Co ${v} rzeczywistych kroków`;return o;}));el('speed').value=dense?stride:1;el('mode').options[0].textContent=META?.dense_ready?`Cały zbiór · 0–500 co ${META.global_steps[1]-META.global_steps[0]} kroków`:'Cały zbiór · dostępne stany 0, 100, 500';}
"""
assert old in s;s=s.replace(old,new,1)
old="el('speed').disabled=mode!=='exact';}"
assert old in s;s=s.replace(old,"setupTimeline();}",1)
old="(el('mode').value==='exact'?Number(el('speed').value):1)"
new="(el('mode').value==='exact'?Number(el('speed').value):META.dense_ready&&el('mode').value==='global'?Math.max(1,Math.round(Number(el('speed').value)/(META.global_steps[1]-META.global_steps[0]))):1)"
assert old in s;s=s.replace(old,new,1)
s=s.replace("el('step').max=el('mode').value==='exact'?500:2;","setupTimeline();",1)
old="'Wszystkie 900 000 cząsteczek, bez próbkowania. Shader obraca i koloruje rzeczywiste pozycje; nie zmienia latentu. Przełączenie kroków pokazuje zapisane stany; nie interpoluje ruchu.'"
new="(META.dense_ready?`Wszystkie 900 000 cząsteczek. ${META.global_steps.length} rzeczywiście obliczonych stanów od 0 do 500. Bez interpolacji.`:'Wszystkie 900 000 cząsteczek. Obecnie tylko trzy zapisane stany; trwa przygotowanie pełnego ruchu co 5 kroków. Bez interpolacji.')"
assert old in s;s=s.replace(old,new,1)
s=s.replace('pełne stany w krokach 0, 100, 500. Podgląd globalny', 'gęsty zapis co 5 kroków pojawi się po ukończeniu i weryfikacji nowych obliczeń. Do tego czasu dostępne są pełne stany 0, 100 i 500. Podgląd globalny',1)
path.write_text(s,encoding='utf-8')
(Path(__file__).parent/'viewer_inline_check.js').write_text('\n'.join(re.findall(r'<script>(.*?)</script>',s,re.S)),encoding='utf-8')
print('Podgląd gotowy do gęstych rzeczywistych stanów po weryfikacji transferu.')
