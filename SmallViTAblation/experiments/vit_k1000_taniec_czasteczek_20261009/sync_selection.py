"""Refresh only the owned inline image/patch inspector, preserving animation."""
from pathlib import Path
p=Path(__file__).with_name('selection-inline.js')
h=p.parents[2]/'outputs/vit_k1000_taniec_czasteczek_20261009/WIZUALIZACJA.html'
v=h.read_text(encoding='utf-8')
start=v.index('// Stable identities:');end=v.index('async function init(){',start)
v=v[:start]+p.read_text(encoding='utf-8')+'\n'+v[end:]
v=v.replace("'Wszystkie 600 cząsteczek tego obrazu. Dostępne rzeczywiste stany 0, 100, 500.'", "(META.dense_ready?'Wszystkie 600 cząsteczek tego obrazu. Rzeczywiste stany 0–500 co 5 kroków.':'Wszystkie 600 cząsteczek tego obrazu. Dostępne rzeczywiste stany 0, 100, 500.')")
v=v.replace("color:d.norm,cmin:", "color:d.norm,cmin:")
v=v.replace("color:colorMode?d.tokens.map((t,j)=>identityColor(colorMode===1?d.images[j]:t)):d.norm,cmin:", "color:colorMode?d.tokens.map((t,j)=>identityColor(colorMode===1?d.images[j]:t)):d.norm,showscale:colorMode===0,cmin:")
v=v.replace("marker:{size:o.image<0?4:6,color}", "marker:{size:o.image<0?4:6,color:colorMode===0?ids.map(j=>d.norm[j]):ids.map(j=>identityColor(colorMode===1?d.images[j]:t)),cmin:0,cmax:d.norm_scale,colorscale:[[0,'#53728b'],[.45,'#62d4dd'],[1,'#ffb967']],showscale:false}")
h.write_text(v,encoding='utf-8')
p.with_name('viewer_inline_check.js').write_text(v.split('<script>')[-1].split('</script>')[0],encoding='utf-8')
