from pathlib import Path
p=Path(__file__).with_name('space-inline.js')
h=p.parents[2]/'outputs/vit_k1000_taniec_czasteczek_20261009/WIZUALIZACJA.html'
s=h.read_text(encoding='utf-8')
start=s.index('// These views change');end=s.index('// Stable identities:',start)
extension=p.read_text(encoding='utf-8').replace('To mapa przestrzenna ruchu, nie odległości między latentami.', 'Osie mają niezależne skale. To mapa przestrzenna ruchu, nie odległości między latentami.')
s=s[:start]+extension+'\n'+s[end:]
s=s.replace('async function updateInspection(requestedStep){', "async function updateInspection(requestedStep,frame=null){if(spaceView()==='raw'){await updateRawInspection(frame||await rawFrame(opts()));return;}el('profile-panel').classList.add('hidden');")
s=s.replace("try{const o=opts();let d=provided||await loadFrame(o);d=transformSpace(d,o);if(ticket!==sequence)return;", "try{const o=opts(),viewRequested=spaceView(),axesRequested=channelAxes().join(','),patchRequested=selectedPatch;let d=provided||await loadFrame(o);if(ticket!==sequence||o.mode!==el('mode').value||o.image!==Number(el('image').value||0)||o.movement!==el('movement').checked||viewRequested!==spaceView()||axesRequested!==channelAxes().join(',')||patchRequested!==selectedPatch)return;d=transformSpace(d,o);")
s=s.replace("el('space-view').value='pca';el('raw-axes')", "el('space-view').value='pca';labelSpace();el('raw-axes')")
s=s.replace("el('sampling-note').textContent=o.mode==='exact'?", "el('sampling-note').textContent=spaceView()==='raw'?'600 cząsteczek w oryginalnych kanałach 768D. Rzeczywiste pełne stany: 0, 100, 500.':o.mode==='exact'?")
h.write_text(s,encoding='utf-8')
h.write_text(s,encoding='utf-8')
p.with_name('viewer_inline_check.js').write_text(s.split('<script>')[-1].split('</script>')[0],encoding='utf-8')
