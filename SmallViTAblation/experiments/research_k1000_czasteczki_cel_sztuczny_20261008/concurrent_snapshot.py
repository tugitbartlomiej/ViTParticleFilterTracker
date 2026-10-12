from pathlib import Path
import hashlib,json,datetime
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/Path(__file__).parent.name
SNAP=OUT/'evidence/final-read'
SNAP.mkdir(exist_ok=True)
files=['PLAN.md','pelne_latenty.py','test_pelne_latenty.py','przygotuj_manifest.py','uruchom_oracle.py']
records=[]
for name in files:
    rel='experiments/vit_k1000_1500_pelne_latenty_90_20261008/'+name
    blob=(ROOT/rel).read_bytes();(SNAP/name).write_bytes(blob)
    records.append({'path':rel,'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)})
(SNAP/'MANIFEST.json').write_text(json.dumps({'read_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records},ensure_ascii=False,indent=2),encoding='utf-8')
p=OUT/'RAPORT.md';s=p.read_text(encoding='utf-8')
a=s.index('## 13.');b=s.index('## 14.',a)
s=s[:a]+'''## 13. Pamięć, kontrakt zapisu i równoległe zmiany

[CPU] Jeden pełny stan 1500×600×768 FP32 zajmuje 2 764 800 000 bajtów. Zapis wszystkich 501 stanów 0–500 wymaga 1 385 164 800 000 bajtów, czyli około 1,385 TB, bez gradientów i momentów. To wyliczenie scenariusza gęstego, nie stwierdzenie, że taki zapis wykonano.

[REPO, pierwsza migawka] Pierwotnie odczytany plan przewidywał before/after/delta oraz trzy pełne milestone, surowo 16 588 800 000 bajtów. Gęsty panel 256 cząsteczek, 501 stanów, 768 osi i czterech pól FP32 to dodatkowe 1 576 009 728 bajtów przy jednakowej długości wszystkich pól. Panel musi oznaczać 256 cząsteczek łącznie; 256 per obraz zwiększyłoby ten składnik do około 2,364 TB. Pełne rzadkie stany nie umożliwiają identyfikacji gęstej dynamiki wszystkich 900 000 tokenów.

Pierwsza migawka wykazała także niezgodność: PLAN wymagał FP32, a writer pelne_latenty.py:102 i test:80 używały float16. Dowody tej wersji pozostają zachowane; nie należy traktować ich jako gwarantowanego bieżącego stanu repozytorium.

**[REPO, końcowy ponowny odczyt] Równoległy wykonawca zmienił plan i kod podczas researchu.** Writer używa teraz float32, również przy zbieraniu milestone. Bieżący plan zachowuje pełne kroki 100 i 500; krok 0 jest before bez duplikacji. Panel to 16 obrazów × 8 tokenów, czyli 128 cząsteczek łącznie. Runner przekazuje tokeny z panelu przypisanego do indeksu obrazu, więc nie jest to 128 per każdy z 1500 obrazów. Zapis z obejmuje 501 stanów, a gradient/m/v po 500 stanów przed aktualizacją. Rzeczywisty surowy budżet tak zadeklarowanych tensorów wynosi:

- before/after/delta: 8 294 400 000 bajtów;
- dwa milestone: 5 529 600 000 bajtów;
- panel 128×768×4×(501+3×500): 786 825 216 bajtów;
- razem: 14 610 825 216 bajtów, około 14,611 GB przed małą telemetrią i narzutem plików.

Bieżący plan deklaruje wstrzymanie deploy i dużych zapisów do zakończenia audytu miejsca. Ten research nie łączył się z Edenem, nie sprawdzał aktualnego df i nie potwierdza wartości wolnego miejsca z planu. Wymagane jest rzeczywiste ponowne sprawdzenie przed uruchomieniem.

Końcowe kopie i hashe są w evidence/final-read/MANIFEST.json. Nie uruchamiano cudzych testów ani ich produkcji. Korekta dtype w writerze jest odczytanym faktem, nie potwierdzeniem całego kontraktu zapisu. Przy wznowieniach trzeba sprawdzać dtype, identyfikatory stanów i zgodność momentów z krokiem.

Wszystkie twierdzenia o częstotliwościach i pamięci muszą dotyczyć gęstego panelu. Obecne pełne stany 0/100/500 są jeszcze rzadsze niż wcześniejsze 0/25/100/500. Próbkowanie panelu powinno pokrywać filmy, narzędzie/tło, trudność i wielkość ruchu. Nie sklejać niezależnych trajektorii w fałszywe przejścia.

'''+s[b:]
anchor='W rzeczywistym historycznym J128 delta ma 74,647 procent energii null. Dlatego MSE pełne 768 może mierzyć głównie coś, co nie ma wpływu na detekcję frozen tail.'
s=s.replace(anchor,anchor+' Po aktualizacji wag W2 jego kernel może się zmienić; niewidoczność dotyczy tego checkpointa i stałego residual. W ocenie uczenia potrzebna jest projekcja dla właściwego stanu oraz rzeczywisty efekt zadaniowy, nie uniwersalne odrzucenie tych współrzędnych.')
s=s.replace('nie ślepo porównywany surowych współrzędnych','nie ślepo porównywany w surowych współrzędnych')
s=s.replace('cel na stan po końcowym LayerNorm','cel na stanie po końcowym LayerNorm')
s=s.replace('64 mody POD wyjaśnia','64 mody POD wyjaśniają')
s=s.replace('po filmach lub niezależnych trajektorii','po filmach lub niezależnych trajektoriach')
s=s.replace('sondach×600×768','sondach × 600×768')
s=s.replace('control 1,121694;variant 8:1,121515','kontrola 1,121694; wariant 8: 1,121515')
s=s.replace('RMSE do celu:start','RMSE do celu: start')
s=s.replace('effect wariant minus kontrola cosine z target','cosine efektu (wariant minus kontrola) z kierunkiem celu:')
s=s.replace('podmian:IoU','podmian: IoU')
p.write_text(s,encoding='utf-8')
p=OUT/'CLAIMS_EVIDENCE_FAILTEST.md';s=p.read_text(encoding='utf-8')
s=s.replace('Plan: pełne stany 0/25/100/500 oraz gęsty panel 256','Pierwsza migawka: 0/25/100/500 i panel 256. Końcowy odczyt: 0/100/500 i panel 128')
s=s.replace('REPO, migawka: writer i test float16, PLAN FP32 | Niespójność odczytanej wersji','REPO: pierwsza migawka writer/test float16 przy PLAN FP32; końcowy writer już float32 | Niespójność pierwszej wersji; dtype writer poprawiony równolegle')
p.write_text(s,encoding='utf-8')
p=OUT/'REKOMENDACJE.md';s=p.read_text(encoding='utf-8')
s=s.replace('zgodność FP32 z writerem. Panel 256 oznacza 256 cząsteczek łącznie.',
    'zgodność FP32 z writerem i wznowieniem. Końcowy odczyt: writer już float32, panel 128 globalnie, budżet tensorów około 14,611 GB. Wciąż sprawdzić miejsce i kontrakt przed uruchomieniem.')
p.write_text(s,encoding='utf-8')
print('Zapisano końcową migawkę cudzych równoległych zmian; cudzych plików nie edytowano.')
