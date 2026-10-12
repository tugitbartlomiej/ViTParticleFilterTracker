"""Research-only OA PDF retrieval; read JSON input from stdin, keep signed URLs private."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, sys, urllib.request
import fitz

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/Path(__file__).parent.name/'literature'
OUT.mkdir(parents=True,exist_ok=True)
items=json.load(sys.stdin)
def get(item):
    key=item['key']
    try:
        req=urllib.request.Request(item['url'],headers={'User-Agent':'AcademicResearch/1.0'})
        with urllib.request.urlopen(req,timeout=45) as f:
            blob=f.read()
        if not blob.startswith(b'%PDF'):
            raise ValueError('response is not PDF')
        p=OUT/(key+'.pdf');p.write_bytes(blob)
        doc=fitz.open(p)
        pages=[page.get_text() for page in doc]
        (OUT/(key+'.txt')).write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n{t}' for i,t in enumerate(pages)),encoding='utf-8')
        return dict(key=key,status='downloaded_and_extracted',pages=len(pages),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),public_source=item.get('public_source','Undermind OA PDF cache'),pdf_file=key+'.pdf',text_file=key+'.txt')
    except Exception as e:
        # Never log signed download URLs.
        return dict(key=key,status='failed',error=type(e).__name__,public_source=item.get('public_source','Undermind OA PDF cache'))
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(get,items))
manifest=OUT/'FULLTEXT_MANIFEST.json'
prior=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
bykey={x['key']:x for x in prior}
bykey.update({x['key']:x for x in results})
manifest.write_text(json.dumps(list(bykey.values()),indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(results,indent=2))
