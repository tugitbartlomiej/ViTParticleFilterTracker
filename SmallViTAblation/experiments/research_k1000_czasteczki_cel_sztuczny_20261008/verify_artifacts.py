from pathlib import Path
import hashlib, json, re
import fitz
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'/Path(__file__).parent.name
LIT=OUT/'literature'
manifest=json.loads((LIT/'FULLTEXT_MANIFEST.json').read_text(encoding='utf-8-sig'))
checks=[]
for entry in manifest:
    pdf=LIT/entry['pdf_file']
    txt=LIT/entry['text_file']
    with fitz.open(pdf) as doc:
        good=(doc.page_count==entry['pages'] and
              hashlib.sha256(pdf.read_bytes()).hexdigest()==entry['sha256'] and
              len(txt.read_text(encoding='utf-8'))>1000)
        checks.append({'file':pdf.name,'pages':doc.page_count,'valid':good})
assert len(checks)==24 and all(x['valid'] for x in checks)
for fn, terms in {
    'AdamW_1711.05101v3.pdf':['Algorithm 2','Proposition 2','Proposition 3'],
    'Hao20_undermind.pdf':['(7)','(8)','(9)','(10)'],
    'Mis26b_undermind.pdf':['Theorem 3.1','Theorem 4.2','Theorem 4.4','Theorem 4.5'],
    'Mak23_undermind.pdf':['3.1','Table 1'],
    'Chu22c_undermind.pdf':['Manifold','Posterior Sampling','C.6'],
    'Bae22b_undermind.pdf':['(8)','PBRF'],
    'Liu26p_undermind.pdf':['Theorem 3.1','(1)','(2)'],
    'Den26b_undermind.pdf':['AdamW','(7)'],
}.items():
    with fitz.open(LIT/fn) as doc:
        pages=[p.get_text() for p in doc]
    print(fn, json.dumps({term:[i+1 for i,t in enumerate(pages) if term.casefold() in t.casefold()] for term in terms}))
report=(OUT/'RAPORT.md').read_text(encoding='utf-8')
refs=sorted(set(int(i) for i in re.findall(r'\[R(\d+)\]\(https?://',report)))
assert refs==list(range(1,25)),refs
audit=json.loads((OUT/'evidence/CPU_AUDIT.json').read_text(encoding='utf-8-sig'))
copy_checks=[]
for entry in audit['evidence']:
    saved=OUT/'evidence'/entry['path'].replace('/','__')
    copy_checks.append({'path':entry['path'],'saved_hash_valid':hashlib.sha256(saved.read_bytes()).hexdigest()==entry['sha256'],
        'live_matches_snapshot':hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()==entry['sha256']})
assert all(x['saved_hash_valid'] for x in copy_checks)
qa={'pdf_count_distinct_references':24,'pdf_checks':checks,'report_reference_ids':refs,'snapshot_checks':copy_checks,
    'no_training':audit['no_training'],'no_image_forward':audit['no_image_forward'],
    'scope':'research documents and CPU probes only', 'cpu_audit_checkpoint_sha256':audit['checkpoint_sha256']}
(OUT/'QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print('QA: 24 pełne teksty, 24 cytowania, migawki źródeł i hashe prawidłowe.')
