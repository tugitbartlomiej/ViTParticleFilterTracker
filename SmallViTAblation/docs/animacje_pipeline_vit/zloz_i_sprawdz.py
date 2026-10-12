"""Montaż rozdziałów, napisy, rozdziały MP4, indeks HTML i kontrola plików."""
from pathlib import Path
import json
import subprocess
import html
import hashlib
import argparse
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
V=HERE/'videos'
C=json.loads((HERE/'storyboard.json').read_text(encoding='utf-8'))

def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True,encoding='utf-8',errors='replace').stdout

def probe(p):
    return json.loads(run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(p)]))

def srt_time(t):
    ms=round(t*1000); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

def srt(entries,p):
    # Napisy dzielimy na krótkie zdania, bez nakładania segmentów.
    chunks=[]
    for e in entries:
        sentences=e['text'].replace('?', '?.').split('. ')
        sentences=[s.strip().rstrip('.')+'.' for s in sentences if s.strip()]
        weight=sum(len(s) for s in sentences)
        begin=e['start']+.1
        for sentence in sentences:
            duration=(e['end']-e['start']-.7)*len(sentence)/max(weight,1)
            chunks.append((begin,begin+duration,sentence));begin+=duration
    p.write_text('\n\n'.join(f'{i+1}\n{srt_time(a)} --> {srt_time(b)}\n{t}' for i,(a,b,t) in enumerate(chunks))+'\n',encoding='utf-8')
    p.with_suffix('.vtt').write_text('WEBVTT\n\n'+'\n\n'.join(
        f'{srt_time(a).replace(",", ".")} --> {srt_time(b).replace(",", ".")}\n{t}'
        for a,b,t in chunks)+'\n',encoding='utf-8')

def review_frames(folder=V,quick=False):
    out=HERE/('kontrola_proba' if quick else 'kontrola');out.mkdir(exist_ok=True)
    frames=[]
    for c in C:
        video=folder/f"Rozdzial{c['id']:02d}.mp4"
        if not video.exists():continue
        timeline=json.loads((HERE/'timelines'/f"{c['slug']}.json").read_text(encoding='utf-8'))
        duration=float(probe(video)['format']['duration'])
        for i,e in enumerate(timeline):
            t=min(duration-.1,e['start']+.68*(e['end']-e['start']))
            dest=out/f"{c['id']:02d}_{i:02d}.jpg"
            run(['ffmpeg','-y','-v','error','-ss',str(t),'-i',str(video),'-frames:v','1','-vf','scale=640:360',str(dest)])
            frames.append((dest,f"{c['id']:02d}.{i+1} {e['title']}"))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    sheets=[]
    for start in range(0,len(frames),12):
        page=frames[start:start+12];im=Image.new('RGB',(1280,390*((len(page)+1)//2)),(11,18,32));draw=ImageDraw.Draw(im)
        for j,(p,title) in enumerate(page):
            x=(j%2)*640;y=(j//2)*390
            im.paste(Image.open(p),(x,y));draw.text((x+8,y+365),title[:70],fill='white',font=font)
        dest=out/f'plansza_{start//12+1:02d}.jpg';im.save(dest,quality=90);sheets.append(str(dest))
    return sheets

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review-only',action='store_true');ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    if args.review_only:
        print(json.dumps(review_frames(V/'proba' if args.quick else V,args.quick),ensure_ascii=False));return
    items=[];all_entries=[];offset=0.;meta=[';FFMETADATA1','title=Cząstki ViT, tożsamość, Koopman i Langevin','artist=Materiały projektu ViTParticleFilterTracker']
    for c in C:
        p=V/f"Rozdzial{c['id']:02d}.mp4";info=probe(p)
        video=next(s for s in info['streams'] if s['codec_type']=='video')
        audios=[s for s in info['streams'] if s['codec_type']=='audio']
        assert (video['width'],video['height'])==(1920,1080),p
        assert audios, f'Brak lektora: {p}'
        duration=float(info['format']['duration'])
        entries=json.loads((HERE/'timelines'/f"{c['slug']}.json").read_text(encoding='utf-8'))
        assert len(entries)==len(c['slides'])
        assert abs(entries[-1]['end']-duration)<.3,(p,entries[-1]['end'],duration)
        # Dekodowanie całego filmu: wykrywa uszkodzone strumienie i błędy kontenera.
        run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'])
        srt(entries,p.with_suffix('.srt'))
        for e in entries:all_entries.append({**e,'start':e['start']+offset,'end':e['end']+offset})
        meta+=['[CHAPTER]','TIMEBASE=1/1000',f'START={round(offset*1000)}',f'END={round((offset+duration)*1000)}',f"title={c['id']:02d}. {c['title']}"]
        items.append(dict(id=c['id'],title=c['title'],file=p.name,duration=duration,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),start=offset))
        offset+=duration
    (HERE/'chapters.ffmetadata').write_text('\n'.join(meta)+'\n',encoding='utf-8')
    (HERE/'concat.txt').write_text('\n'.join("file '"+str(V/i['file']).replace('\\','/')+"'" for i in items),encoding='utf-8')
    total=V/'CALOSC_ViT_Tozsamosc_Koopman_Langevin.mp4'
    subtitles=total.with_suffix('.srt');srt(all_entries,subtitles)
    run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(HERE/'concat.txt'),'-i',str(HERE/'chapters.ffmetadata'),'-i',str(subtitles),
         '-map','0:v:0','-map','0:a:0','-map','2:0','-map_metadata','1','-map_chapters','1','-c','copy','-c:s','mov_text','-metadata:s:s:0','language=pol','-movflags','+faststart',str(total)])
    check=probe(total)
    assert abs(float(check['format']['duration'])-offset)<1.
    page=['<!doctype html><html lang="pl"><meta charset="utf-8"><title>ViT: cząstki i tożsamość</title>',
          '<style>body{background:#0b1220;color:#edf4ff;font:18px system-ui;max-width:1100px;margin:35px auto;padding:20px}a{color:#50cef5}video{width:100%;background:black;border-radius:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.card{background:#152236;padding:20px;border-radius:14px}h1{font-size:38px}small{color:#9aadc7}li{margin:12px 0}</style>',
          '<h1>Cząstki ViT, tożsamość, Koopman i Langevin</h1>',
          f'<p>12 rozdziałów • {offset/60:.1f} min • Full HD • polski lektor • 51 ujęć</p>',
          '<p>Pomiar jest oznaczony osobno od schematów. Animacja między dwoma checkpointami jest interpolacją. Kierunki ruchu są kandydatami, nie dowodem przyczynowej tożsamości.</p>',
          f'<video id="calosc" controls preload="metadata" src="videos/{total.name}"><track kind="subtitles" src="videos/{total.stem}.vtt" srclang="pl" label="Polski"></video>',
          '<p><a href="SCENARIUSZ.md">Scenariusz i pełny tekst</a> · <a href="ZRODLA.json">Źródła</a> · <a href="videos/'+subtitles.name+'">Napisy SRT</a></p>',
          '<h2>Rozdziały</h2><div class="grid">']
    for i in items:
        page += [f'<section class="card"><h3>{i["id"]:02d}. {html.escape(i["title"])}</h3><small>{i["duration"]:.0f} sekund</small>',
                 f'<video controls preload="none" src="videos/{i["file"]}"><track kind="subtitles" src="videos/{Path(i["file"]).stem}.vtt" srclang="pl" label="Polski"></video>',
                 f'<p><a href="videos/{i["file"]}">Otwórz film</a> · <a href="videos/{Path(i["file"]).stem}.srt">Napisy</a></p></section>']
    page+=['</div></html>'];(HERE/'index.html').write_text('\n'.join(page),encoding='utf-8')
    report=dict(chapters=items,total_seconds=offset,total_file=str(total),total_bytes=total.stat().st_size,
                resolution=[1920,1080],fps=24,voice='Microsoft Paulina Desktop pl-PL',renderer='ManimGL 1.7.2',
                decode_validation='all 12 full chapter streams decoded without errors',scientific_status='educational; measured and illustrative scenes distinguished')
    (HERE/'MANIFEST_FILMOW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    sheets=review_frames()
    print(json.dumps({'minutes':offset/60,'total':str(total),'bytes':total.stat().st_size,'review':sheets},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
