"""Czytelny raport z już wykonanych etapów; nie uruchamia modeli."""
from __future__ import annotations
import argparse
from pathlib import Path
from dane import read_json

def make_report(root):
    root=Path(root);result=read_json(root/'REPORT.json')
    lines=['# Wynik pilota Koopman–FastGAN','',f"Stan: {result['state']}.",
           f"Powód zakończenia: {result.get('stop_reason','test techniczny')}.",'',
           'Wygenerowane obrazy służą wyszukiwaniu. Do treningu trafiają wyłącznie oznaczone, rzeczywiste klatki.','']
    for rd in sorted(root.glob('round_*')):
        lines.extend([f'## {rd.name}',''])
        forecast=read_json(rd/'FORECAST.json')
        lines.extend([forecast['reason'],'', '| Prognoza | Błąd |','|---|---:|'])
        for name,score in forecast['scores'].items():lines.append(f"| {name} | {score['mean_vector_error']:.6g} |")
        lines.append('')
        from PIL import Image,ImageDraw,ImageFont
        try:font=ImageFont.truetype('DejaVuSans.ttf',16)
        except OSError:font=ImageFont.load_default()
        for gen in sorted(rd.glob('generation_*')):
            if not gen.is_dir() or not (gen/'report.json').exists():continue
            panels=[]
            for name,path in [('Kotwica',gen/'anchor.png'),('Cel Koopmana',gen/'target'/'best.png'),
                              ('Cel obecny',gen/'current'/'best.png'),('Losowy przyrost',gen/'random_norm'/'best.png')]:
                if path.exists():
                    with Image.open(path) as im:panels.append((name,im.convert('RGB').resize((256,256))))
            q=gen.name.removeprefix('generation_');retrieval=rd/f'retrieval_{q}.json'
            if retrieval.exists():
                for i,row in enumerate(read_json(retrieval)['candidates'][:4]):
                    path=Path(row['path'])
                    if path.is_file():
                        with Image.open(path) as im:panels.append((f"Realny {i+1}, cos={row['score']:.3f}",im.convert('RGB').crop(row['crop_xyxy']).resize((256,256))))
            if panels:
                canvas=Image.new('RGB',(4*256,((len(panels)+3)//4)*288),(240,240,240));draw=ImageDraw.Draw(canvas)
                for j,(label,im) in enumerate(panels):
                    x=(j%4)*256;y=(j//4)*288;canvas.paste(im,(x,y+32));draw.text((x+5,y+6),label,font=font,fill='black')
                canvas.save(gen/'PODGLAD.png')
                lines.extend([f'![{gen.name}]({rd.name}/{gen.name}/PODGLAD.png)',''])
        if (rd/'REPORT.json').exists():
            r=read_json(rd/'REPORT.json');lines.extend(['| Gałąź | mAP walidacji |','|---|---:|'])
            for name,value in r['validation_map'].items():lines.append(f'| {name} | {value:.6g} |')
            lines.extend(['',f"Dalsza gałąź: {r['chosen_next']}.",''])
    lines.extend(['## Ograniczenia','', 'Jeden seed i mały budżet. Trafna prognoza latentu nie dowodzi użyteczności klatki.',
                  'Dopasowanie maski generatora nie jest dowodem poprawnego kształtu narzędzia w RGB.'])
    (root/'RAPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();make_report(a.run)
