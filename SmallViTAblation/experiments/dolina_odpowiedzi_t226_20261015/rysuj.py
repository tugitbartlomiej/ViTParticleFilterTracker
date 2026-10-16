"""Film 1080p i plansze: powierzchnia stałej odpowiedzi i dolina niezgodności.

Stylistyka i eksport jak w film_krysztal (Pillow + ffmpeg, ciemny obraz,
świecąca cząsteczka, stały punkt odniesienia i zsynchronizowane liczby).
python -X utf8 .../rysuj.py --stills  /  python -X utf8 .../rysuj.py
"""
from pathlib import Path
import argparse
import json
import math
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/dolina_odpowiedzi_t226_20261015'
D = np.load(OUT/'geometria.npz')
META = json.loads((OUT/'POMIAR.json').read_text(encoding='utf-8'))
W, H, SS, FPS, DURATION = 1920, 1080, 1.5, 30, 50
BG = (6, 11, 19)
CYAN, GOLD, ORANGE, PURPLE = (69, 222, 227), (255, 218, 112), (255, 141, 87), (185, 145, 251)
WHITE, MUTED = (231, 240, 248), (152, 174, 193)
FONT = Path('C:/Windows/Fonts/segoeui.ttf')
BOLD = Path('C:/Windows/Fonts/segoeuib.ttf')
FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(str(BOLD if bold else FONT), round(size*SS))
    return FONTS[key]


def fmt(v, digits=2):
    return f'{float(v):.{digits}f}'.replace('.', ',')


class Painter:
    def __init__(self):
        self.im = Image.new('RGB', (round(W*SS), round(H*SS)), BG)
        self.d = ImageDraw.Draw(self.im, 'RGBA')

    def text(self, xy, txt, size=24, color=WHITE, bold=False, anchor=None):
        self.d.text(tuple(np.array(xy)*SS), txt, fill=color, font=font(size,bold), anchor=anchor)

    def line(self, pts, color, width=1):
        self.d.line([tuple(p) for p in np.asarray(pts)*SS], fill=color, width=max(1,round(width*SS)), joint='curve')

    def rect(self, box, fill, outline=None, width=1, radius=0):
        b = tuple(np.array(box)*SS)
        if radius:
            self.d.rounded_rectangle(b, radius=round(radius*SS), fill=fill, outline=outline, width=max(1,round(width*SS)))
        else:
            self.d.rectangle(b, fill=fill, outline=outline, width=max(1,round(width*SS)))

    def poly(self, pts, fill):
        self.d.polygon([tuple(p) for p in np.asarray(pts)*SS], fill=fill)

    def dot(self, xy, radius, col=GOLD):
        x,y = xy
        for mult, alpha in ((3.8,10),(2.8,15),(2.,30),(1.4,70),(1.,255),(.45,255)):
            r=radius*mult
            c=col+(alpha,) if mult>.45 else (255,252,229,255)
            self.d.ellipse(tuple(np.array([x-r,y-r,x+r,y+r])*SS), fill=c)

    def arrow(self, a, b, col, width=3):
        a,b=np.array(a),np.array(b)
        self.line([a,b], col, width)
        dv=b-a; dv/=max(np.linalg.norm(dv),1e-12)
        perp=np.array([-dv[1],dv[0]])
        self.poly([b,b-15*dv+6*perp,b-15*dv-6*perp], col)

    def image(self, img, xy, size):
        img=img.resize(tuple(round(a*SS) for a in size), Image.Resampling.LANCZOS)
        self.im.paste(img, tuple(round(a*SS) for a in xy))

    def result(self):
        return self.im.resize((W,H),Image.Resampling.LANCZOS)


def projection(points, az=-48, elev=25, scale=104, center=(628,615), zcenter=.6):
    a,b=math.radians(az),math.radians(elev)
    right=np.array([math.cos(a),math.sin(a),0])
    up=np.array([-math.sin(a)*math.sin(b),math.cos(a)*math.sin(b),math.cos(b)])
    depth=np.cross(right,up)
    q=np.asarray(points,dtype=float).reshape(-1,3).copy(); q[:,2]-=zcenter
    pts=np.column_stack([center[0]+scale*(q@right),center[1]-scale*(q@up)])
    return pts, q@depth


def draw_surface(p, xyz, project, energy=False, alpha=190):
    h,w,_=xyz.shape
    flat=xyz.reshape(-1,3)
    xy,dep=project(flat)
    a=np.arange((h-1)* (w-1)).reshape(h-1,w-1)
    iy,ix=np.indices((h-1,w-1))
    idx=np.stack([iy*w+ix,iy*w+ix+1,(iy+1)*w+ix+1,(iy+1)*w+ix],-1).reshape(-1,4)
    vertices=flat[idx]
    normals=np.cross(vertices[:,1]-vertices[:,0],vertices[:,3]-vertices[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    light=np.abs(normals@np.array([-.2,-.4,.894]))
    order=np.argsort(dep[idx].mean(1))
    for k in order:
        zz=vertices[k,:,2].mean()
        if energy:
            v=min(zz/5,1)
            base=(1-v)*np.array(CYAN)+v*np.array(ORANGE)
        else:
            base=np.array(CYAN)
        rgb=np.clip(base*(.28+.45*light[k]),0,255).astype(int)
        p.poly(xy[idx[k]],tuple(rgb)+(alpha,))
    for row in range(0,h,4):
        pp,_=project(xyz[row]);p.line(pp,CYAN+(80,) if not energy else (135,196,213,80),1.2)
    for col in range(0,w,4):
        pp,_=project(xyz[:,col]);p.line(pp,CYAN+(80,) if not energy else (135,196,213,80),1.2)
    for edge in (xyz[0],xyz[-1],xyz[:,0],xyz[:,-1]):
        pp,_=project(edge);p.line(pp,CYAN+(210,) if not energy else (135,196,213,180),1.8)


def axes(p,project, R, mode='latent', grid=True):
    if grid:
        for u in np.linspace(-R,R,9):
            for line in ([[u,-R,-.12],[u,R,-.12]],[[-R,u,-.12],[R,u,-.12]]):
                xy,_=project(line);p.line(xy,(63,87,109,70),1)
    endpoints=[([R*.91,0,0],'d₁',CYAN),([0,R*.91,0],'d₂',CYAN),([0,0,3.2],'n',PURPLE)]
    if mode=='energy':
        endpoints=[([R*.94,0,0],'d₁',CYAN),([0,R*.94,0],'poprzecznie',ORANGE),([0,0,4.5],'E',WHITE)]
    origin=project([[0,0,0]])[0][0]
    for point,label,color in endpoints:
        xy=project([point])[0][0]
        p.arrow(origin,xy,color,2)
        p.text(xy+[10,-14],label,24,color,True)


def state(t):
    if t<6:
        return 'intro',np.array([0.,0.,0.]),D['target'],.995,1.,0.,0
    if t<18:
        i=round((t-6)/12*360)
        return 'sheet',D['path_xyz'][i],D['path_o'][i],D['path_p'][i],D['path_iou'][i],D['path_E'][i],i
    if t<29:
        s=12*math.sin(2*math.pi*(t-18)/11)
        i=int(np.argmin(abs(D['normal_s']-s)))
        return 'normal',np.array([0.,0.,D['normal_s'][i]]),D['normal_o'][i],D['normal_p'][i],D['normal_iou'][i],D['normal_E'][i],i
    if t<38:
        i=round((t-29)/9*360)
        return 'volume',D['path_xyz'][i],D['path_o'][i],D['path_p'][i],D['path_iou'][i],D['path_E'][i],i
    # W drugiej połowie ostatniej fazy cząsteczka wchodzi na zbocze E.
    tt=t-38
    u=3.*math.sin(2*math.pi*tt/12)
    ds=0 if tt<6 else -7.*math.sin(math.pi*(tt-6)/6)
    i=int(np.argmin(abs(D['energy_u']-u)));j=int(np.argmin(abs(D['energy_delta']-ds)))
    prob,iou,E=D['energy_metrics'][j,i]
    return 'energy',np.array([D['energy_u'][i],D['energy_delta'][j]/2.3,E*3]),D['energy_o'][j,i],prob,iou,E,(j,i)


TITLES={
    'intro':('Jedna odpowiedź. Wiele położeń cząsteczki.','Rzeczywista głowa detektora · token 226 · zamrożone wagi z kroku 65 640'),
    'sheet':('Po dolinie: cząsteczka się przesuwa, odpowiedź pozostaje ta sama.','Pewność 99,5% i ta sama ramka · zakrzywiony wycinek zbioru równoważnych położeń'),
    'normal':('W poprzek doliny: zmieniają się pewność i ramka.','Fioletowa oś n to jedna z pięciu kombinacji zmian, na które reaguje głowa.'),
    'volume':('Dobra odpowiedź zajmuje obszar wokół powierzchni.','Złote próbki spełniają oba warunki: pewność ≥ 99% oraz IoU ≥ 0,90.'),
    'energy':('Tak wygląda dolina niezgodności odpowiedzi.','Tutaj wysokość oznacza błąd E. Na błękitnym dnie odpowiedź jest zgodna z wybranym celem.')}


def right_panel(p,o,prob,iou,E,phase):
    p.text((1270,161),'Odpowiedź tej samej głowy',27,WHITE,True)
    p.text((1270,199),'Zbliżenie oryginalnego zdjęcia',18,MUTED)
    crop=np.array([150,75,380,180])
    p.image(Image.fromarray(D['rgb']).crop(tuple(crop)),(1270,235),(588,268))
    sx,sy=588/(crop[2]-crop[0]),268/(crop[3]-crop[1])
    gt=D['gt']; box=np.exp(o[1:])*16
    cx,cy=(226%30+.5)*16,(226//30+.5)*16
    bb=np.array([cx-box[0],cy-box[1],cx+box[2],cy+box[3]])
    def bb2(b):
        return [1270+(b[0]-crop[0])*sx,235+(b[1]-crop[1])*sy,1270+(b[2]-crop[0])*sx,235+(b[3]-crop[1])*sy]
    p.rect(bb2(gt),None,(246,248,253,230),3)
    col=CYAN if prob>=.99 and iou>=.9 else ORANGE
    p.rect(bb2(bb),None,col,2)
    xy=[1270+(cx-crop[0])*sx,235+(cy-crop[1])*sy]
    p.dot(xy,3,GOLD)
    p.line([(1280,524),(1310,524)],WHITE,3);p.text((1320,510),'ramka oznaczona',18,MUTED)
    p.line([(1560,524),(1590,524)],col,3);p.text((1600,510),'ramka detektora',18,MUTED)
    p.text((1270,560),'PEWNOŚĆ',18,MUTED,True)
    p.text((1578,560),'IoU Z OZNACZENIEM',18,MUTED,True)
    p.text((1270,584),fmt(100*prob,2)+'%',52,CYAN if prob>=.99 else ORANGE,True)
    p.text((1578,584),fmt(iou,4),52,CYAN if iou>=.9 else ORANGE,True)
    for x,v,limit in ((1270,prob,.99),(1578,iou,.9)):
        p.rect((x,656,x+260,663),(44,60,76),radius=3)
        p.rect((x,656,x+260*v,663),CYAN if v>=limit else ORANGE,radius=3)
        p.line([(x+260*limit,650),(x+260*limit,669)],WHITE,2)
    p.text((1270,690),'ODLEGŁOŚCI DO KRAWĘDZI · kratki',18,MUTED,True)
    for k,label in enumerate(('lewa','góra','prawa','dół')):
        x=1270+k*146
        p.text((x,721),label,18,MUTED)
        p.text((x,747),fmt(np.exp(o[k+1]),3),27,WHITE,True)
    p.text((1270,799),'Niezgodność E = '+fmt(E,5),22,WHITE)
    p.text((1270,840),'Po treningu: 94,95% · IoU 0,8858',18,MUTED)


def frame(t):
    phase,point,o,prob,iou,E,idx=state(t)
    p=Painter()
    p.text((44,25),TITLES[phase][0],37,WHITE,True)
    p.text((44,80),TITLES[phase][1],23,CYAN if phase in ('sheet','intro') else GOLD if phase=='volume' else MUTED)
    p.rect((36,143,1220,884),(11,20,31),(38,57,74),radius=12)
    p.rect((1246,143,1884,884),(11,20,31),(38,57,74),radius=12)
    R=float(D['uv'][-1])
    # Gęsta siatka E jest zredukowana wyłącznie podczas rysowania.
    if phase=='energy':
        p.text((60,165),'KRAJOBRAZ ODPOWIEDZI',23,WHITE,True)
        p.text((60,203),'Wysokość = E · dwie osie ruchu, jedna oś błędu',20,MUTED)
        uu,dd=np.meshgrid(D['energy_u'][::2],D['energy_delta'][::2])
        xyz=np.stack([uu,dd/2.3,D['energy_metrics'][::2,::2,2]*3],-1)
        project=lambda pp:projection(pp,az=-48+2*math.sin(t/6),elev=25,scale=93,center=(625,656),zcenter=.5)
        axes(p,project,R,'energy')
        draw_surface(p,xyz,project,True)
        axes(p,project,R,'energy',False)
        # Prawdziwe dno, E=0 (do dokładności numerycznej).
        line=np.column_stack([D['energy_u'],np.zeros(len(D['energy_u'])),np.zeros(len(D['energy_u']))])
        p.line(project(line)[0],CYAN,5)
        p.dot(project([point])[0][0],9,GOLD)
        p.text((60,817),'Wzdłuż dna: 187 swobodnych kierunków lokalnie.',21,CYAN)
        p.text((60,848),'E jest miarą niezgodności; to nie energia pola laserów ani strata treningowa.',18,MUTED)
    else:
        p.text((60,165),'POŁOŻENIA W LATENCIE',23,WHITE,True)
        p.text((60,203),'Wysokość = rzut na kierunek n · wszystkie osie w jednostkach stanu tokenu',20,MUTED)
        scale=102 if phase in ('intro','sheet') else 38 if phase=='normal' else 53
        center=(628,610) if phase in ('intro','sheet') else (628,525) if phase=='normal' else (628,645)
        az=-48+5*math.sin(t/10)
        zcenter=.4+point[2]*.45 if phase=='normal' else .4
        project=lambda pp:projection(pp,az=az,elev=25,scale=scale,center=center,zcenter=zcenter)
        axes(p,project,R)
        if phase=='volume':
            xyz=D['volume_xyz'].reshape(-1,3)
            m=D['volume_metrics'].reshape(-1,3)
            mask=(m[:,0]>=.99)&(m[:,1]>=.9)
            inds=np.flatnonzero(mask)[::7]
            xy,depth=project(xyz[inds])
            for j in np.argsort(depth):
                x,y=xy[j]
                if 55<x<1200 and 245<y<805:
                    p.rect((x-1.5,y-1.5,x+1.5,y+1.5),GOLD+(55,))
        draw_surface(p,D['sheet_xyz'],project)
        axes(p,project,R,grid=False)
        # Dwie proste styczne w centrum. Poza centrum nie są ścieżkami stałej odpowiedzi.
        if phase in ('intro','sheet'):
            for axis in (0,1):
                ends=np.zeros((2,3));ends[:,axis]=[-R*.9,R*.9]
                p.line(project(ends)[0],CYAN+(120,),1.5)
            trail=D['path_xyz'][:idx+1] if phase=='sheet' else D['path_xyz']
            p.line(project(trail)[0],GOLD+(170 if phase=='sheet' else 55,),2.5)
        if phase=='normal':
            p.line(project([[0,0,min(-.5,point[2])],[0,0,max(3,point[2])]])[0],PURPLE+(130,),2)
            p.arrow(project([[0,0,0]])[0][0],project([point])[0][0],ORANGE,3)
        p.dot(project([point])[0][0],9,GOLD)
        p.text((60,817),'Błękitna powierzchnia: p = 99,5% · ta sama ramka · IoU = 1,0000',22,CYAN,True)
        p.text((60,850),'Pokazujemy dwa spośród 187 kierunków stycznych i jedną normalną. To wycinek 192D.',18,MUTED)
    right_panel(p,o,prob,iou,E,phase)
    # Pełna liczba zweryfikowanych niezależnych kierunków, bez udawania ich rzutu na 3D.
    p.rect((36,904,1884,1036),(11,20,31),(38,57,74),radius=12)
    p.text((57,917),'187',62,CYAN,True)
    p.text((60,992),'kierunków stycznych',18,MUTED)
    for i in range(187):
        x=240+(i%17)*16;y=922+(i//17)*8.8
        p.rect((x,y,x+11,y+5.3),CYAN+(220 if i<2 else 75,))
    p.text((550,919),'Każdy kafelek = jeden niezależny kierunek bazy jądra jakobianu.',22,WHITE)
    p.text((550,953),'Na powierzchni utrzymujemy pięć liczb wyjścia, korygując krzywiznę.',21,MUTED)
    p.text((550,986),'Weryfikacja w 192D: maksymalny błąd odpowiedzi < 10⁻¹².',21,CYAN)
    p.text((1550,918),'5',57,ORANGE,True)
    p.text((1610,931),'kierunków',23,ORANGE,True)
    p.text((1610,964),'odczytu głowy',22,MUTED)
    for i in range(5):
        p.rect((1558+i*25,1004,1575+i*25,1014),ORANGE+(190,))
    p.text((43,1052),'MalyViT s47 · checkpoint 65 640 · token 226 · stan po ostatnim bloku, przed LayerNorm · cel p=0,995 wybrany do wizualizacji.',17,MUTED)
    # Dyskretny pasek faz zamiast osi uczenia: podczas filmu wagi się nie zmieniają.
    for i,(start,end) in enumerate(((0,6),(6,18),(18,29),(29,38),(38,50))):
        x0=36+1848*start/DURATION;x1=36+1848*end/DURATION
        p.rect((x0,888,x1-3,892),CYAN if t>=end else GOLD if t>=start else (46,63,79))
    return p.result()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stills',action='store_true');args=ap.parse_args()
    for t,label in ((3,'01_powierzchnia'),(12,'02_ruch_po_dolinie'),(20.75,'03_w_poprzek'),(33,'04_obszar_jakosci'),(47,'05_krajobraz_bledu')):
        frame(t).save(OUT/f'{label}.png')
    if args.stills:
        print('Zapisano 5 plansz',flush=True);return
    fn=OUT/'dolina_odpowiedzi_t226.mp4'
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
         '-an','-c:v','h264_nvenc','-preset','p5','-cq','19','-pix_fmt','yuv420p','-movflags','+faststart',str(fn)]
    t0=time.time()
    with (OUT/'render_ffmpeg.log').open('w',encoding='utf-8') as err:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
        try:
            for i in range(FPS*DURATION):
                proc.stdin.write(frame(i/FPS).tobytes())
                if i%(FPS*5)==0:print(f'klatka {i}/{FPS*DURATION}; {time.time()-t0:.1f}s',flush=True)
        finally:
            proc.stdin.close()
            code=proc.wait()
        assert code==0, (OUT/'render_ffmpeg.log').read_text(encoding='utf-8')
    print('Zapisano',fn,'w',round(time.time()-t0,1),'s',flush=True)


if __name__=='__main__':
    main()
