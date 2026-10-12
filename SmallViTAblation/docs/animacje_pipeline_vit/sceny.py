"""Dwanaście rozdziałów ManimGL. Dane, ilustracje i ograniczenia są oznaczone.
Każde ujęcie ma polski głos i podpis. Brak treningu sieci w tym rendererze.
"""
from pathlib import Path
import json
import textwrap
import wave
import os
import re
import numpy as np
from manimlib import *

HERE=Path(__file__).resolve().parent
STORY=json.loads((HERE/'storyboard.json').read_text(encoding='utf-8'))
D=json.loads((HERE/'assets/dane.json').read_text(encoding='utf-8'))
BG='#0b1220'; FG='#edf4ff'; MUTED='#9aadc7'; CYAN='#50cef5'; GREEN='#50e0a6'; ORANGE='#ffbd66'; PURPLE='#b298ff'; RED='#ff7187'

def text(s, size=25, color=FG, width=None):
    # Arial/Pango does not reliably render these Unicode indices on this host.
    indices=str.maketrans('₀₁₂₃₄₅₆₇₈₉ₖₙ', '0123456789kn')
    display=re.sub('[₀₁₂₃₄₅₆₇₈₉ₖₙ]+',lambda match:'_'+match.group().translate(indices),str(s))
    display=display.replace('⁻¹','^(-1)').replace('∝','proporcjonalny do')
    m=Text(display,font='Arial',font_size=size,fill_color=color)
    if width and m.get_width()>width: m.set_width(width)
    return m

def label(s, pos, size=24, color=FG, width=None):
    return text(s,size,color,width).move_to(np.array(pos,float))

def card(s,pos,w=2.5,h=.85,color=CYAN,size=23):
    r=RoundedRectangle(width=w,height=h,corner_radius=.12).set_stroke(color,1.7).set_fill(color,.09).move_to(pos)
    lines=str(s).split('\n')
    words=VGroup(*[text(t,size,color,w-.25) for t in lines]).arrange(DOWN,buff=.1).move_to(pos)
    return VGroup(r,words)

def arrow(a,b,color=CYAN,width=2):
    return Arrow(np.array(a,float),np.array(b,float),buff=.10,stroke_width=width,fill_color=color).set_color(color)

def poly(points,color=CYAN,width=3):
    return VMobject().set_points_as_corners(np.asarray(points,float)).set_stroke(color,width)

def note(lines,x=4,y=1.7,color=FG):
    return VGroup(*[label(t,[x,y-i*.75,0],27,color,3.8) for i,t in enumerate(lines)])

class PipelineFilm(Scene):
    chapter_index=0

    def construct(self):
        c=STORY[self.chapter_index]
        slides=c['slides']
        if os.getenv('VIT_FILM_SMOKE'): slides=slides[:1]
        self.timeline=[]
        for index,s in enumerate(slides):
            start=self.get_time()
            old=[m for m in self.mobjects if m is not self.camera.frame]
            if old:
                self.play(*[FadeOut(m) for m in old],run_time=.3)
            heading=label(s['title'],[0,3.35,0],36,FG,12.4)
            chapter=label(f"{c['id']:02d} / 12   •   {c['title']}",[-1.4,3.85,0],14,MUTED,10.7)
            mode=label(s['mode'],[0,2.78,0],14,ORANGE if 'POMIAR' in s['mode'] or 'DANE' in s['mode'] else MUTED,12)
            caplines=textwrap.wrap(s['caption'],width=91,break_long_words=False)
            caption=VGroup(*[text(t,24,FG,12.5) for t in caplines]).arrange(DOWN,buff=.10).move_to([0,-2.98,0])
            footer=label('ViTParticleFilterTracker  •  kod i źródła w ZRODLA.json  •  01.10.2026',[0,-3.78,0],12,MUTED)
            self.add(heading,chapter,mode,caption,footer)
            wav=HERE/'audio'/f"{c['id']:02d}_{index:02d}.wav"
            with wave.open(str(wav)) as wf: length=wf.getnframes()/wf.getframerate()
            quick=bool(os.getenv('VIT_FILM_CHECK'))
            if quick: length=1.3
            else: self.add_sound(str(wav),time_offset=.1)
            visual,anim=self.diagram(s['kind'])
            self.play(FadeIn(visual),run_time=.7)
            if anim: self.play(*anim,run_time=.5 if quick else min(7.0,max(3.0,length*.43)),rate_func=smooth)
            remaining=length+1.0-(self.get_time()-start)
            if remaining>0: self.wait(remaining)
            self.timeline.append(dict(start=start,end=self.get_time(),title=s['title'],text=s['narration']))
        (HERE/'timelines').mkdir(exist_ok=True)
        (HERE/'timelines'/f"{c['slug']}.json").write_text(json.dumps(self.timeline,ensure_ascii=False,indent=2),encoding='utf-8')

    def diagram(self,k):
        if k in ('frame','grid','mask','mask_compare','mask_weights'): return self.image_diagram(k)
        if k=='one_pixel': return self.one_pixel()
        if k in ('cloud','trajectories'): return self.cloud(k)
        if k in ('history','averaging'): return self.history(k)
        if k=='rollout': return self.forecast()
        if k=='spectrum': return self.spectrum()
        if k=='results': return self.results()
        if k in ('directions','subspace','lda','score'): return self.projection(k)
        if k in ('attention','position','dimensions','architecture'): return self.network(k)
        if k in ('aliasing','memory'): return self.oscillation(k)
        if k in ('langevin','forces','full_latent','optimizer','explanation','no_leak'): return self.equations(k)
        return self.flow(k)

    def image_diagram(self,k):
        img=ImageMobject(str(HERE/'assets/klatka_00.png')).set_width(6.5).move_to([-2.8,.15,0])
        elems=[img]
        w,h=6.5,6.5*2/3
        x0,y0=-2.8-w/2,.15+h/2
        cell=w/30
        grid=VGroup(*[Line([x0+i*cell,y0,0],[x0+i*cell,y0-h,0]).set_stroke(CYAN,.6,.48) for i in range(31)],
                    *[Line([x0,y0-i*cell,0],[x0+w,y0-i*cell,0]).set_stroke(CYAN,.6,.48) for i in range(21)])
        if k!='frame': elems.append(grid)
        selected=VGroup()
        for j,yes in enumerate(D['tool']):
            if not yes: continue
            col=GREEN if D['tool25'][j] else ORANGE
            r=Rectangle(width=cell,height=cell).set_fill(col,.35).set_stroke(col,1.8)
            r.move_to([x0+(j%30+.5)*cell,y0-(j//30+.5)*cell,0]); selected.add(r)
        if k in ('mask','mask_compare','mask_weights'): elems.append(selected)
        if k=='frame': lines=['480 × 320 px','RGB','Pełna klatka','Cataract-LMM']
        elif k=='grid': lines=['30 × 20','600 tokenów','Krok siatki: 16 px','Pola widzenia się nakładają']
        elif k=='mask': lines=['Obrys cystotomu','Piksele maski','Przypisanie do siatki','Zachowujemy 600 tokenów']
        elif k=='mask_compare': lines=['Ten obraz: 20 tokenów','Stary próg: 9 tokenów','Nowy próg: > 0','Dodatkowe pola przy brzegu']
        else: lines=['1 piksel z 256','25% = 64 piksele','Pokrycie jest zapisane','Tło: z dala od maski']
        elems.append(note(lines,x=4.0,y=1.6))
        focus=Rectangle(width=cell,height=cell).set_stroke(WHITE,3).move_to([x0+15.5*cell,y0-10.5*cell,0])
        elems.append(focus)
        return Group(*elems),[focus.animate.move_to([x0+23.5*cell,y0-10.5*cell,0])]

    def one_pixel(self):
        left=-4.9; top=2.1; d=.25
        cells=VGroup(*[Square(side_length=d).set_stroke(MUTED,.4).set_fill('#152336',1).move_to([left+(j%16)*d,top-(j//16)*d,0]) for j in range(256)])
        one=cells[136]; one.set_fill(GREEN,1)
        count=note(['Pole: 16 × 16','256 pikseli','Już 1 piksel maski','Token zostaje wybrany'],x=3.4,y=1.7)
        a=arrow([-.4,.25,0],[1.0,.25,0],GREEN)
        return VGroup(cells,count,a),[one.animate.scale(1.8).set_fill(ORANGE,1)]

    def cloud(self,k):
        coords=np.asarray(D['coords'])
        p=np.column_stack([coords[0,:,0]*5.1,coords[0,:,1]*2.0,np.zeros(600)])
        q=np.column_stack([coords[1,:,0]*5.1,coords[1,:,1]*2.0,np.zeros(600)])
        mask=np.asarray(D['tool'],bool)
        background=DotCloud(p[~mask],color=MUTED,radius=.028,opacity=.45)
        tool=DotCloud(p[mask],color=GREEN,radius=.065)
        axes=VGroup(Line([-5.5,0,0],[5.5,0,0]).set_stroke(MUTED,.8),Line([0,-2.2,0],[0,2.2,0]).set_stroke(MUTED,.8),
                    label('rzut 1',[5.4,-.3,0],17,MUTED),label('rzut 2',[.55,2.1,0],17,MUTED))
        legend=label('600 tokenów • zielone pola przecinają maskę • tylko 2 z 768 współrzędnych',[0,-2.3,0],18,MUTED,12)
        g=Group(axes,background,tool,legend)
        if k=='trajectories':
            lines=VGroup(*[Line(p[j],q[j]).set_stroke(GREEN,1,.5) for j in np.flatnonzero(mask)])
            g.add(lines)
            return g,[tool.animate.set_points(q[mask]),background.animate.set_points(q[~mask])]
        return g,[tool.animate.set_radius(.10)]

    def chart_axes(self,xmax,ymin,ymax):
        def point(x,y): return np.array([-5.6+11.2*x/xmax,-1.7+3.65*(y-ymin)/max(ymax-ymin,1e-9),0])
        axes=VGroup(Line(point(0,ymin),point(xmax,ymin)).set_stroke(MUTED,1),Line(point(0,ymin),point(0,ymax)).set_stroke(MUTED,1))
        for t in np.linspace(0,xmax,5): axes.add(label(f'{t:g}',point(t,ymin)+[0,-.28,0],16,MUTED))
        for v in np.linspace(ymin,ymax,4): axes.add(label(f'{v:.2f}',point(0,v)+[-.5,0,0],15,MUTED))
        return axes,point

    def history(self,k):
        vals=np.asarray(D['raw_history'])[:101]
        axes,p=self.chart_axes(1000,float(vals.min()-.3),float(vals.max()+.3))
        line=poly([p(i*10,v) for i,v in enumerate(vals)],CYAN,2)
        dots=DotCloud(np.array([p(i*10,v) for i,v in enumerate(vals)]),color=CYAN,radius=.026)
        g=Group(axes,line,dots,label('aktualizacje wag',[0,-2.25,0],18,MUTED))
        if k=='averaging':
            blocks=np.asarray(D['averaged_history'])
            avg=poly([p((i+1)*50,v) for i,v in enumerate(blocks)],ORANGE,3)
            g.add(label('surowe: niebieskie    średnie: pomarańczowe',[0,2.28,0],20,ORANGE))
            g.add(avg)
            return g,[ShowCreation(avg)]
        cursor=Dot(p(0,vals[0]),color=ORANGE,radius=.07);g.add(cursor)
        return g,[MoveAlongPath(cursor,line)]

    def spectrum(self):
        vals=D['spectrum']; axes,p=self.chart_axes(64,0,1)
        curve=poly([p(i+1,v) for i,v in enumerate(vals)],CYAN)
        threshold=Line(p(0,.9),p(64,.9)).set_stroke(ORANGE,1)
        g=VGroup(axes,curve,threshold,label('90% energii',[4.4,1.95,0],20,ORANGE),label('Liczba kierunków',[0,-2.25,0],20,MUTED))
        return g,[ShowCreation(curve)]

    def forecast(self):
        history=np.asarray(D['raw_history'])
        allvalues=np.r_[history,*[v['values'] for v in D['forecast'].values()]]
        axes,p=self.chart_axes(1500,float(allvalues.min()-.3),float(allvalues.max()+.3))
        actual=poly([p(i*10,v) for i,v in enumerate(history)],MUTED,2)
        divide=Line(p(1000,allvalues.min()-.3),p(1000,allvalues.max()+.3)).set_stroke(ORANGE,1.5)
        g=VGroup(axes,actual,divide,label('dopasowanie do 1000 | prognoza do 1500',[0,2.24,0],21,MUTED))
        anim=[]
        for model,col,y in [('koopman',CYAN,-2.12),('langevin',GREEN,-2.48)]:
            prediction=D['forecast'][model]
            curve=poly([p(1000,history[100])]+[p(1010+j*10,v) for j,v in enumerate(prediction['values'])],col,3)
            g.add(curve,label(f"{model}: RMSE {prediction['rmse']:.2f} | brak zmiany: {prediction['rmse_persistence']:.2f}",[0,y,0],17,col))
            anim.append(ShowCreation(curve))
        return g,anim

    def results(self):
        rows=[('Norma ruchu',.506,CYAN),('Średni kierunek',.669,GREEN),('16 osi SVD',.611,PURPLE),('16 losowych osi',.608,ORANGE)]
        g=VGroup(); animations=[]
        for j,(name,v,col) in enumerate(rows):
            y=1.7-j*.9
            g.add(label(name,[-4.6,y,0],24,FG,3.5))
            full=Rectangle(width=6.6,height=.40).set_fill('#1b2a41',1).set_stroke(width=0).move_to([1.1,y,0]);g.add(full)
            bar=Rectangle(width=max(.01,6.6*v),height=.40).set_fill(col,1).set_stroke(width=0).move_to([-2.2+3.3*v,y,0]);g.add(bar)
            g.add(label(f'{v:.3f}',[5.25,y,0],26,col))
            animations.append(GrowFromEdge(bar,LEFT))
        g.add(label('ROC AUC: 0,5 = brak rozdzielenia; 1,0 = pełne rozdzielenie',[0,-2.2,0],19,MUTED))
        return g,animations

    def projection(self,k):
        rng=np.random.default_rng(6)
        a=rng.normal(size=(40,2))*.48+[-2.2,-.6]
        b=rng.normal(size=(40,2))*.5+[1.9,.7]
        to3=lambda x:np.column_stack([x,np.zeros(len(x))])
        g=Group(DotCloud(to3(a),color=MUTED,radius=.055),DotCloud(to3(b),color=GREEN,radius=.055))
        direction=arrow([-4,-1.7,0],[4,1.7,0],PURPLE,3);g.add(direction)
        g.add(label('tło',[-3.2,-1.15,0],23,MUTED),label('narzędzie',[2.8,1.2,0],23,GREEN),label('kierunek w',[4.6,1.6,0],22,PURPLE))
        if k=='subspace':
            plane=Rectangle(width=8,height=2.5).set_fill(PURPLE,.08).set_stroke(PURPLE,1).rotate(.34)
            g.add(plane,label('Podprzestrzeń W: kilka osi, wiele informacji',[0,-2.2,0],21,PURPLE))
        elif k=='lda':
            g.add(label('Różnica średnich + kowariancja + regularyzacja',[0,-2.2,0],21,CYAN))
        elif k=='score':
            for v in b[::5]:
                u=np.array([1,.425]);proj=u*np.dot(v,u)/np.dot(u,u)
                g.add(Line([*v,0],[*proj,0]).set_stroke(GREEN,1,.5))
            g.add(label('Rzut daje skalarny odczyt; reszta wektora pozostaje',[0,-2.2,0],21,CYAN))
        else:g.add(label('Token może mieć równocześnie cechy narzędzia i tła',[0,-2.2,0],21,MUTED))
        return g,[ShowCreation(direction)]

    def network(self,k):
        if k=='dimensions':
            left=card('600 tokenów\n192 cechy',[-4,.4,0],3,1.5,CYAN,29)
            mid=card('MLP + GELU',[0,.4,0],2.7,1.2,PURPLE,25)
            right=card('600 tokenów\n768 aktywacji',[4,.4,0],3.2,1.5,GREEN,28)
            a=arrow([-2.4,.4,0],[-1.5,.4,0]);b=arrow([1.5,.4,0],[2.3,.4,0],GREEN)
            bars=VGroup(*[Rectangle(width=.075,height=.3+.7*(j%5)/5).set_fill(PURPLE,.7).set_stroke(width=0).move_to([-4.7+j*.19,-1.4,0]) for j in range(50)])
            return VGroup(left,mid,right,a,b,bars),[ShowCreation(a),ShowCreation(b)]
        if k=='position':
            cards=VGroup(card('cechy treści',[-4,.6,0],3,1,CYAN),card('kod miejsca',[0,.6,0],3,1,ORANGE),card('token + pozycja',[4,.6,0],3.2,1,GREEN))
            g=VGroup(cards,label('+',[-2,.6,0],40),label('=',[2,.6,0],40))
            xx=np.linspace(-5.5,5.5,150)
            wave1=poly([[x,-1.2+.38*np.sin(2*x),0] for x in xx],ORANGE)
            g.add(wave1,label('sinus / cosinus • stały kod dla tego samego miejsca',[0,-2.15,0],20,MUTED))
            return g,[ShowCreation(wave1)]
        if k=='architecture':
            labels=['RGB','4 konwolucje','6 bloków ViT','głowa detekcji']; xs=[-5,-1.8,1.8,5]
            g=VGroup(*[card(s,[x,.9,0],2.7,1,CYAN if i<2 else PURPLE,22) for i,(x,s) in enumerate(zip(xs,labels))])
            arr=VGroup(*[arrow([xs[i]+1.4,.9,0],[xs[i+1]-1.4,.9,0]) for i in range(3)])
            g.add(arr,card('Odczyt: MLP ostatniego bloku',[1.5,-1,0],5.2,.9,GREEN),arrow([1.8,.3,0],[1.8,-.45,0],GREEN))
            return g,[ShowCreation(arr)]
        # Attention: pokazujemy schemat wag, nie zmierzoną macierz uwagi.
        points=[np.array([-5+i*2,1.1,0]) for i in range(6)]
        nodes=VGroup(*[Circle(radius=.27).set_fill(GREEN if i==2 else CYAN,.7).set_stroke(width=0).move_to(p) for i,p in enumerate(points)])
        target=np.array([0,-1,0]);center=Circle(radius=.4).set_fill(PURPLE,.8).move_to(target)
        edges=VGroup(*[arrow(p+[0,-.3,0],target+[0,.4,0],GREEN if i==2 else MUTED,1+(i%3)) for i,p in enumerate(points)])
        g=VGroup(nodes,center,edges,label('ważona suma informacji',[0,-1.9,0],25,PURPLE),label('Q × K → wagi uwagi → mieszanina V',[0,2.1,0],24,CYAN))
        return g,[LaggedStart(*[ShowCreation(e) for e in edges],lag_ratio=.12)]

    def oscillation(self,k):
        xs=np.linspace(0,1,160)
        g=VGroup()
        for j,col in enumerate([CYAN,ORANGE]):
            ys=(np.sin(xs*12*np.pi)*.7 if j==0 else np.zeros_like(xs)) if k=='aliasing' else np.exp(-xs*(4 if j==0 else 1.2))*np.cos(xs*10)
            line=poly([[-5.5+x*11,ysv+(1 if j==0 else -.8),0] for x,ysv in zip(xs,ys)],col)
            g.add(line)
        g.add(label('sygnał / zależność w czasie',[0,2.1,0],24,CYAN),label('wygładzenie / inna pamięć',[0,-1.9,0],24,ORANGE))
        return g,[ShowCreation(g[0]),ShowCreation(g[1])]

    def equations(self,k):
        formulas={
            'langevin':('Δs = k·s + suma zmian z historii + c + reszta',['stan','pamięć','stała sondy','reszta']),
            'forces':('następny ruch = dryf + pamięć + reszta',['dryf','historia','niepewność','suma']),
            'full_latent':('Δz(t+1) ≈ 0,83 Δz(t) + 0,98 S(t)',['768 cech','poprzedni ruch','siła paczki','następny ruch']),
            'optimizer':('S(t) = z(θ + dnowa) − z(θ)',['paczka','gradient','część kroku AdamW','odczyt sond']),
            'explanation':('U = 100 · (1 − SSEmodel / SSEbrak_zmiany)',['mediana ≈ 85%','10 przebiegów','jeden krok','znana paczka']),
            'no_leak':('dostępne: historia i stan optymalizatora',['przeszłość','chwila prognozy','przyszła paczka','jej gradient'])}
        f,labels=formulas[k]
        g=VGroup(card(f,[0,1.65,0],12,.85,PURPLE,28))
        cols=[CYAN,GREEN,ORANGE,RED] if k=='no_leak' else [CYAN,PURPLE,GREEN,ORANGE]
        boxes=VGroup(*[card(t,[-4.8+i*3.2,-.35,0],2.9,1.05,cols[i],22) for i,t in enumerate(labels)])
        arrows=VGroup(*[arrow([-3.3+i*3.2,-.35,0],[-3.1+i*3.2,-.35,0],cols[i]) for i in range(3)])
        g.add(boxes,arrows)
        msg='Nie znamy przyszłego gradientu w chwili prognozy' if k=='no_leak' else 'Każdy składnik ma inne znaczenie i wymaga osobnej kontroli'
        g.add(label(msg,[0,-1.9,0],23,MUTED,12))
        return g,[LaggedStart(*[Indicate(b,color=cols[i]) for i,b in enumerate(boxes)],lag_ratio=.35)]

    def flow(self,k):
        descriptions={
        'overview':['Obraz + maska','600 tokenów','Kierunki informacji','Historia uczenia','Koopman / Langevin','Dobór klatek'],
        'reset':['Klatka A','ViT: te same wagi','Tokeny A','Klatka B','ViT: te same wagi','Tokeny B'],
        'two_times':['Film: obraz A','obraz B','obraz C','Uczenie: θ₁','θ₂','θ₃; ten sam obraz'],
        'invariance':['Ten sam cystotom','obrót','przesunięcie','inne tło','inna długość','odczyt tożsamości'],
        'pooling':['Token 1','Token 2','Token n','maska narzędzia','średnia po tokenach','jeden wektor z'],
        'causal':['Oryginalny latent','usuń kierunek','zmierz detekcję','Kontrola','usuń losowy kierunek','porównaj różnicę'],
        'experiment':['Stały obraz','wagi θ₁','600 wektorów z₁','Ten sam obraz','wagi θ₂','600 wektorów z₂'],
        'background_drift':['Ruch narzędzia','Ruch tła','wspólny składnik','Δz / Δt','odejmij referencję','pozostały ruch'],
        'dense':['krok 0','krok 1','krok 2','…','krok 999','krok 1000'],
        'svd':['Ruch tokena 1','Ruch tokena 2','Ruch tokena n','macierz R','rozkład SVD','kierunki V'],
        'holdout':['8 filmów','dopasowanie osi','zamrożone kierunki','4 inne filmy','rzuty ruchu','porównanie z tłem'],
        'fair_compare':['ten sam start','surowa historia','wygładzona historia','ten sam horyzont','te same cele','błąd poza próbą'],
        'koopman':['stan x','dynamika F','następny stan','obserwacja ψ(x)','model liniowy K','prognoza ψ(F(x))'],
        'lags':['Δs(t−3)','Δs(t−2)','Δs(t−1)','Δs(t)','współczynniki a','prognoza Δs(t+1)'],
        'fit':['wiersze historii X','następne zmiany Y','tylko przeszłość','regresja + kara','współczynniki','test przyszłości'],
        'rollout':['ostatni stan','prognoza zmiany','dodaj do stanu','wpisz do historii','następna prognoza','błąd może narastać'],
        'koopman_limit':['maska: które tokeny','LDA: jaki odczyt','Koopman: jaka zmiana','trzy osobne operacje','trzy osobne testy','bez skracania wniosku'],
        'selection_model':['dostępna historia','ukryj końcówkę','dopasuj K oraz L','porównaj błędy','dopasuj zwycięzcę','prognozuj przyszłość'],
        'retrieval':['kotwica obrazu','prognoza Δs','sterowanie FastGAN','obraz do szukania','podobne klatki DINO','trening na realnych'],
        'diversity':['słaby odczyt','zaskoczenie','błąd detektora','kandydaci','różnorodność','ogranicz sąsiadów'],
        'controls':['losowe klatki','sam odczyt','ruch + odczyt','ten sam budżet','ten sam trening','odłożone operacje'],
        'evidence':['maski: są','latent: zmierzony','kierunki: kandydaci','znaczenie: testuj','prognozę: sprawdź','zysk: osobny trening'],
        'end':['Obraz i maska','Osobne tokeny','Kierunki konceptu','Dynamika uczenia','Dobór klatek','Lepsza detekcja?']}
        titles=descriptions[k]
        pos=[[-4.7,1.05,0],[0,1.05,0],[4.7,1.05,0],[-4.7,-1.0,0],[0,-1.0,0],[4.7,-1.0,0]]
        cols=[CYAN,CYAN,GREEN,PURPLE,PURPLE,ORANGE]
        boxes=VGroup(*[card(t,p,3.7,1.05,col,24) for t,p,col in zip(titles,pos,cols)])
        lines=VGroup()
        for a,b in [(0,1),(1,2),(3,4),(4,5)]:
            lines.add(arrow(np.array(pos[a])+[1.9,0,0],np.array(pos[b])+[-1.9,0,0]))
        if k not in ('reset','two_times','experiment','causal','holdout'):
            lines.add(arrow(np.array(pos[2])+[0,-.6,0],np.array(pos[3])+[0,.6,0],PURPLE,1.4))
        g=VGroup(boxes,lines)
        return g,[LaggedStart(*[Indicate(b,color=cols[i]) for i,b in enumerate(boxes)],lag_ratio=.22)]


class Rozdzial01(PipelineFilm): chapter_index=0
class Rozdzial02(PipelineFilm): chapter_index=1
class Rozdzial03(PipelineFilm): chapter_index=2
class Rozdzial04(PipelineFilm): chapter_index=3
class Rozdzial05(PipelineFilm): chapter_index=4
class Rozdzial06(PipelineFilm): chapter_index=5
class Rozdzial07(PipelineFilm): chapter_index=6
class Rozdzial08(PipelineFilm): chapter_index=7
class Rozdzial09(PipelineFilm): chapter_index=8
class Rozdzial10(PipelineFilm): chapter_index=9
class Rozdzial11(PipelineFilm): chapter_index=10
class Rozdzial12(PipelineFilm): chapter_index=11
