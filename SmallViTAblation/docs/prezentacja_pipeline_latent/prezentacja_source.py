"""Prezentacja pipeline'u dla promotora; dane wynikowe pozostają niezmienione."""
from pathlib import Path
import json, hashlib, shutil, datetime, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/prezentacja_pipeline_latent'
ASSET=OUT/'materialy'
OUT.mkdir(parents=True,exist_ok=True); ASSET.mkdir(exist_ok=True)
W,H=1280,720
NAVY='#142D42'; TEAL='#087E8B'; ORANGE='#D97725'; LIGHT='#F0F5F7'; MUTED='#506779'
FONT_DIR=Path('C:/Windows/Fonts')
for name,file in [('Arial','arial.ttf'),('ArialB','arialbd.ttf'),('ArialI','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONT_DIR/file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialB',italic='ArialI',boldItalic='ArialB')
sources={}
def asset(rel):
    p=ROOT/rel
    if not p.is_file():raise FileNotFoundError(p)
    key=str(p.relative_to(ROOT)).replace('\\','/')
    sources[key]=hashlib.sha256(p.read_bytes()).hexdigest()
    if p.suffix.lower() in ('.png','.jpg','.jpeg'):
        dst=ASSET/(hashlib.sha256(key.encode()).hexdigest()[:8]+'_'+p.name)
        if not dst.exists():shutil.copy2(p,dst)
        return dst
    return p
def read(rel):return json.loads(asset(rel).read_text(encoding='utf-8-sig'))
single=read('outputs/vit_latent128_jedno_zdjecie_90/RESULT.json')
nearest=read('outputs/vit_sd_obraz_najblizszy/RESULT.json')
bound=read('outputs/vit_latent768_pelny_zbior/DIAGNOSTYKA_OSIAGALNOSCI.json')
oldsd=read('outputs/vit_latent768_sd_od_szumu/RESULT.json')
new_sd=read('outputs/vit_latent768_sd_od_najblizszego/RESULT.json')
pooled=read('outputs/vit_latent128_maksymalizacja/RESULT.json')
asset('docs/23_METODA_PRZEWIDYWANIA_LATENTU.md')
asset('experiments/maly_vit_tozsamosc90_bez_pca_langevin/WYNIKI.md')
asset('experiments/vit_latent768_pelny_zbior/model_snapshot.py')
asset('outputs/vit_latent128_jedno_zdjecie_90/WIZUALIZACJA768/JAK_CZYTAC.md')
pairs=read('experiments/maly_vit_test_orientacji_narzedzia_400/PARY.json')
pairroot='experiments/maly_vit_test_orientacji_narzedzia_400/'
atlas='outputs/vit_latent128_jedno_zdjecie_90/WIZUALIZACJA768/'
source=asset('outputs/vit_latent128_jedno_zdjecie_90/ZDJECIE_ZRODLOWE.jpg')
source480=asset('outputs/vit_latent128_jedno_zdjecie_90/WEJSCIE_VIT.png')
sd=asset('outputs/vit_sd_obraz_najblizszy/OBRAZ_ZAPYTANIA.png')
found=asset('outputs/vit_sd_obraz_najblizszy/NAJBLIZSZE_ZDJECIE.png')
latent=asset('outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz')
with np.load(latent) as f:before=f['before'];after=f['after'];J=f['channels0']
assert before.shape==after.shape==(600,768) and J.size==128
assert np.array_equal(before[:,np.setdiff1d(np.arange(768),J)],after[:,np.setdiff1d(np.arange(768),J)])
PDF=OUT/'PIPELINE_LATENT_VIT_SD_LANGEVIN.pdf'
c=canvas.Canvas(str(PDF),pagesize=(W,H),pageCompression=1)
c.setTitle('Sterowanie uczeniem ViT przez cel w latencie - pipeline i eksperymenty')
c.setAuthor('Projekt badawczy użytkownika; opracowanie Codex na podstawie artefaktów')
pages=[]; overflows=[]
def text(x,y,txt,size=22,color=NAVY,bold=False):
    c.setFillColor(HexColor(color));c.setFont('ArialB' if bold else 'Arial',size);c.drawString(x,y,txt)
def para(x,top,width,txt,size=23,color=NAVY,leading=None,maxh=None):
    sty=ParagraphStyle('p',fontName='Arial',fontSize=size,leading=leading or size*1.32,textColor=HexColor(color))
    p=Paragraph(txt,sty);aw,ah=p.wrap(width,H)
    if maxh and ah>maxh:overflows.append(dict(page=len(pages),text=txt[:70],height=ah,max=maxh))
    p.drawOn(c,x,top-ah);return ah
def rect(x,y,w,h,color=LIGHT,stroke=None,r=13):
    c.setFillColor(HexColor(color))
    c.setStrokeColor(HexColor(stroke or color));c.roundRect(x,y,w,h,r,fill=1,stroke=bool(stroke))
def img(p,x,y,w,h):
    im=Image.open(p);iw,ih=im.size;s=min(w/iw,h/ih);dw,dh=iw*s,ih*s
    c.drawImage(ImageReader(im),x+(w-dw)/2,y+(h-dh)/2,dw,dh,mask='auto')
def panel_image(p,x,y,w,h,label,note=None):
    rect(x,y,w,h);img(p,x+8,y+45,w-16,h-55)
    para(x+14,y+32,w-28,label,size=18,maxh=28)
    if note:para(x,y-12,w,note,size=15,color=MUTED,maxh=54)
def slide(title,kicker,section='PIPELINE',status=None):
    if pages:c.showPage()
    pages.append(title)
    c.setFillColor(HexColor('#FFFFFF'));c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor(HexColor(TEAL));c.rect(0,H-12,W,12,fill=1,stroke=0)
    text(52,H-48,section.upper(),15,TEAL,True)
    text(52,H-98,title,34,NAVY,True)
    para(52,H-127,1176,kicker,size=18,color=MUTED,maxh=50)
    c.setStrokeColor(HexColor('#DFE7EB'));c.line(52,51,W-52,51)
    text(52,28,'SmallViTAblation | Stan materiału: 5 października 2026',12,MUTED)
    c.setFont('Arial',12);c.drawRightString(W-52,28,f'{len(pages):02d}')
    if status:text(W-280,H-47,status,14,ORANGE,True)
def foot(txt):para(52,94,1176,txt,size=15,color=MUTED,maxh=35)
def card(x,y,w,h,title,body,color=LIGHT):
    rect(x,y,w,h,color);text(x+20,y+h-35,title,22,TEAL,True)
    para(x+20,y+h-57,w-40,body,size=20,maxh=h-70)
def arrow(x1,y1,x2,y2,color=TEAL):
    c.setStrokeColor(HexColor(color));c.setLineWidth(3);c.line(x1,y1,x2,y2)
    a=math.atan2(y2-y1,x2-x1)
    p=c.beginPath();p.moveTo(x2,y2);p.lineTo(x2-12*math.cos(a-.45),y2-12*math.sin(a-.45));p.lineTo(x2-12*math.cos(a+.45),y2-12*math.sin(a+.45));p.close()
    c.setFillColor(HexColor(color));c.drawPath(p,fill=1,stroke=0)
def box(x,y,w,h,title,body):
    rect(x,y,w,h,LIGHT,TEAL);para(x+16,y+h-18,w-32,'<b>'+title+'</b><br/>'+body,size=19,maxh=h-25)
def rows(items,x=60,top=520,width=1140,size=23,gap=20):
    for head,body in items:
        ah=para(x,top,width,'<b>'+head+'</b> '+body,size=size);top-=ah+gap
    return top
def table(headers,data,x,y,width,rowh=48,colweights=None,size=19):
    ws=np.array(colweights or [1]*len(headers),dtype=float);ws=width*ws/ws.sum()
    rect(x,y-rowh,width,rowh,TEAL,r=3)
    xx=x
    for h,w in zip(headers,ws):para(xx+12,y-12,w-24,'<b>'+h+'</b>',size=size,color='#FFFFFF',maxh=rowh-12);xx+=w
    for i,row in enumerate(data):
        yy=y-(i+2)*rowh;rect(x,yy,width,rowh-2,LIGHT if i%2==0 else '#FFFFFF',r=0);xx=x
        for val,w in zip(row,ws):para(xx+12,yy+rowh-12,w-24,str(val),size=size,maxh=rowh-12);xx+=w

# 01
slide('Od celu w latencie do wyboru danych','','CEL BADAŃ')
panel_image(source,52,243,370,300,'Sonda i checkpoint ViT')
panel_image(sd,455,243,370,300,'Obraz sterowany latentem - SD')
panel_image(found,858,243,370,300,'Rzeczywisty kandydat z train')
para(66,210,1130,'<b>Pytanie:</b> czy korzystna zmiana reprezentacji pozwala dobrać zdjęcia i kolejność paczek tak, aby szybciej poprawić detekcję?',size=26,maxh=100)
foot('Fotografie pochodzą z lokalnych artefaktów projektu. Nie traktujemy poprawy po interwencji jako dowodu poprawy po treningu.')
# 02
slide('Cały pipeline w jednym schemacie','Każda strzałka oznacza osobną zależność, którą trzeba zweryfikować eksperymentalnie.','MAPA ALGORYTMU')
nodes=[('1. Checkpoint + sondy','Stan sieci, etykiety i pamięć optymalizatora'),('2. Cel w latencie','Ograniczona korekta wybranych J128'),('3. Obrazy SD','Różne szumy; pomiar zgodności z celem'),('4. Wyszukiwanie','Podobne prawdziwe zdjęcia z puli train'),('5. Paczki + prognoza','Siły, kolejność i czas zbliżenia do celu'),('6. Uczenie + ocena','Aktualizacja i niezależna jakość detekcji')]
for i,(title,body) in enumerate(nodes):
    row=i//3;col=i%3 if row==0 else 5-i;x=52+col*408;y=350-row*190;box(x,y,360,150,title,body)
    if row==0 and col<2:arrow(x+365,y+75,x+402,y+75)
    if row==1 and col>0:arrow(x-5,y+75,x-42,y+75)
arrow(1048,347,1048,315)
foot('Pętla zamknięta: po krótkim uczeniu aktualizujemy stan, cel i prognozę. Równolegle testujemy wariant bez SD.')
# 03
slide('Jak działa badany SmallViT','To konkretny hybrydowy model: 600 tokenów obrazu, wymiar stanu bloku 192, mierzony latent MLP 768.','MODEL')
im=Image.open(source480).convert('RGB');dr=ImageDraw.Draw(im)
for x in range(0,480,16):dr.line((x,0,x,320),fill=(20,190,210),width=1)
for y in range(0,320,16):dr.line((0,y,480,y),fill=(20,190,210),width=1)
grid=ASSET/'siatka600.png';im.save(grid)
panel_image(grid,52,219,520,337,'480 × 320 pikseli → 20 × 30 = 600 tokenów')
steps=[('Obraz RGB','Normalizacja i 4 konwolucje, krok przestrzenny 16 px.'),('6 bloków transformer','192 współrzędne stanu bloku, 3 głowy uwagi.'),('Punkt pomiaru','GELU w MLP ostatniego bloku: 600 × 768.'),('Dalsza część sieci','Projekcja MLP do 192 + połączenie resztowe; norm i głowa: pewność oraz ramka.')]
rows(steps,610,550,610,21,17)
foot('Źródło: model_snapshot.py. Latent 768D jest wewnętrzną aktywacją MLP, a nie bezpośrednim wejściem końcowej głowy.')
# 04
slide('Cząsteczki, makrowektory i mikrowektory','Stała baza współrzędnych i zmienne położenia tokenów - zgodnie z terminologią projektu.','JAK CZYTAĆ LATENT')
card(52,340,570,200,'Cząsteczka','Jeden token obrazu: wektor z_i w przestrzeni 768D. Dla jednego obrazu śledzimy 600 takich wektorów.')
card(658,340,570,200,'Makrowektory','768 wspólnych osi e_j. J128 to 128 wybranych osi; w sztucznej interwencji baza się nie obraca.')
card(52,120,570,195,'Mikrowektory','Składowe z_ij e_j. Zmieniamy liczby z_ij. Cały wektor tokenu może zmienić długość i kierunek.')
card(658,120,570,195,'Dwa różne zegary','Inne zdjęcie przy stałych wagach ≠ ten sam obraz po kroku uczenia. Prognoza treningu dotyczy drugiego przypadku.')
# 05
slide('Skąd wzięły się kanały powiązane z narzędziem','400 par: prawdziwy obraz i ta sama klatka po lokalnym usunięciu cystotome przez LaMa.','ETAP 1 | IDENTYFIKACJA')
for row,idx in enumerate([0,2]):
    p=pairs[idx];yy=340-row*190
    for col,(key,label) in enumerate([('original','Oryginał'),('mask_edit','Maska edycji'),('synthetic','Po LaMa')]):
        panel_image(asset(pairroot+p[key].replace('\\','/')),52+col*405,yy,370,175,label)
foot('Te same tokeny oryginalnej maski w obu obrazach. Poza maską edycji piksele zachowano; odtworzone tło jest syntetyczne.')
# 06
slide('J128: selekcja i zakres znaczenia','Kanały wybrano na parach oryginał / LaMa. To test obecności narzędzia w znanym obszarze.','ETAP 1 | WYNIK',status='WYKONANO')
rows([('Ranking wyłącznie train:','|średnia różnica par| / odchylenie różnicy; 240 par z 69 filmów.'),('Dobór na validation:','75 par / 23 filmy; najmniejsze sprawdzone K z BA > 90%: K = 128. Bez PCA.'),('Test:','85 par / 24 filmy; film-weighted balanced accuracy 92,83%. CI95: 88,17-96,99%.'),('Interpretacja:','J128 wspólnie niesie sygnał obecności. Nie dowodzi wyłącznej odpowiedzialności kanałów ani pełnego domknięcia detektora.')],52,548,1140,25,21)
foot('Źródło: maly_vit_tozsamosc90_bez_pca_langevin/WYNIKI.md. Dane bazowe należą do train ViT; to nie nowy, nietknięty test detektora.')
# 07
slide('Sztuczna korekta na jednej sondzie','Piksele i wagi zachowano. Gradient zmienia wyłącznie aktywacje wybranych 128 kanałów.','ETAP 2 | INTERWENCJA',status='WYKONANO')
img(asset('outputs/vit_latent128_jedno_zdjecie_90/DETEKCJA_PRZED_PO.png'),52,185,810,365)
rows([('Przed:','pewność 86,79%; IoU z etykietą 85,66%.'),('Po korekcie:','pewność 93,17%; IoU 99,93%.'),('Co to pokazuje:','dalsza część sieci potrafi wykorzystać zmieniony stan.'),('Co pozostaje:','sprawdzić, czy prawdziwe uczenie może wytworzyć korzystną zmianę.')],900,540,330,21,20)
foot('Film TR_0154_S1_P03, klatka 1926. Użyto etykiety tej sondy w optymalizacji; nie jest to niezależna poprawa generalizacji.')
# 08
slide('Wszystkie 768 kanałów: przed i po','Pomarańczowe obwódki wyróżniają J128. Pozostałe 640 współrzędnych zachowano bitowo.','WIZUALIZACJA')
img(asset(atlas+'LATENT768_PRZED_PO.png'),52,110,1176,445)
foot('Układ kafelków jest indeksem kanałów, nie geometrycznym rzutem 768D. Mapy na obrazie odnoszą aktywacje do siatki tokenów.')
# 09
slide('768 makrowektorów: przed zmianą, po zmianie i różnica','Każdy makrowektor odpowiada jednej osi latentu. Wykres pokazuje średnią wartość jej kanału.','WIZUALIZACJA')
para(52,550,1176,'Pomarańczowy: wybrane 128 kanałów. Szary: pozostałe 640, zachowane dokładnie.',size=18,maxh=35)
img(asset('outputs/prezentacja_pipeline_latent/makrowektory_wartosci/MAKROWEKTORY_SLAJD9.png'),52,177,1176,327)
para(52,146,1176,'<b>Kierunki makrowektorów pozostają stałe.</b> Porównujemy wartości przypisane ich kanałom.<br/>PRZED i PO mają tę samą skalę; RÓŻNICA ma osobną. Wartości są średnimi aktywacjami kanałów, nie długościami osi bazowych.',size=17,maxh=80)
# 10
slide('Cel musi być osiągalny lub mieć osiągalne przybliżenie','Znaleziony sztuczny stan przekracza konieczne granice aktywacji przy zamrożonym checkpointcie.','ETAP 2 | DIAGNOSTYKA',status='WYKONANO')
card(52,308,360,224,'51,35','Największa wartość sztucznego celu.')
card(460,308,360,224,'10,25','Największa górna granica kanału wynikająca z wag i LayerNorm tego checkpointa.')
card(868,308,360,224,'85 kanałów','15 658 wartości token-kanał przekracza indywidualne granice.')
para(66,260,1140,'<b>Wniosek:</b> żaden obraz nie odtworzy dokładnie tego celu w tej zamrożonej sieci. Ograniczenie dotyczy tego checkpointa; po zmianie wag granice trzeba przeliczyć.',size=26,maxh=110)
foot('Konieczna dolna granica MSE768: 6,2603. Spełnienie samych granic nie gwarantuje istnienia obrazu. Źródło: DIAGNOSTYKA_OSIAGALNOSCI.json.')
# 11
slide('Co oznacza wynik bliski 90 procent','Pewność, IoU, AP50, mAP i zgodność latentu opisują różne własności.','METRYKI')
table(['Miara','Co mierzy','Czego nie oznacza'],[
('Pewność detektora','Wynik dla przewidzianej ramki','Odsetek poprawnych detekcji'),
('IoU','Zgodność jednej ramki z etykietą','Jakość całego zbioru'),
('AP50 / AP75','Precision-recall przy progu IoU','Pewność pojedynczego zdjęcia'),
('mAP / AP50:95','Jakość po klasach / progach IoU','Podobieństwo latentów'),
('MSE / cosinus latentu','Odległość / zgodność kierunku','Samodzielnie: lepsza detekcja')],52,543,1176,67,[1.15,1.75,1.9],20)
foot('Na 400 obrazach osobna korekta z etykietami dała AP50 ≈ 0,885 wobec ≈ 0,438. To pomiar z wyrocznią, nie deployowalny detektor.')
# 12
slide('Stable Diffusion jako most do przestrzeni obrazów','Stały cel ViT kieruje odszumianiem; prior SD pomaga utrzymać strukturę fotografii.','ETAP 3 | GENEROWANIE')
panels=[('outputs/vit_latent768_sd_od_szumu/SZUM_STARTOWY.png','Szum startowy'),('outputs/vit_latent768_sd_od_szumu/OBRAZ_SD_PO_DDIM.png','Po odszumianiu DDIM'),('outputs/vit_latent768_sd_od_szumu/OBRAZ_SD.png','Po dawnej dalszej optymalizacji')]
for col,(p,label) in enumerate(panels):panel_image(asset(p),52+col*405,248,370,285,label)
para(62,210,1150,'Generator otrzymuje sygnał błędu: jak różni się naturalny latent wygenerowanego obrazu od celu. W starym biegu użyto pustego tekstu i zerowych masek; nie podano obrazu źródłowego.',size=23,maxh=95)
foot('To próba odwrócenia reprezentacji. Nie daje gwarancji, że obraz wskazuje cechy korzystne podczas treningu.')
# 13
slide('Pierwsza generacja: obraz powstał, cel nie został odtworzony','Zniekształcenia są wynikiem optymalizacji, a nie niedokończonym renderowaniem.','ETAP 3 | WYNIK',status='WYKONANO')
panel_image(sd,52,157,570,390,'Obraz z szumu - wcześniejszy eksperyment')
table(['Miara','Bez prowadzenia','Z prowadzeniem'],[
('MSE768','14,29887','14,33022'),('MSE128','85,49728','85,76305'),('Pewność','67,84%','68,71%')],660,523,568,63,[1.2,1,1],18)
para(673,240,535,'Ważona energia celu spadła o 29,85%, ale surowe MSE128 i MSE768 wzrosło. Dlatego nie interpretujemy artefaktów jako mapy brakujących cech narzędzia.',size=22,maxh=140)
foot('Źródło: vit_latent768_sd_od_szumu/RESULT.json. Cele i miary trzeba dobrać tak, by nie nagradzać poprawy tylko łatwiejszej części stanu.')
# 15
slide('Wyszukiwanie rzeczywistego odpowiednika obrazu SD','Każdy kandydat przechodzi przez ten sam zamrożony ViT i identyczne przygotowanie wejścia.','ETAP 4 | WYNIK',status='WYKONANO')
panel_image(sd,52,215,565,330,'Zapytanie: kolorowy obraz wygenerowany przez SD')
panel_image(found,663,215,565,330,'Trafienie: TR_0006_S1_P03, klatka 1193, train')
table(['Zakres','MSE768','Cosinus768','Pewność'],[('461 547 zdjęć / 170 ZIP','0,02959','0,8748','60,99%')],52,205,1176,48,[1.9,1,1,1],20)
foot('Minimum dotyczy naturalnego latentu obrazu SD, nie pierwotnego sztucznego celu. Cosinus nie jest procentem poprawnych detekcji.')
# 16
slide('Jak wyszukujemy zdjęcia i tworzymy paczki','Praktyczna selekcja kandydatów, bez założenia, że podobieństwo gwarantuje korzystny gradient.','ETAP 4 | ALGORYTM')
rows([('A. Wzorzec:','naturalny latent obrazu SD albo bezpośrednio ograniczony cel sondy; oba warianty porównujemy.'),('B. Indeks:','latent lub jego roboczy skrót dla puli train. Aktualizacja indeksu po istotnej zmianie checkpointa.'),('C. Ranking:','odległość pełnego stanu oraz J128; usunięcie duplikatów i sąsiednich niemal identycznych klatek.'),('D. Różnorodność:','różne filmy, pozycje, skale i tła. Cztery generacje mogą wskazać cztery grupy kandydatów.'),('E. Paczka:','32 zdjęcia na aktualizację; pula może obejmować wiele paczek. Dobór całej paczki weryfikuje gradient.')],52,546,1150,23,19)
foot('Całe lokalne dane użyto do retrieval. Do uczenia i selekcji treningowej używamy train; niezależne filmy pozostają do oceny.')
# 17
slide('Siła paczki: skutek aktualizacji, nie cecha samych zdjęć','Siłę mierzymy na tej samej sondzie, przy aktualnych wagach i stanie AdamW.','ETAP 5 | APARAT')
box(52,349,350,158,'Paczka B_t','Zdjęcia + etykiety → strata → gradient g_t')
box(463,349,350,158,'Aktualizacja wag','Nowy gradient oraz pamięć optymalizatora')
box(874,349,350,158,'Sonda x','Różnica latentu z(theta + d, x) - z(theta, x)')
arrow(408,428,455,428);arrow(819,428,866,428)
para(66,298,1140,'<b>S_nowa:</b> wpływ części aktualizacji od bieżącego gradientu.<br/><b>S_pęd:</b> wpływ pamięci wcześniejszych gradientów.<br/>Te same zdjęcia mogą dawać inną siłę przy innych wagach.',size=26,maxh=145)
foot('Dodatkowe forwardy sond kosztują czas. Historyczne siły liczone na średnich tokenów narzędzia nie stanowią automatycznie modelu całego600×768.')
# 18
slide('Langevin: co jest opisem, a co prognozą','Wynik warunkowy ze zmierzonymi siłami nie gwarantuje autonomicznej prognozy wielokrokowej.','ETAP 5 | RÓWNANIE')
card(52,319,570,222,'Model ze zmierzoną siłą','Delta z_(t+1) = a · Delta z_t + b · S_t + reszta<br/><br/>Historyczne a ≈ 0,83; b ≈ 0,98. S_t zależy od gradientu bieżącej paczki.')
card(658,319,570,222,'Prognoza z pamięci','Delta z_(t+1) ≈ c · S_pęd,t + reszta<br/><br/>Historyczne c ≈ 0,94. Trzeba znać stan optymalizatora i pilnować informacji w mianowniku AdamW.')
para(65,272,1140,'Dla wielu kroków potrzebujemy przyszłych sił <b>przewidzianych</b> z obecnych danych. Przy stałej sile jest to scenariusz z założeniem, a przy rzeczywistych przyszłych gradientach - rekonstrukcja.',size=25,maxh=110)
foot('Źródło: docs/23_METODA_PRZEWIDYWANIA_LATENTU.md. Współczynniki przytoczono z dokumentu, nie dopasowano tutaj nowego modelu.')
# 19
slide('Zadanie odwrotne: które paczki i ile kroków?','Planowanie dojścia do celu przy użyciu modelu dynamiki, a nie algebraiczne odwrócenie równania.','ETAP 5 | HIPOTEZA')
rows([('Cel:','dla stałych sond określamy dopuszczalną odległość od korzystnego stanu i brak pogorszenia poprawnych detekcji.'),('Sterowanie:','wybór paczki, kolejności, liczby powtórzeń oraz mieszania z resztą train.'),('Prognoza:','symulujemy krótkie sekwencje i szukamy najkrótszej, która spełnia cel; dopuszczamy brak dojścia.'),('Weryfikacja:','wykonujemy kilka rzeczywistych kroków, mierzymy stan, odświeżamy siły i ponownie planujemy.'),('Jednostka:','kroki optymalizatora. Epoki zależą od wielkości puli; jedna paczka nie jest epoką całego zbioru.')],52,546,1150,24,17)
foot('Zmieniane wagi mogą zmieniać także odczyt głowy i granice aktywacji. Cel wyznaczony dla starego checkpointa wymaga kontroli.')
# 20
slide('Bieżące testy: cztery szumy i paczka32','Status jest migawką przygotowaną przy składzie PDF. Nowych wyników nie zastępujemy oczekiwaniami.','NAJBLIŻSZE EKSPERYMENTY')
for x,folder,title,body in [(52,'vit_sd_cztery_szumy','Cztery generacje SD','Cztery niezależne szumy, wspólny sztuczny cel, pomiar MSE i różnorodności. Zachować prior SD; bez nieograniczonej dalszej optymalizacji VAE.'),(658,'vit_langevin_paczka32_cel_sztuczny','Prognoza wpływu paczki32','32 unikalne podobne zdjęcia train. Oryginalna sonda i ten sam cel. Prognoza odległości i czasu dojścia; jawne założenia oraz porównanie z krótkim uczeniem.')]:
    pp=ROOT/'outputs'/folder/'PROGRESS.json'
    st=json.loads(pp.read_text(encoding='utf-8-sig')) if pp.exists() else {'status':'preparation by delegated agent'}
    sources['snapshot:'+folder]=st
    status_labels={'running':'obliczenia trwają','complete':'zakończono','queued':'oczekuje na GPU','queued_gpu':'oczekuje na wolną pamięć GPU','preparation by delegated agent':'przygotowanie przez agenta','fetching_actual_images_groundtruth_cpu':'przygotowanie zdjęć i etykiet','failed':'błąd - zobacz raport'}
    card(x,230,570,307,title,body+'<br/><br/><b>Status przy składzie:</b> '+status_labels.get(st.get('status'),str(st.get('status','brak odczytu'))))
foot('Pełne artefakty w outputs odpowiedniej rodziny. Testy uruchamiane w tle nie są dowodem osiągnięcia celu.')
new_images=sorted((ROOT/'outputs/vit_sd_cztery_szumy').glob('OBRAZ_*_prowadzony.png'))
if len(new_images)==4:
    variants=read('outputs/vit_sd_cztery_szumy/VARIANTS.json')
    slide('Cztery różne szumy, jeden cel ViT','MSE768 mniejsze niż w dopasowanych kontrolach o 0,016-0,315%. To niewielkie zbliżenie, nie odtworzenie celu.','ETAP 3 | CZTERY WARIANTY',status='WYGENEROWANO')
    for i,p in enumerate(new_images):
        v=next(v for v in variants if v['tag']==p.stem.replace('OBRAZ_',''))
        m=v['metrics']
        label=f"Wariant {i+1} | MSE768 {m['mse768']:.4f} | pewność {100*m['confidence']:.2f}%".replace('.',',')
        panel_image(asset(str(p.relative_to(ROOT))),52+(i%2)*612,330-(i//2)*213,560,202,label)
    foot('Wspólny cel pozostaje nieosiągalny w zamrożonym checkpointcie. Różne zdjęcia nie oznaczają równych latentów ani poprawy treningu.')
# 21
slide('Test rozstrzygający: czy wybrane dane uczą szybciej?','Wszystkie ramiona zaczynają z tych samych wag, buforów, optymalizatora i budżetu.','WALIDACJA')
table(['Ramię','Dobór danych','Po co'],[
('Kontrola','Losowe różnorodne paczki','Punkt odniesienia'),('Trudne przypadki','Selekcja po stracie detektora','Czy prostszy dobór wystarcza?'),('Cel bez SD','Wyszukiwanie z celu w latencie','Czy generator wnosi wartość?'),('SD + retrieval','Podobieństwo do generacji','Czy obraz pomaga selekcji?'),('SD + retrieval + prognoza','Dobór i długość uczenia','Czy Langevin poprawia decyzje?')],52,542,1176,64,[1.3,1.7,1.7],20)
foot('Ocena: AP50, AP75, AP50:95 na osobnych filmach, czas GPU oraz cały czas pipeline. Kilka ziaren; checkpoint wybierany bez testu.')
# 22
slide('Co pokaże sukces, a co obali mechanizm','Liczymy korzyść całego procesu, razem z generowaniem, wyszukiwaniem i oceną sił.','KRYTERIA')
card(52,285,570,255,'Sygnał sukcesu','Wybrane paczki przesuwają latent sond bliżej celu; prognoza zgadza się z rzeczywistą zmianą; docelowe AP osiągamy szybciej niż w kontrolach.')
card(658,285,570,255,'Sygnał porażki','Podobne zdjęcia nie dają korzystnego gradientu; cel pozostaje nieosiągalny; poprawa sond nie przenosi się na inne filmy; selekcja kosztuje więcej niż oszczędza.')
para(65,239,1145,'<b>Najważniejszy wynik badawczy:</b> czy cel w reprezentacji przewiduje użyteczność danych treningowych? Generator i Langevin są elementami, których wkład mierzymy oddzielnie.',size=27,maxh=105)
# 23
slide('Plan dla promotora: co wiemy i co mierzymy dalej','Oddzielamy wykazane własności checkpointa od proponowanego sterowania całym treningiem.','WNIOSEK')
rows([('Wykazano:','J128 rozpoznaje obecność w zadaniu par; sztuczna korekta zmienia detekcję; pełny retrieval znajduje najbliższy naturalny odpowiednik obrazu SD.'),('Ograniczenie:','pierwotny cel przekracza granice zamrożonej sieci, a SD nie odtworzyło jego aktywacji.'),('Następny krok:','cztery generacje, paczka podobnych zdjęć i prognoza vs rzeczywista zmiana oryginalnej sondy.'),('Główna teza do sprawdzenia:','dobór danych według korzystnego celu latentu oraz ich wpływu skraca czas osiągnięcia ustalonej jakości detekcji.')],52,546,1150,25,23)
foot('Materiał nie potwierdza prawie100% mAP deployowalnego modelu ani dokładnej prognozy liczby epok do takiego wyniku.')
# 24
slide('Źródła i możliwość odtworzenia','Zdjęcia i liczby pochodzą z zapisanych artefaktów; pełne ścieżki i SHA256 są w manifestach obok PDF.','DOKUMENTACJA')
rows([('Identyfikacja J128:','maly_vit_test_orientacji_narzedzia_400/PARY.json i maly_vit_tozsamosc90_bez_pca_langevin/WYNIKI.md.'),('Sztuczna korekta + wizualizacja:','vit_latent128_jedno_zdjecie_90/RESULT.json, LATENT_PELNY.npz i WIZUALIZACJA768.'),('SD i wyszukiwanie:','vit_latent768_sd_od_szumu, vit_latent768_sd_od_najblizszego, vit_sd_obraz_najblizszy.'),('Równanie i granice:','docs/23_METODA_PRZEWIDYWANIA_LATENTU.md oraz DIAGNOSTYKA_OSIAGALNOSCI.json.'),('Inspiracja metodologiczna:','Rousselot, Wendebourg, Köthe (2026), Show Me What You Don’t Know: Efficient Sampling from Invariant Sets for Model Validation. arXiv:2603.21782.')],52,547,1150,21,20)
c.linkURL('https://arxiv.org/abs/2603.21782',(52,119,1228,196),relative=0,thickness=0)
foot('Artykuł bada niezmienniczości przez generowanie obrazów o zgodnych cechach; nie dowodzi przyspieszenia treningu proponowanym pipeline.')

c.save()
manifest={'created_local':datetime.datetime.now().astimezone().isoformat(),'pdf':str(PDF),'pages':pages,'inputs':sources,'layout_overflows':overflows,'scope':'wykonane wyniki + jawnie oznaczone propozycje; bez nowych obliczeń treningowych w autorze PDF'}
(OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(__file__,OUT/'prezentacja_source.py')
reader=PdfReader(str(PDF))
assert len(reader.pages)==len(pages) and len(pages) in (24,25)
assert not overflows,overflows
for i,p in enumerate(reader.pages):
    t=p.extract_text();assert len(t)>150,(i,t)
(OUT/'TEKST_PREZENTACJI.txt').write_text('\n\n'.join(p.extract_text() for p in reader.pages),encoding='utf-8')
(OUT/'README.md').write_text(f'# Prezentacja pipeline latentu\n\nPDF ma {len(pages)} stron 16:9. Zdjęcia są kopiami artefaktów źródłowych. MANIFEST.json zapisuje pochodzenie i sumy kontrolne. Kod autora: tools/prezentacja_pipeline_latent.py.\n\nStatus nowych zadań to migawka chwili składu, a nie obietnica zakończenia.\n',encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':len(pages),'overflows':overflows,'bytes':PDF.stat().st_size},ensure_ascii=False))
