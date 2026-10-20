"""Create the explicitly requested report and short conclusion from measured data."""
from pathlib import Path
import json, html, hashlib, csv, zipfile
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
DATA = ROOT / "Dane"
REPORT = ROOT / "Raport"
SHORT = ROOT / "Krótki_Wniosek"
for p in [REPORT, SHORT]: p.mkdir(exist_ok=True)
S = json.loads((DATA / "SUMMARY.json").read_text(encoding="utf-8"))
INDEX = json.loads((ROOT / "Wykresy/INDEX.json").read_text(encoding="utf-8"))
META = json.loads((DATA / "DETAILS_METADATA.json").read_text(encoding="utf-8"))
font = Path("C:/Windows/Fonts/arial.ttf")
bold = Path("C:/Windows/Fonts/arialbd.ttf")
pdfmetrics.registerFont(TTFont("Body", str(font)))
pdfmetrics.registerFont(TTFont("BodyBold", str(bold)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TextPL", fontName="Body", fontSize=10.5, leading=15.5, spaceAfter=10, textColor=colors.HexColor("#263D52")))
styles.add(ParagraphStyle("TitlePL", fontName="BodyBold", fontSize=23, leading=28, spaceAfter=18, textColor=colors.HexColor("#18314B")))
styles.add(ParagraphStyle("HeadingPL", fontName="BodyBold", fontSize=16, leading=22, spaceAfter=14, textColor=colors.HexColor("#18314B")))
styles.add(ParagraphStyle("CellPL", fontName="Body", fontSize=8.7, leading=12, textColor=colors.HexColor("#263D52")))
story, webpage = [], []

def p(text):
    story.append(Paragraph(text, styles["TextPL"]))
    webpage.append("<p>"+text+"</p>")
def heading(text, first=False):
    if not first:
        story.append(PageBreak())
    story.append(Paragraph(text, styles["TitlePL"] if first else styles["HeadingPL"]))
    webpage.append("<h1>" + text + "</h1>" if first else "<h2>"+text+"</h2>")
def table(headers, rows, widths=None):
    cells=[[Paragraph(str(v),styles["CellPL"]) for v in row] for row in [headers]+rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E2EBF3")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F6F9")]),
        ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#A8BBCC")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
    ]))
    story.extend([t,Spacer(1,13)])
    webpage.append("<table><thead><tr>"+"".join("<th>"+str(x)+"</th>" for x in headers)+"</tr></thead><tbody>"+
        "".join("<tr>"+"".join("<td>"+str(x)+"</td>" for x in row)+"</tr>" for row in rows)+"</tbody></table>")

heading("Co trening robi z wiązkami MLP?",True)
p("<b>Wynik:</b> siła skupia się w mniejszej liczbie dominujących wkładów. Nie oznacza to masowego wyzerowania pozostałych wiązek. Średni kąt lekko maleje, ale rozkład kątów rozszerza się w obie strony względem 90°.")
p("Raport dotyczy jednej ukończonej ścieżki treningu MalyViT, seed3047. Odtworzono zapisane checkpointy; nie trenowano modelu ponownie. Każdy blok działał normalnie, bez wyłączania wiązek.")
table(["Element","Zakres"],[
 ["Model","6 bloków ViT; MLP 192 -> 768 -> GELU -> 192"],
 ["Punkty pomiarowe","11 checkpointów od kroku 0 do 55 000"],
 ["Dane pomiarowe","50 niezmiennych klatek TEST, po 2 z 25 filmów; 600 tokenów na klatkę"],
 ["Checkpoint końcowy","55 000: najlepszy IoU w osobnym porównaniu VAL; TEST nie wybierał checkpointu"],
 ["Epoki","35 000 obrazów TRAIN, batch 32; 55 000 × 32 / 35 000 = 50,29 epoki"],
 ["Wykresy",f"{len(INDEX)} plansz w PNG, SVG i PDF; galeria i wspólny atlas w Wykresy/"],
],[105,390])
p("<b>Co pokazuje badanie:</b> zmianę sił i geometrii tych samych wkładów w czasie. <b>Czego nie sprawdza:</b> czy usunięcie słabych wiązek zachowa IoU ani czy zmiany geometrii powodują poprawę detekcji. Do tego potrzebna jest osobna interwencja.")
p("Przy czytaniu wykresów zacznij od planszy 01. Plansze 02-08 wyjaśniają aktywność; 09-18 dotyczą kątów i kontroli; kolejne pokazują detekcję, rozkłady sił oraz szczegóły bloków i pojedynczego tokena.")

heading("1. Co mierzyliśmy i co oznacza „wygaszanie”")
p("Wiązka to <b>jeden wkład jednego neuronu dla konkretnego tokena</b>: u[t,c] = z[t,c] × W2[:,c], gdzie z jest aktywacją po GELU. Jej siła to |z[t,c]| × ||W2[:,c]||. Mamy 768 takich wkładów, a każdy jest wektorem 192D. Bias fc2 jest osobnym składnikiem.")
p("Za znaczącą uznajemy wiązkę o sile większej niż 5% najsilniejszej wiązki tego samego tokena. To próg względny. <b>Przykład objaśniający, nie pomiar:</b> maksimum rośnie z 1 do 10, więc próg rośnie z 0,05 do 0,5. Wiązka rosnąca z 0,1 do 0,2 wypada poniżej progu, chociaż staje się silniejsza.")
table(["Blok",">5%: start",">5%: 55k","N_eff: start","N_eff: 55k","Zera: 55k"],[
 [r["block"],f'{r["activity"]["active_rel_0.05"]["initial"]:.1f}',f'{r["activity"]["active_rel_0.05"]["best"]:.1f}',
  f'{r["activity"]["effective"]["initial"]:.1f}',f'{r["activity"]["effective"]["best"]:.1f}',f'{r["activity"]["exactzero"]["best"]:.2f}'] for r in S["blocks"]
],[40,90,90,90,90,95])
p("Wartości to średnie liczby na token, spośród 768. Efektywna liczba N_eff = (suma sił)² / suma kwadratów sił mierzy koncentrację bez ustalania progu. Jej spadek potwierdza koncentrację siły, a nie tylko zmianę arbitralnej granicy.")
p("W każdym bloku rośnie suma norm wkładów i mediana siły. Jednocześnie liczba dokładnych zer w najlepszym checkpoincie wynosi średnio około 0,01-3,31 na token. Setki wkładów tracą więc przede wszystkim znaczenie względne.")
r=S["blocks"][3]["endpoint_transitions"]
p(f'<b>Blok 4:</b> ponad próg na obu końcach pozostaje {r["persist"]["mean_beams_per_token"]:.1f} wiązki/token; '
  f'{r["lost"]["mean_beams_per_token"]:.1f} traci próg, a {r["new"]["mean_beams_per_token"]:.1f} wchodzi ponad próg. '
  f'Wśród tracących próg siła rzeczywiście maleje u {100*r["lost"]["fraction_absolute_decrease"]:.1f}%, '
  f'a spadek co najmniej 10-krotny dotyczy {100*r["lost"]["fraction_decrease10x"]:.1f}%. Pozostałe mogą rosnąć, lecz wolniej od dominantów.')

heading("2. Czy kąty rosną, czy maleją?")
p("Śledzimy tożsamości obraz-token-neuron. Główna grupa zawiera wiązki ponad progiem 5% na początku <b>i</b> w kroku 55k, dodatkowo niezerowe we wszystkich zapisach. Mogą one spaść poniżej progu w środku treningu i wrócić. Każda para dotyczy zawsze tego samego tokena.")
p("Kąt liczony jest w pełnych 192D, z uwzględnieniem znaku aktywacji. Ujemne GELU odwraca kierunek wkładu. Nie stosujemy projekcji do 3D. Wszystkie pary w zdefiniowanym zbiorze są liczone; histogramy mają przedziały 1°.")
table(["Blok","Kąt start -> 55k","Spadek >1°","Wzrost >1°","|kąt-90°| start -> 55k"],[
 [r["block"],f'{r["angles"]["persistent"]["mean_angle_start"]:.2f}° -> {r["angles"]["persistent"]["mean_angle_best"]:.2f}°',
  f'{100*r["angles"]["persistent"]["decreasing_gt1_fraction"]:.1f}%',
  f'{100*r["angles"]["persistent"]["increasing_gt1_fraction"]:.1f}%',
  f'{r["angles"]["persistent"]["mean_distance_from90_start"]:.2f}° -> {r["angles"]["persistent"]["mean_distance_from90_best"]:.2f}°'] for r in S["blocks"]
],[36,132,90,90,147])
p("<b>Oba kierunki zmian są częste.</b> Średnia maleje o około 0,44-2,70°, ale część par się oddala. Rosnąca średnia odległość od 90° oznacza szerszy rozkład: więcej par jest bliższych zgodności, a inne bardziej przeciwne. Nie oznacza to, że wszystkie kierunki zbliżają się do 180°.")
p("Najsilniejszy spadek średniej występuje w bloku 4: około 89,98° -> 87,28°. Jednocześnie odległość od 90° rośnie z 3,31° do 8,79°. Ta różnica między średnią a rozrzutem jest głównym powodem pokazania pełnych rozkładów.")
p("Kontrole progów 1% i 10% zachowują kierunek zmiany średniej, lecz zmieniają jej wielkość i skład grupy. Osobne krzywe pokazują same wagi oraz wagi ze znakami aktywacji zamrożonymi z początku. Zmiana rzeczywistych kątów obejmuje oba składniki.")

heading("3. Jak to się ma do IoU, score i epok?")
p("Jeden krok to aktualizacja modelu na batchu 32 obrazów. Sampler przechodzi przez kolejne losowe permutacje 35 000 obrazów, bez odrzucania resztek batcha. Epoka na wykresach jest zatem przeliczeniem liczby obejrzanych przykładów, a nie dodatkową etykietą checkpointu.")
table(["Krok","Odpowiednik epok","Średnie IoU","Średni score"],[
 [f'{r["step"]:,}'.replace(","," "),f'{r["step"]*32/35000:.2f}',f'{r["mean_iou"]:.4f}',f'{r["mean_score"]:.4f}'] for r in S["detection"]
],[110,145,120,120])
p("Duża część poprawy IoU następuje już do kroku 3000 (około 2,74 epoki): IoU osiąga około 0,672, wobec około 0,729 w kroku 55k. Koncentracja siły i poszerzanie rozkładu kątów trwają dalej.")
p("IoU mierzy nakładanie ramki tokena z najwyższym logitem z najlepiej pasującą ramką GT. Score jest sigmoidem tego logitu. Wysoki score nie gwarantuje wysokiego IoU. Końcowy score jest niższy niż w kroku 44k, mimo nieco wyższego IoU.")
p("Wybór 55k wynikał z osobnego VAL (średnie IoU około 0,7619). Wynik około 0,7293 tutaj pochodzi z 50 klatek TEST. Te liczby dotyczą różnych zbiorów i nie są sprzeczne.")

heading("4. Pojedynczy token i wiarygodność pomiaru")
p(f'Plansze szczegółowe śledzą również wszystkie 768 wiązek pojedynczego tokena: obraz <b>{META["frame"]}</b>, '
  f'token <b>{META["fixed_token"]}</b>, wiersz {META["token_row"]}, kolumna {META["token_col"]} (indeksy od 0). '
  "Jest to pierwszy obraz z zamrożonej listy i token wybrany przez końcową detekcję. Jego pozycja pozostaje taka sama dla wcześniejszych checkpointów.")
p("Mapa aktywności zachowuje identyfikatory wiązek; kolejność wierszy ustalono raz według końcowej siły. Kolor oznacza przekroczenie progu 5%. Rozrzut siły pokazuje każdą wiązkę oddzielnie, a słupki porównują te same 24 najsilniejsze na końcu ID na początku i końcu.")
p("Dla każdego bloku pokazano trzy pary z tego tokena: największy końcowy kąt, największy spadek i największy wzrost kąta w stałej grupie. Dobór celowo wybiera skrajności. Krzywych tych nie wolno traktować jako reprezentatywnych dla całego modelu.")
p("<b>Kontrole:</b> sumy kontrolne skopiowanych danych porównano z manifestem zakończonej analizy. Obliczenia kątów porównano z bezpośrednim przeliczaniem par na danych syntetycznych i rzeczywistych podzbiorach; maksymalny błąd był mniejszy niż 3 × 10^-14. Zweryfikowano liczbę par, stałość ich tożsamości i zgodność średniej zmiany ze zmianą średnich.")
p("Na wykresach filmy mają równe wagi. Przedziały 95% powstają przez 2000 losowań bootstrap 25 filmów. Nie opisują zmienności między niezależnymi treningami, ponieważ badany jest tylko seed3047. Warunek niezerowości zachował co najmniej 99,9969% wyjściowych instancji wiązek w grupach stałych.")
p("Suma norm i norma sumy są różnymi wielkościami. Sama mała norma sumy wielu wkładów nie dowodzi znoszenia konkretnej pary. Do oceny wpływu na IoU potrzebne są osobne ablacje wiązek.")

heading("5. Pliki i odtworzenie")
table(["Miejsce względem folderu eksperymentu","Zawartość"],[
 ["Opis/","Cel, protokół i ograniczenia badania"],
 ["Kod_eksperymentu/","Konfiguracja i wskazanie istniejącego kodu pomiarów na Edenie"],
 ["Wyniki/Dane/","Oryginalne tabele, histogramy, dodatkowe szczegóły, metryka i kontrole"],
 ["Wyniki/Code_Interpretujacy_Wyniki/","Skrypty odczytu wyników, rysowania i raportowania"],
 ["Wyniki/Wykresy/","GALERIA.html, ATLAS_WYKRESOW.pdf oraz każdy wykres w PNG, SVG i PDF"],
 ["Wyniki/Raport/","Niniejszy raport w HTML i PDF"],
 ["Wyniki/Krótki_Wniosek/","WNIOSEK.txt: skrót głównych ustaleń"],
],[200,295])
p("make_plots.py odczytuje zapisane dane i generuje cały atlas. assemble_atlas.py przygotowuje raport, krótki wniosek, kontrolę plików i archiwum. prepare_details_eden.py czyta zapisane stany na Edenie i eksportuje małe tablice dla pojedynczych tokenów; nie uruchamia treningu.")
p("Historyczny eksperyment źródłowy na Edenie: outputs/beam_survival_angles_training_20261020/run_v1. Zadania 1904199 i 1904200 zakończyły ekstrakcję i analizę. Zadanie 1904672 wyeksportowało szczegóły do atlasu. Pliki źródłowe zachowano bez zmiany.")
p("Oś czasu symlog pokazuje wyraźnie początek treningu i długą końcówkę. Nie ma punktów pomiarowych pomiędzy zapisanymi checkpointami. Linie są wyłącznie łącznikami; mapy checkpointów używają równych kolumn lub wierszy i nie przedstawiają równych odstępów czasu.")
p("Własny opis użytkownika i zasady organizacji eksperymentów są zapisane w AGENTS.md oraz CLAUDE.md na poziomie projektu i experiments/. Każdy nowy eksperyment wpisuje się do experiments/experiments_index.md. Wykresy, raport i wniosek przygotowano po wyraźnej zgodzie użytkownika.")

def footer(canvas,doc):
    canvas.saveState()
    canvas.setFont("Body",8)
    canvas.setFillColor(colors.HexColor("#68788A"))
    canvas.drawString(42,25,"Wiązki MLP | seed3047 | zapisane checkpointy | 2026-10-20")
    canvas.drawRightString(A4[0]-42,25,str(doc.page))
    canvas.restoreState()
doc=SimpleDocTemplate(str(REPORT/"RAPORT.pdf"),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=43,
    title="Wiązki MLP podczas treningu - wyniki",author="Analiza zapisanych pomiarów")
doc.build(story,onFirstPage=footer,onLaterPages=footer)
page='''<!doctype html><html lang="pl"><meta charset="utf-8"><title>Wiązki MLP - raport</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{font:17px/1.65 system-ui;max-width:980px;margin:40px auto;padding:0 24px;color:#263d52;background:#fff}
h1,h2{color:#18314b;line-height:1.25}h2{margin-top:48px}table{border-collapse:collapse;width:100%;font-size:15px}
td,th{padding:10px;text-align:left;border-bottom:1px solid #dbe3eb}th{background:#e2ebf3}tr:nth-child(even){background:#f3f6f9}
a{color:#0072b2}nav{padding:18px;background:#eef3f7;border-radius:8px}</style>
<nav><a href="../Wykresy/GALERIA.html">Galeria wykresów</a> · <a href="RAPORT.pdf">Raport PDF</a> · <a href="../Krótki_Wniosek/WNIOSEK.txt">Krótki wniosek</a></nav>'''
(REPORT/"RAPORT.html").write_text(page+"\n".join(webpage)+"</html>",encoding="utf-8")
short="""WNIOSKI: WIĄZKI MLP PODCZAS TRENINGU

1. Siła skupia się w mniejszej liczbie dominujących wiązek we wszystkich 6 blokach.
2. Spadek poniżej 5% maksimum nie oznacza wyzerowania. Wiele takich wiązek
   nadal działa; część nawet zwiększa siłę. Średnio tylko 0,01-3,31 wiązki
   na token ma dokładnie zerową aktywację w końcowym pomiarze.
3. Kąty tych samych par zmieniają się w obie strony. Średnia lekko maleje,
   ale rozkład się poszerza. Największy spadek średniej: blok 4, 89,98° -> 87,28°.
4. IoU na 50 klatkach TEST rośnie do 0,7293; score wynosi 0,8686.
   55k to około 50,29 epoki (35 000 obrazów TRAIN, batch 32).
5. To obserwacja jednego treningu. Nie dowodzi, że te zmiany powodują poprawę
   IoU ani że słabsze wiązki można bezkarnie usunąć.
"""
(SHORT/"WNIOSEK.txt").write_text(short,encoding="utf-8")
atlas=PdfReader(ROOT/"Wykresy/ATLAS_WYKRESOW.pdf")
report=PdfReader(REPORT/"RAPORT.pdf")
assert len(atlas.pages)==len(INDEX)
assert all((ROOT/"Wykresy"/(r["stem"]+"."+ext)).is_file() for r in INDEX for ext in ["png","svg","pdf"])
assert all(len(p.extract_text() or "")>50 for p in report.pages)
manifest={p.relative_to(EXP).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
          for folder in [EXP/"Opis",ROOT/"Wykresy",REPORT,SHORT,ROOT/"Code_Interpretujacy_Wyniki"]
          for p in folder.rglob("*") if p.is_file() and p.suffix.lower()!=".zip"}
(DATA/"DELIVERY_MANIFEST.json").write_text(json.dumps(dict(status="PASS",plots=len(INDEX),atlas_pages=len(atlas.pages),
    report_pages=len(report.pages),files=manifest),ensure_ascii=False,indent=2),encoding="utf-8")
with zipfile.ZipFile(REPORT/"PELNY_ZESTAW.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in EXP.rglob("*"):
        if p.is_file() and p.suffix!=".zip" and "__pycache__" not in p.parts and "_QA" not in p.parts:
            z.write(p,p.relative_to(EXP))
print(json.dumps(dict(status="PASS",plots=len(INDEX),atlas_pages=len(atlas.pages),report_pages=len(report.pages)),ensure_ascii=False))
