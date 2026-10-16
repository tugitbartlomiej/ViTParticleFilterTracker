from ruch_a_jakosc import *

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    c=pd.read_csv(OUT/'korelacje.csv');m=pd.read_csv(OUT/'porownania_dopasowane.csv');labels=json.loads((OUT/'ETYKIETY_I_KONTROLE.json').read_text())
    eq=[]
    for metric,label in zip(PRIMARY,LABELS):
        a=m[(m.metric==metric)&(m.radius==1000)]
        eq.append([label]+[f'{a[a.image==i].median_paired_pct.iloc[0]:+.2f}%' for i in [1,2]])
    corrrows=[]
    for metric,label in zip(PRIMARY,LABELS):
        a=c[(c.metric==metric)&(c.knots==6)]
        corrrows.append([label]+[f'{a[(a.image==i)&(a.target==target)].residual.iloc[0]:+.3f}' for target in ['iou','confidence'] for i in [1,2,3]])
    report='''# Co robią cząsteczki, gdy ramka i pewność są dobre?

Analiza zapisanych historii, 15 października 2026. Nie zmieniano aplikacji ani checkpointów.

**Najbliższą zaobserwowaną wskazówką jest nieco mniejsze odchylenie cząsteczek obszaru ramki od ich wygładzonej trajektorii. Nie znaleźliśmy jednak jednoznacznego zachowania, po którym można rozpoznać jednocześnie wysokie IoU i wysoką pewność.** Na dwóch obrazach z dobrą detekcją różnica prędkości cząsteczek między pobliskimi dobrymi i gorszymi checkpointami jest mała. Ruch „pułapek” nie daje zgodnego wzorca. Trzeci obraz pokazuje, że silniejsze współdziałanie aktywnych wkładów może towarzyszyć większej pewności bez wysokiej jakości lokalizacji.

To wynik pomiaru konkretnych historii, a nie twierdzenie, że geometria reprezentacji nigdy nie zawiera informacji o jakości. Pomiar nie obejmuje wszystkich możliwych cech ani innych warstw sieci.

## Co porównaliśmy

Trzy niezmienione obrazy RGB, jeden trening, 647 stanów od kroku 1000 do 65600 co 100 kroków. Jedna epoka to 1094 kroki. Wszystkie ruchy policzono względem dokładnie 100 kroków wcześniej i przypisano końcowi przedziału. Nie używano różnic po indeksach nieregularnych zapisów. To dynamika w czasie uczenia, nie przejście tokenu przez blok w jednym forwardzie.

„Dobra detekcja” oznacza **IoU >= 0,8 oraz pewność >= 0,8 jednocześnie**. Ten próg spełnia 552/647 stanów obrazu 1, 573/647 obrazu 2 i 0/647 obrazu 3. Przy kontroli progu 0,75 liczby wynoszą 596, 610 i 4; przy 0,85: 412, 443 i 0. Nie zmienialiśmy progu, aby uzyskać dodatnie przykłady na trzecim obrazie. Pewność to wynik modelu, nie zweryfikowane prawdopodobieństwo poprawnej ramki.

W analizie głównej użyto ośmiu stałych tokenów najbliższych środkowi GT według odległości znormalizowanej rozmiarem ramki i ośmiu stałych tokenów tła odległych ponad 32 piksele od GT. Ten wybór nie oznacza segmentacji. Sprawdzono położenia: we wszystkich trzech obrazach osiem wybranych środków tokenów centrum rzeczywiście znajduje się wewnątrz GT, a osiem tła spełnia warunek odległości. Tokeny centrum pozostają te same nawet wtedy, gdy detektor wybierze inny zwycięski token.

Dodatkową analizę wszystkich tokenów w bbox i dalekiego tła dodano na doprecyzowanie użytkownika po obliczeniach głównych; jest eksploracyjna. Środki wewnątrz bbox: 18, 20 i 72 tokeny. Dalekie tło: 526, 523 i 444 tokeny. Ramka może zawierać piksele tła, więc poprawna nazwa to **obszar bbox narzędzia**, a nie segmentacja samego narzędzia.

## Co naprawdę oznaczają ruch i pułapki

- Prędkość centrum: średnia długość zmiany wektora residual-pre ostatniego MLP w **192 wymiarach**, podzielona przez 100 kroków. Wartości podajemy w jednostkach współrzędnych reprezentacji na krok; nie są prędkością fizyczną. Warstwa ukryta MLP ma 768 aktywacji.
- Stosunek prędkości centrum do tła: odniesienie do ruchu w tym samym obrazie i przedziale czasu.
- Zgodność ruchu: `||Σ Δx_i|| / Σ ||Δx_i||`, od zera do jedności. Duża wartość oznacza zgodny ruch grupy w tym przedziale, nie poprawną ramkę ani wzajemną siłę między tokenami.
- `vib`: odległość punktu od wygładzonej trajektorii w aplikacji, w rzucie 2D po Prokrustesie. `wezly` są średnią aktywacji w oknie **±300 kroków**, a ich rzut wykorzystuje bazę dopasowaną do całej historii. Nie są wyznaczonym minimum potencjału ani zmierzoną pułapką optyczną. Zawierają przyszłe stany, więc ten pomiar jest retrospektywny.
- Prędkość węzłów: zmiana tych wygładzonych punktów w 3D aplikacji na krok. Jednostka wizualizacji wynika z jej skali, nie z metrów.
- Aktywne η: `||Σ u_c||² / Σ ||u_c||²`, gdzie `u_c=z_c v_c`, bez biasu. Mierzy stosunek normy sumy wkładów do sumy kwadratów ich norm. Nie jest udziałem reflektorów z przeciwnym partnerem, nie jest też energią światła. Dotyczy działania MLP w obrębie checkpointu; nie jest siłą wywołującą ruch między checkpointami.

## Dobre i gorsze checkpointy blisko siebie w czasie

Dobieraliśmy stany z tego samego obrazu w odległości do 1000 kroków, bez ponownego użycia checkpointu. Algorytm Hungarian maksymalizuje liczbę par, następnie minimalizuje ich łączną odległość czasową. Otrzymano **64 pary na obrazie 1 i 56 na obrazie 2**, średnio odległe o 262,5 i 242,9 kroku. Nie każdy zły stan ma dobrego sąsiada; te porównania dotyczą obszaru wspólnego wsparcia czasowego, nie całego treningu.

Poniżej mediana procentowej różnicy wewnątrz pary: `(dobra − gorsza) / gorsza`. To mediana ilorazów, nie iloraz median.

'''+table(['Miara','Obraz 1','Obraz 2'],eq)+'''

Prędkość centrum różni się o około −1,9% oraz +0,05%. Wysoka jakość nie oznacza zatem, że cząsteczki muszą nagle zastygnąć. Zgodność kierunków oraz prędkość węzłów zmieniają znak między obrazami.

Największa zgodna różnica to niższe `vib`. Ale na obrazie 1 jest ono niższe tylko w **59,4%** par, a na obrazie 2 w **57,1%**. To słaba wskazówka, nie kryterium rozpoznawania jakości. Po rozszerzeniu dopuszczalnego odstępu do 2000 kroków otrzymujemy 70/63 pary, a mediana różnicy `vib` zmienia się na −11,3%/−5,3%. Wrażliwość wielkości efektu na sposób dopasowania jest istotna.

### Uwzględnienie etykiety bbox i całego tła

Dla wszystkich tokenów ze środkiem w bbox mediana `vib` jest w dobrych stanach niższa o **17,3% i 11,6%**. Dla dalekiego tła odpowiednio −3,6% i +3,1%. Stosunek `vib` w bbox do `vib` tła spada o **14,6% i 13,9%**, lecz dzieje się to tylko w **59,4% i 60,7%** par. To najbardziej spójny opis wizualny w tym pilocie: przy dobrej detekcji okolica narzędzia bywa mniej odchylona od swojej wygładzonej trajektorii względem tła. Nie jest to niezawodny znak i nie dowodzi fizycznego uwięzienia.

Prędkość wszystkich węzłów bbox zmienia się o −4,8% i −2,6%, ale prędkość cząsteczek bbox w rzucie residual-pre 3D zmienia się o +5,4% i −5,4%. Ta ostatnia projekcja odejmuje środek wszystkich tokenów w każdym checkpointcie i traci większość kierunków pełnej przestrzeni. Nie należy utożsamiać jej z normą ruchu w 192D.

## Czy to tylko trend treningu?

Dla każdej miary oraz IoU/pewności osobno odjęto sześcienny spline czasu z sześcioma węzłami. Poniżej korelacje Pearsona pozostałych odchyleń. Są to powiązania współwystępowania, nie dowód mechanizmu ani sprawdzony predyktor.

'''+table(['Miara','IoU 1','IoU 2','IoU 3','Pewność 1','Pewność 2','Pewność 3'],corrrows)+'''

Na obrazie 1 mniejsza prędkość centrum towarzyszy lepszemu IoU z korelacją około **−0,254**. Kontrola 4/10 węzłów daje od −0,266 do −0,252. Orientacyjny 95-procentowy przedział bootstrapu blokowego reszt wynosi [−0,343; −0,141] dla bloków 20 stanów i [−0,311; −0,143] dla bloków 50 stanów. Na obrazie 2 związek jest słabszy (−0,078), a na trzecim praktycznie nie występuje (−0,014). Nie jest to uniwersalna sygnatura.

Korelacja przyrostów prędkości centrum i IoU wynosi −0,150, +0,005 i −0,006. Wyniki pełne, Spearman reszt i kontrole węzłów są w CSV. Nie stosowano p-values zakładających niezależność checkpointów. Bootstrap jest orientacyjny i warunkowy względem dopasowanego trendu; nie oddaje niepewności między treningami.

Po odniesieniu `vib` całego bbox do tła korelacje po usunięciu czasu z IoU wynoszą −0,137, −0,063 i +0,047; z pewnością −0,085, −0,037 i +0,059. Doprecyzowanie etykiet nie zmienia wniosku o ograniczonej sile wskaźnika.

### Szczególnie pouczający trzeci obraz

Aktywne η wiąże się z pewnością z korelacją **+0,353** po usunięciu trendu (kontrola 4/10 węzłów: +0,320 do +0,353; przyrosty: +0,275). Orientacyjne przedziały blokowe: [0,270; 0,439] i [0,238; 0,470]. Związek η z IoU jest znacznie słabszy: +0,132, a jego przedziały obejmują zero.

Mimo tego ani jeden z analizowanych stanów trzeciego obrazu nie osiąga jednocześnie progu 0,8 dla obu wyników. W **57 checkpointach pewność przekracza 0,8, a IoU jest poniżej 0,5**. Wysoka pewność i dobre pokrycie ramki są rozdzielnymi własnościami odczytu. Nie twierdzimy, że η przyczynowo zwiększa pewność.

## Ilustracja konkretnego przejścia

Na obrazie 1 para kroków **10400 i 10500** ma odstęp 100 kroków. Wybrano ją spośród dopasowanych par o minimalnym odstępie, przy remisie wybierając największą różnicę IoU; jest więc ilustracją widocznej zmiany, a nie reprezentatywnym efektem średnim.

Przy kroku 10400 IoU=0,930 i pewność=0,905. Przy 10500 IoU spada do 0,675, podczas gdy pewność nadal wynosi 0,879. Prędkość centrum wzrasta z 0,125 do 0,169 jednostki/krok, a `vib` z 0,392 do 0,718. Ten przykład pokazuje pogorszenie ramki przy silniejszym ruchu, lecz nie usuwa licznych wyjątków widocznych w analizie wszystkich par.

## Co można powiedzieć użytkownikowi mikroskopu

Na tych dwóch zdjęciach dobra detekcja bywa związana z mniejszym odchyleniem cząsteczek okolicy narzędzia od wygładzonego toru, także względem tła. Nie występuje natomiast charakterystyczne zatrzymanie całego układu ani stabilna zmiana kierunków ruchu. W ruchu są słabe wskazówki jakości, ale nie znaleźliśmy rozpoznawalnego, pewnego odpowiednika „cząsteczka złapana, więc ramka jest dobra”.

Poprzednie ablacje wykazały, że wkłady neuronów mogą się kompensować i razem przełączyć zwycięską detekcję. Obecna analiza dotyczy innej kwestii: czy naturalny przebieg ruchu daje widoczną sygnaturę jakości. Ten pilot nie wykazał, że któraś z sześciu miar jest wystarczająca. Dalszy eksperyment powinien badać ruch w kierunkach rzeczywiście wpływających na cztery krawędzie i konkurencję detekcji oraz sprawdzić wskaźnik na nowych filmach i treningach.

## Zakres, ograniczenia i kontrola

- Jeden trening, trzy obrazy, z czego tylko dwa dają liczne stany dobrej jakości. Nie potwierdzono niezależności filmów. 647 checkpointów nie oznacza 647 niezależnych prób naukowych.
- Definicja jakości używa GT, więc analiza nie jest metodą samodzielnej oceny bez etykiet. Nie oceniono skuteczności klasyfikatora jakości na odłożonych filmach.
- Czas jest kontrolowany dopasowaniem i spline, ale nie usuwa to wszystkich możliwych wspólnych przyczyn. Nie wykonano interwencji w tej analizie.
- Wizualne miary aplikacji zależą od projekcji, Prokrustesa i przyszłych stanów. Nie nazywamy ich temperaturą, siłą ani sztywnością fizycznej pułapki.
- Miary 192D są liczone w stałych współrzędnych sieci; nie usuwano zmian współrzędnych poprzez dopasowanie obrotów między checkpointami.
- Sprawdzono dokładne kroki, zgodność obrazów, rozmiary tablic, zakres współczynnika zgodności, brak zerowych mianowników i brak nieskończonych wartości. Niezależne odtworzenie IoU z ramek daje maksymalny błąd <4,8e−7; niezależne odtworzenie wybranych prędkości <8,4e−17. Brak par obrazu 3 zapisano jako brak pomiaru, nie zero efektu.
- Nie powtarzano forwardów ani treningu; użyto istniejących pomiarów float32 i zapisów aplikacji.

## Pliki i odtwarzanie

- `przebiegi.png`: jakość i sześć miar dla trzech obrazów.
- `korelacje_po_czasie.png`: korelacje po kontroli czasu.
- `maski_obszarow.png`: sprawdzone tokeny obszaru bbox i dalekiego tła.
- `przyklad_pary.png`: ilustracja GT i detekcji w krokach 10400/10500.
- `historia.csv`, `porownania_dopasowane.csv`, `pary_checkpointow.csv`, `korelacje.csv`, `progi.csv`: pełne dane główne.
- `historia_etykiety.csv`, `porownania_etykiety.csv`, `korelacje_etykiety.csv`: eksploracyjna analiza bbox/tła.
- `MANIFEST.json`: wejścia i kod analizy głównej z hashami; `MANIFEST_UZUPELNIENIE.json`: dodatkowy kod i wyniki.
- `WERYFIKACJA.json`, `ETYKIETY_I_KONTROLE.json`: kontrole i maski tokenów.

Z katalogu SmallViTAblation:

```powershell
python experiments/wspoldzialanie_neuronow_20261015/ruch_a_jakosc.py
python experiments/wspoldzialanie_neuronow_20261015/ruch_a_jakosc_etykiety.py
python experiments/wspoldzialanie_neuronow_20261015/ruch_a_jakosc_raport.py
```
'''
    (OUT/'RAPORT.md').write_text(report,encoding='utf-8')
    files=[Path(__file__),Path(__file__).with_name('ruch_a_jakosc_etykiety.py')]+[f for f in OUT.iterdir() if f.is_file() and not f.name.startswith('MANIFEST')]
    (OUT/'MANIFEST_UZUPELNIENIE.json').write_text(json.dumps([dict(path=str(f),size=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],indent=2),encoding='utf-8')
    print('Raport zapisany:',OUT/'RAPORT.md')

if __name__=='__main__':main()
