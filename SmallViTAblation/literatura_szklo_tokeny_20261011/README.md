# Literatura: ruch tokenów małego ViT jak cząstek w szkle

Zbiór PDF-ów do badania „ruch tokenów małego ViT jak cząstki w szkle" (cage rattling,
heterogeniczność dynamiczna, wspólny ośrodek, pułapki, NTK). Wyniki tego badania są w
`SmallViTAblation\outputs\porownanie_szklo_20261010\README.md`, sekcje 6b i 6c.

Założony 2026-10-11 wspólnymi narzędziami `F:\Studia\Articles\Zbiory\_Narzedzia\PdfDownloader`
(`nowy_zbior.py`). Zbiór jest samodzielny: PDF-y leżą TUTAJ w `PDF_all\`, bez odsiewania po innych
katalogach (`skip_dirs` w `config\config.json` wskazuje tylko własny `PDF_all`).

## Skąd listy

Dwie listy linków skopiowane bez zmian z `F:\Studia\Articles\Zbiory\JEPA\Scripts\PdfDownloader\linki\`
do `linki\`:

| lista | linków | pobrano | nieudane |
|---|---|---|---|
| `szklo-cage-rattling-2026-10-10.txt` | 46 | 39 | 7 |
| `tokeny-pulapki-wspolny-osrodek-ntk-2026-10-11.txt` | 56 | 27 | 29 |
| razem | 102 | 66 | 36 |

Pobranie: `py -3.11 pobierz_pdfy.py --zbior <ten katalog> --watki 8` (Unpaywall + arXiv + preprinty),
czas około 2 minut. Pełny log: `pobieranie.log`.

## Nieudane (36 pozycji, głównie wydawcy z paywallem: APS, AIP, Elsevier, Science, Wiley)

Klikalny raport: `nieudane\nieudane_linki.html` (stan maszynowy: `nieudane\nieudane_stan.json`).
Ponowienie samych nieudanych: `py -3.11 pobierz_pdfy.py --zbior <ten katalog> --nieudane`.
Dostęp instytucjonalny (Primo) wymaga logowania, więc nie został użyty; te prace trzeba
pobrać ręcznie lub przez Primo (`scripts\pobieranie\odswiez_kolejke.bat`).

- `10.1126/science.287.5453.627` — Weeks i in. 2000, Science, pierwszy obraz 3D relaksacji strukturalnej przy przejsciu szklistym
- `10.1126/science.287.5451.290` — Kegel, van Blaaderen 2000, Science, bezposrednia obserwacja heterogenicznosci dynamicznej
- `10.1146/annurev.physchem.51.1.99` — Ediger 2000, Annu. Rev. Phys. Chem., przestrzennie heterogeniczna dynamika cieczy przechlodzonych
- `10.1063/1.1605094` — Lacevic i in. 2003, JCP, czteropunktowa funkcja korelacji chi4 i dlugosc korelacji
- `10.1103/PhysRevLett.110.025501` — Warren, Rottler 2013, PRL, zamrozenie, rownowagowanie i podstarzenie w szklach strukturalnych
- `10.1103/PhysRevE.98.022134` — Cherstvy i in. 2018, PRE, srednie czasowe procesu OU i ich rozrzut (kontrola OU na czastke)
- `10.1103/PhysRevA.44.1169` — Loewen, Hansen, Roux 1991, PRA, dynamika Browna i kinetyczne przejscie szkliste w koloidach (wzor modelu BD)
- `10.1103/PhysRevLett.85.888` — Crocker i in. 2000, PRL, mikroreologia dwupunktowa niejednorodnych materialow miekkich
- `10.1103/PhysRevLett.100.146001` — Larsen, Furst 2008, PRL, mikroreologia przejscia ciecz-cialo stale przy zelowaniu (krzywa wzorcowa MSD)
- `10.1103/PhysRevLett.84.2275` — Cipelletti i in. 2000, PRL, uniwersalne starzenie zeli fraktalnych; eksponens scisniety beta > 1
- `10.1103/PhysRevLett.96.088101` — Polin, Grier, Quake 2006, PRL, mody zbiorowe szeregu kul w pulapkach holograficznych
- `10.1103/PhysRevLett.79.2574` — Segre, Herbolzheimer, Chaikin 1997, PRL, dalekozasiegowe korelacje predkosci w sedymentacji
- `10.1146/annurev.fluid.40.111406.102224` — Salazar, Collins 2009, Annu. Rev. Fluid Mech., dyspersja dwuczastkowa w turbulencji izotropowej
- `10.1209/0295-5075/28/7/011` — Castillo, Goldbart, Zippelius 1994, EPL, rozklad dlugosci lokalizacji przy wulkanizacji
- `10.1103/PhysRevLett.85.3656` — Zahn, Maret 2000, PRL, parametr Lindemanna z przesuniec wzglednych sasiadow w 2D
- `10.1034/j.1600-0706.2000.890119.x` — Ripa 2000, Oikos, efekt Morana a rozproszenie
- `10.1103/PhysRevLett.61.2620` — Martin, Adolf, Wilcoxon 1988, PRL, lepkosprezystosc zeli bliskich punktu krytycznego
- `10.1111/j.0030-1299.2007.15856.x` — Abbott 2007, Oikos, sam spadek synchronii z odlegloscia nie rozpoznaje efektu Morana
- `10.1111/j.0030-1299.2007.16101.x` — Vasseur 2007, Oikos, szum autokorelowany wzmacnia efekt Morana (ped AdamW)
- `10.1103/PhysRevLett.93.135701` — Widmer-Cooper, Harrowell, Fynewever 2004, PRL, zespol izokonfiguracyjny i propensity
- `10.1111/j.2006.0030-1299.14686.x` — Hugueny 2006, Oikos, twierdzenie Morana dla roznych dynamik: czynnik demograficzny x korelacja szumu
- `10.1038/nphys788` — Larini i in. 2008, Nat. Phys., tau_alpha ~ exp(a^2 / 2<u^2>)
- `10.1039/c2sm07445a` — Henkes i in. 2012, Soft Matter, mody drgan z kowariancji przesuniec
- `10.1086/303240` — Lande, Engen, Saether 1999, Am. Nat., skala przestrzenna synchronii: srodowisko kontra rozproszenie
- `10.1038/337754a0` — Doster, Cusack, Petry 1989, Nature, przejscie dynamiczne mioglobiny (harmoniczne cialo stale ponizej 180 K)
- `10.1016/S1359-0278(97)00024-2` — Bahar, Atilgan, Erman 1997, Fold. Des., model sieci gaussowskiej GNM
- `10.1006/jmbi.1998.1807` — Abseher, Nilges 1998, JMB, czy sa nietrywialne korelacje krzyzowe w bialkach (usuwanie ruchu globalnego)
- `10.1103/PhysRevLett.113.245702` — Lemaitre 2014, PRL, relaksacja strukturalna jako proces bezskalowy; korelacje maksymalne przy tau_alpha
- `10.1103/PhysRevLett.62.1916` — Iben i in. 1989, PRL, szkliste zachowanie bialka
- `10.1126/science.1121726` — Bourgoin i in. 2006, Science, rola dyspersji par w przeplywie turbulentnym (Batchelor, Richardson t^3)
- `10.1103/PhysRevLett.111.066001` — Chattoraj, Lemaitre 2013, PRL, sprezysty slad zdarzen plyniecia przy scinaniu
- `10.1002/prot.340110305` — Ichiye, Karplus 1991, Proteins, ruchy zbiorowe w bialkach
- `10.1051/jp1:1992238` — Bouchaud 1992, J. Phys. I, starzenie w modelu pulapek
- `10.1086/167771` — Gorski i in. 1989, ApJ, korelacje predkosci kosmologicznych
- `10.1038/s41593-022-01225-z` — Qin i in. 2023, Nat. Neurosci., skoordynowany dryf pol recepcyjnych przy zaszumionym uczeniu
- `10.1016/j.cell.2017.07.021` — Driscoll i in. 2017, Cell, dryf reprezentacji w korze ciemieniowej

## Dopisek 2026-10-12: nowość publikacyjna

Lista `linki/nowosc-tokeny-czastki-czas-uczenia-2026-10-12.txt` (62 linki, deep research „czy ktoś już robił taki układ”).
Raport: `outputs/porownanie_szklo_20261010/NOWOSC_PUBLIKACYJNA_2026-10-12.md`. Pobrano 54, 2 już były, 6 nieudanych
(log `pobieranie_nowosc_2026-10-12.log`):

- `10.1103/physreve.111.045310` — Avidan i in. 2025, PRE, faza gradientowa potem dyfuzyjna
- `10.1039/d0sm01686a` — Soft Matter 2021, przegląd: brak dynamiki szklistej w typowych architekturach
- `10.1073/pnas.1919995117` — Verpoort i in. 2020, PNAS, archetypowe krajobrazy (brak wysokich barier)
- `10.59400/cai4258` — Rahman, Ottoborgo 2025, dryf wyjaśnień w czasie uczenia
- `10.1007/s44163-026-01593-6` — La Quang i in. 2026, Discover AI, taksonomia stanów uczenia
- `10.2312/eurovisshort.20161164` — Rauber i in. 2016, EuroVis, dynamiczny t-SNE

## Układ katalogu

`PDF_all\` pobrane PDF-y, `linki\` listy wejściowe, `nieudane\` raport nieudanych, `config\config.json`
ustawienia, `primo\` i `doi\` kolejka dostępu instytucjonalnego (pusta), `scripts\` cienkie `.bat`-y
wołające wspólne narzędzia. Nie wykonano OCR ani indeksowania RAG.
