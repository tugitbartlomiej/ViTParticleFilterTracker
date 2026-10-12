"""Film ManimGL: jak wyznaczono cząsteczki i kierunki tożsamości narzędzia (pary 400, LaMa).

Bez lektora: każdy krok ma napis na ekranie i czas na przeczytanie. Wszystkie liczby i punkty
pochodzą z assets/dane.json (przygotuj.py), czyli z zapisanych wyników eksperymentu
experiments/maly_vit_tozsamosc_czasteczki_pary_400. Siatka, ramki i strzałki to nakładka rysunkowa.

    python render.py sceny.py Scena1 ... Scena9 -w --hd   (FILM_CHECK=1 skraca pauzy)
"""
from pathlib import Path
import json
import os
import textwrap

import numpy as np
from manimlib import *

HERE = Path(__file__).resolve().parent
A = HERE / 'assets'
D = json.loads((A / 'dane.json').read_text(encoding='utf-8'))
CHECK = bool(os.getenv('FILM_CHECK'))
FG = '#edf4ff'; MUTED = '#9aadc7'; CYAN = '#50cef5'; GREEN = '#50e0a6'; ORANGE = '#ffbd66'
PURPLE = '#b298ff'; RED = '#ff7187'; YELLOW = '#ffe066'; PANEL = '#1b2a41'
TOOL = [int(t) for t in D['token_ids']]           # numery od 0
COV = D['pokrycie']
C280 = D['czastka']['numer'] - 1                  # 279


def pl(x, n=1):
    return f'{x:.{n}f}'.replace('.', ',')


def text(s, size=24, color=FG, width=None):
    m = Text(str(s), font='Arial', font_size=size, fill_color=color)
    if width and m.get_width() > width:
        m.set_width(width)
    return m


def label(s, pos, size=24, color=FG, width=None):
    return text(s, size, color, width).move_to(np.array(pos, float))


def left_label(s, x, y, size=20, color=FG, width=None):
    m = text(s, size, color, width)
    return m.move_to([x + m.get_width() / 2, y, 0])


def card(s, pos, w=2.5, h=.85, color=CYAN, size=22):
    r = RoundedRectangle(width=w, height=h, corner_radius=.12).set_stroke(color, 1.7).set_fill(color, .09).move_to(pos)
    words = VGroup(*[text(t, size, color, w - .25) for t in str(s).split('\n')]).arrange(DOWN, buff=.08).move_to(pos)
    return VGroup(r, words)


def image(name, cx, cy, w):
    return ImageMobject(str(A / name)).set_width(w).move_to([cx, cy, 0])


def grid_geometry(cx, cy, w):
    """Siatka 30 x 20 pól nałożona na obraz o szerokości w (proporcje 3:2)."""
    h = w * 2 / 3
    x0, y0, cell = cx - w / 2, cy + h / 2, w / 30

    def center(j):
        return np.array([x0 + (j % 30 + .5) * cell, y0 - (j // 30 + .5) * cell, 0])
    return x0, y0, cell, h, center


def grid_lines(cx, cy, w, color=CYAN, opacity=.45):
    x0, y0, cell, h, _ = grid_geometry(cx, cy, w)
    return VGroup(*[Line([x0 + i * cell, y0, 0], [x0 + i * cell, y0 - h, 0]).set_stroke(color, .6, opacity) for i in range(31)],
                  *[Line([x0, y0 - i * cell, 0], [x0 + w, y0 - i * cell, 0]).set_stroke(color, .6, opacity) for i in range(21)])


def cell_box(j, cx, cy, w, color=WHITE, width=3.0):
    _, _, cell, _, center = grid_geometry(cx, cy, w)
    return Square(side_length=cell).set_stroke(color, width).set_fill(opacity=0).move_to(center(j))


def bars_path(values, x0, x1, base, scale, color, width=1.6):
    """768 słupków jako jedna ścieżka: szybciej niż 768 osobnych linii."""
    m = VMobject()
    xs = np.linspace(x0, x1, len(values))
    for x, v in zip(xs, values):
        m.start_new_path(np.array([x, base, 0]))
        m.add_line_to(np.array([x, base + v * scale + (1e-3 if v == 0 else 0), 0]))
    return m.set_stroke(color, width)


class Ujecie(Scene):
    numer = 0
    tytul = ''
    tryb = ''

    def naglowek(self):
        self.add(label(self.tytul, [0, 3.38, 0], 34, FG, 12.6),
                 label(f'{self.numer} / 9', [-6.55, 3.85, 0], 15, MUTED),
                 label('Mały ViT, ziarno 47, krok 1000  •  400 par: oryginał i LaMa  •  '
                       'experiments/maly_vit_tozsamosc_czasteczki_pary_400', [0, -3.8, 0], 12, MUTED, 13.5))
        self.badge = label(self.tryb, [0, 2.86, 0], 15, ORANGE, 12.5)
        self.add(self.badge)
        self.podpis = None

    def krok(self, napis, *anims, run_time=1.6, tryb=None, hold=0.0, then=()):
        lines = textwrap.wrap(napis, width=92, break_long_words=False)
        new = VGroup(*[text(t, 25, FG, 13.2) for t in lines]).arrange(DOWN, buff=.12).move_to([0, -2.93, 0])
        swap = [FadeIn(new)]
        if self.podpis is not None:
            swap.append(FadeOut(self.podpis))
        if tryb:
            nb = label(tryb, [0, 2.86, 0], 15, ORANGE, 12.5)
            swap += [FadeOut(self.badge), FadeIn(nb)]
            self.badge = nb
        self.play(*swap, run_time=.45)
        self.podpis = new
        if anims:
            self.play(*anims, run_time=.4 if CHECK else run_time)
        if then:
            self.play(*then, run_time=.4 if CHECK else run_time)
        self.wait(.15 if CHECK else max(3.0, .36 * len(napis.split()) + 1.2) + hold)


class Scena1(Ujecie):
    numer = 1
    tytul = 'Para zdjęć: z narzędziem i bez niego'
    tryb = 'PRAWDZIWE ZDJĘCIA  •  Cataract-LMM, film TR_0053, klatka 881 (film odłożony do testu)'

    def construct(self):
        self.naglowek()
        W, L, R = 6.3, (-3.4, .45), (3.4, .45)
        a, b = image('obraz_oryginal.png', *L, W), image('obraz_lama.png', *R, W)
        la = label('oryginał: narzędzie jest', [L[0], -1.98, 0], 20, GREEN)
        lb = label('LaMa: narzędzie wymazane', [R[0], -1.98, 0], 20, ORANGE)
        self.krok('To prawdziwa klatka z operacji zaćmy. Narzędzie to cystotom: cienka igła, która wchodzi do oka od lewej.',
                  FadeIn(a), FadeIn(la))
        self.krok('Program LaMa wymazał igłę i domalował w jej miejscu tło. Poza miejscem igły oba obrazy są identyczne co do piksela.',
                  FadeIn(b), FadeIn(lb))
        m = image('maska.png', *L, W)
        self.krok('Zielona maska to obrys igły z ręcznej adnotacji. Takich par mamy 400, ze 116 filmów.', FadeIn(m))
        self.krok('Obrazy różnią się tylko tam, gdzie była igła. Każdą zmianę wewnątrz sieci łączymy więc z narzędziem.',
                  FadeOut(m))
        c = image('obraz_lama.png', *L, W)
        for _ in range(1 if CHECK else 2):
            self.play(FadeIn(c), run_time=.5)
            self.wait(.5)
            self.play(FadeOut(c), run_time=.5)
            self.wait(.5)
        self.wait(.5)


class Scena2(Ujecie):
    numer = 2
    tytul = 'Pola obrazu to cząsteczki'
    tryb = 'SCHEMAT SIATKI na prawdziwym zdjęciu  •  pokrycie pól policzone z maski'

    def construct(self):
        self.naglowek()
        W, cx, cy = 7.2, -2.55, .05
        img = image('obraz_oryginal.png', cx, cy, W)
        grid = grid_lines(cx, cy, W)
        _, _, cell, _, center = grid_geometry(cx, cy, W)
        self.add(img)
        notes = VGroup(label('480 × 320 pikseli', [4.6, 1.7, 0], 26),
                       label('pole 16 × 16', [4.6, 1.1, 0], 26),
                       label('30 × 20 = 600 pól', [4.6, .5, 0], 26, CYAN))
        self.krok('Sieć widzi klatkę zmniejszoną do 480 na 320 pikseli. Dzieli ją na pola 16 na 16: 30 kolumn i 20 wierszy.',
                  ShowCreation(grid), FadeIn(notes[0]), FadeIn(notes[1]), FadeIn(notes[2]))
        n2 = label('pole = cząsteczka', [4.6, -.25, 0], 26, YELLOW)
        probe = Square(side_length=cell).set_stroke(YELLOW, 3).move_to(center(2 * 30 + 4))
        self.krok('Każde pole to jeden token. Traktujemy go jak cząsteczkę: ma stałe miejsce na obrazie i własny stan, czyli zestaw liczb.',
                  FadeIn(n2), FadeIn(probe), run_time=.8, then=[probe.animate.move_to(center(15 * 30 + 24))])
        boxes = VGroup()
        for j, cv in zip(TOOL, COV):
            boxes.add(Square(side_length=cell).set_fill(GREEN, .15 + .8 * cv).set_stroke(GREEN, 1.4).move_to(center(j)))
        n3 = label(f'{D["n_czastek_maski"]} cząsteczek pod maską', [4.6, -.95, 0], 26, GREEN)
        self.krok(f'Wybieramy pola, które maska przecina choćby jednym pikselem. Na tej klatce jest ich {D["n_czastek_maski"]}.',
                  FadeOut(probe), LaggedStartMap(FadeIn, boxes, lag_ratio=.08), FadeIn(n3))
        n4 = label(f'udział igły: {round(100 * min(COV))} do {round(100 * max(COV))} %', [4.6, -1.6, 0], 22, MUTED)
        self.krok('Jasność pola to udział igły w polu. Ten udział jest wagą cząsteczki we wszystkich dalszych rachunkach.', FadeIn(n4))
        frame = Square(side_length=cell).set_stroke(WHITE, 3.5).move_to(center(C280))
        big = frame.copy().scale(6).set_stroke(WHITE, 2)
        tag = label('cząsteczka 280', center(C280) + np.array([0, 1.0, 0]), 22, WHITE).add_background_rectangle(opacity=.75)
        self.krok(f'Cząsteczka 280 leży na środku igły: wiersz {D["czastka"]["wiersz"] + 1}, kolumna {D["czastka"]["kolumna"] + 1}. '
                  f'Igła zajmuje {round(100 * D["czastka"]["pokrycie"])} procent jej pola.',
                  TransformFromCopy(big, frame), FadeIn(tag))


class Scena3(Ujecie):
    numer = 3
    tytul = 'Stan cząsteczki: 768 liczb'
    tryb = 'POMIAR  •  zamrożona sieć, aktywacje ostatniego bloku (MLP po GELU), cząsteczka 280'

    def construct(self):
        self.naglowek()
        zo, zn = np.array(D['czastka']['z_o']), np.array(D['czastka']['z_n'])
        dz = zo - zn
        s = .85 / np.abs(np.r_[zo, zn, dz]).max()
        x0, x1 = -2.9, 6.7
        rows = [(1.15, zo, CYAN), (-.05, zn, ORANGE), (-1.45, dz, GREEN)]
        bases = VGroup(*[Line([x0, y, 0], [x1, y, 0]).set_stroke(MUTED, .8) for y, _, _ in rows])
        paths = [bars_path(v, x0, x1, y, s, col) for y, v, col in rows]
        po = Group(image('latka_oryginal.png', -6.2, 1.45, 1.15),
                   Square(side_length=1.15 / 3).set_stroke(WHITE, 2).move_to([-6.2, 1.45, 0]))
        pn = Group(image('latka_lama.png', -6.2, .25, 1.15),
                   Square(side_length=1.15 / 3).set_stroke(WHITE, 2).move_to([-6.2, .25, 0]))
        lo = left_label('oryginał', -5.45, 1.45, 22, CYAN)
        ln = left_label('bez igły', -5.45, .25, 22, ORANGE)
        ld = VGroup(left_label('różnica', -6.75, -1.3, 22, GREEN), left_label('oryginał − LaMa', -6.75, -1.7, 18, GREEN))
        ticks = VGroup(*[label(str(k), [x0 + (x1 - x0) * (k - 1) / 767, .93, 0], 13, MUTED) for k in (1, 256, 512, 768)])
        self.krok('Sieć to mały ViT, czyli transformer, który patrzy na obraz polami. Jest zamrożona po 1000 krokach uczenia od zera.',
                  FadeIn(po), FadeIn(lo))
        self.krok('Każdą cząsteczkę opisuje 768 liczb z ostatniego bloku, tuż przed częścią, która rysuje ramkę. Tu wszystkie 768 dla cząsteczki 280.',
                  ShowCreation(bases[0]), ShowCreation(paths[0]), FadeIn(ticks), run_time=2.4)
        self.krok('Ta sama cząsteczka na obrazie bez igły ma inne liczby. Miejsce jest to samo, zmienił się tylko obraz.',
                  FadeIn(pn), FadeIn(ln), ShowCreation(bases[1]), ShowCreation(paths[1]), run_time=2.4)
        self.krok('Odejmujemy: oryginał minus obraz bez igły. Ta różnica to ślad igły w tej cząsteczce. Jedne liczby rosną, inne maleją.',
                  FadeIn(ld), ShowCreation(bases[2]), ShowCreation(paths[2]), run_time=2.4)
        marks = VGroup()
        for o in D['osie'][:3]:
            x = x0 + (x1 - x0) * o['idx'] / 767
            y = -1.45 + o['czastka'] * s
            marks.add(Dot([x, y, 0], radius=.06, fill_color=YELLOW),
                      label(f'oś {o["os"]}', [x, (y + .3) if o['czastka'] > 0 else (y - .3), 0], 16, YELLOW))
        self.krok('Każda z 768 pozycji to jedna oś stanu. Żółte osie okażą się za chwilę najbardziej powtarzalne na wszystkich zdjęciach.',
                  LaggedStartMap(FadeIn, marks, lag_ratio=.2))


class Scena4(Ujecie):
    numer = 4
    tytul = 'Osie, które przy igle zmieniają się zawsze tak samo'
    tryb = 'POMIAR  •  ranking na 240 parach uczących (69 filmów), sprawdzenie na 85 parach z 24 innych filmów'

    def construct(self):
        self.naglowek()
        osie = D['osie']
        zero, scale = -.3, 3.0 / max(max(abs(o['srednia']) + o['sd'] for o in osie), max(abs(o['czastka']) for o in osie))
        ys = [1.85 - i * .68 for i in range(len(osie))]
        axis = Line([zero, 2.3, 0], [zero, ys[-1] - .4, 0]).set_stroke(MUTED, 1)
        names = VGroup(*[label(f'oś {o["os"]}', [-5.5, y, 0], 24) for o, y in zip(osie, ys)])
        heads = VGroup(label('maleje przy igle', [zero - 1.6, 2.38, 0], 16, ORANGE),
                       label('rośnie przy igle', [zero + 1.6, 2.38, 0], 16, GREEN))
        self.krok('Szukamy osi, które przy igle zmieniają się zawsze w tę samą stronę, na wielu zdjęciach i w wielu filmach.',
                  FadeIn(names), ShowCreation(axis), FadeIn(heads))
        bars, whiskers, dots, pct = VGroup(), VGroup(), VGroup(), VGroup()
        for o, y in zip(osie, ys):
            v, sd = o['srednia'], o['sd']
            col = GREEN if v > 0 else ORANGE
            bars.add(Rectangle(width=abs(v) * scale, height=.34).set_fill(col, .9).set_stroke(width=0)
                     .move_to([zero + v * scale / 2, y, 0]))
            whiskers.add(Line([zero + (v - sd) * scale, y, 0], [zero + (v + sd) * scale, y, 0]).set_stroke(FG, 2))
            dots.add(Dot([zero + o['czastka'] * scale, y, 0], radius=.08, fill_color=WHITE).set_stroke(BLACK, 1))
            pct.add(label(f'{pl(100 * o["znak_test"])} %', [5.6, y, 0], 24, CYAN))
        self.krok('Słupek to średnia zmiana osi na 240 parach uczących, kreska to jej rozrzut. Oś 500 rośnie, gdy igła jest, a osie 472, 735 i 512 maleją.',
                  LaggedStart(*[GrowFromEdge(b, LEFT if o['srednia'] > 0 else RIGHT) for b, o in zip(bars, osie)], lag_ratio=.12),
                  FadeIn(whiskers))
        self.krok('Biała kropka to ta sama oś w cząsteczce 280. Pojedyncza cząsteczka idzie w tę samą stronę co średnia, tylko z inną siłą.',
                  LaggedStartMap(FadeIn, dots, lag_ratio=.12))
        head = VGroup(label('ten sam kierunek', [5.6, 2.55, 0], 15, CYAN), label('na parach testowych', [5.6, 2.27, 0], 15, CYAN))
        self.krok('Na 85 parach z 24 innych filmów kierunek zmiany tych osi zgadza się w 93 do 98 procentach.',
                  FadeIn(head), LaggedStartMap(FadeIn, pct, lag_ratio=.12))
        self.krok('Takie osie nazywamy makrowektorami. To ranking powtarzalności: pojedyncza oś nie dowodzi, że to ona rozpoznaje igłę.')


class Scena5(Ujecie):
    numer = 5
    tytul = 'Chmura zmian i osiem kierunków'
    tryb = 'POMIAR  •  zmiany cząsteczek z 240 par uczących, ważone udziałem igły'

    def construct(self):
        self.naglowek()
        ch = D['chmura']
        cx, cy, hw, hh = -2.0, .15, 4.4, 2.2

        def to_screen(xy, ref):
            xy = np.asarray(xy, float)
            k = np.percentile(np.abs(np.asarray(ref, float)), 99, axis=0)
            return np.column_stack([cx + xy[:, 0] / k[0] * hw * .95, cy + xy[:, 1] / k[1] * hh * .95, np.zeros(len(xy))])

        raw_t, raw_b = to_screen(ch['surowe'], ch['surowe']), to_screen(ch['tlo_surowe'], ch['surowe'])
        svd_t, svd_b = to_screen(ch['svd'], ch['svd']), to_screen(ch['tlo_svd'], ch['svd'])
        # Punkty spoza ramki (skrajny 1 procent) pomijamy w obu widokach, zamiast spychać je na brzeg.
        inside = lambda p: (np.abs(p[:, 0] - cx) <= hw) & (np.abs(p[:, 1] - cy) <= hh)
        kt, kb = inside(raw_t) & inside(svd_t), inside(raw_b) & inside(svd_b)
        raw_t, svd_t, raw_b, svd_b = raw_t[kt], svd_t[kt], raw_b[kb], svd_b[kb]
        axes = VGroup(Line([cx - hw, cy, 0], [cx + hw, cy, 0]).set_stroke(MUTED, 1),
                      Line([cx, cy - hh, 0], [cx, cy + hh, 0]).set_stroke(MUTED, 1))
        ax1 = label(f'oś {ch["osie"][0]}', [cx + hw - .4, cy - .3, 0], 18, MUTED)
        ax2 = label(f'oś {ch["osie"][1]}', [cx + .6, cy + hh - .1, 0], 18, MUTED)
        bg = DotCloud(raw_b, color=MUTED, radius=.022, opacity=.5)
        tool = DotCloud(raw_t, color=GREEN, radius=.026, opacity=.75)
        legend = VGroup(label('zielone: cząsteczki igły', [4.95, 1.6, 0], 21, GREEN),
                        label(f'pokazane {len(raw_t)} z {ch["n_czastek"]}', [4.95, 1.15, 0], 17, MUTED),
                        label('szare: dalekie tło', [4.95, .45, 0], 21, MUTED),
                        label(f'pokazane {len(raw_b)} z {ch["n_tla"]}', [4.95, 0, 0], 17, MUTED))
        self.krok('Każdy zielony punkt to zmiana jednej cząsteczki igły na osiach 500 i 472. Szare punkty to zmiany cząsteczek dalekiego tła.',
                  ShowCreation(axes), FadeIn(ax1), FadeIn(ax2), FadeIn(bg), FadeIn(tool), FadeIn(legend))
        nx1 = label('kierunek 1', [cx + hw - .55, cy - .3, 0], 18, YELLOW)
        nx2 = label('kierunek 2', [cx + .8, cy + hh - .1, 0], 18, YELLOW)
        self.krok('Metoda SVD obraca układ osi. Pierwszy kierunek wskazuje tam, gdzie zmiany cząsteczek igły są największe i najczęstsze.',
                  tool.animate.set_points(svd_t), bg.animate.set_points(svd_b),
                  FadeOut(ax1), FadeOut(ax2), FadeIn(nx1), FadeIn(nx2), run_time=3.0)
        mt, mb = svd_t.mean(0), svd_b.mean(0)
        mean_arrow = Arrow([cx, cy, 0], mt, buff=0, thickness=4).set_color(YELLOW)
        mean_tag = label('średnia zmiana igły', mt + np.array([.2, .45, 0]), 18, YELLOW).add_background_rectangle(opacity=.7)
        mean_bg = VGroup(Dot(mb, radius=.09, fill_color=FG).set_stroke(BLACK, 2),
                         label('średnia tła', mb + np.array([0, -.4, 0]), 16, FG).add_background_rectangle(opacity=.7))
        self.krok('Żółta strzałka to średnia zmiana cząsteczek igły: wskazuje wzdłuż pierwszego kierunku. Średnia zmiana tła leży bliżej środka.',
                  GrowArrow(mean_arrow), FadeIn(mean_tag), FadeIn(mean_bg))
        self.krok('Tło zmienia się słabiej, ale nie zero, bo sieć miesza informację z całego obrazu.')
        # Energia: ile kierunków potrzeba.
        e = np.array(D['energia'])
        gx0, gx1, gy0, gy1 = -5.6, 3.2, -1.75, 2.2
        P = lambda k, v: np.array([gx0 + (gx1 - gx0) * (k - 1) / (len(e) - 1), gy0 + (gy1 - gy0) * v, 0])
        frame = VGroup(Line(P(1, 0), P(len(e), 0)).set_stroke(MUTED, 1), Line(P(1, 0), P(1, 1)).set_stroke(MUTED, 1),
                       *[label(str(k), P(k, 0) + DOWN * .25, 15, MUTED) for k in (1, 5, 8, 20, 40)],
                       *[label(f'{v} %', P(1, v / 100) + LEFT * .45, 15, MUTED) for v in (0, 50, 100)],
                       label('liczba kierunków', [-1.2, gy0 - .55, 0], 17, MUTED))
        curve = VMobject().set_points_as_corners([P(k + 1, v) for k, v in enumerate(e)]).set_stroke(CYAN, 3)
        dots = DotCloud([P(k + 1, v) for k, v in enumerate(e)], color=CYAN, radius=.035)
        m8 = VGroup(Line(P(8, 0), P(8, e[7])).set_stroke(YELLOW, 1.5), Dot(P(8, e[7]), radius=.08, fill_color=YELLOW),
                    label(f'8 kierunków: {pl(100 * D["energia8"])} %', P(8, e[7]) + np.array([.3, .4, 0]), 20, YELLOW))
        m5 = VGroup(Dot(P(5, e[4]), radius=.06, fill_color=FG), label('5: połowa', P(5, e[4]) + np.array([-.35, .35, 0]), 17, FG))
        side = VGroup(label('część całej zmiany,', [5.4, 1.2, 0], 19, MUTED), label('którą obejmuje', [5.4, .85, 0], 19, MUTED),
                      label('k pierwszych kierunków', [5.4, .5, 0], 19, MUTED),
                      label(f'90 % dopiero przy {D["r90"]}', [5.4, -.3, 0], 21, ORANGE))
        self.play(*[FadeOut(m) for m in (axes, nx1, nx2, bg, tool, legend, mean_arrow, mean_tag, mean_bg)], run_time=.6)
        self.krok(f'Ile kierunków potrzeba? Pięć obejmuje połowę całej zmiany, osiem obejmuje {pl(100 * D["energia8"])} procent, a do 90 procent trzeba aż {D["r90"]}.',
                  ShowCreation(frame), ShowCreation(curve), FadeIn(dots), FadeIn(m5), FadeIn(m8), FadeIn(side), run_time=2.4)
        self.play(*[FadeOut(m) for m in (frame, curve, dots, m5, m8, side)], run_time=.6)
        k1 = D['kierunek1']
        wmax = max(abs(c['weight']) for c in k1)
        rows = VGroup()
        for i, c in enumerate(k1):
            y = 1.6 - i * .72
            col = GREEN if c['weight'] > 0 else ORANGE
            rows.add(label(f'oś {c["channel1"]}', [-5.0, y, 0], 24),
                     Rectangle(width=abs(c['weight']) / wmax * 3.3, height=.36).set_fill(col, .9).set_stroke(width=0)
                     .move_to([.6 + np.sign(c['weight']) * abs(c['weight']) / wmax * 1.65, y, 0]),
                     label(f'{"+" if c["weight"] > 0 else "−"}{pl(abs(c["weight"]), 3)}', [5.3, y, 0], 22, col))
        zline = Line([.6, 2.1, 0], [.6, -1.75, 0]).set_stroke(MUTED, 1)
        t1 = label('kierunek 1: pięć największych wag z 768', [0, 2.38, 0], 19, YELLOW)
        self.krok('Kierunek nie jest jedną osią. Pierwszy to mieszanka wszystkich 768 osi; największe wagi mają osie 226, 655, 80, 390 i 6.',
                  FadeIn(t1), ShowCreation(zline), LaggedStartMap(FadeIn, rows, lag_ratio=.06))
        self.krok('Osiem takich kierunków tworzy podprzestrzeń tożsamości: płaski wycinek przestrzeni 768 liczb, w którym najlepiej widać igłę.')


class Scena6(Ujecie):
    numer = 6
    tytul = 'Czy z ośmiu kierunków da się odczytać igłę?'
    tryb = 'POMIAR  •  85 par z 24 filmów odłożonych do testu'

    def construct(self):
        self.naglowek()
        o = D['odczyt']
        po, pn, thr = np.array(o['p_oryginal']), np.array(o['p_lama']), o['prog']
        X = lambda p: -5.4 + 10.8 * np.asarray(p)
        rng = np.random.default_rng(3)
        yo, yn = 1.0, -.45
        pts_o = np.column_stack([X(po), yo + rng.uniform(-.38, .38, len(po)), np.zeros(len(po))])
        pts_n = np.column_stack([X(pn), yn + rng.uniform(-.38, .38, len(pn)), np.zeros(len(pn))])
        axis = VGroup(Line([-5.4, -1.15, 0], [5.4, -1.15, 0]).set_stroke(MUTED, 1),
                      *[label(t, [X(v), -1.42, 0], 16, MUTED) for v, t in ((0, '0'), (.5, '0,5'), (1, '1'))],
                      label('szansa, że igła jest', [0, -1.78, 0], 17, MUTED))
        ro = left_label('oryginały', -7.0, yo + .55, 19, GREEN)
        rn = left_label('bez igły (LaMa)', -7.0, yn + .55, 19, ORANGE)
        flow = VGroup(card('cząsteczki\npod maską', [-4.6, .4, 0], 2.4, 1.1, GREEN, 21),
                      card('średnia ważona\nudziałem igły', [-1.6, .4, 0], 2.6, 1.1, CYAN, 21),
                      card('8 liczb:\nrzut na kierunki', [1.4, .4, 0], 2.6, 1.1, YELLOW, 21),
                      card('klasyfikator\nliniowy', [4.4, .4, 0], 2.4, 1.1, PURPLE, 21))
        arrows = VGroup(*[Arrow([x + 1.25, .4, 0], [x + 1.75, .4, 0], buff=0).set_color(MUTED) for x in (-4.6, -1.6, 1.4)])
        self.krok('Teraz cały obraz naraz. Uśredniamy cząsteczki pod maską z wagą udziału igły i rzutujemy wynik na osiem kierunków.',
                  LaggedStartMap(FadeIn, flow, lag_ratio=.25), FadeIn(arrows), run_time=2.0)
        self.krok('Klasyfikator liniowy, czyli ważona suma tych ośmiu liczb, zamienia je na szansę, że igła jest.')
        self.play(FadeOut(flow), FadeOut(arrows), run_time=.5)
        do = DotCloud(pts_o, color=GREEN, radius=.06)
        dn = DotCloud(pts_n, color=ORANGE, radius=.06)
        self.krok(f'Każda kropka to jeden obraz z filmów, których nie użyliśmy do wyznaczania kierunków: {o["n_test"]} oryginałów i {o["n_test"]} kopii bez igły.',
                  ShowCreation(axis), FadeIn(ro), FadeIn(rn), FadeIn(do), FadeIn(dn))
        tl = VGroup(DashedLine([X(thr), -1.15, 0], [X(thr), 1.75, 0]).set_stroke(YELLOW, 2),
                    label(f'próg {pl(thr, 3)}', [X(thr), 2.0, 0], 18, YELLOW))
        ok_o, ok_n = int((po >= thr).sum()), int((pn < thr).sum())
        self.krok(f'Próg ustalono wcześniej, na osobnych filmach. Powyżej leży {ok_o} z {o["n_test"]} oryginałów, poniżej {ok_n} z {o["n_test"]} kopii bez igły.',
                  FadeIn(tl))
        res = VGroup(card(f'8 kierunków: trafność {pl(100 * o["trafnosc"])} %', [-3.0, 2.38, 0], 4.9, .5, CYAN, 19),
                     card(f'kierunek LDA: {pl(100 * o["lda_trafnosc"])} %', [3.0, 2.38, 0], 4.2, .5, PURPLE, 19))
        self.krok(f'Trafność, liczona z równą wagą filmów, wynosi {pl(100 * o["trafnosc"])} procent. Kierunek LDA, czyli jedna oś najlepiej rozdzielająca obie grupy, daje {pl(100 * o["lda_trafnosc"])}.',
                  FadeOut(tl[1]), FadeIn(res))
        self.krok('Odczyt zna miejsce igły z maski. Mówi więc tylko, czy w tym miejscu igła jest, a nie gdzie jej szukać.')


class Scena7(Ujecie):
    numer = 7
    tytul = 'Które cząsteczki niosą tożsamość, a które decydują'
    tryb = 'POMIAR  •  klatka 881 filmu TR_0053 (film testowy), 18 cząsteczek pod maską'

    def construct(self):
        self.naglowek()
        W, cx, cy = 6.4, -3.35, .2
        img = image('obraz_oryginal.png', cx, cy, W)
        _, _, cell, _, center = grid_geometry(cx, cy, W)
        mapa = np.array(D['mapa'])
        vmax = mapa.max()
        heat = VGroup(*[Square(side_length=cell).set_fill(interpolate_color(ORANGE, RED, mapa[j] / vmax),
                                                         .2 + .75 * mapa[j] / vmax).set_stroke(YELLOW, .8).move_to(center(j)) for j in TOOL])
        # Wykresy: górny = reakcja (długość zmiany w 8 kierunkach), dolny = wpływ na detektor.
        gx0, gx1 = .85, 6.75
        xs = np.linspace(gx0, gx1, len(TOOL))
        sc_r, sc_w = 1.75 / vmax, 1.75 / max(abs(t['dq']) for t in D['wplyw'])
        wpl = {t['token'] - 1: t['dq'] for t in D['wplyw']}
        top = VGroup(*[Rectangle(width=.22, height=max(mapa[j] * sc_r, .01)).set_fill(RED if j == C280 else ORANGE, .9)
                       .set_stroke(width=0).move_to([x, .35 + mapa[j] * sc_r / 2, 0]) for x, j in zip(xs, TOOL)])
        bot = VGroup(*[Rectangle(width=.22, height=max(abs(wpl[j]) * sc_w, .012)).set_fill(CYAN, .95)
                       .set_stroke(width=0).move_to([x, -.35 - abs(wpl[j]) * sc_w / 2, 0]) for x, j in zip(xs, TOOL)])
        nums = VGroup(*[label(str(j + 1), [x, 0, 0], 11, WHITE if j in (C280, 247) else MUTED) for x, j in zip(xs, TOOL)])
        ttop = label('reakcja na usunięcie igły', [(gx0 + gx1) / 2, 2.4, 0], 17, ORANGE)
        tbot = label('spadek wyniku detektora', [(gx0 + gx1) / 2, -2.3, 0], 17, CYAN)
        self.add(img)
        self.krok('Kolor pola pokazuje, jak mocno cząsteczka zmienia się w ośmiu kierunkach, gdy igła znika.',
                  LaggedStartMap(FadeIn, heat, lag_ratio=.05), FadeIn(nums), FadeIn(ttop), LaggedStartMap(GrowFromEdge, top, edge=DOWN, lag_ratio=.05))
        f280 = cell_box(C280, cx, cy, W, WHITE, 3.5)
        l280 = label(f'280: {pl(D["czastka"]["rzut"], 2)}', center(C280) + np.array([0, 1.05, 0]), 20, WHITE).add_background_rectangle(opacity=.75)
        self.krok(f'Najmocniej reaguje cząsteczka 280: długość jej zmiany w ośmiu kierunkach wynosi {pl(D["czastka"]["rzut"], 2)}.',
                  FadeIn(f280), FadeIn(l280))
        self.krok('Potem podmienialiśmy po jednej cząsteczce na wersję bez igły i liczyliśmy detektor do końca.')
        self.krok('Mierzymy pewność wykrycia pomnożoną przez to, jak dobrze ramka pokrywa igłę. Dolne słupki pokazują spadek tego wyniku.',
                  FadeIn(tbot), LaggedStartMap(GrowFromEdge, bot, edge=UP, lag_ratio=.05))
        f248 = cell_box(247, cx, cy, W, CYAN, 3.5)
        l248 = label(f'248: −{pl(abs(wpl[247]), 3)}', center(247) + np.array([-.2, -1.0, 0]), 20, CYAN).add_background_rectangle(opacity=.75)
        self.krok(f'Wynik obniża tylko podmiana cząsteczki 248, o {pl(abs(wpl[247]), 3)}. Podmiana 280 nie zmienia nic. Mocna reakcja nie znaczy więc wpływu na decyzję.',
                  FadeIn(f248), FadeIn(l248))
        self.krok('Za tą warstwą detektor patrzy na każde pole osobno. Informacja z innych pól trafiła do 248 wcześniej, przez mieszanie między polami w sieci.')


class Scena8(Ujecie):
    numer = 8
    tytul = 'Sprawdzian: czy te kierunki wpływają na detektor?'
    tryb = 'POMIAR  •  85 zdjęć testowych, ingerencja tylko w cząsteczkach pod maską'

    def construct(self):
        self.naglowek()
        d = D['detekcja']
        base = d['bez']
        rows = [('8 kierunków tożsamości', d['tozsamosc'], YELLOW)] + \
               [(f'losowe 8 kierunków, próba {i + 1}', v, MUTED) for i, v in enumerate(d['losowe'])] + \
               [(f'losowe, ta sama siła na wyjściu, próba {i + 1}', v, PURPLE) for i, v in enumerate(d['losowe_fc2'])]
        zero, scale = 4.6, 6.0 / .022
        ys = [1.9 - i * .6 for i in range(len(rows))]
        names = VGroup(*[left_label(n, -6.9, y, 19, c if c != MUTED else FG, 4.6) for (n, _, c), y in zip(rows, ys)])
        bars = VGroup(*[Rectangle(width=max((base - v) * scale, .01), height=.34).set_fill(c, .9).set_stroke(width=0)
                        .move_to([zero - (base - v) * scale / 2, y, 0]) for (_, v, c), y in zip(rows, ys)])
        vals = VGroup(*[label(f'−{pl(base - v, 4)}', [zero + .75, y, 0], 18, c if c != MUTED else FG) for (_, v, c), y in zip(rows, ys)])
        axis = VGroup(Line([zero, 2.3, 0], [zero, ys[-1] - .35, 0]).set_stroke(MUTED, 1),
                      label(f'spadek wyniku od {pl(base, 3)}', [zero - 2.4, 2.45, 0], 16, MUTED))
        self.krok('W cząsteczkach igły przesuwamy tylko składowe w ośmiu kierunkach, w stronę obrazu bez igły. Reszta stanu zostaje bez zmian.')
        self.krok(f'Wynik detektora, czyli pewność razy pokrycie ramki, spada z {pl(base, 3)} do {pl(d["tozsamosc"], 3)}. Przedział niepewności tego spadku nie obejmuje zera.',
                  FadeIn(axis), FadeIn(names[0]), GrowFromEdge(bars[0], RIGHT), FadeIn(vals[0]))
        self.krok('Losowe kierunki z tak samo długim przesunięciem zmieniają wynik kilka razy słabiej, także po wyrównaniu siły na wyjściu warstwy.',
                  LaggedStartMap(FadeIn, names[1:], lag_ratio=.1), LaggedStartMap(GrowFromEdge, bars[1:], edge=RIGHT, lag_ratio=.1),
                  LaggedStartMap(FadeIn, vals[1:], lag_ratio=.1), run_time=2.2)
        ap = card(f'AP50: {pl(100 * d["ap_bez"])} → {pl(100 * d["ap_tozsamosc"])} %   •   przedział zmiany: '
                  f'{pl(100 * d["ap_ci"][0])} do +{pl(100 * d["ap_ci"][1])} punktu', [0, 2.1, 0], 11.5, .55, ORANGE, 20)
        self.play(*[FadeOut(m) for m in (axis, names, bars, vals)], run_time=.6)
        self.krok(f'Trafność wykrywania AP50 spadła z {pl(100 * d["ap_bez"])} do {pl(100 * d["ap_tozsamosc"])} procent. Ten spadek mieści się jednak w szumie, więc go nie rozstrzygamy.',
                  FadeIn(ap))
        sh = D['sham']
        s2 = 5.0 / sh['usuniecie']
        b1 = Rectangle(width=sh['atrapa'] * s2, height=.5).set_fill(MUTED, .9).set_stroke(width=0)
        b2 = Rectangle(width=sh['usuniecie'] * s2, height=.5).set_fill(GREEN, .9).set_stroke(width=0)
        b1.move_to([-1.5 + sh['atrapa'] * s2 / 2, .55, 0]); b2.move_to([-1.5 + sh['usuniecie'] * s2 / 2, -.35, 0])
        sham = VGroup(left_label('atrapa: zamalowane tło obok, igła zostaje', -6.9, .55, 18, FG, 5.2),
                      left_label('usunięta igła', -6.9, -.35, 18, GREEN, 5.2), b1, b2,
                      label(pl(sh['atrapa'], 2), [-1.5 + sh['atrapa'] * s2 + .45, .55, 0], 20, FG),
                      label(pl(sh['usuniecie'], 2), [-1.5 + sh['usuniecie'] * s2 + .45, -.35, 0], 20, GREEN),
                      label(f'reakcja cząsteczek igły w 8 kierunkach  •  {sh["n"]} par z filmów testowych', [0, 1.35, 0], 17, MUTED))
        self.krok(f'Kontrola: LaMa zamalowuje kawałek tła obok, a igła zostaje. Cząsteczki igły zmieniają się wtedy o {pl(sh["atrapa"], 2)}, a po usunięciu igły o {pl(sh["usuniecie"], 2)}.',
                  FadeIn(sham))
        self.krok('Kierunki reagują więc na samą igłę, a nie na każdą edycję LaMy. Wszystkich śladów tej metody kontrola jednak nie usuwa.')


class Scena9(Ujecie):
    numer = 9
    tytul = 'Cała droga i jej granice'
    tryb = 'PODSUMOWANIE  •  co pomiar pokazuje, a czego nie'

    def construct(self):
        self.naglowek()
        steps = [('para zdjęć\nz igłą i bez', GREEN), ('18 cząsteczek\npod maską', GREEN), ('różnica\n768 liczb', CYAN),
                 ('powtarzalne\nosie', CYAN), ('8 kierunków\nSVD', YELLOW), ('odczyt\ni sprawdzian', PURPLE)]
        xs = np.linspace(-5.75, 5.75, len(steps))
        cards = VGroup(*[card(s, [x, 1.75, 0], 1.95, 1.0, c, 19) for (s, c), x in zip(steps, xs)])
        arrows = VGroup(*[Arrow([xs[i] + 1.0, 1.75, 0], [xs[i + 1] - 1.0, 1.75, 0], buff=0).set_color(MUTED) for i in range(len(xs) - 1)])
        self.krok('Cała droga w skrócie: para zdjęć, cząsteczki pod maską, ich różnice, powtarzalne osie, osiem kierunków, odczyt i sprawdzian.',
                  LaggedStartMap(FadeIn, cards, lag_ratio=.2), FadeIn(arrows), run_time=2.4)
        limits = [('Tożsamość znaczy tu: igła jest kontra igła wymazana.\nInnych narzędzi nie rozróżnialiśmy.', ORANGE),
                  ('Jedna zamrożona sieć: ziarno 47, krok 1000.\nRuchu kierunków w czasie uczenia ten pomiar nie widzi.', ORANGE),
                  ('Zdjęcia pochodzą ze zbioru uczącego detektora.\nOdłożone filmy sprawdzają wyznaczanie kierunków, nie sieć.', ORANGE)]
        boxes = [card(t, [0, .35 - i * 1.0, 0], 11.0, .85, c, 19) for i, (t, c) in enumerate(limits)]
        self.krok('Tożsamość znaczy tu tylko obecność igły kontra jej wymazanie. Innych narzędzi ten pomiar nie rozróżnia.', FadeIn(boxes[0]))
        self.krok('Kierunki wyznaczono na jednej zamrożonej sieci. Jak zmieniają się podczas uczenia, trzeba zmierzyć osobno.', FadeIn(boxes[1]))
        self.krok('Wszystkie zdjęcia sieć widziała w uczeniu. Odłożone filmy sprawdzają więc wyznaczanie kierunków, a nie samą sieć.',
                  FadeIn(boxes[2]), hold=1.5)
