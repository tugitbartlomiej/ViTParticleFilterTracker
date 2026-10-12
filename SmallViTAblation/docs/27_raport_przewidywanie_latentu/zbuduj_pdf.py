"""Składa raport 27 do PDF (reportlab). Liczby w tabelach czytane z plików wyników, nie przepisywane ręcznie.

Uruchomienie: py -3.11 wykresy.py && py -3.11 zbuduj_pdf.py
Wynik: ../27_PRZEWIDYWANIE_LATENTU_RAPORT.pdf
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
RYS = HERE / 'rys'
EXP = HERE.parents[1] / 'experiments'
SUF = EXP / 'maly_vit_sufit_przewidywalnosci_100'
PAC = EXP / 'maly_vit_gesta_historia_paczka'
OUT = HERE.parent / '27_PRZEWIDYWANIE_LATENTU_RAPORT.pdf'

FD = Path(matplotlib.__file__).parent / 'mpl-data' / 'fonts' / 'ttf'
pdfmetrics.registerFont(TTFont('DV', str(FD / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DVB', str(FD / 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DVI', str(FD / 'DejaVuSans-Oblique.ttf')))
pdfmetrics.registerFont(TTFont('DVM', str(FD / 'DejaVuSansMono.ttf')))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DVB', italic='DVI', boldItalic='DVB')

INK, INK2, NIEB, OS = colors.HexColor('#0b0b0b'), colors.HexColor('#52514e'), colors.HexColor('#2a78d6'), colors.HexColor('#c3c2b7')
S = {
    'tytul': ParagraphStyle('t', fontName='DVB', fontSize=17, leading=21, textColor=INK, spaceAfter=4),
    'pod': ParagraphStyle('p', fontName='DV', fontSize=9.5, leading=13, textColor=INK2, spaceAfter=10),
    'h1': ParagraphStyle('h1', fontName='DVB', fontSize=13, leading=17, textColor=INK, spaceBefore=12, spaceAfter=5),
    'h2': ParagraphStyle('h2', fontName='DVB', fontSize=10.8, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3),
    'b': ParagraphStyle('b', fontName='DV', fontSize=9.6, leading=13.6, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5),
    'lista': ParagraphStyle('l', fontName='DV', fontSize=9.6, leading=13.4, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=2),
    'pods': ParagraphStyle('c', fontName='DVI', fontSize=8.4, leading=11, textColor=INK2, alignment=TA_LEFT, spaceBefore=2, spaceAfter=9),
    'ramka': ParagraphStyle('r', fontName='DV', fontSize=9.6, leading=13.6, textColor=INK, alignment=TA_LEFT),
    'tab': ParagraphStyle('tb', fontName='DV', fontSize=8.3, leading=10.5, textColor=INK),
    'tabh': ParagraphStyle('th', fontName='DVB', fontSize=8.3, leading=10.5, textColor=INK),
    'mono': ParagraphStyle('m', fontName='DVM', fontSize=7.6, leading=10, textColor=INK2),
}
SZER = A4[0] - 4 * cm


def P(t, s='b'): return Paragraph(t, S[s])
def L(items): return [Paragraph(t, S['lista'], bulletText='•') for t in items]
def pl(x, n=2): return f'{x:.{n}f}'.replace('.', ',')


def rys(nazwa, podpis, szer=SZER):
    from PIL import Image as PI
    w, h = PI.open(RYS / nazwa).size
    return KeepTogether([Image(str(RYS / nazwa), width=szer, height=szer * h / w), P(podpis, 'pods')])


def wzor(nazwa, szer_cm):
    from PIL import Image as PI
    w, h = PI.open(RYS / nazwa).size
    im = Image(str(RYS / nazwa), width=szer_cm * cm, height=szer_cm * cm * h / w)
    im.hAlign = 'CENTER'
    return im


def tabela(wiersze, kol, naglowek=True):
    dane = [[c if isinstance(c, Paragraph) else Paragraph(str(c), S['tabh' if (naglowek and i == 0) else 'tab'])
             for c in w] for i, w in enumerate(wiersze)]
    t = Table(dane, colWidths=kol, hAlign='LEFT')
    t.setStyle(TableStyle([('LINEBELOW', (0, 0), (-1, 0), 0.8, INK2), ('LINEBELOW', (0, -1), (-1, -1), 0.5, OS),
                           ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 3),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f7f5')])]))
    return t


def ramka(tekst, kolor='#eef4fc', obw=NIEB):
    t = Table([[Paragraph(tekst, S['ramka'])]], colWidths=[SZER])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(kolor)), ('BOX', (0, 0), (-1, -1), 0.8, obw),
                           ('LEFTPADDING', (0, 0), (-1, -1), 9), ('RIGHTPADDING', (0, 0), (-1, -1), 9),
                           ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    return t


def stopka(c, doc):
    c.saveState(); c.setFont('DV', 7.5); c.setFillColor(INK2)
    c.drawString(2 * cm, 1.2 * cm, 'Mały ViT: przewidywanie latentu w trakcie uczenia  ·  raport 27  ·  2026-10-04')
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f'str. {doc.page}')
    c.restoreState()


def main():
    rz = json.loads((SUF / 'wyniki' / 'rownanie_z_pedem' / 'ROWNANIE_Z_PEDEM.json').read_text(encoding='utf-8'))
    pacz = {}
    for f in (PAC / 'wyniki' / 'porownanie' / 'POROWNANIE.json', PAC / 'wyniki' / 'porownanie_p128' / 'POROWNANIE.json'):
        pacz.update(json.loads(f.read_text(encoding='utf-8'))['paczki'])
    acf = json.loads((SUF / 'wyniki' / 'autokorelacja' / 'ACF.json').read_text(encoding='utf-8'))
    cj = json.loads((SUF / 'wyniki' / 'co_jeszcze' / 'CO_JESZCZE.json').read_text(encoding='utf-8'))
    wf = json.loads((SUF / 'wyniki' / 'wczesna_faza' / 'WCZESNA_FAZA.json').read_text(encoding='utf-8'))
    s47, s48 = rz['s47_test'], rz['s48_przeniesienie']
    a = acf['300-1000']['acf']

    st = []
    st += [P('Przewidywanie latentu małego ViT w trakcie uczenia', 'tytul'),
           P('Jak to liczymy, jakie równanie Langevina stosujemy i co wychodzi na 1–100 kroków naprzód. '
             'Repozytorium SmallViTAblation, pomiary z 2–4 października 2026, ziarna 47 i 48, bez nowych założeń '
             'poza opisanymi. Wszystkie liczby w tabelach są czytane z plików wyników.', 'pod')]
    st.append(ramka(
        '<b>Najkrótszy wniosek.</b> Latent sieci reaguje na kroki uczenia prawie liniowo: równanie Langevina, które '
        f'dostaje prawdziwą siłę każdego kroku, odtwarza latent na 10 kroków z wynikiem {pl(s47["pelna_sila_znana_iter"][9], 3)}. '
        'To jednak wyjaśnienie, nie prognoza, bo siła kroku jest znana dopiero po kroku. '
        f'Prawdziwa prognoza (tylko przeszłość i stan optymalizatora) daje na 10 kroków {pl(s47["hist_ped_wprost"][9])}, '
        'na 1 krok ' + pl(s47['hist_ped_wprost'][0]) + '. Brakuje siły z przyszłych paczek. Większa paczka tego nie '
        'zmienia; drgania tworzy sam optymalizator AdamW.'))
    st.append(Spacer(1, 8))

    # 1. słowniczek
    st.append(P('1. Słowniczek: zwykłymi słowami', 'h1'))
    st += L([
        '<b>Latent</b>: liczby, którymi sieć opisuje obraz w środku swojej pracy. Tu: 768 liczb na każdy z 600 '
        'kawałków obrazu, wzięte z ostatniego bloku sieci. Mówią, które „neurony się zapalają”.',
        '<b>Token (cząsteczka)</b>: jeden kawałek obrazu 16 × 16 pikseli. Obraz 480 × 320 daje siatkę 30 × 20 = 600 tokenów.',
        '<b>Krok uczenia</b>: jedna poprawka wag sieci. Sieć ogląda paczkę 32 obrazów, liczy, w którą stronę zmienić wagi '
        '(to jest <i>gradient</i>), i robi mały krok w tę stronę.',
        '<b>AdamW</b>: przepis na ten krok. Ma <i>pęd</i>, czyli uśrednioną pamięć poprzednich kierunków (sięga około '
        '10 kroków wstecz), i dzieli krok przez typowy rozmiar gradientu, więc kroki mają podobną długość.',
        '<b>Prognoza „bez zmian”</b>: najprostsza prognoza, że latent za h kroków będzie taki jak teraz. To punkt odniesienia.',
        '<b>Wynik prognozy</b>: o ile mniejszy jest błąd kwadratowy prognozy od błędu prognozy „bez zmian”. '
        '1 to ideał, 0 to tyle co „bez zmian”, wartości ujemne są gorsze. Cel użytkownika: ponad 0,85.',
        '<b>Równanie Langevina</b>: opis ruchu jako suma części przewidywalnej (dryf, ściąganie do średniej, pamięć) '
        'i losowych pchnięć (szum). Tu stosujemy je do latentu w czasie uczenia.',
    ])
    st.append(wzor('w_umiejetnosc.png', 9))
    st.append(P('Sumy biegną po wszystkich tokenach, wszystkich 768 liczbach i wszystkich chwilach startu prognozy w części testowej.', 'pods'))

    # 2. architektura
    st.append(PageBreak())
    st.append(P('2. Architektura sieci i skąd bierzemy latent', 'h1'))
    st.append(P('Sieć to mały transformer obrazowy (ViT) uczony od zera do wykrywania narzędzia chirurgicznego na '
                'klatkach operacji zaćmy (zbiór Cataract-LMM). Łodyga z czterech konwolucji zamienia obraz na siatkę '
                '600 tokenów po 192 liczby. Sześć bloków transformera pozwala tokenom wymieniać informację. Głowa na '
                'każdym tokenie mówi, czy tu jest środek narzędzia, i jak daleko są krawędzie ramki. Latent bierzemy ze '
                'środka ostatniego bloku: to 768 liczb po funkcji GELU, czyli wartości, które blok „zapala” dla danego tokenu.'))
    st.append(rys('01_architektura.png', 'Rysunek 1. Architektura małego ViT (kod: experiments/maly_vit_gesta_historia_paczka/kod/model.py). '
                  'Zielonym: miejsce, z którego czytamy latent (768 liczb na token).'))
    st.append(P('3. Jak zbieramy dane', 'h1'))
    st.append(P('Uczymy sieć 1000 kroków od zera (AdamW, lr 5·10<super>−4</super> z rozgrzewką przez 300 kroków, paczka 32). '
                'Po każdym kroku przepuszczamy przez sieć te same 12 obrazów w trybie pomiaru (sieć się wtedy nie uczy) '
                'i zapisujemy cały latent. Tak powstaje „gęsta historia”: 1001 stanów × 12 obrazów × 600 tokenów × 768 liczb '
                '(22 GB). Osobne nagranie z wcześniejszego eksperymentu zapisuje też <i>siły</i>: o ile przesunąłby się latent, '
                'gdyby wagi przesunąć tylko o jedną część kroku AdamW (część z pędu, część z nowej paczki, część z zaniku wag).'))
    st.append(rys('02_potok_pomiaru.png', 'Rysunek 2. Potok pomiaru. Analiza nie uczy sieci od nowa; korzysta z zapisanych stanów i sił.',
                  szer=0.78 * SZER))

    # 4. co da się przewidzieć
    st.append(PageBreak())
    st.append(P('4. Co w ruchu latentu w ogóle da się przewidzieć', 'h1'))
    st.append(P('Ruch latentu składa się z powolnego dryfu i z drgań. Dryf to spokojne przesuwanie się poziomu, które '
                'można by przedłużyć w przyszłość. Drgania to szybkie odbicia w górę i w dół wokół tego poziomu.'))
    st.append(rys('03_ruch_dryf_drgania.png', 'Rysunek 3. Siła tożsamości narzędzia (jedna liczba na obraz: średni rzut tokenów narzędzia '
                  'na kierunek, który najlepiej odróżnia narzędzie od tła) w 1000 krokach uczenia. Na dole: drgania, czyli '
                  'różnica między przebiegiem a gładką ścieżką.'))
    st.append(P('Kluczowe jest to, jak długo ruch „pamięta” sam siebie. Mierzy to autokorelacja przyrostów: podobieństwo '
                f'zmiany latentu w jednym kroku do zmiany w kroku o k później. Po 1 kroku wynosi {pl(a[1])}, czyli ruch jest '
                f'gładki, co robi pęd AdamW. Po 4 krokach spada do {pl(a[4])}: latent odbija w przeciwną stronę. Od około '
                '10 kroków nie ma już związku z przeszłością.'))
    st.append(rys('04_autokorelacja.png', 'Rysunek 4. Autokorelacja przyrostów latentu tokenów narzędzia (gęsta historia, paczka 32).'))
    st.append(KeepTogether([
        P('Sufit i podłoga', 'h2'),
        P('Sufit to wynik wyroczni, która zna całą gładką ścieżkę przyszłości i myli się tylko o drgania. '
          'Podłoga to wynik samego odszumienia punktu startowego, bez żadnego dryfu. Przybliżony wzór na sufit, '
          'gdzie D to przesunięcie dryfu w h krokach, a σ to rozmiar drgań:'),
        wzor('w_sufit.png', 5.5)]))
    st.append(rys('07_sufit_100.png', 'Rysunek 5. Prognoza na 100 kroków dla tokenów narzędzia: osiągnięte (model liniowy z historii, '
                  'sprawdzony na innych obrazach), podłoga i sufit przy dwóch szerokościach wygładzania. Wyrocznia daje '
                  '0,75–0,94, ale jej „dryf” zawiera powolne krążenie, którego z przeszłości nie widać.'))

    # 5. równania
    st.append(PageBreak())
    st.append(P('5. Równania, które sprawdziliśmy', 'h1'))
    st.append(P('Wszystkie równania są liniowe i mają mało parametrów: te same współczynniki dla wszystkich 768 liczb '
                'i wszystkich tokenów. Dopasowujemy je na wcześniejszych krokach albo innych obrazach, a sprawdzamy na '
                'krokach i obrazach, których dopasowanie nie widziało.'))
    st.append(P('Ściąganie do średniej (proces Ornsteina-Uhlenbecka, OU)', 'h2'))
    st.append(P('Latent wraca do średniej z ostatnich k stanów. Odchylenie od niej gaśnie wykładniczo z szybkością κ.'))
    st.append(wzor('w_ou.png', 11))
    st.append(P('Jądro pamięci i pęd optymalizatora (forma z projektu grantowego, bez wymuszenia)', 'h2'))
    st.append(P('Zmiana latentu w następnym kroku to: ściąganie do średniej, ważona pamięć ostatnich J zmian (jądro '
                'pamięci K<sub>j</sub>), siła pędu AdamW i siła zaniku wag. Pęd i zanik są znane w chwili t, bo zależą tylko '
                'od przeszłości.'))
    st.append(wzor('w_jadro.png', 15))
    st.append(wzor('w_ped.png', 12))
    st.append(P('Granica: równanie z pełną siłą kroku', 'h2'))
    st.append(P('Żeby sprawdzić, czy forma równania jest dobra, dajemy mu prawdziwą siłę każdego przyszłego kroku. '
                'S<super>krok</super> to przesunięcie latentu z(θ), gdy wagi θ przesunąć o cały krok AdamW d<sub>t</sub>, '
                'policzone przez sieć w jej bieżącym stanie. Tego w prognozie nie wolno używać (reguła: bez przyszłych '
                'paczek), więc to granica, nie metoda.'))
    st.append(wzor('w_pelna.png', 14))

    # 6. wyniki 1..10
    st.append(PageBreak())
    st.append(P('6. Wyniki na 1–10 kroków naprzód', 'h1'))
    st.append(rys('05_rownania_1_10.png', 'Rysunek 6. Wynik prognozy w funkcji horyzontu. Dopasowanie na krokach 1000–2000 ziarna 47, '
                  'test na krokach 2000–3000 ziarna 47 (gruba linia) oraz ziarna 48 z tymi samymi współczynnikami (cienka linia). '
                  'Latent: średnie po tokenach narzędzia 180 sond.'))
    nazwy = [('hist_wprost', 'Prognoza: sama historia latentu'), ('ped_wprost', 'Prognoza: sam pęd i zanik wag'),
             ('hist_ped_wprost', 'Prognoza: historia + pęd + zanik'), ('hist_ped_iter', 'To samo, iterowane krok po kroku'),
             ('paczki_znane_iter', 'Granica: znane przyszłe paczki, pęd liniowo'),
             ('pelna_sila_znana_iter', 'Granica: znana pełna siła każdego kroku')]
    w = [['model', 'h = 1', 'h = 2', 'h = 5', 'h = 10', 'h = 10, ziarno 48']]
    for k, n in nazwy:
        w.append([n] + [pl(s47[k][i]) for i in (0, 1, 4, 9)] + [pl(s48[k][9])])
    st.append(tabela(w, [6.6 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.9 * cm]))
    st.append(P('Tabela 1. Wynik prognozy (1 − błąd² / błąd² „bez zmian”). Plik: experiments/maly_vit_sufit_przewidywalnosci_100/'
                'wyniki/rownanie_z_pedem/ROWNANIE_Z_PEDEM.json.', 'pods'))
    st.append(P('Co z tego wynika', 'h2'))
    st += L([
        f'Na 1 krok pęd optymalizatora pomaga: wynik rośnie z {pl(s47["hist_wprost"][0])} do {pl(s47["hist_ped_wprost"][0])}.',
        f'Na 10 kroków wszystkie prognozy zatrzymują się przy około {pl(s47["hist_ped_wprost"][9])}. Pęd gaśnie po kilku '
        'krokach, a dalej ruch zależy od gradientów kolejnych paczek, których jeszcze nie ma.',
        f'Równanie z prawdziwą pełną siłą każdego kroku daje {pl(s47["pelna_sila_znana_iter"][9], 3)} na 10 kroków. '
        'Forma równania jest więc dobra; brakuje tylko siły.',
        f'Znane przyszłe paczki, ale z pędem składanym liniowo, dają {pl(s47["paczki_znane_iter"][4])} na 5 kroków i '
        f'{pl(s47["paczki_znane_iter"][9])} na 10. Siła tej samej paczki zależy od stanu sieci, który odpływa.',
        'Iterowanie równania jednego kroku bez przyszłych sił psuje prognozę: błędy się składają.',
        'Współczynniki przenoszą się między ziarnami prawie bez straty.',
    ])
    st.append(P('Od checkpointu 90 do 100, krok po kroku', 'h2'))
    st.append(P('Ten sam odcinek, na którym przegrał model NNDMD (sieć Koopmana przewidująca cały latent): start w kroku 900, '
                'prognoza do 1000. Równanie OU zyskuje 20–30 procent na prawie każdym kroku; NNDMD jest gorszy od „bez zmian” '
                'na wszystkich 10 checkpointach, bo pcha ruch do przodu, a ruch zawraca.'))
    st.append(rys('../../../experiments/maly_vit_sufit_przewidywalnosci_100/wyniki/prognoza_900/PROGNOZA_900.png',
                  'Rysunek 7. Prognoza od kroku 900 do 1000. Panel A: jeden obraz i trzy prognozy. Panele B–D: wynik na każdym '
                  'kroku dla tokenów losowych, tokenów narzędzia i siły tożsamości; szara linia przerywana to sufit wyroczni.'))

    # 7. paczki
    st.append(PageBreak())
    st.append(P('7. Czy większa paczka uspokaja ruch', 'h1'))
    st.append(P('Teoria dyfuzji dla zwykłego SGD mówi, że losowe pchnięcia maleją jak 1 przez rozmiar paczki. Jeśli drgania '
                'latentu pochodzą z losowania obrazów do paczki, większa paczka powinna je zmniejszyć i podnieść wynik prognozy. '
                'Puściliśmy dwa nowe biegi na Edenie (paczka 128 i 320, ten sam przepis i ziarno, po około 5 minut na H100) '
                'i porównaliśmy je z paczką 32 na tych samych tokenach.'))
    st.append(rys('06_paczki.png', 'Rysunek 8. Lewy panel: wynik prognozy z historii przy trzech paczkach. Prawy: autokorelacja przyrostów.'))
    ocena32 = json.loads((EXP / 'maly_vit_ruch_czastek_gesta_historia' / 'wyniki' / 'pelny' / 'dane' / 'OCENY.json')
                         .read_text(encoding='utf-8'))[-1]['mAP50']
    w = [['paczka', 'h = 1', 'h = 10', 'h = 100', 'sufit h = 10', 'autokorelacja po 1 kroku', 'względny krok', 'mAP@0,5 na końcu']]
    for k in sorted(pacz, key=int):
        q = pacz[k]['wszystkie']; m = pacz[k].get('mAP50_koniec') or (ocena32 if k == '32' else None)
        w.append([k, pl(q['rownanie_J5'][0]), pl(q['rownanie_J5'][9]), pl(q['rownanie_J5'][10]), pl(q['sufit'][9]),
                  pl(q['autokorelacja_przyrostow']['1']), pl(q['sredni_krok_wzgledny'], 3), pl(m) if m is not None else '-'])
    st.append(tabela(w, [1.6 * cm, 1.4 * cm, 1.5 * cm, 1.6 * cm, 2.0 * cm, 3.1 * cm, 2.0 * cm, 2.5 * cm]))
    st.append(P('Tabela 2. Paczki 32, 128, 320 (458 tokenów: narzędzie plus 20 tokenów tła na obraz). Pliki: '
                'experiments/maly_vit_gesta_historia_paczka/wyniki/porownanie*/POROWNANIE.json.', 'pods'))
    st.append(P('Wynik obala tę tezę. Drgania mają tę samą wielkość przy każdej paczce, a prognoza na 10 kroków daje 0,20–0,24. '
                'Większa paczka czyni ruch nawet mniej gładkim. Powód leży w AdamW: dzieli krok przez typowy rozmiar '
                'gradientu, więc blisko dna robi pełne kroki nawet wtedy, gdy kierunek zmienia się z kroku na krok. '
                'Sieć odbija się po dnie. Większa paczka celniej wskazuje dno, ale odbijania nie uspokaja.'))

    # 8. wnioski
    st.append(P('8. Wnioski i co dalej', 'h1'))
    st += L([
        'Forma równania Langevina dla latentu jest dobra: z prawdziwą siłą kroku odtwarza 10 kroków niemal idealnie.',
        'Prognoza bez przyszłych paczek kończy się na około 0,22 na 10 kroków. Granica leży w danych, nie w modelu: '
        'sprawdziliśmy 768 osobnych współczynników, jądra pamięci do 40 kroków, cechy wspólnego ruchu, pęd optymalizatora '
        f'i większe paczki. Wspólny ruch wszystkich tokenów to tylko {pl(cj["udzial_ruchu_wspolnego_w_bledzie_bez_zmian"][0])} '
        'błędu prognozy „bez zmian”.',
        f'We wczesnej fazie uczenia (kroki 0–400, 10 ziaren, co 10 kroków) prognoza na 100 kroków daje {pl(wf["skill_wszystkie_starty"])}: '
        'latent też zawraca.',
        'Większa paczka nie pomaga, bo drgania robi AdamW, a nie losowanie obrazów.',
    ])
    st.append(ramka(
        '<b>Propozycja następnego biegu (nieuruchomiony, czeka na zgodę).</b> Pęd AdamW z β<sub>1</sub> = 0,99 zamiast 0,9 '
        'pamięta około 100 kroków zamiast 10. Wtedy ruch w najbliższych 10 krokach jest w większości wyznaczony przez pęd, '
        'który znamy w chwili prognozy, a nowe paczki dokładają mało. To zmiana przepisu uczenia, nie równania. Koszt: '
        'jeden bieg na Edenie, około 5 minut GPU i 2 GB. Ryzyko: wolniejsza reakcja uczenia i możliwy spadek mAP. '
        'Kod jest gotowy: <font name="DVM" size="8">sbatch bieg_paczka.slurm pelny 32 0.99</font>.', '#fff7e6', colors.HexColor('#eda100')))

    # 9. pliki
    st.append(P('9. Gdzie są pliki', 'h1'))
    pliki = [
        ('Ten raport i jego źródła', r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\27_raport_przewidywanie_latentu'),
        ('Sufit, równania, prognoza 900→1000', r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_sufit_przewidywalnosci_100'),
        ('Biegi z paczką 128 i 320', r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_gesta_historia_paczka'),
        ('Gęsta historia (paczka 32, 22 GB)', r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_ruch_czastek_gesta_historia\wyniki\pelny\dane'),
        ('Nagranie sił (docs/23)', r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_rownanie_latentu\nagranie_s47_sily'),
        ('Eden: biegi paczek', '~/smallvit_runs/2026-10-03_maly_vit_gesta_paczka (zadania 1820332, 1820333, 1820334)'),
    ]
    st.append(tabela([['co', 'gdzie']] + [[a_, Paragraph(b_, S['mono'])] for a_, b_ in pliki], [4.6 * cm, SZER - 4.6 * cm]))
    st.append(Spacer(1, 6))
    st.append(P('Odtworzenie: <font name="DVM" size="8">py -3.11 wykresy.py</font>, potem <font name="DVM" size="8">py -3.11 zbuduj_pdf.py</font> '
                'w katalogu raportu. Opisy wyników: WYNIKI*.md w obu katalogach eksperymentów.', 'pods'))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title='Przewidywanie latentu małego ViT w trakcie uczenia', author='SmallViTAblation, raport 27',
                            subject='Równanie Langevina dla latentu ViT: pomiar, wyniki, architektura')
    doc.build(st, onFirstPage=stopka, onLaterPages=stopka)
    print('zapisano', OUT)


if __name__ == '__main__':
    main()
