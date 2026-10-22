"""Podpowiedzi (dymki) wszystkich kontrolek panelu parametrów i podsekcji rzutów.

`zastosuj(z)` (z = zbudowana `ZakladkaLaserow`) uzupełnia puste i zdawkowe dymki słownikiem `DYMKI` (nazwa atrybutu → opis),
a potem kopiuje dymek pola na etykietę wiersza QFormLayout, żeby wyjaśnienie pokazywało się też po najechaniu na nazwę parametru.
Dobre, już istniejące dymki (dłuższe niż `PROG_ZDAWKOWY` znaków albo wieloliniowe) zostają bez zmian.

Słowniczek: lasery = 768 neuronów MLP bloku 6 (reflektory na szklanej sferze), koloidy = 600 tokenów (patchy zdjęcia),
pułapka optyczna = ognisko wiązek wszystkich laserów wokół jednego koloidu, skok = pchnięcie koloidu przez MLP bloku 6
w jednym kroku uczenia, checkpoint = zapisany stan wag (oś czasu uczenia)."""
from PyQt6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QFormLayout, QLabel, QLineEdit, QPushButton, QToolButton, QWidget)

PROG_ZDAWKOWY = 55          # dymek jednoliniowy do tylu znaków uchodzi za zdawkowy (zostaje zastąpiony opisem z DYMKI)
ZNACZNIK = 'dymek_z_dymki'      # właściwość widżetu, któremu dymki.zastosuj już ustawiło opis (powtórne wywołanie go nie rusza)
TYPY = (QComboBox, QAbstractSpinBox, QCheckBox, QLineEdit, QPushButton, QToolButton)

DYMKI = {
    # ---------------------------------------------------------------- stereo, dane
    'ch_stereo_zam': 'Zamienia lewe i prawe oko w trybie stereo. Włącz, gdy na monitorze 3D głębia jest odwrócona '
                     '(to, co ma być z przodu, widać z tyłu).',
    'cb_historia': 'Historia uczenia: katalog z checkpointami k<krok>.pt jednego biegu małego ViT. Cała oś czasu (sfera, pułapka, rzuty) '
                   'pokazuje właśnie ten bieg. Po zmianie kliknij „Policz / wczytaj” (policzona wcześniej historia wczytuje się z cache od razu).',
    'sp_do': 'Ostatni krok uczenia na osi czasu: checkpointy z większym numerem są pomijane. Wartość 10 000 000 oznacza brak ograniczenia.',
    'bt_licz': 'Liczy na GPU dane do wszystkich widoków dla wybranej historii i zdjęcia: latent 600 × 768, położenia reflektorów i koloidów, '
               'atencję, ramkę modelu i składowe ruchu. Wynik zostaje w cache_uczenia/lasery, więc ten sam wybór wczytuje się potem od razu '
               '(pełna historia 35 000 kroków liczy się kilka minut).',
    'bt_stop': 'Przerywa trwające liczenie historii. Aktywny tylko w czasie liczenia.',
    'lb_gt': 'Opis ramki narzędzia (GT) wybranego zdjęcia i informacja, czy zdjęcie pochodzi ze zbioru uczącego.',
    # ---------------------------------------------------------------- okienka 1–3 (rzut płaski)
    'sp_rozm': 'Wielkość punktów tokenów w okienkach 1–3 w pikselach. Większe = lepiej widoczne, ale zlewają się w gęstych miejscach.',
    'ch_refl': 'Pokazuje reflektory (768 neuronów bloku 6) jako punkty na zewnętrznej sferze. Wyłączenie zostawia same koloidy i wiązki.',
    'sp_jas': 'Jasność „studni” (jasnych plam wokół centrów) w okienkach 1–3. Zwiększ, gdy studnie giną na tle zdjęcia.',
    'sp_sig': 'Szerokość studni: odchylenie σ gaussa w komórkach siatki zdjęcia (film 2: 0,26). Większe σ = szersze, bardziej rozmyte studnie.',
    'sp_strz': 'Mnożnik długości strzałek od adresu (położenie w kroku 0) do centrum studni w okienkach 1–3. 1 = prawdziwa długość.',
    'sp_zbl': 'Szerokość kadru okienka 3 w komórkach siatki. Mniej = większe zbliżenie na narzędzie.',
    'ch_kamera': 'Okienko 3 przesuwa kadr tak, żeby narzędzie było w środku. Wyłączone: kadr stoi w miejscu.',
    'ch_narz2': 'Rysuje pierścienie wokół tokenów należących do narzędzia (według etykiety z sekcji „Etykiety”) w okienkach 1–3.',
    # ---------------------------------------------------------------- sfera: wiązki
    'cb_wiazki': 'Które wiązki rysować na sferze: brak, wszystkie (768 × 600), tylko do wybranych cząsteczek albo tylko z wybranych reflektorów. '
                 'Wybór pojedynczych wiązek odsłania szczegóły, które wszystkie naraz zasłaniają.',
    'sp_grub_w': 'Grubość linii wiązek laserów i linii uwagi na sferze w pikselach.',
    'ch_sfera': 'Pokazuje siatkę szklanej sfery, na której stoją reflektory. Po wyłączeniu krycie obok jest nieaktywne.',
    'sp_sfera': 'Krycie siatki szklanej sfery: 0 = niewidoczna, 1 = pełna.',
    'cb_skala': 'Skala chmury koloidów na sferze: dopasowana do bieżącego kroku (chmura zawsze wypełnia sferę tak samo) albo stała z końca uczenia '
                '(widać, jak chmura rośnie i kurczy się w czasie).',
    'sp_tok': 'Numer wybranej cząsteczki (patcha zdjęcia, 0–599). Dla niej liczą się wiązki do cząsteczki, pułapka optyczna (okienko 11), '
              'dolina (okienko 12), wektory i strzałki. Przycisk „narzędzie” ustawia ją na środek ramki narzędzia.',
    'sp_refl': 'Numer wybranego reflektora (neuronu MLP bloku 6, 0–767). Pokazują go wiązki „z wybranych reflektorów”, igła obrotu, '
               'mechanika w okienku 13 i mapa natężenia w okienku 5.',
    # ---------------------------------------------------------------- sfera: koloidy, wektory, ścieżki
    'sp_rozm3': 'Wielkość punktu koloidu (tokenu) na sferze w pikselach. Większe punkty są lepiej widoczne, ale zasłaniają wiązki.',
    'sp_rozm_pul': 'Średnica poświaty średniej skoków koloidu na sferze w pikselach. Większa = bardziej rozmyta chmura średnich.',
    'sp_jas_pul': 'Jasność poświaty średnich skoków koloidów. Mniej = poświata mniej zasłania koloidy.',
    'cb_wek_dla': 'Dla których cząsteczek rysować wektory i drogi: wszystkich, tylko narzędzia albo tylko wybranych '
                  '(„Cząsteczka” i „+ cząsteczki”).',
    'sp_skala_wek': 'Mnożnik długości wektorów ruchu na sferze (długość drogi zawsze w skali 1). Większy = łatwiej zobaczyć małe ruchy.',
    'sp_dl_obr': 'Mnożnik długości igieł i strzałek obrotu przy reflektorach na sferze. 1 = domyślna długość; zwiększ, gdy strzałki są za krótkie.',
    'sp_wyg_okno': 'Liczba ostatnich checkpointów w średniej ważonej (wagi okno, okno−1, …, 1). Większe okno = gładsza, ale bardziej opóźniona droga.',
    'sp_wyg_alfa': 'Współczynnik EMA: nowa wartość = α·x + (1 − α)·poprzednia. Mniejsze α = silniejsze wygładzenie i większe opóźnienie.',
    # ---------------------------------------------------------------- zapis przejść przez sieć
    'sp_prz_co': 'Zapis wszystkich przejść: co ile checkpointów (ostatni zawsze). Większe = mniej plików i mniej miejsca na dysku.',
    'bt_prz': 'Liczy i zapisuje na dysk przejście przez sieć dla wielu checkpointów historii (co n według pola obok). '
              'Zapisane przejścia wczytują się potem zamiast się liczyć.',
    'bt_prz_stop': 'Przerywa zapisywanie przejść. Aktywny tylko w czasie zapisu.',
    # ---------------------------------------------------------------- pułapka: liczenie
    'sp_og_G': 'Liczba węzłów siatki pola na bok (G³ punktów w komórce cząsteczki). Więcej = dokładniejsze pole i gęstsze strzałki, ale wolniej.',
    'bt_og': 'Wczytuje z cache albo liczy w tle pułapkę dla wybranej cząsteczki i bieżących ustawień siatki (włącza też podgląd, jeśli był wyłączony).',
    'bt_og_stop': 'Przerywa liczenie pułapki w tle. Aktywny tylko w czasie liczenia.',
    'sp_pom_co': 'Pomiar pułapki w podkrokach: co ile checkpointów liczyć (ostatni zawsze). Większe = szybciej, ale rzadsza próbka historii.',
    'bt_pom_stop': 'Przerywa pomiar pułapki w podkrokach dla całej historii. Aktywny tylko w czasie pomiaru.',
    # ---------------------------------------------------------------- pułapka: wiązki
    'sp_og_gw': 'Mnożnik jasności wiązek w okienku 11 (najjaśniejsza wiązka = 1 przed mnożeniem). Większy = wyraźniejsze wiązki.',
    'sp_og_dlw': 'Mnożnik długości wiązek w zbliżeniu. Większy = wiązki dłuższe, ognisko wydaje się mniejsze.',
    'sp_og_jrefl': 'Mnożnik jasności wiązek od reflektorów (widok w skali sfery).',
    'ch_og_linie': 'Dorysowuje cienkie linie wiązek od reflektorów do cząsteczki, niezależnie od światła ogniska.',
    # ---------------------------------------------------------------- pułapka: światło ogniska
    'sp_og_jsw': 'Mnożnik jasności światła ogniska. Większy = jaśniejsza poświata wokół cząsteczki.',
    'sp_og_szer': 'Przewężenie wiązki w ognisku jako ułamek jej długości L. Mniej = ostrzejsze, mniejsze ognisko.',
    'ch_og_osieo': 'Rysuje osie kształtu ogniska (białe odcinki): kierunki, w których światło jest najbardziej rozciągnięte; wartości w opisie okienka.',
    # ---------------------------------------------------------------- pułapka: obrys
    'sp_og_jobr': 'Krycie powierzchni obrysu ogniska: 0 = niemal przezroczysta, 1 = pełna.',
    # ---------------------------------------------------------------- pułapka: wkłady
    'sp_og_dlwk': 'Mnożnik długości strzałek wkładów do skoku (kierunki zawsze prawdziwe).',
    # ---------------------------------------------------------------- pułapka: mgła i strzałki
    'sp_og_jas': 'Jasność mgły lokalnego pola. Większa = wyraźniejsza mgła, ale może zasłaniać strzałki i koloid.',
    'sp_og_tlo': 'Jasność pola poza ogniskiem (0 = widać tylko samo ognisko).',
    'sp_og_pods': 'Rysuje strzałkę co tyle węzłów siatki. Większa wartość = rzadsze, czytelniejsze strzałki.',
    'sp_og_tempo': 'Tempo płynięcia strzałek: liczba pełnych przebiegów na sekundę.',
    # ---------------------------------------------------------------- pułapka: oznaczenia w kadrze
    'ch_og_skok': 'Rysuje pomarańczowy skok koloidu w bloku 6: pchnięcie, jakie koloid dostaje od wszystkich wiązek w tym checkpoincie.',
    'sp_og_jtlo': 'Mnożnik jasności cząsteczek w tle okienka 11. Mniej = tło mniej odwraca uwagę od ogniska i koloidu.',
    # ---------------------------------------------------------------- pułapka: ślad, kadr, odtwarzanie
    'cb_og_slad': 'Ślad w okienku 11: brak, tylko droga koloidu albo droga koloidu i pułapki (punktu zera pola).',
    'cb_og_uklad': 'Układ śladu: w układzie laboratorium (pułapka się rusza) albo względem pułapki (pułapka stoi, rusza się koloid).',
    'ch_og_sfera': 'Pokazuje pułapkę (jej ognisko) także na dużej sferze reflektorów w okienku sfery.',
    # ---------------------------------------------------------------- mapy
    'cb_mapa': 'Mapa na zdjęciu w okienku 5: odpływ centrów, drgania, atencja (średnia z 6 bloków albo wybrany blok) '
               'albo łączne natężenie wybranych reflektorów.',
    'sp_krycie': 'Krycie map na zdjęciu (okienka 4 i 5): 0 = zdjęcie bez mapy, 1 = pełna mapa.',
    'ch_pierw': 'Pierwiastkuje mapę atencji (jak film 16): wzmacnia słabe wartości, więc widać także mało uwagi.',
    'ch_rm': 'Rysuje na mapach błękitną ramkę narzędzia przewidzianą przez model.',
    'ch_rg': 'Rysuje na mapach białą prawdziwą ramkę narzędzia (z adnotacji).',
    # ---------------------------------------------------------------- dolina
    'sp_dol_p': 'Próg pewności p = σ(logit): dolina to miejsca, gdzie głowa jest pewna co najmniej w tym stopniu.',
    'sp_dol_iou': 'Próg IoU: dolina to miejsca, gdzie ramka tej cząsteczki pokrywa się z prawdziwą ramką co najmniej w tym stopniu.',
    'ch_dol_mgla': 'Rysuje punkty doliny jako mgłę: im jaśniejszy punkt, tym dalej nad progami pewności i IoU.',
    'ch_dol_obrys': 'Rysuje brzeg doliny: powierzchnię, na której pewność lub IoU dotyka progu.',
    'sp_dol_jas': 'Jasność mgły doliny pewności i IoU. Większa = wyraźniejszy kształt doliny, ale mgła szybciej się zlewa.',
    'sp_dol_rozm': 'Wielkość punktów mgły doliny w pikselach. Większe punkty zlewają się w jednolitą bryłę, mniejsze pokazują rozkład punktów siatki.',
    'ch_dol_sfera': 'Rysuje dolinę także na sferze reflektorów, wokół wybranej cząsteczki.',
    'ch_dol_kamera': 'Kamera okienka 12 powoli obraca się wokół doliny, co ułatwia zobaczenie jej kształtu 3D.',
    # ---------------------------------------------------------------- mechanika
    'ch_mech_cel': 'Rysuje celownik w_c: kierunek wagi wybranego reflektora (neuronu c), czyli igłę, na którą działa odrzut wiązek.',
    'ch_mech_prog': 'Rysuje suwak progu b_c wybranego reflektora (bias neuronu): pokazuje, jak daleko jest do włączenia neuronu.',
    'ch_mech_nap': 'Pokazuje napisy z wartościami (kąty, siły, jasności) na rysunku mechaniki.',
    # ---------------------------------------------------------------- okienko P (przejście przez sieć)
    'sp_gain_a': 'Mnożnik jasności linii uwagi (razem ze średnią atencją głów).',
    'sp_skala_sil': 'Mnożnik długości wektorów sił uwagi f_ij w okienku P. Większy = dłuższe strzałki, łatwiej zobaczyć słabe siły.',
    'sp_skala_l': 'Mnożnik długości wektorów laserów MLP w okienku P. Większy = dłuższe strzałki, łatwiej zobaczyć słabe lasery.',
    'sp_skala_wyp': 'Mnożnik długości wypadkowych wektorów (suma sił uwagi albo laserów) w podkrokach, na sferze i w okienku P.',
    'sp_sync_co': 'Przy odtwarzaniu z synchronizacją: po pełnym przejściu przez sieć skacze co tyle checkpointów. 1 = każdy checkpoint, większe = szybsze odtwarzanie historii.',
    'cb_kolor_cz': 'Kolor cząsteczek: kolor patcha ze zdjęcia, położenie patcha na obrazie, narzędzie albo obszary oka (narzędzie, źrenica, rogówka, tło).',
    # ---------------------------------------------------------------- sekcja „Podkroki” i pasek czasu
    'sp_pod_tempo': 'Tempo animacji podkroków: ile z 12 podkroków przejścia przez sieć przypada na sekundę.',
    'bt_okna': 'Menu okienek: pokazuje i chowa okienka widoków, a także zapamiętuje i przywraca układy okienek.',
    # ---------------------------------------------------------------- etykiety
    'sp_etyk_prog': 'Próg pokrycia patcha narzędziem (0–1) w trybie „pokrycie ≥ próg”. Większy = tylko patche w większości zajęte przez narzędzie.',
}

# kontrolki z.rzuty (osobna przestrzeń nazw: z.rzuty.sp_rozm to nie z.sp_rozm, z.rzuty.ch_kamera to nie z.ch_kamera)
DYMKI_RZUTY = {
    'bt_rz_stop': 'Przerywa liczenie stanów 192D. Aktywny tylko w czasie liczenia.',
    'sp_B_skala': 'Mnożnik długości wiązek (silnej sieci) w ramie siły R4. Większy = dłuższe strzałki wiązek względem siły netto.',
    'ch_osie': 'Rysuje osie rzutu z nazwami, żeby było widać, jaki kierunek reprezentuje każda z trzech osi.',
    'sp_rozm': 'Wielkość punktów cząsteczek w rzutach R1–R7 w pikselach.',
    'ch_C': 'Liczy ramę Schura (R5) automatycznie przy każdej zmianie checkpointu i cząsteczki, o ile okienko R5 albo R8 jest widoczne. Wyłącz, jeśli przeszkadza w płynności.',
    'cb_C_para': 'Którą parę zespoloną wartości własnych jakobianu pokazać w R5: najszybszy obrót, najmniej stabilną albo najbliższą kierunkowi siły F̂.',
    'cb_C_rzecz': 'Który mod rzeczywisty jakobianu pokazać w R5: najsilniej wypychający, najsilniej ściągający albo najbliższy sile F̂.',
    'cb_C_pole': 'Jakie pole rysować w R5: dokładne (F(x + Qu) − F(x), z nieliniowościami) albo jego część liniowa (QᵀJQ)u.',
    'sp_C_strz': 'Mnożnik długości strzałek pola w ramie Schura R5. Większy = dłuższe strzałki, łatwiej zobaczyć kształt pola.',
    'ch_W': 'R3: rysuje wiązki wybranej cząsteczki jako sieć silną jak w R4 (najsilniejsze wiązki niosące ustalony ułamek energii).',
    'sp_C_G': 'Liczba węzłów siatki pola na bok (G³ punktów) w ramie Schura R5. Więcej = dokładniejsze pole, ale wolniej.',
    'sp_D_zero': 'Liczba losowych torów zerowych (błądzenie losowe z tymi samymi długościami kroków) rysowanych dla porównania w R6.',
    'cb_P': 'Którą miarę wierności rzutu pokazać w R8: rozrzut chmury, energię sił netto, energię wiązek, długość separacji Δ albo ruch między checkpointami.',
    'cb_L_kiedy': 'Kiedy dopasować oś LDA w R9: raz na końcu uczenia (stała kamera dla całego filmu) albo od nowa na bieżącym checkpoincie.',
    'cb_L_osie': 'Jak wybrać osie 2–3 w R9: największy albo najmniejszy rozrzut wewnątrz klas, PCA tła lub PCA narzędzia. Oś 1 to zawsze oś LDA.',
    'sp_L_bl': 'Na ile ciągłych bloków dzielona jest pula checkpointów w walidacji blokowej R9. Więcej = ostrzejsza walidacja, mniej danych w bloku.',
    'cb_R12_widok': 'Układ 3 osi w R12 (kNN): PCA chmury bez etykiet albo oś LDA jak w R9.',
    'sp_R12_k': 'Liczba najbliższych sąsiadów k w R12. „auto” = min(10, n − 1), gdzie n to liczba tokenów narzędzia.',
    'ch_R12_kr': 'Rysuje w R12 linie od tokenów narzędzia do ich najbliższych sąsiadów.',
    'ch_R0_siatka': 'Rysuje w R0 siatkę patchy 16 × 16 pikseli, żeby było widać granice tokenów.',
}

# kontrolki bez atrybutu (widżety lokalne w kodzie budującym): klucz = tekst kontrolki
DYMKI_PO_TEKSCIE = {
    'Inny katalog wag…': 'Wskazuje dowolny katalog z checkpointami k<krok>.pt spoza listy historii i dodaje go do wyboru jako „katalog: <nazwa>”.',
    'Zdjęcie z pliku…': 'Wczytuje własne zdjęcie z pliku (jpg, png; rozciągnięte do 480 × 320). Takie zdjęcie nie ma ramki z manifestu zbioru uczącego.',
}


def _nazwane(z):
    """(widżet, nazwa atrybutu, opis) dla każdego widżetu-atrybutu z (słownik DYMKI) i z.rzuty (DYMKI_RZUTY) z wpisem w słowniku."""
    for obj, opisy in ((z, DYMKI), (getattr(z, 'rzuty', None), DYMKI_RZUTY)):
        if obj is None:
            continue
        for nazwa, w in list(vars(obj).items()):
            if isinstance(w, QWidget) and nazwa in opisy:
                yield w, nazwa, opisy[nazwa]


def zdawkowy(w):
    """Czy dymek kontrolki jest pusty albo zdawkowy: pusty, równy tekstowi wybranej pozycji listy (automat z ZakladkaLaserow)
    albo jednoliniowy i krótki (jedna fraza)."""
    t = w.toolTip()
    if not t.strip():
        return True
    if isinstance(w, QComboBox) and t == w.currentText():
        return True
    return '\n' not in t and len(t) <= PROG_ZDAWKOWY


def _ustaw(w, opis):
    """Ustawia dymek. Lista wyboru dostaje opis z dopiskiem o bieżącym wyborze (dłuższe teksty pozycji są w WaskiCombo skracane), a automat
    „dymek = wybrany tekst” z ZakladkaLaserow jest odłączany, bo nadpisywałby opis przy każdej zmianie wyboru."""
    if isinstance(w, QComboBox):
        try:
            w.currentTextChanged.disconnect(w.setToolTip)
        except (TypeError, RuntimeError):
            pass
        w.setToolTip(f'{opis}\n\nWybrano: {w.currentText()}')
        w.currentTextChanged.connect(lambda t, w=w, opis=opis: w.setToolTip(f'{opis}\n\nWybrano: {t}'))
    else:
        w.setToolTip(opis)


def _sekcje(z):
    """Korzenie przeszukiwania: wszystkie sekcje panelu (z._sek, w tym zagnieżdżone Zwijana w z.rzuty) oraz rzuty."""
    wynik = list(getattr(z, '_sek', {}).values())
    if getattr(z, 'rzuty', None) is not None and z.rzuty not in wynik:
        wynik.append(z.rzuty)
    return wynik


def zastosuj(z):
    """Uzupełnia dymki kontrolek (puste i zdawkowe) i kopiuje je na etykiety wierszy formularzy. Zwraca liczbę kontrolek,
    którym zmieniono dymek (etykiety: `etykiety_z_dymkami`). Powtórne wywołanie nic już nie zmienia."""
    zmienione = 0
    for w, _, opis in _nazwane(z):
        if not w.property(ZNACZNIK) and zdawkowy(w):
            _ustaw(w, opis); w.setProperty(ZNACZNIK, True); zmienione += 1
    for sek in _sekcje(z):
        for w in sek.findChildren(QWidget):
            if isinstance(w, (QPushButton, QToolButton)) and not w.toolTip().strip():
                opis = DYMKI_PO_TEKSCIE.get(w.text())
                if opis:
                    w.setToolTip(opis); zmienione += 1
    etykiety_z_dymkami(z)
    return zmienione


def _dymek_pola(el):
    """Pierwszy niepusty dymek pola wiersza: samego widżetu, a dla układu albo kontenera: pierwszego jego widżetu z dymkiem."""
    if el is None:
        return ''
    w = el.widget()
    if w is not None:
        if w.toolTip().strip():
            return w.toolTip()
        for p in w.findChildren(QWidget):
            if isinstance(p, TYPY) and p.toolTip().strip():
                return p.toolTip()
        return ''
    uklad = el.layout()
    if uklad is not None:
        for i in range(uklad.count()):
            t = _dymek_pola(uklad.itemAt(i))
            if t:
                return t
    return ''


def etykiety_z_dymkami(z):
    """Kopiuje dymek pola na etykietę wiersza QFormLayout (tylko etykiety bez własnego dymka); zwraca liczbę zmienionych etykiet."""
    n = 0
    for sek in _sekcje(z):
        for f in sek.findChildren(QFormLayout):
            for r in range(f.rowCount()):
                et = f.itemAt(r, QFormLayout.ItemRole.LabelRole)
                if et is None or not isinstance(et.widget(), QLabel) or et.widget().toolTip().strip():
                    continue
                t = _dymek_pola(f.itemAt(r, QFormLayout.ItemRole.FieldRole))
                if t:
                    et.widget().setToolTip(t); n += 1
    return n
