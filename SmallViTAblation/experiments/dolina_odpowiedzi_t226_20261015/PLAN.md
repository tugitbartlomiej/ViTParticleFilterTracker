# Dolina odpowiedzi cząsteczki 226

Cel: pokazać na rzeczywistych wagach, czym różni się zbiór identycznej odpowiedzi głowy od obszaru wystarczająco dobrej detekcji. Zachować ciemną stylistykę filmów laserów, świecącą cząsteczkę, strzałki i zsynchronizowaną ramkę na zdjęciu.

Źródło: bieg s47, checkpoint 65640, zdjęcie z historii eafb5a973be3d490. Stan po ostatnim bloku, przed końcową LayerNorm: 192 współrzędne. To inne miejsce sieci niż 768 aktywacji ukrytej warstwy MLP. Głowa i LayerNorm zamrożone.

## Pomiar

1. Odtworzyć prawdziwe wyjście dla tokenu 226; użyć rzeczywistego epsilon końcowej LayerNorm.
2. Znaleźć przykładowe położenie x*, dla którego p=0,995 i ramka jest zgodna z GT. Nie nazywać go jedynym celem ani najbliższym punktem.
3. Policzyć dokładny jakobian 5×192, jego rząd, 187 wektorów bazy jądra oraz 5 kierunków normalnych. Zweryfikować pochodną także różnicą skończoną.
4. Wybrać dwa kierunki styczne o widocznej krzywiźnie. Kontynuować powierzchnię x(u,v)=x*+u t1+v t2+Q c(u,v), rozwiązując pięć równań odpowiedzi w pięciu współrzędnych c. Zapisać pełne stany 192D, błędy i bazę; nie tylko rzut.
5. Wzdłuż powierzchni oraz w poprzek niej policzyć pewność, ramkę i IoU. Dodatkowo próbkować rodzinę x(u,v,delta)=x(u,v)+delta n. Obszar jakości jest tylko próbką tej rodziny, nie objętością całej przestrzeni 192D.
6. Krajobraz niezgodności: E=(logit-logit*)² + suma(((ltrb-ltrb*)/ltrb*)²). To jawnie zdefiniowana miara wizualizacji, a nie oryginalna strata treningowa ani energia pola laserów.

## Animacja

- Najpierw powierzchnia i dwa kierunki styczne. Obok zdjęcie, GT, ramka detektora, pewność i IoU.
- Ruch po zakrzywionej powierzchni: zmiana położenia, odpowiedź stała.
- Ruch w poprzek powierzchni: odpowiedź się zmienia, prosta styczna działa tylko lokalnie.
- Próbkowany obszar p>=0,99 i IoU>=0,90. Suwaki progów w samodzielnym widoku interaktywnym.
- Dodatkowy widok doliny E: wysokość jest niezgodnością odpowiedzi, nie współrzędną latentu. Wyraźnie odróżnić oba znaczenia wysokości.

Kolory: błękit/turkus = identyczna odpowiedź, złoto = cząsteczka, pomarańcz = ruch zmieniający odpowiedź, fiolet = wybrany kierunek normalny. Dwa widoczne kierunki są wycinkiem 187-wymiarowej przestrzeni stycznej; 187 kafelków pokazuje pełną liczbę zweryfikowanych kierunków.

## Wyniki i weryfikacja

Kod w tym katalogu, wszystkie wyniki w outputs/dolina_odpowiedzi_t226_20261015/. Dostarczyć MP4, planszę PNG, obracany widok HTML, NPZ ze stanami i JSON/Markdown z metodą oraz ograniczeniami. Sprawdzić reszty równań, zgodność ręcznej głowy z modułem PyTorch, zmienność odpowiedzi poza powierzchnią, parametry i pełne dekodowanie filmu. Obejrzeć klatki z każdej fazy. Nie zmieniać aplikacji ani starych wyników.
