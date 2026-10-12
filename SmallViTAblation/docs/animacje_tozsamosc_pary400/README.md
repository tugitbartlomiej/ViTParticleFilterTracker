# Film: jak wyznaczono cząsteczki i kierunki tożsamości narzędzia (pary 400, LaMa)

Data: 2026-10-02. Prośba użytkownika: „Zrób animacje w manim jak wyznaczone są wektory i cząsteczki tożsamości
narzędzia. Możesz używać prawdziwych zdjęć”. Wybrana metoda: pary 400 (oryginał kontra LaMa), eksperyment
`experiments/maly_vit_tozsamosc_czasteczki_pary_400`. Bez lektora (decyzja użytkownika): każdy krok ma napis na ekranie.

Film: `videos/Tozsamosc_czasteczki_pary400.mp4` (9 scen, Full HD, 24 kl./s, bez dźwięku). Osobne sceny:
`videos/Scena1.mp4` … `Scena9.mp4`. Długości i kontrola dekodowania: `MANIFEST_FILMU.json`, klatki końcowe: `kontrola/`.

## Sceny

1. Para zdjęć: klatka 881 filmu TR_0053 (film odłożony do testu) i jej wersja z narzędziem wymazanym przez LaMa, maska.
2. Siatka 30 × 20 pól na zdjęciu; 18 cząsteczek pod maską (reguła: choćby jeden piksel), jasność = udział narzędzia; cząsteczka 280.
3. Stan cząsteczki 280: 768 aktywacji (ostatni blok, MLP po GELU) na oryginale, na obrazie bez narzędzia i ich różnica.
4. Osie o najpewniejszej zmianie (makrowektory 500, 472, 735, 512, 647, 326): średnia i rozrzut na 240 parach uczących, wartość w cząsteczce 280, zgodność znaku na 85 parach testowych.
5. Chmura zmian cząsteczek (losowa próbka 1500 z 4059 cząsteczek narzędzia i 600 z 1920 cząsteczek dalekiego tła z par uczących, bez punktów spoza ramki) na osiach 500/472, obrót do kierunków SVD, średnie zmiany; krzywa energii (8 kierunków = 60,53 %, 90 % przy 88); skład kierunku 1.
6. Odczyt obecności na 85 parach z 24 odłożonych filmów: szanse z regresji logistycznej na 8 kierunkach, próg 0,585, trafność 88,6 % (ważona filmami), LDA 95,3 %.
7. Mapa reakcji 18 cząsteczek (najmocniej 280: 6,98) i wpływ pojedynczej podmiany na pewność × IoU (tylko 248: −0,028).
8. Ingerencja w 8 kierunków kontra losowe kierunki (także po wyrównaniu normy po fc2); AP50 bez rozstrzygnięcia; kontrola z atrapą LaMa (0,16 kontra 3,56).
9. Podsumowanie drogi i trzy ograniczenia.

## Co jest pomiarem, a co rysunkiem

Pomiar: zdjęcia, maska, pokrycie pól, 768 aktywacji cząsteczki 280, rankingi osi, punkty chmury, krzywa energii,
wagi kierunku 1, szanse odczytu, mapa reakcji, wpływ podmian, wyniki detektora i kontroli. Wszystko z
`assets/dane.json`, które `przygotuj.py` wyciąga z zapisanych plików eksperymentu (sumy SHA-256 w `ZRODLA.json`).

`przygotuj.py` sprawdza zgodność z zapisem: długość rzutu cząsteczki 280 liczona z `TOKENY_PARY.npz` i bazy `U`
równa się mapie z `PODPRZESTRZEN.npz` (6,984), energia 8 kierunków równa się `IDENTYFIKACJA.json` (0,6053), a AUROC
odczytu przeliczony bez wag (0,9646) równa się zapisanemu wariantowi ważonemu obrazami.

Rysunek: siatka pól, ramki, strzałki, schemat przepływu w scenach 6 i 9. Pierwsze dwie osie chmury to tylko rzut
768 liczb na dwie osie; skala chmury sięga 99. percentyla, a punkty spoza ramki w którymkolwiek widoku są pominięte (legenda podaje, ile pokazano).

Nie liczono niczego na sieci ani na klastrze. Nie zmieniono kodu ani wyników eksperymentu.

## Odtwarzanie

W tym katalogu, kolejno:

```
python przygotuj.py
F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1\Scripts\python.exe render.py sceny.py Scena1 -w --hd   (… Scena9)
python zloz.py
```

`render.py` i `custom_config.yml` są skopiowane z `docs/animacje_pipeline_vit` (ManimGL 1.7.2 z istniejącego środowiska,
bez instalacji). `FILM_CHECK=1` skraca pauzy do szybkiej próby układu.
