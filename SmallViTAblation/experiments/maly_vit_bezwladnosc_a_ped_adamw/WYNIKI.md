# Bezwładność ruchu cząstek a pęd AdamW (2026-10-07)

Pytanie: czy bezwładność w prawie ruchu cząstek małego ViT, v(t+1) ≈ 0,87 v(t) + S(t), wynika z pędu optymalizatora
AdamW (β1 = 0,9), i czy paczki uczenia da się odtworzyć dokładnie, żeby liczyć pchnięcie pojedynczych obrazów.
Bez uczenia nowych modeli; tylko nagrania i zapisane wagi.

## 1. Odtworzenie paczek z ziarna (`kod/sprawdz_paczki.py`, `app/paczki_uczenia.py`)

- Losowanie paczek, odbić i jasności jak w `trening.Uczen.kroki` (generator `default_rng(ziarno)`, nic innego go nie zużywa).
- Nagranie `maly_vit_rownanie_latentu/nagranie_s47_sily` (zapisane `paczka_idx`, `paczka_odb`, `paczka_a`): przy
  przepisie nagrania (kolejka zerowana co 500 kroków, nowe ziarno 1000·47 + 3 od kroku 1000) wszystkie 3000 paczek
  zgodne co do indeksu, odbicia i jasności (różnica 0). Gęsta historia (bez zerowania) jest z nim identyczna w krokach
  0-499, jak powinna.
- Gęsta historia (`maly_vit_ruch_czastek_gesta_historia`): 10 kroków AdamW od zapisanego k0000 na odtworzonych paczkach
  daje wagi różne od zapisanego k0010 o 1,6e-5 względnie, czyli 0,017 % ruchu wag w tych 10 krokach (szum bf16).
  Kontrola z innym ziarnem (48): 0,69 % ruchu, 40 razy więcej. Inicjalizacja `Uczen(47)` = k0000 (4,6e-8).
  Wniosek: paczki gęstej historii są znane dokładnie, a uczenie da się powtórzyć lokalnie (stan AdamW w każdym kroku:
  `app/stan_uczenia.py`).

## 2. Czy bezwładność to pęd AdamW (`kod/bezwladnosc.py`, `WYNIKI_bezwladnosc.json`)

Dane: nagrania sił s47 (kroki 900-3000) i s48 (1500-3000): latent narzędzia 180 sond z narzędziem i rozkład każdego
kroku na części AdamW (nowa paczka, pęd, zanik wag) policzony przez podstawienie wag. Jeden współczynnik wspólny dla
kanałów i sond, najmniejsze kwadraty.

| | s47 900-1500 | s47 1500-3000 | s48 1500-3000 |
|---|---:|---:|---:|
| a: bezwładność obserwowana, Δz_t = a Δz_(t−1) + nowa + zanik | 0,844 | 0,817 | 0,820 |
| b: siła pędu = b · poprzednia prędkość | 0,851 | 0,826 | 0,829 |
| trafność B (pęd wyjaśniony poprzednią prędkością) | 0,920 | 0,874 | 0,883 |
| c: siła pędu = c · (pęd + nowa) z kroku wcześniej | 0,854 | 0,833 | 0,840 |
| ruch wyjaśniony sumą części AdamW | 0,975 | 0,966 | 0,964 |
| ruch wyjaśniony bez pędu (tylko nowa + zanik) | 0,203 | 0,198 | 0,195 |
| norma siły pędu / ruch | 0,89 | 0,88 | 0,88 |
| norma nowej paczki / ruch | 0,60 | 0,56 | 0,57 |

Wnioski:
- Bezwładność to pęd AdamW: siła pędu jest w 87-92 % proporcjonalna do poprzedniej prędkości cząstki, a jej
  współczynnik (0,83-0,85) jest równy obserwowanej bezwładności (0,82-0,84).
- W wagach pęd to dokładnie β1 = 0,9 razy poprzednia aktualizacja (z poprawką mianownika). W latencie wychodzi
  0,83-0,85: część pędu gubi się przez zmianę mianownika (drugi moment) i jakobianu sieci między krokami.
- Bez pędu równanie wyjaśnia tylko 20 % ruchu; pęd jest większy od pchnięcia nowej paczki (0,88 wobec 0,57 ruchu).
- Wartość 0,87 z gęstej historii dotyczy pojedynczych tokenów i kroków 300-700; tu średnia narzędzia sond i kroki
  900-3000, stąd nieco niższa liczba. Ten sam mechanizm.

Czego to nie dowodzi: nie mówi, ile pędu przypada na które paczki w przeszłości (to wymaga rozwinięcia m_t na
historię gradientów), ani że inny β1 zmieniłby jakość detekcji.

## Pliki

- `kod/sprawdz_paczki.py` -> `WYNIKI_sprawdz_paczki.json`
- `kod/bezwladnosc.py` -> `WYNIKI_bezwladnosc.json`
- kod wielokrotnego użytku w aplikacji: `app/paczki_uczenia.py`, `app/stan_uczenia.py`, `app/pchniecie.py`
