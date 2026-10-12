# Wynik: sufit przewidywalności latentu małego ViT na 100 kroków

Data: 2026-10-03. Plan: [PLAN.md](PLAN.md). Liczby: `wyniki/pelny/METRYKI.json`, tabele `wyniki/pelny/TABELA_h{10,50,100}.md`.
Bieg: 12 obrazów, 1580 s, Python 3.11, bez treningu.

## Werdykt

W krokach 100–1000 cel 85 procent na 100 kroków jest poza zasięgiem prognozy z własnej historii latentu.
Najlepsza prosta prognoza przyczynowa daje 0,31–0,35 w MSE (0,17–0,19 w RMSE). Nawet wyrocznia znająca całą
gładką przyszłość daje 0,84–0,89 przy szerokości wygładzania 21 i 0,69–0,80 przy szerokości 51, czyli jest na
granicy celu albo poniżej. Silny, uporządkowany dryf istnieje tylko w pierwszych 100 krokach uczenia.

## Wynik główny, h = 100, MSE (1 - błąd / błąd prognozy bez zmian)

| wielkość | okres | sufit w=21 | sufit w=51 | podłoga w=21 | D²/σ² w=21 | trwałość dryfu | liniowy krzyżowo | liniowy RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tok_narz | 0-100 | 0,94 | 0,84 | 0,21 | 35,9 | - | - | - |
| tok_narz | 100-300 | 0,88 | 0,77 | 0,45 | 6,8 | -0,10 | 0,16 | 0,08 |
| tok_narz | 300-600 | 0,87 | 0,75 | 0,51 | 4,6 | -0,21 | 0,25 | 0,13 |
| tok_narz | 600-koniec | 0,89 | 0,76 | 0,54 | 5,0 | -0,34 | 0,31 | 0,17 |
| sr_narz | 600-koniec | 0,89 | 0,76 | 0,55 | 4,6 | -0,33 | 0,32 | 0,17 |
| sila_obrazu | 0-100 | 0,94 | 0,73 | -0,01 | 46,6 | - | - | - |
| sila_obrazu | 600-koniec | 0,89 | 0,77 | 0,60 | 3,8 | -0,28 | 0,35 | 0,19 |
| sila_obrazu_los | 600-koniec | 0,89 | 0,77 | 0,57 | 4,9 | -0,15 | 0,18 | 0,10 |

Pełna tabela wszystkich dziewięciu wielkości: `wyniki/pelny/TABELA_h100.md`.

## Co z tego wynika

1. **Ruch jest gładki z kroku na krok, ale nie trwały na 100 kroków.** Autokorelacja kolejnych przyrostów
   wynosi 0,62–0,88, więc pojedynczy krok przypomina poprzedni (zgodne z pędem AdamW, beta1 = 0,9, pamięć około
   10 kroków). Na 100 kroków kierunek gładkiej ścieżki odwraca się: cosinus między przesunięciem przyszłym
   a przeszłym jest ujemny we wszystkich okresach po kroku 100 (-0,01 do -0,34). Przedłużenie ostatniego dryfu
   (C1) daje wynik ujemny, około -0,9: gorzej niż prognoza bez zmian.
2. **Latent krąży wokół wolno przesuwającego się poziomu.** Model liniowy daje największe wagi średniej z ostatnich
   51–101 stanów (beta 0,4–0,7). Prognoza działa więc przez ściąganie do średniej, jak w procesie
   Ornsteina-Uhlenbecka, a nie przez przedłużanie ruchu. Sama średnia z ostatnich 101 stanów daje 0,29–0,34.
3. **Uśrednianie po tokenach i kierunek LDA nie pomagają.** `sr_narz` daje to samo co `tok_narz`, a rzut na
   losowy kierunek daje to samo co rzut na LDA (sufit 0,89 wobec 0,89; podłoga 0,57 wobec 0,60). Krążenie jest
   wspólne dla wszystkich tokenów, bo wywołuje je wspólna zmiana wag. Uśrednianie nie wycina wspólnego ruchu.
4. **Pierwsze 100 kroków jest inne.** Dryf jest tam 30–90 razy silniejszy od drgań, a sufit wyroczni wynosi
   0,93–0,96. Tylko w tym okresie cel 85 procent jest fizycznie możliwy. Prognozy przyczynowej dla tego okresu nie
   zmierzono, bo model liniowy wymaga 200 kroków historii.
5. **85 procent w RMSE jest niemożliwe w żadnym okresie po kroku 100.** Sufit w RMSE wynosi tam 0,64–0,70.

## Podłoga

Wyrocznia na samych drganiach (dryf usunięty) daje 0,45–0,60 w okresach po kroku 100. Tyle daje samo odszumienie
punktu startowego bez dryfu. Sufit z dryfem (0,87–0,89) leży wyraźnie wyżej, więc powolny ruch istnieje.
Prognoza przyczynowa (0,16–0,35) leży jednak poniżej samej podłogi: z przeszłości nie da się nawet dokładnie
ustalić bieżącego poziomu, bo poziom sam się przesuwa.

## Zastrzeżenia

- Sufit wyroczni zależy od szerokości wygładzania: w=5 daje 0,99, bo wtedy prawie cały ruch liczy się jako dryf.
  Przy autokorelacji przyrostów 0,6–0,9 drgania nie są białe, więc żadna szerokość nie oddziela ich czysto.
  Raportuję w=21 i w=51 jako rozsądny zakres.
- Sufit wyroczni zalicza do dryfu także powolne krążenie, którego nie da się przewidzieć. Ujemna trwałość dryfu
  pokazuje, że tak tu jest. Prawdziwy sufit dla prognozy z przeszłości leży więc niżej niż 0,87–0,89.
- Model liniowy to dolna granica osiągalnego: osiem cech, wspólne współczynniki, dopasowanie na obrazach 0–5,
  sprawdzenie na 6–11 i odwrotnie. Model nieliniowy z tymi samymi danymi może dać więcej, ale nie powyżej sufitu.
- Jedno ziarno (47), 12 obrazów treningowych, jeden przebieg uczenia. Ruch obejmuje zmianę statystyk BatchNorm.
- Kierunek LDA policzony w kroku 1000 (d' = 3,85). Kontrola losowym kierunkiem daje to samo.

## Czego to NIE dowodzi

- Nie dowodzi, że 85 procent jest nieosiągalne z dodatkową informacją spoza historii latentu, na przykład ze
  stanu optymalizatora (pęd AdamW) albo z wag. Pęd wyjaśnia jednak ruch tylko na około 10 kroków.
- Nie dotyczy pierwszych 100 kroków, gdzie sufit 0,93–0,96 dopuszcza cel; tam potrzebny jest osobny pomiar
  prognozy przyczynowej z krótką historią.
- Nie dotyczy innych ziaren, większej sieci ani innego harmonogramu uczenia.
