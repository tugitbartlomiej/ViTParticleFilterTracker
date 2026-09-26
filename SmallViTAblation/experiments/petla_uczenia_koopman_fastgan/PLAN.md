# Pętla uczenia: prognoza latentu → FastGAN → wyszukanie klatek → dodanie → uczenie, 8 obrotów

Data: 2026-09-26. Stan: **ZATWIERDZONY** („No to wykonuj i potem analizuj”); pilot: ziarno 42, trzy pętle, 8 obrotów.
Cel doprecyzowany przez użytkownika: „dzięki temu wiemy, jak najszybciej uczyć sieć” — główną miarą jest tempo
uczenia (liczba obrotów do progu mAP@0,5 i pole pod krzywą mAP po obrotach), obok wyniku końcowego.

## Realizacja (ustalona przy pisaniu kodu, przed wynikiem)

- Obrót = 165 aktualizacji, nie 103: sampler wymaga epoch_size ≥ liczby klatek, a po 8 obrotach pętla ma
  1500 + 8 × 134 = 2572 klatki. epoch_size 2640 (165 aktualizacji) jednakowy dla wszystkich pętli i obrotów,
  33 zapisy latentu na obrót, czyli co 5 aktualizacji jak w biegu 1810662.
- Latent sond bez ekstrakcji: trening.py zapisuje LATENT_*.npz (opis 192 sond); 72 sondy analizy to ich
  podzbiór, zgodność z lokalnymi tokenami 1e-6 względnie. Wagi pośrednie kasowane w zadaniu, zostaje koniec obrotu.
- Prognoza: wspólny operator na 8 ostatnich przyrostach (grzbiet λ = 0,1), osobno skalar i S_4, dopasowany
  do całej historii pętli; rollout średniej o 33 kroki. Kierunki (w_lda, S_4) stałe z OPERATORY_K73_S48.
- FastGAN lokalnie (RTX 3070), uczenie na Edenie (short/hopper, w tym węzły DGX). Kod: obrot_lokalny.py,
  dyrygent.py, eden/obrot.slurm, eden/kontynuacja_obrot.py, eden/config_seed42.json.

Polecenie użytkownika: „spróbuj zrobić szukanie latentu, generowanie zdjęć FastGAN-em, szukanie nowych zdjęć,
dodawanie do zbioru uczącego, potem uczenie z nowymi zdjęciami i znowu, przez 8 kroków tej pętli. Zobaczymy, jak
się wytrenuje sieć. Jeśli się uda, to uczenie od początku.”

## Jeden obrót pętli (k = 1..8, jeden obrót = jedna epoka = 103 aktualizacje)

1. **Stan.** Detektor po obrocie k−1 (dla k = 1: `model_0000245.pth` biegu 1810662).
2. **Latent.** Zadanie uczenia na Edenie zapisuje wagi co 5 aktualizacji (20 punktów na epokę), od razu liczy
   z nich opis 72 sond (średnie tokenów narzędzia i tła, ta sama funkcja co `ekstrakcja.py`) i kasuje wagi
   pośrednie. Zostają wagi końca epoki i mały plik opisu sond. Historia łańcucha: bieg 1810662 do kroku 245 plus
   wszystkie wcześniejsze obroty.
3. **Prognoza.** Równanie z dryfem Koopmana i pamięcią 8 przyrostów (`../langevin_latent_detektora/langevin.py`,
   najlepsze w teście: 0,151 wobec 0,171 dla L4) dopasowane do historii łańcucha. Cel: latent 3 kotwic
   (00, 24, 48) na koniec następnej epoki.
4. **FastGAN.** Inwersja kotwicy (start z z* z poprzedniego obrotu, mniej restartów) i krok Gaussa–Newtona do
   celu (`pokaz_fastgan_inwersja.py`, `pokaz_fastgan_gn.py`, detektor z bieżącego obrotu).
5. **Wyszukanie.** DINOv2 równoległej sesji (tylko odczyt), top-45 na kotwicę, bez klatek już użytych
   (1500 pretreningu, kotwice i wszystko dodane wcześniej): około 134 nowych klatek na obrót.
6. **Uczenie.** Jedna epoka na 1500 + wszystkie dotąd dodane klatki, świeży optymalizator (jak w 7c i 7f).

Po 8 obrotach dochodzi około 1070 klatek (71 procent zbioru bazowego).

## Kontrole (bez nich wynik jest nie do odczytania)

- **Pętla losowa:** co obrót tyle samo losowych nowych klatek.
- **Pętla bez dodawania:** te same 8 epok tym samym łańcuchem zadań.
Wszystkie trzy pętle mają identyczny łańcuch zadań i restarty optymalizatora, więc różnią się tylko danymi.

## Miary w każdym obrocie

mAP@0,5 i mAP@0,5:0,95 całego val (Eden); AP@0,5 na 134 klatkach val najbliższych kotwicom (lokalnie);
błąd prognozy (latent przewidziany w obrocie k wobec zmierzonego w k+1), czyli sprawdzian samego Koopmana;
wspólna część wyboru z wyszukiwaniem po samej kotwicy (twierdzenie 1 z `TEORIA_LANGEVIN.md`); trudność dodanych
klatek dla detektora na początku obrotu.

## Kryterium (ustalone przed wynikiem)

Po 8 obrotach pętla Koopmana jest lepsza, gdy na top-134 val wygrywa z pętlą losową w każdym ziarnie i gdy na
całym val nie traci więcej niż 0,002 mAP@0,5 wobec pętli bez dodawania. Przewidywanie teorii do sprawdzenia: wybór
pokrywa się z sąsiadami samej kotwicy w co najmniej 90 procentach, więc pętla zachowa się jak pętla sąsiadów.

## Skala i koszt

- **Pilot: jedno ziarno (42), trzy pętle, 8 obrotów.** Obroty idą po kolei, bo obrót k potrzebuje wyniku k−1.
  Na obrót: uczenie na Edenie około 15 min (w tym pełna walidacja i ekstrakcja), pobranie 2 min, FastGAN
  lokalnie 5–15 min, wyszukanie 2 min. Razem około 5–6 h zegarowo, 6 h GPU na Edenie.
- Jedno ziarno pokaże przebieg, ale nie rozstrzygnie różnic poniżej 0,01 (rozrzut między ziarnami lokalnie 0,008).
  Pełna wersja: 3 ziarna, około 9–12 h zegarowo (FastGAN lokalnie jest wąskim gardłem).
- Dysk Edena: 49 GB wolne (w tym 9 GB kampanii 5 epok do sprzątnięcia za zgodą). Pilot zostawia 3 × 8 × 120 MB
  wag końca epok plus małe pliki opisu sond, razem około 3 GB.

## Co trzeba napisać przed zgłoszeniem

1. Zadanie Edena: uczenie jednej epoki z gęstymi punktami kontrolnymi, ekstrakcja opisu sond, kasowanie wag
   pośrednich, pułapka sprzątająca (`trap`), `sed -i 's/\r$//'` na wysłanych skryptach.
2. Lokalny dyrygent (proces odłączony, jeden naraz): czeka na koniec zadania, pobiera, liczy prognozę, FastGAN,
   wyszukanie, zapisuje selekcję, wysyła i zgłasza kolejny obrót; każdy krok zapisuje wynik i da się wznowić.
3. Próba na sucho obrotu 1 bez uczenia (sprawdzenie wejść i czasów).

## Potem: uczenie od początku

Jeśli pilot pokaże zysk, ta sama pętla od wag COCO (krok 0). Teoria mówi, że tam prognoza ma większe znaczenie:
dryf latentu jest duży na początku uczenia i wygasa około kroku 500.

## Propozycja: pętla „przyspieszona” (2026-09-26, 23:25) — CZEKA NA ZATWIERDZENIE, nic nie zgłoszone

Uwaga po pilocie (2026-09-27, `WYNIKI.md`): dryf latentu tożsamości kończy się w pierwszych 2 obrotach od kroku 245
(zmiana zmierzona od obrotu 3 ≈ 0), więc cel „5 obrotów naprzód” od kroku 245 byłby prawie równy stanowi obecnemu.
Sensowne miejsce na pętlę przyspieszoną to uczenie od początku (wagi COCO, krok 0), gdzie latent jeszcze się
przesuwa. Drugi wniosek pilota: stała długość obrotu przy rosnącym zbiorze rozcieńcza powtórzenia bazy; w następnej
pętli liczba prezentacji 1500 klatek bazy na obrót powinna być stała we wszystkich pętlach.

Diagnoza użytkownika po 3 obrotach: pętla Koopmana szuka klatek zgodnych z latentem, do którego uczenie i tak
dojdzie za jeden obrót, więc nie wnosi nic nowego (zgodne z `TEORIA_LANGEVIN.md`, twierdzenie 1: cel ≈ stan
obecny). Polecenie: wygenerować FastGAN-em przypadki, które jeszcze bardziej przyspieszą uczenie latentu (chronią
przed niedouczeniem).

Zmiana względem pętli „koopman” (reszta bez zmian: ziarno 42, 165 aktualizacji na obrót, 134 klatki na obrót):
1. **Cel daleko w przód:** latent przewidziany przez to samo równanie (dryf Koopmana z pamięcią 8) na 5 obrotów
   naprzód (5 × 33 kroków), a nie na jeden. Klatki mają pokazywać to, czego sieć nauczyłaby się dopiero później.
2. **Kotwice niedouczone:** co obrót 3 sondy (oryginały, różne źródła) o największej przewidzianej drodze do
   przebycia ‖z(+5 obrotów) − z(teraz)‖, zamiast stałych kotwic 00, 24, 48.
3. Porównanie z już liczonymi pętlami tego samego ziarna: koopman (cel za jeden obrót), losowe, bez.

Miary: mAP@0,5 po obrotach i obrót dojścia do progu 0,895 (tempo), AP przy kotwicach, oraz **tempo samego latentu**:
jaka część przewidzianej drogi skalara tożsamości zostaje przebyta w obrocie, wobec pętli bez dodawania.
Kryterium (przed wynikiem): pętla przyspieszona dochodzi do 0,895 wcześniej niż bez i losowe, albo w obrotach 1–4
jej latent przebywa co najmniej 1,2 razy dłuższą drogę tożsamości niż w pętli bez.
Koszt: 8 zadań na Edenie (około 20 min każde) i lokalny FastGAN około 6–12 min na obrót; około 4 h, równolegle
z trwającą pętlą (lokalne kroki FastGAN obu pętli po kolei, blokada pliku). Jedno ziarno: tylko obraz przebiegu.
