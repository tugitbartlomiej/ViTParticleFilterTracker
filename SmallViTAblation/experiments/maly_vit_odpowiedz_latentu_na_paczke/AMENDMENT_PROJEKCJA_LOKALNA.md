# Poprawka do planu: funkcja odpowiedzi z projekcji lokalnej i horyzont do 500 kroków

Data: 2026-10-05, po odczytaniu pierwszego wyniku (`wyniki/ODPOWIEDZ.json`). PLAN.md zostaje bez zmian; ten plik
opisuje, co dodano po wyniku i dlaczego. Obie analizy są opisowe, bez kryterium ustalonego z góry.

## Co poszło nie tak z definicją z planu

Plan definiował funkcję odpowiedzi `R(n)` jako współczynnik modelu A (wszystkie pchnięcia w oknie) przy pchnięciu
sprzed `n` kroków i zakładał, że pchnięcie paczki jest losowym wstrząsem, bo paczki są losowane niezależnie od stanu
sieci. To założenie jest prawdziwe dla SKŁADU paczki, ale nie dla jej PCHNIĘCIA: pchnięcie to gradient paczki policzony
w bieżącym stanie sieci, więc zależy od tego, gdzie sieć właśnie jest. Późne pchnięcia w oknie „wiedzą” o
przesunięciu, które już zaszło, i je cofają. Skutek w modelu A: współczynnik przy tym samym opóźnieniu zależy od
długości okna (dla ostatniego pchnięcia 0,98 przy oknie 1 kroku i −0,73 przy oknie 50 kroków). Takie współczynniki
nie są funkcją odpowiedzi.

## Co dodano

1. `kod/analiza_gramow.py` (bez czytania nagrań, z `wyniki/GRAMY.npz`) → `wyniki/ANALIZA_GRAMOW.json`: korelacje
   pchnięć z wcześniejszym przesunięciem, z pędem i z odchyleniem od średniej; autokorelacja pchnięć; ile pchnięcia
   wyznacza to, co znane wcześniej.
2. `kod/odpowiedz_dluga.py` → `wyniki/dluga/ODPOWIEDZ_DLUGA.json`: funkcja odpowiedzi jako współczynnik przy
   pchnięciu bieżącej paczki w modelu C (pchnięcie bieżące, pęd, zanik wag, odchylenie od średniej), dla 22 horyzontów
   od 1 do 500 kroków. W ekonometrii ta konstrukcja nazywa się projekcją lokalną. Wszystkie regresory są znane w
   chwili podawania paczki, więc nie ma w niej późniejszych pchnięć zależnych od stanu.
   Trzy zbiory startów dopasowane osobno: ziarno 47 kroki 1000-1749, ziarno 47 kroki 1750-2499, ziarno 48 kroki
   2000-2499. Dodatkowy regresor kontrolny: pchnięcie paczki sprzed 100 kroków (placebo).

## Czego ta poprawka nie naprawia

Projekcja lokalna nadal jest regresją na zwykłym uczeniu, nie interwencją. Pchnięcie bieżącej paczki zawiera część
wyznaczoną przez stan sieci; pęd i odchylenie tylko częściowo ją kontrolują (wyjaśniają 5,5 procent pchnięcia).
Odpowiedź na samą różnicę między dwiema paczkami podanymi w tym samym stanie zmierzą dopiero pary przebiegów.
