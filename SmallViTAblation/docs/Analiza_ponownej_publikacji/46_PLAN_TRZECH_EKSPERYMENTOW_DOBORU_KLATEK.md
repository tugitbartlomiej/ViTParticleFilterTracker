# 46. Plan trzech eksperymentów doboru klatek z latentu tożsamości

Data: 2026-09-23. Status: plany zatwierdzone do zapisu przez użytkownika; nic z nowych eksperymentów nie uruchomiono.
Każdy trening wymaga osobnej zgody.

## Punkt wyjścia

- W tokenach backbone RF-DETR Nano jest jeden kierunek, który zapala się tam, gdzie widać cystotom (wektor tożsamości).
  Znajduje się też bez etykiet (AUC 0,975). Jest domknięty wobec obrotu o 30 stopni i zmiany długości
  (zapala się w 97–98 procentach klatek), słabiej wobec obcego tła (81 procent).
- Na 25 operacjach testowych najlepsza epoka detektora z 1500 klatek trafia narzędzie w 91 procentach klatek,
  mAP 0,68, tyle co na walidacji. Ostatnia epoka jest gorsza (0,65), a fałszywe alarmy rosną z 40 do 50 procent.
  Uczenie kończy się realnie po 6–7 epokach.
- Większość pewnych chybień to inne cienkie narzędzia (pęseta, drugi instrument) i nietypowe tło.
- Zaskoczenie Koopmana liczone na latencie całego kadru nie wskazuje chybień (AUC 0,58, gorzej niż zwykła różnica
  klatek 0,60). Wektor tożsamości wskazuje je z AUC 0,76, ale tylko pod obrysem z adnotacji.
- Pomiary na teście posłużyły do diagnozy. Dalszy rozwój metod prowadzimy na walidacji, test zostaje do oceny końcowej.

## Trzy eksperymenty

| Nr | Katalog | Pytanie | Pierwszy etap |
|---|---|---|---|
| 1 | `experiments/wybieracz_tozsamosc_koopman` | czy wektor tożsamości, zaskoczenie Koopmana i poprawka Langevina, z różnorodnością, wybierają klatki lepsze od losowych | offline na walidacji: wektor bez etykiet, Koopman w układzie narzędzia |
| 2 | `experiments/koopman_dynamika_uczenia` | czy Koopman na przebiegu uczenia (latent klatki w funkcji kroku, jak Dogra i Redman 2020 dla wag) wskazuje klatki, które dalej uczą sieć, i przyspiesza trening | bieg z sondą na klatkach uczących (wymaga zgody) |
| 3 | `experiments/wlokno_pomylek` | jakie obrazy detektor widzi tak samo jak cystotom (włókno, Rousselot i in. 2026) i czy kierunki błędów z nich wybierają dobre klatki | włókno w prawdziwej puli, bez generatora, lokalnie; model dyfuzyjny odłożony |

Szczegóły, miary, kryteria sukcesu i ograniczenia są w `PLAN.md` każdego katalogu.

## Wspólne zasady

- Równe budżety (250 klatek), ta sama pula i trzy ziarna (42, 1337, 2024) we wszystkich treningach.
- Odniesienie: losowe 250 klatek, mAP 0,6319. Sukces to przewaga większa niż rozrzut między ziarnami.
- Najpierw test offline bez uczenia sieci; trening tylko po jego przejściu i po zgodzie użytkownika.
- Wyniki każdego eksperymentu zapisujemy w jego własnym katalogu.
