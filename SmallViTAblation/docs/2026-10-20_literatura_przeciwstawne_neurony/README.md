# Przeciwstawne neurony w trakcie uczenia: przegląd literatury

Data: 2026-10-20. Indeks katalogu.

## Pytanie

Czy ktoś mierzył przeciwstawne (antyrównoległe) neurony podczas uczenia sieci: śledził je po checkpointach i seedach, razem z interwencjami A/B/AB oraz z rozdzieleniem IoU od score?

## Odpowiedź wprost

Nie znaleziono pracy, która mierzy dokładnie to. Są cztery bliskie precedensy: pary z przeciwnymi wagami wyjściowymi w małych MLP (Mar25l, Kwo05b), opozycja warstw w pretreningu LLM (Pat25d), push-pull w sieciach spikujących (Zhu26l) oraz geometria kolumn w toy modelach superpozycji (Che23an).

## Pliki w tym katalogu

- `PRACE_PELNE_TEKSTY.md`: każda praca z tytułem, kluczem, DOI, tym, co zmierzono, kluczowymi liczbami i wnioskiem dla naszych eksperymentów; na końcu tabela oceny sześciu obserwacji.
- `PLAN_TESTU_KANALU_MAR25L.md`: szkic testu kanału Mar25l na 288 istniejących checkpointach (bez treningu). Eksperyment nie został utworzony ani uruchomiony.

Notatka źródłowa (nie zmieniana): `../../plan_na_przyszle_eksperymenty/2026-10-20_literatura_przeciwstawne_neurony_w_uczeniu.md`

## Deep searche Undermind

Workspace: „ViT: detekcja końcówki narzędzia w chirurgii zaćmy”

- Emergence of opposing antagonistic neurons during neural network training tracked across checkpoints (69 prac):
  https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Emergence%20of%20opposing%20antagonistic%20neurons%20during%20neural%20network%20training%20tracked%20across%20checkpoints
- Measuring synergy redundancy and compensation between neurons through training with interventions (132 prace):
  https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Measuring%20synergy%20redundancy%20and%20compensation%20between%20neurons%20through%20training%20with%20interventions
- Prior art check for specific measured findings on opposing MLP neuron pairs in a ViT detector across training (62 prace; w celu wyszukiwania wpisano nasze liczby z kampanii):
  https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Prior%20art%20check%20for%20specific%20measured%20findings%20on%20opposing%20MLP%20neuron%20pairs%20in%20a%20ViT%20detector%20across%20training

## Narzędzia

Undermind lepiej interpretuje zestawy wyników (cel do 30 000 znaków). Scite służy do sprawdzania cytowań i fragmentów pełnych tekstów.

## Stan czytania

- Pełne teksty: Mar25l, Che23an, Ahm26c, Ran23c, Hut25, Pat25d, Gon26h, Sam26b, Zhu26l, Urb26b, Saf26.
- Pozostałe prace: tylko abstrakt lub metadane.
