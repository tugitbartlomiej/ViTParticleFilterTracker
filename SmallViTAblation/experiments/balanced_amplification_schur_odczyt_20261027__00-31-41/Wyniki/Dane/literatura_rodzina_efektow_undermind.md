# Rodzina efektów pokrewnych balanced amplification (Undermind, 2026-10-27 01:56)

Wyszukiwanie `/Balanced amplification family of effects across cortex fluids ecology photonics and learned networks` (134 prace)
w przestrzeni „ViT: detekcja końcówki narzędzia w chirurgii zaćmy” (40294fa6-…). Podsumowanie Undermind oparte na streszczeniach,
nie pełnych tekstach. Przerwane na polecenie użytkownika po pierwszym pełnym wyniku; równoległy workflow webowy zatrzymany bez
wyników, scite niedostępny (limit miesięczny 250 wywołań, reset 2026-11-13). Uzupełnia `wzmocnienie_przejsciowe_pseudospektrum/
Wyniki/Dane/literatura_undermind.md` (tabela dziedzin) i `audyt_nowosci_undermind.md`.

## Werdykt Undermind

Wspólny obiekt: **skończenie-czasowe, kierunkowo selektywne wzmocnienie w operatorze nienormalnym o stabilnym widmie**. Wartości własne
opisują zanik asymptotyczny; wzmocnienie przejściowe widzą wektory osobliwe propagatora, sprzężenia w formie Schura i pseudospektra
[Mur09, Hen12b, Tre93b, Tre05]. Balanced amplification jest **konkretną realizacją korową**, nie synonimem nienormalności: silne
pobudzenie rekurencyjne stabilizowane hamowaniem daje selektywne, szybkie przebiegi bez wolnych modów [Mur09, Hen12b]. Stabilizacja
hamowaniem i efekt paradoksalny to własności pokrewne, ale **inne diagnostyki**; same nie dowodzą łańcucha przejściowego o niskiej randze
[Oze09, San19g, Sad20f]. Forma Schura ujawnia ukryty składnik feedforward; trenowane RNN pokazują, że łańcuchowa struktura nienormalna
może powstać przez uczenie [Orh19].

Te same równania, inne generatory: lift-up/Orr w przepływach ścinających, reaktywność ekologiczna, wzmocnienie szumu biochemicznego,
kierunkowy zysk fotoniczny — skończony wzrost przy stabilnym widmie, a źródła od narzuconego ścinania lub skierowanej topologii po
projekt zysk–strata i asymetrię sieci reakcji [Sch07f, Asl18, Mcc13, Mak14]. Diagnostyki: wartości osobliwe propagatora, abscysa
numeryczna, pseudospektra / stała Kreissa, zysk wejście–wyjście (rezolwenta). **Zależą od normy, horyzontu i wyboru wejścia / odczytu**,
więc „wzmocnienie” nie jest jedną wymienną miarą [Tre05, Bag19] — uwaga do naszego artykułu: podawać normę, horyzont (jeden blok) i odczyt.

Sieci uczone: najbliżsi [Orh19] (RNN z emergentnymi łańcuchami nienormalnymi) oraz [Aub24, Fer26d] (sprzężenie wektorów osobliwych
i geometria jakobianu po głębokości w transformerach). Są **sąsiednie, nie bezpośrednie**: nikt nie pokazał zaworu i łańcucha Schura
w jakobianie MLP; nadto jakobian warstwy nie jest operatorem rekurencyjnym kory (analogia jest matematyczna, nie mechanistyczna).

## Najważniejsze prace (30 z 134; r = względna trafność, 1 = próg)

| klucz | praca | r | co dla nas |
|---|---|---|---|
| [Orh19] | Orhan & Pitkow, sekwencyjna nienormalna dynamika poprawia pamięć RNN (2019) | 1,98 | łańcuch nienormalny **powstaje w treningu**; najbliższy precedens |
| [Mur09] | Murphy & Miller, balanced amplification (2009) | 1,95 | definicja mechanizmu; Schur E/I |
| [Fer26d] | geometria widmowa strumienia resztkowego transformera (2026) | 1,93 | jakobian całego bloku, LLM; kontrast monotoniczność |
| [Hen12b] | Hennequin, Vogels, Gerstner, nienormalne wzmocnienie w losowych sieciach zbalansowanych (2012) | 1,87 | statystyka wzmocnienia przy losowych wagach (= nasze k0) |
| [Bon18] | kodowanie trajektoriami przejściowymi w RNN (2018) | 1,86 | funkcja wzmocnienia przejściowego |
| [Gol09] | Goldman, pamięć bez sprzężenia zwrotnego (2009) | 1,84 | ukryty łańcuch feedforward = pamięć |
| [Ker19] | nnRNN, parametryzacja Schura (2019) | 1,83 | kontrola nienormalności w uczeniu |
| [Pat23e] | **kora naczelnych wykonuje balanced amplification** (2023) | 1,81 | dowód eksperymentalny in vivo — cytować obok [Mur09] |
| [Hen14] | optymalna kontrola przebiegów przejściowych, ruch (2014) | 1,78 | nienormalność z optymalizacji |
| [Gan08b] | Ganguli, Huh, Sompolinsky, ślady pamięci (2008) | 1,78 | nienormalne (feedforward) macierze dają najdłuższą pamięć |
| [Chr22] | reżimy i mechanizmy wzmocnienia przejściowego (2022) | 1,75 | przegląd mechanizmów w sieciach |
| [Li26ad] | transient reserves w propagatorze uwagi (2026) | 1,74 | nienormalność **uwagi** LLM; nasz: MLP |
| [Asl18] | struktura i dynamika sieci nienormalnych (2018) | 1,67 | sieci skierowane |
| [Mid22] | topologiczne wzmocnienie kierunkowe (2022) | 1,65 | fotonika: zawór kierunkowy |
| [Aub24] | sprzężenie bloków transformera a generalizacja (2024) | 1,65 | kontrast: u nas brak sprzężenia u₁ᵇ→v₁ᵇ⁺¹, jest v₁ᵇ↔v₁ᵇ⁺¹ |
| [Sym17] | klasyfikacja mechanizmów wzmocnienia w analizie rezolwenty (2017) | 1,64 | słownik: wzmocnienie modalne vs nienormalne |
| [Tre93b] | stabilność hydrodynamiczna bez wartości własnych (1993) | 1,62 | teoria niemodalna |
| [Mak14] | anomalne wzmocnienie przejściowe fal w nienormalnych ośrodkach fotonicznych (2014) | 1,61 | optymalne wzbudzenie = prawy wektor osobliwy |
| [Sch07f] | Schmid, teoria stabilności niemodalnej (2007) | 1,58 | przegląd |
| [Bag19] | komunikacja w sieciach dynamicznych a nienormalność (2019) | 1,58 | nienormalność jako zasób komunikacyjny; zależność od normy |
| [Sad20f] | stabilizacja hamowaniem a obliczenia korowe (2020) | 1,55 | ISN ≠ balanced amplification (diagnostyki) |
| [Mcc13] | wzmocnienie bez niestabilności: płyny → chemia i biologia (2013) | 1,52 | przegląd międzydziedzinowy |
| [Jeo26] | geometryczna ciągłość w głębokich sieciach; połączenia resztkowe (2026) | 1,52 | nowe, do sprawdzenia |
| [Dua22] | strukturalne pochodzenie niestabilności w dużych układach (2022) | 1,51 | sieci skierowane |
| [Tre05] | Trefethen & Embree, Spectra and pseudospectra (2005) | 1,50 | podstawowa monografia |
| [Ram20e] | topologiczna teoria wejście–wyjście wzmocnienia kierunkowego (2020) | 1,44 | zawór jako niezmiennik topologiczny |
| [Dix16c] | wzmocnienie przejściowe ogranicza tłumienie szumu w sieciach biochemicznych (2016) | 1,43 | biochemia |
| [Sai26] | **wnioskowanie geometrii nienormalnego wzmocnienia z szeregów czasowych** (2026) | 1,43 | metoda pomiaru bez dostępu do macierzy — możliwe zastosowanie do aktywacji |
| [Rub15] | stabilized supralinear network (2015) | 1,42 | motyw korowy pokrewny |
| [Ioa06] | uogólniona teoria stabilności w prognozowaniu (2006) | 1,42 | pogoda: wektory osobliwe |

## Co z tego dla artykułu

1. Nazewnictwo: pisać „nienormalne wzmocnienie przejściowe o strukturze balanced amplification”, nie „balanced amplification” wprost,
   bo [Oze09, San19g, Sad20f] rozróżniają ISN, efekt paradoksalny i łańcuch Schura; my mamy łańcuch Schura i zawór, nie mamy E/I.
2. Nowe cytowania do dodania: [Pat23e] (in vivo), [Sym17] (klasyfikacja mechanizmów), [Ram20e, Mid22] (zawór kierunkowy jako topologia),
   [Sai26] (pomiar z szeregów czasowych), [Gol09, Gan08b] (łańcuch feedforward = pamięć), [Bag19] (zależność miar od normy).
3. Zastrzeżenie metodyczne z [Tre05, Bag19]: zawsze podawać normę (2), horyzont (jeden blok / złożenie), kierunek wejścia (losowy,
   najgorszy) i odczyt (głowa, następny blok) — nasz pomiar to robi, warto to podkreślić jako cechę.
4. Luka potwierdzona po raz drugi: brak demonstracji zaworu i łańcucha Schura w jakobianie MLP transformera; sąsiedzi [Orh19, Aub24, Fer26d, Li26ad].
