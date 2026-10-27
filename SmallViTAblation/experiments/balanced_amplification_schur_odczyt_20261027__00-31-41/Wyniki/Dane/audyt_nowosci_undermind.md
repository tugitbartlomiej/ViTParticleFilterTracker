# Audyt nowości (Undermind, 2026-10-27 00:44): nienormalność i balanced amplification w podwarstwie MLP transformera

Wyszukiwanie `/Novelty audit non normality and balanced amplification structure of the MLP sublayer Jacobian in trained transformers`
(21 prac) w przestrzeni „ViT: detekcja końcówki narzędzia w chirurgii zaćmy” (40294fa6-…), uzupełnione o streszczenia najbliższych prac
z wyszukiwania 26.10 (`wzmocnienie_przejsciowe_pseudospektrum/Wyniki/Dane/literatura_undermind.md`). Podsumowanie Undermind oparte na
streszczeniach, nie pełnych tekstach.

## Werdykt Undermind

Żadna z odnalezionych prac nie raportuje pomiaru jakobianu podwarstwy MLP per token (odchylenie Henriciego, stała Kreissa, pseudospektrum)
w funkcji checkpointu z kontrolami (bez weight decay, losowe etykiety), ani struktury Schura / balanced amplification, wyrównania z gradientem
odczytu, nieliniowej weryfikacji wzmocnienia na IoU / marginesie, rozkładu na wagi vs bramkowanie, ani ablacji frozen_mlp / parallel w tym
kontekście. Najbliższy precedens w transformerach: pełna geometria widmowa jakobianu całego bloku strumienia resztkowego w LLM [Fer26d].

## Najbliższe prace i czym się różnimy

| praca | co zmierzyli | różnica |
|---|---|---|
| [Li26ad] Li 2026 (arXiv 2607.09279) | nienormalność propagatora **uwagi** w GPT-2, Pythia, Llama-3: Kreiss, pseudospektra, predykcje pre-rejestrowane, cenzus checkpointów, interwencja przyczynowa | uwaga, nie MLP; LLM; bez kontroli losowych etykiet / wd i ablacji |
| [Fer26d] Fernando & Guitchounts 2026 | pełne widmo jakobianu **całego bloku** w LLM: monotoniczny gradient od nienormalnych wczesnych do niemal symetrycznych późnych; wyuczone; „rozpuszcza się po usunięciu nienormalności strukturalnej” | blok, nie MLP; monotoniczność po głębokości (u nas: szczyt w blokach 3–5, w CIFAR ostatni blok najwyższy); bez kontroli |
| [Aub24] Aubry i in., ICLR 2024 | sprzężenie górnych wektorów osobliwych jakobianów bloków po głębokości i tokenach koreluje z generalizacją (LLM i ViT) | my: w detekcji najgorsze kierunki kolejnych bloków MLP **nie** są zgodne (W5 FAIL), w CIFAR słabo — kontrast do zacytowania |
| [Jha26c] NerVE 2026 | miary widmowe (entropia, PR) kowariancji aktywacji FFN w LLM; nieliniowość „reinjektuje” wariancję między modami | widmo kowariancji aktywacji, nie jakobianu; bez nienormalności |
| [Whi26] Whipp 2026 | liniowa odtwarzalność bloku FFN (R²_lin) jest wyuczona, niemonotoniczna po głębokości | miara liniowości, nie nienormalności; zgodna z naszym „własność wyuczona, niemonotoniczna” |
| [Li22ap] Li i in., ICLR 2023 | rzadkość aktywacji MLP powstaje także przy losowych etykietach i losowych wejściach | to samo kontrolne rozumowanie (losowe etykiety) zastosowane do innej własności; cytować jako precedens metodologiczny |
| [Mud26] Mudarisov i in. 2026 | FFN jako pole sterujące na sferze; składowa styczna konieczna; defekt komutatora decyduje o równoległości bloku (LLM) | zgodne z naszymi P4–P6, C9b; bez jakobianów / nienormalności |
| [Wan25ae] Wang i in. 2025 | tokeny o dużej normie wyzwalane przez prawy wiodący wektor osobliwy przybliżenia liniowego modułu (ViT, LLM) | ten sam obiekt (v₁) w innym celu; możliwy związek: wzmacniany kierunek a „singular defects” |
| [Son23h] Sonkar & Baraniuk 2023 | PAF (równoległy FFN) w RoBERTa/BERT: FFN utrzymuje izotropię tokenów | precedens dla naszej ablacji parallel (A2), w LLM |
| [Per26d] Pervez & Locatello 2026 | nienormalne, niekomutujące jakobiany → wzmocnienie przejściowe błędu rozwinięć; regularyzacja normalności | symulatory fizyczne; cel inżynierski; pokazuje, że „normalizowanie” jakobianów ma sens praktyczny |
| [Gho26b] Ghosh 2026 | nienormalność operatora optymalizatora (Adam) jako precursor niestabilności | inny operator |
| [Mur09, Hen12b, Hen14, Gan08b, Orh19, Ker19, Bon18, Chr22] | balanced amplification, ukryty feedforward, nienormalne RNN | sieci rekurencyjne i kora; nie MLP transformera |

## Co jest nowe (do sformułowania w artykule)

1. Obiekt: jakobian **podwarstwy MLP** per token (nie uwaga, nie cały blok), w ViT detektorze i klasyfikatorach (nie LLM).
2. Dynamika treningu z kontrolami przyczynowymi: 24 seedy, bez wd, losowe etykiety (kształt zostaje, skala zanika), frozen_mlp / frozen_attn /
   parallel / frozen_both.
3. Struktura balanced amplification w MLP (forma Schura, zawór, wyrównanie z odczytem) — pomiar w toku (ten eksperyment).
4. Nieliniowa weryfikacja wzmocnienia (σ_max co do 1 %) i jego skutek dla IoU / marginesu.
5. Pochodzenie: wagi W₂W₁ vs bramkowanie, żywe reflektory; niemonotoniczność po głębokości (kontrast z [Fer26d]); brak sprzężenia kierunków
   między blokami (kontrast z [Aub24]).

## Ryzyko czasowe

[Li26ad] (lipiec 2026) i [Fer26d] (maj 2026) pokazują, że język pseudospektrów wchodzi do analizy transformerów; pomiar MLP jest naturalnym
następnym krokiem dla tych grup. Argument za szybkim preprintem (arXiv) po zakończeniu bieżących pomiarów, przed pełnym tekstem do czasopisma.
