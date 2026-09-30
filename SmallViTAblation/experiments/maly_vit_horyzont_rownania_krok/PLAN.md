# Horyzont prognozy latentu tożsamości równaniem ruchu

Pytanie użytkownika (2026-09-30): „sprawdź, jak daleko kroków uczenia jest w stanie równanie przewidzieć do przodu
(prognoza latentu tożsamości) z dokładnością do 90 procent”.

## Co mierzę

Równanie jednego kroku z `../maly_vit_tozsamosc_jeden_krok` (commit a5813796): Δz(t+1) = S_pęd(t) + S_paczka(t),
95 procent umiejętności na nagraniach. Tutaj: prognoza przyrostu siły tożsamości 180 sond z narzędziem na H kroków
naprzód, H od 1 do 500, i największe H, do którego umiejętność nie spada poniżej 90 procent.

Umiejętność = 100 · (1 − SSE / SS) przyrostu między t0 a t0 + H wobec „braku zmiany” (ta sama miara co w a5813796).

Warianty, różniące się tym, co wiadomo w chwili t0 (kod `horyzont.py`):

| wariant | co zna | rola |
|---|---|---|
| A | równanie liczone co krok na prawdziwych wagach i paczkach, zsumowane | górna granica, nie prognoza |
| B | klatki przyszłych kroków (w pętli wybiera je pętla), wagi z chwili t0; krok AdamW symulowany w przestrzeni wag | prognoza przy znanych klatkach |
| B2 | jak B, ale suma pchnięć liczonych osobno przy wagach z t0 (dosłowne przedłużenie S_pęd + S_paczka) | prognoza przy znanych klatkach |
| C1 | sam pęd optymalizatora z t0, wygasający | prognoza bez przyszłych klatek |
| C2 | pęd z t0 i średni gradient całego zbioru w każdym kroku | prognoza bez przyszłych klatek |

Reguła o przyszłych klatkach (memory użytkownika): prognoza właściwa to C1 i C2. B i B2 używają klatek przyszłych
kroków, co jest dozwolone tylko wtedy, gdy klatki są decyzją pętli (wybór zestawu), a nie obserwacją.

## Aparat

Zwykłe uczenie małego ViT na bazie 1500 klatek (jak rozgrzewka pętli 4.2), bez dokładania klatek, lokalnie RTX 3070.
Rozgrzewka 1000 kroków, wektor tożsamości z kroków 960-1000, potem 4 rozłączne okna po 500 kroków (t0 = 1000, 1500,
2000, 2500). Bufory BatchNorm prognoz z chwili t0. Latent fp32.

Uruchomienie: `uruchom_lokalnie.cmd horyzont.py --ziarno 47`, wyniki `wyniki/HORYZONT_s47.json`, `wyniki/horyzont_s47.npz`.
