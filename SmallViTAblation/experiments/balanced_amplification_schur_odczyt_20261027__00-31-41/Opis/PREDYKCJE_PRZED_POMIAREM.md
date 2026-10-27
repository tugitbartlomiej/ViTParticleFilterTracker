# Predykcje zapisane przed pomiarem (2026-10-27, 00:40, Europe/Warsaw)

Zapisane przed uruchomieniem jakiegokolwiek zadania tego eksperymentu na Edenie. Progi nie będą zmieniane po pomiarze; uwagi po fakcie
będą oznaczone. Pytanie: **czy nienormalność pola MLP ma strukturę balanced amplification** (Murphy & Miller 2009; Hennequin 2012, 2014)
i nienormalnego wzmacniacza fotonicznego (Makris 2014; Midya 2022): ukryty łańcuch feedforward o niskiej randze, rozdzielny kierunek
wejścia i wyjścia wzmacnianego modu (zawór), wzmocnienie wyrównane z odczytem; oraz **gdzie siedzi nienormalność** (wagi W₂W₁ czy
bramkowanie GELU) i jak zależy od liczby żywych reflektorów. Miary w `Kod_eksperymentu/balanced_eden.py` (docstring); ocena w
`Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py`.

Biegi jak w rodzicu wzmocnienie_przejsciowe: detekcja s3047 / s4047 / s5047 (k0, 3000, 65 640), MałyViT CIFAR s47 i DeiT CIFAR s47
(k0, 3000, 93 780); 5 biegów, test znaku 5/5. Blok środkowy b* = NB // 2 + 1. Mediany po tokenach (narzędzie i tło razem) i klatkach.
Odniesienia losowe: dla kierunku losowego w 192 wymiarach cos² ≈ 1/192 = 0,0052, |cos| ≈ 0,058.

| nr | predykcja | warunek liczbowy (blok b*, mediana tokenów) | próg |
|---|---|---|---|
| BA1 | trening koncentruje część nilpotentną w kilku modach (ukryty łańcuch feedforward o niskiej randze) | c5(N): koniec > k0 **i** PR_N: koniec < k0 | 5/5 |
| BA2a | wzmacniany mod ma rozdzielne wejście i wyjście (feedforward, nie normalny) | \|⟨v₁, u₁⟩\| dla pary osobliwej J na końcu < 0,3 | 5/5 |
| BA2b | zawór: to, co pole napisało, nie jest przez nie znów czytane | ‖J u₁‖ / σ₁(J) na końcu < 0,6 | 5/5 |
| BA3a | wzmacniany kierunek wyjściowy jest wyrównany z odczytem bardziej niż wejściowy | mediana cos²(u₁, r) > mediana cos²(v₁, r) na końcu | 5/5 |
| BA3b | wyrównanie z odczytem jest nieprzypadkowe i powstaje w treningu | mediana cos²(u₁, r) na końcu > 3/192 **i** > wartość przy k0 | 5/5 |
| BA4a | nienormalność siedzi w wagach, nie w bramkowaniu | ‖N‖/‖J‖ macierzy W₂W₁ na końcu > 0,85 | 5/5 |
| BA4b | bramkowanie nie zmienia nienormalności | \|‖N‖/‖J‖(W₂·diag(ḡ)·W₁) − mediana ‖N‖/‖J‖(J)\| < 0,05 na końcu | 5/5 |
| BA5a | wzmocnienie rośnie z liczbą żywych reflektorów (GELU′ ∈ (0,1; 0,9)) | korelacja Pearsona (żywe, G₁) po tokenach bloku b* na końcu > 0,3 | 5/5 |
| BA5b | tokeny narzędzia mają mniej żywych reflektorów niż tło (detekcja) | mediana „żywe” narzędzia < tła w bloku 4 na końcu | 3/3 |
| BA6 | reflektory są niewzajemne (czytają z innego kierunku, niż piszą) także po treningu | średnia \|cos(w₁ᵢ, w₂ᵢ)\| ważona ḡᵢ na końcu < 0,2 **i** nie większa niż 2 × wartość przy k0 | 5/5 |

Poprawka definicji przed pomiarem (00:50, po teście lokalnym z losowymi wagami, przed jakimkolwiek zadaniem na Edenie): para osobliwa
(u₁, v₁, σ₁) liczona na polu J, nie na A = I + J, bo przy małym J wektory osobliwe A są zdominowane przez I i |⟨v₁, u₁⟩| ≈ 0,96 niezależnie
od struktury (test lokalny). Odniesienie losowe (Ginibre, k0) będzie podane obok każdej wartości.

Uwaga do BA3 dla DeiT: odczyt r jest gradientem marginesu CLS po stanie tokenu obrazu (przez uwagę), więc kierunek odczytu jest pośredni;
BA3 dla DeiT zapisuję jako test, nie oczekiwanie pewne.
Uwaga do BA6: wcześniejsze eksperymenty tego projektu znalazły pary neuronów przeciwstawnych (w₂ antypodalne); gdyby trening wyrównywał
w₂ᵢ z ±w₁ᵢ, J stawałoby się bardziej normalne, a pomiar rodzica mówi odwrotnie. BA6 sprawdza więc spójność obu obserwacji.

## Co uznam za potwierdzenie

- Struktura balanced amplification: BA1, BA2a, BA2b, BA3a, BA3b. Wtedy zdanie do artykułu: gradient uczy MLP transformera ukrytego łańcucha
  feedforward o niskiej randze (czytaj tu → pisz tam), ze stabilnym widmem i wzmocnieniem wyrównanym z odczytem, jak w korze.
- Pochodzenie: BA4a, BA4b (wagi), BA5a, BA5b (bramkowanie steruje wielkością, nie kształtem), BA6 (niewzajemność reflektorów).
- Jeżeli BA3 nie zajdzie: łańcuch jest, ale nie jest wyrównany z zadaniem — struktura treningu, nie funkcji (zgodne z A6 i C1z rodziców).

## Uzupełnienie BA3c / BA3d (zapisane 2026-10-27 01:21, po ocenie BA1–BA6, przed ponownym uruchomieniem)

Wynik BA3 był odwrotny do predykcji: w detekcji z gradientem logitu ciepła wyrównane jest wejście v₁ (cos² 0,06–0,11), nie wyjście u₁
(0,003–0,008). Pytanie uzupełniające: **dokąd pisze wzmacniany mod?** Dwie możliwości testowane przed pomiarem; progi dla biegów detekcji
(3) i CIFAR (2) osobno, bo podprzestrzenie mają inny wymiar (detekcja: 5 wyjść głowy, w tym 4 ramki; CIFAR: 10 logitów).
Odniesienie losowe: energia losowego kierunku w podprzestrzeni k-wymiarowej = k/192 (5 wyjść: 0,026; 4 ramki: 0,021; 10 logitów: 0,052).

| nr | predykcja | warunek liczbowy (blok b*, mediana tokenów, koniec) | próg |
|---|---|---|---|
| BA3c | wyjście wzmacnianego modu leży w podprzestrzeni odczytu ramki (czyta „czy narzędzie”, pisze „jaka ramka”) | detekcja: E(u₁ \| ramka) > 3 × 4/192 **i** E(u₁ \| ramka) > E(u₁ \| ramka) przy k0 | 3/3 |
| BA3c′ | w CIFAR wyjście leży w podprzestrzeni wszystkich logitów | E(u₁ \| 10 logitów) > 3 × 10/192 | 2/2 (test) |
| BA3d | następny blok czyta to, co ten napisał (łańcuch między blokami) | cos²(u₁ᵇ, v₁ᵇ⁺¹) dla tego samego tokenu, mediana po blokach b < NB: koniec > 3/192 **i** > k0 | 5/5 |

Jeżeli BA3c nie zajdzie, a BA3d zajdzie: MLP pisze do współrzędnych czytanych przez kolejne bloki, nie przez głowę; „odczyt” w sensie
Murphy–Miller trzeba definiować przez następną warstwę. Jeżeli oba nie zajdą: wzmacniany mod rozprasza współrzędną odczytu poza to, co
czyta sieć dalej — odwrócenie jest realne. Wartości BA1–BA6 z pierwszego biegu zostają (ten sam kod, dodane kolumny).
