# Fala 5: odpowiedzi na cztery zarzuty recenzentów. Instrukcje na Edena (najpierw małe biegi)

Data: 2026-10-12. Źródło zarzutów: `outputs/porownanie_szklo_20261010/NOWOSC_PUBLIKACYJNA_2026-10-12.md`.

| pakiet | zarzut recenzenta | co mierzymy | czy trzeba uczyć |
|---|---|---|---|
| **A** | „temperatura = długość kroku” to artefakt stale aktywnego obcinania gradientu | klatka i τ bez obcinania, SGD i AdamW, paczki 16–128 | tak, małe biegi |
| **B** | jeden mały, niestandardowy model | ten sam pomiar na **DeiT-Tiny z timm** (Imagenette, potem CIFAR-10) | tak, małe biegi |
| **C** | δz = J δθ to tożsamość, brak teorii | przewidzenie funkcji korelacji ruchu C(r) z jądra i z kodowania pozycji, potem **test na nowych danych** (zmienione częstości PE) | najpierw bez uczenia, potem 2–3 małe biegi |
| **D** | brak zysku praktycznego poza zbiorem uczącym | czy wczesna ruchliwość niewidzianego zdjęcia przewiduje jego końcowy błąd lepiej niż pewność i rozrzut predykcji | najpierw bez uczenia, uczenie dopiero w D2 |

**Pliki do zmiany lub dopisania przed pierwszym biegiem.** Wszystkie są opisane w sekcjach „Kod do dopisania”; stan kodu z fali 4 zostaje jako kopia `*_fala4.py`.

| plik | zmiana | pakiet |
|---|---|---|
| `kod/trening_pokretla.py` | `--obciecie` z zapisem norm przed obcięciem; `--pe_baza`, `--pe_skala`, `--pe przetasowane` | A, C |
| `kod/trening_klasyfikacja.py` | `--obciecie`; `--model deit_ti` (klasa `DeitTi`); `--pe` | A, B |
| `kod/obserwator_rp.py` | klucz `siatka` w `SONDY.json` | B |
| `kod/teoria_korelacji.py` i `teoria.slurm` (nowe, slurm wzorowany na `jadro_pola.slurm`) | P0, P1, P2 wobec M | C |
| `eden/siatka_gesty.py`, `eden/fonony.py`, `eden/lindemann_wzgledny.py` | siatka z `SONDY.json` zamiast stałego 20 × 30 | B, C |
| `eden/ruchliwosc_wczesna.py` i `ruchliwosc_wczesna.slurm` (nowe, wzorowane na `pomiar_eden.py` i `pomiar_eden.slurm`) | cechy wczesne i końcowe IoU | D |

Sprawdzone na Edenie 2026-10-12:
- biegi `lr_5e-4_rp`, `bezpoz_lr_5e-4_rp` i `cnn_lr_5e-4_rp_zostaw` mają `pelny/stan_e02.pt`, `stan_e03.pt` i pełne `gesty/`;
- pomiar 4000 zdjęć leży w `~/smallvit_runs/2026-10-10_szklo_koloidalne/wyniki/pomiar_pelny60_3000tr_1000val`;
- wektorowany timm w `$E35/deps/timm` ma `deit.py`;
- w venv jest torch 2.9.1, ale nie ma torchvision.

Każdy pakiet ma trzy stopnie:
1. **próba** (`--proba`: 512 klatek, 2 epoki, 16 sond) sprawdza tylko, czy kod działa;
2. **mały bieg** (4 sondy, 2–9 epok) daje pierwsze liczby i sprawdza analizę od początku do końca;
3. **produkcja** (12 sond, 3 ziarna). Puszczamy ją dopiero, gdy mały bieg jest czysty.

---

## 0. Przygotowanie (raz)

### 0.1 Kwota: najpierw miejsce

Stan 2026-10-12: `lfs quota -p 1299` = **188,4 GB z 200 GB**, a każdy bieg wymaga co najmniej `--min_start_gb` wolnego miejsca. Największe pozycje:

| katalog | rozmiar | uwagi |
|---|---|---|
| `2026-10-10_szklo_pokretla/wyniki/klas_vit_od_zera_60ep_lr_5e-4/gesty` | 12 GB | sprawdzić, czy to nie bieg innej sesji |
| `…/wyniki/dlugi_lr_5e-4_rp/gesty` | 7,6 GB | zostawiony do testów jądra |
| `…/wyniki/{lr_5e-4_rp, bezpoz_lr_5e-4_rp, cnn_lr_5e-4_rp_zostaw, lr_2e-3_rp_zostaw, lr_5e-4_rp_z48_zostaw, lr_5e-4_rp_z49_zostaw}/gesty` | 6 × 4,1 GB | **trzy pierwsze są potrzebne w C1** (bez uczenia); trzy pozostałe po kontrolach siatki można skasować |
| `vit_k1000_replay900k_co5_20261009` | 24 GB | inny wątek projektu |
| `2026-10-10_szklo_koloidalne` | 26 GB | symulacje szkła BD |

Kasowanie należy do Ciebie. Wcześniej sprawdź, czy `ANALIZA*.json` danego biegu jest lokalnie w `outputs/porownanie_szklo_20261010/eden/wyniki/pokretla/`. Rozmiary sprawdzaj tylko na konkretnych katalogach, nigdy na `~`:

```bash
EXP=~/smallvit_runs/2026-10-10_szklo_pokretla
du -sh $EXP/wyniki/*/gesty | sort -h | tail -15
lfs quota -h -p 1299 /mnt/evafs | tail -1
```

Do małych biegów wystarczy około 10 GB wolnego miejsca:
- mały bieg z 4 sondami zajmuje około 1,4 GB surowych trajektorii (12 sond to 4,1 GB);
- `ZOSTAW=0` kasuje surowe trajektorie po analizie;
- `ZOSTAW=1` zostawia je dla pakietu C.

### 0.2 Zmienne, których używają wszystkie polecenia

```bash
B=/home2/faculty/bpiotrowski
EXP=$B/smallvit_runs/2026-10-10_szklo_pokretla
E35=$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok
M35=$E35/wejscia/MANIFEST_35000.jsonl; S100=$E35/wejscia/SONDY_100.jsonl
cd $EXP
```

Konwencje:
- **Nazwy biegów:** prefiks `f5A_`, `f5B_`, `f5C_`, `f5D_`.
- **Tory:** `--job-name=f5A --dependency=singleton` sprawia, że w torze działa najwyżej jeden bieg naraz, co pilnuje kwoty.
- **Zmienne `ANALIZY`, `ZOSTAW`, `WARSTWY`** przekazuj przez środowisko (`ANALIZY=300:2188 sbatch …`). Opcja `--export` jest na Edenie zabijana jako „CANCELLED by 0”.
- **Końce linii:** pliki `.py` i `.slurm` muszą mieć LF, bo przy CRLF sbatch odrzuca skrypt.

### 0.3 Wgranie kodu

Nowy i zmieniony kod (spis w sekcjach „Kod do dopisania”) trafia do `$EXP/kod/` (skrypty treningu i teorii) i do `$E35/kod/` (skrypty pomiaru D). Najpierw zrób kopię starych wersji na Edenie:

```bash
ssh eden-cluster 'cd ~/smallvit_runs/2026-10-10_szklo_pokretla/kod && for f in trening_pokretla.py trening_klasyfikacja.py obserwator_rp.py; do cp -n $f ${f%.py}_fala4.py; done'
tar cf - -C experiments/maly_vit_szklo_pokretla_20261010/kod trening_pokretla.py trening_klasyfikacja.py obserwator_rp.py teoria_korelacji.py \
  | ssh eden-cluster 'tar xf - -C ~/smallvit_runs/2026-10-10_szklo_pokretla/kod'
```

Wspólny `trening.py` biegu 35 000 (`$E35/kod/trening.py`) **zostaje bez zmian**. Nowe zachowania są dopisane łatkami w skryptach fali 5, więc główny bieg pozostaje odtwarzalny. `metadata.json` każdego biegu zapisuje sumy SHA-256 kodu.

### 0.4 Pobieranie wyników na dysk lokalny

```bash
mkdir -p outputs/porownanie_szklo_20261010/eden/wyniki/fala5
ssh eden-cluster 'cd ~/smallvit_runs/2026-10-10_szklo_pokretla/wyniki && tar cf - f5*/ANALIZA*.json f5*/dane/*.json f5*/dane/*.npz f5*/TEORIA_*.json f5*/SIATKA*.json f5*/FONONY*.json 2>/dev/null' \
  | tar xf - -C outputs/porownanie_szklo_20261010/eden/wyniki/fala5
```

---

## A. Obcinanie gradientu: czy „temperatura = długość kroku” przetrwa

**Zarzut.** We wszystkich dotychczasowych biegach `clip_grad_norm_(…, 1.0)` było aktywne praktycznie w każdym kroku (w `PACZKI.npz` norma gradientu = 1,0). Wtedy dla SGD długość kroku wynosi zawsze lr, więc paczka nie może zmienić klatki: wynik „paczka 16–128 daje klatkę 0,39/0,40/0,38/0,31” mógł wynikać z samego obcinania.

**Przewidywania:**
- **SGD bez obcinania:** klatka² ∝ lr / paczka, czyli klasyczna temperatura SGD (Mignacco i Urbani, Sclocchi i in.). Na wykresie log–log nachylenie bliskie 1.
- **AdamW bez obcinania:** klatka zależy od lr, a słabo od paczki, bo Adam normuje krok (|Δθ| ≈ lr).

Jeśli tak wyjdzie, teza zmienia się na: „temperaturą tokenów jest długość kroku w przestrzeni wag; przy SGD bez obcinania wynosi ona lr/B, przy Adamie i przy obcinaniu lr”. To jest mocniejsza teza od obecnej.

### Kod do dopisania (`trening_pokretla.py` i `trening_klasyfikacja.py`)

```python
a.add_argument('--obciecie', type=float, default=1.0, help='max norma gradientu; 0 = bez obcinania (fala 5)')
...
# po parse_args, przed utworzeniem Uczen: łatka na torch.nn.utils.clip_grad_norm_ (trening.py woła ją co krok)
_clip0 = torch.nn.utils.clip_grad_norm_
NORMY = dict(krok=[], norma=[])
def _clip(params, max_norm, *ar, **kw):
    n = _clip0(params, a.obciecie if a.obciecie > 0 else float('inf'), *ar, **kw)   # zwraca normę PRZED obcięciem
    NORMY['norma'].append(float(n)); NORMY['krok'].append(len(NORMY['norma']))
    return n
torch.nn.utils.clip_grad_norm_ = _clip
```

- `np.savez(out/'dane'/'NORMY_GRAD.npz', **NORMY)` przy końcu każdej epoki i na końcu biegu.
- `meta['pokretla']['obciecie'] = a.obciecie`.
- W `meta` dopisz `odsetek_obcietych` (udział kroków z normą > 1) i medianę normy.

### A0. Próba (około 5 minut)

```bash
ANALIZY=- ZOSTAW=1 sbatch --job-name=f5A --dependency=singleton pokretla_pelny.slurm f5A_proba 0.03 2 $M35 - 4 \
    --rzut rp --opt sgd --obciecie 0 --proba --min_start_gb 4
```

Sprawdź:
- w logu `pokrętła f5A_proba … extra: … --obciecie 0`;
- `dane/NORMY_GRAD.npz` ma około 64 wpisów;
- strata w `dane/LOG.json` jest skończona.

Potem `rm -r $EXP/wyniki/f5A_proba`.

### A1. Skan stabilności SGD bez obcinania (1 epoka, 1 sonda, bez analizy)

Bez obcinania SGD przy lr 0,1 prawdopodobnie się rozjedzie: przy obcinaniu efektywne lr wynosiło lr/|g|. Najpierw szukamy największego stabilnego lr, η*:

```bash
for lr in 0.003 0.01 0.03 0.1; do
  ANALIZY=- ZOSTAW=0 sbatch --job-name=f5A --dependency=singleton pokretla_pelny.slurm f5A_skan_sgd_${lr} $lr 1 $M35 $S100 1 \
      --rzut rp --opt sgd --obciecie 0 --min_start_gb 4
done
# odniesienie: ten sam przepis Z obcinaniem, tylko po to, by zapisać normy przed obcięciem (dowód, że obcinanie było stale aktywne)
ANALIZY=- sbatch --job-name=f5A --dependency=singleton pokretla_pelny.slurm f5A_ref_sgd_0.1_obc1 0.1 1 $M35 $S100 1 \
    --rzut rp --opt sgd --obciecie 1 --min_start_gb 4
ANALIZY=- sbatch --job-name=f5A --dependency=singleton pokretla_pelny.slurm f5A_ref_adamw_obc1 5e-4 1 $M35 $S100 1 \
    --rzut rp --obciecie 1 --min_start_gb 4
```

Czytaj z `NORMY_GRAD.npz` i `LOG.json`:
- medianę i 95. percentyl normy;
- odsetek kroków z normą > 1 w biegach `_ref_` (oczekiwane bliskie 100 %);
- czy strata maleje.

**η\*** to największe lr bez NaN, przy którym strata po 1094 krokach jest niższa niż w kroku 300.

### A2. Mały bieg właściwy: paczka przy stałym lr, bez obcinania

Liczby epok dobrane tak, by okno analizy 300–2188 się mieściło, jak w fali 4:

```bash
ETA=<η* z A1>
for bp in 16:2 32:3 64:5 128:9; do b=${bp%%:*}; e=${bp#*:}
  ANALIZY=300:2188 sbatch --job-name=f5A --dependency=singleton pokretla_pelny.slurm f5A_sgd_${ETA}_b${b}_bezobc $ETA $e $M35 $S100 4 \
      --rzut rp --opt sgd --paczka $b --obciecie 0 --min_start_gb 4
done
for bp in 16:2 32:3 128:9; do b=${bp%%:*}; e=${bp#*:}
  ANALIZY=300:2188 sbatch --job-name=f5A --dependency=singleton pokretla_pelny.slurm f5A_adamw_5e-4_b${b}_bezobc 5e-4 $e $M35 $S100 4 \
      --rzut rp --paczka $b --obciecie 0 --min_start_gb 4
done
for lr in 2.5e-4 1e-3; do     # wykładnik klatka ~ lr^x bez obcinania (z obcinaniem było 0,43)
  ANALIZY=300:2188 sbatch --job-name=f5A --dependency=singleton pokretla_pelny.slurm f5A_adamw_${lr}_bezobc $lr 3 $M35 $S100 4 \
      --rzut rp --obciecie 0 --min_start_gb 4
done
```

Czas: około 15–25 minut na bieg (trening, dane i analiza okna). Miejsce: około 1,4 GB chwilowo, kasowane po analizie.

**Co czytać** z `ANALIZA.json`:
- `S1.lindemann_shell` (klatka w odstępach sąsiadów);
- `S1.tau_1e`, `S1.beta`;
- `energia_768.E_300_400`;
- `rozklad_wariancji.udzial` przy τ = 1;
- z `NORMY_GRAD.npz`: średnia norma i długość kroku |Δθ|. Długość kroku można policzyć ze stanów `pelny/stan_eNN.pt` albo dopisać do łatki zapis `lr·|g|`.

**Decyzja (wykres log–log klatka² wobec lr/B):**

| wynik | interpretacja |
|---|---|
| SGD bez obcinania: nachylenie 0,8–1,2 i dobra zgodność z liniowością | klasyczna temperatura SGD; dotychczasowa niezależność od paczki była artefaktem obcinania. Napisać to wprost |
| AdamW: rozrzut klatki między paczkami < 10 % | dla Adama temperatura = lr niezależnie od obcinania; stara teza dla AdamW stoi |
| AdamW bez obcinania ≈ AdamW z obcinaniem (`lr_5e-4_rp`, klatka 0,50) | obcinanie nie zmieniało wyników głównego biegu |

**Produkcja:** te same biegi z 12 sondami i ziarnami 47, 48, 49 (`--ziarno`). Dopiero po decyzji.

---

## B. Standardowy ViT na publicznym zbiorze: DeiT-Tiny

**Dlaczego DeiT-Tiny z timm:** szerokość 192, 3 głowy, MLP 768, jak MałyViT, więc „cząstka” to ta sama warstwa GELU 768D ostatniego bloku. Różni się jednak w trzech kluczowych miejscach:

1. **Łatki 16 × 16 bez zakładki** zamiast łodygi konwolucyjnej, więc znika sprzężenie sąsiadów przez nakładanie pól widzenia łodygi (zmierzone wcześniej: 0,5 i 0,27).
2. **Uczone kodowanie pozycji** zamiast stałego sin-cos.
3. **Token CLS i 12 bloków** zamiast średniej i 6 bloków.

Jeśli udział pola wspólnych wag, siatka i zamrożona heterogeniczność przetrwają w DeiT, wynik jest ogólny. Jeśli siatka zniknie, wiemy, że tworzy ją łodyga, a nie kodowanie pozycji.

**Zbiór:**
- **Mały bieg:** Imagenette, który już leży na Edenie (`~/smallvit_data/imagenette/imagenette2-320`). `img_size=320` daje siatkę 20 × 20 = 400 tokenów, dokładnie jak istniejące biegi `klas_lr_*` MałegoViT, więc porównanie jest jeden do jednego.
- **Produkcja:** CIFAR-10 (163 MB, pobranie na węźle logowania: `wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`, czytanie przez `pickle`, bo w venv nie ma torchvision), skalowany do 224 (196 tokenów).
- **ImageNet-100** tylko wtedy, gdy kwota pozwoli (około 13 GB).

### Kod do dopisania (`trening_klasyfikacja.py`, opcja `--model deit_ti`)

```python
class DeitTi(nn.Module):
    """timm deit_tiny_patch16_224 od zera (wektorowany timm z $E35/deps), img_size 320 → siatka 20×20; cząstka = GELU fc1 ostatniego bloku,
    tylko tokeny łatek (bez CLS). --pe: uczone (domyślnie timm) | sincos (stałe jak MalyViT) | brak (zera) | przetasowane (sincos, wiersze permutowane rng 0)."""
    def __init__(self, n_klas=10, img=320, pe='uczone'):
        super().__init__()
        import timm
        self.vit = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=n_klas, img_size=img)
        self.np_ = self.vit.num_prefix_tokens; g = img // 16
        if pe != 'uczone':
            P = M.sincos_2d(g, g, 192, 'cpu')
            if pe == 'brak': P = torch.zeros_like(P)
            if pe == 'przetasowane': P = P[torch.from_numpy(np.random.default_rng(0).permutation(g * g))]
            with torch.no_grad():
                self.vit.pos_embed.zero_(); self.vit.pos_embed[0, self.np_:] = P
            self.vit.pos_embed.requires_grad_(False)
        self._latent = None; self._hak = None
    @property
    def bloki(self): return self.vit.blocks
    def zapisuj_latent(self, tak):
        if tak and self._hak is None:
            self._hak = self.vit.blocks[-1].mlp.act.register_forward_hook(lambda m, i, o: setattr(self, '_latent', o[:, self.np_:]))
        elif not tak and self._hak is not None:
            self._hak.remove(); self._hak = None; self._latent = None
    def forward(self, x):
        x = (x - M.MEAN.to(x.device, x.dtype)) / M.STD.to(x.device, x.dtype)
        return self.vit(x), self._latent
```

- W `UczenKlas.__init__`: `self.net = DeitTi(10, ROZMIAR, a.pe) if a.model == 'deit_ti' else Klasyfikator(...)`.
- Do argumentów dopisz `choices=['vit','cnn','deit_ti']` i `--pe`.
- `meta['pokretla']` uzupełnij o `pe` i `siatka=[20, 20]`.
- `ObserwatorRP._opis_sond` zapisuje `siatka` do `SONDY.json`.

**Analizy z sztywną siatką 20 × 30** dostają odczyt `H, W = SONDY.get('siatka', [20, 30])` (albo argument `--siatka 20x20`): `outputs/porownanie_szklo_20261010/eden/siatka_gesty.py`, `fonony.py` (H, W), `lindemann_wzgledny.py` (TOK). `pokretla_analiza.py` już czyta liczbę tokenów z `SONDY.json`.

### B0. Próba

```bash
ANALIZY=- ZOSTAW=1 sbatch --job-name=f5B --dependency=singleton klasyfikacja.slurm f5B_proba_deit 5e-4 2 4 --model deit_ti --proba
```

Sprawdź w logu:
- `model deit_ti (≈5,5 mln parametrów)`;
- obserwator zapisuje latent kształtu (S, 400, 768);
- acc > 0,1 po 2 epokach próby.

### B1. Małe biegi (8 epok = 2368 kroków, 4 sondy)

```bash
ANALIZY=300:2188 ZOSTAW=1 sbatch --job-name=f5B --dependency=singleton klasyfikacja.slurm f5B_deit_lr_5e-4 5e-4 8 4 --model deit_ti
ANALIZY=300:2188 ZOSTAW=1 sbatch --job-name=f5B --dependency=singleton klasyfikacja.slurm f5B_deit_pe_sincos 5e-4 8 4 --model deit_ti --pe sincos
ANALIZY=300:2188 ZOSTAW=1 sbatch --job-name=f5B --dependency=singleton klasyfikacja.slurm f5B_deit_pe_brak 5e-4 8 4 --model deit_ti --pe brak
```

Następnie siatka i fonony na zachowanych trajektoriach. Polecenia są jak w `eden/kontrole_siatki.slurm`, z `--siatka 20x20`, dla okna 300–2188. Po nich `ZOSTAW` można skasować ręcznie.

**Odniesienie:** MałyViT na tym samym zbiorze (`klas_lr_5e-4`, plik ANALIZA jest lokalnie).

**Co porównać, DeiT kontra MałyViT:**

| miara | skąd | pytanie |
|---|---|---|
| udział pola (wspólny + zdjęcie + sąsiedztwo) przy τ = 1 | `rozklad_wariancji.udzial` | czy dalej 85–90 % |
| C(r) przemieszczeń | `korelacja_przesuniec.C` | czy sąsiedzi nadal 0,6–0,85, skoro nie ma łodygi |
| klatka, τ, β, trwałość rankingu ruchliwości | `S1`, `trwalosc_ruchliwosci_CR` | czy heterogeniczność dalej jest zamrożona |
| R² numeru patcha i odsetek sąsiadów-patchy wśród 6 NN (losowo 8/399 = 2 %) | `siatka_gesty.py` | czy uczone PE też tworzy siatkę; `pe_brak` jako kontrola |
| wykładnik widma fononów | `fonony.py` | k^−2,7 jak w MałymViT czy płasko |

**Produkcja:** CIFAR-10 (DeiT-Ti 224) i Imagenette, 3 ziarna, 12 sond; opcjonalnie DeiT-Small (`deit_small_patch16_224`, szerokość 384, MLP 1536), żeby pokazać zależność od szerokości.

---

## C. Teoria: funkcja korelacji ruchu przewidziana z jądra i z kodowania pozycji

**Szkic teorii do sprawdzenia:**
- Krok tokenu to δz_i = J_i δθ. Dla izotropowego δθ korelacja ruchu dwóch tokenów to znormowane jądro tokenów, K_ij = tr(J_i J_jᵀ) / √(tr(J_i J_iᵀ) tr(J_j J_jᵀ)).
- Dla samych wag fc1 ostatniego bloku (∂z_ic/∂w_c = g_ic u_i, gdzie g = GELU′) jądro rozkłada się dokładnie na iloczyn: **K^fc1_ij = (u_i · u_j) · (Σ_c g_ic g_jc)**, czyli podobieństwo wejść razy wspólną bramkę.
- Wejście u_i = LN(treść_i + PE_i + …), więc podobieństwo wejść zawiera jądro kodowania pozycji. Dla sin-cos jest ono **analityczne i zależy tylko od przesunięcia na siatce**: κ(Δx, Δy) = Σ_f [cos(ω_f Δx) + cos(ω_f Δy)], gdzie ω_f = baza^(−f/48).

**Hierarchia przewidywań:**

| poziom | co | skąd | ile kosztuje |
|---|---|---|---|
| P0 | κ_PE(Δr) | wzór analityczny | nic |
| P1 | (u_i·u_j)(g_i·g_j) | jeden forward w danym stanie | sekundy |
| P2 | pełne jądro tokenów | JVP z losowymi kierunkami δθ (izotropowymi i z prekondycjonerem AdamW ze stanu) | minuty na GPU |
| M | zmierzona korelacja przyrostów | `gesty/` w oknie przed stanem | — |

Teza do obrony: **M ≈ P2** (już wiemy: losowe kierunki dają C1 0,82 wobec zmierzonego 0,85). Pytanie brzmi, ile z tego daje P1, a ile P0, czyli kodowanie pozycji. Wartość dla recenzenta daje **przewidzenie przed uczeniem**: zmieniamy częstości PE i z P0 i P1 w kroku 0 przewidujemy nowe C(r), długość korelacji ξ i nachylenie widma fononów.

### Kod do dopisania

1. **`trening_pokretla.py`:** opcje `--pe_baza` (domyślnie 10000), `--pe_skala` (domyślnie 1,0) i `--pe przetasowane`. Łatka jak przy `--bez_pozycji`, czyli podmiana `M_.sincos_2d` na wersję z inną bazą i skalą albo z permutacją wierszy (rng 0). Wartości trafiają do `meta['pokretla']`.
2. **Nowy `kod/teoria_korelacji.py <bieg> <stan.pt> <out.json> [--n_kier 64] [--okno od do]`:**
   - buduje sieć według `metadata.json` biegu (vit lub cnn, `bez_pozycji`, `pe_*`) i wczytuje stan **w trybie eval** (pułapka z fali 4: bufory BatchNorm łodygi zmieniają się w trybie train, więc ładuj `load_state_dict` całego stanu);
   - wczytuje zdjęcia sond z `SONDY.json`, tak jak robi to `jadro_pola.py`;
   - liczy P0, P1 (haki na `bloki[-1].mlp.fc1`: wejście u, preaktywacja, z której g = GELU′), P2 (`torch.func.jvp` na `functional_call`, przesunięcia rzutowane tą samą macierzą R 128D co obserwator, ziarno 0), a M bierze z `gesty/` w oknie;
   - wynik:
     - krzywe C(r) w pierścieniach odległości na siatce, osobno poziomo i pionowo;
     - Spearman par (P0, P1, P2 wobec M);
     - R² krzywych;
     - długość ξ (gdzie C spada do 1/e);
     - dopasowanie M ≈ a·P1 + b·P0 + c;
   - zapis `TEORIA_<stan>.json`.

### C1. Bez uczenia: istniejące biegi z zachowanymi trajektoriami

Biegi `lr_5e-4_rp` (`pelny/stan_e02.pt`, krok 2188), `bezpoz_lr_5e-4_rp` i `cnn_lr_5e-4_rp_zostaw` mają zachowane `gesty/`. Najpierw próba z `--n_kier 8`, potem pełne liczenie:

```bash
# sbatch teoria.slurm <bieg> <stan> <n_kier> <od> <do>   (slurm jak jadro_pola.slurm, wywołuje teoria_korelacji.py)
sbatch --job-name=f5C --dependency=singleton teoria.slurm lr_5e-4_rp stan_e02.pt 8 1988 2188     # próba
sbatch --job-name=f5C --dependency=singleton teoria.slurm lr_5e-4_rp stan_e02.pt 64 1988 2188
sbatch --job-name=f5C --dependency=singleton teoria.slurm bezpoz_lr_5e-4_rp stan_e02.pt 64 1988 2188
sbatch --job-name=f5C --dependency=singleton teoria.slurm cnn_lr_5e-4_rp_zostaw stan_e02.pt 64 1988 2188
```

Oczekiwane:
- Spearman(P2, M) ≥ 0,8;
- P1 wyjaśnia co najmniej połowę kształtu C(r);
- P0 tłumaczy część długozasięgową i **znika w `bezpoz`** (kontrola wewnętrzna).

### C2. Test na nowych danych: zmienione częstości kodowania pozycji

Najpierw przewidywanie, potem uczenie. Kolejność jest ważna, bo to rejestracja przewidywania.

1. Krótki bieg zapisuje stan kroku 0 (`wagi/k00000.pt` zapisuje się zawsze). Dla każdego wariantu liczymy P0 i P1 w kroku 0 i zapisujemy `PRZEWIDYWANIE_<wariant>.json` (ξ, nachylenie widma, C(1), C(2)) **przed** obejrzeniem wyników. Sumę SHA tego pliku zapisz w notatce.
2. Małe biegi, 3 epoki, 4 sondy, z zachowanymi trajektoriami do teorii i fononów:

```bash
for w in "baza100:--pe_baza 100" "baza1e6:--pe_baza 1e6" "skala2:--pe_skala 2" "przetas:--pe przetasowane"; do n=${w%%:*}; arg=${w#*:}
  ANALIZY=300:2188 ZOSTAW=1 sbatch --job-name=f5C --dependency=singleton pokretla_pelny.slurm f5C_pe_${n} 5e-4 3 $M35 $S100 4 \
      --rzut rp $arg --min_start_gb 4
done
```

3. Po biegach liczymy C(r) zmierzone (`TEORIA_stan_e02.json` z pola M), fonony i siatkę i porównujemy z zapisanym przewidywaniem.

**Kryteria sukcesu:**
- kolejność ξ (baza100 < baza 10⁴ < baza1e6) zgodna z przewidywaniem;
- stosunki ξ w granicach ±25 %;
- dla PE przetasowanego siatka w przestrzeni obrazu znika (sąsiedzi-patche wśród NN spadają do poziomu losowego), ale **pojawia się na siatce permutowanej**. To najmocniejszy dowód, że kryształ jest zadany przez kodowanie pozycji.

**Opcjonalnie C3 (rozmiar klatki z jądra):** energia kroku tokenu E = ⟨|J_i Δθ|²⟩ dla kroku AdamW o długości ∝ lr, a klatka² ≈ E · τ/2 (OU). Przewidzieć klatkę przy lr 1e-3 i 2e-3 z jądra i τ zmierzonych przy 2,5e-4 i 5e-4, potem porównać z istniejącymi biegami `lr_1e-3_rp` i `lr_2e-3_rp_z48/49` (bez nowego uczenia).

---

## D. Zysk praktyczny poza zbiorem uczącym

Dotychczas Spearman ruchliwość–błąd wynosił −0,37 na TRAIN i −0,08 na VALID. Na VALID to za mało. Zamiast korelacji ruchliwości z błędem w tym samym momencie sprawdzamy **przewidywanie z wyprzedzeniem bez etykiet**: czy z ruchu tokenów zdjęcia **niewidzianego w uczeniu**, zmierzonego wcześnie (kroki 300–2200), da się przewidzieć jego **końcowe** IoU po 60 epokach. Praktycznie to wybór klatek do anotacji: aktywne uczenie bez etykiet.

**Uczciwe punkty odniesienia** (wszystkie bez etykiet, z tych samych checkpointów):
1. średnia pewność;
2. rozrzut pewności;
3. **niestabilność predykcji**: 1 − IoU kolejnych przewidzianych ramek oraz rozrzut środka ramki względem jej rozmiaru;
4. |z − z0| (`delta_l2` z pomiaru).

### D1. Bez uczenia (wagi 806 checkpointów są na Edenie)

**Kod do dopisania:** `outputs/porownanie_szklo_20261010/eden/ruchliwosc_wczesna.py`, wzorowany na `pomiar_eden.py`. Uruchamiany w `$E35/kod` ze zmiennymi `MALY_VIT_*`.
- Wejścia:
  - `--wagi $E35/wyniki/pelny/wagi`;
  - `--pomiar ~/smallvit_runs/2026-10-10_szklo_koloidalne/wyniki/pomiar_pelny60_3000tr_1000val`: lista tych samych 4000 zdjęć z jego `manifest.json` i końcowe IoU z `metrics.npy` (kształt 4000 × 806 × 5, kolumny `[confidence, iou, delta_l2, training_step, both90]`);
  - `--kroki 300:2200:100`;
  - `--n_val`, `--n_train`.
- Na każdym checkpoincie i zdjęciu liczy:
  - latent 600 × 768 względem średniej zdjęcia;
  - ruchliwość: średnie po tokenach |z_i(t) − z_i(t−1)|;
  - amplitudę wokół średniej okna, jak `ruchliwosc_zdjec.py`;
  - pewność i ramkę (`dekoduj`).
- Cechy w oknie:
  - ruchliwość: średnia i wartość z końca okna;
  - amplituda;
  - pewność: średnia i rozrzut;
  - niestabilność ramki;
  - `delta_l2`.
- Cele:
  - IoU w ostatnim checkpoincie (krok 65 640), osobno w końcu 21. epoki;
  - porażka = IoU < 0,5.
- Statystyki osobno dla VALID i TRAIN:
  - Spearman każdej cechy z końcowym IoU;
  - AUROC porażki;
  - 1000 prób bootstrap z 95-procentowym przedziałem;
  - częściowy Spearman ruchliwości po kontroli najlepszej cechy odniesienia;
  - **wychwyt budżetu**: jaki odsetek końcowych porażek leży w 20 % zdjęć wskazanych przez daną cechę.
- Wynik: `RUCHLIWOSC_WCZESNA.json` i `ruchliwosc_wczesna.npz` (cechy na zdjęcie).

```bash
# próba: 200 zdjęć VALID, 7 checkpointów (~5 min GPU)
sbatch --job-name=f5D ruchliwosc_wczesna.slurm --n_val 200 --n_train 0 --kroki 300:2200:300
# mały pełny: 1000 VALID + 1000 TRAIN, 20 checkpointów (~15 min GPU)
sbatch --job-name=f5D --dependency=singleton ruchliwosc_wczesna.slurm --n_val 1000 --n_train 1000 --kroki 300:2200:100
```

**Decyzja:**

| wynik | interpretacja |
|---|---|
| na VALID ruchliwość ma Spearman z końcowym IoU wyższy od najlepszego odniesienia o co najmniej 0,05, z przedziałem bootstrap bez zera, i wyższy wychwyt budżetu | zysk praktyczny, wchodzi do artykułu |
| ruchliwość ≈ niestabilność predykcji | ruch tokenów to tylko „wnętrze” niestabilności wyjścia; uczciwy wynik negatywny, do dyskusji |
| ruchliwość < odniesień | nie budować na tym tezy praktycznej |

### D2. Z uczeniem (dopiero gdy D1 da sygnał): wybór danych do uczenia

1. `ruchliwosc_wczesna.py --n_train all` liczy ruchliwość dla wszystkich 35 000 klatek TRAIN (35 000 × 7 checkpointów, kilka minut). Dodatkowo ten sam skrypt liczy EL2N (strata zdjęcia w kroku około 1000, wymaga etykiet).
2. Podzbiory po 10 000 klatek: losowy, EL2N-wysokie, ruchliwość-wysoka, ruchliwość-niska. Pliki `wejscia/MANIFEST_10000_<kryterium>.jsonl` mają ten sam format co `MANIFEST_1500_z35k`.
3. Małe biegi: 3 epoki, 1 ziarno, 1 sonda, bez analizy fizyki. Liczy się tylko mAP@0,5 z `dane/OCENY.json`.

```bash
for k in losowy el2n ruch_wys ruch_nis; do
  ANALIZY=- sbatch --job-name=f5D --dependency=singleton pokretla_pelny.slurm f5D_wybor_${k} 5e-4 3 $EXP/wejscia/MANIFEST_10000_${k}.jsonl - 1 \
      --rzut rp --min_start_gb 4
done
```

**Produkcja D2:** 3 ziarna, 21 epok, budżety 30 i 50 %; punkt odniesienia z literatury to EL2N i zdarzenia zapominania. To łączy się bezpośrednio z wątkiem selekcji danych (`AdvancedDatasetSelection`).

---

## Kolejność i budżet małych biegów

| kolejność | pakiet | zadania | GPU | dysk chwilowo |
|---|---|---|---|---|
| 1 | A0, B0 | 2 próby | około 10 min | < 1 GB |
| 2 | C1 | 3 liczenia teorii na istniejących biegach | około 30 min | 0 (czyta istniejące) |
| 3 | D1 | próba i mały pełny | około 20 min | < 1 GB |
| 4 | A1 | 4 skany i 2 odniesienia (1 epoka) | około 1 h | po około 0,4 GB |
| 5 | A2 | 9 biegów | około 3 h | po około 1,4 GB, kasowane |
| 6 | B1 | 3 biegi DeiT | około 1,5 h | 3 × około 1,2 GB (`ZOSTAW=1`) |
| 7 | C2 | przewidywanie, potem 4 biegi | około 1,5 h | 4 × około 1,4 GB (`ZOSTAW=1`) |
| 8 | D2 | 4 biegi (tylko po sygnale z D1) | około 1,5 h | małe |

Tory `f5A`–`f5D` mogą iść równolegle; w obrębie toru najwyżej jeden bieg naraz. Najszybciej pierwsze odpowiedzi dadzą **C1 i D1**, bo nie wymagają uczenia.

## Po każdym stopniu

1. Pobierz wyniki (sekcja 0.4) do `outputs/porownanie_szklo_20261010/eden/wyniki/fala5/`.
2. Dopisz wiersz do tabeli wyników w sekcji „Fala 5” w `outputs/porownanie_szklo_20261010/README.md`.
3. Skasuj zachowane trajektorie (`ZOSTAW=1`), gdy analizy siatki, fononów i teorii są już lokalnie.
