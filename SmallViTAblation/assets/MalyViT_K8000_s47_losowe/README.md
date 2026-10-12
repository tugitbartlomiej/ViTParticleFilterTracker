# Mały ViT po 8000 krokach — seed 47, dobór losowy

Plik: `MALY_VIT_K8000_s47_losowe.pth`.

Checkpoint detektora narzędzia MalyViT z eksperymentu `2026-10-01_petla43_sd_prowadzenie`, bieg `petla43b_s47`, wariant `losowe`. To model po treningu na wielu obrazach, nie dopasowanie jednego obrazu do sztucznego latentu.

## Trening i wynik

- Seed: **47**.
- Wariant doboru danych: **losowe**.
- Zakończony krok: **8000**.
- Zakończony obrót pętli eksperymentalnej: **16**. Obrót nie jest automatycznie pełną epoką całego zbioru.
- Zapisany historyczny wynik **AP50 / mAP50: 0.6112320512935991 (61,12%)**, przy kroku 8000.
- Wynik pochodzi z `OCENY.json` tego biegu. Przy przenoszeniu modelu nie wykonano ponownej walidacji. Nie jest to mAP uśrednione po progach IoU od 0.5 do 0.95 ani odsetek poprawnie rozpoznanych zdjęć.

## Architektura i format

Wagi zostały wczytane z `strict=True` do klasy `MalyViT` z `experiments/vit_latent128_jedno_zdjecie_90/model_snapshot.py`; wszystkie klucze pasują. To ta sama architektura co mały ViT używany w eksperymentach latentowych, z reprezentacją po końcowym GELU o układzie 600 tokenów × 768 współrzędnych dla wejścia 480 × 320.

Plik przechowuje bezpośredni PyTorch `state_dict`, nie kontener z kluczem `net`. Ładowanie:

```python
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

## Pochodzenie i integralność

Źródło na Edenie:

```text
/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_petla43_sd_prowadzenie/wyniki/petla43b_s47/losowe/wagi_koncowe.pth
```

Źródło metryk na Edenie: `OCENY.json` w tym samym katalogu.

Pobrano i zweryfikowano 6 października 2026. Lokalna kopia została następnie przeniesiona z `outputs/audyt_checkpointow_vit_20261006/` do tego folderu na polecenie użytkownika. Model na Edenie pozostaje w swoim miejscu.

- Rozmiar: **11 744 005 bajtów**.
- SHA256 lokalnego pliku i źródła na Edenie:

```text
0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b
```

## Zastosowanie do generacji SD

Zgodność architektury nie oznacza zgodności wartości i znaczenia reprezentacji. Nowe SD wytrenowane na Edenie korzystało z latentów `MODEL_VIT_K1000.pt`. Nie potwierdzono zgodności jego adaptera z latentami tego modelu po 8000 krokach. Dla bezpośredniego warunkowania należy odpowiednio dostroić adapter lub osobno sprawdzić kompatybilność.

Raport audytu i historyczne oceny kandydatów: `outputs/audyt_checkpointow_vit_20261006/RAPORT.md` oraz `EDEN.json`.
