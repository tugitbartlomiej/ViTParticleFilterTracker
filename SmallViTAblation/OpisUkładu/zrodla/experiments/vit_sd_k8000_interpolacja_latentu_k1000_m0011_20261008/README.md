# SD K8000: interpolacja sztucznego latentu K1000 dla m0011

Eksperyment generuje obrazy bezpośrednio z 11 warunków latentowych leżących na odcinku od sztucznego latentu `m0011` do jego oryginalnego latentu. Używa modelu `SD_ViT_K8000_A_control_K10500` i tego samego szumu dla całej serii.

Uruchomienie z `SmallViTAblation`:

```powershell
F:/Python/python.exe -B experiments/vit_sd_k8000_interpolacja_latentu_k1000_m0011_20261008/generuj.py
```

Wyniki są zapisywane do `outputs/latent_cel_do_obrazow/sd_k8000_m0011_interpolacja`.

To celowy test zgodności krzyżowej. Warunek pochodzi z K1000, a generator był uczony na K8000. Wyników nie należy opisywać jako wiernej rekonstrukcji przestrzeni K8000.

