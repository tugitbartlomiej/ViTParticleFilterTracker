# SD z najbliższego zdjęcia, prowadzone sztucznym latentem

Uruchom w katalogu SmallViTAblation:

```powershell
$env:PYTHONPATH='F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/deps'
$env:PYTHONUTF8='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:USE_TF='0'
$env:USE_FLAX='0'
F:/Studia/PhD_projekt/VIT/venvs/rfdetr_1_10_1/Scripts/python.exe -u experiments/vit_latent768_sd_od_najblizszego/generuj.py
```

Nie nadpisuje wcześniejszej próby od szumu. Cztery wyniki prowadzone i dwie kontrole. Obraz wybrany: outputs/vit_latent768_sd_od_najblizszego/OBRAZ_SD.png. Detektor działa naturalnie, bez hooka podmieniającego aktywacje.
