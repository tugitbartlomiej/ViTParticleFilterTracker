# Generowanie z szumu według pełnego latentu

Uruchomienie z katalogu projektu:

```powershell
$env:PYTHONPATH='F:/Studia/PhD_projekt/VIT/eden_archiwum/sd_modele/deps'
$env:PYTHONUTF8='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:USE_TF='0'
$env:USE_FLAX='0'
& F:/Studia/PhD_projekt/VIT/venvs/rfdetr_1_10_1/Scripts/python.exe -u experiments/vit_latent768_sd_od_szumu/generuj.py
```

Obraz końcowy: outputs/vit_latent768_sd_od_szumu/OBRAZ_SD.png. Plan określa warunek, kontrolę i ograniczenia. Program nie wczytuje zdjęcia źródłowego ani etykiet. UNet przenoszony na CPU podczas różniczkowania VAE, aby zmieścić się na RTX3070.
