@echo off
rem Lokalne uruchomienie petli wersji 4 (PLAN.md): sciezki wejsc jak w wersji 1-3 (kopie lokalne), Stable Diffusion
rem z eden_archiwum\sd_modele, adnotacje Cataract-LMM z Dataset. Uzycie: uruchom_lokalnie.cmd skrypt.py [argumenty]
set PYTHONUTF8=1
set KMP_DUPLICATE_LIB_OK=TRUE
set PY=F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1\Scripts\python.exe
set EXP=F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments
set SDM=F:\Studia\PhD_projekt\VIT\eden_archiwum\sd_modele
set MALY_VIT_WEJSCIA=%EXP%\maly_vit_od_zera\eden\wejscia
set MALY_VIT_DANE=%EXP%\koopman_fastgan_petla\local_inputs\data
set MALY_VIT_CACHE=%EXP%\maly_vit_od_zera\cache
set MALY_VIT_WYNIKI=%EXP%\maly_vit_petla_sd_rownanie\wyniki
set MALY_VIT_ADNOTACJE=F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Dataset\Cataract-LMM\3_Object_Tracking\annotations
set MALY_VIT_SD_UNET=%SDM%
set MALY_VIT_SD_VAE=%SDM%\sd-vae-ft-mse
set MALY_VIT_SD_TEKST=%SDM%\pusty_tekst.pt
set PYTHONPATH=%SDM%\deps
cd /d %~dp0
"%PY%" -u %*
