# img2img SD K8000 sterowane tylko sztucznym latentem (2026-10-08)

Jeden obraz: klatka TRAIN 0055/4019 (TR_0055_S1_P03), na której ViT K8000 nie wykrywa cystotomu. Sztuczny latent z*
z `outputs/latent_cel_do_obrazow/k8000_trening/zb_j10389_minmax.npy` (korekta detekcji w zakresach kanałów,
||δ|| = 2,9 ||z0||, odległość od naturalnych tokenów 19,8 wobec 3,1 dla zdjęcia). Metoda jak `sd_prowadzone.py`
użytkownika: zdjęcie celu -> VAE -> zaszumienie do siły s -> DDIM 50 leading; prowadzenie energią średnia (z − z*)^2
po pełnym latencie (2 kroki gradientu po x̂0 na krok, eta 0,15 σ_t, ostatnie 5 kroków bez prowadzenia), 3 rundy.
Różnica wobec dawnego prowadzenia: warunkiem SD jest teraz z*, nie z0. Bez straty detekcji.

Kod `kod/img2img_sterowanie.py`, wyniki `outputs/sd_img2img_sterowanie_latentem/` (WYNIK.json, SIATKA.jpg, `obrazy/`,
zdjęcie celu ZDJECIE_CELU_0055_4019_*.png/jpg).

| wariant | siła | ułamek δ | ułamek widocznej dla fc2 | cos do δ | bliżej z* | poza | PSNR do zdjęcia |
|---|---|---|---|---|---|---|---|
| kontrola: warunek z0 | 0,3 / 0,5 / 0,7 | 0,3 / 0,0 / 0,0 % | 1 / 0 / −1 % | ≈ 0 | −1 do −2 % | 2,7-2,9 | 26 / 24 / 22 dB |
| warunek z*, bez prowadzenia | 0,3 / 0,5 / 0,7 | 4,4 / 5,5 / 5,2 % | 0 / 1 / 2 % | 0,12-0,15 | −1 do −2 % | 2,5-2,6 | 25 / 22 / 20 dB |
| warunek z* + prowadzenie, runda 1 | 0,3 / 0,5 / 0,7 | 9,2 / 10,7 / 11,8 % | 6,6 / 9,9 / 10,8 % | 0,24-0,29 | 1,6-3,6 % | 2,8-3,3 | 25 / 22 / 20 dB |
| warunek z* + prowadzenie, najlepsza runda | 0,3 r2 / 0,7 r3 | 12,7 / 12,4 % | 11,9 / 12,0 % | 0,30 | 3,5 / 4,1 % | 3,3 | 22 / 16 dB |

- img2img zachowuje scenę celu; sam warunek z* dokłada ok. 5 % przyrostu, prawie nic z części widocznej dla fc2.
- Prowadzenie odległością od z* podwaja to do ok. 12 %, a część widoczna dla fc2 rośnie do 12 %, przy tokenach nadal
  naturalnych (poza 3,3 wobec 3,1 zdjęcia). Najlepszy kompromis: siła 0,3, druga runda (PSNR 22 dB).
- Dalsze rundy i większa siła nie dokładają kierunku, tylko oddalają obraz od zdjęcia.
- Wizualnie przyrost pojawia się jako cienkie niebiesko-metaliczne kreski na źrenicy (ten sam motyw co obraz zastępczy
  SD z szumu), więcej przy prowadzeniu.
- Odległość od z* maleje tylko o 3-4 %: większa część δ jest dla obrazów nieosiągalna. Wcześniej z szumu bez img2img
  i bez prowadzenia: ok. 2 % netto; prowadzenie k1000 przy sztucznym celu: 0,5 %.

Jeden obraz i jedno ziarno na rundę, więc to pokaz metody, nie pomiar istotności.
