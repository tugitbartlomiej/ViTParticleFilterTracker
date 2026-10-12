# Wybrane checkpointy

Oba modele zaczynają od `MALY_VIT_K8000_s47_losowe.pth` i wykonują 500 aktualizacji z identycznymi ustawieniami. Pula ma 256 obrazów: 255 wspólnych i jeden zmieniany.

- `control_step500.pt` - gałąź kontrolna, slot 256 zawiera `TR_0084_S1_P03/567`; SHA256 `ad8dff0e4113df6d534be679ef8b80337e18ba8ee5688cb3d628c5daec38861e`.
- `intervention_05_TR_0007_S1_P03_frame1165_step500.pt` - najlepszy kompromis w tej pojedynczej próbie; slot 256 zawiera `TR_0007_S1_P03/1165`; SHA256 `a152bad6289643653c0c5f11e060585cd4111f359d4ee618901faeafede5f1c5`.

Checkpoint interwencji 05 poprawił względem kontroli przyrost best IoU sondy o `+0.0081` i przyrost guard AP50 o `+0.0077`; efekt confidence był `0.0000`. To wynik jednego seeda i wymaga replikacji przed uznaniem obrazu za wiarygodnie korzystny.

Oba pliki zostały ponownie wczytane lokalnie z `strict=True`; każdy zawiera stan modelu o 2 926 832 parametrach.
