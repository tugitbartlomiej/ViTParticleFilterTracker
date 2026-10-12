# SD od szumu, cel wyłącznie w latencie ViT

Celem jest uzyskanie obrazu, którego naturalny forward zamrożonego ViT zbliża się do zapisanego `after` 600×768. Nie wczytujemy zdjęcia, ramki, segmentacji ani manifestu. Jedyny warunek zależny od przykładu to LATENT_PELNY.npz. Używamy własnego SD EMA020000, pustego tekstu i zerowych trzech kanałów masek. Latent ViT nie jest bezpośrednim wejściem wytrenowanego UNet; steruje gradientem przez dekoder VAE i ViT.

Kontrola: ten sam czysty szum i DDIM bez prowadzenia. Prowadzenie: 40 kroków DDIM, po pierwszych 8 krokach korekta przewidywanego czystego latentu VAE. Następnie 120 kroków optymalizacji końcowego latentu VAE, z karą odejścia od wyniku SD. Wszystkie wagi są zamrożone.

Cel: średnia z błędu standaryzowanego dla J128 i pozostałych640, po wszystkich600 tokenach. Skala każdej współrzędnej to RMS docelowej aktywacji z podłogą0.25. Osobno raportujemy surowy MSE768/MSE128/MSE640, pewność i ramkę naturalnego detektora. Wynik wybieramy według błędu latentu, nie pewności. PNG ponownie wczytujemy i oceniamy, bez podmiany aktywacji.

Brak redukcji błędu względem identycznego losowania SD obala skuteczność prowadzenia w tej próbie. Jeden obraz nie dowodzi poprawy treningu ani istnienia dokładnego obrazu dla sztucznego celu. Pewność detektora nie jest trafnością ani mAP. Nie gwarantujemy zachowania wyniku93.17% sztucznej interwencji.
