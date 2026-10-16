# Ramki obiektów ImageNet dla Imagenette (2026-10-11)

- ILSVRC2012_bbox_train_v2.tar.gz (MD5 9271167e2176350e65cfe4e546f14b17) i ILSVRC2012_bbox_val_v3.tgz (MD5 f4cd18b5ea29fe6bbea62ec9c20d80f0),
  pobrane z https://image-net.org/data/ILSVRC/2012/ ; nierozpakowane (kwota plików).
- imagenette_bbox.json: ramki dopasowane po nazwie pliku do imagenette2-320. Klucz = ścieżka względna (train/<wnid>/<plik>.JPEG).
  Pola: split, klasa, rozmiar_xml, rozmiar_pliku, ramki_xml, ramki_plik (piksele pliku 320), ramki_wyciecie320 (środkowe wycięcie 320x320 jak w treningu).
  Format ramki: {"wnid", "xyxy": [x1, y1, x2, y2]}.
- Pokrycie: 5505 z 13394 obrazów ma ramki (train 3891 z 9469, val 1614 z 3925); reszta obrazów ImageNet nie ma ramek w ILSVRC2012.
- Skrypt: ~/smallvit_runs/2026-10-10_szklo_pokretla/kod/ramki_imagenette.py ; kontrola wzrokowa: kontrola_ramek.png
