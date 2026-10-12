# Wynik podmiany 8 z 256 obrazów

Obie gałęzie zaczęły od tego samego checkpointa K8000 i wykonały 500 aktualizacji. Kontrola zawierała 256 oryginalnych obrazów; wariant zawierał 248 tych samych obrazów i osiem unikalnych obrazów DINO.

- Efekt best IoU ponad kontrolę: +0.002301
- Efekt top confidence ponad kontrolę: +0.003985
- Efekt IoU ramki top-confidence ponad kontrolę: -0.000269
- Efekt guard AP50 ponad kontrolę: +0.000042

Wynik jest pojedynczą próbą z jednym seedem. Do potwierdzenia wpływu zestawu potrzebne są powtórzenia na kilku seedach i innych ośmiu usuwanych obrazach kontrolnych.
