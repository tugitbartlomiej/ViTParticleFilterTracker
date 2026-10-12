# Cztery niezależne szumy SD

Wyłącznie przybliżenia oryginalnego sztucznego celu. Dokładny cel wyklucza diagnostyka osiągalności. Wagi ViT zamrożone. Bez zdjęcia źródłowego, treningu i optymalizacji po DDIM.

Ziarno 2026100511: MSE768=14.319647, kontrola=14.338458, zmiana=0.131 procent. MSE128=85.424622, MSE640=0.098652, cosinus=0.129833, pewność=0.424514.

Ziarno 2026100523: MSE768=14.287478, kontrola=14.332698, zmiana=0.315 procent. MSE128=85.503624, MSE640=0.044248, cosinus=0.141585, pewność=0.878570.

Ziarno 2026100537: MSE768=14.279775, kontrola=14.282085, zmiana=0.016 procent. MSE128=85.404564, MSE640=0.054814, cosinus=0.141318, pewność=0.720236.

Ziarno 2026100549: MSE768=14.317536, kontrola=14.338115, zmiana=0.144 procent. MSE128=85.508049, MSE640=0.079433, cosinus=0.134358, pewność=0.562365.

Różnorodność: DIVERSITY.json, pary obrazów i ich naturalnych latentów dla obu ramion. Pewność detektora nie jest mAP ani dowodem poprawnej detekcji. Prowadzenie może obniżyć MSE bez zachowania realizmu; obrazy wymagają oceny człowieka. Ta próba nie dotyczy treningu na zdjęciu z wyszukiwania.
