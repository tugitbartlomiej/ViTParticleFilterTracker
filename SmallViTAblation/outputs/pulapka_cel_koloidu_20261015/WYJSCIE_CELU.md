# Do jakiego punktu dąży koloid: wyjście głowy tokenu 226 (środek ramki)

Cel: heat = 1 (logit → +∞), ltrb = [4.380000114440918, 0.25, 4.420000076293945, 1.1699999570846558] (w krokach siatki 16 px).
- krok 0: logit -4.41 (p = 0.0120), siła na logit 1.07e+00; ltrb [3.06, 2.18, 3.28, 2.42] (średni błąd 1.407); |W1| 7.99, |W2| 1.32, |x| 14.6
- krok 500: logit -0.06 (p = 0.4855), siła na logit 3.22e-01; ltrb [3.99, 0.61, 4.48, 1.35] (średni błąd 0.247); |W1| 8.01, |W2| 1.30, |x| 31.5
- krok 1100: logit -0.18 (p = 0.4554), siła na logit 3.74e-01; ltrb [4.75, 0.44, 4.51, 1.24] (średni błąd 0.182); |W1| 8.05, |W2| 1.27, |x| 33.0
- krok 10100: logit +2.33 (p = 0.9114), siła na logit 2.02e-03; ltrb [3.87, 0.15, 4.41, 0.98] (średni błąd 0.199); |W1| 9.22, |W2| 1.30, |x| 53.9
- krok 28444: logit +2.15 (p = 0.8961), siła na logit 3.24e-03; ltrb [4.39, 0.23, 4.47, 1.11] (średni błąd 0.036); |W1| 10.89, |W2| 1.67, |x| 60.9
- krok 46800: logit +2.65 (p = 0.9338), siła na logit 8.51e-04; ltrb [4.44, 0.32, 4.79, 1.15] (średni błąd 0.132); |W1| 11.89, |W2| 1.99, |x| 60.6
- krok 65640: logit +2.93 (p = 0.9495), siła na logit 3.81e-04; ltrb [4.5, 0.35, 4.33, 1.13] (średni błąd 0.090); |W1| 12.42, |W2| 2.22, |x| 56.6

logit ~ 0.70·ln t + -4.67 (R² 0.721, kroki ≥ 1000); siła na logit ~ t^-1.76.
Błąd ltrb: krok 1000 0.185 → koniec 0.090.
