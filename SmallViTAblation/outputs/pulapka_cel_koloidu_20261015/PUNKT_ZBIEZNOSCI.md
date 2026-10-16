# Do czego zbiega pułapka uczenia: prognoza punktu końcowego z części uczenia

m(t) = m_∞ + A·t^−α dopasowane na krokach ≤ t_c; błąd względny = |m_∞ − m_T| / |m(t_c) − m_T| (1 = brak zysku z prognozy).

## 226
- t_c = 8000: α = 0.10; błąd względny prognozy 5.08; prognoza pokrywa 171% prawdziwej drogi do końca (w jej kierunku)
- t_c = 16000: α = 0.10; błąd względny prognozy 6.91; prognoza pokrywa 25% prawdziwej drogi do końca (w jej kierunku)
- t_c = 32000: α = 0.10; błąd względny prognozy 10.05; prognoza pokrywa -93% prawdziwej drogi do końca (w jej kierunku)

## w ramce (średnio)
- t_c = 8000: α = 0.10; błąd względny prognozy 5.91; prognoza pokrywa 176% prawdziwej drogi do końca (w jej kierunku)
- t_c = 16000: α = 0.10; błąd względny prognozy 7.95; prognoza pokrywa 91% prawdziwej drogi do końca (w jej kierunku)
- t_c = 32000: α = 0.10; błąd względny prognozy 11.51; prognoza pokrywa -111% prawdziwej drogi do końca (w jej kierunku)
