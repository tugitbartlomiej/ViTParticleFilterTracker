# Wyniki: vit_k1000_sd_regresja_1500

## 2026-10-08: warunek wstępny niespełniony, generacji nie uruchomiono

Szczegóły i tabela są w [PLAN.md](PLAN.md).

- **Pudło naturalne:** sztuczny latent K1000 zdjęcia 0055/4019, liczony procedurą etapu 8, daje pewność 0,055 przy IoU 0,70, czyli detekcji nie ma. Przy 3000 krokach wynik jest taki sam jak przy 300, bo optimum leży na granicy pudła min–max puli.
- **Samo pudło z wag:** detekcja się udaje, z pewnością 0,93 i IoU 1,00. Latent zmienia się jednak o 11 norm, a „poza” wynosi 91,8 wobec 1,7 dla prawdziwych zdjęć.
- **Porównanie z K8000:** z* K8000 tego zdjęcia ma pewność 0,81 i IoU 1,00 przy „poza” 19,8.
- **Koszt generacji:** stare SD na RTX 3070 robi obraz w 0,88 s. 1500 startów po 166 obrazów to około 62 h lokalnie, a na Edenie szacunkowo 10–15 h karty.

Generacja czeka na decyzję użytkownika o celu.
