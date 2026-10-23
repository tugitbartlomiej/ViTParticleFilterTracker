"""Ocena predykcji P1–P7 z Opis/PREDYKCJE_PRZED_POMIAREM.md na danych replikacji (seed 3047). Progi są tu zakodowane tak, jak zapisano
przed pomiarem; skrypt wypisuje liczby i PASS/FAIL i zapisuje Wyniki/Dane/ocena_predykcji.json.
Uruchomienie: python ocena_predykcji.py <katalog Wyniki/Dane>"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
d = np.load(D / 'plaszczyzna_3047.npz'); kr = d['krok']; K, F = d['fisher'].shape[:2]
PK = ['u1', 'm1', 'u2', 'm2', 'u3', 'm3', 'u4', 'm4', 'u5', 'm5', 'u6', 'm6']
ik = lambda k: int(np.where(kr == k)[0][0])
wyn = {}


def werdykt(nazwa, ok, liczby):
    wyn[nazwa] = dict(PASS=bool(ok), **liczby)
    print(f'{nazwa}: {"PASS" if ok else "FAIL"}  {json.dumps(liczby, ensure_ascii=False)}')


# P1: kolaps uwagi przed detekcją
iou_sr = d['iou'].mean(1)
k_star = int(np.argmax(iou_sr > 0.3)) if (iou_sr > 0.3).any() else None
neff = d['att_neff'].mean(3)                                                      # (K, F, 6) średnia po głowach
med = np.median(neff, 1)                                                          # (K, 6) mediana po klatkach
if k_star is None:
    werdykt('P1', False, dict(uwaga='IoU nigdy > 0,3'))
else:
    b4, b5 = med[k_star, 3], med[k_star, 4]; b4k, b5k = med[ik(1000), 3], med[ik(1000), 4]
    werdykt('P1', b4 < 100 and b5 < 100 and b4k < 50 and b5k < 50,
            dict(k_star=int(kr[k_star]), iou_k_star=round(float(iou_sr[k_star]), 3), zrodla_b4_b5_k_star=[round(float(b4), 1), round(float(b5), 1)],
                 zrodla_b4_b5_k1000=[round(float(b4k), 1), round(float(b5k), 1)], zrodla_po_krokach_b4=[[int(k), round(float(v), 1)] for k, v in zip(kr[:12], med[:12, 3])]))

# P2: gwiazda na narzędziu
late = kr >= 10000
hub = d['hub_top3'][late][:, :, 3:5]                                              # (K', F, 2)
wzb = d['att_bg2tool'].mean(3)[late][:, :, 3:5]                                   # (K', F, 2)
udz_hub = float(hub.mean()); klatek_ok = (wzb >= 10).sum(1)                       # (K', 2) ile klatek z wzbogaceniem ≥ 10
werdykt('P2', udz_hub >= 0.9 and bool((klatek_ok >= 8).all()),
        dict(udzial_top3_w_obrysie_lub_halo=round(udz_hub, 3), min_klatek_z_wzbogaceniem_ge10_b4_b5=[int(v) for v in klatek_ok.min(0)],
             wzbogacenie_mediana_b4_b5=[round(float(np.median(wzb[..., j])), 1) for j in range(2)], checkpointow=int(late.sum())))

# P3: ruch prosty, potem zygzak
bal = d['bal'][:, :, 0].mean(1)                                                    # (K,) średnia po klatkach
b0 = float(bal[ik(0)]); bmax = float(max(bal[ik(300)], bal[ik(1000)], bal[ik(3000)])); b_late = bal[kr >= 24000]
werdykt('P3', 0.8 <= b0 <= 1.2 and bmax >= 2.0 and bool((b_late <= 1.2).all()),
        dict(bal_k0=round(b0, 3), bal_maks_300_1000_3000=round(bmax, 3), bal_maks_po_24000=round(float(b_late.max()), 3),
             bal_po_krokach=[[int(k), round(float(v), 2)] for k, v in zip(kr, bal)]))

# P4: złożenie siodeł nie kurczy (jakobiany)
J = D / 'jakobiany_3047.npz'
if J.exists():
    j = np.load(J); sv = j['sv_prod'][..., 11, :]                                  # (Kj, F, 6 tokenów, 192) po 12. podkroku
    udz = (sv < 1).mean(-1)                                                        # (Kj, F, 6)
    nk = j['nakl']; nakl = nk[..., 0] / np.maximum(np.minimum(nk[..., 1], nk[..., 2]), 1); los = np.maximum(nk[..., 1], nk[..., 2]) / 192
    roz = (nakl - los).mean(-1)                                                    # (Kj, F, 6) średnia po 5 parach
    werdykt('P4', bool(((udz >= 0.45) & (udz <= 0.55)).all()) and bool((roz <= 0.03).all()),
            dict(udzial_sigma_lt1_min_maks=[round(float(np.nanmin(udz)), 3), round(float(np.nanmax(udz)), 3)],
                 nakladanie_minus_losowe_maks=round(float(np.nanmax(roz)), 4), nakladanie_minus_losowe_srednia=round(float(np.nanmean(roz)), 4),
                 blad_odtworzenia_maks=float(np.nanmax(j['blad']))))
else:
    print('P4: brak jakobiany_3047.npz')

# P5: front rozdziału w bloku 3–4, blok 6 nie podnosi
fi = d['fisher'][kr >= 24000]                                                      # (K', F, 13)
skok = np.argmax(np.diff(fi, axis=2), axis=2)                                      # (K', F) indeks podkroku 0..11 z największym przyrostem
w_b34 = np.isin(skok, [4, 5, 6, 7]).sum(1)                                         # u3, m3, u4, m4
m6_ok = (fi[:, :, 12] <= 1.05 * fi[:, :, 11]).sum(1)
ok5 = ((w_b34 >= 6).mean() >= 0.8) and ((m6_ok >= 6).mean() >= 0.8)
werdykt('P5', bool(ok5), dict(odsetek_checkpointow_skok_b34_ge6=round(float((w_b34 >= 6).mean()), 2), odsetek_checkpointow_m6_ge6=round(float((m6_ok >= 6).mean()), 2),
                              klatek_ze_skokiem_b34_srednio=round(float(w_b34.mean()), 1), rozklad_podkroku_skoku={PK[s]: int((skok == s).sum()) for s in range(12)},
                              fisher_srednio_m4_m5_m6=[round(float(fi[:, :, s].mean()), 2) for s in (8, 10, 12)]))

# P6: uwaga nie wygładza po siatce
sas = d['att_sas8'].mean(3)                                                        # (K, F, 6)
dD = np.diff(d['dir'], axis=2).mean(1)                                             # (K, 12) średnia po klatkach
mlp_s, uw_s = dD[:, 1::2].sum(1), dD[:, 0::2].sum(1)
late1 = kr >= 1000
werdykt('P6', bool((sas < 0.026).all()) and bool((mlp_s[late1] < 0).all()) and bool((mlp_s[late1] < uw_s[late1]).all()),
        dict(sas8_maks=round(float(sas.max()), 4), suma_zmian_dirichlet_mlp_min_maks=[round(float(mlp_s[late1].min()), 3), round(float(mlp_s[late1].max()), 3)],
             suma_zmian_dirichlet_uwaga_min_maks=[round(float(uw_s[late1].min()), 3), round(float(uw_s[late1].max()), 3)]))

# P7: przejście ściska, narzędzie mocniej
if J.exists():
    ld = j['logdet'].sum(-1)                                                       # (Kj, F, 6)
    kj = j['krok']; narz_sr = ld[:, :, :3].mean(2); tlo_sr = ld[:, :, 3:].mean(2)   # (Kj, F)
    po10 = kj >= 10000; po24 = kj >= 24000
    ok7a = bool((narz_sr[po10].mean(1) < 0).all()); ok7b = bool(((narz_sr[po24] < tlo_sr[po24]).sum(1) >= 6).all())
    werdykt('P7', ok7a and ok7b, dict(logdet_narz_po_krokach=[[int(k), round(float(v), 2)] for k, v in zip(kj, narz_sr.mean(1))],
                                     logdet_tlo_po_krokach=[[int(k), round(float(v), 2)] for k, v in zip(kj, tlo_sr.mean(1))],
                                     klatek_narz_lt_tlo_po_24000=[int(v) for v in (narz_sr[po24] < tlo_sr[po24]).sum(1)]))
json.dump(wyn, open(D / 'ocena_predykcji.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('PASS:', sum(v['PASS'] for v in wyn.values()), '/', len(wyn))
