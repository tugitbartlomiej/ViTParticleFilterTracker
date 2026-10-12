"""Wyciąga do assets/ dane filmu o cząsteczkach i kierunkach tożsamości (pary 400, LaMa).

Nic nie liczy na sieci. Czyta tylko zapisane wyniki eksperymentu
experiments/maly_vit_tozsamosc_czasteczki_pary_400 i zdjęcia par z
experiments/maly_vit_test_orientacji_narzedzia_400. Każda liczba w filmie pochodzi stąd.

    python przygotuj.py  ->  assets/dane.json, assets/*.png, ZRODLA.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent / 'experiments'
PARY = EXP / 'maly_vit_tozsamosc_czasteczki_pary_400'
ZDJ = EXP / 'maly_vit_test_orientacji_narzedzia_400'
A = HERE / 'assets'
SRC = 8            # TR_0053_S1_P03/00000881, przykład z PODSUMOWANIE.md
TOKEN0 = 279       # cząsteczka 280 liczona od 1


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    A.mkdir(exist_ok=True)
    rows = json.loads((ZDJ / 'manifest.json').read_text(encoding='utf-8'))
    podzial = json.loads((PARY / 'PODZIAL.json').read_text(encoding='utf-8'))
    split = np.array([r['split'] for r in podzial['records']])
    r = rows[SRC]
    assert r['frame_id'] == 'cataract_lmm/tracking/TR_0053_S1_P03/00000881'

    # Zdjęcia: tak jak pomiar, INTER_AREA do 480 x 320.
    orig = cv2.imread(str(ZDJ / r['image_path']))
    lama = cv2.imread(str(ZDJ / r['synthetic_path']))
    mask = cv2.imread(str(ZDJ / r['mask_480_path']), cv2.IMREAD_GRAYSCALE)
    for name, im in (('obraz_oryginal.png', orig), ('obraz_lama.png', lama)):
        cv2.imwrite(str(A / name), cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA))
    # Powiększenie 3 x 3 pól wokół cząsteczki 280 (środek: wiersz 9, kolumna 9), bez wygładzania.
    r0, c0 = TOKEN0 // 30, TOKEN0 % 30
    for name, im in (('latka_oryginal.png', orig), ('latka_lama.png', lama)):
        small = cv2.resize(im, (480, 320), interpolation=cv2.INTER_AREA)
        crop = small[(r0 - 1) * 16:(r0 + 2) * 16, (c0 - 1) * 16:(c0 + 2) * 16]
        cv2.imwrite(str(A / name), cv2.resize(crop, (288, 288), interpolation=cv2.INTER_NEAREST))
    rgba = np.zeros((320, 480, 4), np.uint8)
    rgba[mask > 0] = (80, 224, 166, 200)          # BGR + alfa: zielona maska
    cv2.imwrite(str(A / 'maska.png'), rgba)

    c = np.load(PARY / 'TOKENY_PARY.npz')
    P = np.load(PARY / 'PODPRZESTRZEN.npz')
    U = P['U'].astype(np.float64)
    a, b = c['offsets'][SRC:SRC + 2]
    ids = c['token_ids'][a:b].astype(int)
    cov = c['coverage'][a:b].astype(float)
    st = c['states'][a:b].astype(np.float64)
    j = int(np.flatnonzero(ids == TOKEN0)[0])
    z_o, z_n = st[j, 0], st[j, 1]
    delta = z_o - z_n

    # Mapa reakcji: długość rzutu zmiany na 8 kierunków, zapisana w PODPRZESTRZEN.npz.
    mapa = P['identity_projection_maps'][SRC].astype(float)
    sprawdz = float(np.linalg.norm(U.T @ delta))
    assert abs(sprawdz - mapa[TOKEN0]) < 1e-3, (sprawdz, mapa[TOKEN0])

    # Osie o najpewniejszej zmianie (ranking z części uczącej).
    makro = json.loads((PARY / 'MAKROWEKTORY.json').read_text(encoding='utf-8'))[:6]
    osie = [dict(os=m['channel1'], idx=m['channel0'], srednia=m['train_mean_delta'], sd=m['train_std_delta'],
                 znak_test=m['test_image_sign_agreement'], czastka=float(delta[m['channel0']])) for m in makro]

    # Chmura zmian cząsteczek z części uczącej: najpierw w dwóch surowych osiach, potem w dwóch kierunkach SVD.
    img_of = np.repeat(np.arange(400), np.diff(c['offsets']))
    tr = split[img_of] == 'train'
    D = (c['states'][:, 0] - c['states'][:, 1]).astype(np.float64)[tr]
    covtr = c['coverage'][tr].astype(float)
    bimg = np.repeat(np.arange(400), 8)
    Dbg = (c['background_states'][:, 0] - c['background_states'][:, 1]).astype(np.float64)[split[bimg] == 'train']
    rng = np.random.default_rng(0)
    pick = rng.choice(len(D), size=min(1500, len(D)), replace=False)
    pickb = rng.choice(len(Dbg), size=min(600, len(Dbg)), replace=False)
    o1, o2 = osie[0]['idx'], osie[1]['idx']
    chmura = dict(
        surowe=D[pick][:, [o1, o2]].round(4).tolist(),
        svd=(D[pick] @ U[:, :2]).round(4).tolist(),
        tlo_surowe=Dbg[pickb][:, [o1, o2]].round(4).tolist(),
        tlo_svd=(Dbg[pickb] @ U[:, :2]).round(4).tolist(),
        pokrycie=covtr[pick].round(3).tolist(),
        osie=[osie[0]['os'], osie[1]['os']],
        n_czastek=int(tr.sum()), n_tla=int(len(Dbg)))

    ident = json.loads((PARY / 'IDENTYFIKACJA.json').read_text(encoding='utf-8'))
    ev = P['eigenvalues']
    energia = (np.cumsum(ev) / ev.sum())[:40]
    assert abs(energia[7] - ident['energy_selected_rank']) < 1e-6

    # Odczyt obecności na odłożonych filmach (test): rzut uśrednionego latentu na 8 kierunków i regresja logistyczna.
    te = np.flatnonzero(split == 'test')
    pooled = c['pooled'].astype(np.float64)
    def prob(X):
        v = ((X @ U - P['probe_mu']) / P['probe_sd']) @ P['probe_coef'] + float(P['probe_intercept'])
        return 1 / (1 + np.exp(-np.clip(v, -60, 60)))
    p_o, p_n = prob(pooled[te, 0]), prob(pooled[te, 1])
    auc_bez_wag = float(roc_auc_score(np.r_[np.ones(len(te)), np.zeros(len(te))], np.r_[p_o, p_n]))
    test = ident['presence']['test']['film_weighted']
    test_obrazy = ident['presence']['test']['image_weighted']

    wplyw = json.loads((PARY / 'CZASTECZKI_WPLYW.json').read_text(encoding='utf-8'))[0]
    assert wplyw['source_index'] == SRC
    det = json.loads((PARY / 'DETEKCJA.json').read_text(encoding='utf-8'))['summary']
    fc2 = json.loads((PARY / 'DETEKCJA_FC2.json').read_text(encoding='utf-8'))
    sham = json.loads((PARY / 'SHAM_PODSUMOWANIE.json').read_text(encoding='utf-8'))['heldout']

    dane = dict(
        klatka=r['frame_id'], token_ids=ids.tolist(), pokrycie=cov.round(3).tolist(),
        n_czastek_maski=len(ids), n_25=int((cov >= .25).sum()),
        czastka=dict(numer=TOKEN0 + 1, wiersz=TOKEN0 // 30, kolumna=TOKEN0 % 30, pokrycie=float(cov[j]),
                     z_o=z_o.round(4).tolist(), z_n=z_n.round(4).tolist(), rzut=float(mapa[TOKEN0])),
        osie=osie, chmura=chmura, energia=energia.round(4).tolist(), r50=int(ident['r50']), r90=int(ident['r90']),
        energia8=float(ident['energy_selected_rank']),
        kierunek1=ident['mixed_directions'][0]['top_channels'][:5],
        odczyt=dict(p_oryginal=p_o.round(4).tolist(), p_lama=p_n.round(4).tolist(), prog=float(ident['chosen_threshold']),
                    auroc=float(test['auroc']), trafnosc=float(test['balanced_accuracy']), auroc_bez_wag=auc_bez_wag, auroc_obrazy_zapis=float(test_obrazy['auroc']),
                    lda_trafnosc=float(ident['lda']['test']['balanced_accuracy']), n_test=len(te),
                    n_filmow=sum(v == 'test' for v in podzial['films'].values())),
        mapa=mapa.round(4).tolist(),
        wplyw=[dict(token=t['token1'], dq=t['delta_q']) for t in wplyw['token_influence']],
        detekcja=dict(bez=det['nohook']['q'], tozsamosc=det['identity']['q'],
                      losowe=[det[k]['q'] for k in ('random11', 'random22', 'random33')],
                      losowe_fc2=[fc2['summary'][k]['q'] for k in ('fc2random11', 'fc2random22', 'fc2random33')],
                      ap_bez=det['nohook']['AP50'], ap_tozsamosc=det['identity']['AP50'],
                      ap_ci=det['identity']['AP50_paired_delta_bootstrap95'],
                      q_ci=det['identity']['paired_delta_film_bootstrap95']['q']),
        sham=dict(atrapa=sham['original_tool_roi_projection_l2'], usuniecie=sham['removal_tool_projection_mean'],
                  n=sham['n'], zmiany_decyzji=sham['presence_call_flips']),
    )
    (A / 'dane.json').write_text(json.dumps(dane, ensure_ascii=False, indent=1), encoding='utf-8')

    zrodla = {str(p): sha(p) for p in [ZDJ / r['image_path'], ZDJ / r['synthetic_path'], ZDJ / r['mask_480_path'],
                                        PARY / 'TOKENY_PARY.npz', PARY / 'PODPRZESTRZEN.npz', PARY / 'MAKROWEKTORY.json',
                                        PARY / 'IDENTYFIKACJA.json', PARY / 'CZASTECZKI_WPLYW.json', PARY / 'DETEKCJA.json',
                                        PARY / 'DETEKCJA_FC2.json', PARY / 'SHAM_PODSUMOWANIE.json', PARY / 'PODZIAL.json']}
    (HERE / 'ZRODLA.json').write_text(json.dumps(zrodla, ensure_ascii=False, indent=1), encoding='utf-8')
    print('cząsteczek maski', len(ids), '| rzut 280', round(sprawdz, 3), '| energia 8', round(energia[7], 4),
          '| AUROC test zapis', round(test['auroc'], 4), 'bez wag', round(auc_bez_wag, 4),
          '| trafność', round(test['balanced_accuracy'], 4), '| chmura', len(pick), len(pickb))


if __name__ == '__main__':
    main()
