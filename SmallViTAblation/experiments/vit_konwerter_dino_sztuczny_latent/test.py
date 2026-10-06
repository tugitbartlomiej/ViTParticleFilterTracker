"""Sztuczny latent J128 -> konwerter v2 -> cechy DINO -> wyszukiwanie w bazie 35000 klatek TRAIN.

Sztuczny latent: outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz (before / after, 600x768), zmiana
128 kanałów na zdjęciu TR_0154_S1_P03 kl. 1926 (confidence 0,868 -> 0,932, IoU 0,857 -> 0,999).

Pytanie: czy zdjęcia znalezione cechami przewidzianymi ze sztucznego latentu mają naturalny latent bliższy
sztucznemu celowi na 128 kanałach (i lepszą detekcję) niż zdjęcia znalezione z latentu przed zmianą i losowe?
Weryfikacja: każde znalezione zdjęcie przechodzi przez zamrożony ViT; cos128/MSE128 do celu i IoU do etykiety.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KONW = ROOT / 'experiments' / 'vit_konwerter_dino_filmy_eden'
sys.path.insert(0, str(KONW))
from dane import przygotuj  # noqa: E402
from model import KonwerterGotowy, cechy_dino, latent_vit, wczytaj_dino, wczytaj_vit  # noqa: E402
from model_vit import dekoduj  # noqa: E402

OUT = ROOT / 'outputs' / HERE.name
ZRODLO = ROOT / 'outputs/vit_latent128_jedno_zdjecie_90'
KONWERTER = ROOT / 'outputs/vit_konwerter_dino_filmy_eden/v2/KONWERTER.pt'
VIT = ZRODLO / 'MODEL_VIT_K1000.pt'
BAZA = ROOT / 'outputs/vit_dino_faiss_35000'
DANE = ROOT.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
DINO_REPO = Path('C:/Users/bartl/.cache/torch/hub/facebookresearch_dinov2_main')
DINO_WAGI = Path('C:/Users/bartl/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
FILM, KLATKA = 'TR_0154_S1_P03', 1926
TOPK = 32


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jpeg(archiwum, nazwa):
    with zipfile.ZipFile(archiwum) as z:
        return np.asarray(Image.open(io.BytesIO(z.read(nazwa))).convert('RGB'))


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    vit, _ = wczytaj_vit(VIT, dev)
    dino, _ = wczytaj_dino(DINO_REPO, DINO_WAGI, dev)
    konw = KonwerterGotowy(KONWERTER, device=dev)
    ck = torch.load(KONWERTER, map_location='cpu', weights_only=False)
    mu = F.normalize(ck['y_mu'].float(), dim=0).to(dev)
    x_mean, x_std = konw.model.x_mean, konw.model.x_std

    L = np.load(ZRODLO / 'LATENT_PELNY.npz')
    before, after = (torch.from_numpy(L[k]).to(dev) for k in ('before', 'after'))
    kan = torch.from_numpy(L['channels0']).to(dev)
    baza = torch.from_numpy(np.load(BAZA / 'features.npy')).to(dev)
    wiersze = [json.loads(l) for l in (BAZA / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]

    # 1. Zdjęcie źródłowe: nasz przepis obrazu kontra zapisany latent "before".
    zrodlo = jpeg(DANE / f'{FILM}.zip', f'{FILM}/{FILM}_{KLATKA:06d}.jpg')
    v, c, p = (t[None].to(dev) for t in przygotuj(zrodlo))
    with torch.no_grad():
        z_nasz = latent_vit(vit, v)[0]
        y_cls, _ = cechy_dino(dino, c, p)
        pb, tb = konw.predict(before)
        pa, ta = konw.predict(after)
        pn, _ = konw.predict(z_nasz)
    zgodnosc_przepisu = float((z_nasz - before).norm() / before.norm())

    # 2. Jak daleko sztuczny latent leży od rozkładu uczącego (w standaryzacji konwertera).
    sb, sa = (before - x_mean) / x_std, (after - x_mean) / x_std
    odleglosc = dict(
        maks_odchylen_przed=float(sb.abs().max()), maks_odchylen_po=float(sa.abs().max()),
        sredni_modul_128_przed=float(sb[:, kan].abs().mean()), sredni_modul_128_po=float(sa[:, kan].abs().mean()),
        udzial_wartosci_powyzej_10_odchylen_po=float((sa.abs() > 10).float().mean()))

    cc = lambda a, b: float((F.normalize(a - mu, dim=-1) * F.normalize(b - mu, dim=-1)).sum())
    konwerter = dict(
        cos_przed_do_prawdziwego=float((pb[0] * y_cls[0]).sum()), cos_wycentr_przed_do_prawdziwego=cc(pb[0], y_cls[0]),
        cos_po_do_przed=float((pa[0] * pb[0]).sum()), cos_wycentr_po_do_przed=cc(pa[0], pb[0]),
        cos_po_do_prawdziwego=float((pa[0] * y_cls[0]).sum()),
        cos_nasz_przepis_do_przed=float((pn[0] * pb[0]).sum()),
        cos_tokenow_po_do_przed=float((ta[0] * tb[0]).sum(-1).mean()))

    # 3. Wyszukiwanie w bazie. Czy samo zdjęcie źródłowe jest w bazie i na której pozycji?
    j_src = next((i for i, w in enumerate(wiersze) if w['film'] == FILM and w['frame'] == KLATKA), None)
    S = {'prawdziwe_DINO': y_cls[0] @ baza.T, 'z_latentu_przed': pb[0] @ baza.T, 'z_latentu_po': pa[0] @ baza.T}
    top = {k: s.topk(TOPK).indices.cpu() for k, s in S.items()}
    gen = torch.Generator().manual_seed(20261006)
    top['losowe'] = torch.randperm(len(baza), generator=gen)[:TOPK]
    wyszukiwanie = dict(
        zrodlo_w_bazie=j_src is not None,
        pozycja_zrodla={k: (int((s > s[j_src]).sum()) + 1 if j_src is not None else None) for k, s in S.items()},
        wspolne_top32_po_i_przed=len(set(top['z_latentu_po'].tolist()) & set(top['z_latentu_przed'].tolist())),
        wspolne_top32_po_i_prawdziwe=len(set(top['z_latentu_po'].tolist()) & set(top['prawdziwe_DINO'].tolist())))

    # 4. Weryfikacja ViT: naturalny latent znalezionych zdjęć kontra sztuczny cel na 128 kanałach + detekcja.
    cel128 = after[:, kan].flatten()
    przed128 = before[:, kan].flatten()

    @torch.no_grad()
    def ocen(idx):
        out = []
        for j in idx.tolist():
            w = wiersze[j]
            rgb = jpeg(w['archive'], w['member'])
            H, W = rgb.shape[:2]
            v, _, _ = przygotuj(rgb)
            o, z = vit(v[None].to(dev).float() / 255)
            vit._latent = None
            z128 = z[0][:, kan].flatten().float()
            bx, sc = dekoduj(o, 1)
            boxes, scores = bx[0], sc[0]
            gt = w['gt'][0]['bbox_xywh'] if w.get('gt') else None
            iou_ = None
            if gt is not None:
                sx, sy = 480 / W, 320 / H
                g = [gt[0] * sx, gt[1] * sy, (gt[0] + gt[2]) * sx, (gt[1] + gt[3]) * sy]
                iou_ = iou(boxes[0].tolist(), g)
            out.append(dict(indeks=j, film=w['film'], klatka=w['frame'],
                            cos128_do_celu=float(F.cosine_similarity(z128, cel128, dim=0)),
                            mse128_do_celu=float(((z128 - cel128) ** 2).mean()),
                            cos128_do_przed=float(F.cosine_similarity(z128, przed128, dim=0)),
                            pewnosc=float(scores[0]), iou=iou_))
        return out

    weryfikacja = {k: ocen(v) for k, v in top.items()}
    podsum = {}
    for k, lista in weryfikacja.items():
        ious = [r['iou'] for r in lista if r['iou'] is not None]
        podsum[k] = dict(cos128_do_celu_srednia=float(np.mean([r['cos128_do_celu'] for r in lista])),
                         cos128_do_celu_maks=float(np.max([r['cos128_do_celu'] for r in lista])),
                         mse128_do_celu_srednia=float(np.mean([r['mse128_do_celu'] for r in lista])),
                         pewnosc_srednia=float(np.mean([r['pewnosc'] for r in lista])),
                         iou_srednia=float(np.mean(ious)) if ious else None,
                         filmow=len({r['film'] for r in lista}))
    # Odniesienie: zdjęcie źródłowe przed zmianą (naturalny forward) względem sztucznego celu.
    odniesienie = dict(cos128_przed_do_celu=float(F.cosine_similarity(przed128, cel128, dim=0)),
                       mse128_przed_do_celu=float(((przed128 - cel128) ** 2).mean()))

    wynik = dict(meta=dict(konwerter_sha256=sha(KONWERTER), latent_sha256=sha(ZRODLO / 'LATENT_PELNY.npz'),
                           vit_sha256=sha(VIT), baza_sha256=sha(BAZA / 'features.npy'), topk=TOPK,
                           zrodlo=f'{FILM} kl. {KLATKA}'),
                 zgodnosc_przepisu_obrazu=zgodnosc_przepisu, odleglosc_od_rozkladu=odleglosc, konwerter=konwerter,
                 wyszukiwanie=wyszukiwanie, odniesienie=odniesienie, podsumowanie_weryfikacji=podsum,
                 weryfikacja=weryfikacja)
    (OUT / 'WYNIKI.json').write_text(json.dumps(wynik, indent=2, ensure_ascii=False), encoding='utf-8')
    rysuj(zrodlo, weryfikacja, wiersze, (ta[0] * tb[0]).sum(-1).cpu().numpy(),
          (after - before).norm(dim=-1).cpu().numpy())
    print(json.dumps({k: wynik[k] for k in ('zgodnosc_przepisu_obrazu', 'odleglosc_od_rozkladu', 'konwerter',
                                            'wyszukiwanie', 'odniesienie', 'podsumowanie_weryfikacji')},
                     indent=2, ensure_ascii=False))


def rysuj(zrodlo, weryfikacja, wiersze, cos_tok, delta_tok):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'
    H, W = zrodlo.shape[:2]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].imshow(zrodlo); ax[0].set_title(f'zdjęcie źródłowe {FILM} kl. {KLATKA}', fontsize=10)
    ax[1].imshow(zrodlo); m = ax[1].imshow(delta_tok.reshape(20, 30), extent=(0, W, H, 0), cmap='magma', alpha=.6)
    ax[1].set_title('siła sztucznej zmiany latentu na tokenach', fontsize=10); fig.colorbar(m, ax=ax[1], fraction=.04)
    ax[2].imshow(zrodlo); m = ax[2].imshow(cos_tok.reshape(20, 30), extent=(0, W, H, 0), cmap='viridis', alpha=.6)
    ax[2].set_title('tokeny DINO z konwertera: cos(po, przed)\nciemne = konwerter zmienił odpowiedź', fontsize=10)
    fig.colorbar(m, ax=ax[2], fraction=.04)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / 'zmiana_latentu_i_odpowiedzi.png', dpi=100); plt.close(fig)

    n = 8
    rzedy = [('z_latentu_po', 'ze sztucznego latentu'), ('z_latentu_przed', 'z latentu przed zmianą'),
             ('prawdziwe_DINO', 'prawdziwe DINO źródła'), ('losowe', 'losowe z bazy')]
    fig, ax = plt.subplots(len(rzedy), n, figsize=(2.6 * n, 2.3 * len(rzedy)))
    for r, (k, nazwa) in enumerate(rzedy):
        for i, w in enumerate(weryfikacja[k][:n]):
            row = wiersze[w['indeks']]
            ax[r, i].imshow(jpeg(row['archive'], row['member']))
            iou_ = f'{w["iou"]:.2f}' if w['iou'] is not None else '-'
            ax[r, i].set_title(f'{w["film"][:7]} kl.{w["klatka"]}\ncos128 {w["cos128_do_celu"]:.2f} IoU {iou_}', fontsize=8)
            ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
        ax[r, 0].set_ylabel(nazwa, fontsize=9)
    fig.suptitle('Pierwsze 8 z 32 znalezionych zdjęć; cos128 = zgodność ich naturalnego latentu ze sztucznym celem na 128 kanałach',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'znalezione_zdjecia.png', dpi=95); plt.close(fig)


if __name__ == '__main__':
    main()
