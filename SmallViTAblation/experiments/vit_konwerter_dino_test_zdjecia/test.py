"""Test konwertera v2 (latent ViT k1000 -> DINOv2) na kilku zdjęciach z filmów VAL.

Dla każdego zdjęcia JPEG (z archiwum adnotacji, ta sama domena co baza cech 35000):
  - zamrożony ViT liczy latent 600x768, konwerter przewiduje CLS 384 i tokeny 20x30x384,
  - zamrożone DINOv2 liczy prawdziwe cechy tym samym przepisem co w uczeniu (dane.przygotuj),
  - porównujemy przewidziane i prawdziwe cechy oraz wyszukiwanie w bazie 35000 klatek TRAIN
    (outputs/vit_dino_faiss_35000/features.npy, przeszukiwanie dokładne iloczynem skalarnym).
Wyniki: outputs/vit_konwerter_dino_test_zdjecia (WYNIKI.json, RAPORT.md, rysunki PNG).
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
from dane import przygotuj  # noqa: E402  ten sam przepis obrazów co w uczeniu
from model import KonwerterGotowy, cechy_dino, latent_vit, wczytaj_dino, wczytaj_vit  # noqa: E402

OUT = ROOT / 'outputs' / HERE.name
KONWERTER = ROOT / 'outputs/vit_konwerter_dino_filmy_eden/v2/KONWERTER.pt'
VIT = ROOT / 'outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt'
FILMY = ROOT / 'outputs/vit_sd_pelny_latent_filmy_eden_20261006/FILMY.json'
BAZA = ROOT / 'outputs/vit_dino_faiss_35000'
DANE = ROOT.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
DINO_REPO = Path('C:/Users/bartl/.cache/torch/hub/facebookresearch_dinov2_main')
DINO_WAGI = Path('C:/Users/bartl/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
LICZBA = 8
TOPK = 5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jpeg(archiwum, nazwa):
    with zipfile.ZipFile(archiwum) as z:
        return np.asarray(Image.open(io.BytesIO(z.read(nazwa))).convert('RGB'))


def wybierz_zdjecia():
    """Po jednym zdjęciu z LICZBA filmów VAL, rozłożonych równo po liście; klatka ze środka filmu."""
    val = [r for r in json.loads(FILMY.read_text(encoding='utf-8')) if r['split'] == 'val']
    val.sort(key=lambda r: r['video_id'])
    idx = np.linspace(0, len(val) - 1, LICZBA).round().astype(int)
    wybor = []
    for i in idx:
        r = val[i]
        nr = int(r['frame_count']) // 2
        film = r['video_id']
        wybor.append(dict(film=film, klatka=nr, archiwum=str(DANE / f'{film}.zip'),
                          nazwa=f'{film}/{film}_{nr:06d}.jpg'))
    return wybor


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # jak w uczeniu
    vit, _ = wczytaj_vit(VIT, dev)
    dino, _ = wczytaj_dino(DINO_REPO, DINO_WAGI, dev)
    konw = KonwerterGotowy(KONWERTER, device=dev)
    mu = F.normalize(torch.load(KONWERTER, map_location='cpu', weights_only=False)['y_mu'].float(), dim=0).to(dev)

    baza = torch.from_numpy(np.load(BAZA / 'features.npy')).to(dev)
    wiersze = [json.loads(l) for l in (BAZA / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(wiersze) == len(baza) == 35000

    zdjecia = wybierz_zdjecia()
    if {z['film'] for z in zdjecia} & {w['film'] for w in wiersze}:
        raise RuntimeError('Film testowy jest w bazie TRAIN')
    rgb = [jpeg(z['archiwum'], z['nazwa']) for z in zdjecia]
    obrazy = [przygotuj(x) for x in rgb]
    b = {k: torch.stack([o[i] for o in obrazy]).to(dev) for i, k in enumerate(('vit', 'cls', 'patch'))}
    with torch.no_grad():
        z = latent_vit(vit, b['vit'])
        y_cls, y_tok = cechy_dino(dino, b['cls'], b['patch'])
        p_cls, p_tok = konw.predict(z)

    cos = (p_cls * y_cls).sum(-1)
    cos_c = (F.normalize(p_cls - mu, dim=-1) * F.normalize(y_cls - mu, dim=-1)).sum(-1)
    cos_tok = (p_tok * y_tok).sum(-1)                      # N x 600
    S_test = (p_cls @ y_cls.T).cpu()                       # przewidziane kontra prawdziwe w zestawie testowym
    S_true, S_pred = y_cls @ baza.T, p_cls @ baza.T        # podobieństwo do bazy 35000
    top_true, top_pred = S_true.topk(TOPK, 1).indices.cpu(), S_pred.topk(TOPK, 1).indices.cpu()
    # Gdzie w kolejności z predykcji ląduje najbliższe zdjęcie według prawdziwego DINO.
    poz_true1 = ((S_pred > S_pred.gather(1, top_true[:, :1].to(dev))).sum(1) + 1).cpu()
    # Jak blisko prawdziwym cechom zapytania jest zdjęcie znalezione predykcją (jakość wyszukiwania).
    cos_znal = S_true.gather(1, top_pred[:, :1].to(dev)).squeeze(1).cpu()
    cos_najl = S_true.gather(1, top_true[:, :1].to(dev)).squeeze(1).cpu()
    los = torch.randint(0, len(baza), (2000,), generator=torch.Generator().manual_seed(0)).to(dev)
    cos_losowe = S_true[:, los].mean(1).cpu()

    wyniki = []
    for i, zd in enumerate(zdjecia):
        def opis(j):
            w = wiersze[int(j)]
            return dict(film=w['film'], klatka=w['frame'], cos_do_prawdziwego=float(S_true[i, int(j)]))
        wyniki.append(dict(
            **{k: zd[k] for k in ('film', 'klatka', 'nazwa')},
            cos_cls=float(cos[i]), cos_cls_wycentrowany=float(cos_c[i]),
            cos_tokenow_srednia=float(cos_tok[i].mean()), cos_tokenow_min=float(cos_tok[i].min()),
            trafienie_w_zestawie=int(S_test[i].argmax()) == i, pozycja_w_zestawie=int((S_test[i] > S_test[i, i]).sum()) + 1,
            pozycja_prawdziwego_top1_w_predykcji=int(poz_true1[i]),
            wspolne_top5=len(set(top_true[i].tolist()) & set(top_pred[i].tolist())),
            cos_znalezionego_top1=float(cos_znal[i]), cos_najlepszego_w_bazie=float(cos_najl[i]),
            cos_losowego_z_bazy=float(cos_losowe[i]),
            top5_prawdziwe=[opis(j) for j in top_true[i]], top5_predykcja=[opis(j) for j in top_pred[i]]))

    podsumowanie = dict(
        zdjec=len(zdjecia), cos_cls_srednia=float(cos.mean()), cos_cls_wycentrowany_srednia=float(cos_c.mean()),
        cos_tokenow_srednia=float(cos_tok.mean()),
        trafienia_w_zestawie=f'{sum(w["trafienie_w_zestawie"] for w in wyniki)}/{len(wyniki)}',
        mediana_pozycji_prawdziwego_top1=float(poz_true1.float().median()),
        wspolne_top5_srednio=float(np.mean([w['wspolne_top5'] for w in wyniki])),
        cos_znalezionego_top1_srednia=float(cos_znal.mean()), cos_najlepszego_srednia=float(cos_najl.mean()),
        cos_losowego_srednia=float(cos_losowe.mean()))
    meta = dict(konwerter=str(KONWERTER), konwerter_sha256=sha(KONWERTER), vit_sha256=sha(VIT),
                kod_konwertera_sha256={n: sha(KONW / n) for n in ('dane.py', 'model.py', 'model_vit.py')},
                baza=str(BAZA / 'features.npy'), baza_sha256=sha(BAZA / 'features.npy'),
                zrodlo_zdjec='JPEG z archiwów adnotacji, filmy VAL, klatka ze środka filmu')
    (OUT / 'WYNIKI.json').write_text(json.dumps(dict(meta=meta, podsumowanie=podsumowanie, zdjecia=wyniki),
                                                indent=2, ensure_ascii=False), encoding='utf-8')
    np.save(OUT / 'MACIERZ_ZESTAWU.npy', S_test.numpy())
    rysuj(rgb, zdjecia, wyniki, cos_tok.cpu().numpy(), top_true, top_pred, wiersze, S_test.numpy())
    print(json.dumps(podsumowanie, indent=2, ensure_ascii=False))


def rysuj(rgb, zdjecia, wyniki, cos_tok, top_true, top_pred, wiersze, S_test):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'DejaVu Sans'
    cache = {}

    def baza_obraz(j):
        w = wiersze[int(j)]
        if j not in cache:
            cache[j] = jpeg(w['archive'], w['member'])
        return cache[j]

    for i, zd in enumerate(zdjecia):
        fig, ax = plt.subplots(2, TOPK + 1, figsize=(3.3 * (TOPK + 1), 5.6))
        ax[0, 0].imshow(rgb[i]); ax[0, 0].set_title(f'ZAPYTANIE (film VAL)\n{zd["film"]} kl. {zd["klatka"]}', fontsize=10)
        ax[1, 0].imshow(rgb[i])
        h, w = rgb[i].shape[:2]
        ax[1, 0].imshow(cos_tok[i].reshape(20, 30), extent=(0, w, h, 0), cmap='magma', vmin=0, vmax=1, alpha=.55)
        ax[1, 0].set_title(f'zgodność tokenów 20x30 (jasne = dobra)\nśredni cosinus {cos_tok[i].mean():.2f}', fontsize=10)
        for wiersz, (top, nazwa) in enumerate(((top_true, 'prawdziwe DINO'), (top_pred, 'z latentu ViT'))):
            for k in range(TOPK):
                j = int(top[i, k]); a = ax[wiersz, k + 1]
                a.imshow(baza_obraz(j))
                a.set_title(f'{nazwa} #{k + 1}\n{wiersze[j]["film"]} kl. {wiersze[j]["frame"]}', fontsize=9)
        for a in ax.flat:
            a.set_xticks([]); a.set_yticks([])
        r = wyniki[i]
        fig.text(.01, .01, f'Górny wiersz: najbliższe zdjęcia w bazie 35000 TRAIN według prawdziwych cech DINO. '
                 f'Dolny: według cech przewidzianych z latentu ViT.\n'
                 f'cos CLS {r["cos_cls"]:.3f}, po odjęciu średniej {r["cos_cls_wycentrowany"]:.3f} | '
                 f'wspólne top5: {r["wspolne_top5"]}/5 | prawdziwe top1 na pozycji {r["pozycja_prawdziwego_top1_w_predykcji"]} '
                 f'| podobieństwo znalezionego top1 do zapytania {r["cos_znalezionego_top1"]:.3f} '
                 f'(najlepsze w bazie {r["cos_najlepszego_w_bazie"]:.3f}, losowe {r["cos_losowego_z_bazy"]:.3f})', fontsize=10)
        fig.tight_layout(rect=(0, .07, 1, 1))
        fig.savefig(OUT / f'zdjecie_{i + 1:02d}_{zd["film"]}.png', dpi=100)
        plt.close(fig)

    fig, a = plt.subplots(figsize=(6.5, 5.5))
    m = a.imshow(S_test, cmap='viridis')
    a.set_xticks(range(len(zdjecia)), [z['film'][:7] for z in zdjecia], rotation=60, fontsize=8)
    a.set_yticks(range(len(zdjecia)), [z['film'][:7] for z in zdjecia], fontsize=8)
    a.set_xlabel('prawdziwe cechy DINO'); a.set_ylabel('cechy przewidziane z latentu ViT')
    a.set_title('Podobieństwo przewidzianych i prawdziwych CLS\n(trafienie = maksimum na przekątnej)', fontsize=10)
    fig.colorbar(m); fig.tight_layout(); fig.savefig(OUT / 'macierz_zestawu.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
