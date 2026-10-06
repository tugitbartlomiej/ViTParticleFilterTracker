"""Uczenie konwertera latent ViT -> cechy DINOv2 na strumieniu filmów, DDP na kilku GPU.

Na każdej paczce, w locie na GPU: zamrożony ViT liczy naturalny latent 600x768, zamrożone DINOv2 liczy
cele (CLS 384 wg DINO_SPEC i tokeny 20x30x384), konwerter uczy się mapowania. Klatki, latenty i cechy
nie są zapisywane na dysk; zapisujemy tylko wagi konwertera, optymalizator, logi i metryki.

torchrun --standalone --nproc_per_node=4 run.py --config config.json --manifest FILMY.json \
    --vit-checkpoint MODEL_VIT_K1000.pt --dino-repo <hub/facebookresearch_dinov2_*> \
    --dino-weights dinov2_vits14_pretrain.pth --output wyniki/trening
"""
import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dane import StrumienUczacy, ZbiorOceny, wczytaj_manifest, zloz  # noqa: E402
from model import (Konwerter, KontrolaLiniowa, cechy_dino, latent_vit, sha256,  # noqa: E402
                   wczytaj_dino, wczytaj_vit)

DOMYSLNE = dict(steps=20000, batch_size=64, workers=3, przeplot=4, bufor=256, hflip=0.0, crop_min=1.0, jitter=0.0,
                dropout=0.0, drop_path=0.0, maska=0.0,
                lr=3e-4, lr_kontrola=1e-3, wd=0.05, warmup=500, dim=384, glebokosc=6, glowy=6,
                lambda_patch=0.5, lambda_nce=0.2, tau=0.07, okno_nce=30, stats_batches=40,
                eval_every=2000, eval_krok=15, eval_max_filmow=None, eval_workers=2,
                save_every=1000, log_every=20, max_hours=11.5, seed=20261006, tf32=True,
                resume='auto', smoke=False, zamiana=None, pozwol_inny_kod_dino=False)

STOP = {'flaga': False}


def argumenty():
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    for k in ('manifest', 'vit_checkpoint', 'dino_repo', 'dino_weights', 'output'):
        p.add_argument('--' + k.replace('_', '-'), required=True)
    for k, v in DOMYSLNE.items():
        if isinstance(v, bool):
            p.add_argument('--' + k.replace('_', '-'), action=argparse.BooleanOptionalAction, default=None)
        else:
            p.add_argument('--' + k.replace('_', '-'), type=type(v) if v is not None else str, default=None)
    a = vars(p.parse_args())
    cfg = dict(DOMYSLNE)
    if a['config']:
        cfg.update(json.loads(Path(a['config']).read_text(encoding='utf-8')))
    cfg.update({k: v for k, v in a.items() if v is not None and k != 'config'})
    if cfg['eval_max_filmow'] is not None:
        cfg['eval_max_filmow'] = int(cfg['eval_max_filmow'])
    return cfg


def log0(rank, *a):
    if rank == 0:
        print(time.strftime('%H:%M:%S'), *a, flush=True)


def harmonogram(krok, cfg):
    if krok < cfg['warmup']:
        return (krok + 1) / cfg['warmup']
    t = (krok - cfg['warmup']) / max(1, cfg['steps'] - cfg['warmup'])
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def info_nce(pred, y_all, film_all, kl_all, film, kl, offset, tau, okno):
    """pred: lokalne B x 384 (z gradientem); y_all: cele ze wszystkich GPU. Sąsiednie klatki tego samego
    filmu (|dk| < okno) wypadają z negatywów, bo są prawie tym samym obrazem."""
    logits = pred @ y_all.T / tau
    idx = torch.arange(len(pred), device=pred.device) + offset
    blisko = (film[:, None] == film_all[None]) & ((kl[:, None] - kl_all[None]).abs() < okno)
    blisko[torch.arange(len(pred)), idx] = False
    return F.cross_entropy(logits.masked_fill(blisko, float('-inf')), idx)


def zbierz(t, world):
    if world == 1:
        return t
    out = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out)


def na_gpu(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def cele_i_latent(vit, dino, b):
    z = latent_vit(vit, b['vit'])
    y_cls, y_patch = cechy_dino(dino, b['cls'], b['patch'])
    return z, y_cls, y_patch


def statystyki(it, vit, dino, n, dev, world):
    """Stała standaryzacja kanałów latentu i średni CLS DINO, liczone na pierwszych n paczkach TRAIN."""
    s1 = torch.zeros(768, dtype=torch.float64, device=dev)
    s2 = torch.zeros_like(s1)
    sy = torch.zeros(384, dtype=torch.float64, device=dev)
    cnt = torch.zeros(2, dtype=torch.float64, device=dev)
    for _ in range(n):
        z, y, _ = cele_i_latent(vit, dino, na_gpu(next(it), dev))
        zz = z.double().reshape(-1, 768)
        s1 += zz.sum(0); s2 += (zz ** 2).sum(0); sy += y.double().sum(0)
        cnt += torch.tensor([zz.shape[0], y.shape[0]], dtype=torch.float64, device=dev)
    if world > 1:
        for t in (s1, s2, sy, cnt):
            dist.all_reduce(t)
    mean = s1 / cnt[0]
    std = (s2 / cnt[0] - mean ** 2).clamp_min(0).sqrt()
    martwe = int((std < 1e-3).sum())
    return mean.float(), std.clamp_min(1e-3).float(), sy / cnt[1], martwe


@torch.no_grad()
def zbuduj_ocene(rows, cfg, rank, world, vit, dino, dev):
    """Zbiór VAL liczony raz i trzymany w pamięci GPU (nie na dysku): latent bf16 + cele."""
    ds = ZbiorOceny(rows, rank, world, cfg['eval_krok'], cfg['eval_max_filmow'])
    nw = min(cfg['eval_workers'], len(ds.rows))
    zbior = dict(z=[], y=[], yp=[], film=[], kl=[])
    if ds.rows:
        dl = DataLoader(ds, batch_size=cfg['batch_size'], num_workers=nw, collate_fn=zloz, pin_memory=True)
        for b in dl:
            z, y, yp = cele_i_latent(vit, dino, na_gpu(b, dev))
            zbior['z'].append(z.bfloat16()); zbior['y'].append(y); zbior['yp'].append(yp.bfloat16())
            zbior['film'].append(b['film']); zbior['kl'].append(b['klatka'])
    return zbior


@torch.no_grad()
def ocen(model, kontrola, zbior, y_mu, cfg, rank, world, dev):
    model.eval()
    p, pk, pc = [], [], []
    for z, yp in zip(zbior['z'], zbior['yp']):
        with torch.autocast('cuda', torch.bfloat16, enabled=dev.type == 'cuda'):
            zc, zp = model(z.float())
            zk = kontrola(z.float(), model.x_mean, model.x_std)
        p.append(zc.cpu()); pk.append(zk.cpu())
        pc.append(F.cosine_similarity(zp, yp.float(), dim=-1).mean(1).cpu())
    lok = dict(p=p, pk=pk, pc=pc, y=[t.cpu() for t in zbior['y']], film=zbior['film'], kl=zbior['kl'])
    lok = {k: torch.cat(v) if v else torch.empty(0) for k, v in lok.items()}
    if world > 1:
        wszystkie = [None] * world
        dist.all_gather_object(wszystkie, lok)
    else:
        wszystkie = [lok]
    model.train()
    if rank != 0:
        return None
    a = {k: torch.cat([w[k] for w in wszystkie if w[k].numel()]) for k in lok}
    y, film, kl, mu = a['y'], a['film'], a['kl'], y_mu.cpu()
    wynik = dict(n=len(y), filmow=int(film.unique().numel()), krok_klatek=cfg['eval_krok'])
    srednia = mu[None].expand_as(y)
    for nazwa, pred in (('konwerter', a['p']), ('kontrola_liniowa', a['pk']), ('srednia_train', srednia)):
        cos = (pred * y).sum(-1)
        r = dict(cos=float(cos.mean()))
        if nazwa != 'srednia_train':
            cc = (F.normalize(pred - mu, dim=-1) * F.normalize(y - mu, dim=-1)).sum(-1)
            S = pred @ y.T
            rank_ = (S > S.diagonal()[:, None]).sum(1) + 1
            j = S.argmax(1)
            okno = (film[j] == film) & ((kl[j] - kl).abs() <= cfg['eval_krok'])
            per_film = [float(cc[film == f].median()) for f in film.unique()]
            r.update(cos_wycentrowany=float(cc.mean()), top1=float((rank_ == 1).float().mean()),
                     top5=float((rank_ <= 5).float().mean()), mediana_pozycji=float(rank_.float().median()),
                     top1_sasiad=float(okno.float().mean()), top1_film=float((film[j] == film).float().mean()),
                     mediana_filmow_cos_wycentrowany=float(np.median(per_film)))
        wynik[nazwa] = r
    wynik['konwerter']['cos_tokenow'] = float(a['pc'].mean())
    # Odniesienie: ile wynosi top1_film przy losowym wyborze (udział filmu w galerii).
    _, licz = film.unique(return_counts=True)
    wynik['top1_film_losowo'] = float(((licz.float() / len(film)) ** 2).sum())
    return wynik


def zapisz_atomowo(obj, sciezka):
    tmp = sciezka.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, sciezka)


def main():
    cfg = argumenty()
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    lokal = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        dist.init_process_group('nccl')
    dev = torch.device('cuda', lokal) if torch.cuda.is_available() else torch.device('cpu')
    if dev.type == 'cuda':
        torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = cfg['tf32']
    torch.backends.cudnn.allow_tf32 = cfg['tf32']
    torch.manual_seed(cfg['seed'] + rank)
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    for s in (signal.SIGTERM, signal.SIGINT) + ((signal.SIGUSR1,) if hasattr(signal, 'SIGUSR1') else ()):
        signal.signal(s, lambda *_: STOP.update(flaga=True))
    start = time.time()

    zamiana = tuple(cfg['zamiana'].split('=', 1)) if cfg['zamiana'] else None
    rows = wczytaj_manifest(cfg['manifest'], zamiana)
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    if not val or {r['video_id'] for r in train} & {r['video_id'] for r in val}:
        raise ValueError('Potrzebne rozłączne filmy TRAIN i VAL')

    vit, vit_krok = wczytaj_vit(cfg['vit_checkpoint'], dev)
    dino, dino_kod = wczytaj_dino(cfg['dino_repo'], cfg['dino_weights'], dev, cfg['pozwol_inny_kod_dino'])
    model = Konwerter(cfg['dim'], cfg['glebokosc'], cfg['glowy'], dropout=cfg['dropout'],
                      drop_path=cfg['drop_path'], maska=cfg['maska']).to(dev)
    kontrola = KontrolaLiniowa().to(dev)
    opt = torch.optim.AdamW([dict(params=model.parameters(), lr=cfg['lr']),
                             dict(params=kontrola.parameters(), lr=cfg['lr_kontrola'])], weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: harmonogram(k, cfg))
    y_mu = torch.zeros(384, device=dev)

    ckpts = sorted(out.glob('ckpt_*.pt'))
    krok = 0
    if ckpts and cfg['resume'] == 'auto' and not cfg['smoke']:
        ck = torch.load(ckpts[-1], map_location=dev, weights_only=False)
        model.load_state_dict(ck['model']); kontrola.load_state_dict(ck['kontrola'])
        opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        y_mu = ck['y_mu'].to(dev); krok = ck['krok']
        log0(rank, f'wznowienie z {ckpts[-1].name}, krok {krok}')

    aug = dict(hflip=cfg['hflip'], crop_min=cfg['crop_min'], jitter=cfg['jitter'])
    ds = StrumienUczacy(rows, rank, world, cfg['seed'] + krok, cfg['przeplot'], cfg['bufor'], aug)
    dl = DataLoader(ds, batch_size=cfg['batch_size'], num_workers=cfg['workers'], collate_fn=zloz,
                    pin_memory=True, persistent_workers=cfg['workers'] > 0,
                    prefetch_factor=4 if cfg['workers'] > 0 else None)
    it = iter(dl)

    meta = dict(cfg=cfg, world=world, vit_checkpoint_sha256=sha256(cfg['vit_checkpoint']), vit_krok=vit_krok,
                dino_kod_sha256=dino_kod, filmy_train=len(train), filmy_val=len(val),
                klatki_train=sum(int(r['frame_count']) for r in train),
                parametry_konwertera=sum(p.numel() for p in model.parameters()),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu')
    smoke = {}

    if krok == 0:
        t0 = time.time()
        if cfg['smoke']:
            b = na_gpu(next(it), dev)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            z32, y32, p32 = cele_i_latent(vit, dino, b)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
            z_tf, y_tf, p_tf = cele_i_latent(vit, dino, b)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = cfg['tf32']
            smoke['tf32_kontra_fp32'] = dict(
                cls_cos_min=float((y32 * y_tf).sum(-1).min()), tokeny_cos_min=float((p32 * p_tf).sum(-1).min()),
                latent_blad_wzgledny=float((z32 - z_tf).norm() / z32.norm()))
            smoke['ksztalty'] = dict(vit=list(b['vit'].shape), cls=list(b['cls'].shape), patch=list(b['patch'].shape),
                                     latent=list(z32.shape), y_cls=list(y32.shape), y_patch=list(p32.shape))
        x_mean, x_std, y_sr, martwe = statystyki(it, vit, dino, cfg['stats_batches'], dev, world)
        y_mu = F.normalize(y_sr, dim=0).float()
        model.x_mean.copy_(x_mean); model.x_std.copy_(x_std)
        meta['statystyki'] = dict(paczek=cfg['stats_batches'], martwe_kanaly=martwe, sekund=time.time() - t0,
                                  norma_sredniego_cls=float(y_sr.norm()))
        log0(rank, f'statystyki latentu: {cfg["stats_batches"]} paczek, martwe kanały {martwe}, {time.time() - t0:.0f} s')

    if rank == 0:
        (out / ('SMOKE_CONFIG.json' if cfg['smoke'] else 'CONFIG.json')).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lokal]) if world > 1 else model
    ddp_k = torch.nn.parallel.DistributedDataParallel(kontrola, device_ids=[lokal]) if world > 1 else kontrola

    log0(rank, 'buduję zbiór VAL w pamięci GPU')
    t0 = time.time()
    zbior = zbuduj_ocene(rows, cfg, rank, world, vit, dino, dev)
    log0(rank, f'VAL gotowy w {time.time() - t0:.0f} s')

    def zapisz(krok):
        if rank != 0:
            return
        sciezka = out / f'ckpt_{krok:08d}.pt'
        zapisz_atomowo(dict(model=model.state_dict(), kontrola=kontrola.state_dict(), opt=opt.state_dict(),
                            sched=sched.state_dict(), y_mu=y_mu.cpu(), krok=krok, konf=model.konf, cfg=cfg), sciezka)
        for stary in sorted(out.glob('ckpt_*.pt'))[:-2]:
            stary.unlink()
        (out / 'LATEST.json').write_text(json.dumps(dict(krok=krok, plik=sciezka.name,
                                                         bajty=sciezka.stat().st_size)), encoding='utf-8')

    historia = (out / 'HISTORIA.jsonl').open('a', encoding='utf-8') if rank == 0 else None
    model.train(); kontrola.train()
    t_log, n_log, t_dane = time.time(), 0, 0.0
    pierwszy_krok = krok
    while krok < cfg['steps']:
        td = time.time()
        b = na_gpu(next(it), dev)
        t_dane += time.time() - td
        z, y, yp = cele_i_latent(vit, dino, b)
        with torch.autocast('cuda', torch.bfloat16, enabled=dev.type == 'cuda'):
            pc, pp = ddp(z)
            pk = ddp_k(z, model.x_mean, model.x_std)
        y_all = zbierz(y, world)
        f_all, k_all = zbierz(b['film'], world), zbierz(b['klatka'], world)
        off = rank * len(y)
        l_cos = 1 - (pc * y).sum(-1).mean()
        l_patch = 1 - F.cosine_similarity(pp, yp, dim=-1).mean()
        l_nce = info_nce(pc, y_all, f_all, k_all, b['film'], b['klatka'], off, cfg['tau'], cfg['okno_nce'])
        l_k = 1 - (pk * y).sum(-1).mean() + cfg['lambda_nce'] * info_nce(
            pk, y_all, f_all, k_all, b['film'], b['klatka'], off, cfg['tau'], cfg['okno_nce'])
        strata = l_cos + cfg['lambda_patch'] * l_patch + cfg['lambda_nce'] * l_nce
        opt.zero_grad(set_to_none=True)
        (strata + l_k).backward()
        if cfg['smoke'] and krok == pierwszy_krok:
            g = [p.grad for p in model.parameters() if p.requires_grad]
            smoke['gradient'] = dict(
                konwerter_skonczony=all(bool(torch.isfinite(x).all()) for x in g if x is not None),
                konwerter_norma=float(torch.norm(torch.stack([x.norm() for x in g if x is not None]))),
                bez_gradu=sum(x is None for x in g),
                vit_dino_bez_gradientu=all(p.grad is None for p in list(vit.parameters()) + list(dino.parameters())))
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(kontrola.parameters()), 1.0)
        opt.step(); sched.step()
        krok += 1; n_log += len(y) * world

        if krok % cfg['log_every'] == 0 or krok == cfg['steps']:
            if not math.isfinite(float(strata.detach())):
                raise FloatingPointError(f'Strata nieskończona w kroku {krok}')
            dt = time.time() - t_log
            wpis = dict(krok=krok, strata=float(strata), cos=1 - float(l_cos), cos_tokenow=1 - float(l_patch),
                        nce=float(l_nce), kontrola=float(l_k), lr=sched.get_last_lr()[0],
                        klatek_na_s=n_log / dt, udzial_czekania_na_dane=t_dane / dt, godzin=(time.time() - start) / 3600)
            if historia:
                historia.write(json.dumps(wpis) + '\n'); historia.flush()
            log0(rank, ' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in wpis.items()))
            t_log, n_log, t_dane = time.time(), 0, 0.0
            if (out / 'STOP').exists() or (time.time() - start) / 3600 > cfg['max_hours']:
                STOP['flaga'] = True
        if world > 1:
            f = torch.tensor([float(STOP['flaga'])], device=dev)
            dist.all_reduce(f, op=dist.ReduceOp.MAX)
            STOP['flaga'] = bool(f.item())

        if krok % cfg['eval_every'] == 0 or krok == cfg['steps'] or STOP['flaga']:
            w = ocen(model, kontrola, zbior, y_mu, cfg, rank, world, dev)
            if w:
                w['krok'] = krok
                with (out / 'OCENA.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(w) + '\n')
                log0(rank, 'VAL', json.dumps({k: w[k] for k in ('konwerter', 'kontrola_liniowa')}))
        if krok % cfg['save_every'] == 0 or krok == cfg['steps'] or STOP['flaga']:
            zapisz(krok)
        if STOP['flaga']:
            log0(rank, f'zatrzymanie w kroku {krok}')
            break

    if rank == 0:
        ostatnia = [json.loads(x) for x in (out / 'OCENA.jsonl').read_text(encoding='utf-8').splitlines()][-1]
        meta['ocena_koncowa'] = ostatnia
        meta['krok_koncowy'] = krok
        meta['godzin'] = (time.time() - start) / 3600
        if krok == cfg['steps']:
            zapisz_atomowo(dict(model=model.state_dict(), konf=model.konf, y_mu=y_mu.cpu(), meta=meta),
                           out / 'KONWERTER.pt')
        if cfg['smoke']:
            ck = torch.load(sorted(out.glob('ckpt_*.pt'))[-1], map_location=dev, weights_only=False)
            m2 = Konwerter(**ck['konf']).to(dev).eval(); m2.load_state_dict(ck['model'])
            model.eval()
            z0 = zbior['z'][0][:8].float() if zbior['z'] else z[:8]
            with torch.no_grad():
                a1, b1 = model(z0); a2, b2 = m2(z0)
            smoke['odczyt_checkpointu_identyczny'] = bool(torch.equal(a1, a2) and torch.equal(b1, b2))
            h = [json.loads(x) for x in (out / 'HISTORIA.jsonl').read_text(encoding='utf-8').splitlines()]
            smoke['klatek_na_s'] = h[-1]['klatek_na_s']
            smoke['udzial_czekania_na_dane'] = h[-1]['udzial_czekania_na_dane']
            smoke['ocena'] = ostatnia
            t = smoke['tf32_kontra_fp32']; g = smoke['gradient']
            smoke['passed'] = bool(g['konwerter_skonczony'] and g['konwerter_norma'] > 0 and g['bez_gradu'] == 0
                                   and g['vit_dino_bez_gradientu'] and smoke['odczyt_checkpointu_identyczny']
                                   and t['cls_cos_min'] > 0.999 and t['tokeny_cos_min'] > 0.99)
            (out / 'SMOKE.json').write_text(json.dumps(smoke, indent=2, ensure_ascii=False), encoding='utf-8')
            log0(rank, 'SMOKE', json.dumps(smoke, ensure_ascii=False))
        (out / ('SMOKE_CONFIG.json' if cfg['smoke'] else 'CONFIG.json')).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    os._exit(0)  # procesy ładujące z nieskończonym strumieniem nie kończą się same


if __name__ == '__main__':
    main()
