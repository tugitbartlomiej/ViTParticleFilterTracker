#!/usr/bin/env python3
"""Dostrajanie Stable Diffusion 1.5 na latentach klatek Cataract-LMM z maskami układu sceny, na wielu kartach (PLAN.md).

UNet SD 1.5 (860 mln parametrów) dostaje na wejściu zaszumiony latent (4 kanały) i trzy maski w rozdzielczości latentu
(narzędzie, rogówka, źrenica); pierwsza warstwa rozszerzona z 4 do 7 kanałów, nowe wagi od zera. W `--bez-maski`
przykładów maski są zerowane (model umie też rysować bez warunku). Tekst: stałe osadzenie pustego napisu.
Cel: przewidywany szum (jak SD 1.5), przesunięcie szumu `--offset`.

Rozproszenie: torchrun, DDP; stała paczka globalna `--paczka` = `--na-karte` × liczba kart × akumulacja, więc wynik nie
zależy od liczby kart. Zapis `ostatni.pt` co `--zapis-co` kroków (atomowo), migawki EMA w formacie diffusers co
`--ema-co`, podgląd co `--podglad-co`. Plik `STOP` w katalogu wyjścia (tworzy go skrypt SLURM przed końcem czasu)
kończy uczenie po bieżącym kroku z zapisem; kolejne uruchomienie z tym samym `--out` wznawia.

    torchrun --standalone --nproc_per_node=8 trenuj_sd.py --dane D --podglad-dane V --sd SD --vae VAE --out O
    python trenuj_sd.py --maly ...                      (próba lokalna: mały UNet z losowymi wagami)
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import wspolne as W

SD_SZUM = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')


class Klatki(torch.utils.data.Dataset):
    """Latenty i maski wszystkich przygotowanych filmów; pliki otwierane leniwie (każdy proces ładujący osobno)."""

    def __init__(self, katalog, odbicie=True):
        self.k = Path(katalog)
        self.filmy = sorted(p.stem for p in (self.k / 'indeks').glob('*.json') if not p.stem.endswith('.tmp'))
        dl = [json.loads((self.k / 'indeks' / f'{f}.json').read_text(encoding='utf-8'))['klatki'] for f in self.filmy]
        self.od = np.concatenate([[0], np.cumsum([len(x) for x in dl])])
        self.odbicie = odbicie
        self._L = self._M = None

    def __len__(self):
        return int(self.od[-1])

    def _otworz(self):
        self._L = [np.load(self.k / 'latenty' / f'{f}.npy', mmap_mode='r') for f in self.filmy]
        self._M = [np.load(self.k / 'maski' / f'{f}.npy', mmap_mode='r') for f in self.filmy]

    def __getitem__(self, i):
        if self._L is None:
            self._otworz()
        f = int(np.searchsorted(self.od, i, side='right') - 1)
        j = int(i - self.od[f])
        lat = torch.from_numpy(np.array(self._L[f][j], np.float32))
        mas = torch.from_numpy(np.array(self._M[f][j], np.float32) / 255.0)
        if self.odbicie and torch.rand(()) < 0.5:                        # odbicie poziome latentu i masek razem
            lat, mas = lat.flip(-1), mas.flip(-1)
        return lat, mas


def zbuduj_unet(a):
    from diffusers import UNet2DConditionModel
    if a.maly:
        unet = UNet2DConditionModel(sample_size=(W.LWYS, W.LSZER), in_channels=4, out_channels=4, layers_per_block=1,
                                    block_out_channels=(32, 64, 64, 64), cross_attention_dim=768, attention_head_dim=8,
                                    down_block_types=('CrossAttnDownBlock2D',) * 3 + ('DownBlock2D',),
                                    up_block_types=('UpBlock2D',) + ('CrossAttnUpBlock2D',) * 3)
    else:
        unet = UNet2DConditionModel.from_pretrained(a.sd, subfolder='unet', variant=a.wariant, torch_dtype=torch.float32)
    stary = unet.conv_in
    nowy = torch.nn.Conv2d(4 + len(W.OBIEKTY), stary.out_channels, stary.kernel_size, stary.stride, stary.padding)
    with torch.no_grad():
        nowy.weight.zero_()
        nowy.weight[:, :4] = stary.weight
        nowy.bias.copy_(stary.bias)
    unet.conv_in = nowy
    unet.register_to_config(in_channels=4 + len(W.OBIEKTY))
    return unet


@torch.no_grad()
def pusty_tekst(a, dev):
    if a.maly:
        return torch.zeros(1, 77, 768, device=dev)
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(a.sd, subfolder='tokenizer')
    te = CLIPTextModel.from_pretrained(a.sd, subfolder='text_encoder', variant=a.wariant).to(dev).eval()
    ids = tok([''], padding='max_length', max_length=tok.model_max_length, return_tensors='pt').input_ids.to(dev)
    e = te(ids)[0].float()
    del te
    return e


def maski_podgladu(katalog, n=16):
    """n klatek rozłożonych po filmach z `katalog` (filmy val): latenty i maski."""
    ds = Klatki(katalog, odbicie=False)
    idx = np.linspace(0, len(ds) - 1, n).round().astype(int)
    L, M = zip(*(ds[int(i)] for i in idx))
    return torch.stack(L), torch.stack(M)


@torch.no_grad()
def probkuj(unet, maski, emb, dev, kroki=50, cfg=1.0, ziarno=1234):
    """DDIM od czystego szumu; cfg > 1 wzmacnia warunek masek (bez warunku = maski zerowe)."""
    from diffusers import DDIMScheduler
    s = DDIMScheduler(**SD_SZUM, clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    s.set_timesteps(kroki)
    n = len(maski)
    g = torch.Generator(device=dev).manual_seed(ziarno)
    x = torch.randn((n, 4, W.LWYS, W.LSZER), generator=g, device=dev)
    e_ = emb.expand(n, -1, -1)
    for t in s.timesteps:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.startswith('cuda')):
            e = unet(torch.cat([x, maski], 1), t, encoder_hidden_states=e_).sample.float()
            if cfg != 1.0:
                e0 = unet(torch.cat([x, torch.zeros_like(maski)], 1), t, encoder_hidden_states=e_).sample.float()
                e = e0 + cfg * (e - e0)
        x = s.step(e, t, x).prev_sample
    return x


@torch.no_grad()
def zapisz_podglad(vae, lat_prawdziwe, lat_gen, maski, sciezka):
    """Dwa rzędy na przykład: klatka prawdziwa po koderze i dekoderze VAE, obraz z modelu; kontur masek na obu."""
    import cv2
    def obraz(lat):
        x = vae.decode(lat.to(vae.device, vae.dtype) / W.SKALA_VAE).sample.float().clamp(-1, 1)
        return ((x + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    P, G = obraz(lat_prawdziwe), obraz(lat_gen)
    kolory = ((255, 40, 40), (40, 255, 40), (60, 120, 255))
    kafle = []
    for p, g, m in zip(P, G, maski.cpu().numpy()):
        para = []
        for im in (p, g):
            im = np.ascontiguousarray(im)
            for k in range(m.shape[0]):
                mk = cv2.resize((m[k] > 0.5).astype(np.uint8), (W.SZER, W.WYS), interpolation=cv2.INTER_NEAREST)
                kont, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, kont, -1, kolory[k], 1)
            para.append(im)
        kafle.append(np.concatenate(para, 0))
    wiersze = [np.concatenate(kafle[i:i + 8], 1) for i in range(0, len(kafle), 8)]
    siatka = np.concatenate(wiersze, 0)
    cv2.imwrite(str(sciezka), cv2.cvtColor(siatka, cv2.COLOR_RGB2BGR))


def zapisz_atomowo(obj, sciezka):
    tmp = sciezka.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, sciezka)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dane', required=True)
    ap.add_argument('--podglad-dane', default=None)
    ap.add_argument('--sd', default=None, help='katalog Stable Diffusion 1.5 (diffusers)')
    ap.add_argument('--vae', default=None, help='katalog VAE do podglądu')
    ap.add_argument('--wariant', default='fp16')
    ap.add_argument('--out', required=True)
    ap.add_argument('--kroki', type=int, default=20000)
    ap.add_argument('--paczka', type=int, default=256, help='paczka globalna')
    ap.add_argument('--na-karte', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--rozgrzewka', type=int, default=500)
    ap.add_argument('--ema', type=float, default=0.9999)
    ap.add_argument('--offset', type=float, default=0.05)
    ap.add_argument('--bez-maski', type=float, default=0.2)
    ap.add_argument('--checkpointing', action='store_true')
    ap.add_argument('--workery', type=int, default=4)
    ap.add_argument('--zapis-co', type=int, default=1000)
    ap.add_argument('--ema-co', type=int, default=5000)
    ap.add_argument('--podglad-co', type=int, default=2000)
    ap.add_argument('--log-co', type=int, default=50)
    ap.add_argument('--ziarno', type=int, default=0)
    ap.add_argument('--maly', action='store_true')
    a = ap.parse_args()

    rank, world, lrank = W.rozproszenie()
    rozpr = 'RANK' in os.environ
    cuda = torch.cuda.is_available()
    dev = f'cuda:{lrank}' if cuda else 'cpu'
    if cuda:
        torch.cuda.set_device(dev)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if rozpr:
        dist.init_process_group('nccl' if cuda else 'gloo')
    torch.manual_seed(a.ziarno * 1000 + rank)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    akum = a.paczka // (a.na_karte * world)
    assert akum >= 1 and akum * a.na_karte * world == a.paczka, (a.paczka, a.na_karte, world)

    dane = Klatki(a.dane)
    sampler = torch.utils.data.distributed.DistributedSampler(dane, num_replicas=world, rank=rank, shuffle=True,
                                                              seed=a.ziarno, drop_last=True)
    ladowanie = torch.utils.data.DataLoader(dane, batch_size=a.na_karte, sampler=sampler, num_workers=a.workery,
                                            pin_memory=cuda, drop_last=True, persistent_workers=a.workery > 0)
    unet = zbuduj_unet(a).to(dev)
    if a.checkpointing:
        unet.enable_gradient_checkpointing()
    emb = pusty_tekst(a, dev)
    from diffusers import DDPMScheduler
    harm = DDPMScheduler(**SD_SZUM, clip_sample=False, prediction_type='epsilon')
    opt = torch.optim.AdamW(unet.parameters(), lr=a.lr, weight_decay=1e-2)
    ema = copy.deepcopy(unet).requires_grad_(False).eval() if rank == 0 else None
    krok = epoka = 0
    sciezka = out / 'ostatni.pt'
    if sciezka.exists():
        ck = torch.load(sciezka, map_location='cpu', weights_only=False, mmap=True)
        unet.load_state_dict(ck['unet']); opt.load_state_dict(ck['opt'])
        krok, epoka = ck['krok'], ck['epoka']
        if ema is not None:
            ema.load_state_dict(ck['ema'])
        del ck
        print(f'[{rank}] wznowienie od kroku {krok}', flush=True)
    model = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[lrank] if cuda else None) if rozpr else unet
    npar = sum(p.numel() for p in unet.parameters())
    vae = pl_lat = pl_mas = None
    if rank == 0:
        (out / 'KONFIG.json').write_text(json.dumps(dict(vars(a), karty=world, akumulacja=akum, parametry=npar,
                                                         klatek=len(dane), filmow=len(dane.filmy)), indent=1), encoding='utf-8')
        if a.podglad_dane and a.vae:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval().requires_grad_(False)
            pl_lat, pl_mas = maski_podgladu(a.podglad_dane)
            pl_lat, pl_mas = pl_lat.to(dev), pl_mas.to(dev)
        print(f'UNet {npar / 1e6:.0f} mln parametrów, {len(dane)} klatek z {len(dane.filmy)} filmów, {world} kart, '
              f'{a.na_karte} na kartę × akumulacja {akum} = {a.paczka}', flush=True)

    sampler.set_epoch(epoka)
    it = iter(ladowanie)
    t_log, suma_log, n_log = time.time(), 0.0, 0
    log = open(out / 'LOG.jsonl', 'a', encoding='utf-8') if rank == 0 else None
    while krok < a.kroki:
        for g in opt.param_groups:
            g['lr'] = a.lr * min(1.0, (krok + 1) / a.rozgrzewka)
        strata_kroku = 0.0
        for m in range(akum):
            try:
                lat, mas = next(it)
            except StopIteration:
                epoka += 1; sampler.set_epoch(epoka); it = iter(ladowanie)
                lat, mas = next(it)
            lat, mas = lat.to(dev, non_blocking=True), mas.to(dev, non_blocking=True)
            B = lat.shape[0]
            szum = torch.randn_like(lat) + a.offset * torch.randn(B, lat.shape[1], 1, 1, device=dev)
            t = torch.randint(0, harm.config.num_train_timesteps, (B,), device=dev)
            wej = torch.cat([harm.add_noise(lat, szum, t), mas * (torch.rand(B, 1, 1, 1, device=dev) >= a.bez_maski)], 1)
            ctx = model.no_sync() if (rozpr and m < akum - 1) else contextlib.nullcontext()
            with ctx:
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cuda):
                    pred = model(wej, t, encoder_hidden_states=emb.expand(B, -1, -1)).sample
                strata = F.mse_loss(pred.float(), szum) / akum
                strata.backward()
            strata_kroku += float(strata.detach())
        norma = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        krok += 1
        if ema is not None:
            d = min(a.ema, (1 + krok) / (10 + krok))
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), unet.parameters()):
                    pe.lerp_(p.detach(), 1 - d)
        suma_log += strata_kroku; n_log += 1
        flaga = torch.tensor([float((out / 'STOP').exists())], device=dev)
        if rozpr:
            dist.all_reduce(flaga)
        koniec = bool(flaga.item() > 0) or krok >= a.kroki
        if rank == 0 and (krok % a.log_co == 0 or koniec):
            dt = time.time() - t_log
            wpis = dict(krok=krok, epoka=epoka, strata=round(suma_log / n_log, 5), norma=round(float(norma), 4),
                        lr=opt.param_groups[0]['lr'], s_na_krok=round(dt / n_log, 3), obrazow_na_s=round(n_log * a.paczka / dt, 1),
                        pamiec_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0)
            log.write(json.dumps(wpis) + '\n'); log.flush()
            print(json.dumps(wpis), flush=True)
            (out / 'STATUS.json').write_text(json.dumps(dict(wpis, stan='trwa', kroki=a.kroki, karty=world,
                                                             godzin_do_konca=round((a.kroki - krok) * dt / n_log / 3600, 2))), encoding='utf-8')
            t_log, suma_log, n_log = time.time(), 0.0, 0
        if rank == 0 and (krok % a.zapis_co == 0 or koniec):
            zapisz_atomowo(dict(unet=unet.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), krok=krok, epoka=epoka,
                                konfig=vars(a)), sciezka)
        if rank == 0 and (krok % a.ema_co == 0 or koniec):
            kopia = copy.deepcopy(ema).to('cpu', torch.float16)
            kopia.save_pretrained(out / f'ema_{krok:06d}', safe_serialization=True)
            del kopia
        if rank == 0 and vae is not None and (krok % a.podglad_co == 0 or koniec):
            gen = probkuj(ema, pl_mas, emb, dev)
            zapisz_podglad(vae, pl_lat, gen, pl_mas, out / f'podglad_{krok:06d}.jpg')
        if rozpr:
            dist.barrier()
        if koniec:
            break
    if rank == 0:
        stan = 'koniec' if krok >= a.kroki else 'przerwane'
        (out / 'STATUS.json').write_text(json.dumps(dict(stan=stan, krok=krok, kroki=a.kroki, karty=world)), encoding='utf-8')
        if stan == 'przerwane' and (out / 'STOP').exists():
            (out / 'STOP').unlink()
        print(f'{stan} w kroku {krok}', flush=True)
    if rozpr:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
