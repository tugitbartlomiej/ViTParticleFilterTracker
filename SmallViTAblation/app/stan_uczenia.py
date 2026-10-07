"""Stan uczenia gęstej historii w dowolnym kroku: wagi, bufory BatchNorm i stan AdamW (momenty m, v, licznik kroków).

Zapisane są tylko wagi co 10 kroków (wyniki/pelny/wagi/kNNNN.pt), bez stanu optymalizatora. Paczki odtwarzają się
z ziarna dokładnie (paczki_uczenia.py; sprawdzone: 10 kroków od k0000 daje k0010 z różnicą 0,017 % ruchu wag, kontrola
z innym ziarnem 40 razy więcej). Tu powtarzamy uczenie kodem tego biegu (kopia kod/ eksperymentu, bez zmian) i co
10 kroków podmieniamy wagi na zapisane (kotwica: wagi zawsze zgodne z biegiem), a momenty AdamW biorą się z powtórki.
Migawki stanu (wagi + AdamW) co 100 kroków w app/cache_uczenia/adamw_<klucz>/, więc stan w kroku k to migawka
i najwyżej 99 kroków powtórki (kilka sekund).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

import analiza_uczenia as AU
from paczki_uczenia import odtworz_paczki

CO_MIGAWKA = 100


def _srodowisko(bieg):
    """Ścieżki danych jak w uruchom_lokalnie.cmd (baza 1500 w cache maly_vit_od_zera) i import kodu biegu."""
    exp = Path(bieg['dane']).parents[2]                      # experiments/<eksperyment>
    root = exp.parent
    os.environ.setdefault('MALY_VIT_WEJSCIA', str(root / 'maly_vit_od_zera' / 'eden' / 'wejscia'))
    os.environ.setdefault('MALY_VIT_DANE', str(root / 'koopman_fastgan_petla' / 'local_inputs' / 'data'))
    os.environ.setdefault('MALY_VIT_CACHE', str(root / 'maly_vit_od_zera' / 'cache'))
    kod = str(exp / 'kod')
    # moduły biegu (dane, trening, model) w izolacji: aplikacja może mieć już inny moduł „model” w sys.modules
    nazwy = ('dane', 'trening', 'model')
    stare = {n: sys.modules.pop(n) for n in nazwy if n in sys.modules}
    sys.path.insert(0, kod)
    try:
        import importlib
        M = importlib.import_module('model')
        D = importlib.import_module('dane')
        T = importlib.import_module('trening')
    finally:
        sys.path.remove(kod)
        for n in nazwy:
            sys.modules.pop(n, None)
        sys.modules.update(stare)
    return D, T, M


class Odtworzenie:
    def __init__(self, bieg, ziarno=None):
        self.bieg = bieg
        self.D, self.T, self.M = _srodowisko(bieg)
        meta = AU._json(Path(bieg['dane']) / 'metadata.json', {})
        self.ziarno = int(ziarno if ziarno is not None else meta.get('seed', 47))
        self.kroki = bieg['ksztalt'][0] - 1
        self.baza = self.D.baza()
        self.I, self.O, self.A = odtworz_paczki(self.kroki, len(self.baza), self.ziarno)
        self.kat = AU.CACHE / f"adamw_{bieg['klucz']}"
        self.kat.mkdir(parents=True, exist_ok=True)
        self.wagi = Path(bieg['wagi'])

    # ------------------------------------------------------------ paczka kroku
    def paczka(self, k, dev='cuda'):
        """Paczka kroku k (aktualizacja k -> k+1) dokładnie jak w uczeniu: x, heat, ltrb, poz (na dev) i indeksy bazy."""
        zb, idx, odb, a = self.baza, self.I[k], self.O[k], self.A[k]
        cel0, cel1 = zb.cele
        x = torch.from_numpy(zb.X[idx]).to(dev).permute(0, 3, 1, 2).float() / 255.
        ob = torch.from_numpy(odb).to(dev)
        x = torch.where(ob[:, None, None, None], x.flip(3), x)
        at = torch.from_numpy(a).to(dev)
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * at[:, 0, None, None, None] + m * at[:, 1, None, None, None]).clamp(0, 1)
        it = torch.from_numpy(idx)
        heat = torch.where(torch.from_numpy(odb)[:, None, None], cel1[0][it], cel0[0][it]).to(dev)
        ltrb = torch.where(torch.from_numpy(odb)[:, None, None, None], cel1[1][it], cel0[1][it]).to(dev)
        poz = torch.where(torch.from_numpy(odb)[:, None, None], cel1[2][it], cel0[2][it]).to(dev)
        return x, heat, ltrb, poz, idx

    # ------------------------------------------------------------ powtórka
    def _krok(self, u, k):
        """Jeden krok uczenia k -> k+1 (jak trening.Uczen.kroki, bez zapisu)."""
        x, heat, ltrb, poz, _ = self.paczka(k, u.dev)
        u.net.train()
        for g in u.opt.param_groups:
            g['lr'] = u._lr()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o, _ = u.net(x)
        L, _ = self.M.strata(o, heat, ltrb, poz)
        u.opt.zero_grad(set_to_none=True)
        L.backward()
        torch.nn.utils.clip_grad_norm_(u.net.parameters(), 1.0)
        u.opt.step()
        u.krok += 1

    def _kotwica(self, u):
        """Wagi i bufory z zapisanego checkpointu kroku u.krok (gdy istnieje), momenty AdamW zostają. Zwraca dryf
        powtórki przed podmianą: ||θ_powtórka − θ_zapis|| / ||θ_zapis − θ_(krok−10)|| (ułamek ruchu wag w 10 krokach)."""
        p = self.wagi / f'k{u.krok:04d}.pt'
        if not p.exists():
            return None
        w = torch.load(p, map_location=u.dev, weights_only=True)['net']
        dryf = None
        pop = getattr(self, '_poprz', None)
        if pop is not None:
            num = sum(float(((u.net.state_dict()[n].float() - w[n].float()) ** 2).sum()) for n in w if w[n].dtype.is_floating_point)
            den = sum(float(((pop[n].float() - w[n].float()) ** 2).sum()) for n in w if w[n].dtype.is_floating_point)
            dryf = (num / max(den, 1e-30)) ** .5
        u.net.load_state_dict(w)
        self._poprz = {n: t.clone() for n, t in w.items()}
        return dryf

    def _migawka(self, k):
        return self.kat / f'k{k:04d}.pt'

    def przygotuj(self, postep=None):
        """Powtórka całego uczenia i migawki co CO_MIGAWKA kroków (raz; kolejne wywołania nic nie robią)."""
        if all(self._migawka(k).exists() for k in range(0, self.kroki + 1, CO_MIGAWKA)):
            return
        torch.backends.cudnn.benchmark = False
        u = self.T.Uczen(self.ziarno)
        self._poprz = None
        self._kotwica(u)
        torch.save(u.stan(), self._migawka(0))
        self.dryf = []
        for k in range(self.kroki):
            self._krok(u, k)
            if u.krok % 10 == 0:
                self.dryf.append((u.krok, self._kotwica(u)))
            if u.krok % CO_MIGAWKA == 0:
                torch.save(u.stan(), self._migawka(u.krok))
            if postep and k % 20 == 0:
                postep(k, self.kroki)

    def stan(self, k):
        """Uczen w stanie kroku k (wagi, bufory i AdamW przed aktualizacją k -> k+1)."""
        k0 = (k // CO_MIGAWKA) * CO_MIGAWKA
        u = self.T.Uczen(self.ziarno)
        u.wczytaj(torch.load(self._migawka(k0), map_location=u.dev, weights_only=False))
        while u.krok < k:
            self._krok(u, u.krok)
            if u.krok % 10 == 0:
                self._kotwica(u)
        return u
