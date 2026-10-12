"""Wybór zdjęć do trajektorii i historie checkpointów (bez karty graficznej)."""
import json

import pytest

import historie_checkpointow as HC
import zbiory_zdjec as Z


def test_parsuj_i_zwiez():
    assert Z.parsuj('5, 0-2, 9-7; 2', 20) == [0, 1, 2, 5, 7, 8, 9]
    assert Z.zwiez([0, 1, 2, 5, 7, 8, 9]) == '0-2, 5, 7-9'
    assert Z.parsuj('', 10) == []
    with pytest.raises(ValueError):
        Z.parsuj('3-12', 10)


def test_losowe_powtarzalne():
    assert Z.losowe('1500', 7) == Z.losowe('1500', 7) and len(set(Z.losowe('1500', 7))) == 7


def test_podzbior_1500_zgodny_z_manifestem(tmp_path, monkeypatch):
    monkeypatch.setattr(Z, 'WYJSCIE', tmp_path)
    manifest, katalog = Z.zbuduj('1500', [4, 2, 2])
    rows = [json.loads(l) for l in open(manifest, encoding='utf-8')]
    zrodlo = Z.wiersze('1500')
    assert [r['index'] for r in rows] == [0, 1] and [r['zrodlo_indeks'] for r in rows] == [2, 4]
    assert all(r['image_sha256'] == zrodlo[r['zrodlo_indeks']]['image_sha256'] for r in rows)
    assert all((katalog / r['image_name']).is_file() for r in rows)


@pytest.mark.skipif(not Z.ZIPY.is_dir() or not Z.PULA_35000.is_file(), reason='brak zipów Cataract-LMM albo puli 35 000')
def test_podzbior_puli_35000(tmp_path, monkeypatch):
    monkeypatch.setattr(Z, 'WYJSCIE', tmp_path)
    monkeypatch.setattr(Z, 'CACHE_35000', tmp_path / 'obrazy')
    manifest, katalog = Z.zbuduj('35000', [0, 34999])
    rows = [json.loads(l) for l in open(manifest, encoding='utf-8')]
    for r in rows:
        W, H = r['source_width'], r['source_height']
        x, y, w, h = r['bbox_xywh_source']
        assert r['bbox_xyxy_480'] == [x * 480 / W, y * 320 / H, (x + w) * 480 / W, (y + h) * 320 / H]
        assert (katalog / r['image_name']).is_file()


def test_historie_opis_krokow():
    h = HC.historia('35k')
    assert h.opis_kroku(2188) == 'epoka 2 (krok 2188)' and h.opis_kroku(0) == 'epoka 0 (krok 0)'
    assert HC.historia('k1000').opis_kroku(120) == 'krok 120'
    if h.kroki():
        assert h.epoki()[0] == 0 and h.sciezka(h.kroki()[-1]).is_file()
