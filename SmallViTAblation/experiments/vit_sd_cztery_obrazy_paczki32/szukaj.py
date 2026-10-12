"""Które 32 unikalne zdjęcia train są najbliższe naturalnemu latentowi każdego
z czterech obrazów SD w rzeczywiście zapisanym podzbiorze? Użytkownik przerwał
pełny skan; --finalize-existing tylko weryfikuje CSV i przelicza finalistów.
Niezgodne liczby lub kolejność CSV, błędny split, zmiana wag lub duplikaty
w paczce obalają poprawność wyniku. Ranking dotyczy wyłącznie deterministycznego
przetworzonego prefiksu, nie pełnego train i nie próbki losowej. Nie mierzymy
generalizacji ani realizmu; brak ustalonego progu bliskości.
"""
print('IMPORTING', flush=True)
import csv
import gc
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from model_snapshot import MalyViT, dekoduj

ROOT = Path(__file__).resolve().parents[2]
FAMILY = 'vit_sd_cztery_obrazy_paczki32'
OUT = ROOT / 'outputs' / FAMILY
SOURCE = ROOT / 'outputs/vit_latent128_jedno_zdjecie_90'
QUERY_SOURCE = ROOT / 'outputs/vit_sd_cztery_szumy'
SEEDS = [2026100511, 2026100523, 2026100537, 2026100549]
ARCHIVES = ROOT.parent / 'Dataset/Cataract-LMM/3_Object_Tracking/annotations'
SPLITS = ROOT / 'feature_cache/eden_static36/data/tracking_static36/split_manifest.json'
BATCH = 4
METRICS = ['mse768', 'mse128', 'mse640', 'cosine768', 'cosine128', 'cosine640']
FIELDS = ['ordinal', 'film', 'split', 'archive', 'member', 'pixel_sha256', *METRICS, 'confidence']
sys.path.insert(0, str(ROOT))
import src.narzedzia.deklaracja_wynikow as deklaracja
deklaracja._brudne_drzewo = lambda: True  # Known dirty tree, no huge git status.


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def js(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def text_atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def decode(raw):
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError('Image decode failed')
    # Dedup hashes full-resolution decoded RGB pixels with explicit dimensions.
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h = hashlib.sha256()
    h.update(str(rgb.shape).encode('ascii'))
    h.update(rgb.tobytes())
    return cv2.resize(rgb, (480, 320), interpolation=cv2.INTER_AREA), h.hexdigest()


def priority_jobs():
    jobs = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            args = process.info['cmdline'] or []
            # Inspect script arguments, not incidental shell text or stale PID files.
            if process.pid != os.getpid() and 'python' in (process.info['name'] or '').lower():
                if any('vit_sd_zgodnosc90' in a.replace('\\', '/') for a in args[1:]):
                    jobs.append(process.pid)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return jobs


def wait_priority(processed, total, phase):
    started = time.monotonic()
    while jobs := priority_jobs():
        js(OUT / 'PROGRESS.json', dict(status='queued', phase=phase, processed=processed,
                                      total=total, priority_pids=jobs, pid=os.getpid()))
        print('GPU QUEUED priority SD', jobs, 'processed', processed, flush=True)
        if time.monotonic() - started > 5400:
            raise TimeoutError('Priority SD remained active for 90 minutes; no indefinite queue')
        time.sleep(10)


def trim_unique(records, limit=64):
    seen = set()
    result = []
    for r in sorted(records, key=lambda r: (r['mse768'], r['ordinal'])):
        if r['pixel_sha256'] in seen:
            continue
        seen.add(r['pixel_sha256'])
        result.append(r)
        if len(result) == limit:
            break
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'RESULT.json').exists():
        raise RuntimeError('Completed result exists; no overwrite')
    partial = '--finalize-existing' in sys.argv
    if (OUT / 'STOP_REQUEST.json').exists() and not partial:
        raise RuntimeError('User stopped exhaustive scan; use --finalize-existing only')
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    query_meta = []
    for seed in SEEDS:
        source = QUERY_SOURCE / f'OBRAZ_{seed}_prowadzony.png'
        if not source.is_file():
            raise FileNotFoundError(source)
        qdir = OUT / str(seed)
        qdir.mkdir(exist_ok=True)
        destination = qdir / 'OBRAZ_ZAPYTANIA.png'
        if destination.exists() and sha(destination) != sha(source):
            raise RuntimeError('Existing query differs; no overwrite')
        if not destination.exists():
            shutil.copy2(source, destination)
        query_meta.append(dict(seed=seed, source=str(source), copy=str(destination), sha256=sha(source)))
    js(OUT / 'QUERIES.json', query_meta)
    wait_priority(0, 310257, 'partial_finalize_query_forward' if partial else 'before_query_forward')
    with np.load(SOURCE / 'LATENT_PELNY.npz') as data:
        channels_np = data['channels0'].copy()
    assert channels_np.shape == (128,) and len(np.unique(channels_np)) == 128
    assert np.all((channels_np >= 0) & (channels_np < 768))
    channels = torch.as_tensor(channels_np, dtype=torch.long, device='cuda')
    other = torch.ones(768, dtype=torch.bool, device='cuda')
    other[channels] = False
    net = MalyViT().eval().requires_grad_(False)
    net.load_state_dict(torch.load(SOURCE / 'MODEL_VIT_K1000.pt', map_location='cpu', weights_only=True)['net'])
    net.zapisuj_latent(True)

    def weight_hash():
        h = hashlib.sha256()
        for name, x in net.state_dict().items():
            h.update(name.encode())
            h.update(x.cpu().numpy().tobytes())
        return h.hexdigest()

    weights_before = weight_hash()
    net.cuda()
    query_latents = []
    with torch.inference_mode():
        for meta in query_meta:
            print('QUERY FORWARD', meta['seed'], flush=True)
            rgb, pixel_hash = decode(Path(meta['copy']).read_bytes())
            output, z = net(torch.from_numpy(rgb).permute(2, 0, 1)[None].float().cuda() / 255)
            natural = z[0].cpu().numpy().copy()
            assert natural.shape == (600, 768) and np.isfinite(natural).all()
            query_latents.append(natural)
            np.savez_compressed(OUT / str(meta['seed']) / 'LATENT_ZAPYTANIA.npz',
                                query=natural, natural=natural, channels0=channels_np,
                                grid_shape=np.array([20, 30]), input_rgb=rgb,
                                detector_output=output.cpu().numpy())
            meta['pixel_sha256'] = pixel_hash
    targets = torch.from_numpy(np.stack(query_latents)).cuda()

    def measure(z):
        # Sequential differences against 4 targets avoid a Bx4x600x768 allocation.
        results = []
        for target in targets:
            diff = (z - target[None]).square()
            values = [diff.mean((1, 2)), diff[:, :, channels].mean((1, 2)), diff[:, :, other].mean((1, 2))]
            for subset in (slice(None), channels, other):
                a = z[:, :, subset].flatten(1)
                b = target[:, subset].flatten()[None].expand(z.shape[0], -1)
                values.append(torch.nn.functional.cosine_similarity(a, b, dim=1))
            results.append(torch.stack(values, 1))
        return torch.stack(results, 1).cpu().numpy()

    with torch.inference_mode():
        smoke = measure(targets)
    diagonal = smoke[np.arange(4), np.arange(4)]
    assert np.allclose(diagonal[:, :3], 0) and np.allclose(diagonal[:, 3:], 1, atol=1e-5)
    assert weight_hash() == weights_before
    js(OUT / 'SMOKE.json', dict(status='passed', shape=list(targets.shape), self_metrics=diagonal.tolist(),
                               query_hashes=[m['sha256'] for m in query_meta], batch_size=BATCH,
                               checkpoint_sha256=sha(SOURCE / 'MODEL_VIT_K1000.pt'),
                               channels_source_sha256=sha(SOURCE / 'LATENT_PELNY.npz'), weights_unchanged=True))
    print('SMOKE PASSED 4 natural queries', flush=True)
    if '--smoke' in sys.argv:
        return
    groups = json.loads(SPLITS.read_text())['groups']
    inventory = []
    membership_hash = hashlib.sha256()
    for path in sorted(ARCHIVES.glob('*.zip')):
        if groups.get(path.stem) != 'train':
            continue
        with zipfile.ZipFile(path) as archive:
            members = sorted((x for x in archive.infolist() if x.filename.lower().endswith(('.jpg', '.jpeg', '.png'))), key=lambda x: x.filename)
            assert len({x.filename for x in members}) == len(members)
            for member in members:
                membership_hash.update(f'{path.name}|{member.filename}|{member.CRC}|{member.file_size}\n'.encode())
            inventory.append(dict(path=str(path), film=path.stem, split='train', count=len(members),
                                  members=[x.filename for x in members]))
    total = sum(entry['count'] for entry in inventory)
    assert total == 310257, total
    js(OUT / 'TRAIN_MEMBERSHIP.json', dict(total=total, archives=inventory, split_manifest_sha256=sha(SPLITS),
                                         train_membership_sha256=membership_hash.hexdigest(),
                                         rule="groups[archive.stem] == 'train'; no val/test/unknown"))
    shutil.copy2(__file__, OUT / 'szukaj_source.py')
    shutil.copy2(Path(__file__).with_name('model_snapshot.py'), OUT / 'model_snapshot_source.py')
    text_atomic(OUT / 'RAPORT.md', '# Wyszukiwanie w toku\n\nCztery naturalne zapytania. Jeden pełny przebieg 310257 zdjęć train, ranking raw MSE 600x768. Docelowo 32 różne obrazy na zapytanie, deduplikacja pełnych zdekodowanych pikseli RGB. Duplikaty między paczkami są dozwolone. Wyniki częściowe nie stanowią wyniku końcowego.\n')
    if partial:
        print('PARTIAL FINALIZATION ONLY; no candidate scan', flush=True)
        top = []
        counts = []
        order_hashes = []
        seen = Counter()
        inputs = None
        for qi, seed in enumerate(SEEDS):
            rows = []
            count = 0
            order_hash = hashlib.sha256()
            expected = ((entry['path'], name) for entry in inventory for name in entry['members'])
            with open(OUT / str(seed) / 'RANKING_TRAIN.csv', newline='', encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    count += 1
                    assert int(row['ordinal']) == count
                    assert row['split'] == 'train' and groups[row['film']] == 'train'
                    assert (row['archive'], row['member']) == next(expected)
                    order_hash.update(f"{row['archive']}|{row['member']}|{row['pixel_sha256']}\n".encode())
                    record = dict(row, ordinal=count, confidence=float(row['confidence']))
                    record.update({name: float(row[name]) for name in METRICS})
                    assert all(np.isfinite(record[name]) for name in METRICS)
                    rows.append(record)
                    if len(rows) >= 512:
                        rows = trim_unique(rows)
                    if qi == 0:
                        seen[row['film']] += 1
            top.append(trim_unique(rows))
            counts.append(count)
            order_hashes.append(order_hash.hexdigest())
            print('RECOVERED', seed, count, 'complete CSV records', flush=True)
        assert len(set(counts)) == 1 and len(set(order_hashes)) == 1, (counts, order_hashes)
        processed = counts[0]
        assert 32 <= processed < total and sum(seen.values()) == processed
        js(OUT / 'SCANNED_MEMBERSHIP.json', dict(status='partial_verified', processed=processed,
            available_train=total, ordered_rows_sha256=order_hashes[0], seen_by_archive=dict(seen),
            order='sorted ZIP paths; sorted JPEG/JPG/PNG member names; deterministic prefix, not random',
            csvs=[str(OUT / str(seed) / 'RANKING_TRAIN.csv') for seed in SEEDS],
            csv_sha256={str(seed): sha(OUT / str(seed) / 'RANKING_TRAIN.csv') for seed in SEEDS}))
        js(OUT / 'INTERIM_TOP64.json', dict(status='partial_scan_stopped', processed=processed,
            available_train=total, queries=dict(zip(map(str, SEEDS), top))))
    else:
        pending = queue.Queue(maxsize=2)
        stop = threading.Event()

        def put(item):
            while not stop.is_set():
                try:
                    pending.put(item, timeout=.5)
                    return
                except queue.Full:
                    pass

        def producer():
            try:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for entry in inventory:
                        with zipfile.ZipFile(entry['path']) as archive:
                            for lo in range(0, entry['count'], BATCH):
                                if stop.is_set():
                                    return
                                names = entry['members'][lo:lo + BATCH]
                                raw = [archive.read(name) for name in names]
                                decoded = list(executor.map(decode, raw))
                                records = [dict(archive=entry['path'], member=name, film=entry['film'], split='train', pixel_sha256=d[1]) for name, d in zip(names, decoded)]
                                put((records, np.stack([d[0] for d in decoded])))
                put(None)
            except BaseException as e:
                put(e)

        threading.Thread(target=producer, daemon=True).start()
        inputs = None  # May yield GPU before first candidate batch.
        top = [[] for _ in SEEDS]
        processed = 0
        seen = Counter()
        last_report = time.perf_counter()
        scan_start = last_report
        paused_seconds = 0.0
        print('SCAN START', total, 'train images; batch', BATCH, flush=True)
        js(OUT / 'PROGRESS.json', dict(status='running', phase='scan', processed=0, total=total, pid=os.getpid()))
        try:
            with ExitStack() as stack, torch.inference_mode():
                streams = [stack.enter_context(open(OUT / str(seed) / 'RANKING_TRAIN.csv', 'w', newline='', encoding='utf-8')) for seed in SEEDS]
                writers = [csv.DictWriter(stream, fieldnames=FIELDS, extrasaction='ignore') for stream in streams]
                for writer in writers:
                    writer.writeheader()
                while True:
                    if (OUT / 'STOP_REQUEST.json').exists():
                        break
                    # Yield GPU if the priority SD worker starts while retrieval is running.
                    if priority_jobs():
                        for stream in streams:
                            stream.flush()
                        net._latent = None
                        del output, z, inputs
                        net.cpu()
                        targets = targets.cpu()
                        channels = channels.cpu()
                        other = other.cpu()
                        gc.collect()
                        torch.cuda.empty_cache()
                        pause = time.perf_counter()
                        wait_priority(processed, total, 'scan_paused_for_sd')
                        paused_seconds += time.perf_counter() - pause
                        net.cuda()
                        targets = targets.cuda()
                        channels = channels.cuda()
                        other = other.cuda()
                    item = pending.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    records, images = item
                    inputs = torch.from_numpy(images).permute(0, 3, 1, 2).float().cuda() / 255
                    output, z = net(inputs)  # One forward per candidate, shared by all 4 queries.
                    values = measure(z)
                    boxes, confidence = dekoduj(output)
                    boxes = boxes.cpu().numpy()
                    confidence = confidence.cpu().numpy()
                    assert np.isfinite(values).all()
                    for k, base in enumerate(records):
                        processed += 1
                        seen[base['film']] += 1
                        base.update(ordinal=processed, confidence=float(confidence[k, 0]), bbox_xyxy=boxes[k, 0].tolist())
                        for qi in range(4):
                            record = dict(base, **dict(zip(METRICS, map(float, values[k, qi]))))
                            writers[qi].writerow(record)
                            top[qi].append(record)
                    top = [trim_unique(rows) for rows in top]
                    now = time.perf_counter()
                    if now - last_report >= 10 or processed == total:
                        for stream in streams:
                            stream.flush()
                        fps = processed / max(now - scan_start - paused_seconds, .001)
                        js(OUT / 'INTERIM_TOP64.json', dict(status='interim', processed=processed, queries=dict(zip(map(str, SEEDS), top))))
                        js(OUT / 'PROGRESS.json', dict(status='running', phase='scan', processed=processed, total=total,
                                                      pid=os.getpid(), images_per_second=fps, eta_seconds=(total - processed) / fps,
                                                      best_mse=[rows[0]['mse768'] for rows in top], seen_by_archive=dict(seen)))
                        print('PROGRESS', processed, total, 'fps', round(fps, 2), 'best', [round(rows[0]['mse768'], 6) for rows in top], flush=True)
                        last_report = now
                if not (OUT / 'STOP_REQUEST.json').exists():
                    assert processed == total and seen == Counter({e['film']: e['count'] for e in inventory})
                else:
                    partial = True
        finally:
            stop.set()
    js(OUT / 'COVERAGE.json', dict(status='partial_verified' if partial else 'verified',
        processed=processed, ranking_denominator=processed, available_train=total,
        total=total, seen_by_archive=dict(seen), train_membership_sha256=membership_hash.hexdigest(),
        exhaustive=not partial, sampling=False, random_sample=False,
        scope='nearest within deterministic processed prefix only' if partial else 'all train'))
    finalists = {(r['archive'], r['member']): r for rows in top for r in rows}
    refined = [[] for _ in SEEDS]
    cache = {}
    print('REFINE START unique finalists', len(finalists), flush=True)
    with torch.inference_mode():
        for number, (key, record) in enumerate(finalists.items(), 1):
            if priority_jobs():
                net._latent = None
                net.cpu()
                targets = targets.cpu()
                channels = channels.cpu()
                other = other.cpu()
                del output, z, inputs
                gc.collect()
                torch.cuda.empty_cache()
                wait_priority(processed, total, 'refine_paused_for_sd')
                net.cuda()
                targets = targets.cuda()
                channels = channels.cuda()
                other = other.cuda()
            with zipfile.ZipFile(key[0]) as archive:
                raw = archive.read(key[1])
            image, pixel_hash = decode(raw)
            assert pixel_hash == record['pixel_sha256']
            inputs = torch.from_numpy(image).permute(2, 0, 1)[None].float().cuda() / 255
            output, z = net(inputs)
            values = measure(z)[0]
            boxes, confidence = dekoduj(output)
            cache[key] = dict(raw=raw, natural=z[0].cpu().numpy().copy(), detector=output[0].cpu().numpy().copy())
            for qi in range(4):
                r = dict(record, **dict(zip(METRICS, map(float, values[qi]))), confidence=float(confidence[0, 0]), bbox_xyxy=boxes[0, 0].cpu().tolist())
                refined[qi].append(r)
                for old in top[qi]:
                    if (old['archive'], old['member']) == key:
                        assert np.isclose(r['mse768'], old['mse768'], rtol=1e-4, atol=2e-5)
            js(OUT / 'PROGRESS.json', dict(status='running', phase='refine', processed=processed, total=total, ranking_denominator=processed, partial_scan=partial, finalists_done=number, finalists_total=len(finalists)))
    selected = [trim_unique(rows, 32) for rows in refined]
    assert all(len(rows) == 32 for rows in selected)
    # Resolve every archive/member alias of selected pixel hashes without redoing forwards.
    selected_hashes = {r['pixel_sha256'] for rows in selected for r in rows}
    aliases = {h: [] for h in selected_hashes}
    with open(OUT / str(SEEDS[0]) / 'RANKING_TRAIN.csv', newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['pixel_sha256'] in aliases:
                aliases[row['pixel_sha256']].append({k: row[k] for k in ['archive', 'member', 'film', 'split', 'ordinal']})
    js(OUT / 'PIXEL_DEDUP_PROVENANCE.json', aliases)
    for qi, seed in enumerate(SEEDS):
        qdir = OUT / str(seed)
        photo_dir = qdir / 'ZDJECIA_32'
        photo_dir.mkdir(exist_ok=True)
        sheet = np.full((8 * 180, 4 * 240, 3), 245, dtype=np.uint8)
        for rank, r in enumerate(selected[qi], 1):
            key = (r['archive'], r['member'])
            data = cache[key]
            destination = photo_dir / f'{rank:02d}_{Path(r["member"]).name}'
            destination.write_bytes(data['raw'])  # Exact original bytes, original filename extension.
            r.update(rank=rank, original_photo=str(destination), original_sha256=sha(destination),
                     pixel_duplicate_alias_count=len(aliases[r['pixel_sha256']]))
            bgr = cv2.imdecode(np.frombuffer(data['raw'], np.uint8), cv2.IMREAD_COLOR)
            thumb = cv2.resize(bgr, (240, 160), interpolation=cv2.INTER_AREA)
            yy, xx = divmod(rank - 1, 4)
            sheet[yy * 180:yy * 180 + 160, xx * 240:(xx + 1) * 240] = thumb
            cv2.putText(sheet, f'{rank:02d} MSE {r["mse768"]:.6f}', (xx * 240 + 4, yy * 180 + 175), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 0, 0), 1, cv2.LINE_AA)
        assert cv2.imwrite(str(qdir / 'STYKOWKA_4x8.png'), sheet)
        with open(qdir / 'TOP32.csv', 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['rank', *FIELDS, 'original_photo', 'original_sha256', 'pixel_duplicate_alias_count'], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(selected[qi])
        js(qdir / 'TOP32.json', dict(seed=seed, ranking='raw full MSE600x768', unique_pixels=32, selected=selected[qi]))
        np.savez_compressed(qdir / 'LATENT_WYBRANE32.npz',
                            natural=np.stack([cache[(r['archive'], r['member'])]['natural'] for r in selected[qi]]),
                            query=query_latents[qi], channels0=channels_np, grid_shape=np.array([20, 30]),
                            detector_output=np.stack([cache[(r['archive'], r['member'])]['detector'] for r in selected[qi]]),
                            ranks=np.arange(1, 33), pixel_sha256=np.array([r['pixel_sha256'] for r in selected[qi]]),
                            archives=np.array([r['archive'] for r in selected[qi]]), members=np.array([r['member'] for r in selected[qi]]))
    sets_pixels = [{r['pixel_sha256'] for r in rows} for rows in selected]
    sets_members = [{(r['archive'], r['member']) for r in rows} for rows in selected]
    overlap = [[len(a & b) for b in sets_pixels] for a in sets_pixels]
    overlap_members = [[len(a & b) for b in sets_members] for a in sets_members]
    assert weight_hash() == weights_before
    assert all(sha(Path(m['copy'])) == m['sha256'] for m in query_meta)
    result = dict(status='complete', ziarno=20261005, query_seeds=SEEDS, queries=query_meta,
                  train_images_scanned=processed, train_archives=len(inventory), scan_candidate_forwards=processed,
                  unique_refinement_forwards=len(finalists), batch_size=BATCH, weights_unchanged=True, known_dirty_tree=True,
                  train_membership_sha256=membership_hash.hexdigest(), split_manifest_sha256=sha(SPLITS),
                  checkpoint_sha256=sha(SOURCE / 'MODEL_VIT_K1000.pt'), channels_source_sha256=sha(SOURCE / 'LATENT_PELNY.npz'),
                  overlap_pixel_counts=overlap, overlap_archive_member_counts=overlap_members,
                  total_selections=128, unique_selected_pixels=len(set.union(*sets_pixels)),
                  unique_selected_members=len(set.union(*sets_members)), seconds=time.perf_counter() - start,
                  partial_scan=partial, ranking_denominator=processed, available_train_images=total, scanned_archives=len(seen),
                  coverage_order='deterministic sorted ZIP/member prefix, not random sampling',
                  selection='nearest within processed train prefix only; raw natural latent full MSE; top64 distinct pixels per query and singleton union refinement; top32 each' if partial else 'train-only exhaustive natural latent retrieval',
                  limitation='No claim of 128 globally unique photos. No visual realism or generalization guarantee. Detector confidence is not accuracy or mAP.')
    deklaracja.zapisz_wynik(__file__, 'train_cztery_paczki32', result)
    report = '# PARTIAL SCAN: cztery paczki po 32 zdjęcia train\n\n' if partial else '# Cztery paczki po 32 rzeczywiste zdjęcia train\n\n'
    report += f'Zakres rankingu: {processed} przetworzonych zdjęć train z {len(seen)} rozpoczętych ZIP; cały dostępny train liczy {total}. Ranking najbliższych wyłącznie w tym podzbiorze. Kolejność to posortowane ścieżki ZIP i nazwy plików wewnątrz; deterministyczny początek, nie losowa próbka. Jeden forward na kandydata; {len(finalists)} dodatkowych forwardów finalistów. Naturalne zapytania z czterech obrazów SD, bez sztucznego celu. Zamrożone wagi, RGB 480x320 INTER_AREA, ranking surowe MSE 600x768.\n\n'
    report += f'Każda paczka ma 32 różne hashe pełnych pikseli RGB i różne pary archive/member. Łącznie 128 wyborów, ale {result["unique_selected_pixels"]} różnych obrazów. Dopuszczamy powtórzenia pomiędzy zapytaniami. Macierz wspólnych pikseli (kolejność {SEEDS}):\n\n{json.dumps(overlap)}\n\n'
    report += 'Ranking nie gwarantuje podobieństwa wizualnego. To retrieval ze zbioru train, bez pomiaru generalizacji. Pewność detektora nie jest dokładnością. Znane niezacommitowane zmiany repozytorium. TOP32, pełne CSV, latenty, oryginalne zdjęcia i stykówki są w katalogach zapytań.\n'
    text_atomic(OUT / 'RAPORT.md', report)
    text_atomic(Path(__file__).parent / 'WYNIKI_TRAIN.md', report)
    js(OUT / 'RESULT.json', result)
    js(OUT / 'PROGRESS.json', dict(status='complete', phase='complete', processed=processed, total=total, ranking_denominator=processed, partial_scan=partial, unique_selected_pixels=result['unique_selected_pixels']))
    exclude = {'HASHES.json', 'stdout.log', 'stderr.log', 'PID.txt', 'JOB.json', 'smoke_stdout.log', 'smoke_stderr.log', 'finalize_stdout.log', 'finalize_stderr.log'}
    js(OUT / 'HASHES.json', {str(p.relative_to(OUT)).replace('\\', '/'): sha(p) for p in OUT.rglob('*')
                           if p.is_file() and p.name not in exclude and 'pycache' not in p.parts and not p.name.endswith('.tmp')})
    print('COMPLETE', processed, 'unique selected pixels', result['unique_selected_pixels'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        js(OUT / 'FAILURE.json', dict(status='failed', type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
        js(OUT / 'PROGRESS.json', dict(status='failed', error=str(error)))
        raise
