"""Cataract-LMM adapters; no inferred patient identity or synthetic tip boxes.

Dataset specification: https://doi.org/10.1038/s41597-026-07464-0
Tracking CSV boxes describe instruments, whereas *_tip contains point labels.
Segmentation and tracking are deliberately imported as separate experiments.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import shutil
import uuid
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path, PurePosixPath


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding='utf-8')
    tmp.replace(path)


def box_xywh(value, width: int, height: int) -> list[float]:
    """Fail on invalid geometry instead of silently changing the training target."""
    if isinstance(value, str):
        value = json.loads(value)
    if len(value) != 4:
        raise ValueError(f'Expected a COCO xywh box, received {value!r}')
    x, y, w, h = map(float, value)
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError('Nonfinite bounding box')
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > width+1e-3 or y+h > height+1e-3:
        raise ValueError(f'Box {value} outside image {width}x{height}; review source labels')
    return [x, y, w, h]


class ImageSource:
    """Bounded ZIP handle cache; reads members without extracting whole archives."""

    def __init__(self, max_archives=4):
        self.max_archives = max_archives
        self.archives = OrderedDict()

    def bytes(self, row: dict) -> bytes:
        if row.get('archive'):
            key = row['archive']
            if key not in self.archives:
                self.archives[key] = zipfile.ZipFile(key)
            self.archives.move_to_end(key)
            while len(self.archives) > self.max_archives:
                self.archives.popitem(last=False)[1].close()
            return self.archives[key].read(row['member'])
        return Path(row['path']).read_bytes()

    def image(self, row: dict):
        from PIL import Image
        with Image.open(io.BytesIO(self.bytes(row))) as image:
            return image.convert('RGB')

    def close(self):
        for archive in self.archives.values():
            archive.close()
        self.archives.clear()


def _base(video, group, number, width, height, subset, fps):
    return dict(frame_id=f'cataract_lmm/{subset}/{video}/{number:08d}',
                source_dataset='cataract_lmm', subset=subset, video_id=video,
                group_id=group, patient_id=None, patient_identity_verified=False,
                frame_index=number, timestamp=(number-1)/fps if fps else number,
                time_unit='seconds' if fps else 'frames', native_fps=fps,
                timestamp_origin='frame_index/fps' if fps else 'frame_index',
                native_width=width, native_height=height,
                label_origin='dataset_provided', annotations=[])


def tracking_rows(root: Path, stride: int, fps: float | None,
                  audit_path: Path | None = None) -> tuple[list[dict], dict]:
    """Use audited operation groups; preserve uncertainty about patient identity."""
    from PIL import Image
    audit_path = audit_path or root/'_audyt_pacjentow.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    groups = {}
    for group, videos in audit['grupy_operacji'].items():
        for video in videos:
            if video in groups:
                raise ValueError(f'Overlapping audit groups for {video}')
            groups[video] = group
    archives = sorted((root/'3_Object_Tracking/annotations').glob('*.zip'))
    available = {p.stem for p in archives}
    excluded = set()
    # Only documented exact duplicates; similar but shifted clips remain grouped.
    for pair in audit.get('duplikaty', []):
        present = sorted(set(pair) & available)
        excluded.update(present[1:])
    rows = []
    for archive in archives:
        video = archive.stem
        if video not in groups:
            raise ValueError(f'Missing operation group for {video}; update the audit mapping')
        if video in excluded:
            continue
        with zipfile.ZipFile(archive) as z:
            csv_names = [n for n in z.namelist() if PurePosixPath(n).name == 'annotation.csv']
            if len(csv_names) != 1:
                raise ValueError(f'{archive}: expected one annotation.csv')
            csv_name = csv_names[0]
            parent = PurePosixPath(csv_name).parent
            reader = csv.DictReader(io.StringIO(z.read(csv_name).decode('utf-8-sig')))
            size = None
            for raw in reader:
                number = int(raw['frame_number'])
                if (number-1) % stride:
                    continue
                member = str(parent/PurePosixPath(raw['filename']))
                if size is None:
                    with Image.open(io.BytesIO(z.read(member))) as im:
                        size = im.size
                row = _base(video, groups[video], number, *size, 'tracking', fps)
                row.update(archive=str(archive.resolve()), member=member,
                           group_origin='local_operation_audit',
                           target_kind='instrument_box', label_status='unknown')
                # Empty CSV cells are not silently certified negative annotations.
                bbox = raw.get('cystotome_bbox', '').strip()
                if bbox:
                    row['annotations'] = [dict(category_id=1, category_name='instrument',
                        source_category='cystotome', bbox=box_xywh(bbox, *size), iscrowd=0)]
                    row['label_status'] = 'provided'
                if raw.get('cystotome_tip', '').strip():
                    point = json.loads(raw['cystotome_tip'])
                    if len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
                        raise ValueError(f'{video}/{number}: invalid tip point')
                    row['tip_point'] = [float(v) for v in point]
                rows.append(row)
        print(f'[index] {video}: {len(rows)} candidate frames', flush=True)
    return rows, dict(subset='tracking', audit_sha256=hashlib.sha256(
        audit_path.read_bytes()).hexdigest(), excluded_duplicate_videos=sorted(excluded),
        grouping='conservative_operation_groups_not_verified_patients',
        target='instrument_bbox_not_tip_bbox', candidate_stride=stride)


def segmentation_rows(root: Path, stride: int, fps: float | None) -> tuple[list[dict], dict]:
    base = root/'2_Instance_Segmentation/annotations/coco_json'
    coco_path = base/'_annotations.coco.json'
    coco = json.loads(coco_path.read_text(encoding='utf-8'))
    categories = {c['id']: c['name'] for c in coco['categories']}
    ann_by_image = defaultdict(list)
    for ann in coco['annotations']:
        if ann['category_id'] not in categories:
            raise ValueError('Unknown source COCO category')
        if categories[ann['category_id']].lower() not in {'cornea', 'pupil', 'my-dataset-main'}:
            ann_by_image[ann['image_id']].append(ann)
    zip_path = base/'images/images.zip'
    zipped = {}
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.endswith('/'):
                    continue
                leaf = PurePosixPath(name).name
                if leaf in zipped:
                    raise ValueError(f'Ambiguous image basename in {zip_path}: {leaf}')
                zipped[leaf] = name
    rows = []
    for image in coco['images']:
        original = image.get('extra', {}).get('name') or image['file_name']
        match = re.match(r'(SE_\d+_(\d+)_S\d+)_(\d+)', original)
        if not match:
            raise ValueError(f'Cannot recover Global ID and frame index: {original}')
        video, global_id, number = match.group(1), match.group(2), int(match.group(3))
        row = _base(video, f'global_{global_id}', number, image['width'], image['height'],
                    'segmentation', fps)
        row.update(group_origin='dataset_global_operation_id', target_kind='instrument_box',
                   label_status='provided', source_image_id=image['id'])
        candidates = [base/'images'/image['file_name'], base/image['file_name']]
        path = next((p for p in candidates if p.is_file()), None)
        if path:
            row['path'] = str(path.resolve())
        elif image['file_name'] in zipped:
            row.update(archive=str(zip_path.resolve()), member=zipped[image['file_name']])
        else:
            raise FileNotFoundError(f'Missing image {image["file_name"]}')
        row['annotations'] = [dict(category_id=1, category_name='instrument',
            source_category=categories[a['category_id']], source_annotation_id=a['id'],
            bbox=box_xywh(a['bbox'], image['width'], image['height']),
            iscrowd=int(a.get('iscrowd', 0))) for a in ann_by_image[image['id']]]
        rows.append(row)
    # Already sparse annotated frames: never apply the dense tracking stride.
    return rows, dict(subset='segmentation', candidate_stride=1,
        grouping='global_operation_id_not_verified_patients',
        source_coco_sha256=hashlib.sha256(coco_path.read_bytes()).hexdigest(),
        target='all_instrument_boxes_single_class')


def assign_splits(rows: list[dict], seed: int, fractions: tuple[float, float, float],
                  split_map: Path | None) -> dict:
    if not rows:
        raise ValueError('No input frames')
    if len({r['frame_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate frame_id in candidate manifest')
    groups = sorted({r['group_id'] for r in rows})
    if split_map:
        raw = json.loads(split_map.read_text(encoding='utf-8'))
        mapping = raw.get('groups', raw)
        if any(mapping.get(g) not in {'train', 'val', 'test'} for g in groups):
            raise ValueError('Split map must assign train/val/test to every operation group')
        mapping = {g: mapping[g] for g in groups}
    else:
        if min(fractions) <= 0 or not math.isclose(sum(fractions), 1, abs_tol=1e-6):
            raise ValueError('Three positive split fractions must sum to one')
        if len(groups) < 3:
            raise ValueError('At least three groups required to create a split')
        order = sorted(groups, key=lambda g: digest([seed, g]))
        n_val = max(1, int(len(groups)*fractions[1]))
        n_test = max(1, int(len(groups)*fractions[2]))
        if n_val+n_test >= len(groups):
            raise ValueError('Split fractions leave no training groups')
        mapping = {g: ('val' if i < n_val else 'test' if i < n_val+n_test else 'train')
                   for i, g in enumerate(order)}
    for row in rows:
        row['split'] = mapping[row['group_id']]
    return dict(groups=mapping, seed=seed, unit='operation_group',
                patient_disjointness_verified=False)


def export_coco(rows: list[dict], destination: Path, source: ImageSource,
                accept_empty_tracking: bool = False) -> dict:
    """Export ALL target instances, images at native resolution, plus provenance."""
    unknown = [r['frame_id'] for r in rows if r['label_status'] == 'unknown']
    if unknown and not accept_empty_tracking:
        raise ValueError(f'{len(unknown)} frames have unverified empty labels. '
                         'Review them or explicitly use --accept-empty-tracking-labels; '
                         'use --no-export for annotation acquisition.')
    images, annotations = [], []
    for i, row in enumerate(rows, 1):
        leaf = f'{row["video_id"]}_{row["frame_index"]:08d}'
        suffix = Path(row.get('member') or row['path']).suffix.lower()
        relative = Path('images')/(leaf+suffix)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Every run owns a new output directory; source archives are never changed.
        if row.get('archive'):
            target.write_bytes(source.bytes(row))
        else:
            shutil.copyfile(row['path'], target)
        images.append(dict(id=i, file_name=relative.as_posix(), width=row['native_width'],
            height=row['native_height'], frame_id=row['frame_id'], video_id=row['video_id'],
            group_id=row['group_id'], label_status=row['label_status']))
        for ann in row['annotations']:
            annotations.append(dict(id=len(annotations)+1, image_id=i, category_id=1,
                bbox=ann['bbox'], area=ann['bbox'][2]*ann['bbox'][3],
                iscrowd=ann.get('iscrowd', 0), source_category=ann.get('source_category')))
    coco = dict(images=images, annotations=annotations,
        categories=[dict(id=1, name='instrument', supercategory='tool')],
        info=dict(target='instrument_bbox_not_tip_bbox',
                  area_semantics='bbox_width_times_height',
                  unverified_empty_labels_accepted=len(unknown)))
    write_json(destination/'annotations.coco.json', coco)
    return dict(images=len(images), annotations=len(annotations),
                empty_images=sum(not r['annotations'] for r in rows),
                unverified_empty_labels_accepted=len(unknown))
