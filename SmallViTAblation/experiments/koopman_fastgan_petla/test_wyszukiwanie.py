"""Testy CPU wyszukiwarki; bez modeli, pobierania, treningów i dostępu do GPU.

Uruchomienie: python -m unittest discover -s experiments/koopman_fastgan_petla
              -p test_wyszukiwanie.py -v
"""

from __future__ import annotations

import hashlib
import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))
try:
    import wyszukiwanie as w
except ModuleNotFoundError as exc:
    if exc.name != "wyszukiwanie":
        raise
    w = None


class ExactNumpyIndex:
    """Dokładny substytut FAISS wyłącznie do testów algorytmu overfetch."""

    def __init__(self, features):
        self.features = np.asarray(features, dtype=np.float32)
        self.ntotal, self.d = self.features.shape
        self.requests = []

    def search(self, query, k):
        self.requests.append(k)
        scores = np.asarray(query, dtype=np.float32) @ self.features.T
        # Remisy celowo odwrotnie niż docelowy ranking po identyfikatorze.
        order = np.argsort(-scores, axis=1, kind="stable")
        for i, row in enumerate(order):
            order[i] = sorted(row, key=lambda j: (-float(scores[i, j]), -int(j)))
        order = order[:, :k]
        return np.take_along_axis(scores, order, axis=1), order


class CpuTensor:
    """Minimalne tensory na granicy Torch; obliczenia testowe są rzeczywiste NumPy."""

    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, key):
        return CpuTensor(self.values[key])

    def mean(self, dim=None, axis=None):
        return CpuTensor(self.values.mean(axis=dim if dim is not None else axis))

    def to(self, device):
        if device != "cpu":
            raise AssertionError("Test nie może użyć GPU")
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def metadata(frame_id, group="g", split="train", annotation_index=0):
    return {
        "frame_id": frame_id, "group_id": group, "split": split,
        "path": f"/original/{frame_id}.png", "annotation_index": annotation_index,
        "bbox": [0, 0, 4, 2], "crop_xyxy": [-1, -2, 5, 4],
    }


class ModuleTestCase(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(w, "Brak implementacji wyszukiwanie.py")


class SearchTests(ModuleTestCase):
    def test_import_and_help_do_not_import_heavy_dependencies(self):
        code = """
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','transformers','faiss','PIL','huggingface_hub'}:
            raise AssertionError('Niedozwolony import: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['wyszukiwanie.py', '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=MODULE_DIR,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("build", result.stdout)
        self.assertIn("query", result.stdout)

    def test_cosine_is_normalized_and_preserves_negative_scores(self):
        scores = w.normalized_dot([[3, 4], [-3, -4], [0, 7]], [6, 8])
        np.testing.assert_allclose(scores, [1, -1, .8], atol=1e-6)
        np.testing.assert_allclose(w.normalize_features([[3e250, 4e250], [3e-250, 4e-250]]),
                                   [[.6, .8], [.6, .8]], atol=1e-6)

    def test_zero_nonfinite_and_dimension_mismatch_are_rejected(self):
        for bad in ([[0, 0]], [[np.nan, 1]], [[1, np.inf]], [[[1, 2]]], [1, 1j]):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                w.normalize_features(bad)
        with self.assertRaises(ValueError):
            w.normalized_dot([[1, 2]], [1, 2, 3])

    def test_pooling_excludes_cls_and_registers(self):
        tokens = np.array([[[100, 100], [200, 200], [1, 3], [5, 7]]], dtype=float)
        np.testing.assert_array_equal(w.pool_tokens(tokens, "meanpatch", 1), [[3, 5]])
        np.testing.assert_array_equal(w.pool_tokens(tokens, "cls", 1), [[100, 100]])
        with self.assertRaises(ValueError):
            w.pool_tokens(tokens[:, :2], "meanpatch", 1)

    def test_ranking_filters_used_and_nontrain_deduplicates_and_diversifies(self):
        rows = [metadata("used"), metadata("test", split="test"),
                metadata("a", "A"), metadata("a", "A", annotation_index=1),
                metadata("b", "A"), metadata("c", "B"), metadata("d", "C")]
        ranked = w.rank_candidates(rows, [1, .99, .98, .97, .96, .6, -.1],
                                   exclude_ids=["used"], top_k=4)
        self.assertEqual([r["frame_id"] for r in ranked], ["a", "c", "d", "b"])
        self.assertEqual([r["similarity_rank"] for r in ranked], [1, 3, 4, 2])
        self.assertEqual([r["diversity_round"] for r in ranked], [1, 1, 1, 2])
        self.assertEqual(ranked[0]["annotation_index"], 0)
        self.assertEqual(ranked[0]["path"], "/original/a.png")
        self.assertEqual(ranked[2]["score"], -.1)

    def test_ties_have_stable_frame_and_crop_order(self):
        rows = [metadata("z"), metadata("a", annotation_index=1),
                metadata("a", annotation_index=0)]
        a = w.rank_candidates(rows, [.5, .5, .5], top_k=2)
        b = w.rank_candidates(rows[::-1], [.5, .5, .5], top_k=2)
        self.assertEqual([(r["frame_id"], r["annotation_index"]) for r in a],
                         [("a", 0), ("z", 0)])
        self.assertEqual(a, b)

    def test_overfetch_reaches_eligible_tail_and_reports_exhaustion(self):
        rows = [metadata(str(i), str(i)) for i in range(12)]
        features = w.normalize_features([[20 - i, i + 1] for i in range(12)])
        index = ExactNumpyIndex(features)
        report = w.search_index(index, [1, 0], rows, exclude_ids=list(range(10)), top_k=5)
        self.assertEqual([r["frame_id"] for r in report["candidates"]], ["10", "11"])
        self.assertEqual(index.requests[-1], 12)
        self.assertTrue(report["exhausted"])
        self.assertEqual(report["top_selection"]["path"], "/original/10.png")
        self.assertEqual(report["selection_status"], "requires_review")

    def test_diversity_overfetches_for_low_scoring_group(self):
        rows = [metadata(str(i), "A") for i in range(9)] + [metadata("tail", "B")]
        index = ExactNumpyIndex(w.normalize_features([[20 - i, i + 1] for i in range(10)]))
        report = w.search_index(index, [1, 0], rows, top_k=3)
        self.assertEqual([r["frame_id"] for r in report["candidates"]], ["0", "tail", "1"])
        self.assertEqual(index.requests[-1], 10)

    def test_tie_at_fetch_boundary_cannot_hide_lexically_first_id(self):
        rows = [metadata(f"{i:02}") for i in range(10)]
        report = w.search_index(ExactNumpyIndex(np.tile([1, 0], (10, 1))), [1, 0], rows,
                                top_k=2)
        self.assertEqual([r["frame_id"] for r in report["candidates"]], ["00", "01"])
        self.assertEqual(report["retrieved_rows"], 10)

    def test_early_stop_matches_full_ranking(self):
        rows = [metadata(str(i), str(i % 2)) for i in range(20)]
        features = w.normalize_features([[25 - i, i + 1] for i in range(20)])
        index = ExactNumpyIndex(features)
        report = w.search_index(index, [1, 0], rows, top_k=2)
        self.assertEqual([r["frame_id"] for r in report["candidates"]], ["0", "1"])
        self.assertLess(report["retrieved_rows"], 20)

    def test_adaptive_fetch_matches_exhaustive_search_across_groups_and_exclusions(self):
        rng = np.random.default_rng(42)
        for diversify in (False, True):
            for _ in range(40):
                features = w.normalize_features(rng.normal(size=(40, 8)))
                query = w.normalize_features(rng.normal(size=8))
                rows = [metadata(f"{i // 2:02}", str((i // 2) % 4),
                                 "test" if i // 2 == 7 else "train", i % 2) for i in range(40)]
                excluded = rng.choice([f"{i:02}" for i in range(20)], size=8, replace=False).tolist()
                k = int(rng.integers(1, 16))
                expected = w.rank_candidates(rows, features @ query, exclude_ids=excluded,
                                             top_k=k, diversify_groups=diversify)
                result = w.search_index(ExactNumpyIndex(features), query, rows, exclude_ids=excluded,
                                        top_k=k, diversify_groups=diversify)["candidates"]
                self.assertEqual([r["frame_id"] for r in result], [r["frame_id"] for r in expected])
                self.assertEqual([r["annotation_index"] for r in result],
                                 [r["annotation_index"] for r in expected])

    def test_empty_eligibility_returns_no_selection_without_search(self):
        index = ExactNumpyIndex([[1, 0]])
        result = w.search_index(index, [1, 0], [metadata(1)], exclude_ids=["1"])
        self.assertEqual(result["candidates"], [])
        self.assertIsNone(result["top_selection"])
        self.assertEqual(result["selection_status"], "no_candidates")
        self.assertEqual(index.requests, [])
        with self.assertRaises(ValueError):
            w.search_index(index, [1, 0], [metadata(1)], top_k=0)

    def test_exclusions_accept_json_list_or_file_and_reject_ambiguous_types(self):
        self.assertEqual(w.load_exclusions('[1, "a", 1]'), {"1", "a"})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "exclude.json"
            path.write_text('["used"]', encoding="utf-8")
            self.assertEqual(w.load_exclusions(str(path)), {"used"})
        for invalid in ('{"a": 1}', '[true]', '[1.5]', '[null]', '[""]'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                w.load_exclusions(invalid)


class FilesTestCase(ModuleTestCase):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_manifest(self, rows):
        path = self.root / "pool.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def row(self, frame_id, split="train", group=None, path=None):
        return {"frame_id": frame_id, "path": path or f"{frame_id}.png", "split": split,
                "group_id": group or f"g-{frame_id}", "annotations": [{"bbox": [0, 1, 4, 2]}]}


class ManifestCropTests(FilesTestCase):
    def test_only_train_is_prepared_and_nontrain_files_need_not_exist(self):
        from PIL import Image
        Image.new("RGB", (6, 4), (30, 50, 70)).save(self.root / "a.png")
        path = self.write_manifest([self.row("a"), self.row("v", "val"), self.row("t", "test")])
        rows = w.read_manifest(path, self.root)
        prepared = list(w.iter_crops(rows, margin=.5))
        self.assertEqual(len(prepared), 1)
        crop, meta = prepared[0]
        self.assertEqual(crop.size, (8, 8))
        self.assertEqual(meta["crop_xyxy"], [-2, -2, 6, 6])
        self.assertEqual(meta["padding_ltrb"], [2, 2, 0, 2])
        self.assertEqual(meta["path"], str((self.root / "a.png").resolve()))
        self.assertEqual(meta["image_size"], [6, 4])
        self.assertEqual(crop.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(crop.getpixel((2, 2)), (30, 50, 70))

    def test_manifest_rejects_duplicate_ids_paths_and_group_split_leakage(self):
        cases = [
            [self.row(1), self.row("1")],
            [self.row("a"), self.row("b", path="a.png")],
            [self.row("a", group="shared"), self.row("b", "test", group="shared")],
            [self.row("a", group="shared"), self.row("b", "val", group="shared")],
            [self.row("a", "trian")],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                w.read_manifest(self.write_manifest(rows), self.root)

    def test_invalid_bbox_is_rejected_instead_of_silently_cropped(self):
        for bbox in ([1, 2, 0, 3], [1, 2, -1, 3], [1, 2, np.nan, 3], [1, 2, 3]):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                w.square_crop_box(bbox, .1)
        with self.assertRaises(ValueError):
            w.square_crop_box([0, 0, 4, 2], -1)

    def test_fractional_bbox_is_fully_contained_in_square(self):
        x0, y0, x1, y1 = w.square_crop_box([.9, 1.2, 4, 1], 0)
        self.assertEqual(x1 - x0, y1 - y0)
        self.assertLessEqual(x0, .9)
        self.assertGreaterEqual(x1, 4.9)

    def test_query_padding_preserves_pixels_aspect_and_orientation(self):
        from PIL import Image
        source = Image.new("RGB", (4, 2), "red")
        source.putpixel((3, 1), (0, 0, 255))
        square, box = w.prepare_query_image(source)
        self.assertEqual(square.size, (4, 4))
        self.assertEqual(box, [0, -1, 4, 3])
        self.assertEqual(square.getpixel((3, 2)), (0, 0, 255))
        self.assertEqual(square.getpixel((0, 0)), (0, 0, 0))

    def test_hub_preprocessing_has_imagenet_values_and_does_not_rotate(self):
        from PIL import Image
        image = Image.new("RGB", (256, 256), (255, 0, 0))
        image.paste((0, 0, 255), (128, 0, 256, 256))
        values = w.hub_preprocess(image)
        self.assertEqual(values.shape, (3, 224, 224))
        np.testing.assert_allclose(values[:, 40, 40], [(.999999999 - .485) / .229,
                                   -.456 / .224, -.406 / .225], atol=1e-6)
        np.testing.assert_allclose(values[:, 40, 180], [-.485 / .229,
                                   -.456 / .224, (1 - .406) / .225], atol=1e-6)

    def test_model_modes_require_local_files_and_paired_hub_arguments(self):
        for args in ({}, {"torch_hub_dir": self.root}, {"weights": self.root / "w.pth"},
                     {"model_dir": self.root, "torch_hub_dir": self.root,
                      "weights": self.root / "w.pth"}):
            with self.subTest(args=args), self.assertRaises((ValueError, FileNotFoundError)):
                w.resolve_model_source(**args)
        repo = self.root / "hub"
        repo.mkdir()
        (repo / "hubconf.py").write_text("# lokalny fixture", encoding="utf-8")
        weights = self.root / "weights.pth"
        weights.write_bytes(b"wagi testowe, nigdy nie ladowane")
        source = w.resolve_model_source(torch_hub_dir=repo, weights=weights)
        self.assertEqual(source["backend"], "torch_hub")
        self.assertEqual(source["weights_sha256"], hashlib.sha256(weights.read_bytes()).hexdigest())

    def test_cli_accepts_both_local_model_modes_and_query_overrides(self):
        parser = w.make_parser()
        args = parser.parse_args(["build", "--manifest", "p.jsonl", "--data-root", ".",
                                  "--out", "idx", "--torch-hub-dir", "hub",
                                  "--weights", "w.pth", "--device", "cpu"])
        self.assertEqual(args.torch_hub_dir, "hub")
        args = parser.parse_args(["query", "--index", "idx", "--image", "q.png",
                                  "--exclude-json", '["used"]', "--out", "q.json",
                                  "--top-k", "20", "--torch-hub-dir", "hub", "--weights", "w.pth"])
        self.assertEqual(args.top_k, 20)
        args = parser.parse_args(["build", "--manifest", "p.jsonl", "--data-root", ".",
                                  "--out", "idx", "--model-dir", "cached-model", "--device", "cpu"])
        self.assertEqual(args.model_dir, "cached-model")

    def test_hub_encoder_is_local_frozen_and_uses_patch_mean_or_cls(self):
        from PIL import Image
        calls = {}
        model = SimpleNamespace(
            load_state_dict=lambda state, strict: calls.update(state=state, strict=strict),
            requires_grad_=lambda value: calls.update(requires_grad=value),
            eval=lambda: calls.update(eval=True),
            to=lambda device: calls.update(device=device),
            forward_features=lambda tensor: {
                "x_norm_clstoken": CpuTensor([[0, 5]]),
                "x_norm_patchtokens": CpuTensor([[[2, 4], [4, 4]]]),
            })

        def local_hub(repo, entrypoint, *, source, pretrained):
            if (repo, entrypoint, source, pretrained) != ("local-hub", "dinov2_vits14", "local", False):
                raise AssertionError("Próba użycia innego modelu lub pobierania")
            return model

        def local_weights(path, *, map_location, weights_only):
            if (path, map_location, weights_only) != ("local.pth", "cpu", True):
                raise AssertionError("Wagi wymagają bezpiecznego, lokalnego odczytu CPU")
            return {"fixture": 123}

        torch = SimpleNamespace(hub=SimpleNamespace(load=local_hub), load=local_weights,
                                from_numpy=CpuTensor, inference_mode=nullcontext)
        with patch.dict(sys.modules, {"torch": torch, "transformers": None}):
            source = {"backend": "torch_hub", "torch_hub_dir": "local-hub", "weights": "local.pth"}
            encoder = w.DinoEncoder(source, device="cpu", pooling="meanpatch")
            np.testing.assert_allclose(encoder.encode([Image.new("RGB", (8, 8))]), [[.6, .8]], atol=1e-6)
            cls_encoder = w.DinoEncoder(source, device="cpu", pooling="cls")
            np.testing.assert_array_equal(cls_encoder.encode([Image.new("RGB", (8, 8))]), [[0, 1]])
        self.assertFalse(calls["requires_grad"])
        self.assertTrue(calls["eval"])
        self.assertTrue(calls["strict"])
        self.assertEqual(calls["state"], {"fixture": 123})
        self.assertEqual(calls["device"], "cpu")


@unittest.skipUnless(importlib.util.find_spec("faiss"), "Opcjonalny test rzeczywistego FAISS CPU")
class BundleTests(FilesTestCase):
    def test_bundle_roundtrip_hashes_and_corruption_detection(self):
        manifest = self.write_manifest([self.row("a"), self.row("b")])
        output = self.root / "index"
        w.save_index(output, [[3, 4], [0, 5]], [metadata("a", "A"), metadata("b", "B")],
                     {"pooling": "meanpatch", "model_dir": "unused-in-numerical-test"}, manifest)
        index, features, rows, config = w.load_index(output)
        np.testing.assert_allclose(features, [[.6, .8], [0, 1]], atol=1e-6)
        result = w.search_index(index, [1, 0], rows, top_k=2)
        self.assertEqual(result["top_selection"]["frame_id"], "a")
        self.assertEqual(config["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        hashes = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
        self.assertEqual(hashes["config.json"], hashlib.sha256((output / "config.json").read_bytes()).hexdigest())
        (output / "metadata.jsonl").write_text('{}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            w.load_index(output)

    def test_save_rejects_nontrain_misalignment_and_nonempty_output(self):
        manifest = self.write_manifest([self.row("a")])
        with self.assertRaises(ValueError):
            w.save_index(self.root / "bad", [[1, 0]], [metadata("a", split="test")], {}, manifest)
        with self.assertRaises(ValueError):
            w.save_index(self.root / "bad2", [[1, 0], [0, 1]], [metadata("a")], {}, manifest)
        output = self.root / "occupied"
        output.mkdir()
        (output / "foreign.txt").write_text("zachowaj", encoding="utf-8")
        with self.assertRaises((ValueError, FileExistsError)):
            w.save_index(output, [[1, 0]], [metadata("a")], {}, manifest)
        self.assertEqual((output / "foreign.txt").read_text(encoding="utf-8"), "zachowaj")

    def test_build_query_roundtrip_and_changed_weights_are_rejected(self):
        from PIL import Image

        class ColorEncoder:
            preprocessing = dict(w.HUB_PREPROCESS)

            def __init__(self, source, *, device, pooling):
                if device != "cpu":
                    raise AssertionError("Wyłącznie CPU")

            def encode(self, images):
                return np.array([[1, 0] if image.getpixel((image.width // 2, image.height // 2))[0] else [0, 1]
                                 for image in images], dtype=np.float32)

        Image.new("RGB", (6, 4), "red").save(self.root / "a.png")
        Image.new("RGB", (6, 4), "blue").save(self.root / "b.png")
        Image.new("RGB", (5, 3), "red").save(self.root / "query.png")
        a = self.row("a")
        a["annotations"].append({"bbox": [1, 0, 3, 3]})
        manifest = self.write_manifest([a, self.row("b"), self.row("t", "test")])
        repo = self.root / "hub"
        repo.mkdir()
        (repo / "hubconf.py").write_text("# test", encoding="utf-8")
        weights = self.root / "weights.pth"
        weights.write_bytes(b"fake weights")
        output = self.root / "indeks-zażółć"
        with patch.object(w, "DinoEncoder", ColorEncoder):
            config = w.build_index(manifest, self.root, output, torch_hub_dir=repo,
                                   weights=weights, device="cpu", batch_size=2)
            self.assertEqual(config["indexed_frames"], 2)
            self.assertEqual(config["feature_count"], 3)
            report = w.query_index(output, self.root / "query.png", '["a"]', self.root / "query.json")
            self.assertEqual(report["top_selection"]["path"], str((self.root / "b.png").resolve()))
            self.assertEqual(report["top_selection"]["score"], 0)
            self.assertEqual(report["config_sha256"], hashlib.sha256((output / "config.json").read_bytes()).hexdigest())
            saved = json.loads((self.root / "query.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            own_query = w.query_index(output, self.root / "a.png")
            self.assertEqual([r["frame_id"] for r in own_query["candidates"]], ["b"])
            weights.write_bytes(b"different weights")
            with self.assertRaises(ValueError):
                w.query_index(output, self.root / "query.png")

    def test_known_test_image_cannot_drive_selection(self):
        from PIL import Image
        Image.new("RGB", (3, 3)).save(self.root / "test.png")
        manifest = self.write_manifest([self.row("a"), self.row("test", "test")])
        output = self.root / "index"
        w.save_index(output, [[1, 0]], [metadata("a")],
                     {"data_root": str(self.root), "model_source": {}}, manifest)
        with self.assertRaisesRegex(ValueError, "val|test"):
            w.query_index(output, self.root / "test.png")


if __name__ == "__main__":
    unittest.main()
