"""Read-only validation of downloaded checkpoint metadata; no model forward."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch


def main():
    audit_root = Path(__file__).resolve().parent
    weight_root = audit_root.parent / "wagi"
    manifest = json.loads((audit_root / "MANIFEST_ZDALNY_2026-10-01.json").read_text(encoding="utf-8"))
    expected_names = [f"k{step:04d}.pt" for step in range(0, 1001, 10)]
    assert [entry["name"] for entry in manifest["files"]] == expected_names
    reference_schema = None
    records = []
    torch.set_num_threads(1)
    for entry in manifest["files"]:
        path = weight_root / entry["name"]
        assert path.stat().st_size == entry["bytes"], path.name
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert isinstance(checkpoint, dict) and set(checkpoint) == {"net", "krok"}, path.name
        assert isinstance(checkpoint["krok"], int) and checkpoint["krok"] == entry["step"], path.name
        state = checkpoint["net"]
        assert isinstance(state, dict) and state, path.name
        assert all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()), path.name
        schema = {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in sorted(state.items())}
        if reference_schema is None:
            reference_schema = schema
        assert schema == reference_schema, path.name
        bn_means = sorted(key for key in state if key.endswith("running_mean"))
        bn_variances = sorted(key for key in state if key.endswith("running_var"))
        bn_counts = sorted(key for key in state if key.endswith("num_batches_tracked"))
        assert bn_means and len(bn_means) == len(bn_variances) == len(bn_counts), path.name
        records.append({"name": path.name, "filename_step": entry["step"], "stored_krok": checkpoint["krok"], "top_level_keys": sorted(checkpoint), "state_tensors": len(state), "bn_running_mean_count": len(bn_means), "bn_running_variance_count": len(bn_variances), "bn_counter_count": len(bn_counts), "schema_matches": True})
        del checkpoint, state
    schema_json = json.dumps(reference_schema, sort_keys=True, separators=(",", ":"))
    result = {"status": "metadata_and_schema_verified", "verified_at_utc": datetime.now(timezone.utc).isoformat(), "torch_version": torch.__version__, "load_mode": "weights_only=True, map_location=cpu", "count": len(records), "all_stored_steps_match_filenames": True, "all_state_schemas_match": True, "network_forward_executed": False, "latents_reconstructed": False, "tensor_values_numerically_audited": False, "state_schema_sha256": hashlib.sha256(schema_json.encode()).hexdigest(), "reference_schema": reference_schema, "files": records}
    out = audit_root / "WERYFIKACJA_STRUKTURY_2026-10-01.json"
    if out.exists():
        raise RuntimeError("Existing structure audit preserved; do not overwrite.")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "count", "all_stored_steps_match_filenames", "all_state_schemas_match", "network_forward_executed", "state_schema_sha256")}))


if __name__ == "__main__":
    main()
