"""Eden-only two-epoch integration of real RF-DETR, selection and optimizer continuity."""
import argparse
import json
import os
from pathlib import Path
import math
from .dane import write_json, read_jsonl
from .przygotuj import prepare
from .trenuj import run
from .gates import source_hashes


def integration(data_root, context_root, out, initial_weights):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('integration is authorized on Eden through Slurm only')
    import torch
    from .preflight import check
    check(initial_weights)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    config.update(integration_fixture=True, fit_cap=96, val_cap=24, epochs=2, budget=32,
                  batch_size=4, diagnostic_groups=6, diagnostic_frames=24, reference_frames=6,
                  support_frames=24, candidate_limit=48, glister_rounds=2, num_workers=2,
                  min_support=3, min_groups=2, gle_epochs=1, gle_max_windows=256,
                  require_history_fraction=.1)
    write_json(out/'fixture_config.json', config)
    protocol = out/'protocol'
    prepare(data_root, context_root, protocol, out/'fixture_config.json')
    # Exercise baseline and full scientific path with the same warm-up and fresh weights.
    for arm in ('random', 'features_kg_glister'):
        ledger = run(protocol, out/arm, initial_weights, arm, 42)
        expected = math.ceil(config['budget']/config['batch_size'])
        assert [r['optimizer_steps'] for r in ledger] == [expected, 2*expected], 'optimizer/global step restarted'
        assert [r['trained_exposures_total'] for r in ledger] == [config['budget'], 2*config['budget']]
        fit_ids = {r['frame_id'] for r in read_jsonl(protocol/'fit.jsonl')}
        steps = []
        for epoch in (1, 2):
            round_dir = out/arm/'rounds'/f'epoch_{epoch:03d}'
            trained = json.loads((round_dir/'trained_frame_ids.json').read_text(encoding='utf-8'))
            assert set(trained) <= fit_ids, 'D/val entered training'
            state = torch.load(round_dir/'training_state.ckpt', map_location='cpu', weights_only=False)
            assert state['optimizer_states'], 'missing optimizer state'
            steps.append(max(float(v['step']) for v in state['optimizer_states'][0]['state'].values() if 'step' in v))
        assert steps[1] > steps[0] > 0, 'AdamW state was reset'
        first = torch.load(out/arm/'rounds/epoch_001/weights.pth', map_location='cpu', weights_only=False)['model']
        second = torch.load(out/arm/'rounds/epoch_002/weights.pth', map_location='cpu', weights_only=False)['model']
        assert any(not torch.equal(first[k], second[k]) for k in first), 'second epoch changed no parameters'
    a = json.loads((out/'random/initial_frame_ids.json').read_text(encoding='utf-8'))
    b = json.loads((out/'features_kg_glister/initial_frame_ids.json').read_text(encoding='utf-8'))
    assert a == b, 'arms have different warm-up'
    assert (out/'features_kg_glister/rounds/epoch_001/dynamics/audit.json').is_file()
    assert (out/'features_kg_glister/rounds/epoch_001/next_frame_ids.json').is_file()
    write_json(out/'INTEGRATION_PASSED.json', dict(status='passed', epochs=2, arms=['random', 'features_kg_glister'],
               code_sha256=source_hashes(),
               input_archives_sha256={k: v['sha256'] for k, v in json.loads(Path(os.environ['INPUT_PROVENANCE']).read_text()).items()},
               scientific_conclusion=False, note='Execution contract only; not an AP efficacy experiment.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'context-root', 'out', 'initial-weights'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    integration(a.data_root, a.context_root, a.out, a.initial_weights)


if __name__ == '__main__':
    main()
