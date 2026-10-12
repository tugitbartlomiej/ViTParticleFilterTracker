"""Single uninterrupted adaptive RF-DETR run. Invoke on Eden through Slurm."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from .dane import ARMS, read_jsonl, write_json, write_jsonl, sha256, balanced_sample, validate_partitions
from .sampler import AdaptiveSampler


def selection_rows(fit, diagnostic):
    if any(r['split'] != 'train' for r in fit+diagnostic):
        raise ValueError('selection sees source train only')
    if {str(r['group_id']) for r in fit} & {str(r['group_id']) for r in diagnostic}:
        raise ValueError('diagnostic groups cannot enter FIT')
    return fit+diagnostic


def placebo_gain(gain, valid, groups, losses, seed):
    """Shuffle within operation and observed-token BCE tertiles, never across missing history."""
    result = np.asarray(gain).copy()
    rng = np.random.default_rng(seed)
    groups, losses, valid = np.asarray(groups), np.asarray(losses), np.asarray(valid)
    shuffled_strata = 0
    for group in sorted(set(groups)):
        ids = np.flatnonzero((groups == group) & valid & np.isfinite(losses))
        if len(ids) < 2:
            continue
        bins = np.searchsorted(np.quantile(losses[ids], [1/3, 2/3]), losses[ids], side='right')
        for b in range(3):
            part = ids[bins == b]
            if len(part) > 1:
                result[part] = rng.permutation(result[part])
                shuffled_strata += 1
    result[~valid] = 0.
    return result, shuffled_strata


class Controller:
    def __init__(self, protocol, out, config, arm, fit, diagnostic, sampler):
        self.protocol, self.out, self.config, self.arm = Path(protocol), Path(out), config, arm
        self.fit, self.diagnostic, self.sampler = fit, diagnostic, sampler
        self.pool = selection_rows(fit, diagnostic)
        self.meta = json.loads((self.protocol/'protocol.json').read_text(encoding='utf-8'))
        self.features = np.load(self.protocol/'features.npy')
        self.names = self.meta['feature_names']
        self.by_id = {r['frame_id']: r for r in self.pool}
        self.fit_index = {r['frame_id']: i for i, r in enumerate(fit)}
        self.support = [self.by_id[f] for f in self.meta['support_ids']]
        self.reference = [self.by_id[f] for f in self.meta['reference_ids']]
        self.seen = set()
        self.exposures = 0
        self.started = time.monotonic()
        self.ledger = []
        self.observed_indices = []

    def choose(self, backend, epoch, out, task_losses):
        from .cechy import spatial_descriptor
        from .diagnoza import fit_probes, predict_probes, discover_needs
        from .selekcja import retrieve_candidates
        from .dynamika import compute_dynamics
        c = self.config
        seed = c['seed'] + 1009*(epoch+1)
        n_fit = len(self.fit)
        needs, reasons = [], []
        if self.arm == 'random':
            ids = balanced_sample(self.fit, c['budget'], seed)
            return ids, dict(needs=[], candidate_count=0, strategy='operation-balanced random')
        if self.arm == 'glister':
            ids = balanced_sample(self.fit, c['candidate_limit'], seed)
            candidates = np.array([self.fit_index[f] for f in ids])
            reasons = [dict(frame_id=f, selection_reason='operation_balanced_random_prefilter', need_ids=[]) for f in ids]
        else:
            use_dynamics = self.arm in ('features_kg_glister', 'features_placebo_glister')
            latent_rows = read_jsonl(self.protocol/'context.jsonl') if use_dynamics else self.support+self.diagnostic
            tokens, inputs, grid = backend.extract(latent_rows, out/'latent')
            lookup = {r['frame_id']: i for i, r in enumerate(latent_rows)}
            descriptors = np.stack([spatial_descriptor(tokens[lookup[r['frame_id']]], r, grid)
                                    for r in self.support+self.diagnostic])
            support_index = [self.fit_index[r['frame_id']] for r in self.support]
            probes = fit_probes(descriptors[:len(self.support)], self.features[support_index])
            np.savez(out/'probes.npz', **probes)
            prediction = predict_probes(probes, descriptors[len(self.support):])
            with np.errstate(invalid='ignore', divide='ignore'):
                latent_errors = ((prediction-self.features[n_fit:])/probes['feature_scale'])**2
            gain = None
            if use_dynamics:
                result = compute_dynamics(latent_rows, tokens, inputs, self.pool,
                    self.meta['support_ids'], self.meta['excluded_fit_groups'], out/'dynamics',
                    grid_shape=grid, seed=seed, device=str(backend.device), epochs=c['gle_epochs'],
                    max_windows=c['gle_max_windows'], context=c['gle_context'], step=c['context_step'], folds=c['gle_folds'])
                valid = result['valid_history']
                # Missing history cannot silently turn a KG experiment into a static experiment.
                if np.mean(valid[n_fit:]) < c['require_history_fraction']:
                    raise RuntimeError('too few diagnostic frames have valid causal history; inspect context archive')
                positive = result['positive_gain']
                shuffled = 0
                if self.arm == 'features_placebo_glister':
                    positive, shuffled = placebo_gain(positive, valid,
                        [str(r['group_id']) for r in self.pool], result['e_obs'], seed)
                gain = positive[n_fit:]
                write_json(out/'dynamics_summary.json', dict(valid_fit_fraction=float(np.mean(valid[:n_fit])),
                           valid_diagnostic_fraction=float(np.mean(valid[n_fit:])), placebo_strata=shuffled,
                           score='max(observed token readout BCE - predicted token readout BCE, 0)',
                           detector_training_improvement_measured=False))
            needs = discover_needs(self.features[:n_fit], self.features[n_fit:], task_losses,
                       [str(r['group_id']) for r in self.diagnostic], self.names,
                       latent_errors=latent_errors, dynamics_gain=gain, min_support=c['min_support'],
                       min_groups=c['min_groups'], max_needs=c['max_needs'])
            np.savez(out/'diagnostic_errors.npz', task_loss=task_losses, latent_errors=latent_errors,
                     feature_predictions=prediction, frame_ids=np.array([r['frame_id'] for r in self.diagnostic]))
            candidates, reasons = retrieve_candidates(self.features[:n_fit], self.names, needs,
                [r['frame_id'] for r in self.fit], limit=c['candidate_limit'],
                exploration_fraction=c['exploration_fraction'], seed=seed)
            # Only the newly created scratch array is removed, never source/context data.
            del tokens
            if not c['keep_tokens']:
                (out/'latent'/'tokens.npy').unlink()
        write_json(out/'needs.json', needs)
        write_jsonl(out/'candidates.jsonl', reasons)
        local = backend.select([self.fit[i] for i in candidates], self.reference, c['budget'], seed, out)
        ids = [self.fit[int(candidates[i])]['frame_id'] for i in local]
        write_jsonl(out/'chosen_reasons.jsonl', [reasons[i] for i in local])
        return ids, dict(needs=needs, candidate_count=len(candidates), strategy=self.arm)

    def completed_epoch(self, trainer, module):
        import torch
        from .rfdetr_backend import DetectorBackend, diagnostic_transaction
        epoch = int(trainer.current_epoch)
        out = self.out/'rounds'/f'epoch_{epoch+1:03d}'
        out.mkdir(parents=True, exist_ok=False)
        consumed = list(self.sampler)
        if self.observed_indices != consumed:
            raise RuntimeError('actual loader exposure differs from selected sampler order')
        consumed_ids = [self.fit[i]['frame_id'] for i in consumed]
        self.seen.update(consumed_ids)
        self.exposures += len(consumed)
        write_json(out/'trained_frame_ids.json', consumed_ids)
        torch.save(dict(model={k: v.detach().cpu() for k, v in module.model.state_dict().items()},
                        epoch=epoch, global_step=trainer.global_step,
                        model_config=module.model_config.model_dump(), selection_arm=self.arm), out/'weights.pth')
        # This is archival state; automatic resume is deliberately unsupported in v1.
        trainer.save_checkpoint(str(out/'training_state.ckpt'))
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if hasattr(value, 'numel') and value.numel() == 1 and bool(torch.isfinite(value)):
                metrics[str(key)] = float(value.detach().cpu())
        if not 0 <= metrics.get('val/mAP_50_95', -1) <= 1:
            raise RuntimeError('independent validation did not produce a valid val/mAP_50_95')
        selection_start = time.monotonic()
        metric_elapsed = selection_start-self.started
        info = dict(strategy='training_complete', candidate_count=0, needs=[])
        with diagnostic_transaction(module):
            backend = DetectorBackend(module, self.config)
            task_losses = backend.losses(self.diagnostic)
            from .raport import condition_report
            write_json(out/'diagnostic_conditions.json', condition_report(
                self.features[:len(self.fit)], self.features[len(self.fit):], task_losses,
                [str(r['group_id']) for r in self.diagnostic], self.names))
            if epoch+1 < self.config['epochs']:
                ids, info = self.choose(backend, epoch, out, task_losses)
        if epoch+1 < self.config['epochs']:
            self.sampler.update([self.fit_index[f] for f in ids], epoch=epoch+1)
            write_json(out/'next_frame_ids.json', ids)
        entry = dict(epoch=epoch+1, optimizer_steps=int(trainer.global_step),
                     trained_unique_total=len(self.seen), trained_exposures_total=self.exposures,
                     epoch_selected_unique=len(set(consumed_ids)), epoch_exposures=len(consumed_ids),
                     elapsed_seconds=time.monotonic()-self.started,
                     metric_elapsed_seconds=metric_elapsed,
                     selection_seconds=time.monotonic()-selection_start, metrics=metrics,
                     selection=info, weights_sha256=sha256(out/'weights.pth'))
        write_json(out/'summary.json', entry)
        self.ledger.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != 'selection'}), flush=True)


def run(protocol, out, initial_weights, arm, seed, *, config_override=None):
    from .rfdetr_backend import build_training
    from pytorch_lightning import Callback
    protocol, out, initial_weights = map(Path, (protocol, out, initial_weights))
    config = json.loads((protocol/'config.json').read_text(encoding='utf-8'))
    if config_override:
        # Used only by Eden integration; campaign configuration is immutable.
        config.update(config_override)
    config['seed'] = seed
    if arm not in ARMS:
        raise ValueError('unknown arm')
    fit, diagnostic, val = [read_jsonl(protocol/f'{k}.jsonl') for k in ('fit', 'diagnostic', 'val')]
    validate_partitions(fit, diagnostic, val)
    meta = json.loads((protocol/'protocol.json').read_text(encoding='utf-8'))
    for name, digest in meta['files_sha256'].items():
        if sha256(protocol/name) != digest:
            raise ValueError(f'modified protocol artifact: {name}')
    # Common warm-up: support + random remainder; identical between arms for a seed.
    support = meta['support_ids']
    rest = [r for r in fit if r['frame_id'] not in set(support)]
    initial = list(support) + (balanced_sample(rest, config['budget']-len(support), seed) if config['budget'] > len(support) else [])
    index = {r['frame_id']: i for i, r in enumerate(fit)}
    sampler = AdaptiveSampler([index[f] for f in initial], config['batch_size'], seed)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'initial_frame_ids.json', initial)
    write_json(out/'run_config.json', dict(**config, arm=arm, initial_weights_sha256=sha256(initial_weights),
               protocol_sha256=sha256(protocol/'protocol.json'),
               assignment_sha256=meta['assignment_sha256'],
               features_sha256=meta['files_sha256']['features.npy'],
               protocol_config_sha256=meta['config_sha256'],
               input_archives=json.loads(Path(os.environ['INPUT_PROVENANCE']).read_text()) if os.environ.get('INPUT_PROVENANCE') else None,
               code_sha256={str(p.relative_to(Path(__file__).parent)): sha256(p)
                            for p in Path(__file__).parent.rglob('*.py')},
               local_tests_executed=False, resume_supported=False))
    controller = Controller(protocol, out, config, arm, fit, diagnostic, sampler)

    class SelectionCallback(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            controller.observed_indices = []

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            _, targets = batch
            controller.observed_indices.extend(int(t['image_id'].item())-1 for t in targets)

        def on_train_epoch_end(self, trainer, pl_module):
            controller.completed_epoch(trainer, pl_module)

    module, datamodule, trainer = build_training(protocol, out, initial_weights, config, sampler, fit)
    trainer.callbacks.insert(0, SelectionCallback())
    trainer.fit(module, datamodule=datamodule)
    write_json(out/'completed.json', dict(arm=arm, seed=seed, epochs=len(controller.ledger),
               optimizer_steps=int(trainer.global_step), elapsed_seconds=time.monotonic()-controller.started,
               trained_exposures=controller.exposures, trained_unique=len(controller.seen),
               status='completed', test_used=False, rounds=controller.ledger))
    return controller.ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'out', 'initial-weights'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    run(a.protocol, a.out, a.initial_weights, a.arm, a.seed)


if __name__ == '__main__':
    main()
