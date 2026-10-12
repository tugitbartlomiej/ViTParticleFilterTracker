"""Contract tests. Execute on Eden only; local execution deliberately deferred."""
import numpy as np
import pytest
from ..dane import split_operations, balanced_sample, validate_partitions
from ..sampler import AdaptiveSampler


def rows():
    return [dict(frame_id=f'{g}/{i}', group_id=str(g), split='train')
            for g in range(10) for i in range(10)]


def test_diagnostic_operations_never_enter_fit():
    fit, diagnostic = split_operations(rows(), diagnostic_groups=3, seed=42)
    assert len({r['group_id'] for r in diagnostic}) == 3
    assert not ({r['group_id'] for r in fit} & {r['group_id'] for r in diagnostic})
    assert split_operations(rows()[::-1], 3, 42) == (fit, diagnostic)


def test_partition_validator_rejects_cross_split_operation():
    fit, diagnostic = split_operations(rows(), 3, 42)
    val = [dict(fit[0], frame_id='different_image', split='val')]
    with pytest.raises(ValueError, match='group'):
        validate_partitions(fit, diagnostic, val)


def test_balanced_sample_retains_groups_and_is_not_input_order_dependent():
    a = balanced_sample(rows(), 23, 9)
    b = balanced_sample(rows()[::-1], 23, 9)
    assert a == b and len(a) == len(set(a)) == 23
    assert len({x.split('/')[0] for x in a}) == 10


def test_partition_validator_rejects_reused_ids_in_different_groups():
    fit, diagnostic = split_operations(rows(), 3, 42)
    diagnostic[0] = dict(diagnostic[0], frame_id=fit[0]['frame_id'])
    val = [dict(frame_id='val_unique', group_id='val_unique', split='val')]
    with pytest.raises(ValueError, match='frame_id'):
        validate_partitions(fit, diagnostic, val)


def test_sampler_changes_indices_without_changing_epoch_length():
    sampler = AdaptiveSampler([0, 1, 2], batch_size=2, seed=4)
    assert len(sampler) == 4
    first = list(sampler)
    assert set(first) == {0, 1, 2}
    sampler.update([10, 11, 12], epoch=1)
    assert set(sampler) == {10, 11, 12} and len(sampler) == 4
    with pytest.raises(ValueError):
        sampler.update([1, 1, 2], epoch=2)
    with pytest.raises(ValueError):
        sampler.update([1, 2], epoch=2)


def test_no_adaptive_selection_uses_official_val():
    from ..trenuj import selection_rows
    fit, diagnostic = split_operations(rows(), 3, 42)
    assert selection_rows(fit, diagnostic) == fit + diagnostic
    with pytest.raises(ValueError):
        selection_rows(fit, [dict(diagnostic[0], split='val')])
