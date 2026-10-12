import numpy as np
from ..trenuj import placebo_gain


def test_placebo_never_moves_signal_across_operations_or_missing_history():
    values = np.arange(40, dtype=float)
    valid = np.ones(40, dtype=bool)
    valid[[0, 20]] = False
    groups = ['a']*20 + ['b']*20
    loss = np.ones(40)
    shuffled, strata = placebo_gain(values, valid, groups, loss, 42)
    assert strata == 2
    assert shuffled[0] == shuffled[20] == 0.
    assert set(shuffled[1:20]) == set(values[1:20])
    assert set(shuffled[21:]) == set(values[21:])
