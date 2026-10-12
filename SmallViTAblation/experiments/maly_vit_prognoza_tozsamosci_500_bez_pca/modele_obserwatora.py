"""Causal observers in the unchanged 128 native channels.

An observer estimates velocity through the origin. Forecasts thereafter use no
measurements. Position is reset to the observed origin, so smoothing of position
is never confused with forecasting the displacement from the actual origin.
"""
import numpy as np


def configs():
    out = [dict(id='persistence', family='persistence', kind='persistence'),
           dict(id='velocity', family='velocity', kind='velocity', tau=None)]
    for tau in (10, 50, 200, 1000):
        out.append(dict(id=f'damped_velocity_t{tau}', family='damped_velocity',
                        kind='velocity', tau=tau))
    for window in (16, 64, 128):
        for tau in (50, 200, None):
            out.append(dict(id=f'ols_w{window}_t{tau}', family='ols_trend',
                            kind='ols', window=window, tau=tau))
    for window in (50, 200):
        for tau in (50, 200):
            out.append(dict(id=f'reversion_w{window}_t{tau}', family='reversion',
                            kind='reversion', window=window, tau=tau))
    for ratio in (0.0001, 0.01, 1.0):
        for tau in (20, 100, 500, None):
            out.append(dict(id=f'kalman_q{ratio}_t{tau}', family='kalman_velocity',
                            kind='kalman', ratio=ratio, tau=tau))
    for tau_bg in (100, 500):
        for tau_res in (10, 50):
            out.append(dict(id=f'coupled_observer_bg{tau_bg}_res{tau_res}',
                            family='coupled_observer', kind='coupled',
                            tau_bg=tau_bg, tau_res=tau_res,
                            ratio_bg=0.01, ratio_res=0.1))
    assert len(out) == 35
    return out


def fit(qs, bgs, weights, end, config):
    if not (len(qs) == len(bgs) == len(weights)):
        raise ValueError('unequal image lists')
    m = dict(config=config.copy(), fit_end=int(end), channels=128,
             transformation='none; native coordinates', future_measurements=False)
    if config['kind'] == 'coupled':
        # Each image has equal mass; particles use coverage weights. Estimate a
        # diagonal response to common background motion, not a rotated basis.
        mean_x = np.zeros(128); mean_y = np.zeros(128)
        xx = np.zeros(128); xy = np.zeros(128)
        for q, bg, w in zip(qs, bgs, weights):
            dx = np.diff(bg[:end + 1], axis=0)
            dy = np.einsum('tnc,n->tc', np.diff(q[:end + 1], axis=0), w)
            mean_x += dx.mean(0) / len(qs)
            mean_y += dy.mean(0) / len(qs)
            xx += (dx * dx).mean(0) / len(qs)
            xy += (dx * dy).mean(0) / len(qs)
        var = np.maximum(xx - mean_x**2, 0)
        cov = xy - mean_x * mean_y
        # Fixed 0.1 relative ridge, estimated exclusively from the fit prefix.
        m['beta'] = cov / (1.1 * var + 1e-12)
    return m


def _rho(tau):
    return 1.0 if tau is None else float(np.exp(-1.0 / tau))


def _sum_velocity(v, rho, max_h):
    h = np.arange(max_h + 1, dtype=np.float64)
    factor = h if rho == 1.0 else rho * (-np.expm1(h * np.log(rho))) / (1 - rho)
    return factor.reshape((-1,) + (1,) * v.ndim) * v[None]


def _kalman_velocity(history, ratio, rho):
    """Filter independent coordinates; identical fixed noise-ratio covariance.

    F=[[1,rho],[0,rho]] means the next displacement is the next velocity.
    Q=diag(0.001*ratio,ratio), R=1. Position reset happens only AFTER filtering.
    Joseph covariance update avoids loss of symmetry/positive semidefiniteness.
    """
    position = history[0].copy()
    velocity = np.zeros_like(position)
    F = np.array([[1.0, rho], [0.0, rho]])
    Q = np.diag([0.001 * ratio, ratio])
    P = np.eye(2)
    for value in history[1:]:
        position = position + rho * velocity
        velocity = rho * velocity
        P = F @ P @ F.T + Q
        gain = P[:, 0] / (P[0, 0] + 1.0)
        innovation = value - position
        position += gain[0] * innovation
        velocity += gain[1] * innovation
        I_KH = np.eye(2)
        I_KH[:, 0] -= gain
        P = I_KH @ P @ I_KH.T + np.outer(gain, gain)
    return velocity


def predict(q, bg, w, origin, max_h, model):
    # A slice is taken before ALL computations; tests poison the unused suffix.
    history = np.asarray(q[:origin + 1], dtype=np.float64)
    background = np.asarray(bg[:origin + 1], dtype=np.float64)
    current = history[-1]
    c = model['config']; kind = c['kind']
    if kind == 'persistence':
        delta = np.zeros((max_h + 1,) + current.shape)
    elif kind == 'velocity':
        v = history[-1] - history[-2] if len(history) > 1 else np.zeros_like(current)
        delta = _sum_velocity(v, _rho(c['tau']), max_h)
    elif kind == 'ols':
        recent = history[-c['window']:]
        t = np.arange(len(recent), dtype=np.float64)
        t -= t.mean()
        v = np.einsum('t,tnc->nc', t, recent) / max(float(t @ t), 1.0)
        delta = _sum_velocity(v, _rho(c['tau']), max_h)
    elif kind == 'reversion':
        equilibrium = history[-c['window']:].mean(0)
        factor = -np.expm1(-np.arange(max_h + 1) / c['tau'])
        delta = factor[:, None, None] * (equilibrium - current)[None]
    elif kind == 'kalman':
        rho = _rho(c['tau'])
        v = _kalman_velocity(history, c['ratio'], rho)
        delta = _sum_velocity(v, rho, max_h)
    elif kind == 'coupled':
        beta = model['beta']
        residual = history - background[:, None, :] * beta
        rho_bg = _rho(c['tau_bg']); rho_res = _rho(c['tau_res'])
        v_bg = _kalman_velocity(background, c['ratio_bg'], rho_bg)
        v_res = _kalman_velocity(residual, c['ratio_res'], rho_res)
        delta = (_sum_velocity(v_res, rho_res, max_h)
                 + _sum_velocity(v_bg, rho_bg, max_h)[:, None, :] * beta)
    else:
        raise ValueError(kind)
    result = current[None] + delta
    assert result.shape == (max_h + 1,) + current.shape
    assert np.array_equal(result[0], current)
    return result


def audit():
    """Meaningful invariants independent of experiment target values."""
    rng = np.random.default_rng(8142)
    q = rng.normal(size=(31, 3, 128)).cumsum(0)
    bg = rng.normal(size=(31, 128)).cumsum(0)
    w = np.array([0.2, 0.3, 0.5]); rows = []
    for c in configs():
        m = fit([q], [bg], [w], 15, c)
        q_bad = q.copy(); bg_bad = bg.copy()
        q_bad[21:] = np.nan; bg_bad[21:] = np.nan
        a = predict(q, bg, w, 20, 10, m)
        b = predict(q_bad, bg_bad, w, 20, 10, m)
        assert np.array_equal(a, b), c['id']
        m_bad = fit([q_bad], [bg_bad], [w], 15, c)
        assert np.array_equal(a, predict(q, bg, w, 20, 10, m_bad))
        shift = rng.normal(size=(3, 128))
        shifted = predict(q + shift, bg, w, 20, 10, m)
        assert np.allclose(shifted, a + shift, atol=1e-11), c['id']
        rows.append(dict(id=c['id'], prefix_only=True, translation_equivariant=True))
    constant = np.ones_like(q) * 2.0
    for c in configs():
        m = fit([constant], [bg * 0], [w], 15, c)
        pred = predict(constant, bg * 0, w, 20, 10, m)
        assert np.allclose(pred, 2.0), c['id']
    linear = np.arange(31)[:, None, None] * np.ones((1, 3, 128)) * 0.3
    c = next(c for c in configs() if c['id'] == 'ols_w16_tNone')
    m = fit([linear], [bg], [w], 15, c)
    assert np.allclose(predict(linear, bg, w, 20, 10, m), linear[20:31])
    return dict(status='passed', tested_configs=rows,
                constant_path_preserved=True, exact_linear_ols=True)


if __name__ == '__main__':
    import json
    from pathlib import Path
    Path(__file__).with_name('AUDYT_OBSERWATORA.json').write_text(
        json.dumps(audit(), indent=2) + '\n', encoding='utf-8')
    print('35 configurations: prefix, translation, constant and exact trend checks passed')
