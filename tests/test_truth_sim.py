"""Tests for the ground-truth simulator.

These assert the *contract* the estimator will rely on and the statistical
properties of the sensor model (checked over enough samples to be stable)."""

import numpy as np
import pytest

from kalman_sandbox.truth_sim import (
    IFF_FOE,
    IFF_FRIEND,
    SPEED_HIGH,
    SPEED_LOW,
    GroundTruth,
    Label,
    Observation,
    TruthConfig,
    TruthSim,
)


def test_run_shapes_and_contract():
    cfg = TruthConfig(n_steps=40, seed=1)
    obs, gt = TruthSim(cfg).run()

    assert len(obs) == cfg.n_steps
    assert isinstance(gt, GroundTruth)
    assert gt.states.shape == (cfg.n_steps, 4)
    assert gt.is_turning.shape == (cfg.n_steps,)

    for t, o in enumerate(obs):
        assert isinstance(o, Observation)
        assert o.t == t
        assert o.position.shape == (2,)
        assert o.iff in (IFF_FRIEND, IFF_FOE, None)
        assert o.speed_bucket in (SPEED_HIGH, SPEED_LOW)


def test_reproducible_with_seed():
    cfg = TruthConfig(seed=42)
    obs_a, gt_a = TruthSim(cfg).run()
    obs_b, gt_b = TruthSim(cfg).run()

    assert gt_a.label == gt_b.label
    np.testing.assert_array_equal(gt_a.states, gt_b.states)
    for a, b in zip(obs_a, obs_b):
        np.testing.assert_array_equal(a.position, b.position)
        assert a.iff == b.iff
        assert a.speed_bucket == b.speed_bucket


def test_different_seeds_differ():
    # The true trajectory is deterministic given config (no motion noise) --
    # only the *sensors* consume the RNG, so seeds must change measurements.
    obs1, gt1 = TruthSim(TruthConfig(seed=1, label=Label.HOSTILE)).run()
    obs2, gt2 = TruthSim(TruthConfig(seed=2, label=Label.HOSTILE)).run()
    np.testing.assert_array_equal(gt1.states, gt2.states)
    meas1 = np.array([o.position for o in obs1])
    meas2 = np.array([o.position for o in obs2])
    assert not np.array_equal(meas1, meas2)


def test_constant_velocity_then_turn():
    # Before the turn the velocity is exactly constant; during the turn the
    # heading changes while speed is preserved (coordinated turn).
    cfg = TruthConfig(
        n_steps=50, turn_start=20, turn_duration=10, meas_noise_std=0.0, seed=0
    )
    _, gt = TruthSim(cfg).run()
    v = gt.velocities

    # Straight leg: velocity identical across the pre-turn steps.
    np.testing.assert_allclose(v[:20], np.tile(v[0], (20, 1)), atol=1e-9)

    # Heading changes once turning starts.
    assert not np.allclose(v[25], v[0])

    # Coordinated turn preserves speed to numerical precision.
    speeds = gt.speeds
    np.testing.assert_allclose(speeds, speeds[0], atol=1e-9)


def test_turning_flags_align_with_config():
    cfg = TruthConfig(n_steps=50, turn_start=20, turn_duration=10)
    _, gt = TruthSim(cfg).run()
    expected = np.zeros(50, dtype=bool)
    expected[20:30] = True
    np.testing.assert_array_equal(gt.is_turning, expected)


def test_position_measurement_noise_is_unbiased():
    # With zero noise the measurement equals the true position exactly.
    cfg = TruthConfig(meas_noise_std=0.0, seed=3)
    obs, gt = TruthSim(cfg).run()
    for t, o in enumerate(obs):
        np.testing.assert_allclose(o.position, gt.positions[t], atol=1e-12)


def _iff_stats(cfg: TruthConfig):
    obs, _ = TruthSim(cfg).run()
    replies = [o.iff for o in obs]
    n = len(replies)
    n_silent = sum(r is None for r in replies)
    heard = [r for r in replies if r is not None]
    n_friend = sum(r == IFF_FRIEND for r in heard)
    return n, n_silent, len(heard), n_friend


def test_iff_intermittent():
    # ~60% of ticks should be silent given iff_query_success=0.4.
    cfg = TruthConfig(n_steps=4000, iff_query_success=0.4,
                      label=Label.FRIENDLY, seed=7)
    n, n_silent, _, _ = _iff_stats(cfg)
    assert 0.55 < n_silent / n < 0.65


def test_iff_conditioned_on_label():
    # Friendly targets mostly reply "friend"; hostiles rarely do.
    friendly = TruthConfig(n_steps=4000, iff_query_success=1.0,
                           label=Label.FRIENDLY, seed=11)
    hostile = TruthConfig(n_steps=4000, iff_query_success=1.0,
                          label=Label.HOSTILE, seed=11)

    _, _, heard_f, friend_f = _iff_stats(friendly)
    _, _, heard_h, friend_h = _iff_stats(hostile)

    assert friend_f / heard_f > 0.9   # friendly ~0.97
    assert friend_h / heard_h < 0.1   # hostile ~0.05


def test_speed_bucket_flip_rate():
    # A slow, straight target sits below threshold; reported bucket should be
    # "low" except for ~flip_prob of ticks.
    cfg = TruthConfig(
        n_steps=4000, v0=(0.5, 0.0), turn_duration=0, speed_threshold=2.0,
        speed_flip_prob=0.1, seed=5,
    )
    obs, _ = TruthSim(cfg).run()
    high_rate = np.mean([o.speed_bucket == SPEED_HIGH for o in obs])
    assert 0.06 < high_rate < 0.14  # ~0.1 flips


def test_label_resolution():
    assert TruthSim(TruthConfig(label=Label.HOSTILE)).run()[1].label is Label.HOSTILE
    assert TruthSim(TruthConfig(label=Label.FRIENDLY)).run()[1].label is Label.FRIENDLY
    # p_hostile at the extremes forces the draw.
    assert TruthSim(TruthConfig(label=None, p_hostile=1.0)).run()[1].label is Label.HOSTILE
    assert TruthSim(TruthConfig(label=None, p_hostile=0.0)).run()[1].label is Label.FRIENDLY


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
