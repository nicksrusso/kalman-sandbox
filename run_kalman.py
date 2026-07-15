"""Drive the hand-written KalmanFilter over the truth sim and grade it.

Run with:  poetry run python run_kalman.py
Saves kalman_run.png (true path / measurements / filtered estimate) and prints
RMSE broken out over the straight legs vs. the maneuver.

This is pure infrastructure: it builds the constant-velocity model matrices,
runs the predict/update loop, and scores the result. The estimator itself lives
in kalman_sandbox/kalman.py.
"""

import numpy as np

from kalman_sandbox.kalman import KalmanFilter
from kalman_sandbox.truth_sim import Label, TruthConfig, TruthSim


def build_cv_matrices(dt: float, meas_std: float, q: float):
    """Constant-velocity model for state [px, py, vx, vy], position-only sensor.

    q is the acceleration-noise power -- the single Q tuning knob. Bigger q lets
    the filter believe the velocity can change (helps through the turn) at the
    cost of noisier straight-leg tracking.
    """
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    R = (meas_std ** 2) * np.eye(2)

    # Discrete white-noise-acceleration process noise, laid out for the
    # [px, py, vx, vy] ordering (x-axis couples 0<->2, y-axis couples 1<->3).
    q11 = q * dt ** 4 / 4
    q13 = q * dt ** 3 / 2
    q33 = q * dt ** 2
    Q = np.array([
        [q11, 0,   q13, 0],
        [0,   q11, 0,   q13],
        [q13, 0,   q33, 0],
        [0,   q13, 0,   q33],
    ], dtype=float)

    return F, H, Q, R


def run(cfg: TruthConfig, q: float = 0.2):
    obs, gt = TruthSim(cfg).run()

    F, H, Q, R = build_cv_matrices(cfg.dt, cfg.meas_noise_std, q)

    # Seed the belief from the first measurement; velocity unknown -> big P0.
    z0 = obs[0].position
    x0 = np.array([z0[0], z0[1], 0.0, 0.0])
    P0 = np.diag([cfg.meas_noise_std ** 2, cfg.meas_noise_std ** 2, 100.0, 100.0])

    kf = KalmanFilter(F, H, Q, R, x0, P0)

    est = []
    nis = []
    for o in obs:
        kf.predict()
        kf.update(o.position)
        est.append(kf.x.copy())
        # Normalized innovation squared: yᵀ S⁻¹ y. For a consistent filter this
        # is chi-square with m (=2) degrees of freedom, so it should hover near 2.
        nis.append(float(kf.y @ np.linalg.inv(kf.S) @ kf.y))
    est = np.array(est)  # (n_steps, 4)
    nis = np.array(nis)  # (n_steps,)

    meas = np.array([o.position for o in obs])
    return obs, gt, est, meas, nis


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def grade(gt, est, meas, nis):
    turn = gt.is_turning
    straight = ~turn

    print("position RMSE (Euclidean):")
    print(f"  raw measurements : {rmse(meas, gt.positions):.3f}")
    print(f"  filtered (all)   : {rmse(est[:, :2], gt.positions):.3f}")
    print(f"  filtered straight: {rmse(est[straight, :2], gt.positions[straight]):.3f}")
    print(f"  filtered turn    : {rmse(est[turn, :2], gt.positions[turn]):.3f}")
    print(f"velocity RMSE      : {rmse(est[:, 2:], gt.velocities):.3f}")

    # NIS should average ~m (=2) if the filter is consistent. A mean well above 2
    # -- especially concentrated in the turn -- means the model can't explain the
    # surprises it's seeing (the unmodeled maneuver).
    print("NIS (expected mean ~2.0):")
    print(f"  mean straight : {nis[straight].mean():.2f}")
    print(f"  mean turn     : {nis[turn].mean():.2f}")
    print(f"  max           : {nis.max():.2f}")


def plot(cfg, gt, est, meas, path="kalman_run.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping plot)")
        return

    pos = gt.positions
    turn = gt.is_turning
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(pos[:, 0], pos[:, 1], "-", color="0.4", label="true path")
    ax.plot(pos[turn, 0], pos[turn, 1], "-", color="crimson", lw=3, label="maneuver")
    ax.scatter(meas[:, 0], meas[:, 1], s=14, c="steelblue", alpha=0.4,
               label="measurements")
    ax.plot(est[:, 0], est[:, 1], "-", color="darkorange", lw=2, label="filtered")
    ax.scatter(*pos[0], c="green", marker="o", s=80, zorder=5, label="start")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Kalman vs truth — label: {gt.label.value}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def plot_nis(gt, nis, path="kalman_nis.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    t = np.arange(len(nis))
    turn = gt.is_turning
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, nis, "-o", ms=3, color="darkorange", label="NIS")
    ax.axhline(2.0, color="0.4", ls="--", label="expected (m=2)")
    # 95% chi-square(2) interval: a consistent filter mostly stays in [0.05, 5.99].
    ax.axhspan(0.05, 5.99, color="green", alpha=0.08, label="95% band")
    ax.fill_between(t, 0, nis.max() * 1.05, where=turn, color="crimson",
                    alpha=0.12, label="maneuver")
    ax.set_xlabel("timestep")
    ax.set_ylabel("NIS  (yᵀ S⁻¹ y)")
    ax.set_title("Normalized innovation squared — spikes flag the unmodeled turn")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    cfg = TruthConfig(label=Label.HOSTILE, seed=0)
    obs, gt, est, meas, nis = run(cfg, q=0.2)
    grade(gt, est, meas, nis)
    plot(cfg, gt, est, meas)
    plot_nis(gt, nis)


if __name__ == "__main__":
    main()
