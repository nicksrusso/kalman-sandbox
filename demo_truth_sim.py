"""Quick visual sanity check of the truth sim.

Run with:  poetry run python demo_truth_sim.py
Saves truth_sim_demo.png (true path, noisy measurements, maneuver segment).
"""

import numpy as np

from kalman_sandbox.truth_sim import IFF_FRIEND, Label, TruthConfig, TruthSim


def main() -> None:
    cfg = TruthConfig(label=Label.HOSTILE, seed=0)
    obs, gt = TruthSim(cfg).run()

    meas = np.array([o.position for o in obs])
    heard = [(o.t, o.iff) for o in obs if o.iff is not None]
    n_friend = sum(1 for _, r in heard if r == IFF_FRIEND)

    print(f"hidden label      : {gt.label.value}")
    print(f"steps             : {cfg.n_steps}")
    print(f"turn segment      : [{cfg.turn_start}, {cfg.turn_start + cfg.turn_duration})")
    print(f"IFF replies heard : {len(heard)}/{cfg.n_steps} "
          f"({n_friend} 'friend', {len(heard) - n_friend} 'foe')")
    print(f"speed range       : {gt.speeds.min():.2f} .. {gt.speeds.max():.2f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping plot)")
        return

    pos = gt.positions
    turn = gt.is_turning
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pos[:, 0], pos[:, 1], "-", color="0.4", label="true path")
    ax.plot(pos[turn, 0], pos[turn, 1], "-", color="crimson", lw=3, label="maneuver")
    ax.scatter(meas[:, 0], meas[:, 1], s=12, c="steelblue", alpha=0.6,
               label="noisy measurements")
    ax.scatter(*pos[0], c="green", marker="o", s=80, label="start", zorder=5)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Truth sim — hidden label: {gt.label.value}")
    fig.tight_layout()
    fig.savefig("truth_sim_demo.png", dpi=120)
    print("wrote truth_sim_demo.png")


if __name__ == "__main__":
    main()
