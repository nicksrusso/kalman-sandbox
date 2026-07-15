"""Drive the hand-written IFFClassifier over the truth sim and grade it.

Run with:  poetry run python run_bayes_iff.py
Prints the belief trajectory for one target, then classification accuracy over
many random targets. Saves bayes_iff.png (belief in HOSTILE over time).

Pure infrastructure: builds the prior + likelihood CPT, feeds the sim's IFF
stream to the classifier, and scores the posterior against the hidden label.
The estimator itself lives in kalman_sandbox/BayesIFF.py.
"""

import numpy as np

from kalman_sandbox.BayesIFF import IFFClassifier
from kalman_sandbox.truth_sim import (
    IFF_FOE,
    IFF_FRIEND,
    Label,
    TruthConfig,
    TruthSim,
)

# --- the estimator's a-priori model -------------------------------------------
PRIOR = {Label.FRIENDLY: 0.5, Label.HOSTILE: 0.5}

# P(reply | label). Matches the sim's generative params (well-specified filter).
LIKELIHOOD = {
    Label.FRIENDLY: {IFF_FRIEND: 0.97, IFF_FOE: 0.03},
    Label.HOSTILE:  {IFF_FRIEND: 0.05, IFF_FOE: 0.95},
}


def run_one(cfg: TruthConfig):
    """Return (ground-truth label, belief-in-HOSTILE trajectory, replies heard)."""
    obs, gt = TruthSim(cfg).run()
    clf = IFFClassifier(PRIOR, LIKELIHOOD)

    trajectory = []
    replies = []
    for o in obs:
        clf.update(o.iff)
        trajectory.append(clf.belief[Label.HOSTILE])
        if o.iff is not None:
            replies.append(o.iff)
    return gt.label, np.array(trajectory), replies


def demo():
    cfg = TruthConfig(label=Label.HOSTILE, seed=0)
    truth, traj, replies = run_one(cfg)

    n_friend = replies.count(IFF_FRIEND)
    n_foe = replies.count(IFF_FOE)
    print(f"hidden label     : {truth.value}")
    print(f"replies heard    : {len(replies)}  ({n_friend} friend, {n_foe} foe)")
    print(f"final P(hostile) : {traj[-1]:.4f}")
    print(f"decision         : {'HOSTILE' if traj[-1] > 0.5 else 'FRIENDLY'}"
          f"  ({'correct' if (traj[-1] > 0.5) == (truth is Label.HOSTILE) else 'WRONG'})")
    return traj


def accuracy(n_targets=400, n_steps=60, base_seed=1000):
    """Classify many random targets; report accuracy and mean final confidence."""
    correct = 0
    confidences = []
    for i in range(n_targets):
        cfg = TruthConfig(label=None, p_hostile=0.5, n_steps=n_steps, seed=base_seed + i)
        truth, traj, _ = run_one(cfg)
        p_hostile = traj[-1]
        decided_hostile = p_hostile > 0.5
        if decided_hostile == (truth is Label.HOSTILE):
            correct += 1
        # Confidence assigned to the *true* class.
        confidences.append(p_hostile if truth is Label.HOSTILE else 1 - p_hostile)

    print(f"\naccuracy over {n_targets} random targets : {correct / n_targets:.3f}")
    print(f"mean confidence in true label          : {np.mean(confidences):.3f}")


def plot(traj, path="bayes_iff.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping plot)")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(traj, "-o", ms=3, color="crimson")
    ax.axhline(0.5, color="0.5", ls="--", label="decision boundary")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("timestep")
    ax.set_ylabel("P(hostile | IFF so far)")
    ax.set_title("Bayesian belief updating from intermittent IFF replies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    traj = demo()
    accuracy()
    plot(traj)


if __name__ == "__main__":
    main()
