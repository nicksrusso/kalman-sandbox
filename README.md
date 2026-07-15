# kalman-sandbox

From-scratch **Kalman filter** + **Bayes net** over a 2D target-tracking sim.
Interview-prep sandbox.

## Layout

- `kalman_sandbox/truth_sim.py` — the world + sensors (Part 1). A black box that
  emits an `Observation` per timestep and privately holds the `GroundTruth`.
- `tests/` — pytest suite.
- `demo_truth_sim.py` — visual sanity check.

## The truth sim

A 2D target with state `[px, py, vx, vy]` flies **constant velocity**, then
executes a **hard coordinated turn**, then flies straight again. It carries a
hidden `Label` (`FRIENDLY` / `HOSTILE`) the estimator never sees.

Each step it emits an `Observation`:

| field          | feeds        | model                                             |
| -------------- | ------------ | ------------------------------------------------- |
| `position`     | Kalman       | true `(px, py)` + Gaussian noise, every tick      |
| `iff`          | Bayes net    | `"friend"` / `"foe"` / `None`; intermittent, label-conditioned |
| `speed_bucket` | Bayes net    | `"high"` / `"low"` from true speed, with flip noise |

```python
from kalman_sandbox.truth_sim import TruthConfig, TruthSim, Label

obs, gt = TruthSim(TruthConfig(label=Label.HOSTILE, seed=0)).run()
# obs: list[Observation]  -> what the estimator consumes
# gt:  GroundTruth        -> hidden truth, for grading only
```

## Setup

```bash
poetry install
poetry run pytest
poetry run python demo_truth_sim.py
```
