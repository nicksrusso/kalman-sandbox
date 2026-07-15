"""Ground-truth simulator for a maneuvering 2D target.

This module is deliberately *independent* of any estimator (Kalman / Bayes).
It plays the role of the physical world plus the sensors bolted to it:

  world  ->  a target with state [px, py, vx, vy] that flies constant-velocity,
             then executes a hard coordinated turn, then flies straight again.
  label  ->  a hidden ground-truth allegiance (HOSTILE / FRIENDLY) that the
             estimator never sees directly.
  sensors -> per timestep it emits an ``Observation``:
               * a noisy position measurement  (feeds the Kalman filter)
               * an IFF reply, or ``None``      (feeds the Bayes net)
               * a speed bucket, high/low       (feeds the Bayes net)

The estimator you build later consumes the stream of ``Observation``s and is
graded against the ``GroundTruth`` that the sim holds privately.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class Label(str, Enum):
    """Hidden ground-truth allegiance of the target."""

    FRIENDLY = "friendly"
    HOSTILE = "hostile"


# IFF reply channel. A tick yields one of these string values, or ``None`` when
# the interrogation gets no response at all (the common, realistic case).
IFF_FRIEND = "friend"
IFF_FOE = "foe"

# Speed bucket channel.
SPEED_HIGH = "high"
SPEED_LOW = "low"


@dataclass
class TruthConfig:
    """All knobs for the world + sensor model. Defaults give a clean CV/CT/CV
    profile with a maneuver hard enough to make a naive filter lag."""

    # --- time base ---
    dt: float = 1.0
    n_steps: int = 60

    # --- motion profile: constant velocity, then a coordinated turn, then CV ---
    p0: tuple[float, float] = (0.0, 0.0)
    v0: tuple[float, float] = (2.0, 0.5)
    turn_start: int = 25          # step index at which the turn begins
    turn_duration: int = 12       # number of steps spent turning
    turn_rate_rad: float = 0.20   # angular velocity while turning (rad per step*dt)

    # --- position sensor ---
    meas_noise_std: float = 3.0   # per-axis Gaussian std on (px, py)

    # --- hidden label ---
    label: Optional[Label] = None  # if None, drawn from p_hostile
    p_hostile: float = 0.5

    # --- IFF sensor (conditioned on the hidden label) ---
    # Probability the interrogation gets *any* reply on a given tick. The rest
    # of the time the channel is silent (None) -> sensors don't answer every tick.
    iff_query_success: float = 0.4
    # P(reply == "friend" | label) given that a reply arrived. Foe otherwise.
    p_friend_if_friendly: float = 0.97
    p_friend_if_hostile: float = 0.05  # hostiles rarely (spoof) reply "friend"

    # --- speed-bucket sensor (derived from true velocity) ---
    speed_threshold: float = 2.0  # speed >= threshold -> "high"
    speed_flip_prob: float = 0.10  # chance the reported bucket is flipped

    seed: int = 0


@dataclass
class Observation:
    """What the estimator is allowed to see at one timestep."""

    t: int
    position: np.ndarray            # shape (2,), noisy (px, py) measurement
    iff: Optional[str]              # IFF_FRIEND / IFF_FOE / None (no reply)
    speed_bucket: str               # SPEED_HIGH / SPEED_LOW


@dataclass
class GroundTruth:
    """Everything the sim knows but the estimator does not. For grading only."""

    label: Label
    states: np.ndarray              # shape (n_steps, 4): [px, py, vx, vy] per step
    is_turning: np.ndarray          # shape (n_steps,) bool: was the target maneuvering?

    @property
    def positions(self) -> np.ndarray:
        return self.states[:, :2]

    @property
    def velocities(self) -> np.ndarray:
        return self.states[:, 2:]

    @property
    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.velocities, axis=1)


class TruthSim:
    """Black-box world + sensors. Call :meth:`run` to get the observation
    stream; read :attr:`ground_truth` afterwards to grade against."""

    def __init__(self, config: TruthConfig | None = None):
        self.config = config or TruthConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self.ground_truth: GroundTruth | None = None

    # ------------------------------------------------------------------ motion
    def _step_state(self, state: np.ndarray, turning: bool) -> np.ndarray:
        """Advance [px, py, vx, vy] by one dt.

        Straight legs use constant velocity. During the maneuver we apply a
        *coordinated turn*: the velocity vector is rotated by ``turn_rate_rad``
        (speed is preserved, heading changes), then position integrates with the
        rotated velocity (semi-implicit Euler -- stable and easy to read)."""
        dt = self.config.dt
        px, py, vx, vy = state

        if turning:
            w = self.config.turn_rate_rad
            c, s = np.cos(w), np.sin(w)
            vx, vy = c * vx - s * vy, s * vx + c * vy

        px += vx * dt
        py += vy * dt
        return np.array([px, py, vx, vy])

    # ------------------------------------------------------------------ sensors
    def _measure_position(self, state: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(0.0, self.config.meas_noise_std, size=2)
        return state[:2] + noise

    def _iff_reply(self, label: Label) -> Optional[str]:
        # Silent most of the time: intermittent, realistic interrogation.
        if self._rng.random() >= self.config.iff_query_success:
            return None
        p_friend = (
            self.config.p_friend_if_friendly
            if label is Label.FRIENDLY
            else self.config.p_friend_if_hostile
        )
        return IFF_FRIEND if self._rng.random() < p_friend else IFF_FOE

    def _speed_bucket(self, state: np.ndarray) -> str:
        speed = float(np.hypot(state[2], state[3]))
        bucket = SPEED_HIGH if speed >= self.config.speed_threshold else SPEED_LOW
        if self._rng.random() < self.config.speed_flip_prob:
            bucket = SPEED_LOW if bucket == SPEED_HIGH else SPEED_HIGH
        return bucket

    # ------------------------------------------------------------------ driver
    def run(self) -> tuple[list[Observation], GroundTruth]:
        cfg = self.config

        # Resolve the hidden label once, up front.
        if cfg.label is not None:
            label = cfg.label
        else:
            label = Label.HOSTILE if self._rng.random() < cfg.p_hostile else Label.FRIENDLY

        state = np.array([cfg.p0[0], cfg.p0[1], cfg.v0[0], cfg.v0[1]], dtype=float)

        states = np.empty((cfg.n_steps, 4))
        turning_flags = np.zeros(cfg.n_steps, dtype=bool)
        observations: list[Observation] = []

        turn_end = cfg.turn_start + cfg.turn_duration
        for t in range(cfg.n_steps):
            turning = cfg.turn_start <= t < turn_end
            # t == 0 emits the initial state before any motion is applied.
            if t > 0:
                state = self._step_state(state, turning)

            states[t] = state
            turning_flags[t] = turning

            observations.append(
                Observation(
                    t=t,
                    position=self._measure_position(state),
                    iff=self._iff_reply(label),
                    speed_bucket=self._speed_bucket(state),
                )
            )

        self.ground_truth = GroundTruth(
            label=label, states=states, is_turning=turning_flags
        )
        return observations, self.ground_truth
