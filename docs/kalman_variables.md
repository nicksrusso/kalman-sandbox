# Kalman filter — what every variable actually means

Plain-English gloss of each symbol in `kalman_sandbox/kalman.py`. State is
`[px, py, vx, vy]` (n = 4), the sensor measures position only (m = 2).

## The model (fixed — describes the world and the sensor)

- **`F` — state transition matrix `(4,4)`**
  My rule for how the world moves on its own, with no new information. "If I know
  the state now, where does the physics say it'll be one step later?" Here it's
  constant velocity: position advances by velocity·dt, velocity stays put. It's a
  *belief* about the dynamics — and it's deliberately wrong during a turn.

- **`H` — observation matrix `(2,4)`**
  The translator from state-space to what-the-sensor-sees. "Given a full state,
  what measurement would I expect?" Here it just picks out `px, py` and throws
  away velocity, because the sensor can't see velocity.

- **`Q` — process-noise covariance `(4,4)`**
  How much I distrust my own motion model per step. "How much could the world do
  that my `F` doesn't capture?" It's the budget for unmodeled effects — mainly the
  turn. Built from one scalar knob `q` (acceleration noise power).

- **`R` — measurement-noise covariance `(2,2)`**
  How much I distrust the sensor. "How noisy is a single measurement?" Here I know
  it exactly: `meas_noise_std² · I`.

## The belief (mutated every step — my current knowledge)

- **`x` — state estimate `(4,)`**
  My single best guess of the true state right now.

- **`P` — estimate-error covariance `(4,4)`**
  How wrong I think my guess `x` is, and in what correlated way. Diagonal =
  confidence in each component; off-diagonals = coupling (e.g. "if my position is
  off, my velocity is probably off in a related direction"). The off-diagonals are
  what let a position-only measurement teach me about velocity.

- **`x0`, `p0` — initial belief**
  Where I start before any filtering. `x0` seeds position from the first
  measurement (velocity guessed as 0); `p0` says how unsure I am of that seed
  (big on velocity, since I'm really guessing there). Their influence washes out
  after a few steps.

## predict() — the time update (move forward, no measurement)

- **`x = F x`**
  Slide my best guess forward along the physics. "Where should it be now?"

- **`P = F P Fᵀ + Q`**
  Grow my uncertainty. `F P Fᵀ` carries my old uncertainty through the motion
  (the sandwich keeps `P` symmetric), and `+ Q` adds fresh doubt because the model
  isn't perfect. **Predict always makes me less certain.**

## update(z) — the measurement update (fold in evidence)

- **`z` — the measurement `(2,)`**
  What the sensor actually reported this step (a noisy position).

- **`y = z − H x` — the innovation ("the surprise") `(2,)`**
  The difference between my new observation and what I expected to see based on my
  prior state and motion model. Big `y` = the world surprised me; `y ≈ 0` = the
  measurement told me nothing new. Lives in measurement space, and it's a
  *difference*, not an estimate.

- **`S = H P Hᵀ + R` — innovation covariance ("expected surprise") `(2,2)`**
  How big the surprise *should* be, if my model is right. It adds my own
  prediction uncertainty mapped into measurement space (`H P Hᵀ`) to the sensor
  noise (`R`). It's the yardstick I measure `y` against — not whether the model is
  good, just how much surprise is normal.

- **`K = P Hᵀ S⁻¹` — Kalman gain `(4,2)`**
  How much I react to the surprise. Roughly (my uncertainty) / (total
  uncertainty). High `K` → "I'm the unreliable one, trust the measurement." Low
  `K` → "the sensor's the unreliable one, trust my prediction." Computed fresh
  every step. The `P Hᵀ` factor is what smuggles position measurements into
  velocity corrections via `P`'s off-diagonals.

- **`x = x + K y` — corrected estimate**
  Nudge my guess toward the measurement, weighted by how much I decided to trust
  it (`K`) times how wrong I was (`y`).

- **`P = (I − K H) P` — shrunk uncertainty**
  Tighten my confidence, because I just gained information. `I = np.eye(4)`.
  **Update always makes me more certain.**

## Diagnostics (not part of the core recursion)

- **NIS `= yᵀ S⁻¹ y`**
  "How many standard deviations of surprise did I *actually* get, versus how many
  I *expected*?" Should hover near `m` (=2). Sustained values well above that mean
  my model can't explain what it's seeing — the fingerprint of an unmodeled
  maneuver. It's the ratio of reality (`y`) to expectation (`S`).

## The one-sentence mental model

> Predict: *move my guess forward and get fuzzier.*
> Update: *how surprised was I (`y`), how surprised should I have been (`S`),
> therefore how hard do I react (`K`) — then correct the guess and sharpen it.*
