# What the paper does not give us

Every value below is **not** in SkyDreamer (arXiv:2510.14783). Each one has a
default in the code so the system runs end to end, and each default is marked
here with why it was chosen and how to tell if it is wrong.

Legend for **Risk**: how much a wrong choice here costs us.
- 🔴 changes whether we can hit the paper's numbers at all
- 🟡 changes training cost or stability
- 🟢 cosmetic / easily re-tuned

---

## A. Genuine ambiguities in the paper's own equations

### A1. Horizontal advance ratio `mu` — 🔴
The paper prints

> `mu_{xx,yy} = atan((v_x^B2 + v_y^B2) / (r * w_bar))`

Read literally this is a *squared* speed over a speed. At 20 m/s it yields
`atan(400/326) = 0.89 rad`, and with `k_hor = 7.245` the thrust would go up
~640%. The drone is unflyable.

**Default:** `dynamics.ADVANCE_RATIO_MODE = "norm"` → `atan(|v_xy| / (r w_bar))`,
which is dimensionally consistent with the angle-of-attack term printed
immediately above it. Gives ~44% translational lift at 20 m/s.

**Still suspicious:** 44% is high; real translational lift is 5–15%. Either
`k_hor` is defined against a different normalizer, or `r` is not the propeller
radius. Set `ADVANCE_RATIO_MODE = "literal"` to see the pathology for yourself.

**How we'll know:** identify the term from real high-speed flight logs (thrust
residual vs. horizontal speed at fixed RPM), exactly as MonoRace describes for
its own coefficients.

### A2. `r` in the `alpha` and `mu` denominators — 🔴
Never defined in the paper. We use the propeller radius,
`params.PROP_RADIUS = 0.0648 m` (5.1 inch props, per MonoRace arXiv:2601.15222
on the same airframe). This makes `alpha` come out to a few percent of thrust
at realistic climb rates, which is the right order of magnitude, and it gives
the correct *sign* (climbing loses thrust — pinned by
`test_climb_reduces_thrust_descent_increases_it`).

### A3. `k_angle` vs `k_alpha` — 🟢
Table II names it `k_angle`; the force equation names it `k_alpha`. Same
coefficient. We use `k_angle` everywhere.

### A4. `w_i = (w_ci - w_i)/tau` — 🟢
Obvious typo for `w_i_dot`. Implemented as the derivative.

### A5. `k_l` vs `k` — 🟢
The motor-response equation uses `k_l` in `(1 - k_l) u_tilde`; Table II only
lists `k`. Ferede's public implementation (`optimal_quad_control_RL`) uses the
same `k` in both places. We do too.

### A6. Control period — 🟡
The paper says RK4 at 2.2 ms *and* `f_c = 90 Hz`. 1/90 s = 11.111 ms is not a
multiple of 2.2 ms. We use **5 substeps × 2.2 ms = 11.0 ms** (90.9 Hz), the
only integer reading. 1% off; matters only for delay bookkeeping.

### A7. Smoothness-loss action scale — 🟡
`L_smooth = 0.002 * E[||mu_t - mu_{t-1}||^2]`, where the paper's actions are
`u in [0,1]`. DreamerV3's `bounded_normal` head produces a tanh mean in
`[-1,1]`, so the same coefficient is effectively **4× stronger** in our
implementation. Left at the paper's `0.002` for now — see `imag_loss` in
`patches/informed_dreamer.patch`. If the policy comes out sluggish, this is the
first knob to touch.

### A8. GateNet multi-scale head numbering — 🟡
Appendix A's figure numbers the heads `outc0` at the **bottleneck** and `outc4`
at **full resolution**. The loss is `L = 4*L_0 + 2*L_1 + L_2 + L_3 + L_4`,
described as "output-specific scaling factors to emphasize higher-resolution
predictions". Taken together those contradict: `4*L_0` would weight the
*coarsest* map most.

**Default:** we follow the stated intent, not the figure's numbering — index 0
is the full-resolution head and carries weight 4. `GateNet.forward` returns
finest-first and `test_five_heads_finest_first` pins it.

---

## B. Values the paper simply never states

### B1. Gate coordinates for every track — 🔴 (but we own this)
No track in the paper has published gate positions; they exist only in prose
and figures. `track.INVERTED_LOOP` and `track.LADDER_INVERTED_LOOP` are our
reconstructions, sized to the 6×6 m and 8×8 m volumes the paper reports.
**Replace with surveyed coordinates from our own track.** Nothing else in the
codebase depends on them.

The "big track" (Figure 9) is not reconstructed at all — it has ~9 gates in a
larger hall and the figure is not dimensioned.

### B2. Invisible-gate placement — 🔴
The paper says "additional invisible gates are added to enforce the correct
flight maneuvers" and never says where, how many, or how big. These are what
actually *define* the inverted loop and the ladder: without them the drone
would fly the short way round. Our guesses are the `visible=False` entries in
`track.py`. Expect to iterate on these more than on anything else.

### B3. Sensor noise on `Omega_hat` and `omega_hat` — 🟡
The paper is explicit that the privileged information carries *ground-truth*
rates "not the measured ones", which only means something if the measured ones
are noisy. No noise model is given.
**Default: zero** (`env.GYRO_NOISE_STD = env.RPM_NOISE_STD = 0.0`) so we
reproduce the paper as literally as possible. Knobs are in `EnvConfig`.

### B4. Flight-plan increment timing during training — 🟡
"the flight plan index `i` is incremented randomly between the pre- and
post-gate". We increment uniformly at random at one of the three plane
crossings (p = 1/3, 1/2, 1). Any distribution over that window satisfies the
text.

### B5. Privileged decoder head layout — 🟡
The paper decodes `i_t` but never says whether it is one head or many, nor how
the targets are scaled. We split it into 12 semantically named `info_*` keys
(`info_p_w`, `info_v_g`, `info_cam`, `info_dyn`, …) because separate heads are
what make the interpretability claim usable — you can plot decoded position
directly. `info_dyn` is normalized to roughly [-1, 1] against the ±30% training
range; RPM is normalized by nominal `w_max`. DreamerV3's default symlog handles
the rest.

### B6. Gate visual geometry — 🟢
Inner 1.5 m / outer 2.7 m (orange) and 1.5 / 2.1 (MAVLab) are given, but not
the frame profile or thickness. We render a flat square annulus.

### B7. Number of seeds, and which run was deployed — 🟡
The paper admits "the accuracy of parameter identification depends on the
training run". It never says how many runs were done or how the deployed one
was picked. Budget for **several seeds per configuration** and a selection
criterion of our own.

### B8. `d_g` for the ladder track — 🟢
Section III-B says that track trains with `t_g = 0.3 m` rather than 0.8. We
expose `t_g` per config; the default stays 0.8.

---

## C. Datasets — none exist, and two of them need the drone first

| Dataset | Size in paper | Status |
|---|---|---|
| GateNet labels, orange gates | 200 hand-labelled real images | must collect |
| GateNet labels, MAVLab gates | 700 hand-labelled + 8500 synthetic | must collect |
| StochGAN unpaired masks | ~4000 real + rendered | **needs real flight footage** |
| Trained checkpoints | — | none released |

Formats, counts and the collection order are specified in
[`datasets.md`](datasets.md). Note in particular that **policy training needs
no dataset at all** — DreamerV3 generates its own data by flying the simulator,
and the masks are rendered per step rather than loaded.

Implication for sequencing: **the visual sim-to-real half of the paper cannot
start until the drone flies and records.** The pure-simulation reproduction
(rendered masks straight into the policy) has no such dependency, which is why
it is phase 1.

---

## D. Things the paper describes but does not release

- **JAX → PyTorch converter** ("our custom implementation"). Needed for the
  ONNX → TensorRT deployment path. We have to re-derive the RSSM encoder, GRU
  and actor in PyTorch and map weights.
- **The 90 Hz onboard timing stack** — camera-timestamp anchoring, motor
  commands with a desired execution timestamp, buffering. `tudelft/indiflight`
  is the public starting point.
- **The MSc thesis** with all of this spelled out: TU Delft repository
  `uuid:18156042-24d1-4bc8-8c6b-d5ea71f978cc`, **under embargo until
  2027-09-23**. Worth emailing the author.


---

## E. Values taken from the wider ADR literature

The paper is silent on these and they are not derivable from it, so they come
from the autonomous-drone-racing literature instead. Each is a *protocol* or
*budget* choice -- none of them change the policy the paper describes.

### E1. Simulation evaluation protocol
The paper's Table IV is 5 flights x 5 laps per track (25 laps), which is a
*real-world* budget -- 75 laps is a lot of flying. In simulation that limit
does not apply, and Ferede et al. (arXiv:2504.21586, same lab, same dynamics
model) evaluate over **1000 rollouts**.

**Our default: 100 episodes x 5 laps**, i.e. 500 laps -- 20x the paper's
real-world sample and cheap because the env is batched. Raise with
`--episodes`.

### E2. Reported metrics
The ADR survey (arXiv:2301.01755) and the champion-level papers converge on
success rate, lap time, and **mean gate error** (distance from gate centre at
the crossing plane). Kaufmann et al. additionally report **gate margin**, the
smallest clearance to the gate edge. `scripts/evaluate.py` reports all four,
plus max speed and max acceleration so the 13 m/s and 6 g claims are directly
checkable, plus decoded-state error for the interpretability claim.

Lap 1 is reported separately from laps 2+ because Table IV does -- the drone
starts from rest, so its first lap is always slower.

### E3. Seeds
Ferede et al. use **three independent runs** for their architecture comparison.
SkyDreamer admits its parameter-identification quality "depends on the training
run" but never says how many it did. `run.sh` does one; pass `--seed N` for
more, and budget for at least three before trusting a number.

### E4. Dynamics randomization width — confirmed, not changed
The literature recommends roughly **15-25%** randomization on dynamics
parameters for sim-to-real, with the motor time constant being the critical
one. SkyDreamer's Table III uses +-20% (eval) to +-30% (train), with `tau`
included. That is consistent with the literature, so **we changed nothing** --
noted here only so nobody "fixes" it later.

### E5. Sensor noise — deliberately left at zero
The ADR literature is clear that the dominant IMU error on a racing quad is
**propeller vibration, not sensor noise**, and there is no standard value to
borrow: it depends on the airframe. Since our goal is to reproduce the paper's
numbers as literally as possible, `GYRO_NOISE_STD` and `RPM_NOISE_STD` stay at
0 and the knobs stay exposed (see B3). Set them from our own flight logs once
the airframe exists, rather than from someone else's drone.
