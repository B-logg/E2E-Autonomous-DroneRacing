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

### A0. Does training fix `k_hor` by itself?  No — and here is why it matters less than it looks

`k_hor` is a **simulator** parameter, not a learned one. Nothing in the
training loop rewrites the equation.  Two things that look like they might:

- **Domain randomization** varies `k_hor` by ±30% around whatever nominal we
  give it.  That makes the policy robust to the *value*, not to a wrong
  *functional form*.
- **The world model estimates `d` online** (that is the paper's parameter-
  identification result).  But it is inferring which value inside the
  randomization range this particular drone has.  It cannot discover that the
  equation should use `|v_xy|` instead of `v_xy²`.

So: if the form is wrong, the policy simply learns to fly a drone whose physics
are wrong, perfectly happily.

**For the simulation-only reproduction this is not a blocker.** The simulator
*is* the ground truth there — the policy learns whatever physics we hand it,
and success rate, accelerations and decoded-state accuracy are all still
meaningful. The consequence is narrower: our high-speed drag/lift balance
differs from theirs, so top speeds and lap times will not match exactly.

**It becomes a real problem only at sim-to-real**, where the simulator has to
match a physical drone. That is what the high-speed flight logs in
`docs/datasets.md` §4 are for.

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

### B1. Gate coordinates — 🟡 (was 🔴; recovered by measuring the figures)
The paper states gate **sizes** in text (orange: inner 1.5 m, outer 2.7 m;
MAVLab: 1.5 / 2.1) but never publishes gate **positions**.

They are, however, recoverable. Figures 4 and 6 are matplotlib plots with
metric axes, and their captions read "The black blocks mark the gate locations
with exaggerated thickness". Measuring those blocks in pixels and converting
through the axis ticks gives:

| Quantity | Source | Value |
|---|---|---|
| gate outer size | **text** | 2.7 m — measured **2.71 m** ✓ |
| real-to-virtual gate gap | **text** | 2.7 m — measured **2.66–2.70 m** ✓ |
| ladder flight area | **text** | 6×4 m — measured **~6×4 m** ✓ |
| gate centre altitude | figure | **1.35 m** (a 2.7 m gate standing on the floor) |
| gate positions | figure | X = **−2.0** and **≈ +3.0**, both at Y = 0 |
| gate separation | figure | **5.0 m** |
| gate normals | figure | both along **Y** (the two gate planes are parallel) |

The first three rows are the validation: the same pixel-to-metre calibration
reproduces three numbers the text states independently, so the calibration is
right and the derived positions are good to about **±0.05 m**.

**Remaining uncertainty.** Gate 1's block is clipped by the top of the plot, so
its centre is only pinned to **+2.8…+3.0 (±0.2 m)**. The blocks' thickness is
explicitly "exaggerated" and carries no information. And this is a
*reconstruction*, not published data — the only way to get official coordinates
is to ask the authors.

**Consequence:** we should not expect to match the paper's lap times (3.37 s
etc.) exactly, since those depend on the precise track. Success rate, speeds,
accelerations and decoded-state accuracy should be comparable.

The "big track" (Figure 9) is still not reconstructed — ~9 gates, and its
figure is not dimensioned.

### B2. Invisible-gate placement — 🟡 (was 🔴; the loop one is now pinned)
The paper says "additional invisible gates are added to enforce the correct
flight maneuvers" without saying where or how many. These are what actually
*define* the inverted loop and the ladder.

The **inverted loop's** virtual gate is now pinned, from text and figure
agreeing: "the separation between the actual gate and the virtual gate of
**2.7 m**", and the side view of Figure 6 shows the trajectory crossing the
gate plane at altitude 1.35 m (the gate) and again at **4.05 m** — exactly
2.7 m higher. So the virtual gate sits **directly above the real gate, 2.7 m
up**, and is flown inverted. A half loop of diameter 2.7 m has radius 1.35 m,
matching the text's "almost perfect circle with a radius of roughly 1.5 m".

The **ladder's** virtual gates are still ours. The text constrains them only
as "a full 360° left turn, and flies back over it" with the drone "remaining
mostly within 1 m of the gate". Expect to iterate here.

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

### E5b. Replay buffer size on a small disk — a deviation to record
The paper's `replay.size` is 10e6 steps. On disk that is **~38 GB** (measured
3782 B/step, not the 19x compression a synthetic sample suggested), and
DreamerV3 never deletes old chunks, so `scripts/prune_replay.py` has to keep
the directory bounded.

If the machine cannot hold that, shrink it and **write the number down**:

```bash
./run.sh --replay.size 6e6     # ~23 GB disk, ~28 GB RAM
```

DreamerV3's own default is 5e6, so anything in 5e6..10e6 is well-trodden; it
narrows how far back the world model can sample, which matters most for the
long-horizon parameter identification the paper's phase 2 is aimed at.

### E5. Sensor noise — deliberately left at zero
The ADR literature is clear that the dominant IMU error on a racing quad is
**propeller vibration, not sensor noise**, and there is no standard value to
borrow: it depends on the airframe. Since our goal is to reproduce the paper's
numbers as literally as possible, `GYRO_NOISE_STD` and `RPM_NOISE_STD` stay at
0 and the knobs stay exposed (see B3). Set them from our own flight logs once
the airframe exists, rather than from someone else's drone.
