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

### A1. Horizontal advance ratio `mu` — ✅ RESOLVED (was 🔴)
Both SkyDreamer and MonoRace print

> `mu_{xx,yy} = atan((v_x^B2 + v_y^B2) / (r * w_bar))`

which is a *squared* speed over a speed: the argument of `atan` carries m/s.
Read literally, with `k_hor = 7.245`, thrust goes up 438% at 13 m/s.

**Resolved in favour of `atan(|v_xy| / (r w_bar))`** -- our long-standing
default, `dynamics.ADVANCE_RATIO_MODE = "norm"`. Four independent arguments,
all pointing the same way:

1. **MonoRace's own words.** It calls `mu` "the effective advance ratio of the
   blade". An advance ratio is `V / (Omega R)` -- dimensionless by definition.
   The printed formula is not.
2. **Its sibling equation.** `alpha = atan(v_z^B / (r w_bar))` shares the
   denominator and *is* consistent. The two are the vertical and horizontal
   halves of the same construction, so the horizontal one wants the
   magnitude `|v_xy|`, not the sum of squares.
3. **The fitted coefficient.** `k_hor = 7.245` was identified from real flight.
   Against `|v|` it yields +15% at 5 m/s rising to +46% at 20 m/s -- the range
   a rotor actually gains in forward flight. Against `v^2` it yields +438% at
   13 m/s. Had they fitted against `v^2`, the coefficient would have come out
   near 0.5, not 7.245.
4. **The cited source.** The advance-ratio extension is attributed to a
   separate aerodynamics reference, where the standard definition is `V/(Omega R)`.

Most likely a square root lost in typesetting, propagated from MonoRace into
SkyDreamer (which restates the model rather than re-deriving it: "for full
details, we refer the reader to those sources").

**Residual risk:** none for the simulation reproduction -- the simulator is the
ground truth there. It reappears at sim-to-real, where the term has to match a
physical drone.

### A2. `r` in the `alpha` and `mu` denominators — ✅ RESOLVED (was 🔴)
Never defined in SkyDreamer, but MonoRace -- the source SkyDreamer defers to --
calls it "the propeller radius" and gives the value in a footnote:

    PROP_RADIUS = 0.0485775 m    # not randomized

Note it is *not* the geometric radius of the 5.1" props (0.06477 m) but 0.75 of
it, the 75% blade station -- the conventional reference section in blade
element theory. Described as "estimated", so it is a fitted effective radius,
not a measurement.

We previously used 0.0648 (the geometric radius), 33% too large, which
*under*-stated the translational lift by about a third.

### A3. `k_angle` vs `k_alpha` — 🟢
Table II names it `k_angle`; the force equation names it `k_alpha`. Same
coefficient. We use `k_angle` everywhere.

### A4. `w_i = (w_ci - w_i)/tau` — 🟢
Obvious typo for `w_i_dot`. Implemented as the derivative.

### A5. `k_l` vs `k` — ✅ RESOLVED (was 🟢)
The motor-response equation uses both `k` and `k_l`:

    w_ci = (w_max - w_min) * sqrt(k*u^2 + (1 - k_l)*u) + w_min

but Table II lists only `k = 0.50`. `k_l = k` is forced, not guessed: at
`u = 1` the motor must reach `w_max`, which needs the root to be exactly 1, so
`k + 1 - k_l = 1`. Any other value overshoots or undershoots the parameter that
is named for the maximum. Ferede's public implementation agrees.

### A6. Control period — ✅ RESOLVED (was 🟡)
The paper gives RK4 "with a timestep of 2.2 ms" *and* a 90 Hz control frequency
set by the camera, *and* a rate penalty that divides by `f_c = 90 Hz`. Those
cannot all hold: five substeps of 2.2 ms is 11.0 ms, i.e. 90.909 Hz.

90 Hz over five substeps is 2.2222 ms, so the printed 2.2 is the rounded
figure. We now derive the substep from the control frequency:

    CONTROL_FREQ = 90.0; SUBSTEPS = 5; DT_INNER = 1 / (90 * 5)

Taking 2.2 literally, as we did before, ran the simulator 1% fast -- 172 ms of
drift over a 17 s flight -- and left the reward's `f_c` disagreeing with the
integrator.

### A7. Smoothness-loss action scale — 🟢 (downgraded from 🟡)
`L_smooth = 0.002 * E[||mu_t - mu_{t-1}||^2]` applied to "the mean of the
policy". The paper's actions are `u in [0,1]`, while DreamerV3's
`bounded_normal` head has its mean in `[-1,1]` and `NormalizeAction` maps it
onto the env's `[0,1]`. Since `u = (a+1)/2`, a penalty on `a` is 4x one on `u`.

We apply it to `a`, the head's own mean, which is what "the mean of the policy"
denotes in this codebase -- and SkyDreamer builds on this same codebase at this
same commit, with the same wrapper, so their mean lives in `[-1,1]` too.
Matching them is the point, so `0.002` on `a` it is.

Verified active rather than assumed: `train/actsmooth` runs 1e-4 -> 0.21 over
the 17M-step run, contributing ~4e-4 to a policy loss of order 1e-2.

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

The **big track** (Figure 9) *is* measured, like the small ones — but only
from the PDF. The arXiv HTML drops that figure's top-down and side panels
because they are vector rather than raster; page 14 of the PDF has them, with
metric axes.

Validation is the same kind as for Figures 4 and 6: the measured gate blocks
come out **2.10 m** wide and span altitude **0.17–2.34 m**, against the MAVLab
gate's independently stated outer size of **2.1 m**.

| Quantity | Value |
|---|---|
| top straight | X = +6.0, gates at Y = −3.4 and +2.8 |
| bottom straight | X = −6.0, gates at Y = −3.4 and +2.8 |
| start gate (left end) | X = +3.7, Y = −9.5 |
| middle gate, turned 90° to the rest | X = 0, Y = −0.3 |
| right end | two gates tilted ~45°, Y ≈ +8.5 |
| gate centre altitude | 1.25 m |
| extent | 21 × 12 m |

Gate *size* is deliberately left at the training value rather than the paper's
MAVLab 1.5/2.1, so that only the layout is out of distribution — otherwise the
mask would change too and a failure could not be attributed to either.

Its purpose is an **out-of-distribution test**: the paper claims the flight
plan "potentially enables generalization to arbitrary tracks" and never
demonstrates it. `./run.sh --video --track big` and
`scripts/evaluate.py --track big` fly a small-track policy on it zero-shot.

### B2b. No termination for a diverging simulation — 🔴 (we add one; justified)
The paper terminates on exactly two conditions, gate collision and ground
collision, both with a final reward of zero. There is no rate condition.

We add one: `track.diverged`, at 50 rad/s on any axis. It is load-bearing --
without it a real run died at 12.5k steps on a NaN body rate.

**Why the paper's two conditions cannot cover it.** The moment equation has no
rotational damping at all; its only rate-dependent terms are the gyroscopic
products `J_x q r` and friends, which are quadratic and *destabilising*. So
`dOmega/dt ~ J Omega^2` is a finite-time blow-up, and the time to infinity from
`Omega_0` is about `1/(J Omega_0)`:

| from | blow-up in |
|---|---|
| 17 rad/s (the rate penalty's clip) | 65 ms |
| 50 rad/s (our guard) | 22 ms |
| 100 rad/s | 11 ms |

A tumbling drone at 4 m needs roughly 900 ms to fall to the 0.5 m ground-
collision ceiling. The blow-up is forty times faster, so the ground condition
cannot catch it first and the episode reaches NaN while still airborne.

**The handling matches the paper exactly** -- terminate, final reward zero, move
on -- so the deviation is the existence of the condition, not its semantics.
Terminating is also the right thing on its own terms: there is nothing to
recover from past that point, and letting it run would poison the replay buffer
with divergent states.

**What is still unexplained** is why the paper's policy never reaches those
rates and ours does 69% of the time. That is a question about the plant and the
policy, not about the guard, which only reports the failure.

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

### E5b. Memory — solved by packing, not by shrinking the buffer
The paper's `replay.size` is 10e6 steps. DreamerV3 holds that buffer
**uncompressed in RAM** (`embodied/core/replay.py`, `self.chunks`), and at 4596
bytes/step that is **46 GB** — more than a typical rented instance allocates.

The obvious fix, lowering `replay.size`, *is* a deviation. A better one is not:
the 64x64 mask is binary but stored one byte per pixel, so 4096 of those 4596
bytes carry 4096 *bits*. Packing them eight to a byte gives:

| | before | after |
|---|---|---|
| bytes/step | 4596 | **1005** |
| buffer at 10e6 | 46 GB | **10 GB** |

The agent expands it back before the encoder (`packbits` in
`patches/informed_dreamer.patch`), so the CNN receives bit-identical input.
`test_bit_packing_is_lossless_and_matches_the_agent` checks the env's packing
against the agent's unpacking — two separate implementations — on random
masks, and `test_agent_unpacks_before_the_encoder` checks that the encoder's
declared space is the full resolution.

**This is storage, not modelling**, so the paper's settings stand unchanged.

Disk is separate and smaller: chunks are written with `np.savez_compressed`
(measured 3782 B/step before packing) and never pruned, so
`scripts/prune_replay.py` bounds the directory. See
`SKYDREAMER_REPLAY_DISK_STEPS` in `run.sh`.

### E5. Sensor noise — deliberately left at zero
The ADR literature is clear that the dominant IMU error on a racing quad is
**propeller vibration, not sensor noise**, and there is no standard value to
borrow: it depends on the airframe. Since our goal is to reproduce the paper's
numbers as literally as possible, `GYRO_NOISE_STD` and `RPM_NOISE_STD` stay at
0 and the knobs stay exposed (see B3). Set them from our own flight logs once
the airframe exists, rather than from someone else's drone.
