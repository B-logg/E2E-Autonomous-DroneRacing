# Where this repository departs from SkyDreamer

The goal here is **autonomous drone racing with DreamerV3, in simulation and
then on hardware, driving four motors directly.** SkyDreamer (arXiv:2510.14783)
is the reference implementation of that idea and the starting point for this
code, but reproducing its numbers is not the objective. Where the paper's
recipe does not work, this file records what was changed and what measurement
forced the change.

`docs/paper_gaps.md` is the companion to this file and covers the opposite
case: quantities the paper never states, which had to be chosen rather than
changed. Everything not listed in either file follows the paper, and
`tests/test_paper_conformance.py` holds that line.

---

## 1. Table III's `eps_M_fast` is read in degrees — ✅ changed

**The paper's number**: an angular-acceleration disturbance of 125 (train) and
100 (eval), resampled every timestep.

**What this repository does**: treats those as deg/s², i.e. 2.18 and 1.75
rad/s². Taking them as rad/s² makes the environment unflyable.

### The measurement

A hand-written cascade controller with *exact* knowledge of the dynamics — the
true state, the true randomized coefficients, and an analytically inverted
control allocation — was asked to hold a stationary hover for 10 s. Position
RMS error, in metres, against a gate whose half-width is 0.8 m:

| `eps_u` \ `eps_M_fast` [rad/s²] | 0 | 2 | 5 | 10 | 25 | 60 | **125** |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.06 | 0.07 | 0.09 | 0.13 | 3.02 | 27.3 | **24.8** |
| 0.05 | 0.17 | 0.18 | 0.21 | 0.26 | 20.2 | 17.8 | **30.5** |
| 0.2 (Table III) | 1.42 | 1.44 | 1.45 | 1.40 | 13.5 | 19.3 | **20.0** |

There is a cliff between 10 and 25. The value read as radians sits a factor of
12 beyond it, and in that regime the drone's position wanders by 20–25 m. No
policy, learned or otherwise, flies a 1.6 m gate from there.

Turning the disturbances on one at a time confirms this one is responsible.
Flying the inverted loop, 64 episodes:

| disturbance enabled | flight time | gates |
|---|---|---|
| `eps_a` only | 4.70 s | 1.33 |
| `eps_M_slow` only | 4.89 s | 1.33 |
| `eps_u` only | 5.14 s | 1.06 |
| **`eps_M_fast` only** | **1.47 s** | **0.11** |

Reducing it from 125 to 10 takes the same controller from 1.41 s to 4.03 s of
flight and from 0.08 to 1.00 gates.

### Why degrees is the reading rather than a smaller number

- **It has to be flyable.** The paper's policy completes 25 laps at 100%
  success on hardware. A policy trained where hovering is impossible does not
  transfer to a drone that flies.
- **125 deg/s² = 2.18 rad/s² is the same order as `eps_M_slow = 3`.** Read as
  radians, the per-step disturbance is 42x the slow one; a fast disturbance
  larger than the slow one it rides on is backwards.
- **It is 6.7x the drone's steady-state yaw authority.** The four `k_r1..k_r4`
  coefficients give at most 18.6 rad/s² of yaw. A disturbance that size is not
  a disturbance, it is the dominant term.

### Scope

Only `eps_M_fast` is reinterpreted. `eps_M_slow = 3` and `eps_a_slow = 3` are
left as stated: the table above shows both are harmless at those values, so
there is no evidence to overrule the paper with. If the paper's table turns
out to list all three in the same units, `eps_M_slow` should follow.

`skydreamer/params.py` writes the conversion explicitly (`125.0 * _DEG`) so the
paper's number stays visible in the source, and the conformance test asserts
the converted value.

---

## 2. A hand-written controller supplies demonstrations — ✅ added

**The paper's recipe**: end-to-end RL from scratch, no demonstrations.

**What this repository does**: `skydreamer/expert.py` flies the track with a
classical cascade controller, and those episodes seed the replay buffer.

### Why

A full 17M-step run converged to a **value-accurate local optimum**. The
critic's learned value at 12.5M was 53.3; the analytic discounted return of the
behaviour it had settled on — reach 1.42 gates at 16 m/s, then crash — is 52.3.
The critic was right. The problem was that nothing in 238,071 episodes had ever
shown it anything better:

```
episodes reaching 1 lap (303 steps):     6   (0.003%)
episodes reaching 2 laps:                0
```

Completing a single lap is worth 122 against the 52 it had, and five laps 408.
But given that death at ~1.2 s is certain, return rises monotonically with
speed (45.7 at 5 m/s to 53.0 at 20 m/s), so every gradient points at *faster*.
Faster means arriving at the 1.35 m split-S above 9.4 m/s, which needs more
lateral acceleration than 6.07 g of thrust can produce, so the drone cannot
turn and the episode ends — which is what made death certain in the first
place.

Demonstrations break the loop by putting completed laps in the buffer, so the
critic has evidence that 122 exists. They go into the **replay buffer**, not
into a behaviour-cloning loss on the actor: the actor learns only inside
imagined rollouts, so cloning it while the critic still says 53 is undone as
soon as RL resumes. Seeding replay reaches the world model, the critic and the
actor's imagination start states at once.

The demonstrations do not need to be fast. RL is already demonstrably good at
making a trajectory faster; what it could not do is find a feasible one.

### What the controller is

Waypoint pursuit — three waypoints per gate (approach, centre, exit), a fixed
commanded speed, and a cascade of position → velocity → attitude → body rate →
angular acceleration → four motor commands.

An earlier version tracked a smooth spline with a curvature-limited speed
profile, which is the textbook construction and does not survive this track:
the bearing to a gate is undefined directly above that gate, and the thrust
direction is undefined wherever the trajectory is briefly ballistic, which the
split-S descent is. Both are singularities of the construction rather than
demands on the drone, so flying slower does not remove them. They cost a
reference body rate of 30–70 rad/s against a drone that can produce 17.
`skydreamer/expert_path.py` keeps that construction; `control_pursuit` is what
generates the demonstrations.

### Two things the controller design pinned down about the simulator

Both are properties of the plant, not of the controller, and both matter to
anyone tuning the RL side:

- **The loop bandwidths are forced, not chosen.** The motors are a 30 ms
  first-order lag, so `1/tau = 33 rad/s`, the rate loop must stay well under
  half of that, and each outer loop under a third of the one inside it. That
  gives `k_rate <= 17`, `k_att <= 5`, `k_pos <= 3`. At (22, 12, 9) the drone
  answers a 1.4 m position error by rolling past 115 deg in 120 ms.

- **The transient yaw term is not optional.** `M_z` carries
  `k_r5..k_r8` multiplying `dw`, which looks like authority that exists only
  while the motors are slewing. It is larger than the whole steady-state yaw
  axis: a ±500 rad/s differential produces 21.6 rad/s² through `dw` against
  18.6 rad/s² of total steady-state yaw authority. Every roll or pitch
  correction therefore injects more yaw than the yaw axis can produce on
  purpose. Omitting it from the control allocation made the drone tumble to
  163 deg of tilt under Table III's parameter randomization; including it,
  the same episodes fly to within 53 deg.

---

## 3. Actor and critic are reinitialized; the world model is kept — ✅ done

The 9M checkpoint separates cleanly:

| block | parameters | |
|---|---|---|
| `dyn` + `enc` + `dec` + `rew` + `con` | 8.59M | world model — physics and representation |
| `pol` + `val` + `slowval` | 2.50M | actor-critic — entrenched in the local optimum |
| `opt/state/1/1/<same names>` | 20.5M | Adam moments, named 1:1 with the parameters |

The world model is worth keeping: its decode error was still improving at
12.5M (0.32 m on position, 0.26 m measured on a rendered episode), and it is
policy-independent. The actor-critic is not — the critic has correctly learned
that 53 is all there is, and the actor's reflex is full throttle. Both are
discarded along with their Adam moments, so stale momentum does not fight the
reinitialization. This keeps roughly 5M steps of world-model training while
removing the entrenchment entirely.

No surgery on the checkpoint file is needed. `Agent.load` already takes a
regex, so `run.sh --warm-start <ckpt>` passes
`--run.from_checkpoint_regex '^(dyn|enc|dec|rew|con)/'` and everything it does
not match keeps the fresh initialisation it was just built with. The optimiser
state is left out of the regex deliberately: Adam moments for the world model
are a transient worth a thousand warmup steps, and loading them partially while
the actor-critic's are fresh is the kind of asymmetry that is hard to reason
about later.

`from_checkpoint_regex` had to be added to `configs.yaml`. Three of DreamerV3's
run modules read it and none of them declare it, so any run that set
`from_checkpoint` crashed on the attribute.

Because the step counter starts at zero, the run is the paper's full three
phases again rather than a resumption at 9M. That is the point: a freshly
initialised actor-critic should get the whole entropy and learning-rate anneal,
not start at phase 3's settings.

## 4. Demonstrations are seeded into the replay buffer — ✅ done

`run.sh --demos <dir>` copies the chunks that `scripts/collect_demos.py` wrote
into the run's replay directory before training starts. Three things had to be
true for that to work, and none of them were:

- **A fresh run never reads its replay directory.** `cp.load_or_save()` only
  calls `replay.load()` when there is a checkpoint to restore from, so seeded
  chunks sat unread. `embodied/run/train.py` now calls `replay.load()`
  unconditionally after it; `load` skips chunks it already holds, so this is a
  no-op on resume.

- **The pruner deleted them first.** `scripts/prune_replay.py` keeps the disk
  bounded by deleting the oldest chunks, and demonstrations are written before
  training starts, so they were exactly what it reclaimed. It now reads a
  `.demo-chunks` manifest and refuses to touch anything named in it.

- **The key sets have to match.** `Agent.train` asserts the batch carries
  exactly `self.spaces`, and `Replay._assemble_batch` builds its batch from
  whichever sequence it drew first, so a buffer holding two shapes fails at
  random. Two keys differ between a controller's episodes and an agent's:
  `dyn/deter` and `dyn/stoch`, which `Agent.policy` emits when
  `replay_context > 0`, and `finite/*`, which it emits always. The
  demonstrations fill the first pair with zeros — they are a cache that lets a
  sampled sequence skip re-deriving the latent state for its first
  `replay_context` steps, and the agent overwrites them the first time it
  trains on the chunk — and training no longer stores the second, which is
  written and never read.

### Sizing

The buffer holds 10M items and samples uniformly, and one item is one
environment step. 800 episodes is 1.6M steps, so a 17M-step run evicts them
once about 8.4M steps have been collected — after all of phase 1, which is
when the critic needs them. Shorter runs keep them throughout: 8M added steps
leaves the buffer at 9.6M, under capacity, with demonstrations at 16.7%.

Ongoing demonstration injection during training was considered and left out.
Once the policy produces its own successful episodes, continuing to feed it a
2.5 m/s controller biases it toward flying slowly, which is the one thing RL
here has never needed help with.

---

## 5. Evaluation protocol — ✅ corrected

**What was wrong**: `evaluate.py` took `EnvConfig`'s default `train_fraction = 0.7`,
which is section III-A's schedule for *training* episodes (70% from Table III's
training column, started in front of a random gate). Every evaluation we ran,
through all of the third training run, was therefore 67% training-column
episodes. Table IV describes the *evaluation* column flown from the start gate.

**What it cost**, same weights (the 16.99M checkpoint), 200 episodes x 5 laps,
`scripts/evaluate.py`:

| protocol | success (95% CI) | died within 1 gate | what it is |
|---|---|---|---|
| `paper` | **87.5%** (±4.6) | 7% | evaluation column, start gate, `d_g = 1.0` — Table IV |
| `strict` | 77.5% (±5.8) | 16% | `paper` with the training window `d_g = 0.8` |
| `mixed` | 56.0% (±6.9) | 27% | the old accidental default |

So the old number understated the policy against the paper's own protocol, by
about 30 points. Two caveats go the other way:

- `paper` is a nearly in-distribution test: the evaluation column was already 30%
  of training episodes. Nothing here measures generalisation to another track or
  to hardware (`big` is 0%).
- `paper` uses `d_g = 1.0`, but this track's inner gate is 1.5 m, a half-width of
  0.75 m, so 1.0 is more lenient than the physical gate. `strict` is the number to
  read before trusting a policy on a real drone.

**What changed**: `evaluate.py --protocol {paper,strict,mixed}`, default `paper`;
`run_test.sh --protocol` (repeatable) and a history table that labels each row;
`visualize.py` draws the same distribution. Results for non-default protocols are
written to `evaluation_<protocol>.{json,txt}`. History rows written before the flag
existed are shown as `mixed`.

The breakdown of what *remains* — why `strict` and `mixed` lose episodes, and which
failures training does not shrink — is in the early-failure analysis
(`scripts/diagnose_start.py`, `scripts/analyse_start.py`).

---

## 6. Not changed

- **StochGAN (Appendix B)** is unimplemented. It needs real flight footage,
  which this project does not have. Recorded in `docs/paper_gaps.md`.
- **The reward** is the paper's, unchanged: `W_PROG = 5`, `W_GATE = 30`, the
  rate penalty clipped at `||Omega||_1 = 17`, progress zeroed in the tunnel,
  terminal reward zero. Adding a speed penalty would be the direct fix for the
  over-speeding above and was deliberately not taken, because demonstrations
  address the cause rather than the symptom.
- **`imag_length = 16`**, as section II-G states. The horizon is not the
  problem: braking from 16 to 9.4 m/s takes 1.5 m, which is 8 steps, well
  inside the imagination. What lies outside it is the *payoff* for braking,
  and that is the critic's job.
- **Direct motor control.** CTBR with an inner attitude loop would sidestep the
  stability problem entirely and is explicitly not the goal.

## 7. Our drone and gate (`hw_*` task) -- deliberate, not a reproduction

`./run.sh --hw` trains the same method on our flight-test hardware. Everything is in
`HW_PROFILE` (`skydreamer/hardware.py`), values and sources in `docs/hardware.md` section 5.
Changed from the paper: gate frame 2.7 -> 2.1 m, camera 0.10 m forward, camera field of view
114.6 x 92.15 deg (blank top/bottom rows on the nominal K), nominal thrust-to-weight 6.07 -> 4.3
(estimate), gate window d_g 0.8/1.0 -> 0.65/0.60, tunnel t_g 0.8 -> 0.40. Track layout, the
Table III randomisation, the camera tilt (45-55 deg), the reward and the training recipe are
the paper's. Evaluated on the `physical` protocol, which is ours.
