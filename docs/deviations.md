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
114.6 x 92.15 deg (blank top/bottom rows on the nominal K), nominal thrust-to-weight 6.07 -> 4.3 in run 4, 4.1 from run 5
(estimate), gate window d_g 0.8/1.0 -> 0.65/0.60, tunnel t_g 0.8 -> 0.40. Track layout, the
Table III randomisation, the camera tilt (45-55 deg), the reward and the training recipe are
the paper's. Evaluated on the `physical` protocol, which is ours.

### Run 4 and the next run

Run 4 trained the `hw_*` task from scratch (world model and actor-critic both fresh, hw
demonstrations seeded). Stopped at 8.79M steps: 0% success under `physical`; 75-77% of
episodes finish one lap and then crash. The same checkpoint flown with the paper's
thrust-to-weight (6.07) still finishes no run, so the policy is stuck, not merely
under-powered -- the fast-and-crash optimum run 2 also fell into. Run 3's policy, flown with
only its thrust lowered to 4.3, drops from 91% to 45%.

Run 5 repeats run 3's recipe on the `hw_*` task: run 4's world model (8.79M) kept, the
actor-critic fresh, the same hw demonstrations, and the thrust-to-weight the team gave
(4 x 1.024 kgf / 1.0 kg = 4.10; run 4 and its demonstrations used 4.3, which is inside
the randomised range either way):

    ./run.sh --hw --warm-start warm_hw --demos demos_hw

### Run 5, and run 6: fine-tune run 3 instead

Run 5 was stopped at ~5.0M steps. Its online episode length sat at 160-190 from 1M on
(run 3 at the same points: 440 at 3M, 790 at 4M, 990 at 5M) and every one of 100
`physical` episodes at 3.35M and 4.72M hit the ground: 89% pass three gates, then dive
through gate 3 at 15-16 m/s and never pull out. Peak speed rose between the two
evaluations (13.3 -> 15.0 m/s mean); the thrust-to-weight bands made no difference
(5.5-7: 0%, 2.94 gates). Training on `hw_*` from a fresh actor-critic reached the same
fast-and-crash optimum twice, with and without a warm world model.

Run 6 therefore starts from the policy that already pulls out of the loop: run 3's whole
17M agent -- world model, actor, critic, return normaliser and Adam state, nothing
reinitialised -- fine-tuned on our hardware. The step counter starts at 0 and the replay
buffer empty (only the agent is loaded); no demonstrations. Our conditions are switched on
in stages (`skydreamer/hardware.py: HW_STAGES`, `scripts/train.py: FINETUNE`):

| Stage | Steps | Task | Adds |
|---|---|---|---|
| 1 | 0 - 0.5M | `hw1_` | camera 10 cm forward, our FOV, 2.4 m gate frame |
| 2 | 0.5 - 1.5M | `hw2_` | our 2.1 m frame (run 3 scored 0% on it untrained) |
| 3 | 1.5 - 3M | `hw3_` | our thrust-to-weight, 4.10 |
| 4 | 3 - 7M | `hw_` | our gate window (d_g 0.65 / 0.60, t_g 0.40) |
| 5 | 7 - 9M | `hw_` | the paper's phase-3 settings (actent 1e-5, lr 2e-6) |

Where each stage starts: run 3's 17M policy, untrained, 50 episodes x 1 lap, `paper`
protocol on each stage's task -- hw1 76% (2.54 gates), hw2 18% (1.38; 35 of 50 hit a gate),
hw3 6% (0.76). On the full profile under `physical`, 5% (1.30 gates, 20 episodes).

Stages 1-4 use the paper's phase-2 settings (batch_length 256, lr 4e-5, actent 3e-4): the
settings under which run 3 went from 73% to 91% between 6M and 11M without losing what it
could fly. Starting at phase 3's tiny learning rate would leave the world model unable to
learn the new images.

    ./run.sh --finetune warm_ft

`./run_test.sh` scores any stage on the full profile (`physical`), so the history is one
curve towards the target. `./run.sh --resume` finds the `finetune` file in the run
directory and resumes the stages.

Each phase now ends with a checkpoint (`patches/informed_dreamer.patch`, `run/train.py`).
DreamerV3 saves on a 900 s clock only, so before this every phase boundary threw away up to
15 minutes of training (~70k steps) and the next phase redid them under its own settings.
Runs 3-5 lost that at 8M and 13M; with stages of 0.5-1.5M it would have been a real share.

Known costs. The buffer is never cleared between stages, so data from easier stages
stays in it (about a third at 9M) -- in effect a wider randomisation. A stage change can
cost success for a while; judge by the trend across a stage, not right after a switch.

### Run 6's result, and run 7: our real track

Run 6 finished at 9M: `physical`, 5 laps, 69% (73.5% over 200 episodes), 85% in the sim
alone, 79% in the thrust-to-weight band of the real airframe (3.5-4.5). Most of it came at
the switch to phase 3 at 7M: online episode length went 676 -> 1044 between the 6.5M and
7.0M bins, and success 33% (5M) -> 66% (7.8M), with the airframe's worst point through the
opening p90 0.74 -> 0.55 m. Exploration noise, not the window, was what made it miss.
7.8M -> 9M is within the noise of 100 episodes: it had converged.

The real track is fixed (`track.TEAM_TRACK`, from the team's `gate_map.yaml`): 4 gates in
an 8 x 11 x 3.5 m room, 1.2 m opening, 1.5 m outer, G2 standing on G3 for a 1.5 m-diameter
split-S. Run 6's 9M policy untrained on it, 20 episodes x 1 lap, `physical`:

| Layout | Gate | Success | In the sim | Decoded position error, first 20 steps |
|---|---|---|---|---|
| inverted loop | 2.1 / 1.5 m (its own) | 90% | 100% | 0.3 m |
| inverted loop | 1.5 / 1.2 m (ours) | 15% | 30% | 1.0 m |
| ours | 2.1 / 1.5 m | 0% | 0% | 6.7 m |
| ours | ours | 0% | 0% | 6.8 m |
| ours, rotated and shifted onto the old split-S | ours | 0% | 0% | 12 m |

A new layout fails at take-off (0.2 s), even with the coordinates lined up, and the
world model's position estimate is metres off from the first step: trained on one
track, the flight plan has only ever taken three values. The big track (`hw_big`)
scored 0% at every run 6 checkpoint the same way. So run 7 trains on our track.

It fine-tunes run 6's whole 9M agent (`./run.sh --finetune <ckpt> --team`; schedule
`scripts/train.py: FINETUNE["team"]`), the recipe that took run 6 from 5% to 69% where
training our drone from scratch failed twice:

| Stage | Steps | Task | Why |
|---|---|---|---|
| 1 | 0 - 1.5M | `hwg_inverted_loop` | our gate on the layout it knows (starts at 15%) |
| 2 | 1.5 - 4M | `hwg_team` | our layout (starts at 0%), window 0.60 |
| 3 | 4 - 7M | `hw_team` | the real window, train 0.50 (eval 0.45 = 0.60 - 0.15) |
| 4 | 7 - 10M | `hw_team` | phase-3 settings, run 6's biggest gain |

Stages 1-3 use phase-2 settings, which also put the entropy back to 3e-4: the 9M agent
ended on phase 3's 1e-5, too little to explore a new layout. The window narrows in two
steps because run 6 dipped when its window narrowed at 3M. Leaving the room (walls, or the
3.5 m ceiling 0.5 m above G2) ends the episode; Table III's start 2-4 m before the gate is
pulled 0.3 m inside the room and off the floor.

Run 7 is evaluated on our track only (`./run_test.sh` defaults to `team`, 1 and 3 laps;
no big track). Checks, against `physical` on `team`: stage 1 is judged by its training
log (episode length recovering, as run 6's did after each switch), since the track
there is not ours; at 4M our track should finish one lap in the sim at least 30% of the
time (otherwise look at the split-S before going on); at 7M, 60% in the sim (run 6 had
71% before its phase 3). Every run now keeps its checkpoint each 1M in `<run>/snapshots/`
(`scripts/milestone_snapshots.py`), since run 6's 7M was lost for want of one.

Fallback: demonstrations on our track. Runs 4 and 5 had demonstrations and still failed;
run 6 had none and worked, so run 7 starts without. If the 4M check fails they are the first
thing to add, into the running run's buffer:

    ./run.sh --collect-demos --team          # -> demos_team, CPU, minutes
    ./run.sh --resume --demos demos_team

The controller (`scripts/collect_demos.py --task hw_team`) flies the spline through the
gate centres. Screened from the start gate, 512 seeds: at 2.5 m/s 7.2% finish a lap clean,
almost all failures on the way from G2 to G3 (the split-S, path radius 0.41 m); at 2.0 m/s
18.8% (17.0% two laps), the default for `team`. Widening the path (tangent scale 2-3, lead
0.3) did not help, and 1.5 m/s is too slow to fit two laps in 2000 steps. Mixed training
starts, as `--collect-demos` screens them, keep about 11%: 4000 seeds give ~400 episodes.

Known risks. How fast a fine-tune learns a new layout is the open question. The split-S
radius is 0.75 m against the old 1.35 m: at the bottom of the loop, (TWR - 1) g = v^2 / r
allows ~4.8 m/s at TWR 4.1, where run 6 flies 10-14 m/s. Below TWR 3.5 run 6 already
succeeded only 51% of the time.

### Run 7's 4M check, and run 7b: a speed limit for the room

Run 7 failed its 4M check, on exactly the risk above. `physical` on `team`, 100 episodes:
0% at 1 and 3 laps, also 0% in the sim; 82 flights passed G1 and G2, then 75 ended on
a wall or the ceiling and 25 on the floor. Mean peak speed 15.5 m/s (max 25), mean
flight 1.8 s. 3.4M was the same (0%, 1.93 gates). The stages chained correctly (each
loaded the previous stage's checkpoint); the policy learned to charge G2 at ~14 m/s,
which no split-S to G3 1.5 m below can follow in a room with 3 m past G2. Training
episodes almost all ended in a crash, and the training score had flattened (52, 55, 58,
57 from 3.25M to 4M). Stage 3 only narrows the window, so run 7 was stopped at ~4.3M.

Run 7b (`./run.sh --finetune <run 7 4M snapshot>/ckpt --team2 --demos demos_team`, 8M):

- **Speed limit.** Not in the paper. On tracks in `track.V_CAP` (`team`: 4.0 m/s, under
  the ~4.8 m/s the split-S allows) each step pays `W_SPEED * (|v| - v_cap)` above the cap,
  `W_SPEED = 2 W_PROG / f_c`: twice the progress the extra speed can buy, so the reward
  peaks at the cap. It is in the reward, not the dynamics: the policy may still exceed it.
  It applies to every task on `team`, so evaluation is unaffected (reward is not scored).
- **Demonstrations from the start**, the planned fallback (`--collect-demos --team`,
  2.0 m/s, under the cap): the first successful split-S flights the buffer sees.
- **Schedule** `team2`: `hwg_team` (window 0.60) to 3M, `hw_team` (0.50) to 6M, phase 3 to
  8M. From run 7's 4M agent rather than run 6's: it already knows our gate and the first
  two gates of our layout.

Checks, as run 7's: at 3M our track should finish one lap in the sim ≥30% of the time,
and mean peak speed should be near the cap rather than run 7's 15 m/s; at 6M, ≥60%.


### Run 7b's result, and run 7c: the real window and a steeper speed penalty

Run 7b, `physical` on `team`, 1 lap (8M: 200 episodes, the others 100):

| step | physical | sim only | ended on the ground | sim-clean, real gate fails |
|---|---|---|---|---|
| 4.6M | 10% | 33% | 47% | 23% |
| 6.05M | 24% | 55% | 44% | 31% |
| 7.34M | 41% | 65% | 33% | 24% |
| 7.58M | 38% | 59% | 37% | 21% |
| 8.0M | 39.5% | 66.5% | 30.5% | 27% |

Phase 3 (from 6M) took it from 10% to ~40%, flat since 7.3M. 3 laps: 6.0% physical,
which is 0.395^3 -- each lap fails independently, nothing particular to lap 2. Its two
failures, about equal: (1) the split-S exit, 61 of 200 on the ground, 18 before G3 and
42 just after it; the renders show the descent never arrested, at up to 8.5 m/s against
the 4 m/s cap. (2) sim-clean flights the real gate rejects, 54 of 200, at G4 (37) and
G3 (20), both 0.75 m up, right after the split-S.

Run 7c (`./run.sh --finetune <run 7b>/ckpt --team3`, 3M):

- **Train window = eval window, 0.45 m** (hardware `hwp_`), against (2). Run 7b trained
  at 0.50 against the real 0.45.
- **Speed penalty `W_SPEED_STRICT = 5 W_PROG / f_c`**, from 2, against (1): gravity's
  speed in the descent was worth taking at twice the progress.
- **Phase 3 throughout**, not phase 2 then 3: the agent flies the track already, and
  phase 2's entropy would spend the precision phase 3 bought.
- **No demonstrations.** The agent does not need them, and run 7b, which carried 1.5M
  demonstration steps in its buffer, ran out of RAM at 7.62M (with an evaluation running
  alongside).

`hwp_team` is scored as `hw_team`, so the numbers line up with the table above. Checks:
at +1M, sim-only ≥55% (else the penalty was too steep: restart at 3x); at +2M, ground
<20% and physical ≥50%; the aim at 3M is physical ≥60%.

### Evaluation fix: a gate flown past could count as passed

Under `--protocol physical` the sim's own window was 3 m (off), and the real-gate
verdict came from `hardware.check_flight` alone. That check leaves a centre crossing
more than `near` = 1.4 m from a gate to another gate (G2 sits 1.5 m above G3), so a
flight that went past a gate 1.4-3 m to the side had the gate counted by the sim and
was never seen by the check: a success. Such crossings happened -- the largest sim
gate error in run 7b's 1-lap evaluations was 1.5-2.4 m, 3.2 m at 3 laps -- though
whether any of them were in a "successful" flight is not in the saved results.
The sim window under `physical` is now 1.0 m: a crossing past it ends the flight as a
gate strike before the check could miss it, and any centre beyond ~0.45 m fails the
0.6 m opening anyway, so no real pass is cut short. Run 7b's numbers above were taken
before the fix; its 8M checkpoint is re-scored with it for comparison with run 7c.

Training was never affected: it runs at the 0.50 / 0.45 m window, where a crossing
outside ends the episode as a crash.

The renders' side view drew every gate as a vertical line at its centre, which is
right only for a gate facing along Y; our room's gates face along X and are seen
face-on there. It now draws the outer frame and the opening as squares, squeezed by
how far the gate faces the viewer.

### Run 7c's result, and run 7d: the real floor

Run 7c, `physical` on `team`, 1 lap, 200 episodes (run 7b's 8M re-scored under the
fixed protocol for comparison):

| | 7b 8M | 7c 1M | 7c 2M | 7c 3M |
|---|---|---|---|---|
| physical | 37.5% | 13.0% | 28.5% | 25.5% |
| sim only | 59.0% | 42.5% | 44.5% | 41.0% |
| ended on the ground | 58 | 53 | 90 | 104 |
| sim-clean, real gate fails | 43 | 59 | 32 | 31 |
| worst point p90 / p99 [m] | 0.654 / 1.009 | 0.816 / 1.118 | 0.633 / 0.959 | 0.607 / 0.881 |
| 1-lap time [s] | 4.53 | 5.64 | 5.46 | 5.41 |

The 0.45 m window sharpened the gates (real-gate fails 43 -> 31, p99 1.01 -> 0.88).
The 5x speed penalty did the opposite of its purpose: slower over G2, the policy
dropped straight down the split-S instead of looping, and ground deaths doubled
(stopped after 2 gates: 19 -> 44). Both were changed at once, which the next run avoids.

Every render of 7b and 7c ended at exactly 0.5 m: section II-C's ground test
(`z_w > -0.5` while descending or banked past 60 deg, for the paper's podium) covers
the bottom third of G1, G3 and G4's openings (0.15-1.35 m). Some of those flights
were falling at 5-6 m/s and would have hit the floor anyway; how many were not is
what the per-episode dump (`evaluation_*_episodes.json`, every flight's end state and
gate crossings) now records.

Run 7d (`./run.sh --finetune <run 7b>/ckpt --team4`, 2M): from 7b's 8M, phase 3,
- **the floor that is there**: `track.FLOOR`, a crash when the airframe's lowest prop
  tip (`hardware.REACH`, 0.19 m, times the sine of the tilt) comes within 0.10 m of
  the floor. It applies to every task on `team`, evaluation included;
- **7c's window** (train 0.45 = eval, hardware `hwr_`), **7b's speed penalty**.

### Run 7d's result, and run 7e: the real launch and a floor penalty

Run 7d, `physical` on `team`, 1 lap, 200 episodes, mid-air start (the protocol every
run so far was scored under; `evaluate.py --air-start` keeps it):

| | 7b 8M | 7d 1M | 7d 2M |
|---|---|---|---|
| physical | 37.5% | 33.0% | 32.0% |
| at our drone's TWR (3.5-4.5, n=63) | 49.2% | 50.8% | 39.7% |
| sim only | 59.0% | 55.5% | 49.5% |
| on the floor after G3 | 28 | 45 | 56 |
| on the floor before G1 (the start) | 30 | 28 | 27 |
| G3 crossing altitude, finishers / floor deaths [m] | 0.84 / 0.72 | 0.80 / 0.65 | 0.73 / 0.64 |

Run 7d got worse as it trained. Its training score fell 134 -> 81 and its episodes
shortened 335 -> 242 steps, while the world model's position and velocity losses rose
(0.027 -> 0.030, 0.163 -> 0.186) and so did its continuation loss (0.0300 -> 0.0320).
Under the real floor the policy learned to fly lower -- G3 0.85 -> 0.71 m, crossed at
6.9 m/s and still sinking 2.7 m/s -- and the world model, which has no altitude
sensor and is ~0.3 m off, does not foresee the floor, so imagination rated it safe.
Section II-C's ground test had kept 7b off it: it ended any flight descending below
0.5 m. Run 7e starts from 7b's 8M, not 7d's.

**The start was not the real one.** Table III starts 0.7-1.3 m below the start gate;
G1's centre is 0.75 m up, so the room clip put every evaluation flight at 0.3 m,
falling, banked up to 20 deg -- 14-25% of evaluation flights died there, and they
say nothing about the real drone, which takes off from the floor. `track.LAUNCH` now
starts every evaluation-column flight (training's 30%, and all of evaluation) at rest
on the floor in front of G1, level, motors at idle, facing it. The team can put it
anywhere 1-2 m out; 2.0 m was chosen:

| distance from G1's plane | 1.0 m | 1.5 m | 2.0 m |
|---|---|---|---|
| climb angle to G1's centre | 35 deg | 25 deg | 19 deg |
| G1 pixels in the first frame (cam 50 deg) | 402 | 185 | 118 |
| within the 2.0-2.7 m every past start used | no | no | yes |

A TWR-2.4 airframe reaches 0.75 m in 0.37 s, so the climb is no limit at any of them.
What decides it is the world model, whose first 20 steps are its worst (0.36-0.51 m):
2.0 m gives it the most time before the first gate, at the approach it already knows.
Training draws 1.8-2.0 m, +-0.15 m lateral, +-5 deg yaw for placement error.

The floor holds the drone until its thrust lifts it, without sliding or turning; a prop
tip on the floor or a touchdown faster than 1 m/s (II-C's v_z) is a crash, and above
0.29 m (`FLOOR` + reach) `floor_collision` takes over. A launch can be flown: a cascade
controller on ground truth, under evaluation randomization and disturbances, climbing
level to 0.25 m first, lost 2 of 200 takeoffs (0 without the moment disturbances);
pushing toward the gate from the floor lost 7-12%. Run 7d's 2M lost 148 of 200
(8.0% physical): it lifts off at 0.09 s and pitches 2-4 deg per step toward the gate,
as if mid-air, and a prop tip meets the floor at 6-8 cm and 19-28 deg, 0.13 s in.
Mid-air training starts keep Table III's range above `AIR_START` (0.6 m).

**The floor penalty** (`track.floor_penalty`, hardware `hwf_`) charges sinking faster
than the drone could still stop above a buffer. It was calibrated on run 7d's 2M
flights (`evaluate.py --dump`: ground-truth p, v, q per step), after G2, where the
floor deaths happen:

| buffer, decel | finishers pay | floor deaths pay | deaths above finishers' p90 | warning before impact (median / p10) |
|---|---|---|---|---|
| 0.5 m, 12 m/s^2 (physics alone) | 36.5 | 77.7 | 73% | 0.65 / 0.36 s |
| 0.2 m, 12 m/s^2 | 8.0 | 41.6 | 85% | 0.38 / 0.23 s |
| **0.2 m, 20 m/s^2** | **4.1** | **34.1** | **100%** | **0.29 / 0.19 s** |
| 0.1 m, 20 m/s^2 | 1.4 | 22.2 | 100% | 0.23 / 0.13 s |

(m/s over the limit, summed over steps.) Finishers cross G3 sinking 2.0 m/s at a
0.73 m prop tip, so a split-S that bottoms just past G3 is normal flying; the first
choice, from the stopping physics alone, would have charged 9 finishers in 10. At
(0.2, 20) and W_FLOOR = 5x the progress per m/s, a finisher pays 1.1 a flight and a
floor death 9.5 -- a third of a gate -- starting 26 steps before the impact.

Run 7e (`./run.sh --finetune <run 7b>/ckpt --team5`, 3M): from 7b's 8M, phase 3,
7d's window, speed penalty and floor, plus the launch and the floor penalty. Snapshots
every 0.5M, each scored with the launch, so the best point can be kept.

### Run 7e's result, and run 7f: a mask our gate survives, the airframe at the gate, a floor margin

Run 7e, `physical` on `team`, launched from the floor, 200 episodes:

| snapshot | 0.5M | 1.0M | 1.5M | 2.0M | 2.5M | 3.0M |
|---|---|---|---|---|---|---|
| 1 lap | 33.5% | 22.5% | 23.5% | 34.5% | 37.0% | 48.5% |
| 3 laps | 5.5% | 2.0% | 0.0% | 6.5% | 5.5% | 11.5% |
| on the floor before G1 | 51 | 43 | 35 | 29 | 21 | 10 |

It learned to take off (7d's 2M: 8.0% from the floor) and was still improving at 3M.

**The mask erosion wiped our gate.** Section II-F erodes the 64 x 64 mask by 1 pixel
half of the time (3x3 average pooling, thresholded) because real masks came out
thinner -- on the paper's 0.6 m frame. Ours is 0.15 m: 1.9 px at 2 m, 0.9 px at 4 m,
and the 3x3 test removes anything under 3 px, so with erosion on, every gate further
than ~1.25 m was gone (a frame 2 m before G2: 110 pixels -> 4). Run 7e's 3M re-scored
(`evaluate.py --erosion`, same seeds, T4):

| | erosion as trained (0.5) | erosion off |
|---|---|---|
| 1 lap | 45.0% | **65.5%** |
| 3 laps | 6.0% | **36.5%** |
| world model position error (whole flight) | 0.248 m | 0.202 m |

(The 3-lap 6.0% against the A5000's 11.5% at the same seeds is evaluation noise between
machines, ~2 standard errors at n = 200.) `track.ERODE_PX` now thins our frame in the
renderer by 0.25 px per edge -- the same fraction of a 0.15 m frame as the paper's
1 pixel of 0.6 m -- with the paper's probability and hold. Over random poses in front
of G1 it keeps 84 / 67 / 38 / 19% of the frame at 1 / 2 / 4 / 6 m (the expected
(w - 0.5) / w of a w-pixel frame), where the 1-pixel erosion kept 43 / 1 / 0 / 0%.
It applies to every stage on our track, evaluation included.

**What 7e still loses, with erosion off** (3 laps, 200 flights, 127 failures): the floor
between G3 and G4, 54; the airframe past the real opening, 64 (G2 22, G4 20, G3 17,
G1 5); the rest, 9.

*The floor.* Low flight there is a habit, not an accident: half of the G3 -> G4 legs
that survived took the lowest prop tip under 0.45 m and 16% under 0.25 m. The crashes
crossed G3 lower (prop tip p50 0.48 m against 0.66 m) and sinking faster (1.53 against
0.41 m/s); speed did not separate them (6.6 against 6.95 m/s). `floor_penalty` charges
only a sink, so level flight 0.2 m up was free while the world model was 0.2 m off.
`track.altitude_penalty` (hardware `hwa_`): `W_ALT` = 3 per metre of the lowest prop tip
under 0.45 m, per step, once the flight has been above it (a launch climbs through it
for free). On 7e's 3M flights:

| margin | 3-lap finishers charged | m-steps under, finishers / floor crashes | charge starts before a crash (p50 / p10) |
|---|---|---|---|
| 0.35 m | 77% | 2.65 / 5.28 | 0.23 / 0.12 s |
| 0.40 m | 89% | 5.12 / 7.91 | 0.27 / 0.14 s |
| **0.45 m** | 90% | **8.96 / 11.24** | **0.31 / 0.17 s** |
| 0.50 m | 100% | 14.78 / 15.34 | 0.33 / 0.19 s |

It is meant to charge the finishers too -- they fly where the crashes do -- and at
0.45 m it costs 7e's finishers ~9 a lap against ~250 a lap of reward (the 2000-step
training episodes peak at 1149), while starting 28 steps before a crash, beyond the
actor's 16-step imagination. 0.45 m keeps the centre inside G1/G3/G4's window. The
floor penalty is doubled (`W_FLOOR_STRONG`): it already singled out the crashes
(9.0 against 2.8 a flight) but a crash paid a third of one gate.

*The airframe.* Training scored and ended the real plane's crossing on the centre of
mass against d_g = 0.45; the evaluation fails a crossing when any point of the
airframe passes outside the 0.6 m opening, and the airframe reaches 0.08-0.19 m
(p10-p90) beyond its centre. `track.swept_offset` carries each footprint point
(`hardware.footprint`) along the centre's velocity to the plane; against
`check_flight` on 1823 of 7e's crossings its error is -0.015..+0.009 m (p5..p95) and
pass/fail agrees on 99.8%. Under `hwa_` the real plane pays `1 - w / 0.6` and is a
strike past 0.6; the virtual pre/post planes keep d_g. Of the 83 crossings that failed
the airframe test, the old training test let 18 through; the other 65 were already
strikes in training, which says the rest of the precision gap is the centre's spread
(CoM offset p90 0.46 m), i.e. how well the agent knows where it is, not the reward.
The real-plane reward falls from a mean 0.54 to 0.41 per crossing (correlation 0.93).

The speed penalty is unchanged: it does work against the reward's preference for
arriving sooner (the discount), but speed did not separate failures from finishers
(G2 5.43 against 5.25 m/s, G3 6.6 against 6.95), and run 7c's 5x penalty on |v|
doubled the floor deaths by penalizing the split-S's fall. If speed still matters after
7f, the penalty should be on horizontal speed only.

Run 7f (`./run.sh --finetune <run 7e 3M>/ckpt --team6`, 5M): from 7e's 3M, phase 3,
hardware `hwa_`. If it removed the 18 missed airframe strikes and half the floor
crashes, 7e's 127 failures would be ~82, 59% over 3 laps: 80% needs the centre's
spread at the gates to shrink as well.

### Run 7f's result, and run 7g: a margin at the gate

Physical protocol, floor launch, our track. 7e is its 3M with the erosion off; 7f is
evaluated with the 0.25 px thinning on (a slightly harder mask). First failure per
flight; "per crossing" is the airframe's worst point past the 0.6 m opening.

| | 7e 3M (n=200) | 7f 3M (n=100) | 7f 4M (n=200) |
|---|---|---|---|
| 1 lap | 65.5% | **82.0%** | 70.5% |
| 3 laps | 36.5% | **42.0%** | 38.5% |
| 3 laps, ground first | 30% | 14% | 14% |
| 3 laps, airframe first | 32% | 42% | 46% |
| per crossing past 0.6 (3 laps) | 4.6% | 5.2% | 6.8% |
| G2 per crossing (1 lap) | 8.2% | 6.3% | 13.8% |
| lap 1 median | 4.41 s | 5.04 s | 4.73 s |

The altitude margin worked (ground first 30% -> 14%, z ~3.0; on the way to G4 27% ->
6%) and held at 4M. The airframe test changed nothing per crossing, and G2 got worse
from 3M to 4M while the lap got faster again (G1->G2 1.62 -> 1.50 s; G2 lateral sd
0.19 -> 0.24 m). The one-lap drop 82.0 -> 70.5% is z ~2.2.

New in `evaluate.py`: the decoded and true centre in the gate frame at every
crossing. At 4M, G2 (3 laps, 509 crossings): true y +0.126, decoded +0.082 (corr
0.81-0.91 per lap); true z +0.015, decoded +0.104. Decode-minus-truth sd 0.12 / 0.13
against a true spread of 0.23 / 0.24: estimation is ~a quarter of the variance. Most
of the lateral offset is the line the policy chooses while knowing it; the vertical
bias (~0.09 m) is estimation, which a reward cannot fix.

Run 7f put a cost before the floor and the floor stopped killing it; at the gate it
only moved the reward's zero and the strike to the airframe, the same terminal
signal the floor had in 7e. Run 7g puts the same shape at the gate (hardware
`hwm_`, `track.gate_margin_penalty`): 200 per metre of the airframe's worst point
past 0.45 m at the real plane, so a crossing at the opening costs -30 more (one
gate). On 7f's 3M flights 22% of crossings pass 0.45 m, by 0.023 m on average: ~18
a lap, against the altitude term's ~9 and ~250 of reward. Everything else is 7f's.

Run 7g (`./run.sh --finetune <run 7f 3M snapshot>/ckpt --team7`, 3M, snapshots every
0.5M): from 7f's 3M, phase 3. 7f peaked at 3M of fine-tuning, so the best snapshot
is picked by evaluation. If G2 came down to G1's rate (~3% per crossing), 3 laps
would be ~50%; 80% needs ~1.5% per crossing at every gate and the vertical estimate.

Evaluation fixes: the `worst point offset` percentiles skipped gate index 1 (the
inverted loop's invisible gate, i.e. G2 on `team`); they now skip invisible gates.
`run_test.sh` no longer stops on the per-flight `*_episodes.json` dumps.

### Run 7g's result, and how checkpoints are compared from now on

Run 7g's snapshots (physical, floor launch, seed 10 000, n = 100 each; 7f's 3M for
reference). Swept/x is per real-gate crossing with the worst point past 0.6 m.

| | 1 lap | 3 laps | first: swept | first: floor | first: takeoff | swept/x | G2 | lap |
|---|---|---|---|---|---|---|---|---|
| 7f 3M | 82% | 42% | 42% | 10% | 4% | 5.2% | 10.0% | 5.00 s |
| 7g 0.5M | 77% | 39% | 36% | 16% | 7% | 4.8% | 7.4% | 5.02 s |
| 7g 1.0M | 71% | 27% | 50% | 16% | 6% | 7.3% | 14.8% | 5.00 s |
| 7g 1.5M | 74% | 43% | 34% | 15% | 4% | 4.9% | 6.8% | 4.89 s |
| 7g 2.0M | 63% | 20% | 60% | 14% | 4% | 10.6% | 18.7% | 4.63 s |

The margin did not do what it was for: at its best snapshots (0.5M, 1.5M) swept-first
fell 42 -> 34-36% and floor-first rose 10 -> 15-16% (most of it between G3 and G4),
the same 3-lap total. The snapshots also swing far more than n = 100 explains: 1.5M ->
2M is 43 -> 20% on the same flights (McNemar p < 0.001). Run 7g was stopped at 2M.
The best policy is 7f 3M or 7g 1.5M, which these numbers cannot separate (paired
difference -1%, 95% CI -14 to +12 points).

Two things in that data change how checkpoints are scored:

- **Every checkpoint flies the same flights.** evaluate.py deals them from
  `key(seed + k)`, so at the same (seed, episodes) each checkpoint gets the same
  airframes, starts and noise. Flights 43 and 89 crashed on takeoff under five of
  six checkpoints. Records now carry `seed`, `index`, the flight's `params` and
  `start`, and `scripts/compare_evals.py` pairs checkpoints flight by flight
  (McNemar's exact test on the flights they disagree on).
- **Pairing does not rescue n = 100 at 3 laps.** 3-lap outcomes under two
  checkpoints are nearly independent (7f 3M vs 7g 1.5M: both succeed on 20 flights
  where independence predicts 18), so the paired interval is as wide as the
  unpaired one, about +-13 points. Resolving 10 points needs ~400 flights.

`./run_compare.sh NAME=<snapshot> NAME=<snapshot> ...` scores each checkpoint on 400
new flights (seed 20 000) at 1 and 3 laps with `--dump`, which now also keeps the
motor commands `u` and the decoded position `dp`, then writes the comparison and
`scripts/diagnose_failures.py`'s report for each. That report splits the floor
failures that every 7x run has had:

- **Takeoff:** the tip touches once the airframe leans past asin(alt / 0.19 m), 22 deg
  at 7 cm. It reports how far every takeoff leans while low (a habit, with the
  crashes as its tail, or a few bad draws) and ranks the flight's draws -- moment
  coefficients, thrust-to-weight, start -- by how well they separate the crashes.
- **In flight:** at the point of no return (track.floor_penalty's stop, at the
  flight's own thrust-to-weight) a crash is `late` (already near full throttle),
  `estimate` (decoded >= 0.15 m higher than true) or `policy` (knew, and kept
  sinking), next to the finishers' lowest point on the same leg.

Those call for different fixes, so the next run waits for that report.

### The 400-flight comparison, and run 7h: no leaning before the airframe is off the floor

`./run_compare.sh` on 7f's 3M and 7g's 1.5M (400 new flights each, seed 20 000,
identical flights for both; physical protocol, floor launch):

| | 1 lap | 3 laps (95% CI) | swept/x (3 laps) | first: swept | floor in flight | takeoff |
|---|---|---|---|---|---|---|
| 7f 3M | 76.8% | 46.2% (41-51) | 4.4% | 35.5% | 11.7% | 4.2% |
| 7g 1.5M | 75.5% | 42.2% (38-47) | 6.0% | 42.2% | 7.2% | 5.8% |

Paired, 7f minus 7g is +1.2 points at one lap (95% CI -3.8 to +6.3, McNemar p
0.70) and +4.0 at three (-2.3 to +10.3, p 0.25): no ranking at the flight level,
and the airframe failures per crossing are higher under 7g (244 of 4035 against
178 of 4061). The gate margin is dropped; run 7h starts from 7f's 3M.

What the trajectories say about each failure (7f's 3M, 3 laps):

- **The airframe at the opening (35.5%).** At the real plane the centre is 0.52 m
  out on failing crossings, 0.20 m on the others, and the rate climbs with how
  obliquely it crosses (1.5% under 10 deg, 10.5% at 30-45, 26% beyond 45). At G1,
  G2 and G4 the decoded centre was at least as far out as the true one: the agent
  knew. At G4, 44 of its 57 failures are the top edge, on flights that came down
  the G3 -> G4 leg high (the rate is 1.6% when that leg's lowest tip is under
  0.3 m, 9.9% when it is over 0.6 m). Only G3, out of the split-S under G2,
  is an estimate: the decoded centre 0.24 m less far out than the true one.
- **The floor in flight (11.7%; G3 -> G4 5.5, G4 -> G1 5.0).** At the point of no
  return 30 of 47 are `estimate`: the agent decoded itself 0.17-0.24 m (p50)
  higher than it was, where finishers at their lowest point decode +0.01. A 0.15 m
  higher line would have saved ~80% of them, but raising the altitude margin to
  0.60 m charges finishers 33 a lap at today's lines (0.45 m: 8.6) and pushes
  the G3 -> G4 leg up into G4's top edge, which is already failing more the
  higher that leg is flown. Not changed in 7h.
- **Takeoff (4.2%).** Every crash's tip goes under the floor while leaning
  about as far as the takeoffs that clear (25 vs 27 deg), but on low-thrust
  airframes (TWR 2.8 vs 4.0; k_w rank-biserial -0.53) with a higher command and
  more motor spread. 14 of 400 flights crash under both checkpoints. Takeoffs that
  clear keep the tip above 2 cm in 93% of flights.

Run 7h (`hwl_`, `./run.sh --finetune <run 7f 3M snapshot>/ckpt --team8`, 2M,
snapshots every 0.5M): 7f's settings plus `track.launch_tip_penalty`, 600 per metre
of the lowest tip under 2 cm per step until the airframe is clear of the floor:
~31, one gate, for the dip that precedes a typical crash; nothing for a straight
climb. One change, so 7h's snapshots can be read against 7f's 3M on the same 400
flights (`./run_compare.sh 7f3M=... 7h0.5M=...`; results already in `compare/`
are not scored again). If it works, the takeoff share falls toward zero: ~3 points at one lap and
~2 at three (a saved takeoff still has to fly the laps); it does not touch the airframe at the opening or
the floor in flight, which are 47 of the 54 points still missing at 3 laps.

### Run 7h's result, and run 7i: an approach line for each gate

Run 7h's 1.5M against 7f's 3M on the same 400 flights (seed 20000, physical):

| | 1 lap | 3 laps | airframe at the opening, per crossing | takeoff |
|---|---|---|---|---|
| 7f 3M | 76.8% | 46.2% | 4.4% | 4.2% |
| 7h 1.5M | 60.8% | 25.0% | 9.6% | 6.8% |

Paired, 7f minus 7h is +16.0 points at one lap (+10.8 to +21.2) and +21.2 at three
(+15.6 to +26.9), McNemar p < 0.001 both. The takeoff did not change: tilt at steps
5/10/15/20 is 0.2/5.7/11.6/16.1 deg (p50) against 7f's 0.1/5.4/12.9/15.0, and the
tip comes within 1 cm on 2.6% of flights against 2.8%. 13 flights crash on takeoff
under 7h only, 3 under 7f only. Like 7g, and like 7f's own 4M, 7h flew faster
(3.94 -> 4.12 m/s mean; G1 -> G2 in 1.48 s against 1.66) and less precisely. 7h is
stopped; 7f's 3M stays the base.

**Where three laps are lost (7f 3M, 4,057 real crossings).** For 80% over three
laps the airframe has to clip the opening on ~1% of crossings (now 4.3%: 0.957^12 =
58% of flights survive twelve) and the floor take ~1% a lap (now ~4%). Most clips
are near misses (61% of failing crossings are at 0.60-0.70 m), and the spread, not a
bias, makes them: the mean offset per gate is under 0.05 m, the spread 0.17-0.24 m,
and the decoded offsets spread as far (estimation error 0.08-0.14 m). The policy
flies those lines on purpose.

A crossing whose centre is within d_g at both of 7f's training planes (+-0.4 m)
clipped the opening once in 3,080. The pre-plane is where it goes wrong:

| | G1 | G2 | G3 | G4 |
|---|---|---|---|---|
| pre-plane broken (> 0.45 m) | 4% | 25% | 12% | 12% |
| clipped when broken | 26% | 18% | 24% | 49% |
| clipped when not | 1.3% | 2.5% | 0.8% | 0.5% |
| broken toward | the side | below, G1's side | above (split-S) | above |

The breaks hardly depend on the airframe (|rank-biserial| <= 0.16 for every
parameter). They follow II-C's progress term, the straight-line distance to the
pre-plane's centre: from G1 that line meets G2 31 deg off its normal and 1.5 m
below; the flights cross G2's pre-plane 29 deg off (p50), 0.47 m beside and 0.38 m
below its axis a metre out. G2 -> G3 is a vertical drop onto a vertical gate.

**Run 7i** (`hwi_`, `./run.sh --finetune <run 7f 3M snapshot>/ckpt --team9`, 2M,
snapshots every 0.25M): 7f's settings plus `track.APPROACH`, a line along each
gate's normal from G1 0.4, G2 1.0, G3 0.8, G4 1.2 m out to the pre-plane.
`track.approach_distance` (to the line's outer end, then along it) replaces the
straight line in the progress term; it is continuous, so a leg still pays start
minus end -- it changes which way is forward. `track.approach_reward` pays 15 (half
a gate) times 1 - offset / d_g, clipped to [-1, 1], once per gate where the line
starts; it does not end the flight. The lengths: G1's U-turn from G4 leaves no room
past 0.4 (it already comes within 0.16 m of the wall); G3's split-S reaches 0.8 m
back on 93% of laps, 1.0 m on 75%; G2 and G4 have room (no flight went past
x = 6.04 m or above 3.25 m). On 7f's 3M flights, crossings on the line where it
starts (<= 0.45 m) clip the opening on 1.0% (G2), 1.1% (G3), 0.5% (G4) against
8.5%, 6.6%, 16.0% off it, and the reward would have paid them 6-12 more than
the ones that clipped.

Expectation: if the pre-plane breaks at G2-G4 fell to G1's 4%, twelve crossings
would survive 75% instead of 58% (86% with none); with today's takeoff and floor
that is the mid-60s over three laps, not 80. The floor's estimation error (30 of 47
floor crashes decode 0.17-0.24 m high) and the airframe model behind the takeoff
(3-lap success 19% at TWR < 2.8, 59% above 4.3; our drone's thrust is unmeasured)
are the next steps.

### Run 7i's result, and run 7j: the true state as an input (a diagnostic)

Run 7i, `physical` on `team`, 400 paired flights (seed 20000):

| | 1 lap | 3 laps | swept fails per crossing (3 laps) | G2 |
|---|---|---|---|---|
| 7f 3M | 76.8% | 46.2% | 4.4% | 6.2% |
| 7i 0.5M | 63.0% | 31.2% | 6.5% | 8.2% |
| 7i 1M | 68.2% | 27.5% | 7.9% | 10.1% |

The line was learned and did not help. At G2, 7i 0.5M crossed on the line where it
starts (<= 0.45 m) 72% of the time against 7f's 30%, and those crossings clipped the
opening on 7.7% against 7f's 1.0%: 7f's on-line crossings were clean because the
calm flights were the ones on it, not because being on it made them calm. G1, which
7i left alone, went from 2.2% to 7.2% by 1M -- as 7g and 7h, a fine-tune from 7f's
3M drifts whatever it is asked. Stopped at ~1M.

What limits the policy, on 7f's 3M flights (3 laps):
- The window. The opening's 0.6 m less the 0.19 m reach leaves the centre +-0.41 m,
  and the airframe at the crossing reaches 0.15 m past the centre (p50). The centre
  crosses with a spread of 0.17-0.24 m per axis; 1% a crossing needs ~0.14. The
  decoded position is itself 0.10 m off at the gates (p50; p90 0.20).
- Thrust. 3-lap success is 32% below TWR 3.5, 43% at 3.5-4.5, 62% at 4.5-5.5 and
  64% above; floor crashes 39 of 115 flights below 3.5, 2 of 39 above 5.5.
- Rescoring the same flights against a wider opening (an upper bound, not a
  prediction): 46% at 1.2 m, 67% at 1.5 m; 62% and 77% for TWR >= 4.5.
- Not the speed penalty: just before a floor crash the drone was over V_CAP no
  longer than the survivors at their lowest point (G3->G4 79 vs 76%, G4->G1 66 vs
  71%), and pushed the motors as hard (0.51 vs 0.48 after G3).

**Run 7j** (`hwo_`, a diagnostic, never deployable): 7f's settings with the true
state also in the encoder's input -- observation `true_state`, `env.ORACLE_KEYS`
(position, velocity in the world and the gate frame, attitude, rates: 19 numbers).
It still decodes the same `info_*` targets. `scripts/oracle_ckpt.py` appends zero
rows for those inputs to 7f 3M's `enc/mlp0/kernel` and its Adam moments, so the
fine-tune starts as 7f's policy exactly (checked: identical actions over 60 steps).
Scored as `hw_` with the input kept (`hwoe_`, `hardware.keep_oracle`).

    python scripts/oracle_ckpt.py <run 7f 3M snapshot>/ckpt oracle_7f3M/ckpt
    ./run.sh --finetune oracle_7f3M/ckpt --oracle      # 2M, snapshots every 0.25M

Reading it: first check that the input is used -- `decode err p_w` in the
evaluation should fall well below 7f's 0.237 m; if it has not by 1M, phase 3's lr
(2e-6) is too slow for new weights and the run says nothing yet. Then: 3 laps near
90% means the policy is limited by what it can estimate; 50-60% means by the
airframe and the room, i.e. thrust. In between, compare against a fine-tune from
7f's 3M without the input (every one so far lost 15-20 points by 1M).

### Run 7j's result, the raised-gate what-ifs, and run 7k: G1 and G4 raised 0.5 m

`./run_compare.sh` on 7f's 3M and 7j's 1M / 2M (400 identical flights, seed 20 000,
physical protocol, floor launch):

| | 1 lap | 3 laps | sim-only, 3 laps | swept/x | swept lateral / vertical | z sd at gates |
|---|---|---|---|---|---|---|
| 7f 3M | 76.8% | 46.2% | 72.8% | 4.3% | 86 / 89 | 0.20 m |
| 7j 1M | 86.5% | 65.5% | 90.0% | 3.1% | 123 / 17 | 0.14 m |
| 7j 2M | 85.5% | 61.8% | 89.0% | 3.1% | 135 / 4 | 0.12 m |

7j 1M beats 7f 3M by 19.2 points over 3 laps (95% CI +13.2 to +25.3, McNemar p <
0.001); the fine-tunes from 7f's 3M without the input (7g, 7h, 7i) all lost.

- **The floor crashes are the altitude estimate.** Per attempt of each leg (every
  flight, swept check ignored): G3->G4 3.3 -> 0.5-1.1%, G4->G1 3.6 -> 0-0.4%.
- **The vertical half of the swept failures too**: 89 -> 4 crossings.
- **The lateral half is not.** Lateral fails went 86 -> 135 and the lateral spread at
  the gates stayed 0.19-0.21 m. 85-95% of it is from crossing to crossing within a
  flight, not from the airframe draw (correlation with TWR <= 0.34). The decoded
  error explains R^2 0.08-0.13 of it. Straight crossings (0-10 deg off the gate's
  normal) still fail laterally at 2.1%. The input was not fully used (`decode err
  p_w` 0.209 -> 0.185 m), but it was used vertically and not laterally. That fits the
  shaping: the floor penalty and altitude margin pay for knowing the height, and
  nothing but the gate reward pays for the lateral position.
- So even with a perfect estimate, this policy stops at ~62-65% over 3 laps. The
  lateral spread has to come down from training, from ~0.21 to ~0.16 m, with the G1
  and G2 lateral biases (0.06-0.14 m) removed.

The fix for the floor without a new sensor is room: raise the gates the estimate
misses. 7f's 3M scored as it is (no retraining) on four layouts, the same flights:

| 3 laps, floor crash per attempt | team | G4 +0.5 (`team4up`) | G1 +0.25, G4 +0.5 (`team14up`) | G1 +0.5, G4 +0.5 (`team44up`) |
|---|---|---|---|---|
| success | 46.2% | 15.0% | 16.8% | 12.5% |
| takeoff | 4.2% | 2.2% | 2.0% | 2.2% |
| G3->G4 | 3.3% | 4.4% | 4.8% | 7.5% |
| G4->G1 | 3.6% | **13.1%** | 0.3% | **0.2%** |
| swept/x | 4.3% | 11.5% | 11.9% | 15.3% |
| G1 / G4 decoded - true z | 0.00 / -0.05 m | 0.01 / +0.47 | +0.28 / +0.45 | +0.52 / +0.42 |

- Raising G4 alone moves the crash to the U-turn down into G1. Raising G1 with it
  removes that crash, and at 0.5 m as well as at 0.25 m.
- Takeoff does not suffer from the higher G1: 2.0-2.2% against 4.2%, on the same
  airframes.
- Everything else is the world model, not the layout. It decodes G1 and G4 at the
  old height, off by the raise. The first-lap G1 swept failures grow with it (3 ->
  54 -> 141 of 400), all at the top edge (the drone passes 0.23 / 0.41 m high). G3->G4
  cannot be judged without retraining.

**Run 7k** (`--team10`): 7f's 3M, 7f's settings (`hwa_`), on `team44up`. The team
builds stands in 0.5 m steps, and 0.5 m did no worse than 0.25 at takeoff or G4 ->
G1. 3M rather than 2M because the world model has to relearn the layout; snapshots
every 0.25M. Only the track changes.

    ./run.sh --finetune <run 7f 3M snapshot>/ckpt --team10
    ./run_compare.sh --out compare_g44up --track team44up 7f3M=<same> 7k1M=... 7k2M=... 7k3M=...

(`--out compare_g44up` reuses 7f's scores there.) Reading it: (1) G1 and G4 decoded -
true z back near 0 (it was +0.52 / +0.42); (2) floor per attempt at 7j's level, <= 1%
on every leg, and G3->G4 under 7f's 3.3% on `team`; (3) takeoff no worse than 7f's
4.2%; (4) 3 laps against 7f on `team` (46.2%). The lateral spread is the next run's
target, not this one's.

**Run 7k's result** (`team44up`, 400 flights, seed 20000, 3 laps, paired):

| | 7f 3M on `team` | 7k 1.25M | 7k 2.78M | 7k 3.0M |
|---|---|---|---|---|
| success | 46.2% | 27.0% | **49.8%** | 38.8% |
| sim-only | 72.8% | 74.5% | 80.8% | 79.0% |
| swept (first failure) | 35.5% | 60.2% | 39.0% | 47.2% |
| floor, per attempt G3->G4 / G4->G1 | 2.2 / 2.0% | 0.8 / 0.1% | 1.2 / 0.5% | 1.4 / 1.3% |

- 2.78M against 7f: +3.5 points, p = 0.33. On the 208 flights not used to pick it
  (192-399): +0.5, p = 1.0. A tie. 3.0M is 11 points below 2.78M (p = 0.001): the
  G2 bias (+0.14 m high) came back within 0.22M steps. Confirm a pick on flights it
  was not picked on.
- The floor crashes went (G4->G1 2.0 -> 0.5% per attempt). The flights they spared
  failed at a gate instead: swept 35.5 -> 39.0%.

**Where the crossing spread comes from** (7k 2.78M, every crossing traced back in the
gate frame; 7f 3M and 7j alike):

- The final lateral offset is set 0.1-0.3 s (0.5-1.3 m) out, as the turn into the gate
  ends. The last 0.1 s is ballistic (a straight-line extrapolation from it explains
  95-100% of the offset). The median crossing still turns: its velocity is within 10
  degrees of the gate normal for ~0 s before the plane.
- Speed and crossing angle raise the failure rate (the steepest third of crossings
  fails 2-4x as often sideways) but explain 1-17% of the offset's variance.
- Attitude does not matter: the airframe adds 0.15-0.17 m to the centre's offset at
  every gate, and 95% of failures have the centre itself beyond 0.41 m.
- Estimation does not explain the lateral part: the decoded position spreads as much
  as the true one (sd 0.16-0.24 m), its error explains 0-30%, and 7j with the true
  state spreads the same laterally (0.18-0.21 m). Vertically 7j drops to 0.10-0.13 m.
- No run so far moved it: lateral sd 0.175-0.25 m in all nine checkpoints of 7f-7k.
- In training the policy's action std sits at its floor (minstd 0.1; entropy -3.46 for
  4 dims throughout), episodes last 6.7 s and 82% end at a gate.
- A normal model of the centre's offset reproduces the per-crossing failures (G1-G4
  3.6 / 7.8 / 4.1 / 5.1% predicted, 3.4 / 7.4 / 3.5 / 5.4% seen) and the 3-lap scores
  (7k 45 vs 49.8%, 7j 65 vs 65.5%). By it, 80% over 3 laps needs both axes at sd <=
  0.16 m and floor, takeoff and the rest under ~5%. One axis alone tops out near 70%.

**Ablations** (`run_ablation.sh`, no training): the same checkpoint and flights with
one source of randomness off at a time (`scripts/evaluate.py --dist-scale a,Ms,Mf,u
--gyro-noise lo,hi --rpm-noise f`; env `dist_scale` scales Table III's eval bounds
and leaves every draw as it was, so each pairs with the plain score).

    ./run_ablation.sh 7k2.78M=<run 7k>/snapshots/step_000002779840

A1 (no slow acceleration disturbance) bringing lateral sd under 0.16 m says the policy
lacks a measure of the force (no accelerometer input); A0 (nothing at all) still at
0.2 m says the policy itself.
