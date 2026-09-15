# SkyDreamer reproduction

Reproduction of **SkyDreamer: Interpretable End-to-End Vision-Based Drone
Racing with Model-Based Reinforcement Learning** ([arXiv:2510.14783](https://arxiv.org/abs/2510.14783),
Verraest, Bahnam, Ferede, de Croon, De Wagter — TU Delft MAVLab) from the paper
alone. No code, checkpoints or datasets were released by the authors.

**Target:** reproduce the paper's *simulation* results — 100% gate success,
~13 m/s and ~6 g on a small inverted-loop track, with the world model decoding
state and dynamic parameters accurately enough to be interpretable.

Everything the paper does not specify is defaulted and documented in
[`docs/paper_gaps.md`](docs/paper_gaps.md). Read that before trusting a number.

---

## Status

| Component | Paper § | State |
|---|---|---|
| Quadcopter dynamics + RK4 | II-D, Table II | ✅ done, TWR verified = 6.07 |
| Domain randomization, initial states | Table III | ✅ done |
| Track, gate frames, pre/post gates | II-C | ✅ done (our coordinates) |
| Reward, termination | II-C | ✅ done |
| Flight plan logic | II-H | ✅ done |
| Segmentation mask rendering | II-E | ✅ done (JAX raycaster, not PyTorch3D) |
| Rolling shutter, erosion | II-F | ✅ done |
| Informed POMDP env | II-A/B | ✅ done |
| Informed decoder patch for DreamerV3 | II-G | ✅ done, applies to pinned commit |
| Smoothness loss | II-G | ✅ done |
| 3-phase training schedule | III-A | ✅ scripted; **smoke-trained end-to-end**, never at full scale |
| Simulation evaluation vs Table IV | III-C | ✅ `scripts/evaluate.py`, verified against a checkpoint |
| One-command GPU runner | — | ✅ `run.sh`, verified from a clean slate |
| Paper-conformance audit | all | ✅ `tests/test_paper_conformance.py`, 62 assertions |
| StochGAN mask translation | II-F | ⛔ blocked: needs real flight footage |
| GateNet (model, losses, data, train, ONNX export) | II-E, App. A | ✅ implemented + tested; **needs labelled images to train** |
| Onboard deployment | III-A | ⛔ out of scope for phase 1 |

DreamerV3 has been run end-to-end against this environment on CPU (600 steps,
`size1m`) and the training loop is healthy. No real training run has happened:
the policy is untrained and none of the paper's numbers are reproduced yet.

## Layout

```
skydreamer/params.py    Table II coefficients, Table III randomization
skydreamer/dynamics.py  equations of motion, RK4, quaternion attitude
skydreamer/track.py     gates, gate frames, reward, flight plan vector
skydreamer/render.py    binary gate masks by ray casting
skydreamer/augment.py   rolling shutter + erosion
skydreamer/env.py       the informed POMDP: o_t vs info_* split
skydreamer/embodied_env.py   adapter for DreamerV3's driver
skydreamer/gatenet/     U-Net segmentation for REAL images (PyTorch, not in the
                        sim loop; model / losses / data+synthetic / preprocess)
scripts/train_gatenet.py, scripts/export_gatenet.py
scripts/train.py        the paper's 3-phase schedule
scripts/evaluate.py     simulation validation, scored against Table IV
run.sh                  clone -> trained + evaluated policy, one command
patches/informed_dreamer.patch   informed decoding + smoothness loss
docs/paper_gaps.md      every value the paper does not give us
docs/datasets.md        what data each stage needs, and in what order
                        (policy training needs none)
```

## Why no PyTorch3D

The paper renders masks with PyTorch3D. Gates are flat rectangular annuli, so
one ray per pixel intersected with each gate plane gives the mask **exactly**,
in a handful of array ops, and keeps the whole environment inside JAX. Going
through PyTorch would put a host round-trip on every one of 17M env steps.
It is depth-correct: a near gate's hole shows the gate behind it.

## Running it on a GPU server

Everything is in one script. The server needs **only** an NVIDIA GPU with a
CUDA driver, plus `git` and `curl` — no Python, no conda, no root. `uv` installs
its own Python 3.11 (jax 0.4.33, which DreamerV3 pins, does not support 3.13).

```bash
git clone <your-repo-url> skydreamer && cd skydreamer

./run.sh --smoke     # ~5 min  — DO THIS FIRST
./run.sh             # ~50 h   — the paper's run, then evaluation
```

`--smoke` installs everything, runs the 25-test suite, trains 2000 steps with a
tiny model and evaluates it. The resulting policy is useless by design; what it
proves is that setup, training, checkpointing and evaluation all work. Five
minutes here saves finding out at hour 40.

The real run is long, so detach it:

```bash
tmux new -s sd './run.sh 2>&1 | tee run.log'
#   ctrl-b d to detach, tmux attach -t sd to come back
```

Other modes:

| | |
|---|---|
| `./run.sh --setup-only` | install only, train nothing |
| `./run.sh --eval-only` | re-score the most recent run |
| `./run.sh --big` | the 35M-step big-track preset (needs gate coordinates first) |
| `./run.sh --seed 1` | any extra flag is forwarded to `scripts/train.py` |

Override paths with `SKYDREAMER_LOGDIR=/data/logs ./run.sh`.

### Renting on vast.ai (or any cloud)

Two settings there matter more than the GPU, and one of them is easy to miss.

**Disk: set the slider to 150 GB.** The paper's `replay.size` is 10e6 steps
(31 h of flight). Each step stores a 64x64 mask plus vectors = **4596 bytes**,
so the buffer is **~46 GB**, and DreamerV3 keeps it in RAM *and* writes it to
disk. A default 16 GB disk dies part-way through a 30 h run. Extra disk costs
about $0.05/hr — nothing next to the GPU.

**RAM: prefer 128 GB.** 64 GB has to hold that same ~46 GB buffer plus JAX plus
16 environment workers. It may survive; it may OOM at hour 20. If you are stuck
with 64 GB, shrink the buffer explicitly and note the deviation:

```bash
./run.sh --replay.size 5e6     # 15 h of flight instead of 31 h
```

`run.sh` measures both before training and refuses to start rather than dying
later.

**Instances get interrupted.** Use `./run.sh --resume` to continue the newest
run from its checkpoint instead of starting over.

**Template: pick a plain NVIDIA CUDA image.** `jax[cuda12]` ships its own CUDA
libraries as pip wheels, so all the host must provide is the driver — any
recent CUDA 12.x / Ubuntu 22.04 template works, and a PyTorch template just
wastes image size. CUDA base images often lack `git` and `curl`; `run.sh` checks
and tells you the apt line if they are missing.

### GPU requirements

The paper trained on **one A100 80GB, partitioned**: "a 40GB partition with 56
Shared Multiprocessors". 56 SMs out of the A100's 108, with 40 GB, is exactly
NVIDIA's `4g.40gb` MIG profile — **4/7 of one A100**, roughly 178 TFLOPS BF16.
That is the bar to clear, and it is lower than "an A100".

| GPU | BF16 | vs. the paper's slice | notes |
|---|---|---|---|
| RTX 4090 24GB | ✅ Ada | ~0.9x | cheapest way to match the paper's pace |
| A10 / RTX 3090 24GB | ✅ Ampere | ~0.7x | workable, expect ~70 h |
| L40S 48GB | ✅ Ada | ~1.0x | comfortable headroom |
| A100 40/80GB (full) | ✅ Ampere | ~1.75x | ~30–40 h |
| H100 80GB | ✅ Hopper | ~5x on paper | far less in practice, see below |
| V100 / T4 | ❌ **no BF16** | — | needs `--jax.compute_dtype float32`, 2–3x slower |

**BF16 is the hard requirement.** DreamerV3 defaults to `compute_dtype:
bfloat16`, which needs Ampere or newer. Volta and Turing will run only in
float32.

**Minimum memory: 24 GB.** The paper's 40 GB partition is more than the model
needs — `size12m` is ~12M parameters. The peak is phase 2/3, where
`batch_length` goes to 256 and the RSSM backprops through 256 timesteps.
Watch `nvidia-smi` during the smoke test's first training step if you want a
number for your own card before committing to the long run.

**Do not expect a big GPU to help much.** `size12m` at `batch_size 16` is a
small workload — it is bound by kernel launch overhead far more than by FLOPs,
so an H100's 5x paper advantage collapses to maybe 1.5x in practice. Spend on
*more runs in parallel* (the paper concedes results vary per seed) rather than
on one faster card.

### Are we running the same budget as the paper?

Yes — identically, and `tests/test_paper_conformance.py` asserts it: 17M
environment steps, phases at 8M and 13M, `train_ratio` 128, `size12m`,
`replay_context` 16, buffer 10e6. Same steps, same hyperparameters, same
schedule.

Matching the *budget* is not the same as matching the *result*. Three things
stand between us and the paper's numbers, and all three are in
[`docs/paper_gaps.md`](docs/paper_gaps.md): our track geometry is a
reconstruction (B1, B2), the `k_hor` term is dimensionally broken as printed
(A1), and the paper itself says parameter-identification quality "depends on
the training run" (B7). Budget for three seeds before drawing a conclusion.

### What it does, in order

1. installs `uv`, creates a Python 3.11 venv
2. clones DreamerV3 at `cdf5709` — the exact commit section III-A cites — and
   applies `patches/informed_dreamer.patch`
3. installs `jax[cuda12]==0.4.33` and the packages DreamerV3 actually imports
4. trains the paper's three phases (17M steps: defaults → `batch_length` 256 at
   8M → entropy 1e-5 and lr 2e-6 at 13M)
5. evaluates 100 episodes × 5 laps in simulation and writes a report

### Output

```
logdir/small-<timestamp>/
  evaluation.txt     scored against the paper's Table IV
  evaluation.json    same, machine-readable
  metrics.jsonl      per-step training metrics
  ckpt/              weights
  train.log
```

`evaluation.txt` looks like this (numbers here are from an untrained smoke run):

```
=== inverted_loop ===
episodes            100
success rate        0.0%   (paper 100.0%)
lap 1 [s]           n/a    (paper 3.37)
laps 2+ [s]         n/a    (paper 3.25)
max speed [m/s]     n/a    (paper 13.00)
max accel [g]       n/a    (paper 6.00)
mean gate error [m] n/a
min gate margin [m] n/a
decode err p_w      5.551
decode err v_w      3.059
```

**Watch `decode err p_w` first.** It is the distance between the world model's
decoded position and ground truth. That is the paper's central claim — the
world model as an implicit state estimator (Figures 4 and 5) — and it moves
long before lap times do. If it is not falling well under a metre, nothing
downstream will work.

## Local development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,gatenet]'
.venv/bin/python -m pytest tests -q          # 103 passed
```

## What the informed patch does

Informed Dreamer's whole mechanism is a naming convention: observation keys
prefixed `info_` are privileged. The patch splits DreamerV3's encoder and
decoder spaces so the encoder never sees them and the decoder reconstructs only
them. That is what makes the world model a state and parameter estimator
instead of an image autoencoder — the paper's central claim.

Our `info_*` keys, 88 dims total: `p_w`, `p_g`, `v_w`, `v_g`, `att`, `rates`,
`rpm`, `cam` (extrinsics), `dyn` (31 dynamic parameters), plus the measured
quantities and the flight plan. Per the paper, the **image is deliberately not
a decoder target**.

## Verification passes

### Pass 2 (2026-09-14): first real DreamerV3 run

Installing DreamerV3's pinned deps on CPU and actually launching training found
**three more defects that static review could not**:

| # | Defect | Symptom |
|---|---|---|
| 6 | `actsmooth` read `v.mean`, but heads are wrapped in `outs.Agg` which has no `.mean` | `sum()` over an empty generator returned int `0` -> `AttributeError`. Had the guard been written differently it would have *silently disabled the smoothness loss*. Now asserts. |
| 7 | `make_track` built JAX arrays at the host boundary | DreamerV3 sets `jax_transfer_guard='disallow'` process-wide (`embodied/jax/internal.py:38`). Crashes on GPU too. |
| 8 | Same for `jax.random.key(seed)` and the obs/action marshalling in the wrapper | Ditto. Transfers are now allowed only inside the env, which is the real host<->device boundary. |

Confirmed healthy in that run:
- all 12 `info_*` decoder heads train, and **there is no `train/loss/mask`** --
  the encoder sees the image, the decoder does not reconstruct it, which is
  exactly the informed-POMDP split the paper specifies;
- `actsmooth` is live and non-zero;
- reward, value, continue and policy losses all present; episodes terminate.

### Pass 1 (2026-09-04): paper-vs-code re-read

A full re-read of every file against the paper turned up **five real defects**,
all fixed with regression tests:

| # | Defect | Why it mattered |
|---|---|---|
| 1 | `plane_pose` did not wrap the lap index | JAX silently clamps the out-of-range gather, so every lap after the first targets the last gate forever. No error, training just never works. |
| 2 | Image delay was 22 ms, not 33 ms | The mask buffer held `D` frames but is pushed *before* the observation is read, so it needs `D+1`. Off by one step. |
| 3 | `step()` reused RNG keys | `k[0]` drove both the acceleration-disturbance resample and the flight-plan increment, making two unrelated events perfectly correlated. Same for `k[1]`. |
| 4 | Env registration was wrong | DreamerV3's `make_env` uses a **hardcoded ctor dict** in `main.py`; adding to `embodied/envs/__init__.py` does nothing. The patch now edits that dict. |
| 5 | All 16 workers shared seed 0 | Without `use_seed: True` in the env config, every parallel env replays the identical trajectory and the replay buffer collapses to one rollout. |

Also fixed: `env.skydreamer` passed a `track=` kwarg the constructor does not
take (the task name already carries it), and `DynParams.as_vector` normalized
the ±20% motor limits against the ±30% width.

## Verified so far

- Static thrust-to-weight from Table II = **6.07** — matches the paper's 6 g.
- Hover is an exact translational equilibrium.
- Angle-of-attack sign is right: climbing loses thrust, descending gains it.
- Table II's per-motor coefficients are asymmetric, so a uniform motor command
  produces a real trim moment (~1.9 rad/s² roll, ~15 rad/s² pitch). Expected,
  pinned by a test, and something the policy must learn to cancel.
- Rate penalty is <0.01 below 10 rad/s and 1.34 at the clip — it only punishes
  gyro-saturating rates, as intended.
- ~11k env-steps/s on 8 CPU cores; env time will not be the bottleneck on GPU.
- Image delay measured at exactly 3 steps, action delay at exactly 1 step, by
  correlating observed frames against the render history.
- The patch applies cleanly to `cdf5709` and every paper-stated hyperparameter
  round-trips through the config (`informed`, `actsmooth 0.002`, `slowtar`,
  `imag_length 16`, `replay_context 16`, `train_ratio 128`, `size12m`).
- 103 tests, lint clean, including a 62-assertion paper-conformance audit.

## Next

1. First training run, small track, 3 phases. Verify laps complete at all.
2. Check the decoded `info_p_w` / `info_v_w` against ground truth — this is the
   paper's Figure 4/5 claim and the fastest signal that the setup is right.
3. Sweep seeds; the paper concedes parameter-ID quality varies per run.
4. Reconstruct the big track once the small one flies.
