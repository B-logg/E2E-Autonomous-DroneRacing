# Why the 17M-step run does not fly

A record of what is known about the reproduction's failure, so that the next
person to pick it up — GPU in hand — starts from the evidence rather than from
scratch. Written after the first full 17M-step run (`small-20260920-070239`).

## The result

| | paper | ours |
|---|---|---|
| success rate, 5 laps | 100% | **0%** |
| top speed, small track | 13 m/s | 15.5 m/s mean-of-max, 25.7 max |
| peak acceleration | 6 g | 5.1 g mean-of-max, 9.5 max |
| gates passed (3 per lap) | all | mean 1.44 |
| episode ends by | — | **diverged 69%**, gate 29%, ground 2% |

## Training did not succeed either

This is the first thing to understand, because it reframes everything: the
evaluation is not disagreeing with training, it is reporting it.

`episode/length` during training, in steps (one lap = 303):

| step | mean length | |
|---|---|---|
| 3M | 55.5 | 0.61 s |
| 7M | 93.7 | 1.03 s |
| **11M** | **112.6** | **1.24 s** |
| 17M | 112.9 | 1.24 s |

**It plateaued at 11M and did not improve over the last 6M steps.** The longest
episode in the entire run was 328 steps — a single lap, once. A 5-lap
evaluation needs 1515.

What looked paper-like was the *world model*, not the policy.

## What has been ruled out

Each of these was checked against the paper and found to match, so none of them
is the cause.

- **Table II** — all 31 coefficients, verbatim.
- **Table III** — all 17 randomization and initial-state rows, including the
  gate-frame conventions (`z_g` points down, so the drone starts near the floor
  and must climb, which is what the shaped ground condition is for).
- **Dynamics** — `F`, `M` all three rows, `omega_c`, `alpha`, the quaternion
  kinematics, `Omega_dot = M + eps_M`, RK4 with zero-order hold on the action.
- **Reward** — `5 r_prog - r_rate + 30 r_gate`, each term's definition, progress
  zeroed inside the pre/post-gate tunnel, rate penalty on the *true* rates.
- **Termination** — the gate and ground conditions verbatim.
- **Delays** — 33 ms image (3 steps), 11 ms action (1 step); rates are not
  delayed, matching the paper.
- **Training** — 17M steps, size12m, `replay_context` 16, `train_ratio` 128,
  `slowtar`, replay 10e6, three phases at 8M/13M with the stated `batch_length`,
  `actent` and `lr` changes, `imag_length` 16, `lambda_smooth` 0.002 (verified
  active in the logs, not merely configured).
- **The world model** — decoded position error **0.249 m**, the same order as
  the paper's stated 10-15 cm. Parameter identification reproduces the paper's
  §V ordering item by item: `w_max` R²=0.91, `tau` 0.63, `k_w` 0.56 (paper:
  "excels", "accurately"), `k_p1` 0.72, `k_q1` 0.41 (paper: "moderate"), `k_x`
  0.01, `k_r1` 0.01 (paper: "difficult", "not accurately estimated").

The world model works. The policy does not.

## The mechanism, as observed

Traced from the finished checkpoint. Normal flight is fine: 0 to 12 m/s over
64 steps, rates around 6 rad/s, `||Omega||_1` about 10, climbing first and then
accelerating — the behaviour the paper describes. Then:

```
 t   max|w|  ||w||1        (rate penalty clips at ||w||_1 = 17)
186    9.93   16.13
187   13.58   20.53   <- past the clip
188   17.77   25.13
189   21.45   29.02
190   25.04   32.45
191   28.37   36.46   <- monotone, ~400 rad/s^2
```

Two facts make this unrecoverable once it starts:

1. **The rate penalty is clipped at `||Omega||_1 = 17`**, so beyond it the
   penalty is flat and there is no gradient pushing rates back down.
2. **The moment equation has no rotational damping.** Its only rate-dependent
   terms are the gyroscopic products `J_x q r` and friends, which are quadratic
   and destabilising, so `dOmega/dt ~ J Omega^2` is a finite-time blow-up: 22 ms
   from 50 rad/s. A tumbling drone at 4 m needs ~900 ms to reach the ground
   condition, so the paper's two termination conditions cannot catch it. Hence
   our added divergence guard, which reports the failure rather than causing it.

### Why the drone gets there

The loop it is closing is hard. Action delay 11 ms plus motor lag `tau` = 30 ms
is **41 ms** of lag, against a roll authority of 843 rad/s² — 34 rad/s of rate
change is already in flight before a correction lands. A high-gain controller
on that plant oscillates; this is not subtle control theory.

And the control signal shows exactly that: motor commands reverse direction
every 3.1 steps (~29 Hz) while the motor bandwidth is `1/(2 pi tau)` = 5.3 Hz,
and the rates oscillate at ~4.5 Hz, right at that bandwidth. The policy is
commanding faster than the actuator can move.

**Reinforcement learning has no notion of stability margin.** It raises gain
while return improves and stops when something breaks. Something in the paper's
setup must supply that margin — and the smoothness loss, which the paper
introduced precisely because policies "converged to near bang-bang behavior", is
evidence they hit this class of problem too. We have that loss, at the paper's
weight, verified active. So it is not sufficient on its own here.

## Hypotheses, ranked

**1. Zero sensor noise** (acted on; see `paper_gaps.md` B3). The only
unspecified value left that acts on the rate loop. The paper carries measured
and ground-truth rates as separate quantities and decodes both, which is
meaningless if they are equal — and at zero noise they were equal, which the
training log confirms. Perfect rate feedback gives the policy every reason to
chase the 90 Hz disturbance Table III already injects, and nothing to lose by
trying. **Now on by default.**

**2. Track geometry.** The most `[OURS]` input remaining. Turn radius sets the
required rate directly: `omega = v / R`, and at R = 1.35 m the paper's 13 m/s
needs `||Omega||_1` ~ 15.4, just inside the clip, while our 15.5 m/s needs 18.4,
just outside. A small error in gate spacing moves the task across that line.

**3. Run-to-run variance.** The paper notes identification quality "depends on
the training run". It does not note ever failing to complete a lap.

## Hypotheses that were tested and failed

Recorded because a dead hypothesis is still information.

- **Excess thrust from the advance-ratio term.** The idea was that our `k_hor`
  reading over-thrusts at speed, raising top speed past what the rate budget
  allows. Killed by correcting `r` to the value in MonoRace's footnote: the
  correct value makes translational lift *larger*, so our simulator had *less*
  of it than the paper's while flying *faster*. Backwards.
- **Smoothness-loss scale.** The idea was that the paper's `lambda = 0.002` was
  measured on `u in [0,1]` while ours applies to a mean in `[-1,1]`, making ours
  4x too weak. Killed by reading `wrap_env` in the DreamerV3 commit the paper
  cites: `NormalizeAction` is applied unconditionally to every continuous action
  space, so the paper's mean is in `[-1,1]` too. Ours already matches.
- **Poor dynamics-parameter identification.** Killed by measuring it: our
  per-parameter accuracy reproduces the paper's reported ordering, and the paper
  states plainly that poor identification comes "without resulting in a
  significant loss of flight performance".

## Measurement bugs found along the way

None of these caused the failure, but each one distorted the picture while it
was live, and all are fixed.

- The evaluation harness thresholded the **bit-packed** mask, turning every byte
  holding one gate pixel into eight lit pixels. That alone scored the finished
  checkpoint at 0% with a 3.1 m decode error.
- `decode_in_policy` exposed **symlog-space** predictions as if they were metres:
  reported position error 2.219 m against an actual 0.249 m.
- The harness kept **integrating crashed episodes**, which ran body rates to NaN
  and killed the evaluation outright.
- Every paper-comparison metric was computed over successful episodes only, so a
  0% run printed `n/a` on six of seven rows and said nothing about how it failed.
