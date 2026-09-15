# What data this project actually needs

The short version:

| Stage | Dataset | Labels? | Needs the drone to fly? |
|---|---|---|---|
| DreamerV3 policy training (simulation) | **none** | — | no |
| GateNet | image + mask pairs | **yes, by hand** | photos only |
| StochGAN | two unpaired mask folders | no | **yes** |
| Dynamics identification | flight logs | no | **yes** |

## 1. Policy training needs no dataset at all

DreamerV3 is online RL. The policy flies the simulator, its rollouts go into a
replay buffer (10e6 steps = 31 h of flight), and the world model trains on
those. Masks are **rendered every step**, not loaded from disk. There is no
offline dataset to prepare, and nothing to label.

This is why phase 1 needs only a GPU.

## 2. GateNet — the only hand-labelled data

```
data/gatenet/<gate_type>/
  real/
    images/0001.png      RGB, already remapped to the nominal K
    masks/0001.png       {0,255} grayscale, same basename
  backgrounds/*.jpg      scenes with NO gates in them
  cutouts/*.png          RGBA, gate in the colour channels, gate in the alpha
```

**Resolution** (Appendix A, "Gate-specific implementation"):

| Gate | Resolution | `f` | Real labels | Synthetic |
|---|---|---|---|---|
| orange, plain square | 384×384 | 4 | **200** | none needed |
| MAVLab (dark, thin, logos) | 196×196 | 2 | **700** | **8500** |

MonoRace reports a 3500 synthetic : 500 real split for its own training.

**Images must be preprocessed first.** `skydreamer.gatenet.preprocess`
remaps raw frames onto the nominal pinhole model `K = [[25/64 W, 0, W/2], ...]`
using your Kalibr calibration. Do this *before* labelling, so the labels live
in the same geometry the policy will see. Call
`NominalRemapper.check_coverage()` once — if it returns below ~0.99 your crop
is producing black borders the policy has never seen.

**Do not correct extrinsics.** Camera roll/pitch/yaw is what the world model
estimates online; that is the paper's "no extrinsic calibration" claim, and
correcting it would remove the variation the policy is trained to handle.

**Labelling tips**
- Label the gate *structure* (the frame), not the aperture.
- Include frames where the gate runs off the edge. Close passes are mostly
  partial gates, and a net trained only on whole gates fails exactly when
  precision matters most.
- Include the confusers. The paper had to physically cover part of the room
  because GateNet kept classifying it as a gate.
- Perfection is not the goal. The paper deliberately kept a mediocre GateNet
  to show SkyDreamer copes with bad masks.

**Synthetic generation** is handled by `SyntheticGateCompositor`: gate cutouts
under randomized scale / rotation / perspective warp, composited onto a
gate-free background, plus HSV colour augmentation and Gaussian noise. The mask
comes from the cutout's alpha channel, so it is exact and free. You need maybe
a dozen good RGBA cutouts and a few hundred backgrounds of the actual venue.

## 3. StochGAN — no labels, but the drone must fly first

```
data/stochgan/
  rendered/*.png     64×64 binary, straight from our simulator (free, infinite)
  real/*.png         64×64 binary, = GateNet(real flight footage)
```

~4000 images total, **unpaired** — image *i* in one folder has nothing to do
with image *i* in the other. The paper: they "require no manual labeling and
are therefore inexpensive to collect".

The ordering constraint is what matters: **GateNet must be trained before this
folder can exist**, because `real/` is literally GateNet's output on recorded
flights. So the sequence is

    label photos → train GateNet → fly and record → run GateNet over the
    recording → train StochGAN → plug into the sim's augmentation chain

## 4. Dynamics identification — the one that is easy to forget

**Table II's coefficients belong to TU Delft's drone, not ours.** If our
airframe differs, the simulator is wrong and no amount of visual sim-to-real
work will save it. Record, from flights on the actual airframe:

| Signal | Rate | What it identifies |
|---|---|---|
| motor RPM (ESC telemetry) | as high as available | `k_w`, `tau`, `w_max`, thrust/moment coefficients |
| gyroscope | 2 kHz | moment coefficients, `J_x/J_y/J_z` |
| accelerometer | 1 kHz | thrust and drag (identification only — the policy never uses it) |
| camera timestamps | 90 Hz | the clock everything else is aligned to |

MonoRace's method: linear regression on high-speed flight data for the nominal
parameters, and a free-fall spinning throw for the gyroscopic coupling terms.
`tudelft/indiflight` provides this logging.

In particular, the `k_hor` advance-ratio term flagged 🔴 in
[`paper_gaps.md`](paper_gaps.md) A1 — where the paper's printed equation is
dimensionally broken and we had to pick a default — is exactly what these
high-speed logs resolve.

## 5. What you do *not* need

- **MoCap.** The paper's real flights ran with no external aid; MoCap was only
  a reference for the plots, and one experiment has none at all because the
  system broke. Useful for validation, not required.
- **Paired real/rendered masks.** StochGAN is unpaired by design.
- **Any offline dataset for the RL itself.** See §1.
