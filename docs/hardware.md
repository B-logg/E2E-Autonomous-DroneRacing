# Flight-test hardware

What we know about the real drone and gate, what follows from it, and what is still
missing. Nothing here is in the simulator yet unless it says so. Where a number is an
estimate, it is marked **(est.)** and the reasoning is given.

Last updated 2026-10-07. Camera and vision pipeline: GateNet and StochGAN are deferred until real flight video exists; the simulator is aligned to this sheet first.

## 1. Given

| Item | Value | Source |
|---|---|---|
| Gate inner opening | 1.5 m | user |
| Gate outer size | 2.1 m (frame width 0.30 m) | user |
| Gate thickness | 0.05 m | user |
| Frame class | 250 (motor-to-motor diagonal 250 mm) | user |
| Propeller | 5 inch, "pitch 45" | user |
| Motor | RTS RS2205, 2300 KV, 12N14P, 400 W max, ~30 g | [FalconShop listing](https://www.falconshop.co.kr/shop/goods/goods_view.php?goodsno=100000674) |
| Motor thrust, HQ5045 BN prop | **1024 g max per motor** (test voltage not stated) | same listing |
| Motor no-load RPM at 11.1 V | 25,530 (= 2300 x 11.1) | same listing |
| Battery | 4S 1500 mAh | user |
| Total mass | ~800 g | user |
| Camera tilt | 20 degrees up | user |
| Camera FOV | 114.6 deg horizontal, 92.15 deg vertical | user |
| Camera resolution | 720 x 480 | user |
| Camera offset | ~10 cm forward of the centre of mass | user |

"Pitch 45" is read as a 4.5-inch pitch, i.e. an HQ5045 (5.0 x 4.5 in), which is the prop
the listing's thrust figure is for. **Needs confirming.**

## 2. Derived

| Quantity | Value | How |
|---|---|---|
| Prop radius | 0.0635 m | 5 in / 2. The simulator's `PROP_RADIUS` = 0.0486 m is *not* a mismatch: it is the 75% blade station of MonoRace's 5.1 in prop (0.75 x 0.0648), and 0.75 x 0.0635 = 0.0476 m for ours. Left unchanged. The full 0.0635 m is used for the collision footprint only |
| Footprint radius, props included | ~0.19 m **(est.)** | 0.125 m arm tip + 0.0635 m prop radius |
| Footprint radius, frame only | ~0.14 m **(est.)** | arm tip + motor bell |
| Window for the drone's centre | **~0.56 m** with props, ~0.61 m frame only **(est.)** | 0.75 m opening half-width minus footprint radius. Isotropic and conservative: see below |
| Peak thrust-to-weight | **<= 5.1** | 4 x 1.024 kgf / 0.8 kg, unloaded |
| Realistic thrust-to-weight | **~4.0-4.6 (est.)** | 80-90% of the listing's figure for pack sag; no measurement |
| Rotor speed at full throttle | 2700-3400 rad/s **(est.)** | no-load 2300 KV x 14.8-16.8 V = 3565-4046 rad/s, 75-85% under a 5 in prop. The simulator's `w_max = 3100` falls inside |

### What these imply

1. **Takeoff is a live risk.** The simulator's nominal thrust-to-weight is 6.07 and its
   randomisation spans 2.9-10.8 (median ~6). In the 16.99M policy, takeoff-ground deaths
   were 32% at thrust-to-weight <= 4.0, 9.5% at 4.0-4.5, 4.8-8% above 4.5. This drone sits
   in the weak tail where the policy is least reliable.
2. **The window is tighter than anything the policy was trained on.** Trained at 0.8 (70%)
   and 1.0 (30%); the real window is ~0.56-0.61. At 0.57 the current policy succeeds on
   45-68% of 5-lap runs (3-plane tunnel / centre plane only).
3. **The footprint is not a disc.** Seen along the flight direction a quadrotor is a flat
   rectangle, ~0.25 m wide in the plane of its arms and a few centimetres tall, so the
   constraint depends on attitude at the moment of crossing. The simulator's
   `max(|y|, |z|) / d_g` is isotropic. A proper `physical` evaluation projects the body
   onto the gate plane using the attitude at crossing.
4. **A 5 cm gate does not need a 0.8 m tunnel physically.** The simulator's pre/post
   planes at +-0.4 m stand in for body size and attitude. For this gate they are a
   conservative proxy, not a measurement.

## 3. Mismatches with the simulator

| | Simulator | Real | Severity |
|---|---|---|---|
| **Camera tilt** | **45-55 deg up** (Table III) | **20 deg up** | **Blocking.** Measured on the 16.99M policy, paper protocol, same seeds, N=128: 93.0% success at 45-55 deg, 79.7% at 40, **2.3% at 30, 0.0% at 20** (75% die within one gate). Mount at ~45 deg, or retrain with the range moved |
| Image shape | 64 x 64 mask on the nominal K (~104 deg square) | 720 x 480, 114.6 x 92.15 deg | Reflected in the `hw_*` task as blank top/bottom rows. The remap itself (undistortion, crop, resize to the nominal K) is the deferred vision pipeline; lens distortion is unknown |
| Camera position | at the centre of mass | 10 cm forward | Low-medium. Cheap to add to `render.py` |
| Gate outer size | 2.7 m (frame 0.6 m) | 2.1 m (frame 0.3 m) | Medium. The world model learned a frame twice as thick |
| Prop radius | 0.0486 m (75% station) | 0.0476 m (75% station) | None |
| Gate thickness / tunnel | 0.8 m tunnel | 0.05 m | Conservative in simulation |
| Image delay | 3 steps (33 ms) | unknown | -- |
| Action delay | 1 step (11 ms) | unknown | -- |

## 4. Still needed

Marked by what it blocks.

**Blocks the `physical` evaluation protocol**
- Confirm the footprint: the real motor-to-motor diagonal (250 mm assumed) and whether any
  guard or camera mount extends past the props.

**Blocks judging takeoff risk**
- Static thrust at full throttle, at working voltage (a kitchen scale and a rigidly held
  drone is enough). The listing's 1024 g has no stated voltage.
- Battery C rating and sag under load.

**Blocks calibrating the dynamics**
- Motor spin-up time, 10% to 90% (simulator `tau` = 30 ms, +-30%).
- Throttle command to RPM curve, idle RPM (`w_min`, `k`).
- Flight controller and ESC: command protocol and rate, whether RPM telemetry is available
  (the policy observes RPM), its resolution and latency.
- Gyro range and noise (the simulator draws 0.1-0.5 rad/s per episode).
- End-to-end latency, camera to action.

**Blocks the vision pipeline**
- Camera exact mounting (height, lateral offset), lens distortion, exposure.
- Whether motion capture is available (ground truth for evaluation and for GateNet data).
- How the 720 x 480 image becomes a 64 x 64 mask.

**Later**
- Inertia (`J_x`, `J_y`, `J_z`), estimated from mass distribution.
- Real track: gate positions, orientations, heights and how accurately they are known;
  ceiling height (the inverted loop peaks near 4 m); launch podium height.
- Onboard computer and whether a 12M-parameter model runs at 90 Hz.

## 5. The `hw_*` task: what the simulator now does with this sheet

`./run.sh --hw` trains the task `skydreamer_hw_inverted_loop` (and `hw_big`); the name
travels in `config.yaml`, so `run_test.sh`, `evaluate.py` and `visualize.py` pick the
profile up without being told. Source: `HW_PROFILE` in `skydreamer/hardware.py`.

| Parameter | Paper | `hw_*` | Basis |
|---|---|---|---|
| Gate outer size | 2.7 m | **2.1 m** | given |
| Camera position | at CoM | **0.10 m forward** | given |
| Camera field of view | fills the nominal 104.4 deg square K | **114.6 x 92.15 deg**: all columns valid, top and bottom 6 rows (of 64) blank | given; the real image is remapped onto the nominal K, whose vertical extent is wider than the camera's |
| Nominal thrust-to-weight | 6.07 | **4.3** (`k_w` scaled; with Table III's randomisation the sampled median is 4.2, 10-90% 2.9-6.1, 1-99% 2.3-7.2) | **estimate**, unmeasured |
| Window `d_g` train / eval | 0.8 / 1.0 | **0.65 / 0.60** | 0.75 opening half-width less 0.15-0.19 m airframe reach; train a little lenient as the paper's 0.8 is against 0.75 |
| Tunnel `t_g` | 0.8 m | **0.40 m** | pre/post planes at the airframe's reach (+-0.20 m) |
| Camera tilt | 45-55 deg | 45-55 deg (unchanged) | the real camera must be mounted there |
| Prop radius in the dynamics | 0.0486 m | unchanged | already the 75% station of a 5 in prop |

Not changed, and therefore still mismatched: gate height and the 2.7 m gap of the virtual
gate (real track unknown), image shape and FOV (vision pipeline deferred), the moment
coefficients `k_p/k_q/k_r` (only thrust is rescaled, so attitude authority relative to
thrust is higher than a 4.3 drone would have), motor time constant, delays, inertia.

Scoring: `--protocol physical` (default in `run_test.sh` for such runs) turns the
simulator's window off and checks every airframe point against the real 1.5 m opening at
each real gate; the virtual gate asks for the centre within 1.0 m.

Measured on the 16.99M paper-trained policy, 200 episodes x 5 laps, same seeds:

| | success | sim-only |
|---|---|---|
| paper environment, swept footprint check | 73.5% | 90.0% |
| + 2.1 m gate frame | 0.0% | 1.0% |
| + camera 0.10 m forward | 0.0% | 1.0% |
| + thrust-to-weight 4.3 / 3.7 / 5.1 | 0.0% / 0.0% / 0.0% | 0.0% |

The policy cannot read a 2.1 m frame at all, so the effects after it are not separable.
