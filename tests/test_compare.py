"""Checkpoint comparison and floor-failure diagnosis (run_compare.sh)."""

import importlib.util
import json
import math
import pathlib

import jax
import numpy as np

from skydreamer import track as T
from skydreamer.env import EnvConfig

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    import sys

    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flight(success=False, cause=0, gates=4, end=None, phys_fail=None, events=(), **kw):
    e = dict(success=success, sim_success=success or phys_fail is not None, cause=cause,
             gates=gates, end=end, phys_fail=phys_fail, phys_events=list(events),
             gate_times=[1.0, 2.0, 3.0, 4.0][:gates], floor_pen=0.0, twr=3.5)
    e.update(kw)
    return e


def test_intervals_and_mcnemar_match_their_textbook_values():
    ce = _load("compare_evals")
    lo, hi = ce.wilson(40, 100)
    assert abs(lo - 0.3094) < 1e-3 and abs(hi - 0.4980) < 1e-3   # Wilson 95%, 40/100
    assert ce.mcnemar_p(0, 0) == 1.0
    # 2 vs 10 discordant: 2 * P(X <= 2), X ~ Bin(12, 1/2) = 2 * 79 / 4096
    assert abs(ce.mcnemar_p(2, 10) - 2 * 79 / 4096) < 1e-12
    assert ce.mcnemar_p(10, 2) == ce.mcnemar_p(2, 10)


def test_first_failure_is_whichever_came_first():
    ce = _load("compare_evals")
    end = dict(step=500, alt=0.05)
    # swept at step 300, then the floor at 500: the swept check ended it
    assert ce.first_failure(_flight(cause=2, gates=3, end=end,
                                    phys_fail=dict(step=300, gate=1, off=0.7))) == "swept"
    # floor first, swept recorded later in the trace: the floor
    assert ce.first_failure(_flight(cause=2, gates=3, end=end,
                                    phys_fail=dict(step=600, gate=1, off=0.7))) == "ground G3->G4"
    # 11 gates on a 4-gate track: on the way from G3 to G4 of lap 3
    assert ce.first_failure(_flight(cause=2, gates=11, end=end)) == "ground G3->G4"
    assert ce.first_failure(_flight(cause=2, gates=4, end=end)) == "ground G4->G1"
    assert ce.first_failure(_flight(cause=2, gates=0, end=dict(step=13, alt=0.07))) == "launch"
    assert ce.first_failure(_flight(cause=2, gates=0, end=dict(step=200, alt=0.2))) == "ground ->G1"
    assert ce.first_failure(_flight(cause=1, gates=2, end=end)) == "gate"
    assert ce.first_failure(_flight(success=True)) == "ok"


def test_flights_pair_by_seed_and_pool_across_files(tmp_path):
    ce = _load("compare_evals")

    def write(d, tag, seed, outcomes):
        d.mkdir(exist_ok=True)
        stem = f"evaluation_physical_3lap{tag}"
        (d / f"{stem}.json").write_text(json.dumps({"seed": seed, "success_rate": 0}))
        (d / f"{stem}_episodes.json").write_text(json.dumps([_flight(success=o) for o in outcomes]))

    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "", 10_000, [1, 1, 0, 0])
    write(a, "_s20000", 20_000, [1, 0])
    write(b, "", 10_000, [1, 0, 1, 0])
    # a 1-lap file must not be read as 3 laps
    (a / "evaluation_physical_1lap_episodes.json").write_text(json.dumps([_flight(True)] * 9))

    A, B = ce.load([a], 3), ce.load([b], 3)
    assert len(A) == 6 and len(B) == 4
    assert len(ce.load([a, a], 3)) == 6              # the same directory twice counts once
    pr = ce.paired(A, B)                              # only the seed-10000 flights are shared
    assert (pr["n"], pr["both"], pr["a_only"], pr["b_only"], pr["neither"]) == (4, 1, 1, 1, 1)
    assert pr["diff"] == 0.0


def test_point_of_no_return_is_where_full_thrust_stops_too_low():
    df = _load("diagnose_failures")
    # Sinking at a steady 3 m/s from 2 m, level, TWR 3.5: the stop takes
    # 3 * LAG + 9 / (2 * decel); doomed once the lowest tip is within that
    # (plus the crash clearance) of the floor.
    n, sink, twr = 80, 3.0, 3.5
    alt = 2.0 - sink * df.DT * np.arange(n)
    p = np.stack([np.zeros(n), np.zeros(n), -alt], 1)
    v = np.tile([0.0, 0.0, sink], (n, 1))
    tilt = np.zeros(n)
    k = df.point_of_no_return(p, v, tilt, twr)
    need = df.stop_distance(np.array(sink), twr) + df.CLEARANCE
    assert alt[k] <= need < alt[k - 1]
    # Climbing for two steps just after that point, while still above the
    # clearance, means it was not doomed there: the final stretch starts later.
    v2 = v.copy()
    v2[k + 2:k + 4, 2] = -1.0
    assert alt[k + 3] > df.CLEARANCE
    assert df.point_of_no_return(p, v2, tilt, twr) == k + 4


def test_floor_strike_says_whether_it_was_late_unknown_or_chosen():
    df = _load("diagnose_failures")
    n = 60
    alt = np.linspace(1.5, 0.1, n)            # sinking at ~2.1 m/s into the floor
    p = np.stack([np.zeros(n), np.zeros(n), -alt], 1)
    v = np.tile([0.0, 0.0, 1.4 / (n - 1) / df.DT], (n, 1))
    q = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    e = dict(twr=3.5)

    def kind(throttle, bias):
        u = np.full((n, 4), throttle)
        dp = p.copy()
        dp[1:, 2] = p[:-1, 2] - bias          # decoded `bias` m higher, one step behind
        return df.floor_strike(e, dict(p=p, v=v, q=q, u=u, dp=dp))

    late = kind(0.95, 0.0)
    assert late["kind"] == "late"
    est = kind(0.3, 0.3)
    assert est["kind"] == "estimate" and abs(est["dec_alt_bias"] - 0.3) < 1e-9
    assert kind(0.3, 0.0)["kind"] == "policy"


def test_takeoff_reports_how_close_the_tip_came():
    df = _load("diagnose_failures")
    n = 40
    alt = np.linspace(0.05, 0.6, n)
    p = np.stack([np.zeros(n), np.zeros(n), -alt], 1)
    ang = np.radians(np.full(n, 10.0))        # 10 deg of roll throughout
    q = np.stack([np.cos(ang / 2), np.sin(ang / 2), np.zeros(n), np.zeros(n)], 1)
    out = df.takeoff({}, dict(p=p, q=q, u=np.full((n, 4), 0.5)))
    assert abs(out["max_tilt"] - 10.0) < 1e-6
    assert abs(out["min_tip"] - (0.05 - df.REACH * math.sin(math.radians(10)))) < 1e-6
    clear = np.nonzero(alt > df.CLEARANCE + df.REACH)[0][0]
    assert out["steps_to_clear"] == clear


def test_diagnosis_uses_the_simulators_constants():
    from skydreamer import hardware
    from skydreamer.dynamics import CONTROL_DT

    df = _load("diagnose_failures")
    assert (df.REACH, df.CLEARANCE, df.LAG, df.DT) == (
        hardware.REACH, T.FLOOR["team"], T.FLOOR_LAG, CONTROL_DT)


class _ZeroAgent:
    def init_policy(self, b):
        return None

    def policy(self, carry, obs, mode="eval"):
        b = len(obs["is_first"])
        return carry, {"action": np.zeros((b, 4), np.float32)}, {}


def test_rollout_records_what_each_flight_was_dealt():
    """The fields compare_evals.py pairs on and diagnose_failures.py reads."""
    ev = _load("evaluate")
    cfg = EnvConfig(track=T.inverted_loop(), max_steps=6)
    recs, _ = ev.rollout(_ZeroAgent(), cfg, episodes=3, laps=1, seed=7, chunk=2, dump=True)
    assert [r["index"] for r in recs] == [0, 1, 2] and {r["seed"] for r in recs} == {7}
    assert set(recs[0]["params"]) >= {"k_w", "w_max", "tau"}
    assert len(recs[0]["start"]["p"]) == 3
    t = recs[0]["_traj"]
    n = len(t["p"])
    assert t["u"].shape == (n, 4) and t["dp"].shape == (n, 3)
    assert np.isnan(t["dp"]).all()       # no world model, nothing decoded
    # The same seed deals the same flights.
    again, _ = ev.rollout(_ZeroAgent(), cfg, episodes=3, laps=1, seed=7, chunk=2)
    assert [r["params"]["k_w"] for r in again] == [r["params"]["k_w"] for r in recs]
    assert jax.numpy.allclose(np.array(again[1]["start"]["p"]), np.array(recs[1]["start"]["p"]))
