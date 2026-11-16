#!/usr/bin/env python3
"""Register the `ctba` suite and its config in a DreamerV3 checkout that already
carries skydreamer's patches/informed_dreamer.patch.  Idempotent.

    python scripts/patch_dreamerv3.py <path to skydreamer>/third_party/dreamerv3

Adds:
  * main.py make_env:   'ctba': 'ctba.embodied:Ctba'
  * configs.yaml:       defaults.env.ctba, and a `ctba` config block = skydreamer's
                        settings (informed decoder, packed mask, replay 10e6,
                        train_ratio 128, size12m) on task ctba_c1_team44up.
"""
import pathlib
import sys

MARK = "# --- CTBA (hierarchy_control) ---"

CONFIG_BLOCK = f"""
{MARK}
# skydreamer's settings with the CTBA env: 100 Hz policy, (thrust, roll, pitch, yaw rate) action
# flown through Betaflight's ANGLE mode.
ctba:
  task: ctba_c1_team44up
  batch_length: 64
  replay_context: 16
  replay.size: 10e6
  agent:
    informed: True
    packbits: mask
    report_gradnorms: True
    imag_loss: {{slowtar: True, actsmooth: 0.002}}
    imag_length: 16
  run: {{steps: 16e6, train_ratio: 128, envs: 16}}
  env:
    ctba: {{size: 64, max_steps: 2000, use_seed: True}}
"""


OLD_MARK = "# --- " + "CTBR (hierarchy_control) ---"   # the superseded rate-command suite


def _drop_old(m: str, c: str):
    """Remove the earlier `ctbr` registration (its package no longer exists)."""
    old_line = "      'ctbr': 'ctbr.embodied:Ctbr',\n"
    if old_line in m:
        m = m.replace(old_line, "")
        print("main.py: removed old suite ctbr")
    if OLD_MARK in c:
        a = c.index("\n" + OLD_MARK)
        b = c.index("\nmulticpu:", a)
        c = c[:a] + c[b:]
        c = c.replace("    ctbr: {size: 64, max_steps: 2000, use_seed: True}\n", "")
        print("configs.yaml: removed old ctbr config")
    return m, c


def main(root: pathlib.Path) -> int:
    main_py = root / "dreamerv3" / "main.py"
    cfg = root / "dreamerv3" / "configs.yaml"
    m, c0 = _drop_old(main_py.read_text(), cfg.read_text())
    cfg.write_text(c0)
    main_py.write_text(m)
    if "'skydreamer': 'skydreamer.embodied_env:SkyDreamer'" not in m:
        sys.exit("informed_dreamer.patch is not applied to this checkout (run skydreamer's setup first)")
    if "'ctba':" not in m:
        m = m.replace("      'skydreamer': 'skydreamer.embodied_env:SkyDreamer',\n",
                      "      'skydreamer': 'skydreamer.embodied_env:SkyDreamer',\n"
                      "      'ctba': 'ctba.embodied:Ctba',\n")
        main_py.write_text(m)
        print("main.py: registered suite ctba")
    c = cfg.read_text()
    if MARK not in c:
        line = "    skydreamer: {size: 64, max_steps: 2000, use_seed: True}\n"
        assert line in c, "configs.yaml: defaults.env.skydreamer not found"
        c = c.replace(line, line + "    ctba: {size: 64, max_steps: 2000, use_seed: True}\n", 1)
        anchor = "\nmulticpu:"
        assert anchor in c
        c = c.replace(anchor, CONFIG_BLOCK + anchor, 1)
        cfg.write_text(c)
        print("configs.yaml: added defaults.env.ctba and the ctba config")
    print("dreamerv3 ready for ctba")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(pathlib.Path(sys.argv[1]).expanduser().resolve()))
