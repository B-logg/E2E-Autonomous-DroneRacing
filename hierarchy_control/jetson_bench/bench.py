#!/usr/bin/env python3
"""Jetson inference benchmark for the CTBA racing stack.  Copy this folder to the
Jetson and run `./run.sh` (or `python3 bench.py`).  Everything is logged under
results/<host>-<time>/.

What it measures (random weights; latency depends on shapes, not values):
  1. policy_step   one DreamerV3 policy step (size12m), TensorRT FP32 and FP16.
                   The policy runs at 100 Hz: it must finish well inside 10 ms.
  2. GateNet       mask network at several input sizes and widths, TensorRT FP16.
  3. together      policy at 100 Hz while GateNet runs at camera rate (30 and 60 fps),
                   both on the GPU: the policy's latency under that load.
  4. PyTorch       (if installed) the same in eager PyTorch, and a Python 100 Hz loop
                   with GateNet in a thread: deadline misses, i.e. what a plain Python
                   deployment would see.
  5. OpenCV        (if installed) resize + undistort-remap of camera frames on the CPU.
  6. UART / MSP    (with --uart DEV) the flight-controller link: every 10 ms read
                   MSP_RAW_IMU + MSP_ATTITUDE and write MSP_SET_RAW_RC (neutral sticks):
                   round trips, cycle time, bytes, link load, deadline misses, errors,
                   the fastest rate the link sustains, and the same under GPU load.
Plus system info (L4T, power mode, clocks, versions) and tegrastats during the run.

Options: --quick (fewer GateNet sizes, shorter runs), --duration S, --skip-torch, --skip-trt,
         --uart /dev/ttyTHS1 --baud 115200 (FC link), --only-uart (just the FC link)
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ONNX = HERE / "onnx"
sys.path.insert(0, str(HERE))

LAT_RE = re.compile(r"\] Latency: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms, "
                    r"percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, percentile\(99%\) = ([\d.]+) ms")
GPU_RE = re.compile(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms, "
                    r"percentile\(90%\) = ([\d.]+) ms, percentile\(95%\) = ([\d.]+) ms, percentile\(99%\) = ([\d.]+) ms")
QPS_RE = re.compile(r"Throughput: ([\d.]+) qps")
KEYS = ("min", "max", "mean", "median", "p90", "p95", "p99")

QUICK_GATENET = [(4, 240, 320), (4, 480, 640)]     # chosen GateNet (2026-11-17) and its fallback


# ---------------------------------------------------------------------------------- utils
class Run:
    def __init__(self, root: Path):
        self.root = root
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "engines").mkdir(exist_ok=True)
        self.results = {"started": datetime.datetime.now().isoformat(timespec="seconds")}
        self.t0 = time.time()
        self._log = open(root / "bench.log", "a")

    def log(self, msg=""):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n"); self._log.flush()

    def save(self):
        self.results["elapsed_s"] = round(time.time() - self.t0, 1)
        (self.root / "results.json").write_text(json.dumps(self.results, indent=1))


def sh(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def read(path):
    try:
        return Path(path).read_text(errors="ignore").strip("\x00\n ")
    except OSError:
        return None


def system_info():
    info = {
        "host": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
        "model": read("/proc/device-tree/model"), "l4t": read("/etc/nv_tegra_release"),
        "nvpmodel": sh("nvpmodel -q")[1], "jetson_clocks": sh("sudo -n jetson_clocks --show")[1][:2000],
        "memory": sh("free -m")[1], "cpu_count": os.cpu_count(),
        "serial_ports": sorted(str(p) for pat in ("ttyTHS*", "ttyACM*", "ttyUSB*") for p in Path("/dev").glob(pat)),
    }
    for mod in ("numpy", "torch", "tensorrt", "cv2", "onnxruntime"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            info[f"{mod}_version"] = f"not available ({type(e).__name__})"
    try:
        import torch
        info["torch_cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["torch_device"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    info["trtexec"] = find_trtexec()
    return info


def find_trtexec():
    for c in (shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec", "/usr/local/tensorrt/bin/trtexec"):
        if c and Path(c).exists():
            return c
    return None


def stats_ms(xs):
    """min/max/mean/median/p90/p95/p99 in ms (pure Python: numpy is optional)."""
    a = sorted(x * 1e3 for x in xs)
    if not a:
        return dict.fromkeys(KEYS, float("nan"))

    def pct(q):
        k = (len(a) - 1) * q / 100.0
        i = int(k)
        return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (k - i)

    return dict(zip(KEYS, [a[0], a[-1], sum(a) / len(a), pct(50), pct(90), pct(95), pct(99)]))


def guarded(run, name, fn, *args):
    """Run one section; on any error log it, keep the partial results, carry on."""
    try:
        fn(run, *args)
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        run.results.setdefault("errors", {})[name] = traceback.format_exc()[-2000:]
        run.save()
        run.log(f"!! {name} failed: {type(e).__name__}: {e} (traceback in results.json); continuing")


def fmt(s):
    return "  ".join(f"{k} {s[k]:6.2f}" for k in ("median", "p90", "p99", "max")) + " ms"


# ------------------------------------------------------------------------------ tegrastats
class Tegrastats:
    def __init__(self, path):
        self.p = None
        if shutil.which("tegrastats"):
            self.f = open(path, "w")
            self.p = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=self.f, stderr=subprocess.STDOUT)

    def stop(self):
        if self.p:
            self.p.terminate()
            try:
                self.p.wait(5)
            except subprocess.TimeoutExpired:
                self.p.kill()
            self.f.close()


# -------------------------------------------------------------------------------- TensorRT
def trtexec(run: Run, name, args, timeout=1800):
    exe = run.results["system"]["trtexec"]
    cmd = [exe, *args]
    t = time.time()
    rc, out = sh(cmd, timeout=timeout)
    (run.root / "logs" / f"{name}.log").write_text(" ".join(cmd) + "\n\n" + out)
    r = {"rc": rc, "seconds": round(time.time() - t, 1)}
    m = LAT_RE.findall(out)
    if m:
        r["latency_ms"] = dict(zip(KEYS, map(float, m[-1])))
    m = GPU_RE.findall(out)
    if m:
        r["gpu_compute_ms"] = dict(zip(KEYS, map(float, m[-1])))
    m = QPS_RE.findall(out)
    if m:
        r["throughput_qps"] = float(m[-1])
    if rc != 0 or "latency_ms" not in r:
        r["error_tail"] = out[-1500:]
    return r


def build_and_time(run: Run, onnx: Path, fp16: bool, duration: float):
    prec = "fp16" if fp16 else "fp32"
    name = f"{onnx.stem}_{prec}"
    engine = run.root / "engines" / f"{name}.engine"
    args = [f"--onnx={onnx}", f"--saveEngine={engine}", "--warmUp=2000", f"--duration={duration:g}",
            "--useSpinWait", "--avgRuns=1"]
    if fp16:
        args.append("--fp16")
    r = trtexec(run, f"build_{name}", args)
    r["engine"] = str(engine) if engine.exists() else None
    return name, r


def trt_section(run: Run, gn_configs, duration):
    import configs as M
    out = run.results.setdefault("tensorrt", {})
    run.log("== TensorRT: policy step (100 Hz -> must stay well under 10 ms) ==")
    for fp16 in (False, True):
        name, r = build_and_time(run, ONNX / "policy_step.onnx", fp16, duration)
        out[name] = r; run.save()
        run.log(f"{name:28s} " + (fmt(r["latency_ms"]) if "latency_ms" in r else f"FAILED rc={r['rc']}")
                + f"   (build+run {r['seconds']} s)")
    run.log("== TensorRT: GateNet FP16 ==")
    for f, h, w in gn_configs:
        onnx = ONNX / f"{M.gatenet_name(f, h, w)}.onnx"
        name, r = build_and_time(run, onnx, True, duration)
        out[name] = r; run.save()
        if "latency_ms" in r:
            fps = 1000.0 / r["latency_ms"]["median"]
            run.log(f"{name:28s} " + fmt(r["latency_ms"]) + f"   ~{fps:5.0f} fps alone   (build+run {r['seconds']} s)")
        else:
            run.log(f"{name:28s} FAILED rc={r['rc']} (see logs/build_{name}.log)")


def together_section(run: Run, gn_configs, duration):
    """Policy at 100 Hz (idle time fills the rest of each 10 ms) while GateNet runs at a
    camera rate in a second trtexec process: what the policy sees under that load."""
    import configs as M
    trt = run.results.get("tensorrt", {})
    pol = trt.get("policy_step_fp16", {})
    if not pol.get("engine"):
        run.log("together: no policy engine, skipped"); return
    out = run.results.setdefault("together", {})
    pol_lat = pol["latency_ms"]["median"]
    run.log("== TensorRT together: policy @100 Hz + GateNet @camera rate ==")
    for f, h, w in gn_configs:
        g = trt.get(f"{M.gatenet_name(f, h, w)}_fp16", {})
        if not g.get("engine"):
            continue
        for fps in (30, 60):
            g_lat = g["latency_ms"]["median"]
            if g_lat > 1000.0 / fps:
                out[f"{M.gatenet_name(f, h, w)}@{fps}"] = {"skipped": f"GateNet alone {g_lat:.1f} ms > {1000 / fps:.1f} ms frame"}
                continue
            bg_name = f"bg_{M.gatenet_name(f, h, w)}_{fps}fps"
            bg_cmd = [run.results["system"]["trtexec"], f"--loadEngine={g['engine']}", "--warmUp=500",
                      f"--duration={duration + 6:g}", f"--idleTime={max(int(1000 / fps - g_lat), 0)}"]
            bg_log = open(run.root / "logs" / f"{bg_name}.log", "w")
            bg = subprocess.Popen(bg_cmd, stdout=bg_log, stderr=subprocess.STDOUT)
            time.sleep(3)
            r = trtexec(run, f"pol_with_{bg_name}",
                        [f"--loadEngine={pol['engine']}", "--warmUp=500", f"--duration={duration:g}",
                         f"--idleTime={max(int(10 - pol_lat), 0)}"])
            bg.wait(); bg_log.close()
            bg_out = (run.root / "logs" / f"{bg_name}.log").read_text()
            m = LAT_RE.findall(bg_out)
            r["gatenet_latency_ms"] = dict(zip(KEYS, map(float, m[-1]))) if m else None
            key = f"{M.gatenet_name(f, h, w)}@{fps}"
            out[key] = r; run.save()
            if "latency_ms" in r:
                gl = r["gatenet_latency_ms"]
                run.log(f"policy with {key:30s} policy {fmt(r['latency_ms'])}"
                        + (f" | GateNet median {gl['median']:.2f} ms" if gl else ""))


# --------------------------------------------------------------------------------- PyTorch
def torch_section(run: Run, gn_configs, duration):
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        run.log(f"PyTorch not available ({type(e).__name__}): skipped"); return
    if not torch.cuda.is_available():
        run.log("PyTorch has no CUDA: skipped"); return
    import models as M
    out = run.results.setdefault("torch", {})
    dev = "cuda"
    torch.backends.cudnn.benchmark = True

    def timeit(fn, n_warm=50, seconds=duration):
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        ts, end = [], time.time() + seconds
        while time.time() < end:
            t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        return stats_ms(ts)

    run.log("== PyTorch eager (what a plain Python deployment would get) ==")
    for dtype in (torch.float32, torch.float16):
        pol = M.PolicyStep().to(dev, dtype).eval()
        x = M.PolicyStep.example_inputs(dev, dtype)
        with torch.inference_mode():
            s = timeit(lambda: pol(*x))
        name = f"policy_step_{str(dtype).split('.')[-1]}"
        out[name] = s; run.save()
        run.log(f"{name:28s} {fmt(s)}")
    nets = {}
    for f, h, w in gn_configs:
        net = M.GateNet(f).to(dev, torch.float16).eval()
        img = torch.rand(1, 3, h, w, device=dev, dtype=torch.float16)
        with torch.inference_mode():
            s = timeit(lambda: net(img))
        name = M.gatenet_name(f, h, w) + "_float16"
        out[name] = s; nets[(f, h, w)] = (net, img); run.save()
        run.log(f"{name:28s} {fmt(s)}")

    # Python control loop at 100 Hz with GateNet in a thread at 30 fps
    run.log("== PyTorch: 100 Hz policy loop + GateNet thread @30 fps (deadline = 10 ms) ==")
    pol = M.PolicyStep().to(dev, torch.float16).eval()
    x = M.PolicyStep.example_inputs(dev, torch.float16)
    for (f, h, w), (net, img) in nets.items():
        stop = threading.Event()

        def camera():
            s = torch.cuda.Stream()
            nxt = time.perf_counter()
            while not stop.is_set():
                with torch.cuda.stream(s), torch.inference_mode():
                    net(img)
                s.synchronize()
                nxt += 1 / 30
                time.sleep(max(0.0, nxt - time.perf_counter()))

        th = threading.Thread(target=camera, daemon=True); th.start()
        lat, late, nxt = [], 0, time.perf_counter()
        end = time.time() + duration
        with torch.inference_mode():
            while time.time() < end:
                t = time.perf_counter(); pol(*x); torch.cuda.synchronize(); d = time.perf_counter() - t
                lat.append(d)
                nxt += 0.01
                slack = nxt - time.perf_counter()
                if slack < 0:
                    late += 1; nxt = time.perf_counter()
                else:
                    time.sleep(slack)
        stop.set(); th.join()
        s = stats_ms(lat)
        key = f"loop_with_{M.gatenet_name(f, h, w)}@30"
        out[key] = dict(policy_ms=s, missed_deadlines=late, steps=len(lat), rate_hz=len(lat) / duration)
        run.save()
        run.log(f"{key:40s} policy {fmt(s)}  missed {late}/{len(lat)}  ({len(lat) / duration:.0f} Hz)")


# ---------------------------------------------------------------------------------- OpenCV
def cv_section(run: Run, duration):
    try:
        import cv2
        import numpy as np
    except Exception as e:  # noqa: BLE001
        run.log(f"OpenCV not available ({type(e).__name__}): skipped"); return
    out = run.results.setdefault("opencv", {})
    run.log("== OpenCV CPU: camera frame -> resize -> undistort remap ==")
    cams = {"imx219_1640x1232": (1232, 1640)}              # the chosen camera mode
    for cam, (H, W) in cams.items():
        frame = (np.random.rand(H, W, 3) * 255).astype(np.uint8)
        for (h, w) in ((240, 320), (480, 640)):
            mx = np.tile(np.linspace(0, w - 1, w, dtype=np.float32), (h, 1))
            my = np.tile(np.linspace(0, h - 1, h, dtype=np.float32)[:, None], (1, w))
            ts, end = [], time.time() + min(duration, 5)
            while time.time() < end:
                t = time.perf_counter()
                small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                cv2.remap(small, mx, my, cv2.INTER_LINEAR)
                ts.append(time.perf_counter() - t)
            s = stats_ms(ts)
            key = f"{cam}->{h}x{w}"
            out[key] = s; run.save()
            run.log(f"{key:34s} {fmt(s)}")


# ------------------------------------------------------------------------------ UART / MSP
def uart_section(run: Run, dev, baud, seconds):
    import uart_msp as U
    out = run.results.setdefault("uart", {"device": dev, "baud": baud})
    run.log(f"== UART / MSP: {dev} @ {baud} baud (props OFF; neutral RC only) ==")
    budget = U.wire_budget(baud)
    out["budget"] = budget
    run.log(f"per 10 ms cycle: tx {budget['tx_bytes_per_cycle']} B, rx {budget['rx_bytes_per_cycle']} B, "
            f"{budget['wire_time_ms_per_cycle']:.2f} ms on the wire, link load {100 * budget['link_load_100hz']:.0f}% at 100 Hz")
    try:
        ser = U.Serial(dev, baud)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"; run.save()
        hint = ("baud not available on this OS" if isinstance(e, ValueError) else
                "permission? try: sudo usermod -aG dialout $USER, then log out/in (or run with sudo)")
        run.log(f"cannot open {dev}: {out['error']}  ({hint})")
        return
    try:
        ident = U.identify(ser)
        out["fc"] = ident; run.save()
        run.log(f"FC: {ident}")
        if not ident.get("api_version"):
            run.log("no MSP answer: check wiring (TX<->RX crossed, GND), the FC port has MSP enabled at this baud,"
                    " and nothing else (e.g. a serial console) uses the port")
            return
        rd = U.sample_readings(ser)
        out["sample"] = rd; run.save()
        run.log(f"sample reading (sanity): {rd}")

        def report(key, r):
            cyc = stats_ms(r["cycle_s"])
            rtt = {U_NAMES[c]: stats_ms(v) for c, v in r["rtt_s"].items() if v}
            out[key] = dict(cycles=r["cycles"], achieved_hz=r["achieved_hz"], cycle_ms=cyc, rtt_ms=rtt,
                            timeouts=r["timeouts"], errors=r["errors"], missed_deadlines=r["missed_deadlines"],
                            late_frames=r["late_frames"],
                            failed_cycles=len(r["fail_at_s"]),
                            first_fail_s=r["fail_at_s"][0] if r["fail_at_s"] else None,
                            last_fail_s=r["fail_at_s"][-1] if r["fail_at_s"] else None,
                            fc_alive_after=U.alive(ser),
                            tx_bytes_per_s=r["tx_bytes"] / r["seconds"], rx_bytes_per_s=r["rx_bytes"] / r["seconds"])
            run.save()
            run.log(f"{key:30s} cycle {fmt(cyc)} | {r['achieved_hz']:.0f} Hz, missed {r['missed_deadlines']}, "
                    f"timeouts {r['timeouts']}, late frames {r['late_frames']}, bad frames {r['errors']}")
            for name, s in rtt.items():
                run.log(f"{'':32s}{name:14s} round trip {fmt(s)}")
            o = out[key]
            if o["failed_cycles"]:
                run.log(f"{'':32s}cycles with a timeout: {o['failed_cycles']}/{r['cycles']}, "
                        f"first at {o['first_fail_s']:.2f} s, last at {o['last_fail_s']:.2f} s")
            run.log(f"{'':32s}FC answers MSP after this test: {'yes' if o['fc_alive_after'] else 'NO (FC stopped answering)'}")
            run.save()

        U_NAMES = {U.MSP_RAW_IMU: "RAW_IMU", U.MSP_ATTITUDE: "ATTITUDE", U.MSP_SET_RAW_RC: "SET_RAW_RC"}
        report("sequential_100hz", U.run_cycles(ser, seconds, 100.0, pipelined=False))
        report("pipelined_100hz", U.run_cycles(ser, seconds, 100.0, pipelined=True))
        report("sequential_after_pipelined", U.run_cycles(ser, 5.0, 100.0, pipelined=False))
        report("sequential_max_rate", U.run_cycles(ser, 5.0, 0.0, pipelined=False))
        report("pipelined_max_rate", U.run_cycles(ser, min(seconds, 5.0), 0.0, pipelined=True, timeout=0.05))

        # the same while GateNet keeps the GPU busy (CPU/IRQ contention on the Jetson)
        trt = run.results.get("tensorrt", {})
        eng = next((v["engine"] for k, v in trt.items() if k.startswith("gatenet") and v.get("engine")), None)
        if eng and run.results["system"].get("trtexec"):
            bg = subprocess.Popen([run.results["system"]["trtexec"], f"--loadEngine={eng}",
                                   f"--duration={seconds + 4:g}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            report("sequential_100hz_gpu_busy", U.run_cycles(ser, seconds, 100.0, pipelined=False))
            bg.wait()
    finally:
        ser.close()


# ---------------------------------------------------------------------------------- report
def summary(run: Run):
    R = run.results
    lines = ["CTBA Jetson benchmark", f"started {R['started']}, took {R.get('elapsed_s', 0):.0f} s",
             f"device: {R['system'].get('model')}", f"power mode: {R['system'].get('nvpmodel', '').splitlines()[:2]}", ""]
    trt = R.get("tensorrt", {})
    if trt:
        lines.append("TensorRT alone (median / p99 ms):")
        for k, v in trt.items():
            if "latency_ms" in v:
                lines.append(f"  {k:28s} {v['latency_ms']['median']:7.2f} / {v['latency_ms']['p99']:7.2f}")
            else:
                lines.append(f"  {k:28s} FAILED")
        p = trt.get("policy_step_fp16", {}).get("latency_ms")
        if p:
            ok = p["p99"] < 5.0
            lines.append(f"  -> policy FP16 p99 {p['p99']:.2f} ms: "
                         + ("fits 100 Hz with room" if ok else "TIGHT for 100 Hz (budget 10 ms incl. I/O)"
                            if p["p99"] < 10 else "DOES NOT FIT 100 Hz"))
    tog = R.get("together", {})
    if tog:
        lines.append("\nPolicy @100 Hz with GateNet running (policy median / p99 ms):")
        for k, v in tog.items():
            if "latency_ms" in v:
                lines.append(f"  {k:36s} {v['latency_ms']['median']:7.2f} / {v['latency_ms']['p99']:7.2f}")
            else:
                lines.append(f"  {k:36s} {v.get('skipped', 'FAILED')}")
    ua = R.get("uart", {})
    if ua:
        lines.append(f"\nUART / MSP {ua.get('device')} @ {ua.get('baud')} baud: FC {ua.get('fc', {}).get('fc_variant')} "
                     f"{ua.get('fc', {}).get('fc_version')}" + (f"  ERROR {ua['error']}" if "error" in ua else ""))
        if "budget" in ua:
            lines.append(f"  link load at 100 Hz: {100 * ua['budget']['link_load_100hz']:.0f}%")
        for k in ("sequential_100hz", "pipelined_100hz", "sequential_after_pipelined", "sequential_max_rate",
                  "sequential_100hz_gpu_busy", "pipelined_max_rate"):
            if k in ua:
                v = ua[k]
                lines.append(f"  {k:26s} cycle median {v['cycle_ms']['median']:.2f} / p99 {v['cycle_ms']['p99']:.2f} ms, "
                             f"{v['achieved_hz']:.0f} Hz, missed {v['missed_deadlines']}, timeouts {v['timeouts']}"
                             + ("" if v.get("fc_alive_after", True) else "  -> FC STOPPED ANSWERING"))
    tor = R.get("torch", {})
    loops = {k: v for k, v in tor.items() if k.startswith("loop_")}
    if loops:
        lines.append("\nPyTorch 100 Hz loop + GateNet thread @30 fps:")
        for k, v in loops.items():
            lines.append(f"  {k:44s} p99 {v['policy_ms']['p99']:6.2f} ms, missed {v['missed_deadlines']}/{v['steps']}")
    text = "\n".join(lines)
    (run.root / "summary.txt").write_text(text + "\n")
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--duration", type=float, default=None, help="seconds per timing run (default 15, quick 6)")
    ap.add_argument("--skip-trt", action="store_true")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--uart", default=None, help="FC serial device, e.g. /dev/ttyTHS1 (40-pin UART) or /dev/ttyACM0 (USB)")
    ap.add_argument("--baud", type=int, default=115200, help="must equal the FC port's MSP baud")
    ap.add_argument("--uart-seconds", type=float, default=20.0)
    ap.add_argument("--only-uart", action="store_true", help="skip GPU/CPU tests, just the FC link")
    args = ap.parse_args()
    if args.only_uart and not args.uart:
        ap.error("--only-uart needs --uart DEVICE")
    import configs as M
    duration = args.duration or (6.0 if args.quick else 15.0)
    gn = QUICK_GATENET if args.quick else M.GATENET_CONFIGS

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = Run(HERE / "results" / f"{socket.gethostname()}-{stamp}")
    run.results["system"] = system_info()
    run.results["args"] = vars(args)
    run.save()
    run.log(f"results -> {run.root}")
    for k in ("model", "l4t", "nvpmodel", "torch_version", "tensorrt_version", "cv2_version", "trtexec"):
        run.log(f"  {k}: {str(run.results['system'].get(k)).splitlines()[0] if run.results['system'].get(k) else None}")
    if "MAXN" not in (run.results["system"].get("nvpmodel") or ""):
        run.log("  NOTE: not in a MAXN power mode; numbers are for the current mode (recorded above)")

    ts = Tegrastats(run.root / "tegrastats.log")
    try:
        if args.only_uart:
            guarded(run, "uart", uart_section, args.uart, args.baud, args.uart_seconds)
            return
        if not args.skip_trt:
            if run.results["system"]["trtexec"]:
                guarded(run, "tensorrt", trt_section, gn, duration)
                guarded(run, "together", together_section, [c for c in gn if c in QUICK_GATENET] or gn[:2], duration)
            else:
                run.log("trtexec not found (TensorRT): TensorRT section skipped")
        if not args.skip_torch:
            guarded(run, "torch", torch_section, QUICK_GATENET, duration)
        guarded(run, "opencv", cv_section, duration)
        if args.uart:
            guarded(run, "uart", uart_section, args.uart, args.baud, args.uart_seconds)
    except KeyboardInterrupt:
        run.log("interrupted: partial results kept")
    finally:
        ts.stop()
        run.save()
        summary(run)
        run.log(f"done: {run.root}")


if __name__ == "__main__":
    main()
