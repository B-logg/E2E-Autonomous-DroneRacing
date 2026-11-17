"""Camera latency (checklist C4): light hits the sensor -> the frame is in our program.

Method: an LED in front of the lens is switched on by the Jetson itself at a known time
t_on (same clock as the frame arrivals), and we time the first frame in which the LED
is lit.  Repeated ~100 times at random phase against the camera's frame clock:

    L = t_frame_arrives - t_on   ranges over  [d, d + T]   (T = frame period)

d is what the sim calls the camera part of IMAGE_DELAY (scene instant -> frame in hand).
Since the LED switches at a random point within a frame, median(L) - T/2 estimates d.
The capture pipeline is the deployed one: nvarguscamerasrc 1640x1232@30 -> nvvidconv
(hardware resize to 320x240, GateNet's input) -> appsink, newest frame only.

Modes
  --led PIN      LED + 330 ohm resistor from header pin PIN (BOARD numbering) to GND.
                 Accurate (GPIO switches in microseconds).  Recommended.
  --screen       no wiring: a full-screen window on a monitor attached to the Jetson
                 flips black/white.  Includes the monitor's own lag (10-40 ms), so it is
                 an UPPER bound only.
  --simulate MS  no camera: a fake camera with MS ms of delay (checks the script itself).

  --blink        only blink the LED at 1 Hz for 10 s (wiring check).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------- light source
class Led:
    def __init__(self, pin):
        try:
            import Jetson.GPIO as GPIO
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Jetson.GPIO not available ({e}).  Install: sudo pip3 install Jetson.GPIO, "
                     "then: sudo groupadd -f gpio && sudo usermod -aG gpio $USER, log out/in (or run with sudo)")
        self.G, self.pin = GPIO, pin
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

    def set(self, on):
        self.G.output(self.pin, self.G.HIGH if on else self.G.LOW)
        return time.monotonic()

    def close(self):
        self.set(False)
        self.G.cleanup(self.pin)


class Screen:
    def __init__(self):
        import cv2
        import numpy as np
        self.cv2 = cv2
        self.img = {False: np.zeros((480, 640, 3), np.uint8), True: np.full((480, 640, 3), 255, np.uint8)}
        cv2.namedWindow("latency", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("latency", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.set(False)

    def set(self, on):
        self.cv2.imshow("latency", self.img[on])
        self.cv2.waitKey(1)
        return time.monotonic()

    def close(self):
        self.cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------- camera
def pipeline(a):
    exp = ""
    if a.exposure_us:
        ns = int(a.exposure_us * 1000)
        exp = f' exposuretimerange="{ns} {ns}" aelock=true'
    return (f"nvarguscamerasrc sensor-id={a.sensor_id}{exp} ! "
            f"video/x-raw(memory:NVMM),width={a.width},height={a.height},framerate={a.fps}/1,format=NV12 ! "
            f"nvvidconv ! video/x-raw,width={a.out_w},height={a.out_h},format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")


class Camera:
    """Reader thread: every frame -> (arrival time, frame).  Always pulls, so nothing queues."""

    def __init__(self, a):
        import cv2
        self.cv2 = cv2
        self.desc = a.pipeline or pipeline(a)
        import re
        if not re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()):
            sys.exit("this OpenCV has no GStreamer support (cv2.getBuildInformation()).  "
                     "Use the JetPack OpenCV: sudo apt install python3-opencv")
        self.cap = cv2.VideoCapture(self.desc, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            sys.exit(f"cannot open the camera pipeline:\n  {self.desc}\n"
                     "check: ls /dev/video*, the ribbon cable, and that nothing else uses the camera "
                     "(sudo systemctl restart nvargus-daemon)")
        self.lock = threading.Lock()
        self.on_frame = None          # callback(t, frame)
        self.stop = False
        self.arrivals = []
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self.stop:
            ok, f = self.cap.read()
            t = time.monotonic()
            if not ok:
                continue
            self.arrivals.append(t)
            cb = self.on_frame
            if cb:
                cb(t, f)

    def close(self):
        self.stop = True
        self.th.join(1.0)
        self.cap.release()


class FakeCamera:
    """Global-shutter camera at `fps`: exposure instant every T, frame arrives `delay` later."""

    def __init__(self, a, light):
        import numpy as np
        self.np, self.light, self.T, self.d = np, light, 1.0 / a.fps, a.simulate / 1000.0
        self.desc = f"simulated, delay {a.simulate} ms, {a.fps} fps"
        self.on_frame, self.stop, self.arrivals = None, False, []
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        t_exp = time.monotonic() + random.random() * self.T
        while not self.stop:
            lit = self.light.state_at(t_exp)
            time.sleep(max(t_exp + self.d - time.monotonic(), 0))
            t = time.monotonic()
            self.arrivals.append(t)
            f = self.np.full((240, 320, 3), 200 if lit else 30, self.np.uint8)
            cb = self.on_frame
            if cb:
                cb(t, f)
            t_exp += self.T

    def close(self):
        self.stop = True
        self.th.join(1.0)


class FakeLight:
    def __init__(self):
        self.hist = [(0.0, False)]

    def set(self, on):
        t = time.monotonic()
        self.hist.append((t, on))
        return t

    def state_at(self, t):
        s = False
        for tt, on in self.hist:
            if tt <= t:
                s = on
        return s

    def close(self):
        pass


# ------------------------------------------------------------------------------- procedure
def gray(f):
    return f.mean(axis=2)


def locate(cam, light, log):
    """Find the pixels the light changes: mean frame with it on minus off."""
    import numpy as np
    frames = []
    cam.on_frame = lambda t, f: frames.append(f)
    acc = {False: [], True: []}
    for _ in range(4):
        for on in (False, True):
            light.set(on)
            time.sleep(0.3)
            frames.clear()
            time.sleep(0.25)
            acc[on] += [gray(f) for f in frames]
    light.set(False)
    cam.on_frame = None
    if not acc[True] or not acc[False]:
        sys.exit("no frames arrive from the camera")
    diff = np.mean(acc[True], 0) - np.mean(acc[False], 0)
    peak = float(diff.max())
    if peak < 20:
        sys.exit(f"the light barely shows in the image (brightest change {peak:.0f}/255).  "
                 "Put the LED 3-10 cm in front of the lens, in the middle of the view, room light dimmer; "
                 "check the wiring with --blink.")
    mask = diff > 0.5 * peak
    ys, xs = np.nonzero(mask)
    roi = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    off = float(np.mean([g[roi].mean() for g in acc[False]]))
    on = float(np.mean([g[roi].mean() for g in acc[True]]))
    log(f"light found: rows {ys.min()}-{ys.max()}, cols {xs.min()}-{xs.max()} of "
        f"{diff.shape[0]}x{diff.shape[1]}, level off {off:.0f} -> on {on:.0f}")
    return roi, off, on, int(ys.mean())


def trials(cam, light, roi, thr, n, log):
    lat, misses = [], 0
    state = {"t_on": None, "hit": None}

    def cb(t, f):
        if state["t_on"] is not None and state["hit"] is None and t > state["t_on"]:
            if gray(f[roi]).mean() > thr:
                state["hit"] = t

    cam.on_frame = cb
    for i in range(n):
        time.sleep(random.uniform(0.25, 0.45))          # dark for >= 7 frames
        state["hit"] = None
        state["t_on"] = light.set(True)
        end = state["t_on"] + 0.5
        while state["hit"] is None and time.monotonic() < end:
            time.sleep(0.0005)
        if state["hit"] is None:
            misses += 1
        else:
            lat.append(state["hit"] - state["t_on"])
        state["t_on"] = None
        light.set(False)
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{n}  median so far {1e3 * statistics.median(lat):.1f} ms" if lat else f"  {i + 1}/{n}")
    cam.on_frame = None
    return lat, misses


def pct(a, q):
    a = sorted(a)
    k = (len(a) - 1) * q
    lo = int(k)
    return a[lo] + (a[min(lo + 1, len(a) - 1)] - a[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--led", type=int, metavar="PIN", help="40-pin header pin (BOARD numbering), e.g. 15")
    g.add_argument("--screen", action="store_true")
    g.add_argument("--simulate", type=float, metavar="MS")
    ap.add_argument("--blink", action="store_true", help="with --led: blink 10 s and exit")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--width", type=int, default=1640)
    ap.add_argument("--height", type=int, default=1232)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out-w", type=int, default=320)
    ap.add_argument("--out-h", type=int, default=240)
    ap.add_argument("--exposure-us", type=float, default=None, help="fixed exposure (default: auto)")
    ap.add_argument("--pipeline", default=None, help="full GStreamer pipeline ending in appsink (overrides)")
    a = ap.parse_args()

    if a.blink:
        if not a.led:
            ap.error("--blink needs --led PIN")
        led = Led(a.led)
        print(f"blinking pin {a.led} at 1 Hz for 10 s ...")
        for _ in range(10):
            led.set(True); time.sleep(0.5); led.set(False); time.sleep(0.5)
        led.close()
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = HERE / "results" / f"camera-{socket.gethostname()}-{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    light = Led(a.led) if a.led else Screen() if a.screen else FakeLight()
    cam = FakeCamera(a, light) if a.simulate is not None else Camera(a)
    mode = f"LED on pin {a.led}" if a.led else "screen (upper bound: includes monitor lag)" if a.screen else "simulated"
    log(f"camera latency, {mode}")
    log(f"pipeline: {cam.desc}")
    log(f"exposure: {'%g us fixed' % a.exposure_us if a.exposure_us else 'auto'}")
    try:
        time.sleep(2.0)                                  # let auto-exposure settle
        n0 = len(cam.arrivals)
        time.sleep(2.0)
        arr = cam.arrivals[n0:]
        if len(arr) < 10:
            sys.exit("fewer than 10 frames in 2 s: the camera is not streaming")
        dt = [b - x for x, b in zip(arr, arr[1:])]
        T = statistics.median(dt)
        log(f"frames: {1 / T:.1f} fps (interval median {1e3 * T:.1f} ms, max {1e3 * max(dt):.1f} ms)")

        roi, off, on, row = locate(cam, light, log)
        lat, misses = trials(cam, light, roi, 0.5 * (off + on), a.trials, log)
    finally:
        cam.close()
        light.close()
    if len(lat) < 10:
        sys.exit(f"only {len(lat)} detections ({misses} misses): check the light placement")

    ms = [1e3 * x for x in lat]
    med = statistics.median(ms)
    d = med - 0.5e3 * T
    res = dict(mode=mode, pipeline=cam.desc, exposure_us=a.exposure_us, fps=1 / T, frame_period_ms=1e3 * T,
               light_row=row, image_rows=a.out_h, trials=a.trials, misses=misses,
               latency_ms=dict(min=min(ms), p10=pct(ms, .1), median=med, p90=pct(ms, .9), max=max(ms)),
               camera_delay_estimate_ms=d, raw_ms=ms)
    log("")
    log(f"light on -> first lit frame in hand ({len(ms)} trials, {misses} misses):")
    log(f"  min {min(ms):.1f}  p10 {pct(ms, .1):.1f}  median {med:.1f}  p90 {pct(ms, .9):.1f}  max {max(ms):.1f} ms")
    log(f"  spread p90-p10 {pct(ms, .9) - pct(ms, .1):.1f} ms (expected about one frame, {1e3 * T:.1f} ms)")
    log(f"camera delay (scene instant -> frame in program) ~ median - T/2 = {d:.1f} ms "
        f"(it cannot exceed min = {min(ms):.1f} ms)")
    if a.screen:
        log("  (screen mode: this still contains the monitor's lag, so the true value is lower)")
    log(f"  light at image row {row}/{a.out_h}: rolling shutter reads the middle row about half a readout "
        f"(~10 ms for 1640x1232) after the top row")
    (root / "results.json").write_text(json.dumps(res, indent=1))
    (root / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nsaved: {root}")


if __name__ == "__main__":
    main()
