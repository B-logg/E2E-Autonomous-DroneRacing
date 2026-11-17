"""MSP over a UART, no dependencies (Linux termios), and the 100 Hz I/O benchmark:
every 10 ms read MSP_RAW_IMU and MSP_ATTITUDE and write MSP_SET_RAW_RC, the exact
traffic the deployed policy needs.

Safety: the RC frame written is NEUTRAL (roll/pitch/yaw 1500, throttle 1000, every
AUX channel 1000 = off), so nothing arms and nothing moves even if the FC applies it.
Run with the props off anyway.

MSP v1 frames:  request  '$M<' len cmd payload xor     response  '$M>' len cmd payload xor
                (error  '$M!' ...)
"""
from __future__ import annotations

import os
import select
import struct
import termios
import time

MSP_API_VERSION, MSP_FC_VARIANT, MSP_FC_VERSION = 1, 2, 3
MSP_STATUS, MSP_RAW_IMU, MSP_RC, MSP_ATTITUDE = 101, 102, 105, 108
MSP_SET_RAW_RC = 200

NEUTRAL_RC = [1500, 1500, 1000, 1500] + [1000] * 4      # AETR order (BF default map): roll, pitch, throttle, yaw, AUX1-4


def frame(cmd, payload=b""):
    body = bytes([len(payload), cmd]) + payload
    cs = 0
    for b in body:
        cs ^= b
    return b"$M<" + body + bytes([cs])


class Serial:
    def __init__(self, dev, baud):
        rate = getattr(termios, f"B{baud}", None)
        if rate is None:
            raise ValueError(f"baud {baud} not supported by this termios")
        self.fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        a = termios.tcgetattr(self.fd)
        a[0] = 0                                  # iflag: raw
        a[1] = 0                                  # oflag
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        a[3] = 0                                  # lflag: no echo, non-canonical
        a[4] = a[5] = rate
        a[6][termios.VMIN] = 0
        a[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        termios.tcflush(self.fd, termios.TCIOFLUSH)
        self.buf = bytearray()
        self.tx_bytes = self.rx_bytes = 0

    def close(self):
        os.close(self.fd)

    def write(self, data):
        n = 0
        while n < len(data):
            try:
                n += os.write(self.fd, data[n:])
            except BlockingIOError:
                select.select([], [self.fd], [], 0.01)
        self.tx_bytes += len(data)

    def _fill(self, timeout):
        r, _, _ = select.select([self.fd], [], [], max(timeout, 0))
        if r:
            try:
                chunk = os.read(self.fd, 4096)
            except BlockingIOError:
                chunk = b""
            self.buf += chunk
            self.rx_bytes += len(chunk)

    def drain(self):
        """Drop whatever arrived late (answers to a previous cycle that timed out).
        Returns the number of complete late frames dropped."""
        self._fill(0)
        n = self.buf.count(b"$M>") + self.buf.count(b"$M!")
        self.buf.clear()
        return n

    def read_frame(self, deadline):
        """Next MSP response -> (cmd, payload, ok, t_received) or None on timeout.
        ok = False: error frame ('$M!') or bad checksum."""
        while True:
            i = self.buf.find(b"$M")
            if i > 0:
                del self.buf[:i]
            if len(self.buf) >= 5 and self.buf[:2] == b"$M":
                kind, n, cmd = self.buf[2], self.buf[3], self.buf[4]
                if len(self.buf) >= 6 + n:
                    payload = bytes(self.buf[5:5 + n])
                    cs = n ^ cmd
                    for b in payload:
                        cs ^= b
                    ok = (cs == self.buf[5 + n]) and kind == ord(">")
                    del self.buf[:6 + n]
                    return cmd, payload, ok, time.perf_counter()
            elif len(self.buf) >= 2 and self.buf[:2] != b"$M":
                del self.buf[:1]
                continue
            left = deadline - time.perf_counter()
            if left <= 0:
                return None
            self._fill(left)


def request(ser: Serial, cmd, payload=b"", timeout=0.05):
    t = time.perf_counter()
    ser.write(frame(cmd, payload))
    while True:
        r = ser.read_frame(t + timeout)
        if r is None:
            return None, None
        if r[0] == cmd:
            return r[1] if r[2] else None, r[3] - t


def identify(ser: Serial):
    out = {}
    p, rtt = request(ser, MSP_API_VERSION, timeout=0.3)
    out["api_version"] = f"{p[1]}.{p[2]}" if p and len(p) >= 3 else None
    out["first_rtt_ms"] = rtt * 1e3 if rtt else None
    p, _ = request(ser, MSP_FC_VARIANT, timeout=0.3)
    out["fc_variant"] = p.decode(errors="ignore") if p else None
    p, _ = request(ser, MSP_FC_VERSION, timeout=0.3)
    out["fc_version"] = ".".join(map(str, p[:3])) if p and len(p) >= 3 else None
    p, _ = request(ser, MSP_STATUS, timeout=0.3)
    if p and len(p) >= 10:
        out["cycle_time_us"] = struct.unpack_from("<H", p, 0)[0]
        out["flight_mode_flags"] = struct.unpack_from("<I", p, 6)[0]
    return out


def alive(ser: Serial, tries=3):
    """Does the FC still answer MSP?  (A few tries, 0.3 s apart.)"""
    for _ in range(tries):
        ser.drain()
        p, rtt = request(ser, MSP_API_VERSION, timeout=0.3)
        if p:
            return True
        time.sleep(0.3)
    return False


def sample_readings(ser: Serial):
    """One decoded RAW_IMU + ATTITUDE (sanity check of units / frames on the bench)."""
    out = {}
    p, _ = request(ser, MSP_RAW_IMU)
    if p and len(p) >= 12:
        v = struct.unpack_from("<9h", p.ljust(18, b"\0"))
        out["acc_raw"] = v[0:3]          # 512 LSB/g on BF 4.x MSP
        out["gyro_raw"] = v[3:6]         # deg/s
    p, _ = request(ser, MSP_ATTITUDE)
    if p and len(p) >= 6:
        r, pt, y = struct.unpack_from("<hhh", p)
        out["attitude_deg"] = (r / 10, pt / 10, y)
    return out


def cycle_frames():
    rc = struct.pack(f"<{len(NEUTRAL_RC)}H", *NEUTRAL_RC)
    return [frame(MSP_RAW_IMU), frame(MSP_ATTITUDE), frame(MSP_SET_RAW_RC, rc)], \
        [MSP_RAW_IMU, MSP_ATTITUDE, MSP_SET_RAW_RC]


def run_cycles(ser: Serial, seconds=20.0, rate_hz=100.0, pipelined=False, timeout=0.009):
    """The control-loop I/O at `rate_hz` (0 = as fast as possible).  Returns per-cycle
    times, per-request round trips, byte counts, errors."""
    frames, cmds = cycle_frames()
    period = 1.0 / rate_hz if rate_hz else 0.0
    cyc, rtts = [], {c: [] for c in cmds}
    timeouts = errors = late = stale = 0
    fail_at = []                                  # cycle start (s from run start) of every cycle with a timeout
    tx0, rx0 = ser.tx_bytes, ser.rx_bytes
    nxt = start = time.perf_counter()
    end = nxt + seconds
    n = 0
    while time.perf_counter() < end:
        stale += ser.drain()
        t0 = time.perf_counter()
        to0 = timeouts
        if pipelined:
            sent = {}
            for f, c in zip(frames, cmds):
                sent[c] = time.perf_counter(); ser.write(f)
            got = set()
            while len(got) < len(cmds):
                r = ser.read_frame(t0 + timeout)
                if r is None:
                    timeouts += len(cmds) - len(got); break
                c, _, ok, tr = r
                if c in sent and c not in got:
                    got.add(c); rtts[c].append(tr - sent[c])
                    errors += (not ok)
        else:
            for f, c in zip(frames, cmds):
                ts = time.perf_counter(); ser.write(f)
                while True:
                    r = ser.read_frame(ts + timeout)
                    if r is None:
                        timeouts += 1; break
                    if r[0] == c:
                        rtts[c].append(r[3] - ts); errors += (not r[2]); break
        cyc.append(time.perf_counter() - t0)
        if timeouts > to0:
            fail_at.append(t0 - start)
        n += 1
        if period:
            nxt += period
            slack = nxt - time.perf_counter()
            if slack < 0:
                late += 1; nxt = time.perf_counter()
            else:
                time.sleep(slack)
    return dict(cycles=n, seconds=seconds, achieved_hz=n / seconds, cycle_s=cyc, rtt_s=rtts,
                timeouts=timeouts, errors=errors, missed_deadlines=late, late_frames=stale, fail_at_s=fail_at,
                tx_bytes=ser.tx_bytes - tx0, rx_bytes=ser.rx_bytes - rx0)


def wire_budget(baud):
    """Bytes per cycle on the wire and the share of the link at 100 Hz (10 bits/byte)."""
    frames, _ = cycle_frames()
    tx = sum(len(f) for f in frames)
    rx = (6 + 18) + (6 + 6) + (6 + 0)          # RAW_IMU 18 B, ATTITUDE 6 B, SET_RAW_RC ack
    per_dir = max(tx, rx) * 10 * 100 / baud
    return dict(tx_bytes_per_cycle=tx, rx_bytes_per_cycle=rx, link_load_100hz=per_dir,
                wire_time_ms_per_cycle=(tx + rx) * 10 / baud * 1e3)
