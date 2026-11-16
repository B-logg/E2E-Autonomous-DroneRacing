"""Track `team44up` (the real room, G1 and G4 on 0.5 m stands), drawn in the team's
map frame: X, Y on the floor, Z up, m.  Numbers are skydreamer/track.py's
TEAM44UP_TRACK, ROOMS and LAUNCH; the example flight is run 7k 2.78M's first
successful 3-lap evaluation flight (stochgan_handoff/data/flight_states_*.npz),
converted from NED (X = y, Y = x, Z = -z).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
OUT = HERE / "track_team44up.png"
STATES = HERE.parent / "stochgan_handoff/data/flight_states_7k2.78M_team44up.npz"

OUTER, INNER = 1.5, 1.2
STAND = 0.5
# name: (X, Y, Z centre, yaw of the passage direction, deg CCW from +X, on a stand?)
GATES = {"G1": (-2.0, -1.0, 1.25, 90, True),
         "G2": (0.0, 4.0, 2.25, 90, False),
         "G3": (0.0, 4.0, 0.75, 270, False),
         "G4": (2.0, -1.0, 1.25, 270, True)}
ROOM = (-4.0, 4.0, -4.0, 7.0, 3.5)          # X lo, hi, Y lo, hi, ceiling
LAUNCH_D = (1.8, 2.0)                        # m before G1's plane (nominal 2.0)
LAUNCH_LAT, LAUNCH_YAW = 0.15, 5.0
LAUNCH = (-2.0, -1.0 - 2.0, 0.05)

INFO = (
    "트랙 team44up (확정, 2026-11-14)\n"
    "방      8 x 11 x 3.5 m (X -4~4, Y -4~7, Z 0~3.5), 장애물 없음\n"
    "게이트  외곽 1.5 m, 개구 1.2 m, 테두리 0.15 m (두께 무시)\n"
    "순서    G1 → G2 → 스플릿-S → G3 → G4 → U턴 → G1, 3바퀴\n"
    "G1      (-2.0, -1.0, 1.25)  통과 +Y, 0.5 m 받침대\n"
    "G2      ( 0.0,  4.0, 2.25)  통과 +Y, G3 위에 적층 (천장까지 0.5 m)\n"
    "G3      ( 0.0,  4.0, 0.75)  통과 -Y\n"
    "G4      ( 2.0, -1.0, 1.25)  통과 -Y, 0.5 m 받침대\n"
    "이륙    바닥, G1 중심선 위, G1 평면 앞 2.0 m, G1 정면을 봄\n"
    "        학습 범위: 1.8-2.0 m, 좌우 ±0.15 m, 기수 ±5°\n"
    "궤적    run 7k 2.78M 성공 비행 예시 (3바퀴)"
)


def gate_corners(X, Y, Z, half):
    """Square in the X-Z plane (every gate here is passed along ±Y)."""
    return np.array([[X - half, Y, Z - half], [X + half, Y, Z - half],
                     [X + half, Y, Z + half], [X - half, Y, Z + half], [X - half, Y, Z - half]])


def flight():
    z = np.load(STATES)
    ep = int(np.where(z["ep_success"])[0][0])
    m = z["episode"] == ep
    p, v = z["p"][m], z["v"][m]
    P = np.c_[p[:, 1], p[:, 0], -p[:, 2]]
    return P, np.linalg.norm(v, axis=1)


def colored(ax, a, b, speed, lw=1.6):
    pts = np.c_[a, b][:, None, :]
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], 1), cmap="viridis",
                        norm=plt.Normalize(0, 8), lw=lw, alpha=0.9)
    lc.set_array(speed[:-1])
    ax.add_collection(lc)
    return lc


def draw():
    P, spd = flight()
    xl, xh, yl, yh, ceil = ROOM
    fig = plt.figure(figsize=(17, 12.5))
    fig.suptitle("트랙 team44up: 게이트 배치, 방, 이륙 위치 (지도 좌표 X, Y 바닥 / Z 위, m)", fontsize=14)
    fig.text(0.015, 0.94, INFO, fontsize=9.5, va="top", family="AppleGothic", linespacing=1.55,
             bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f2", ec="0.6"))

    # ---- 3D ----
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    for name, (X, Y, Z, yaw, st) in GATES.items():
        for half, c, lw in ((OUTER / 2, "tab:orange", 2.5), (INNER / 2, "tab:orange", 1.2)):
            g = gate_corners(X, Y, Z, half)
            ax.plot(g[:, 0], g[:, 1], g[:, 2], color=c, lw=lw)
        if st:
            for dx in (-OUTER / 2, OUTER / 2):
                ax.plot([X + dx] * 2, [Y] * 2, [0, STAND], color="0.35", lw=3)
        ax.text(X, Y, Z + OUTER / 2 + 0.2, name, fontsize=11, ha="center", color="k")
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color="tab:blue", lw=0.9, alpha=0.8)
    ax.scatter(*LAUNCH, color="crimson", s=60, marker="^")
    # room box
    for zz in (0, ceil):
        ax.plot([xl, xh, xh, xl, xl], [yl, yl, yh, yh, yl], [zz] * 5, color="0.6", lw=0.8)
    for x_ in (xl, xh):
        for y_ in (yl, yh):
            ax.plot([x_, x_], [y_, y_], [0, ceil], color="0.6", lw=0.8)
    ax.set_xlim(xl, xh); ax.set_ylim(yl, yh); ax.set_zlim(0, ceil)
    ax.set_box_aspect((8, 11, 3.5))
    ax.view_init(elev=24, azim=-58)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D (파란 선 = 성공 비행 예시)")

    # ---- top ----
    ax = fig.add_subplot(2, 2, 2)
    ax.add_patch(plt.Rectangle((xl, yl), xh - xl, yh - yl, fill=False, ec="0.3", lw=2))
    lc = colored(ax, P[:, 0], P[:, 1], spd)
    for name, (X, Y, Z, yaw, st) in GATES.items():
        dy = 0.06 if name == "G2" else (-0.06 if name == "G3" else 0)
        ax.plot([X - OUTER / 2, X + OUTER / 2], [Y + dy, Y + dy], color="tab:orange", lw=6,
                solid_capstyle="butt")
        ax.plot([X - INNER / 2, X + INNER / 2], [Y + dy, Y + dy], color="white", lw=2)
        d = 1 if yaw == 90 else -1
        ax.annotate("", (X, Y + 0.7 * d), (X, Y - 0.3 * d),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
    ax.text(-2.9, -0.6, "G1 (Z 1.25)", ha="center", fontsize=10)
    ax.text(2.9, -0.6, "G4 (Z 1.25)", ha="center", fontsize=10)
    ax.text(1.0, 4.35, "G2 (Z 2.25, 위)\nG3 (Z 0.75, 아래)", ha="left", fontsize=10)
    ax.plot(LAUNCH[0], LAUNCH[1], "^", color="crimson", ms=12)
    ax.add_patch(plt.Rectangle((LAUNCH[0] - LAUNCH_LAT, -1.0 - LAUNCH_D[1]), 2 * LAUNCH_LAT,
                               LAUNCH_D[1] - LAUNCH_D[0], color="crimson", alpha=0.25))
    ax.annotate("이륙: G1 앞 2.0 m\n(G1 정면을 봄)", (LAUNCH[0], LAUNCH[1]), (-3.8, -3.75),
                color="crimson", fontsize=10, arrowprops=dict(arrowstyle="->", color="crimson"))
    ax.annotate("", (-2.0, -1.0), (-2.0, -3.0), arrowprops=dict(arrowstyle="<->", color="crimson", lw=1))
    ax.text(-1.85, -2.0, "2.0 m", color="crimson", fontsize=9)
    ax.text(xl + 0.1, yh - 0.35, "벽", color="0.3"); ax.text(xh - 0.4, yl + 0.1, "벽", color="0.3")
    ax.annotate("", (xl, yl - 0.35), (xh, yl - 0.35), arrowprops=dict(arrowstyle="<->"))
    ax.text(0, yl - 0.3, "8 m", ha="center", va="bottom")
    ax.annotate("", (xh + 0.35, yl), (xh + 0.35, yh), arrowprops=dict(arrowstyle="<->"))
    ax.text(xh + 0.45, 1.5, "11 m", va="center", rotation=90)
    ax.set_xlim(xl - 0.6, xh + 0.9); ax.set_ylim(yl - 0.7, yh + 0.3); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("위에서 본 모습 (화살표 = 통과 방향, 색 = 속도)")
    cb = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.02); cb.set_label("속도 (m/s)")

    # ---- side: Y horizontal, Z up (looking from +X toward -X... drawn with Y to the right) ----
    ax = fig.add_subplot(2, 2, 3)
    ax.add_patch(plt.Rectangle((yl, 0), yh - yl, ceil, fill=False, ec="0.3", lw=2))
    colored(ax, P[:, 1], P[:, 2], spd, lw=1.2)
    for name, (X, Y, Z, yaw, st) in GATES.items():
        off = -0.08 if name in ("G1",) else (0.08 if name == "G4" else 0)
        ax.plot([Y + off] * 2, [Z - OUTER / 2, Z + OUTER / 2], color="tab:orange", lw=6, solid_capstyle="butt")
        ax.plot([Y + off] * 2, [Z - INNER / 2, Z + INNER / 2], color="white", lw=2)
        if st:
            ax.add_patch(plt.Rectangle((Y + off - 0.12, 0), 0.24, STAND, color="0.45"))
    ax.text(-1.0, 2.15, "G1 / G4\n(Y -1, 겹쳐 보임)", ha="center", fontsize=10)
    ax.text(4.25, 2.25, "G2", fontsize=11, va="center"); ax.text(4.25, 0.75, "G3", fontsize=11, va="center")
    ax.text(-0.75, 0.2, "받침대 0.5 m", fontsize=9, color="0.3")
    ax.annotate("", (4.0, 3.0), (4.0, ceil), arrowprops=dict(arrowstyle="<->", color="crimson"))
    ax.text(3.6, 3.22, "0.5 m", color="crimson", fontsize=9, ha="right")
    ax.plot(LAUNCH[1], LAUNCH[2], "^", color="crimson", ms=12)
    ax.text(LAUNCH[1], 0.25, "이륙", color="crimson", ha="center")
    ax.text(yl + 0.1, ceil - 0.25, "천장 3.5 m", color="0.3")
    ax.set_xlim(yl - 0.3, yh + 0.3); ax.set_ylim(-0.2, ceil + 0.3); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.set_xlabel("Y (m)"); ax.set_ylabel("Z 높이 (m)")
    ax.set_title("옆에서 본 모습 (Y-Z)")

    # ---- front: X horizontal, Z up ----
    ax = fig.add_subplot(2, 2, 4)
    ax.add_patch(plt.Rectangle((xl, 0), xh - xl, ceil, fill=False, ec="0.3", lw=2))
    for name, (X, Y, Z, yaw, st) in GATES.items():
        o, i = OUTER / 2, INNER / 2
        ax.add_patch(plt.Rectangle((X - o, Z - o), OUTER, OUTER, color="tab:orange"))
        ax.add_patch(plt.Rectangle((X - i, Z - i), INNER, INNER, color="white"))
        ax.text(X, Z, f"{name}\n중심 {Z:.2f}", ha="center", va="center", fontsize=10)
        if st:
            ax.add_patch(plt.Rectangle((X - o, 0), 0.08, STAND, color="0.45"))
            ax.add_patch(plt.Rectangle((X + o - 0.08, 0), 0.08, STAND, color="0.45"))
    ax.annotate("", (-2.0 - 0.75 - 0.25, 0.5), (-2.0 - 0.75 - 0.25, 2.0), arrowprops=dict(arrowstyle="<->"))
    ax.text(-3.05, 1.25, "1.5", ha="right", va="center")
    ax.annotate("", (-1.4, 0.65), (-1.4, 1.85), arrowprops=dict(arrowstyle="<->", color="0.4"))
    ax.text(-1.33, 1.25, "1.2", va="center", color="0.4")
    ax.text(xl + 0.1, ceil - 0.25, "천장 3.5 m", color="0.3")
    ax.set_xlim(xl - 0.3, xh + 0.3); ax.set_ylim(-0.2, ceil + 0.3); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.set_xlabel("X (m)"); ax.set_ylabel("Z 높이 (m)")
    ax.set_title("정면에서 본 게이트 높이 (X-Z, 이륙 쪽에서 봄)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.axes[0].set_position([0.27, 0.50, 0.30, 0.42])
    fig.savefig(OUT, dpi=110)
    return OUT


if __name__ == "__main__":
    print(draw())
