"""INAIR racer geometry from the team's hand measurements (2026-11-13).

Origin: centre of the motor diagonals, at prop-plane height (4 cm above the floor
when resting).  Frame used in this file: x forward, y RIGHT, z UP, cm -- the way
the team measured.  The simulator uses NED body axes (z down): z_sim = -z here.

Measured (team):                    Shape only (from photos, not measured):
  motors   x +-7.75, y +-10.25        shell profile between the measured extremes
  front    x +13.5, z +1.0            antenna, RX whiskers
  rear     x -15.0, z -1.0
  top      x -1.0,  z +6.2
  camera   x +12.0, y +0.2 (?), z +1.5, tilt 40 deg up -> being raised to 50
  antennas inside the shell envelope (VTX tip 0.5 cm inside the tail; RX clear of props)
  stands   x +-8, y -4..+4, z -4 (floor)
  shell    widest 7.7 cm
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).with_suffix(".png")

# ---------------- measured ----------------
MX, MY = 15.66 / 2, 19.75 / 2   # CAD (Part Studio 1-2.step) motor centre holes; tape was 15.5 x 20.5
MOTORS = {"앞오른쪽": (MX, MY), "앞왼쪽": (MX, -MY), "뒤오른쪽": (-MX, MY), "뒤왼쪽": (-MX, -MY)}
PROP_R = 5 * 2.54 / 2            # 5045 prop (team, 2026-11-13): 5.0 in dia -> 6.35 cm radius
PROP_PITCH = 4.5 * 2.54          # 11.43 cm
INFO = (
    "사양 (2026-11-15, 형상은 CAD 기준)\n"
    "기체   전비 0.95 kg (배선 추가로 소폭 증가 가능)\n"
    "       무게중심 (+0.0, +0.1, +1.2) cm (배터리 장착 상태, CAD 모터 위치로 재계산)\n"
    "카메라 렌즈 x +11.7, y +0.1 (기체 오른쪽), z +1.2, 틸트 50° 위 (CAD·현재 40°, 각도만 변경 예정)\n"
    "배치(CAD 중심) 배터리 (-3.0, +0.1, +2.5) 8.1x5.4x5.6 / Jetson (+5.5, +0.1, +2.2)\n"
    "       FC (+0.1, +0.1, -0.4) / ESC (-0.2, +0.1, -1.3) / XT60 (-6.4, +0.1, -1.0)\n"
    "배터리 LiPo 6S 1800 mAh (공칭 22.2 V, 만충 25.2 V), 1개만 사용\n"
    "모터   T-Motor VELOX V2208 V2 x4, 12N14P (14극), 37 g\n"
    "       Ø27.6 x 31.95 mm, 4 mm 티타늄 축, 16x16 마운트, 4-6S\n"
    "       1750 KV, 최대 약 1.46 kg/모터 (제조사, T5143S 6S 100%)\n"
    "프롭   5045 2엽: 직경 127 mm, 피치 114.3 mm\n"
    "       회전(위에서): 앞오른쪽·뒤왼쪽 CCW, 앞왼쪽·뒤오른쪽 CW = props in\n"
    "추력대중량비  약 5.1-6.2 (추정, 중앙 ~5.6) / 호버 스로틀 약 33-38% (추정)\n"
    "       = 4 x 1.2-1.46 kg / 0.95 kg. 실측(호버·펀치 로그)으로 확정 예정\n"
    "FC     SpeedyBee F405 V4 (STM32F405)\n"
    "ESC    SpeedyBee BLS 55A 4-in-1 (8bit, 3-6S, 55A x4, BLHeli_S)\n"
    "       RPM 텔레메트리(양방향 DShot)는 Bluejay 펌웨어로 교체 필요\n"
    "IMU    ICM-42688-P (FC 내장): 자이로 최대 ±2000 dps, 가속도 최대 ±16 g\n"
    "       센서 노이즈 2.8 mdps/√Hz, 70 ug/√Hz (실제는 진동이 지배)\n"
    "컴퓨터 Jetson Orin Nano"
)
# Spin seen from above (team, 2026-11-13): props in.
SPIN = {"앞오른쪽": "CCW", "뒤왼쪽": "CCW", "앞왼쪽": "CW", "뒤오른쪽": "CW"}
# Thrust: T-Motor's V2208 V2 1750KV table (6S, T5143S tri-blade 5.1x4.3) 1460 g at 100%;
# our 2-blade 5045 is taken as 1.2-1.46 kg (2-blades lift less than tri-blades).
TWR_RANGE = (4 * 1.20 / 0.95, 4 * 1.46 / 0.95)


def blade_xy(cx, cy, ang, r0=1.4, r1=PROP_R, chord=1.7):
    """Outline (in plot coords: horizontal = y, vertical = x) of one 2-blade prop."""
    out = []
    for a in (ang, ang + np.pi):
        r = np.linspace(r0, r1, 30)
        w = chord / 2 * np.sin(np.pi * (r - r0) / (r1 - r0)) ** 0.6 + 0.25
        u, v = np.cos(a), np.sin(a)
        side1 = np.c_[cy + r * u - w * v, cx + r * v + w * u]
        side2 = np.c_[cy + r * u + w * v, cx + r * v - w * u]
        out.append(np.r_[side1, side2[::-1]])
    return out
FRONT = (13.56, 0.0, -0.54)      # CAD
REAR = (-16.01, 0.0, 0.43)       # CAD (tape -15.0)
TOP = (2.59, 0.0, 5.71)          # CAD (tape x -1.0, z +6.2)
CAM = (11.73, 0.12, 1.16)        # CAD lens front face; tape (12.0, 0.2, 1.5). y + = drone right
CAM_TILT = 50.0           # CAD 39.8 = measured 40; team raises the tilt to 50, same position (2026-11-14)
FLOOR = -4.0
STAND_X, STAND_HALF_Y = 8.0, 4.0
SHELL_HALF_W = 7.96 / 2          # CAD body width
MASS = 0.95                      # kg, all-up (team, 2026-11-13)
# Centre of mass from 7 hanging photos (com/solve_com.py): hub nut assumed 1.0 cm
# above the prop plane; the CoM is 0.26 cm above the nut.
COM = (0.01, 0.05, 1.15)
COM_LABEL = "무게중심 (+0.0, +0.1, +1.2)"

# ---------------- shell: CAD cross-sections (body, lid, mounts; frame and legs excluded) ----
XS = np.array([13.56, 12.5, 11.0, 9.0, 7.0, 4.0, 1.0, -1.0, -4.0, -7.0, -10.0, -13.0, -15.0, -16.01])
BOT = np.array([-0.54, -1.95, -2.73, -2.32, -1.73, -1.73, -1.73, -1.73, -1.73, -1.73, -0.52, -0.24, -0.12, 0.40])
TOPZ = np.array([-0.54, 0.80, 2.08, 3.41, 4.14, 5.23, 5.61, 5.69, 5.10, 4.02, 2.93, 1.85, 1.13, 0.40])
HALFW = np.array([0.3, 1.99, 2.95, 3.30, 3.50, 3.80, 3.91, 3.71, 3.42, 3.12, 2.77, 2.10, 1.22, 0.0])
# component boxes from CAD: centre (x, y, z), size (fore-aft, lateral, height), cm
BOXES = {"배터리": ((-3.04, 0.11, 2.45), (8.05, 5.40, 5.58), "tab:olive"),
         "Jetson": ((5.52, 0.09, 2.18), (8.46, 5.32, 5.93), "tab:cyan"),
         "FC": ((0.05, 0.11, -0.44), (4.16, 3.94, 0.97), "tab:pink"),
         "ESC": ((-0.17, 0.11, -1.33), (4.16, 4.40, 0.81), "tab:gray")}
CAD_GAPS = {"앞오른쪽": 0.3, "앞왼쪽": 1.4, "뒤오른쪽": 2.9, "뒤왼쪽": 6.2}   # mm, prop disc to nearest part


def shell_at(x):
    xs = XS[::-1]
    return (np.interp(x, xs, BOT[::-1]), np.interp(x, xs, TOPZ[::-1]),
            np.interp(x, xs, HALFW[::-1]))


def shell_surface(nx=60, nt=48):
    x = np.linspace(FRONT[0], REAR[0], nx)
    t = np.linspace(0, 2 * np.pi, nt)
    b, tp, w = shell_at(x)
    zc, hz = (b + tp) / 2, (tp - b) / 2
    c, s = np.cos(t), np.sin(t)
    sup = lambda u: np.sign(u) * np.abs(u) ** 0.5    # superellipse, boxy section  # noqa: E731
    X = np.repeat(x[:, None], nt, 1)
    Y = w[:, None] * sup(c)[None]
    Z = zc[:, None] + hz[:, None] * sup(s)[None]
    return X, Y, Z


def prop_clearance():
    """Smallest gap (cm) between a prop disc edge and the shell side at the prop plane."""
    x = np.linspace(REAR[0], FRONT[0], 2001)
    b, tp, w = shell_at(x)
    inplane = (b <= 0) & (tp >= 0)
    gaps = []
    for mx, my in MOTORS.values():
        d = np.hypot(x - mx, np.abs(my) - w)[inplane]
        gaps.append(d.min() - PROP_R)
    return min(gaps)


def draw():
    X, Y, Z = shell_surface()
    fig = plt.figure(figsize=(17, 12))
    fig.suptitle(f"INAIR 레이서 형상 복원, 전비 {MASS} kg (원점 = 모터 대각선 교점, 프롭 평면 / x 전방, y 오른쪽, z 위, cm)",
                 fontsize=14)
    fig.text(0.015, 0.935, INFO, fontsize=9.5, va="top", ha="left", family="AppleGothic",
             linespacing=1.5, bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f2", ec="0.6"))

    # ---- 3D ----
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot_surface(X, Y, Z, color="#e8e8e8", alpha=0.55, linewidth=0, shade=True)
    for (mx, my) in MOTORS.values():
        ax.plot([0, mx], [0, my], [-1.2, -1.2], color="k", lw=4)
        t = np.linspace(0, 2 * np.pi, 80)
        ax.plot(mx + PROP_R * np.cos(t), my + PROP_R * np.sin(t), 0 * t, color="tab:blue", lw=1)
        ax.plot([mx, mx], [my, my], [FLOOR + 0.3, 0], color="0.3", lw=6)
    for sx in (STAND_X, -STAND_X):
        ax.plot([sx, sx], [-STAND_HALF_Y, STAND_HALF_Y], [FLOOR, FLOOR], color="tab:orange", lw=5)
        ax.plot([sx, sx], [0, 0], [FLOOR, -1.2], color="tab:orange", lw=3)
    d = np.array([np.cos(np.radians(CAM_TILT)), 0, np.sin(np.radians(CAM_TILT))]) * 8
    ax.quiver(*CAM, *d, color="tab:red", lw=2)
    for p, c in ((FRONT, "tab:green"), (REAR, "tab:purple"), (TOP, "tab:brown"), (CAM, "tab:red")):
        ax.scatter(*p, color=c, s=40)
    ax.scatter(0, 0, 0, color="k", s=30, marker="x")
    ax.scatter(*COM, color="magenta", s=120, marker="*", depthshade=False)
    ax.text(COM[0], COM[1], COM[2] + 3, "CoM", color="magenta")
    xx, yy = np.meshgrid([-18, 18], [-18, 18])
    ax.plot_surface(xx, yy, np.full_like(xx, FLOOR, float), color="#d9c7a7", alpha=0.25)
    ax.set_xlim(-18, 18); ax.set_ylim(-18, 18); ax.set_zlim(-6, 12)
    ax.set_box_aspect((36, 36, 18))
    ax.view_init(elev=22, azim=-130)
    ax.set_xlabel("x 전방"); ax.set_ylabel("y 오른쪽"); ax.set_zlabel("z 위")
    ax.set_title("3D (셸 = CAD 단면 근사)")

    # ---- top: x up the page, y to the right ----
    ax = fig.add_subplot(2, 2, 2)
    xs = np.linspace(REAR[0], FRONT[0], 300)
    _, _, w = shell_at(xs)
    ax.fill_betweenx(xs, -w, w, color="#e8e8e8", ec="0.4")
    for name, (mx, my) in MOTORS.items():
        ax.plot([0, my], [0, mx], color="k", lw=4)
        ax.add_patch(plt.Circle((my, mx), PROP_R, fill=False, ec="tab:blue", ls="--"))
        for poly in blade_xy(mx, my, np.radians(35 if mx * my > 0 else -35)):
            ax.add_patch(plt.Polygon(poly, color="tab:blue", alpha=0.35))
        ax.add_patch(plt.Circle((my, mx), 1.4, color="0.3"))
        ccw = SPIN[name] == "CCW"
        a0, a1 = (np.radians(200), np.radians(340)) if mx > 0 else (np.radians(20), np.radians(160))
        if not ccw:
            a0, a1 = a1, a0
        r = PROP_R + 0.9
        ax.annotate("", (my + r * np.cos(a1), mx + r * np.sin(a1)), (my + r * np.cos(a0), mx + r * np.sin(a0)),
                    arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.8,
                                    connectionstyle=f"arc3,rad={0.55 if ccw else -0.55}"))
        ax.text(my, mx + (-2.6 if mx > 0 else 2.0), SPIN[name], color="crimson", fontsize=10,
                ha="center",)
        ax.annotate(f"{name}\n(x {mx:+.2f}, y {my:+.2f})", (my, mx), (my * 1.55, mx * 1.9),
                    ha="center", fontsize=9, arrowprops=dict(arrowstyle="-", color="0.5"))
    for sx in (STAND_X, -STAND_X):
        ax.plot([-STAND_HALF_Y, STAND_HALF_Y], [sx, sx], color="tab:orange", lw=5, alpha=0.7)
    ax.annotate("받침대 x ±8, 폭 8 (y ±4)", (STAND_HALF_Y, -STAND_X), (9, -16), fontsize=9,
                color="tab:orange", arrowprops=dict(arrowstyle="->", color="tab:orange"))
    ax.annotate("", (-MY, -19.5), (MY, -19.5), arrowprops=dict(arrowstyle="<->"))
    ax.text(0, -19.2, "좌우 19.75 (CAD, 실측 20.5)", ha="center", va="bottom")
    ax.annotate("", (-19, MX), (-19, -MX), arrowprops=dict(arrowstyle="<->"))
    ax.text(-18.7, 0, "앞뒤\n15.66\n(실측\n15.5)", va="center")
    ax.plot([-MY, MY], [-MX, MX], color="0.6", ls=":")
    ax.text(3, 0.8, "대각 25.2 (실측 25.5)", fontsize=8, color="0.4", rotation=38)
    ax.annotate("", (SHELL_HALF_W, 2), (-SHELL_HALF_W, 2), arrowprops=dict(arrowstyle="<->", color="0.2"))
    ax.text(0, 2.4, "셸 폭 8.0", ha="center", fontsize=9)
    ax.plot(CAM[1], CAM[0], "o", color="tab:red")
    ax.plot(0, FRONT[0], "o", color="tab:green"); ax.plot(0, REAR[0], "o", color="tab:purple")
    ax.text(0.8, FRONT[0] + 0.6, "가장 앞 +13.6", color="tab:green", fontsize=9)
    ax.text(0.8, REAR[0] - 1.6, "가장 뒤 -16.0", color="tab:purple", fontsize=9)
    for nm, ((cx, cy, cz), (lx, ly, lz), col) in BOXES.items():
        ax.add_patch(plt.Rectangle((cy - ly / 2, cx - lx / 2), ly, lx, fill=False, ec=col, lw=1.6))
    ax.plot(0, 0, "kx")
    ax.plot(COM[1], COM[0], "*", color="magenta", ms=16)
    ax.annotate(COM_LABEL, (COM[1], COM[0]), (-20.5, -16.8), color="magenta", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="magenta"))
    ax.set_xlim(-21, 21); ax.set_ylim(-21, 21); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("y (오른쪽 +)"); ax.set_ylabel("x (전방 +)")
    ax.set_title("위에서 본 모습 (전방 = 위쪽, 프롭 5045: 원 = 회전 영역 Ø127 mm)")
    ax.text(-20.5, 19.5, "빨간 화살표 = 프롭 회전 방향 (위에서 봄), props in", fontsize=8.5,
            color="crimson", va="top")

    # ---- side: x to the LEFT is forward (as the photos), z up ----
    ax = fig.add_subplot(2, 2, 3)
    b, tp, _ = shell_at(xs)
    ax.fill_between(-xs, b, tp, color="#e8e8e8", ec="0.4")
    ax.axhline(FLOOR, color="#b89b6a", lw=3)
    ax.axhline(0, color="tab:blue", ls="--", lw=1)
    ax.text(17.5, 0.3, "프롭 평면 z=0 (5045, Ø12.7)", color="tab:blue", fontsize=9, ha="right")
    for mx in (MX, -MX):
        ax.plot([-mx - PROP_R, -mx + PROP_R], [0, 0], color="tab:blue", lw=3)
        ax.add_patch(plt.Rectangle((-mx - 1.4, FLOOR + 0.6), 2.8, 3.4, color="0.3"))
    for sx in (STAND_X, -STAND_X):
        ax.add_patch(plt.Rectangle((-sx - 0.6, FLOOR), 1.2, 2.8, color="tab:orange", alpha=0.8))
    ax.plot([-MX - 1, MX + 1], [-1.3, -1.3], color="k", lw=4)
    d = np.array([np.cos(np.radians(CAM_TILT)), np.sin(np.radians(CAM_TILT))]) * 6
    ax.annotate("", (-CAM[0] - d[0], CAM[2] + d[1]), (-CAM[0], CAM[2]),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
    ax.plot(-CAM[0], CAM[2], "o", color="tab:red")
    ax.text(-CAM[0] - 6.5, CAM[2] + 5.5, "카메라 렌즈 (x +11.7, z +1.2)\n틸트 50° 위 (CAD·현재 40°)", color="tab:red", fontsize=9)
    for p, c, lab, off in ((FRONT, "tab:green", "가장 앞 (+13.6, z -0.5)", (-20.5, -2.6)),
                           (REAR, "tab:purple", "가장 뒤 (-16.0, z +0.4)", (11, -2.8)),
                           (TOP, "tab:brown", "가장 높은 (x +2.6, z +5.7)", (-3, 8.3))):
        ax.plot(-p[0], p[2], "o", color=c)
        ax.annotate(lab, (-p[0], p[2]), off, color=c, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=c))
    ax.text(-8, FLOOR - 1.3, "받침대 x +8 / -8, z -4 (바닥)", color="tab:orange", fontsize=9, ha="center")
    ax.annotate("", (16.8, FLOOR), (16.8, 0), arrowprops=dict(arrowstyle="<->"))
    ax.text(17.1, -2, "4", va="center")
    for nm, ((cx, cy, cz), (lx, ly, lz), col) in BOXES.items():
        ax.add_patch(plt.Rectangle((-cx - lx / 2, cz - lz / 2), lx, lz, fill=False, ec=col, lw=1.6))
        if nm in ("배터리", "Jetson"):
            ax.text(-cx, cz + lz / 2 + 0.25, nm, color=col, fontsize=9, ha="center")
    ax.plot(0, 0, "kx")
    ax.plot(-COM[0], COM[2], "*", color="magenta", ms=16)
    ax.annotate(COM_LABEL, (-COM[0], COM[2]), (3.5, 3.2), color="magenta", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="magenta"))
    ax.set_xlim(-21, 21); ax.set_ylim(-7, 10); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xticks(range(-20, 21, 5)); ax.set_xticklabels([str(-v) for v in range(-20, 21, 5)])
    ax.set_xlabel("x (전방 +, 왼쪽이 앞)"); ax.set_ylabel("z (위 +)")
    ax.set_title("옆에서 본 모습 (오른쪽 면)")

    # ---- front: looking at the nose, drone's right is on the viewer's LEFT ----
    ax = fig.add_subplot(2, 2, 4)
    x0 = 1.0
    b, tp, w = shell_at(x0)
    t = np.linspace(0, 2 * np.pi, 100)
    sup = lambda u: np.sign(u) * np.abs(u) ** 0.5  # noqa: E731
    ax.fill(-w * sup(np.cos(t)), (b + tp) / 2 + (tp - b) / 2 * sup(np.sin(t)), color="#e8e8e8", ec="0.4")
    ax.axhline(FLOOR, color="#b89b6a", lw=3)
    for my in (MY, -MY):
        ax.plot([-my - PROP_R, -my + PROP_R], [0, 0], color="tab:blue", lw=3)
        ax.add_patch(plt.Rectangle((-my - 1.4, FLOOR + 0.6), 2.8, 3.4, color="0.3"))
    ax.plot([-MY, MY], [-1.3, -1.3], color="k", lw=4)
    ax.add_patch(plt.Rectangle((-STAND_HALF_Y, FLOOR), 2 * STAND_HALF_Y, 1.0, color="tab:orange", alpha=0.8))
    ax.plot(-CAM[1], CAM[2], "o", color="tab:red")
    ax.plot(-COM[1], COM[2], "*", color="magenta", ms=16)
    ax.text(-COM[1] - 0.6, COM[2] + 0.5, "CoM", color="magenta", fontsize=9, ha="right")
    ax.text(-CAM[1] + 0.8, CAM[2] + 0.8, "카메라 y +0.1 (기체 오른쪽)", color="tab:red", fontsize=9)
    ax.text(0, 7.6, "프롭 끝↔셸 (CAD): 앞오른쪽 0.3, 앞왼쪽 1.4, 뒤오른쪽 2.9, 뒤왼쪽 6.2 mm",
            color="crimson", fontsize=9, ha="center")
    ax.text(-MY, -6.3, "기체 오른쪽", ha="center"); ax.text(MY, -6.3, "기체 왼쪽", ha="center")
    ax.set_xlim(-19, 19); ax.set_ylim(-7, 9); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("y (정면에서 볼 때 왼쪽 = 기체 오른쪽)"); ax.set_ylabel("z (위 +)")
    ax.set_title("정면에서 본 모습 (셸 단면은 x=+1 기준)")

    fig.tight_layout()
    fig.axes[0].set_position([0.385, 0.42, 0.22, 0.36])   # 3D panel clear of the spec box
    fig.savefig(OUT, dpi=110)
    return OUT


if __name__ == "__main__":
    print(draw())
    print(f"prop-to-shell clearance (shape model): {prop_clearance():.2f} cm")
    print(f"same, widest section straight beside the motor: {MY - PROP_R - SHELL_HALF_W:.2f} cm")
