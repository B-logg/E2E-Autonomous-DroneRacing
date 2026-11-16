"""Check image: per photo, the clicked hubs (green), the PnP reprojection (cyan),
the string extended downward (yellow) and the solved CoM projected (magenta)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

import solve_com as s

src_dir, out = Path(sys.argv[1]), sys.argv[2]
d = json.loads(Path(__file__).with_name("measurements.json").read_text())
com = np.array(list(map(float, sys.argv[3].split(",")))) * np.array([1, -1, 1])  # to FLU
tiles = []
for name, m in d.items():
    obs = [((*s.MOTOR[k], m["hub_z"]), uv) for k, uv in m["motors"].items()]
    R, t, _ = s.pose(obs)
    rv = Rotation.from_matrix(R).as_rotvec()
    im = Image.open(src_dir / f"{name}.jpg").convert("RGB")
    dr = ImageDraw.Draw(im)
    (x1, y1), (x2, y2) = m["string"]
    k = (x2 - x1) / (y2 - y1)
    dr.line([(x1 + k * (0 - y1), 0), (x1 + k * (1600 - y1), 1600)], fill=(255, 230, 0), width=2)
    for (X, uv) in obs:
        p = s.project(rv, t, np.array([X]))[0]
        dr.ellipse([uv[0] - 9, uv[1] - 9, uv[0] + 9, uv[1] + 9], outline=(0, 255, 0), width=3)
        dr.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(0, 220, 255))
    c = s.project(rv, t, com[None])[0]
    dr.regular_polygon((c[0], c[1], 16), 5, fill=(255, 0, 255))
    dr.text((10, 10), f"{name}  hang {m['hang']}", fill=(255, 255, 0))
    tiles.append(im.crop((150, 200, 1050, 1400)).resize((450, 600)))
sheet = Image.new("RGB", (450 * 4, 600 * 2), "black")
for i, tl in enumerate(tiles):
    sheet.paste(tl, ((i % 4) * 450, (i // 4) * 600))
sheet.save(out, quality=90)
print(out)
