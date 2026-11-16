"""Tile 4x zooms (with a fine pixel grid in full-frame coords) of given points of one photo."""
import sys
from PIL import Image, ImageDraw
src, out = sys.argv[1], sys.argv[2]
pts = [tuple(map(int, p.split(','))) for p in sys.argv[3:]]
im = Image.open(src).convert('RGB')
R, Z = 40, 4
tiles = []
for (cx, cy) in pts:
    t = im.crop((cx - R, cy - R, cx + R, cy + R)).resize((2 * R * Z, 2 * R * Z), Image.LANCZOS)
    d = ImageDraw.Draw(t)
    for k in range(-R, R + 1, 10):
        c = (255, 0, 0) if (cx + k) % 50 == 0 else (255, 170, 170)
        d.line([((k + R) * Z, 0), ((k + R) * Z, 2 * R * Z)], fill=c)
        c = (0, 0, 255) if (cy + k) % 50 == 0 else (170, 170, 255)
        d.line([(0, (k + R) * Z), (2 * R * Z, (k + R) * Z)], fill=c)
    for k in range(-R, R + 1, 20):
        d.text(((k + R) * Z + 2, 2), str(cx + k), fill=(255, 255, 0))
        d.text((2, (k + R) * Z + 2), str(cy + k), fill=(0, 255, 255))
    d.text((5, 2 * R * Z - 14), f"({cx},{cy})", fill=(255, 255, 255))
    tiles.append(t)
W = 2 * R * Z
sheet = Image.new('RGB', (W * min(3, len(tiles)), W * ((len(tiles) + 2) // 3)), 'black')
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 3) * W, (i // 3) * W))
sheet.save(out, quality=90)
