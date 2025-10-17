import os
from PIL import Image, ImageDraw, ImageFont

RESULTS = "results"
IMG1 = os.path.join(RESULTS, "temps.png")
IMG2 = os.path.join(RESULTS, "precip.png")
OUT  = os.path.join(RESULTS, "cover.png")

def fail(msg):
    print(f"❌ {msg}")
    raise SystemExit(1)

if not os.path.exists(IMG1):
    fail(f"Missing {IMG1}. Run `make viz` first.")
if not os.path.exists(IMG2):
    fail(f"Missing {IMG2}. Run `make viz` first.")

img1 = Image.open(IMG1).convert("RGB")
img2 = Image.open(IMG2).convert("RGB")

# same width
w = min(img1.width, img2.width)
h1 = int(img1.height * w / img1.width)
h2 = int(img2.height * w / img2.width)
img1 = img1.resize((w, h1))
img2 = img2.resize((w, h2))

# canvas
pad = 16
title_h = 64
H = h1 + h2 + title_h + pad*3
canvas = Image.new("RGB", (w + pad * 2, H), "white")

# paste charts
y = pad
canvas.paste(img1, (pad, y)); y += h1 + pad
canvas.paste(img2, (pad, y)); y += h2 + pad

# title
draw = ImageDraw.Draw(canvas)
title = "Weather Data Fetcher — Automated Pipeline"
try:
    # Use a default font; system fonts vary
    font = ImageFont.load_default()
except:
    font = None
draw.text((pad, y), title, fill="black", font=font)

os.makedirs(RESULTS, exist_ok=True)
canvas.save(OUT, optimize=True)
print(f"✅ Created {OUT}")
