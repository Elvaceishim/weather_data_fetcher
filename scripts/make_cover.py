from PIL import Image, ImageDraw, ImageFont
import os

RESULTS = "results"
IMG1 = os.path.join(RESULTS, "temps.png")
IMG2 = os.path.join(RESULTS, "precip.png")
OUT  = os.path.join(RESULTS, "cover.png")
ASSETS_DIR = "assets"
ASSET_OUT = os.path.join(ASSETS_DIR, "cover.png")

def fail(msg):
    print(f"❌ {msg}")
    raise SystemExit(1)

# Check inputs
if not os.path.exists(IMG1):
    fail(f"Missing {IMG1}. Run `make viz` first.")
if not os.path.exists(IMG2):
    fail(f"Missing {IMG2}. Run `make viz` first.")

# Load and normalize
img1 = Image.open(IMG1).convert("RGB")
img2 = Image.open(IMG2).convert("RGB")

# Match widths
w = min(img1.width, img2.width)
def resize_to_width(im, target_w):
    new_h = int(im.height * target_w / im.width)
    return im.resize((target_w, new_h))

img1 = resize_to_width(img1, w)
img2 = resize_to_width(img2, w)

# Layout params
pad = 16
title_h = 48
H = img1.height + img2.height + title_h + pad * 4
W = w + pad * 2

# Create canvas (✅ correct)
canvas = Image.new("RGB", (W, H), "white")

# Paste charts
y = pad
canvas.paste(img1, (pad, y)); y += img1.height + pad
canvas.paste(img2, (pad, y)); y += img2.height + pad

# Title text
draw = ImageDraw.Draw(canvas)
title = "Weather Data Fetcher — Automated Pipeline"
try:
    font = ImageFont.load_default()
except Exception:
    font = None

# Center the title horizontally
tw, th = draw.textbbox((0,0), title, font=font)[2:]
tx = (W - tw) // 2
ty = y
draw.text((tx, ty), title, fill="black", font=font)

os.makedirs(RESULTS, exist_ok=True)
canvas.save(OUT, optimize=True)
if ASSETS_DIR:
    os.makedirs(ASSETS_DIR, exist_ok=True)
    canvas.save(ASSET_OUT, optimize=True)
print(f"✅ Created {OUT}")
