"""Tile PNGs into a labelled contact sheet.
Usage: python montage.py <out.png> <cols> <tile_w> <img1> [<img2> ...]   (label = file name, or 'path=label')"""
import sys

from PIL import Image, ImageDraw, ImageFont


def main():
    out, cols, tw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    items = []
    for s in sys.argv[4:]:
        p, _, lab = s.partition("=")
        items.append((p, lab or p.rsplit("/", 1)[-1]))
    ims = []
    for p, lab in items:
        im = Image.open(p).convert("RGB")
        th = round(im.height * tw / im.width)
        ims.append((im.resize((tw, th), Image.LANCZOS), lab))
    th = max(i.height for i, _ in ims)
    lh = 18
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + lh)), "white")
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for k, (im, lab) in enumerate(ims):
        x, y = (k % cols) * tw, (k // cols) * (th + lh)
        d.text((x + 3, y + 2), lab, fill=(180, 0, 0), font=font)
        sheet.paste(im, (x, y + lh))
    sheet.save(out)
    print(out, sheet.size)


if __name__ == "__main__":
    main()
