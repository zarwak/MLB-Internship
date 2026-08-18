\"""
Generates the 15+ sample images used for today's OCR practice.

WHY synthetic instead of real photos: OCR accuracy depends heavily on how
the text got onto the image in the first place (font, layout, lighting,
noise). Rendering the text myself with PIL means I *know* the ground-truth
text for every image, so when EasyOCR gets something wrong later I can
actually tell it's wrong instead of guessing. It's the same reason Day21's
cv_image_studio also shipped a make_sample_images.py instead of relying on
real photos.

Five categories, 3 images each = 15 images, each with a different font,
layout and simulated lighting/noise to keep the OCR comparison honest:
  1. printed_doc   - clean paragraph, standard sans-serif font
  2. receipt       - narrow itemized layout, monospace font
  3. signboard      - big bold text, high contrast, slight rotation
  4. book_page      - serif font, justified paragraph, aged-paper tint
  5. handwritten    - script font + per-character jitter (a *simulation*
                       of handwriting - see README for why this is not
                       the same as real handwriting for OCR purposes)
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT_DIR = Path(__file__).parent
FONT_DIR = Path("C:/Windows/Fonts")
random.seed(42)


def add_noise(img: Image.Image, amount: int) -> Image.Image:
    """Sprinkle random per-pixel brightness noise - simulates a phone photo
    taken under bad lighting instead of a clean digital render."""
    import numpy as np

    arr = np.array(img).astype(int)
    noise = np.random.randint(-amount, amount + 1, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(arr)


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """factor < 1 = darker (underlit), factor > 1 = brighter (washed out)."""
    from PIL import ImageEnhance

    return ImageEnhance.Brightness(img).enhance(factor)


# ---------------------------------------------------------------- printed_doc
PRINTED_TEXT = [
    "QUARTERLY PROGRESS REPORT",
    "",
    "Project: Optical Character Recognition Pipeline",
    "Prepared by: ML Internship Program, Day 22",
    "",
    "This document summarizes the OCR evaluation performed this week.",
    "Multiple engines were benchmarked for accuracy, speed, and ease",
    "of integration into a document processing pipeline. Preprocessing",
    "steps such as grayscale conversion and contrast enhancement were",
    "found to noticeably improve recognition on low-quality scans.",
]


def make_printed_doc(name: str, size: int, brightness: float):
    img = Image.new("RGB", (900, 650), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT_DIR / "arial.ttf"), size)
    y = 40
    for line in PRINTED_TEXT:
        draw.text((50, y), line, fill="black", font=font)
        y += size + 12
    img = adjust_brightness(img, brightness)
    img.save(OUT_DIR / name)


# ------------------------------------------------------------------- receipt
RECEIPT_LINES = [
    "        FRESH MART GROCERY",
    "     123 Market Street, Lahore",
    "------------------------------",
    "Milk 1L            x2   360.00",
    "Bread Loaf         x1   180.00",
    "Eggs (dozen)       x1   320.00",
    "Rice Basmati 5kg   x1  1450.00",
    "Cooking Oil 1L     x1   650.00",
    "------------------------------",
    "SUBTOTAL               2960.00",
    "TAX (5%)                148.00",
    "TOTAL                  3108.00",
    "------------------------------",
    "   THANK YOU FOR SHOPPING",
]


def make_receipt(name: str, size: int, brightness: float):
    img = Image.new("RGB", (500, 620), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT_DIR / "cour.ttf"), size)
    y = 25
    for line in RECEIPT_LINES:
        draw.text((25, y), line, fill="black", font=font)
        y += size + 8
    img = adjust_brightness(img, brightness)
    img.save(OUT_DIR / name)


# ---------------------------------------------------------------- signboard
def make_signboard(name: str, text: str, size: int, angle: float, bg, fg):
    font = ImageFont.truetype(str(FONT_DIR / "arialbd.ttf"), size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 60
    canvas_w, canvas_h = w + 2 * pad, h + 2 * pad
    img = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad - bbox[0], pad - bbox[1]), text, fill=fg, font=font)
    img = img.rotate(angle, expand=True, fillcolor=bg)
    img.save(OUT_DIR / name)


# ---------------------------------------------------------------- book_page
BOOK_PARAGRAPH = [
    "Chapter 7: The Nature of Perception",
    "",
    "It is a curious fact that the eye, for all its precision, is",
    "constantly being fooled by the mind's eagerness to interpret",
    "what it sees. A pattern of light and shadow becomes a face; a",
    "cluster of dots becomes a constellation. Recognition, in this",
    "sense, is not passive reception but active construction -- the",
    "brain reaching out to meet the world halfway.",
]


def make_book_page(name: str, size: int, tint):
    img = Image.new("RGB", (750, 500), tint)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT_DIR / "georgia.ttf"), size)
    y = 40
    for line in BOOK_PARAGRAPH:
        draw.text((55, y), line, fill=(40, 30, 20), font=font)
        y += size + 10
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.save(OUT_DIR / name)


# --------------------------------------------------------------- handwritten
HANDWRITTEN_NOTES = [
    "Buy milk, eggs and bread",
    "Meeting moved to 4pm tomorrow",
    "Remember to submit the report",
]


def make_handwritten(name: str, text: str, size: int):
    """Simulated handwriting: script font + small random jitter per
    character so it isn't perfectly straight like real print."""
    font = ImageFont.truetype(str(FONT_DIR / "segoescb.ttf"), size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    # Per-glyph ink-bbox widths sum to more than the kerned full-string
    # width for this script font, so measure by actually walking the
    # advance loop once (without drawing) to get the real width needed.
    x = 40
    for ch in text:
        bbox = probe.textbbox((0, 0), ch, font=font)
        x += (bbox[2] - bbox[0]) + 2
    canvas_w, canvas_h = x + 40, size + 100

    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(img)
    x, y = 40, 40
    for ch in text:
        jitter_y = random.randint(-4, 4)
        draw.text((x, y + jitter_y), ch, fill="navy", font=font)
        bbox = draw.textbbox((0, 0), ch, font=font)
        x += (bbox[2] - bbox[0]) + 2
    img.save(OUT_DIR / name)


def main():
    # 1. printed_doc x3 - varying font size and lighting
    make_printed_doc("printed_doc_normal.png", size=24, brightness=1.0)
    make_printed_doc("printed_doc_small_text.png", size=15, brightness=1.0)
    make_printed_doc("printed_doc_dim_lighting.png", size=24, brightness=0.55)

    # 2. receipt x3
    make_receipt("receipt_normal.png", size=18, brightness=1.0)
    make_receipt("receipt_faded.png", size=18, brightness=1.5)
    make_receipt("receipt_small.png", size=13, brightness=1.0)

    # 3. signboard x3
    make_signboard("signboard_stop.png", "STOP - CONSTRUCTION AHEAD", 60, 0, "yellow", "black")
    make_signboard("signboard_open.png", "OPEN 24 HOURS", 70, -6, "darkred", "white")
    make_signboard("signboard_exit.png", "EXIT >>", 90, 3, "green", "white")

    # 4. book_page x3
    make_book_page("book_page_clean.png", size=20, tint=(250, 246, 235))
    make_book_page("book_page_aged.png", size=20, tint=(224, 210, 178))
    make_book_page("book_page_large_print.png", size=28, tint=(250, 246, 235))

    # 5. handwritten x3
    make_handwritten("handwritten_note1.png", HANDWRITTEN_NOTES[0], size=42)
    make_handwritten("handwritten_note2.png", HANDWRITTEN_NOTES[1], size=36)
    make_handwritten("handwritten_note3.png", HANDWRITTEN_NOTES[2], size=48)

    # A couple of extra "hard" cases for the preprocessing comparison step
    noisy = adjust_brightness(Image.open(OUT_DIR / "printed_doc_normal.png"), 0.6)
    noisy = add_noise(noisy, 25)
    noisy.save(OUT_DIR / "printed_doc_noisy_lowlight.png")

    receipt_low = add_noise(adjust_brightness(Image.open(OUT_DIR / "receipt_normal.png"), 0.5), 20)
    receipt_low.save(OUT_DIR / "receipt_noisy_lowlight.png")

    made = sorted(p.name for p in OUT_DIR.glob("*.png"))
    print(f"Generated {len(made)} images:")
    for m in made:
        print(" -", m)


if __name__ == "__main__":
    main()
