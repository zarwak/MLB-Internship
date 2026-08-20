"""
Generates the 18 sample images used to test the Day 23 OCR app.

WHY synthetic images instead of real photos (same reasoning as Day 22):
because the text is drawn by this script, we know the exact correct
answer for every image. So when the OCR gets a word wrong later, that is
a real, checkable mistake - not a guess about what a blurry photo said.

Six groups, covering the document types the task asked for plus a few
deliberately hard cases:
  1. document  - plain paragraphs, standard sans-serif font
  2. receipt   - narrow itemised layout, monospace font
  3. invoice   - table layout with columns, totals and an invoice header
  4. form      - labelled fields, ruled lines and [X] checkboxes
  5. hard      - rotated / low-resolution / tiny-text versions
  6. extra     - book page, handwritten note, ID card

Run:  python sample_images/make_sample_images.py
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

OUT_DIR = Path(__file__).parent
FONT_DIR = Path("C:/Windows/Fonts")
random.seed(23)
np.random.seed(23)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size)


def add_noise(img: Image.Image, amount: int) -> Image.Image:
    """Sprinkles random per-pixel brightness noise, which is roughly what
    a phone camera sensor adds when the room is too dark."""
    arr = np.array(img).astype(int)
    arr = np.clip(arr + np.random.randint(-amount, amount + 1, arr.shape), 0, 255)
    return Image.fromarray(arr.astype("uint8"))


def brightness(img: Image.Image, factor: float) -> Image.Image:
    """factor < 1 makes it darker (bad lighting), > 1 washes it out (faded)."""
    return ImageEnhance.Brightness(img).enhance(factor)


def draw_lines(draw, lines, x, y, fnt, fill=(0, 0, 0), gap=10):
    """Draws a list of strings top to bottom and returns the final y."""
    for line in lines:
        draw.text((x, y), line, fill=fill, font=fnt)
        y += fnt.size + gap
    return y


# --------------------------------------------------------------- 1. document
DOCUMENT_TEXT = [
    "PROJECT STATUS REPORT",
    "",
    "Title: Document OCR Web Application",
    "Author: ML Internship Program, Day 23",
    "Date: 19 August 2026",
    "",
    "This report describes the work completed on the optical character",
    "recognition pipeline. The application accepts an uploaded image,",
    "applies preprocessing, and extracts every readable line of text.",
    "Users can review the result on screen and download it as a plain",
    "text file for further processing.",
]


def make_document(name, size=24, bright=1.0, scale=1.0):
    img = Image.new("RGB", (900, 620), "white")
    draw_lines(ImageDraw.Draw(img), DOCUMENT_TEXT, 50, 45, font("arial.ttf", size), gap=12)
    img = brightness(img, bright)
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BICUBIC)
    img.save(OUT_DIR / name)


# ---------------------------------------------------------------- 2. receipt
RECEIPT_LINES = [
    "       GREEN VALLEY MART",
    "    45 Jinnah Road, Islamabad",
    "      Tel: 051-2233445",
    "------------------------------",
    "Basmati Rice 5kg   x1  1450.00",
    "Fresh Milk 1L      x3   540.00",
    "Brown Bread        x2   360.00",
    "Chicken 1kg        x1   780.00",
    "Cooking Oil 1L     x1   650.00",
    "------------------------------",
    "SUBTOTAL               3780.00",
    "TAX 5%                  189.00",
    "TOTAL                  3969.00",
    "CASH                   4000.00",
    "CHANGE                   31.00",
    "------------------------------",
    "   THANK YOU, COME AGAIN",
]


def make_receipt(name, size=18, bright=1.0, noise=0):
    img = Image.new("RGB", (500, 700), "white")
    draw_lines(ImageDraw.Draw(img), RECEIPT_LINES, 25, 25, font("cour.ttf", size), gap=8)
    img = brightness(img, bright)
    if noise:
        img = add_noise(img, noise)
    img.save(OUT_DIR / name)


# ---------------------------------------------------------------- 3. invoice
def make_invoice(name, tint="white", angle=0.0, bright=1.0):
    """An invoice is really a small table, so this one is drawn column by
    column instead of as flat lines - that column layout is exactly what
    makes invoices harder for OCR than a plain paragraph."""
    img = Image.new("RGB", (860, 700), tint)
    draw = ImageDraw.Draw(img)
    bold, reg, small = font("arialbd.ttf", 22), font("arial.ttf", 18), font("arial.ttf", 16)

    draw.text((50, 40), "INVOICE", fill="black", font=font("arialbd.ttf", 38))
    draw_lines(draw, ["Invoice No: INV-2026-0847",
                      "Date: 12 August 2026",
                      "Due Date: 26 August 2026"], 550, 45, small, gap=6)

    draw_lines(draw, ["BILL TO:", "Horizon Tech Solutions",
                      "88 Model Town, Lahore", "NTN: 4410238-6"], 50, 120, reg, gap=6)

    draw.line((50, 240, 810, 240), fill="black", width=2)
    for label, x in [("DESCRIPTION", 55), ("QTY", 470), ("RATE", 570), ("AMOUNT", 690)]:
        draw.text((x, 250), label, fill="black", font=bold)
    draw.line((50, 285, 810, 285), fill="black", width=2)

    rows = [
        ("Web Application Development", "40", "5000", "200000"),
        ("UI and UX Design Services", "15", "4000", "60000"),
        ("Server Setup and Hosting", "1", "35000", "35000"),
        ("Technical Documentation", "8", "3000", "24000"),
    ]
    y = 300
    for desc, qty, rate, amt in rows:
        draw.text((55, y), desc, fill="black", font=reg)
        draw.text((480, y), qty, fill="black", font=reg)
        draw.text((570, y), rate, fill="black", font=reg)
        draw.text((690, y), amt, fill="black", font=reg)
        y += 40

    draw.line((450, y + 10, 810, y + 10), fill="black", width=1)
    for label, value, dy in [("Subtotal", "319000", 25), ("Tax 17%", "54230", 55),
                             ("TOTAL DUE", "373230", 90)]:
        f = bold if label == "TOTAL DUE" else reg
        draw.text((470, y + dy), label, fill="black", font=f)
        draw.text((690, y + dy), value, fill="black", font=f)

    draw.text((50, y + 135), "Payment due within 14 days.", fill="black", font=small)

    img = brightness(img, bright)
    if angle:
        img = img.rotate(angle, expand=True, fillcolor=tint, resample=Image.BICUBIC)
    img.save(OUT_DIR / name)


# ------------------------------------------------------------------- 4. form
def make_form(name, checkboxes=False, ink=(0, 0, 0), bright=1.0):
    """Forms are label + answer pairs sitting on ruled lines. Those lines
    matter: OCR engines often glue a label to the answer beside it, or
    read the ruled line itself as a row of dashes - a genuinely different
    failure mode from a plain paragraph."""
    img = Image.new("RGB", (860, 760), "white")
    draw = ImageDraw.Draw(img)
    title, reg = font("arialbd.ttf", 28), font("arial.ttf", 20)

    draw.text((50, 40), "STUDENT ENROLMENT FORM", fill=ink, font=title)
    draw.line((50, 85, 810, 85), fill=ink, width=2)

    fields = [
        ("Full Name", "Ayesha Khan"),
        ("Father Name", "Imran Khan"),
        ("Date of Birth", "14 / 03 / 2004"),
        ("CNIC Number", "35202-1234567-8"),
        ("Contact Number", "0300-1234567"),
        ("Email Address", "ayesha.khan@example.com"),
        ("Home Address", "House 12, Gulberg, Lahore"),
    ]
    y = 115
    for label, value in fields:
        draw.text((50, y), label + ":", fill=ink, font=reg)
        draw.text((290, y), value, fill=ink, font=reg)
        draw.line((285, y + 28, 810, y + 28), fill=ink, width=1)
        y += 58

    if checkboxes:
        draw.text((50, y + 10), "Programme Applied For:", fill=ink, font=reg)
        y += 55
        for label, checked in [("Computer Science", True), ("Data Science", False),
                               ("Software Engineering", False)]:
            draw.rectangle((55, y, 77, y + 22), outline=ink, width=2)
            if checked:
                draw.text((59, y - 4), "X", fill=ink, font=reg)
            draw.text((95, y - 2), label, fill=ink, font=reg)
            y += 40
    else:
        draw.text((50, y + 20), "Signature:", fill=ink, font=reg)
        draw.line((165, y + 48, 500, y + 48), fill=ink, width=1)

    img = brightness(img, bright)
    img.save(OUT_DIR / name)


# ------------------------------------------------------------- 5 & 6. extras
BOOK_PARAGRAPH = [
    "Chapter 4: Reading Machines",
    "",
    "Long before a computer could read, people imagined one that",
    "could. The idea seemed simple enough: show the machine a page,",
    "and let it say the words back. What took decades to appreciate",
    "was how much of reading happens above the level of the letter,",
    "in expectation, in context, and in the quiet corrections a",
    "reader makes without ever noticing them.",
]


def make_book_page(name, size=20, tint=(248, 243, 230)):
    img = Image.new("RGB", (760, 480), tint)
    draw_lines(ImageDraw.Draw(img), BOOK_PARAGRAPH, 55, 40, font("georgia.ttf", size),
               fill=(40, 30, 20), gap=10)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    img.save(OUT_DIR / name)


def make_handwritten(name, text, size=42):
    """Script font plus a little vertical wobble per letter. It looks
    handwritten, but see the README - it is not as hard as real
    handwriting, so treat the score here as optimistic."""
    fnt = font("segoescb.ttf", size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    # Measure the width the same way we draw it (letter by letter),
    # because for this script font the per-letter widths add up to more
    # than the width of the whole string measured in one go.
    width = 40
    for ch in text:
        bbox = probe.textbbox((0, 0), ch, font=fnt)
        width += (bbox[2] - bbox[0]) + 2

    img = Image.new("RGB", (width + 40, size + 100), "white")
    draw = ImageDraw.Draw(img)
    x = 40
    for ch in text:
        draw.text((x, 40 + random.randint(-4, 4)), ch, fill="navy", font=fnt)
        bbox = draw.textbbox((0, 0), ch, font=fnt)
        x += (bbox[2] - bbox[0]) + 2
    img.save(OUT_DIR / name)


def make_id_card(name):
    img = Image.new("RGB", (620, 380), (235, 240, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 619, 70), fill=(25, 60, 120))
    draw.text((22, 20), "NATIONAL IDENTITY CARD", fill="white", font=font("arialbd.ttf", 26))
    draw.rectangle((30, 100, 170, 290), outline=(120, 130, 150), width=2)
    draw.text((58, 185), "PHOTO", fill=(140, 150, 170), font=font("arial.ttf", 20))

    reg, bold = font("arial.ttf", 16), font("arialbd.ttf", 18)
    y = 105
    for label, value in [("Name", "Bilal Ahmed"), ("Father Name", "Rashid Ahmed"),
                         ("Identity Number", "42101-7654321-3"),
                         ("Date of Birth", "07.11.1998"), ("Expiry", "07.11.2030")]:
        draw.text((200, y), label, fill=(90, 100, 120), font=reg)
        draw.text((200, y + 20), value, fill=(10, 20, 40), font=bold)
        y += 56
    img.save(OUT_DIR / name)


def main():
    # 1. documents
    make_document("document_clean.png", size=24)
    make_document("document_small_text.png", size=14)
    make_document("document_dim_lighting.png", size=24, bright=0.55)

    # 2. receipts
    make_receipt("receipt_clean.png")
    make_receipt("receipt_faded.png", bright=1.55)
    make_receipt("receipt_noisy_lowlight.png", bright=0.55, noise=25)

    # 3. invoices
    make_invoice("invoice_clean.png")
    make_invoice("invoice_scanned_tint.png", tint=(238, 232, 214), bright=0.9)
    make_invoice("invoice_skewed.png", angle=-7.0)

    # 4. forms
    make_form("form_enrolment.png")
    make_form("form_with_checkboxes.png", checkboxes=True)
    make_form("form_low_contrast.png", ink=(120, 120, 120), bright=1.15)

    # 5. deliberately hard versions
    Image.open(OUT_DIR / "document_clean.png").rotate(
        6.0, expand=True, fillcolor="white", resample=Image.BICUBIC
    ).save(OUT_DIR / "document_rotated.png")
    make_document("document_low_resolution.png", size=24, scale=0.45)
    make_receipt("receipt_tiny_text.png", size=11)

    # 6. extras
    make_book_page("book_page.png")
    make_handwritten("handwritten_note.png", "Please collect the invoice on Friday")
    make_id_card("id_card.png")

    made = sorted(p.name for p in OUT_DIR.glob("*.png"))
    print("Generated " + str(len(made)) + " images in " + str(OUT_DIR) + ":")
    for m in made:
        print(" -", m)


if __name__ == "__main__":
    main()
