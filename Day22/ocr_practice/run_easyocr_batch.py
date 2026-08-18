"""
Day 22 - OCR practice: run EasyOCR over every sample image, save the
extracted text, then do a preprocessing before/after comparison.

WHY one shared `get_reader()` instead of calling `easyocr.Reader(...)`
every time: EasyOCR loads two neural networks from disk into memory when
you construct a Reader (a text-detection model that finds *where* text
is, and a text-recognition model that reads *what* it says). That load
takes a few seconds. Reusing one Reader object across all 17 images means
we pay that cost once, not seventeen times - the same "load once, call
many times" idea as any ML model in a real service.
"""

from pathlib import Path

import cv2
import easyocr
import numpy as np

IMAGES_DIR = Path(__file__).parent.parent / "images"
TEXT_DIR = Path(__file__).parent.parent / "extracted_text"
TEXT_DIR.mkdir(exist_ok=True)

_reader = None


def get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        print("Loading EasyOCR models (detection + recognition)... one-time cost.")
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_text(image_path: Path) -> tuple[str, float]:
    """Runs OCR and returns (joined_text, average_confidence).

    EasyOCR's readtext() returns a list of (box, text, confidence) per
    detected text region - confidence is the model's own certainty about
    that single line, useful for spotting which images gave it trouble.
    """
    reader = get_reader()
    results = reader.readtext(str(image_path))
    if not results:
        return "", 0.0
    lines = [text for (_box, text, _conf) in results]
    confidences = [conf for (_box, _text, conf) in results]
    return "\n".join(lines), sum(confidences) / len(confidences)


def run_on_all_images():
    """Task: 'Read text from at least 10 different images' - we have 17."""
    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    print(f"\nFound {len(image_paths)} sample images.\n{'=' * 60}")

    summary = []
    for path in image_paths:
        text, avg_conf = extract_text(path)
        out_file = TEXT_DIR / f"{path.stem}.txt"
        out_file.write_text(text, encoding="utf-8")

        preview = text.replace("\n", " | ")[:70]
        print(f"{path.name:35s} conf={avg_conf:.2f}  \"{preview}\"")
        summary.append((path.name, avg_conf, len(text)))

    print("=" * 60)
    print(f"Saved {len(image_paths)} .txt files to {TEXT_DIR}")
    return summary


# ----------------------------------------------------- preprocessing compare
def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Collapses 3 color channels to 1. OCR only cares about the shape of
    strokes (dark pixels vs light pixels), not hue - color is noise for
    this task, so dropping it removes a source of confusion for the model
    and shrinks the data the network has to process."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance(img: np.ndarray, denoise: bool = False) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) stretches
    local contrast in small tiles across the image, rather than one
    global stretch. That matters for photos with uneven lighting (a
    receipt half in shadow) where a single global contrast boost would
    blow out the bright half trying to fix the dark half.

    denoise=True runs a median blur first. This turned out to matter a
    lot - see the README's "Preprocessing results" section: CLAHE+Otsu
    alone made the noisy test images *worse*, not better, because Otsu's
    hard black/white cutoff turns per-pixel sensor noise into salt-and-
    pepper speckle sitting right on top of the letters. A median blur
    (each pixel replaced by the median of its neighborhood) removes that
    kind of isolated-pixel noise while keeping edges reasonably sharp,
    *before* the threshold gets applied - order matters here.
    """
    gray = to_grayscale(img) if len(img.shape) == 3 else img
    if denoise:
        gray = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Otsu thresholding: picks the brightness cutoff that best splits the
    # image's own histogram into two peaks (text vs background) instead
    # of a fixed number that only works for one lighting condition.
    _thresh_val, binary = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def compare_preprocessing(image_name: str):
    path = IMAGES_DIR / image_name
    img = cv2.imread(str(path))
    reader = get_reader()

    variants = {
        "raw": img,
        "grayscale": to_grayscale(img),
        "grayscale+CLAHE+otsu": enhance(img),
        "grayscale+denoise+CLAHE+otsu": enhance(img, denoise=True),
    }

    print(f"\n--- Preprocessing comparison: {image_name} ---")
    report_lines = [f"Preprocessing comparison for {image_name}\n"]
    for label, variant in variants.items():
        results = reader.readtext(variant)
        text = " ".join(t for (_b, t, _c) in results)
        avg_conf = sum(c for (_b, _t, c) in results) / len(results) if results else 0.0
        line = f"[{label:22s}] avg_conf={avg_conf:.2f}  chars={len(text):4d}  \"{text[:60]}\""
        print(line)
        report_lines.append(line)

    report_path = TEXT_DIR / f"{Path(image_name).stem}_preprocessing_comparison.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def main():
    run_on_all_images()

    # The three "hard" images built specifically to test preprocessing:
    # low light + sensor noise, the kind of thing a real phone photo of a
    # receipt looks like in a dim restaurant.
    for hard_image in [
        "printed_doc_noisy_lowlight.png",
        "receipt_noisy_lowlight.png",
        "printed_doc_dim_lighting.png",
    ]:
        compare_preprocessing(hard_image)


if __name__ == "__main__":
    main()
