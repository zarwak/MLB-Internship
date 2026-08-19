"""
Day 23 - the OCR pipeline, kept separate from the Streamlit interface.

Everything the app actually *does* lives here as small, single-purpose
functions: load an image, clean it up, read the text out of it. app.py
only handles buttons, layout and downloads. Splitting it this way means
batch_test.py can run the exact same pipeline over 18 images from the
command line without touching Streamlit at all - if the app and the
batch script had their own copies of this logic, they would drift apart
the first time either one got fixed.

WHY scikit-image instead of OpenCV for the preprocessing: EasyOCR
already installs scikit-image as one of its own dependencies, so it is
guaranteed to be there wherever EasyOCR runs - including Streamlit
Cloud, where `import cv2` crashed the Day 22 deployment. One less heavy
binary dependency that can fail on a machine we do not control.
(PaddleOCR pulls in opencv as its own dependency regardless - see the
README's "Two engines, two frameworks" section for what that means for
deployment.)

FOUR ENGINES: this pipeline supports EasyOCR, PaddleOCR, Tesseract and
docTR behind one interface. Preprocessing (grayscale, denoise, contrast,
threshold, deskew) is engine-agnostic - every engine just wants a clean
image array. Only the last step, extract_text(), differs per engine,
because each library returns its detections in a different shape.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color, exposure, feature, filters, transform

# EasyOCR's own model-download progress bar prints a filled-block
# character (the U+2588 "full block") to show progress. On Windows,
# stdout defaults to the system's legacy codepage (cp1252) whenever it is
# not attached to a real interactive terminal - which includes every
# subprocess Streamlit launches - and cp1252 cannot encode that
# character, crashing the download with a raw UnicodeEncodeError before
# the model ever finishes fetching. Reconfiguring stdout/stderr to UTF-8
# here (not just setting the PYTHONIOENCODING env var, which only takes
# effect at interpreter *startup* and cannot fix a stream that is already
# open) guarantees this works regardless of how the app gets launched.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Both EasyOCR and PaddleOCR default to downloading their model files into
# the user's home directory (~/.EasyOCR/, ~/.paddleocr/) - which on this
# machine is on C:, and C: has 0 bytes free (the same full-disk problem
# that made Day 22 build its venv on D: in the first place). EasyOCR
# already had its models cached there from before the disk filled up, so
# it kept working; PaddleOCR did not, and failed native model downloads
# with "OSError: No space left on device" the first time this ran.
# Pointing both engines at a folder next to this file keeps every engine's
# model cache on D: instead of relying on C: having room.
MODEL_CACHE_DIR = Path(__file__).parent / ".model_cache"

# Below this width, OCR engines start losing thin strokes: on Day 22 the
# smallest receipt (13px font) dropped EasyOCR to 0.49 confidence and
# turned "GROCERY" into "GROCERZ". Upscaling first gives the recognition
# model more pixels per character to work with.
MIN_WIDTH_FOR_OCR = 1000
MAX_UPSCALE = 3.0

ENGINES = ("easyocr", "paddleocr", "tesseract", "doctr")

# Short display names, for the app's engine picker. ENGINE_LABELS below
# carries the longer descriptions - kept for reference/README use, not
# for rendering inline in the UI (a dropdown option is not the place for
# a paragraph).
ENGINE_NAMES = {
    "easyocr": "EasyOCR",
    "paddleocr": "PaddleOCR",
    "tesseract": "Tesseract",
    "doctr": "docTR",
}
ENGINE_LABELS = {
    "easyocr": "EasyOCR - easiest API, ~100MB models, good all-rounder",
    "paddleocr": "PaddleOCR - lighter models, a second opinion on the same image",
    "tesseract": "Tesseract - classic engine, lightest and fastest, needs the Tesseract binary installed separately",
    "doctr": "docTR - deep-learning engine (mobilenet, PyTorch backend), strong on structured documents",
}

# docTR ships several detection/recognition architectures. The mobilenet
# ones are picked deliberately over the default resnet/master ones - a
# much smaller download and faster CPU inference, at a small accuracy
# cost that is fine for this app's printed-text use case.
DOCTR_DET_ARCH = "db_mobilenet_v3_large"
DOCTR_RECO_ARCH = "crnn_mobilenet_v3_small"


@dataclass
class OcrResult:
    """One OCR run. `lines` keeps the per-line detail so the app can show
    which lines the model was unsure about, instead of only a single
    number for the whole image."""

    text: str
    avg_confidence: float
    lines: list = field(default_factory=list)  # (text, confidence) per line
    preset: str = ""
    engine: str = ""

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def char_count(self) -> int:
        return len(self.text)


# -------------------------------------------------------------- the readers
# One cache slot per engine rather than a single global, because building
# a PaddleOCR reader does not give you an EasyOCR reader for free - each
# engine loads its own separate model files and neither can stand in for
# the other. Switching engines in the app should not force EasyOCR's
# models to reload just because PaddleOCR's cache slot got used.
_readers = {}


def _configure_tesseract_cmd(pytesseract_module) -> None:
    """Points pytesseract at the Tesseract binary. pytesseract is only a
    thin wrapper - it shells out to a separate native install that pip
    cannot provide. TESSERACT_CMD lets anyone override the path, but it
    is not relied on alone - `setx` only writes the registry, and a
    process only picks that up if the app that spawned its terminal was
    itself (re)launched after the `setx` ran, which nested terminals
    (VS Code, etc.) routinely fail to do even across "new" windows. So
    this also checks known install locations directly - including D:,
    the drive every other heavy install in this project already landed
    on because C: has 0 bytes free (see MODEL_CACHE_DIR above)."""
    custom_path = os.environ.get("TESSERACT_CMD")
    if custom_path and Path(custom_path).exists():
        pytesseract_module.pytesseract.tesseract_cmd = custom_path
        return
    candidate_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"D:\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            pytesseract_module.pytesseract.tesseract_cmd = str(candidate)
            return


def get_reader(engine: str = "easyocr", languages=("en",)):
    """Builds (once) and returns the reader object for the requested
    engine. Constructing either reader loads neural networks off disk,
    which takes a few seconds - reusing one object across many images is
    the difference between a batch script that takes a minute and one
    that takes several.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown OCR engine: {engine!r} (choose from {ENGINES})")

    if engine not in _readers:
        if engine == "easyocr":
            import easyocr
            _readers[engine] = easyocr.Reader(
                list(languages), gpu=False,
                model_storage_directory=str(MODEL_CACHE_DIR / "easyocr" / "model"),
                user_network_directory=str(MODEL_CACHE_DIR / "easyocr" / "user_network"))
        elif engine == "paddleocr":
            # PaddleOCR logs a lot at INFO level (model paths, warmup
            # info) by default - quietened here so it does not drown out
            # the app's own output the way it does with default settings.
            logging.getLogger("ppocr").setLevel(logging.ERROR)
            from paddleocr import PaddleOCR
            paddle_dir = MODEL_CACHE_DIR / "paddleocr"
            _readers[engine] = PaddleOCR(
                use_angle_cls=True, lang=languages[0] if languages else "en",
                show_log=False, det_model_dir=str(paddle_dir / "det"),
                rec_model_dir=str(paddle_dir / "rec"),
                cls_model_dir=str(paddle_dir / "cls"))
        elif engine == "tesseract":
            import pytesseract
            _configure_tesseract_cmd(pytesseract)
            try:
                pytesseract.get_tesseract_version()
            except pytesseract.TesseractNotFoundError as exc:
                raise RuntimeError(
                    "Tesseract engine binary not found. pytesseract is only "
                    "a Python wrapper - install the actual engine from "
                    "https://github.com/UB-Mannheim/tesseract/wiki (Windows "
                    "installer), then either leave it at the default path "
                    r"(C:\Program Files\Tesseract-OCR\tesseract.exe) or set "
                    "the TESSERACT_CMD environment variable to wherever "
                    "tesseract.exe ended up."
                ) from exc
            _readers[engine] = pytesseract
        elif engine == "doctr":
            # Both the model cache location and the backend choice have to
            # be set before doctr's own modules are imported - it reads
            # them at import time, not per call - same "point it at D:
            # before it touches the full C:" reasoning as MODEL_CACHE_DIR
            # above for the other two engines' model downloads.
            os.environ.setdefault("DOCTR_CACHE_DIR", str(MODEL_CACHE_DIR / "doctr"))
            os.environ.setdefault("USE_TORCH", "1")
            os.environ.setdefault("USE_TF", "0")
            from doctr.models import ocr_predictor
            _readers[engine] = ocr_predictor(
                det_arch=DOCTR_DET_ARCH, reco_arch=DOCTR_RECO_ARCH, pretrained=True)
    return _readers[engine]


# --------------------------------------------------------- preprocessing 101
def load_image(source) -> Image.Image:
    """Accepts a file path or an uploaded file object and always returns
    RGB. Converting to RGB up front removes surprises later: PNGs can
    carry an alpha channel and scans are sometimes 1-bit, and both break
    code that assumes three channels."""
    return Image.open(source).convert("RGB")


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Three colour channels down to one, as float in 0..1.

    OCR only cares about the shape of the strokes - dark pixels against
    light ones. Colour carries no information about which letter this is,
    so dropping it removes a distraction and shrinks the data."""
    if image.ndim == 2:
        return image.astype(float) / 255.0 if image.dtype == np.uint8 else image
    return color.rgb2gray(image)


def upscale_if_small(gray: np.ndarray, min_width: int = MIN_WIDTH_FOR_OCR) -> np.ndarray:
    """Enlarges images that are too small for the text to survive.

    This is the single cheapest accuracy win in the whole pipeline. A
    photo at 400px wide simply does not have enough pixels per letter for
    the model to tell 'e' from 'c', and no amount of contrast fixing adds
    detail back. Upscaling does not invent detail either, but it does
    give the model the letter *size* it was trained on."""
    if gray.shape[1] >= min_width:
        return gray
    factor = min(min_width / gray.shape[1], MAX_UPSCALE)
    return transform.rescale(gray, factor, anti_aliasing=True)


def denoise(gray: np.ndarray, radius: int = 1) -> np.ndarray:
    """Median filter: every pixel is replaced by the middle value of its
    neighbours, which wipes out isolated speckle while keeping edges
    reasonably sharp.

    Radius is deliberately small (1, so a 3x3 window). Day 22 used a 5x5
    median and it destroyed the text - at normal font sizes a 5-pixel
    window is about as thick as the strokes themselves, so it smeared
    neighbouring letters into each other. Denoising has to stay smaller
    than the thing you are trying to keep."""
    footprint = np.ones((2 * radius + 1, 2 * radius + 1))
    return filters.median(gray, footprint)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 0.02) -> np.ndarray:
    """CLAHE - contrast stretching done in small tiles across the image
    rather than once globally.

    That local part is the point. On a receipt that is half in shadow, a
    single global contrast boost has to pick one setting for both halves,
    so fixing the dark half blows out the bright half. CLAHE gets to make
    a different decision in each tile."""
    return exposure.equalize_adapthist(gray, clip_limit=clip_limit)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold: forces every pixel to pure black or pure white,
    picking the cut-off from the image's own brightness histogram rather
    than a fixed number that only suits one lighting condition.

    Use sparingly. See the README - this genuinely helps faded, washed
    out scans, and genuinely wrecks noisy ones, because it turns every
    noisy pixel near the cut-off into a black speck sitting on the
    letters."""
    return (gray > filters.threshold_otsu(gray)).astype(float)


# ------------------------------------------------------------------- deskew
def estimate_skew_angle(gray: np.ndarray, max_angle: float = 20.0) -> float:
    """Works out how many degrees the page is tilted, using the fact that
    a page of text is full of near-horizontal lines (the text rows, plus
    any rules or table borders).

    Canny finds the edges, then the Hough transform asks 'if I drew a
    straight line through this image at angle X, how many edge pixels
    would sit on it?' - the angles that score highest are the real lines
    in the page. We only search close to horizontal, take the median of
    the strongest lines (median, not mean, so one stray diagonal cannot
    drag the answer), and report the tilt in degrees.
    """
    edges = feature.canny(gray, sigma=2.0)
    if not edges.any():
        return 0.0

    # skimage describes a line by the angle of its *normal*, so a
    # perfectly horizontal line comes back as -90 degrees. Searching a
    # window around that gives us near-horizontal lines only.
    tested = np.deg2rad(np.arange(-90.0, -90.0 + max_angle, 0.2))
    tested = np.concatenate([tested, np.deg2rad(np.arange(90.0 - max_angle, 90.0, 0.2))])

    hspace, angles, dists = transform.hough_line(edges, theta=tested)
    peaks = transform.hough_line_peaks(hspace, angles, dists, num_peaks=25)
    if len(peaks[1]) == 0:
        return 0.0

    degrees = np.rad2deg(peaks[1])
    # Fold both ends of the search window back onto "degrees away from
    # horizontal", so -88 and +88 both mean a 2 degree tilt.
    skews = np.where(degrees < 0, degrees + 90.0, degrees - 90.0)
    return float(np.median(skews))


def deskew(gray: np.ndarray, max_angle: float = 20.0) -> tuple[np.ndarray, float]:
    """Straightens a tilted page. Returns the corrected image and the
    angle that was corrected, so the app can tell the user what it did.

    Skipped entirely for tilts under half a degree - rotating resamples
    every pixel, which very slightly blurs the text, and that is a bad
    trade for a tilt the model would have coped with anyway."""
    angle = estimate_skew_angle(gray, max_angle=max_angle)
    if abs(angle) < 0.5:
        return gray, 0.0
    straightened = transform.rotate(gray, angle, resize=True, cval=1.0, mode="constant")
    return straightened, angle


# ------------------------------------------------------------------ presets
# Each preset is just an ordered recipe of the steps above. Naming them
# and keeping them in one dict means the app's dropdown, the batch test
# and the README all describe the same four things.
PRESETS = {
    "none": [],
    "light": ["grayscale", "upscale"],
    "contrast": ["grayscale", "upscale", "contrast"],
    "scanned": ["grayscale", "upscale", "denoise", "contrast", "threshold"],
}

PRESET_LABELS = {
    "none": "None - send the original image straight to OCR",
    "light": "Light - grayscale + upscale small images (good default)",
    "contrast": "Contrast boost - adds CLAHE, for dim or uneven lighting",
    "scanned": "Scanned document - adds denoise + Otsu threshold, for faded scans",
}


def preprocess(image: Image.Image, preset: str = "light",
               straighten: bool = False) -> tuple[np.ndarray, float]:
    """Runs one preset over a PIL image and hands back the array that
    goes to OCR, plus the skew angle that was corrected (0.0 if none).

    Deskew is a separate flag rather than part of a preset because tilt
    is independent of lighting: a crooked photo can be perfectly lit, and
    a badly lit one can be perfectly straight."""
    array = np.array(image)
    steps = PRESETS.get(preset, PRESETS["light"])

    if not steps and not straighten:
        return array, 0.0

    gray = to_grayscale(array)
    angle = 0.0
    if straighten:
        gray, angle = deskew(gray)

    for step in steps:
        if step == "grayscale":
            continue  # already done above
        elif step == "upscale":
            gray = upscale_if_small(gray)
        elif step == "denoise":
            gray = denoise(gray)
        elif step == "contrast":
            gray = enhance_contrast(gray)
        elif step == "threshold":
            gray = binarize(gray)

    # EasyOCR wants 8-bit pixels, but every step above works in floats
    # from 0 to 1, so convert back at the very end (once) rather than
    # rounding to integers between each step and losing precision.
    return (np.clip(gray, 0, 1) * 255).astype(np.uint8), angle


# --------------------------------------------------------------- run the OCR
def _to_bgr_uint8(image_array: np.ndarray) -> np.ndarray:
    """PaddleOCR is built around OpenCV conventions and wants a 3-channel
    BGR uint8 array. Our preprocessing works in RGB (or single-channel
    grayscale, once the `light`/`contrast`/`scanned` presets have run),
    so this is the one conversion point that bridges the two."""
    if image_array.ndim == 2:
        return np.stack([image_array] * 3, axis=-1)
    return image_array[:, :, ::-1]


def _to_rgb3_uint8(image_array: np.ndarray) -> np.ndarray:
    """docTR wants a 3-channel RGB uint8 array too, but - unlike PaddleOCR
    - it is not built around OpenCV conventions, so there is no channel
    order to reverse, only the same single-to-three-channel stacking as
    the other engines need for a grayscale-preset image."""
    if image_array.ndim == 2:
        return np.stack([image_array] * 3, axis=-1)
    return image_array


def extract_text(reader, image_array: np.ndarray, engine: str = "easyocr",
                 preset: str = "") -> OcrResult:
    """Reads the text and packages up the result. This is the one place
    the two engines' different output shapes get normalised into the
    same OcrResult, so nothing downstream (the app, batch_test.py) needs
    to know which engine produced a given result."""
    if engine == "easyocr":
        # EasyOCR hands back one (box, text, confidence) triple per
        # detected line, because detection and recognition are two
        # separate models inside it.
        detections = reader.readtext(image_array)
        lines = [(text, float(conf)) for (_box, text, conf) in detections]

    elif engine == "paddleocr":
        # PaddleOCR's ocr() takes a *list* of images and returns a list
        # of per-image results, even for a single image - result[0] is
        # our image's detections, or None/[] if nothing was found
        # (both show up across different paddleocr versions).
        raw = reader.ocr(_to_bgr_uint8(image_array), cls=True)
        detections = raw[0] if raw and raw[0] else []
        lines = [(text, float(conf)) for (_box, (text, conf)) in detections]

    elif engine == "tesseract":
        # pytesseract only reads PIL Images (or file paths) - convert back
        # from the numpy array preprocess() hands us, single-channel ("L")
        # or RGB depending on which preset ran.
        pil_image = Image.fromarray(image_array, mode="L") \
            if image_array.ndim == 2 else Image.fromarray(image_array)
        data = reader.image_to_data(pil_image, output_type=reader.Output.DICT)

        # image_to_data returns one row per detected *word*, not per line.
        # Grouping by Tesseract's own block/paragraph/line numbering turns
        # a multi-word line back into one entry, matching the line-level
        # granularity the other three engines already return. A conf of
        # -1 marks a structural row (block/paragraph header) with no text
        # of its own, so it is skipped rather than averaged in.
        grouped: dict = {}
        for i, word in enumerate(data["text"]):
            word = word.strip()
            conf = float(data["conf"][i])
            if not word or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            grouped.setdefault(key, []).append((word, conf))
        lines = [
            (" ".join(w for w, _c in words), sum(c for _w, c in words) / len(words) / 100.0)
            for _key, words in sorted(grouped.items())
        ]

    elif engine == "doctr":
        # docTR groups its own output into pages -> blocks -> lines ->
        # words already, so unlike Tesseract there is no manual grouping
        # needed - just flatten it into the same (text, confidence) shape
        # everything else returns.
        result = reader([_to_rgb3_uint8(image_array)])
        lines = [
            (" ".join(word.value for word in line.words),
             sum(word.confidence for word in line.words) / len(line.words))
            for block in result.pages[0].blocks
            for line in block.lines
            if line.words
        ]

    else:
        raise ValueError(f"Unknown OCR engine: {engine!r} (choose from {ENGINES})")

    if not lines:
        return OcrResult(text="", avg_confidence=0.0, lines=[],
                         preset=preset, engine=engine)

    avg = sum(conf for _t, conf in lines) / len(lines)
    return OcrResult(text="\n".join(t for t, _c in lines), avg_confidence=avg,
                     lines=lines, preset=preset, engine=engine)


def run_ocr(image: Image.Image, preset: str = "light", straighten: bool = False,
            engine: str = "easyocr", reader=None) -> tuple[OcrResult, np.ndarray, float]:
    """The whole pipeline in one call: clean the image, read it, return
    the result plus the processed image (so the app can show what the
    model actually saw) and the corrected skew angle."""
    reader = reader or get_reader(engine=engine)
    processed, angle = preprocess(image, preset=preset, straighten=straighten)
    return extract_text(reader, processed, engine=engine, preset=preset), processed, angle


AUTO_PRESETS = ("light", "contrast", "none")


def run_ocr_auto(image: Image.Image, straighten: bool = False, engine: str = "easyocr",
                 reader=None, candidates=AUTO_PRESETS) -> tuple[OcrResult, np.ndarray, float]:
    """Tries a few presets on one engine and keeps whichever one the
    model was most confident about.

    This exists because Day 22 ended with the uncomfortable finding that
    no single preprocessing recipe wins on every image - heavy clean-up
    rescued the dim document and destroyed the noisy receipt. Rather than
    guess from the file name, this just runs two or three cheap variants
    and lets the engine's own confidence score pick. It costs one extra
    OCR pass per candidate, which is why the app makes it optional and
    the candidate list is short.

    Confidence is a proxy for correctness, not correctness itself: the
    model can be confidently wrong. It is still the best signal available
    without knowing the right answer in advance.
    """
    reader = reader or get_reader(engine=engine)
    best, best_image, best_angle = None, None, 0.0
    for preset in candidates:
        result, processed, angle = run_ocr(image, preset=preset, straighten=straighten,
                                           engine=engine, reader=reader)
        # Tie-break on text length: two runs can score the same average
        # while one of them found several more lines, and more text at
        # equal confidence is the better read.
        if best is None or (result.avg_confidence, result.char_count) > \
                (best.avg_confidence, best.char_count):
            best, best_image, best_angle = result, processed, angle
    return best, best_image, best_angle


def save_text(result: OcrResult, out_path: Path) -> Path:
    """Writes the extracted text to a .txt file, UTF-8 so accented
    characters and currency symbols survive the round trip."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.text, encoding="utf-8")
    return out_path
