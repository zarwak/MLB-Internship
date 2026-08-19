"""
Day 23 - tests the OCR pipeline on all 18 sample images, with both
EasyOCR and PaddleOCR.

The task asked for at least 15 test images across documents, receipts,
invoices and forms, and (once PaddleOCR was added) for both engines to
be compared. Rather than eyeball them one at a time in the app, this
runs every image through all four presets on both engines - 144 OCR
passes - and writes the numbers down, which is what makes the
comparison tables in the README possible.

Outputs (all into ../sample_outputs/):
  <image>.txt                   - text from the single best (engine, preset)
  <image>_easyocr.txt           - text from EasyOCR's best preset
  <image>_paddleocr.txt         - text from PaddleOCR's best preset
  preset_comparison.txt         - every image x every preset x every engine
  results_summary.md            - the readable tables

Run:  python coding_practice/batch_test.py
"""

# Deskew is measured separately (see test_deskew below) rather than being
# switched on for the two tilted images in the main table. Applying it to
# 2 of the 18 rows would mean those rows had an extra step the other 16
# did not, so any difference could not be pinned on the preset - and it
# turned out to matter, because deskew makes the score *worse* on a
# purely-confidence reading (see README).

import sys
import time
from pathlib import Path

# ocr_pipeline.py lives in Day23/, one level above this file - see the
# matching comment in mini_project/app.py. Both the app and this script
# import the same module instead of keeping their own copies.
DAY_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(DAY_DIR))

from ocr_pipeline import ENGINES, PRESETS, get_reader, load_image, run_ocr, save_text

IMAGES_DIR = DAY_DIR / "sample_images"
OUTPUT_DIR = DAY_DIR / "sample_outputs"

# The two images built with a deliberate tilt, tested separately below.
SKEWED = ["document_rotated.png", "invoice_skewed.png"]


def test_one_image(readers: dict, image_path: Path) -> dict:
    """Runs all four presets on both engines for one image. Returns
    results keyed by (engine, preset), plus the best preset per engine
    and the single best (engine, preset) combo overall.

    Deskew is off for every image here, deliberately - see the note at
    the top of the file."""
    image = load_image(image_path)

    results = {}
    for engine in ENGINES:
        reader = readers[engine]
        for preset in PRESETS:
            started = time.time()
            result, _processed, _angle = run_ocr(image, preset=preset, straighten=False,
                                                 engine=engine, reader=reader)
            results[(engine, preset)] = {"result": result, "seconds": time.time() - started}

    best_per_engine = {
        engine: max(PRESETS, key=lambda p: (results[(engine, p)]["result"].avg_confidence,
                                            results[(engine, p)]["result"].char_count))
        for engine in ENGINES
    }
    overall_engine = max(
        ENGINES,
        key=lambda e: (results[(e, best_per_engine[e])]["result"].avg_confidence,
                       results[(e, best_per_engine[e])]["result"].char_count))

    return {"name": image_path.name, "results": results,
            "best_per_engine": best_per_engine, "overall_engine": overall_engine}


def test_deskew(readers: dict) -> list:
    """Deskew on vs off for the two tilted pages, plus a straight page as
    a control - run on both engines, since a detector that boxes text
    horizontally (which is what causes the shredding effect) is a design
    choice either engine could share or not.

    Line count is recorded here, not just confidence, and that turned out
    to be the whole point on EasyOCR - see README. A tilted page can make
    a detector chop each slanted row into short fragments; short
    fragments score individually high, so the *average* confidence goes
    up even though the document has been shredded. Confidence alone would
    have said deskew was harmful; line count says the opposite.
    """
    rows = []
    for name in SKEWED + ["document_clean.png"]:
        image = load_image(IMAGES_DIR / name)
        entry = {"name": name}
        for engine in ENGINES:
            reader = readers[engine]
            for mode, straighten in [("off", False), ("on", True)]:
                result, _processed, angle = run_ocr(image, preset="contrast", straighten=straighten,
                                                    engine=engine, reader=reader)
                entry[(engine, mode)] = {"result": result, "angle": angle}
        rows.append(entry)
    return rows


def write_outputs(all_results: list, deskew_rows: list) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. text files: per-engine best, plus one overall-best per image
    for entry in all_results:
        for engine in ENGINES:
            best_preset = entry["best_per_engine"][engine]
            r = entry["results"][(engine, best_preset)]["result"]
            save_text(r, OUTPUT_DIR / "{}_{}.txt".format(Path(entry["name"]).stem, engine))
        overall = entry["overall_engine"]
        overall_preset = entry["best_per_engine"][overall]
        save_text(entry["results"][(overall, overall_preset)]["result"],
                  OUTPUT_DIR / (Path(entry["name"]).stem + ".txt"))

    # 2. the detailed per-image x per-engine x per-preset log
    lines = ["Day 23 - preset x engine comparison across all sample images",
             "=" * 70,
             "avg_conf = the engine's own average confidence per detected line",
             "chars    = how many characters came out", ""]
    for entry in all_results:
        lines.append(entry["name"])
        for engine in ENGINES:
            lines.append("  [{}]".format(engine))
            for preset in PRESETS:
                data = entry["results"][(engine, preset)]
                r = data["result"]
                marker = "  <-- best for this engine" if preset == entry["best_per_engine"][engine] else ""
                lines.append(
                    "    {:9s} avg_conf={:.2f}  lines={:3d}  chars={:4d}  {:.1f}s{}".format(
                        preset, r.avg_confidence, r.line_count, r.char_count,
                        data["seconds"], marker))
        lines.append("  overall winner: {}".format(entry["overall_engine"]))
        lines.append("")

    lines += ["", "Deskew on vs off (preset: contrast)", "-" * 70]
    for row in deskew_rows:
        lines.append(row["name"])
        for engine in ENGINES:
            for mode in ("off", "on"):
                r = row[(engine, mode)]["result"]
                lines.append(
                    "  [{}] deskew {:3s} avg_conf={:.2f}  lines={:3d}  chars={:4d}"
                    "  corrected={:.1f} deg".format(
                        engine, mode, r.avg_confidence, r.line_count, r.char_count,
                        row[(engine, mode)]["angle"]))
        lines.append("")
    (OUTPUT_DIR / "preset_comparison.txt").write_text("\n".join(lines), encoding="utf-8")

    # 3. the markdown tables used in the README
    md = ["# OCR results on all sample images (EasyOCR vs PaddleOCR)", "",
          "Numbers are each engine's own average confidence (1.00 = completely sure), "
          "using that engine's best-scoring preset for the image.", "",
          "| Image | EasyOCR (best preset) | PaddleOCR (best preset) | Overall winner |",
          "|---|---|---|---|"]
    for entry in all_results:
        row = ["`" + entry["name"] + "`"]
        for engine in ENGINES:
            preset = entry["best_per_engine"][engine]
            conf = entry["results"][(engine, preset)]["result"].avg_confidence
            cell = "{:.2f} ({})".format(conf, preset)
            row.append("**" + cell + "**" if engine == entry["overall_engine"] else cell)
        row.append(entry["overall_engine"])
        md.append("| " + " | ".join(row) + " |")

    md += ["", "## Average confidence per preset, per engine", "",
           "| Engine | Preset | Average confidence | Times it won for that engine |",
           "|---|---|---|---|"]
    for engine in ENGINES:
        for preset in PRESETS:
            confs = [e["results"][(engine, preset)]["result"].avg_confidence for e in all_results]
            wins = sum(1 for e in all_results if e["best_per_engine"][engine] == preset)
            md.append("| {} | {} | {:.3f} | {} |".format(engine, preset, sum(confs) / len(confs), wins))

    md += ["", "## Engine head-to-head (each engine's own best preset per image)", "",
           "| Engine | Average confidence | Images where it won |", "|---|---|---|"]
    for engine in ENGINES:
        confs = [e["results"][(engine, e["best_per_engine"][engine])]["result"].avg_confidence
                 for e in all_results]
        wins = sum(1 for e in all_results if e["overall_engine"] == engine)
        md.append("| {} | {:.3f} | {} / {} |".format(engine, sum(confs) / len(confs), wins, len(all_results)))

    md += ["", "## Deskew on vs off", "",
           "Watch the line count, not the confidence - see README.", "",
           "| Image | Engine | Deskew | Confidence | Lines | Chars | Corrected |",
           "|---|---|---|---|---|---|---|"]
    for row in deskew_rows:
        for engine in ENGINES:
            for mode in ("off", "on"):
                r = row[(engine, mode)]["result"]
                md.append("| `{}` | {} | {} | {:.2f} | {} | {} | {:.1f}° |".format(
                    row["name"], engine, mode, r.avg_confidence, r.line_count,
                    r.char_count, row[(engine, mode)]["angle"]))

    (OUTPUT_DIR / "results_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    images = sorted(IMAGES_DIR.glob("*.png"))
    print("Found {} sample images. Testing both engines x 4 presets = {} OCR passes per image.".format(
        len(images), len(ENGINES) * len(PRESETS)))

    readers = {}
    for engine in ENGINES:
        print("Loading {} models (one-time, takes a while the first time)...".format(engine))
        readers[engine] = get_reader(engine=engine)

    all_results = []
    for i, path in enumerate(images, 1):
        entry = test_one_image(readers, path)
        all_results.append(entry)
        parts = []
        for engine in ENGINES:
            preset = entry["best_per_engine"][engine]
            r = entry["results"][(engine, preset)]["result"]
            parts.append("{}={:.2f}({})".format(engine, r.avg_confidence, preset))
        print("[{:2d}/{}] {:32s} {}  winner={}".format(
            i, len(images), path.name, "  ".join(parts), entry["overall_engine"]))

    print("\nDeskew on vs off on the tilted pages, both engines...")
    deskew_rows = test_deskew(readers)
    for row in deskew_rows:
        for engine in ENGINES:
            for mode in ("off", "on"):
                r = row[(engine, mode)]["result"]
                print("  {:24s} [{}] deskew={:3s} conf={:.2f} lines={:3d} chars={}".format(
                    row["name"], engine, mode, r.avg_confidence, r.line_count, r.char_count))

    write_outputs(all_results, deskew_rows)
    print("\nWrote per-image .txt files + preset_comparison.txt + results_summary.md"
          " to {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
