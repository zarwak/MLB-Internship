# Day 23 - Document OCR Web Application

The app started out EasyOCR-only. Once the brief clarified that the goal
was to compare OCR engines and let the user pick, I added **PaddleOCR**
as a second engine, then later added **Tesseract** and **docTR** too, so
there are four engines now, each selectable from a sidebar dropdown.
There used to be a "Compare both" mode as well, that ran an image
through two engines at once and kept whichever was more confident - I
dropped it once there were four engines to choose from, because a
head-to-head scoreboard stops being a simple side-by-side once there are
four possible answers instead of two, and picking one engine and reading
its result is a much easier thing for a person to actually use. Everything
below reflects the current four-engine version; where a finding is
specific to an earlier build, it says so.

---
## DEMO:
![DEMO VIDEO](demo_video_summarizer.gif)

---

## What the app does

1. You pick an **OCR Engine** in the sidebar: EasyOCR, PaddleOCR,
   Tesseract, or docTR.
2. You upload an image (a document, receipt, invoice, form - anything
   with text on it).
3. The app cleans the image up a bit, because OCR reads cleaner images
   better.
4. The chosen engine reads the text.
5. You see the original image on the left and the text on the right.
6. You can fix any mistakes in the text box, then download it as a
   `.txt` file.

There is also an **Auto mode** (on by default) that tries a few
different clean-up recipes and keeps whichever one worked best, so you
do not have to know which preprocessing setting to pick.

---

## Which OCR libraries I used

**EasyOCR, PaddleOCR, Tesseract and docTR**, selectable from a dropdown.

### EasyOCR

The easiest of the two to use - you write two lines and you get your
text:

```python
reader = easyocr.Reader(["en"])
results = reader.readtext(image)
```

It gives a **confidence score** for every line it reads - a number
between 0 and 1 saying how sure it is. That turned out to be really
useful, both for showing the user which lines to double-check and for
Auto mode, which uses it to pick the best preprocessing result.

The trade-off is that it is the slower of the two and downloads about
100 MB of model files the first time you run it.

### PaddleOCR

A different OCR library from Baidu, built on the PaddlePaddle framework
rather than PyTorch. The API is one step more involved than EasyOCR's -
it wants an explicit angle-classifier flag and returns its detections in
a slightly different shape - but the model files it downloads are
noticeably smaller, and it genuinely does not always agree with EasyOCR
on the same image - which is exactly why it is worth having as a second
opinion, not just a backup.

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")
result = ocr.ocr(image, cls=True)
```

### Tesseract

The oldest and most classic of the four - it started life at HP in the
1980s, and Google maintains it now. `pytesseract` is only a thin Python
wrapper around it; the actual engine is a separate program that has to
be installed on the machine itself, which `pip install` cannot do for
you (see challenge 12 below for how much that one fact cost me):

```python
import pytesseract
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
```

It hands back one row per detected *word*, not per line, so
`ocr_pipeline.py` groups words back into lines using Tesseract's own
block/paragraph/line numbering, to match the line-level shape the other
three engines already return. It is the lightest and fastest of the
four - no deep-learning model to load at all - but only works if
something on the machine actually has the Tesseract program installed.

### docTR

A newer engine built on PyTorch, which means it shares its framework
with EasyOCR rather than adding a third deep-learning framework the way
PaddleOCR did. It ships several detection/recognition architectures, and
I picked the `mobilenet` ones over the default `resnet`/`master` ones on
purpose - a much smaller download and faster on a CPU, for a small
accuracy cost that does not matter for printed text:

```python
from doctr.models import ocr_predictor
model = ocr_predictor(det_arch="db_mobilenet_v3_large",
                      reco_arch="crnn_mobilenet_v3_small", pretrained=True)
result = model([image])
```

Its output already comes grouped into pages -> blocks -> lines -> words,
so unlike Tesseract there is no manual regrouping needed.

`ocr_pipeline.py` normalises all four engines' output into the same
`OcrResult` shape, so nothing else in the app needs to know which engine
actually produced a given result - see "Four engines, one interface"
below.

---

## Four engines, one interface

Adding a second engine after the app already worked with one meant
deciding how much of the pipeline could stay shared versus how much had
to fork per engine.

**Shared, unchanged:** all the preprocessing - grayscale, upscale,
denoise, contrast, threshold, deskew. All four engines just want a clean
image array; none of them care how it got clean. This is most of the
pipeline's actual logic, and none of it needed to know a second (or
third, or fourth) engine existed.

**Forked, on purpose:** only the last step, reading the result back out -
and all four engines shape that result differently. EasyOCR's
`readtext()` returns a flat list of `(box, text, confidence)` triples.
PaddleOCR's `ocr()` wraps that one level deeper - a list *per input
image* (even for a single image) containing `[box, (text, confidence)]`
pairs, and can hand back `None` instead of an empty list when it finds
nothing, which is worth guarding for explicitly rather than assuming "no
results" always looks the same. Tesseract's `image_to_data()` returns one
row *per word*, so it needs grouping back into lines using its own
block/paragraph/line numbers before it matches the others. docTR already
comes pre-grouped into pages -> blocks -> lines -> words, so that one
just needs flattening. `extract_text()` in `ocr_pipeline.py` is the one
place all four differences are handled, so everything above it - the
app, `batch_test.py` - only ever deals with one `OcrResult` shape
regardless of which engine produced it.

**One real conversion needed:** PaddleOCR expects BGR channel order (it
is built around OpenCV conventions), while the preprocessing pipeline
works in RGB throughout. `_to_bgr_uint8()` is the single bridge point -
reverse the channels for a colour image, or stack a grayscale one into
three identical channels if a preset already flattened it. Getting this
backwards would not crash anything (PaddleOCR would still detect *some*
text - flipped red/blue channels do not change stroke shapes), it would
just be a quiet, hard-to-notice accuracy hit rather than an error, which
is exactly the kind of bug worth writing a deliberate conversion
function for instead of leaving to chance. Tesseract and docTR need no
such fix - Tesseract reads a plain PIL image and docTR stays in RGB the
whole way through, same as the pipeline itself.

---

## Four engines, three frameworks

Adding PaddleOCR did not just add a second `pip install` - it added a second
entire deep-learning framework. EasyOCR is built on PyTorch, PaddleOCR is
built on PaddlePaddle, and neither shares any weight with the other. docTR
is PyTorch-based too, so it rides along on EasyOCR's framework for free.
Tesseract is not a Python deep-learning framework at all - it is a separate
compiled C++ program, so it barely adds anything to the Python side:

| | Framework | Approx. size (CPU wheels) |
|---|---|---|
| EasyOCR | PyTorch | ~200MB (`torch==2.13.0+cpu`, pinned via `--extra-index-url` - plain PyPI `torch` bundles several GB of CUDA) |
| PaddleOCR | PaddlePaddle | ~100-150MB (`paddlepaddle==3.0.0` is CPU-only by default on PyPI) |
| docTR | PyTorch (shared with EasyOCR) | just its own package + weights, no extra framework |
| Tesseract | none (native binary) | pytesseract itself is tiny; the actual engine is a separate program installed outside pip entirely |

Add in each engine's own dependencies (opencv, scipy and scikit-learn all
come along as paddleocr's own requirements, and easyocr/paddleocr/doctr
each pull in their own copy of opencv too - see challenge 18 below for what
that caused on Streamlit Cloud) and the full environment is roughly
**3-4GB** - see the install step in `HOW_TO_RUN.txt`.

**The honest trade-off:** Day 22 already hit Streamlit Cloud's free-tier
build limits once, with torch alone. Every extra engine adds to that risk -
a build that runs out of time or disk partway through, silently installing
only Streamlit's own dependencies and giving no warning (see challenge 6 in
"Challenges I faced"). It is now deployed and working (see the deployment
challenges below for everything that took), but I never got to properly
load-test whether running all four engines back to back in one session
stays inside the free tier's 1GB RAM ceiling - see "Possible improvements".

**Why keep all four anyway:** once "Compare both" mode was gone, the reason
for keeping every engine changed - it stopped being about running two
models side by side and became about genuine choice. Each engine actually
trades off differently: Tesseract is the fastest and lightest for a clean
printed page, EasyOCR is the steady all-rounder with a confidence score
per line, PaddleOCR is a second opinion when EasyOCR is unsure, and docTR
tends to do best on more structured documents. If a deploy does hit the
free tier's limit, the fallback is dropping to fewer engines: comment out
the ones you do not need in `requirements.txt` and their branch in
`ocr_pipeline.get_reader()`, trading choice for a build that fits.

---

## What preprocessing I applied

Preprocessing means cleaning up the image *before* the OCR reads it.
The idea is that the model has an easier job if you hand it a tidy
image. I built five separate steps and grouped them into four presets.

### The individual steps

| Step | What it does | Why it helps |
|---|---|---|
| **Grayscale** | Turns a colour image into shades of grey | OCR only cares about the *shape* of letters - dark marks on a light background. Colour tells it nothing useful, so removing it removes a distraction. |
| **Upscale** | Makes small images bigger (up to 3x) | If the text is tiny, there simply are not enough pixels for the model to tell an "e" from a "c". Making it bigger gives the model the letter size it expects. |
| **Denoise** | Replaces each pixel with the middle value of its neighbours | Photos taken in dim light get random specks. This wipes the specks out without blurring the letters too much. |
| **Contrast (CLAHE)** | Stretches out the difference between light and dark, in small patches | If half your receipt is in shadow, one single brightness fix cannot suit both halves. Doing it patch by patch can. |
| **Threshold (Otsu)** | Forces every pixel to be either pure black or pure white | Makes faded, washed-out text crisp again. Also a great way to ruin a noisy image - see below. |

### The four presets

| Preset | Steps included | Best for |
|---|---|---|
| `none` | nothing at all | Images that are already clean |
| `light` | grayscale + upscale | A safe general default |
| `contrast` | grayscale + upscale + contrast | Dim or unevenly lit photos |
| `scanned` | all five steps | Faded, washed-out scans |

There is also a separate **"Straighten a tilted page"** option, which is
its own checkbox rather than part of a preset - because a crooked photo
can be perfectly lit, and a well-aimed photo can be badly lit. The two
problems are unrelated, so they get their own controls.

---

## How well it worked

I tested all **18 sample images** with all four presets - 72 OCR runs.
Full numbers are in [`sample_outputs/results_summary.md`](sample_outputs/results_summary.md).

Here is the average across every image:

| Preset | Average confidence | How often it was the best |
|---|---|---|
| `none` | 0.835 | 5 |
| `light` | 0.836 | 3 |
| `contrast` | 0.828 | 8 |
| `scanned` | 0.672 | 2 |

The four presets score almost identically on average - except
`scanned`, which is clearly the worst. And that is the interesting part.

### The big lesson: more cleaning is not better cleaning

`scanned` is the preset that does the *most* work. It should have been
the best one. Instead it came last, and on some images it was a
disaster:

| Image | `none` | `scanned` |
|---|---|---|
| `document_small_text.png` | 0.87 | **0.09** |
| `receipt_noisy_lowlight.png` | 0.89 | **0.01** |
| `document_low_resolution.png` | 0.77 | **0.23** |

Those are not small drops. On the noisy receipt it went from reading the
receipt fine to reading essentially nothing.

**Why:** the culprit is the threshold step, which forces every pixel to
pure black or pure white. On a noisy photo, every speck of noise sitting
near the cut-off point becomes a solid black dot right on top of the
letters. The text gets buried in confetti.

But here is the thing - `scanned` was not useless. It won on exactly two
images:

| Image | `none` | `scanned` |
|---|---|---|
| `receipt_faded.png` | 0.75 | **0.83** |
| `form_low_contrast.png` | 0.84 | **0.89** |

Both of those are *faded* images - washed out, too little contrast. That
is precisely the problem a threshold is built to solve. So the rule I
ended up with is:

> Thresholding rescues faded images and destroys noisy ones. It is not
> "extra cleaning", it is a specific tool for a specific problem.

This is also exactly why I built the **Auto mode**. Since no single
recipe wins everywhere, the app just tries a few and keeps the best one
instead of guessing.

---

## The mistake that nearly fooled me

I added the "straighten a tilted page" feature and tested it on my two
crooked images. The confidence scores went **down**:

| Image | Deskew off | Deskew on |
|---|---|---|
| `document_rotated.png` | 0.94 | 0.87 |
| `invoice_skewed.png` | 0.90 | 0.89 |

My first thought was that straightening was pointless and I should drop
the feature. Then I looked at the number of *lines* it found, and the
real story showed up:

| Image | Deskew | Confidence | Lines found |
|---|---|---|---|
| `document_rotated.png` | off | 0.94 | **49** |
| `document_rotated.png` | on | 0.87 | **13** |
| `invoice_skewed.png` | off | 0.90 | **61** |
| `invoice_skewed.png` | on | 0.89 | **33** |

The straight versions of those documents have exactly **13** and **33**
lines. So the "worse" setting is the one getting it right.

What was happening: EasyOCR looks for text inside horizontal boxes. When
the page is tilted, a slanted line of text does not fit in a horizontal
box, so the model chops it into several short fragments instead. Short
fragments are easy to read, so each one gets a high score - and the
*average* goes up, even though the document has been shredded.

You can see it in the extracted text. Tilted, with straightening off,
the invoice number `INV-2026-0847` came out as four separate pieces:

```
Date: 12_
-0847
2026
26
```

**The lesson:** a single average score can go up while the actual result
gets worse. I only caught this because I recorded the line count too. If
I had trusted the confidence number on its own, I would have deleted a
feature that works.

(This does mean the app's Auto mode, which picks by confidence, is
trusting an imperfect judge. It only chooses between the preprocessing
presets and never touches the straighten setting, so this particular
trap does not bite it - but it is a real limitation, and it is on the
improvements list below.)

---

## Where it still struggles

**Tiny text is the hard limit.** `receipt_tiny_text.png` (11px font)
scored 0.53, the worst of all 18, and no preprocessing fixed it:

```
GREEN VALLEY MART     <- fine, it is big
4 Jiccal              <- should be "45 Jinnah"
Road,
Jlamabad              <- should be "Islamabad"
1430.00               <- should be 1450.00
3969
0o                    <- ".00" came out as "0o" on every single line
```

(The lines are also broken up oddly, because at this size the model
stopped seeing each row as one piece of text.)

Upscaling helped a bit (0.41 to 0.53) but could not save it. Enlarging
an image does not add detail that was never captured - if the pixels are
not there, they are not there. **The fix is a better photo, not better
code.**

**Tables lose their shape.** Invoices scored highest of everything
(0.94), and nearly every word is correct - but the layout is gone. The
app reads down the columns, so this:

| Description | Qty | Rate | Amount |
|---|---|---|---|
| Web Application Development | 40 | 5000 | 200000 |

comes out as a flat list: `Web Application Development`, `40`, `5000`,
`200000`. The words are right, but "which number belongs to which row"
is lost.

**Single characters go missing.** In the invoice, the quantities `40`
and `15` were read fine, but the quantities `1` and `8` vanished
completely. A lone digit is a very small thing to spot on a big page,
and the detector skipped right over both.

**Small character mix-ups** show up everywhere: a capital `I` read as a
lowercase `l` (`UI and UX` became `Ul and UX`), a comma read as a
semicolon, a full stop read as an underscore. These are the kind of
thing you fix by eye - which is why the app lets you edit the text
before downloading it.

---

## Challenges I faced

**1. The confidence score lied to me.** Covered in detail above. This was
the big one, and the fix was to measure a second thing (line count)
rather than trusting one number.

**2. My preprocessing made things worse before it made them better.**
I assumed more cleaning would mean better results, built the `scanned`
preset with all five steps, and watched it come last overall. Sorting
out *why* is what produced the faded-vs-noisy rule above.

**3. The median blur size mattered more than I expected.** On Day 22 I
used a 5x5 median blur to remove noise, and it wrecked the text. At
normal font sizes, 5 pixels is about as thick as the letter strokes
themselves, so it smeared neighbouring letters together. Dropping it to
3x3 fixed it. The rule: **your denoising has to be smaller than the
thing you are trying to keep.**

**4. Getting the rotation maths the right way round.** To straighten a
page I use a Hough transform, which finds the strong straight lines in
an image. The catch is that scikit-image describes a line by the angle
of its *normal* (the perpendicular), so a perfectly horizontal line
comes back as -90 degrees, not 0. I got the sign backwards at first and
made the tilt worse. I checked it by rotating a straight page by a known
amount and confirming the detector reported that same number back: I
rotated by +6 degrees and it reported -6, which is the correction
needed. Then I tested 0, 5, 10, 15, 20 and 30 degrees to be sure.

**5. A deprecation warning that was already past its deadline.** The app
used `use_container_width=True` for images, which Streamlit had marked
for removal after 31 December 2025 - a date that has now passed. It
still worked locally but could break on Streamlit Cloud, which installs
a recent version. Switched to `width="stretch"` and bumped the minimum
Streamlit version in `requirements.txt` to match.

**6. Not repeating yesterday's deployment mess.** Day 22 failed to
deploy three times, and the real reason turned out to be that
`requirements.txt` was one folder above the app file, where Streamlit
Cloud never looks. Today `app.py` lives in `mini_project/`, so a copy of
`requirements.txt` and `runtime.txt` lives right next to it there too -
same fix as Day 22 used once it split into `mini_project/` and
`ocr_practice/` folders.

**7. PaddleOCR 3.x is a different product, not just a version bump.**
My first instinct was to install the newest PaddleOCR, which is 3.x.
That version is built on PaddleX and defaults to a much heavier pipeline
- document orientation classification, layout analysis, and more -
downloading extra models for stages this app does not need, with a
different API (`.predict()` instead of `.ocr()`). I switched to the
last 2.x release (`paddleocr==2.9.1`) instead, which has the simpler,
widely-documented `PaddleOCR(...).ocr(image, cls=True)` interface and
only downloads the detection + recognition + angle-classification models
this app actually uses. Worth checking a library's major-version changelog
before assuming "newer" means "same thing, more features."

**8. PaddleOCR forces an older numpy, which meant a separate venv.**
`paddleocr==2.9.1` pins `numpy<2.0`. Day 23 originally reused Day 22's
venv (this app was EasyOCR-only at first), and that venv already had
numpy 2.x installed for EasyOCR and scikit-image. Installing paddleocr
into it would have downgraded numpy under packages that did not ask for
it, risking a subtle break in Day 22's already-finished, already-graded
work over a dependency neither of us asked for. Giving Day 23 its own
`.venv/` avoided that entirely - a repeat of the same "don't let one
day's dependencies leak into another's environment" lesson from Day 22's
original C:-drive problem, just triggered by a version conflict instead
of a full disk.

**9. The C: drive problem came back, in a new shape.** Both engines
default to caching their downloaded model files in the user's home
folder (`~/.EasyOCR/`, `~/.paddleocr/`) - and that folder is on C:,
which (per Day 22's original discovery) has 0 bytes free. EasyOCR had
already cached its models there from an earlier day, so it kept working
by accident; PaddleOCR had never run before and failed its very first
model download with `OSError: [Errno 28] No space left on device`. The
fix was to stop relying on either engine's default location at all -
`ocr_pipeline.py` now passes explicit `model_storage_directory` /
`det_model_dir` / `rec_model_dir` / `cls_model_dir` arguments pointing
both engines at a `.model_cache/` folder next to the code, on D:. docTR
got the same treatment (`DOCTR_CACHE_DIR`) the moment it was added,
before it ever got the chance to hit the same wall. Same
underlying problem as Day 22's pip installs, showing up in a completely
different code path - worth remembering that "give it its own venv"
does not automatically mean "nothing on this app writes to C: anymore."

**10. A working model does not mean a working model *on this version of
the framework*.** With everything installed, PaddleOCR crashed on every
single image with `NotFoundError: OneDnnContext does not have the input
Filter` inside a `fused_conv2d` operator - not a missing file or a typo,
a real inference-engine error. The cause was a version mismatch:
`paddleocr==2.9.1` ships model files built for older PaddlePaddle
inference internals, and the newest `paddlepaddle` (3.3.1 at the time)
applies an OneDNN operator-fusion optimisation during CPU inference that
those older model graphs were never built to expect. `paddleocr`'s own
`enable_mkldnn` flag - which looked like the obvious fix - defaults to
off already and had no effect, confirming this was paddle's own default
behaviour, not something paddleocr was asking for. Pinning
`paddlepaddle==3.0.0` (the first 3.x release, much closer to what 2.9.1
was actually built against) fixed it outright: same code, same model,
0.99 confidence instead of a crash. Lesson: "pip install worked" and
"the model runs" are two different claims, and a library that predates
its own dependency's newest release is exactly where the gap between
them shows up.

**11. EasyOCR's own progress bar can crash it, only on Windows, only
sometimes.** Testing EasyOCR crashed with
`UnicodeEncodeError: 'charmap' codec can't encode character '█'`
- EasyOCR prints a filled block character (█) to show download
progress, and Windows defaults new processes to the legacy `cp1252`
console encoding whenever stdout is not a real interactive terminal
(true for every subprocess Streamlit launches, and for background
scripts). `cp1252` cannot represent that character at all, so the crash
happens inside EasyOCR's own code, before the model finishes
downloading - nothing about the OCR itself is broken, only the progress
bar next to it. Setting `PYTHONIOENCODING=utf-8` in the shell fixes it,
but that only helps if whoever launches the app remembers to set it
first. The reliable fix lives in `ocr_pipeline.py` instead:
`sys.stdout.reconfigure(encoding="utf-8")`, which changes an
*already-open* stream's encoding at runtime - unlike the environment
variable, which is only read once at interpreter startup and cannot
retroactively fix a stream Python already opened. Verified by deleting
the cached models and re-downloading them with no environment variable
set at all.

**12. Tesseract needs an actual program installed, not just a pip
package.** `pytesseract` is only a wrapper - it shells out to a real
Tesseract binary that pip cannot install, because it is not a Python
package, it is a separate compiled program. Even after downloading the
Windows installer and pointing `TESSERACT_CMD` at the right path with
`setx`, it still failed the same way twice more, because `setx` only
writes the value to the registry, and any shell that was already open
before you ran it never picks that change up. Not even a brand new
terminal always fixes it, if whatever hosts that terminal was itself
opened before the `setx` ran - it inherits the environment its own
parent process had at *its* startup, not whatever the registry says
right now. The fix that actually stuck was to stop relying on the
environment variable at all: `ocr_pipeline.py` now also checks the
actual common install locations directly, so the app works the same
regardless of which terminal happens to launch it.

**13. The confidence badge lied to me too - a different lie than the
score itself.** While tidying up the UI, `st.metric`'s built-in colour
for the confidence label turned out to be misleading in its own way:
Streamlit only knows how to colour a *number* green or red, based on its
sign. Handed a plain word like "Shaky" instead of a number, it could not
tell that from "Looks good" and was on track to colour both the same
reassuring green with an upward arrow. Fixed by building a small colour
badge by hand instead, so the colour always actually matches what the
label says. Small bug, but the exact same shape as the "confidence score
lied to me" lesson above - a UI number can look confident and still be
wrong.

**14. Streamlit Cloud quietly ignored my Python version pin.** The build
kept using Python 3.14 even with a correctly formatted `runtime.txt`
(`python-3.11`) sitting in three different, sensible-looking folders. The
real problem was nothing to do with the file's content or location - the
Python version for an app on Streamlit Cloud gets locked in when the app
is first created, and does not seem to re-read `runtime.txt` on a later
redeploy. The fix was setting the Python version directly in the app's
own dashboard settings, not in the repo at all. `paddlepaddle==3.0.0`
does not ship a wheel for Python 3.14, so this showed up as a dependency
resolution failure that had nothing to do with dependencies.

**15. `packages.txt` only works from the true repository root.**
`requirements.txt` is read from either the repo root or the folder
`app.py` lives in - I already knew that from Day 22. I assumed
`packages.txt` (the file that installs system-level, non-Python
packages like Tesseract) followed the same rule, and put copies in both
places. It does not: Streamlit Cloud only ever read the copy sitting at
the actual top of the repository, one level above even `Day23/` itself,
since this whole internship's repo is one shared git repository, not one
per day. The giveaway was the build log itself - a working `packages.txt`
prints its own "installing from apt" section before the Python
dependencies install, and that section was simply missing every time,
right up until the root copy existed.

**16. `packages.txt` does not support comments the way `requirements.txt`
does.** Once I found the right folder, the very same file broke the
build in a completely different way: `xargs: unmatched single quote`.
`requirements.txt` is a pip format, and pip properly strips out `#`
comment lines. `packages.txt` is apparently just piped straight into
`apt-get install` with no such stripping, so my own explanatory comments
- which happened to contain a couple of apostrophes, like "app's own
folder" - broke the parsing for the *entire file*, not just that line.
The fix was blunt: no comments at all, just bare package names, one per
line.

**17. Three engines, three different builds of OpenCV, all fighting
each other.** Even with `packages.txt` finally being read, the app still
crashed with `ImportError: libGL.so.1: cannot open shared object file`
the moment any engine that touches `cv2` tried to load. None of my own
code imports `cv2` directly - see challenge 2 back on Day 22 - but
EasyOCR, PaddleOCR and docTR all pull it in anyway, each through its own
dependency chain, and the build log showed all three different builds of
OpenCV (`opencv-python`, `opencv-contrib-python`, and
`opencv-python-headless`) getting installed side by side. `libGL.so.1` is
a system graphics library that Streamlit Cloud's minimal container
simply does not have, because it has no display at all - and the
"headless" build of OpenCV exists precisely so you never need that
library in the first place. The actual fix, confirmed against
Streamlit's own documentation and someone else's working four-engine
build of this exact project, was two things together: add
`opencv-python-headless` explicitly to `requirements.txt`, and add
`libgl1` to `packages.txt` as a fallback for whichever library still
insists on the non-headless build regardless.

**18. A fix that looked right and quietly did nothing at all.** Someone
else's working version of this same project configures PaddleOCR with
lightweight "mobile" models instead of the default heavier ones, to fit
Streamlit Cloud's 1GB free-tier RAM limit. I nearly copied that
configuration straight in. Testing it first against the exact PaddleOCR
version this project is pinned to (`2.9.1`, pinned for a real reason -
see challenge 10) showed it would not have worked: those parameter names
belong to a newer version of PaddleOCR's API, and `2.9.1` does not raise
an error for a parameter it does not recognise, it just silently ignores
it and carries on with its own defaults. The fix would have sat in the
code doing precisely nothing, while looking like it was doing something
- which is worse than a crash, because nothing would ever have told me
it was not working. Left as an open item rather than guessed at - see
"Possible improvements" below.

---

## Possible improvements

- **Pick the best result more intelligently.** Auto mode currently picks
  by average confidence, and this write-up shows that score can be
  misleading. Combining it with the line count and the amount of text
  found would be a better judge.
- **Keep the table structure.** Reading an invoice as a flat list loses
  which number belongs to which row. Using the position of each detected
  box to rebuild rows and columns would fix that, and it is what turns
  "text extraction" into something you could actually feed into an
  accounting system.
- **Detect the document type automatically** and choose the preset from
  that, instead of running OCR two or three times to find out.
- **Pull out specific fields.** For an invoice, the useful output is not
  a wall of text - it is `{invoice_no, date, total}`. That is the
  natural next step.
- **Spell-check the output** against a dictionary to catch the small
  mix-ups like `Ul` for `UI`.
- **Make it faster.** Every OCR pass takes 3-9 seconds on a CPU, and
  Auto mode does three of them. Cropping to the region containing text
  before reading would cut that down.
- **Support more languages.** EasyOCR handles 80+, and the app currently
  only asks for English.
- **Actually load-test the free-tier RAM limit.** It deploys and runs
  now, but I never properly tested what happens if one session cycles
  through all four engines back to back - see challenge 18. Getting
  PaddleOCR onto genuinely lighter models on this pinned version, not
  just confirming the newer-API shortcut does not work, is the real
  version of that fix.

---

## What is in this folder

```
Day23/
  .venv/                   <- this app's own environment (all four engines)
  mini_project/            <- the deliverable: the Streamlit web app
    app.py                 <- start here - has the engine selector
    requirements.txt       <- copy of the root one, kept next to app.py
    runtime.txt            <- (Streamlit Cloud needs them right here)
    packages.txt           <- a spare copy, kept just in case (see below)
    .streamlit/config.toml <- the app's colour theme
  coding_practice/          <- the practice script, run from a terminal
    batch_test.py           <- runs all 18 images through 4 presets x every
                               engine (loops whatever ocr_pipeline.ENGINES is)
  ocr_pipeline.py          <- all the OCR logic - shared by both folders
                             above, not owned by either one
  requirements.txt        <- what to install
  runtime.txt             <- which Python version
  packages.txt            <- another spare copy - not the one that actually
                             works (see below)
  HOW_TO_RUN.txt          <- setup commands
  sample_images/          <- 18 test images + the script that generates them
  sample_outputs/         <- the extracted text (per engine + overall best),
                             plus the results tables
```

`ocr_pipeline.py` sits at the `Day23/` root rather than inside either
folder, because it is genuinely shared: both `mini_project/app.py` and
`coding_practice/batch_test.py` add that root folder to `sys.path` at
startup and import the one copy, instead of each keeping its own. I
split the OCR logic away from the interface (`app.py`) on purpose. It
means `batch_test.py` can run the exact same pipeline from the command
line without Streamlit involved - and if the
two had their own copies of the logic, they would have drifted apart the
first time I fixed anything.

**One more thing about `packages.txt`:** this whole internship is one
shared git repository, with every day's folder inside it - so the true
root of the repository is not `Day23/`, it is the folder one level above
that, containing every `Day1`, `Day2`, and so on. Streamlit Cloud only
ever reads `packages.txt` from that actual top level, which is not shown
in the tree above because it sits outside `Day23/` entirely. The two
copies inside `Day23/` shown above are spares I left in place just in
case, not the ones doing any real work - see challenge 15.

---

## The 18 test images

The task asked for at least 15, covering documents, receipts, invoices
and forms. I generated 18 with a script rather than using real photos,
for one main reason: **I know exactly what every image says.** So when
the OCR gets a word wrong, I can be certain it is wrong instead of
squinting at a blurry photo and guessing.

| Type | Images | What varies |
|---|---|---|
| Documents | `document_clean`, `document_small_text`, `document_dim_lighting`, `document_rotated`, `document_low_resolution` | Font size, lighting, tilt, resolution |
| Receipts | `receipt_clean`, `receipt_faded`, `receipt_noisy_lowlight`, `receipt_tiny_text` | Brightness, sensor noise, font size |
| Invoices | `invoice_clean`, `invoice_scanned_tint`, `invoice_skewed` | Paper tint, tilt |
| Forms | `form_enrolment`, `form_with_checkboxes`, `form_low_contrast` | Checkboxes, faded ink |
| Others | `book_page`, `handwritten_note`, `id_card` | Serif text, script font, card layout |

**One honest caveat:** the "handwritten" note is a script *font* with a
bit of random wobble added, not real handwriting. It looks the part, but
real handwriting is much harder for OCR - everyone's letters are
different, and they join up. Its 0.85 score is optimistic and should not
be read as "this app handles handwriting".

---

## How to run it

Full details are in `HOW_TO_RUN.txt`. The short version:

```bash
cd D:\GitHub\ML-Bench\Day23
.venv\Scripts\activate
streamlit run mini_project\app.py
```

This app has its own virtual environment, separate from Day 22's -
adding PaddleOCR meant adding a second numpy constraint (`numpy<2.0`)
that Day 22's already-working EasyOCR-only venv did not have, so a
fresh environment avoided any risk of breaking that one. Starting fresh
instead (`.venv` missing, or on another machine):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Requirements met

- [x] Accepts an image upload
- [x] Applies preprocessing (grayscale, denoising, contrast, thresholding)
- [x] Extracts text using **EasyOCR, PaddleOCR, Tesseract and docTR**,
      selectable in the app
- [x] Displays the original image and the extracted text side by side
- [x] Lets the user download the text as a `.txt` file
- [x] Handles different document types (18 images across 6 categories)
- [x] Tested on more than 15 images (18, each through 4 presets)
- [x] Code organised into separate functions
- [x] Built as a Streamlit app
- [x] Deployed to Streamlit Cloud - live at
      [ocrtoolapp.streamlit.app](https://ocrtoolapp.streamlit.app) (took a
      real fight to get there - see challenges 14 to 18)
- [x] Public URL added to this README - see above

---

