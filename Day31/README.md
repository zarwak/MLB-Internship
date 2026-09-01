# Day 31 - Vehicle Counting System (YOLOv8n + ByteTrack)

Day 30 in this repo added **tracking** - a persistent ID per object across a
video. Day 31 builds on that directly: **counting** is tracking plus one
more idea, a line the camera is watching, and the rule "count each track the
moment it crosses that line."

**Live demo:** _(recording + hosted URL added after deployment - see
`HOW_TO_RUN.txt`)_


**APP LINK:** _[add after deploying - see HOW_TO_RUN.txt]_
[Click here to go to the app](https://vehicle-counting-system.streamlit.app/)


## What is object counting?

Detection answers "what's here, and where?" for one frame. Tracking (Day 30)
adds "is this the same object as last frame?" across a whole video.
**Counting** adds a third question on top: "has this object passed a
specific point yet?" - answered once per object, not once per frame.

Three ideas make that answer possible:

- **The counting line** - a line the app draws across the frame (horizontal
  or vertical, position adjustable). Every tracked vehicle's centroid is
  checked against it every frame: which side is it on *now*, and which side
  was it on *last* frame? A side change is a crossing.
- **Region of interest (ROI)** - an optional rectangle that counting is
  restricted to. A vehicle detected outside it is still drawn, just never
  counted - useful for ignoring a parking lane, a sidewalk, or the opposite
  carriageway that happens to be visible in frame.
- **Avoiding duplicate counts** - the hard part, and the brief's own
  question, answered in detail below.

Real-world uses: traffic-volume studies, smart-signal timing, parking-lot
occupancy, tollway/bridge throughput, pedestrian-safety audits at
crosswalks - anywhere the question is "how many things passed through
here," not just "what's in view right now."


## How tracking IDs prevent duplicate counting

A vehicle's bounding box is usually 40-150 pixels tall. A counting line is
one pixel row. A car crossing that line doesn't touch it for one frame - it
straddles it for several consecutive frames while its box passes through.
A detector with **no memory** (re-detects fresh every frame, Days 27-29's
approach) has no way to know "the box touching the line in frame 112 is the
same physical car as the box that touched it in frame 111" - it would either
need fragile frame-to-frame heuristics bolted on, or it would risk counting
the same crossing multiple times.

Tracking (`model.track(..., persist=True)`, same mechanism as Day30) solves
this at the source: every vehicle gets **one ID** for its entire time in
frame. `counting.py`'s `CountState` keeps two small pieces of state per ID -
which side of the line it was on last frame, and whether it's already been
counted (`counted_ids: set[int]`) - so the crossing logic becomes: *"side
changed, and this ID isn't in `counted_ids` yet → count it, add the ID to
the set, never count that ID again."* One physical vehicle, seen crossing
the line's pixel-row across 4-6 consecutive frames, is one entry in
`counted_ids` - not four to six.

Concretely, from this project's own output (`sample_outputs/counting_results.md`):
`highway_many_cars.mp4` (466 frames, ~15.5s) has vehicles visible near the
counting line for several frames each crossing, yet the video reports
exactly **28** vehicles counted - the same 28 whether you look at
`counted_ids`'s size or the per-class breakdown (`car=24, bus=1, truck=3`).
A per-frame detector with no ID concept, counting "a box touches the line,"
would have overcounted every one of those 28 by several x.


## Which vehicle types were counted

**Car, motorcycle, bus, truck** - COCO class ids 2, 3, 5, 7. `yolov8n.pt` is
COCO-pretrained, so all four are already classes the model knows; unlike
Day 29, no custom training was needed for this project. The mini-project
app and `coding_practice/02_batch_count_all_videos.py` count all four;
`coding_practice/01_count_vehicles.py` (the brief's literal "Coding
Practice" checklist) restricts detection to **car + truck only**, via
`model.track(..., classes=[2, 7])`.


## Real challenges faced (measured, not guessed)

All numbers below are the actual output of
`coding_practice/01_count_vehicles.py` and
`coding_practice/02_batch_count_all_videos.py`
(`sample_outputs/counting_results.md`) - run once against all 5 videos
before writing this section.

- **A straight-down aerial camera angle breaks a COCO-pretrained
  detector completely.** Two nadir/overhead traffic clips were the first
  videos picked for this project - and YOLOv8n detected **zero** vehicles on
  either, at every confidence threshold tested down to 0.1 (see
  `download_samples.py`'s docstring for the exact test). COCO's
  car/truck/bus/motorcycle training images are essentially all ground-level
  or oblique photos; a car photographed from directly overhead just doesn't
  look like anything in that training distribution. Both clips were dropped
  and replaced with oblique/eye-level highway footage, which detects
  normally. This is a real limitation of using a general-purpose pretrained
  model rather than a mistake in confidence tuning - the same class of issue
  Day29 hit with domain-specific imagery, here caused by camera angle
  instead of subject matter.
- **Restricting the class list changes which class wins for ambiguous
  vehicles, not just which ones get suppressed.** Running
  `highway_many_cars.mp4` through the full 4-class model
  (`02_batch_count_all_videos.py`) gives `car=24, bus=1, truck=3` (28
  total); running the *same clip* through the car+truck-only model
  (`01_count_vehicles.py`, `classes=[2, 7]`) gives `car=24, truck=4` (also
  28 total) - one vehicle the 4-class model called "bus" gets relabeled
  "truck" the moment "bus" isn't an option any more. `classes=` in
  Ultralytics restricts which labels the model is *allowed to output*, so a
  genuinely borderline box (a van-sized minibus, in this case) falls to
  whichever permitted class scored second-best - it doesn't just get
  dropped. Worth knowing before trusting a class-restricted count as
  ground truth for the excluded classes' absence.
- **A "count" measures line crossings, not full transits - stopped
  traffic still produces real crossings.** `urban_intersection_motorcycles.mp4`
  is a dashcam clip of traffic queued at a red light - vehicles barely move
  for the whole 19s clip. It still produced **21** counted crossings
  (`car=1, motorcycle=20`, split `down=12, up=9`) - not from vehicles
  driving through the scene, but from the queue creeping forward a few
  pixels at a time as more motorcycles arrive and everyone shuffles up,
  which is enough motion for tightly-packed centroids to cross a
  mid-frame line one at a time. The count is technically accurate (every
  entry really did cross the line, once), but it's a good reminder that
  "N crossings" answers exactly the question the line's placement asks -
  it doesn't mean N vehicles drove all the way through the intersection.
- **CPU-only inference cost scales with frame count, not clip duration -
  and frame rate varies a lot between real source footage.**
  `urban_intersection_motorcycles.mp4` (19.0s, but shot at **60fps** =
  1139 frames) took **173.3s** to process - more than double every other
  clip, despite being one of the shorter ones by duration. The other four
  clips (25-30fps, 238-466 frames) all took 47-90s. Per-frame cost is
  actually consistent across clips (~0.15-0.2s/frame on this CPU); it's
  purely frame *count* that drives wall-clock time. This is why `app.py`
  caps processing at 300 frames regardless of the source clip's fps.


## The 5 sample videos

All 5 are real Pexels footage (Pexels License - free for commercial/personal
use, no attribution required, safe to commit into a public repo), fetched at
their SD 960x540 encode. Picked for a mix of oblique/eye-level camera angles
(the only kind that actually detects, per "Challenges" above) and enough
variety to exercise all 4 vehicle classes - the intersection clip is the
only one with motorcycles, `highway_cars_buses.mp4` is the only one with a
clean bus example.

| File | Scene | Duration | Notes |
|---|---|---|---|
| `highway_evening.mp4` | Highway, oblique overpass view | 14.1s | Evening traffic, includes an oncoming BRT bus lane (the source of this project's only "down"-direction outlier on an otherwise one-way clip) |
| `highway_cars_buses.mp4` | Highway, side angle | 8.0s | Dense mixed cars + buses |
| `highway_fast_paced.mp4` | Highway, close oblique angle | 17.5s | Fast-moving, motion-blurred - the hardest detection case kept in the set |
| `highway_many_cars.mp4` | Highway, elevated overpass view | 15.5s | Dense multi-lane traffic; used for `coding_practice/01_count_vehicles.py` |
| `urban_intersection_motorcycles.mp4` | Urban intersection, dashcam | 19.0s, 60fps | Static queued traffic at a red light; only clip with motorcycles |

Fetched by `download_samples.py`, which also documents (in its docstring)
the two aerial clips that were tried and dropped - see "Challenges" above.


## Full results

| Video | Frames | Total counted | Per-class | Per-direction | Time |
|---|---|---|---|---|---|
| highway_evening.mp4 | 352 | 13 | car=13 | down=2, up=11 | 65.3s |
| highway_cars_buses.mp4 | 238 | 10 | bus=2, car=8 | up=10 | 46.5s |
| highway_fast_paced.mp4 | 437 | 13 | car=13 | down=13 | 71.4s |
| highway_many_cars.mp4 | 466 | 28 | bus=1, car=24, truck=3 | up=28 | 83.7s |
| urban_intersection_motorcycles.mp4 | 1139 | 21 | car=1, motorcycle=20 | down=12, up=9 | 173.3s |

All 5 used a horizontal counting line, position tuned per clip to sit across
the visible lanes (see `coding_practice/02_batch_count_all_videos.py`).
`coding_practice/01_count_vehicles.py` (car+truck only, same line) counted
**28** vehicles on `highway_many_cars.mp4` too - `car=24, truck=4` - see
"Challenges" above for why the bus/truck split differs from the full-class
run despite the same total.

Annotated output videos for all 6 runs above are in `sample_outputs/`.


## The Streamlit app - Smart Vehicle Counting System

`app.py` - pick a sample video (or upload your own), position a counting
line (horizontal or vertical, adjustable), optionally restrict counting to
a region of interest, tune confidence/IoU, run counting, and get back: the
annotated video (ID-colored boxes + counting line + live running-total
badge), 4 summary metrics, a per-class bar chart, a per-direction bar
chart, and a download button. Tracker is fixed to ByteTrack (Day30 already
covers the ByteTrack-vs-BoT-SORT comparison; this project's variable is the
counting logic, not the tracker).

Run locally: `streamlit run app.py` (see `HOW_TO_RUN.txt`).


## Project layout

```
Day31/
├── app.py                            # Streamlit app - UI only
├── counting.py                       # core module: model load, .track() wrapper,
│                                      #   counting line / ROI, crossing detection,
│                                      #   drawing, per-video stats
├── download_samples.py               # fetches the 5 videos above
├── coding_practice/
│   ├── 01_count_vehicles.py          # brief's literal checklist: car+truck only,
│   │                                  #   one video, line, count, save
│   └── 02_batch_count_all_videos.py  # full 4-class counting on all 5 videos,
│                                      #   saves annotated outputs, writes results table
├── sample_videos/                    # 5 short clips (committed)
├── sample_outputs/                   # 6 annotated output videos + results table
├── yolov8n.pt                        # pretrained COCO weights (gitignored - auto-downloaded
│                                      #   by ultralytics if missing, see .gitignore)
├── requirements.txt / runtime.txt
├── .streamlit/config.toml
└── HOW_TO_RUN.txt
```


## Setup & run

See [`HOW_TO_RUN.txt`](HOW_TO_RUN.txt) for full setup and run instructions.
