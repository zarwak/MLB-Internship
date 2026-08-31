# Day 30 - Multi-Object Tracking (YOLOv8 + ByteTrack / BoT-SORT)

Days 27-29 in this repo did **detection**: find objects in a frame, fresh,
every frame, with no memory of what came before. Day 30 adds **tracking**:
give every detected object a persistent identity and follow it across the
whole video - including while it's briefly hidden behind something else, or
crossing paths with another object.

**Live demo:** _(recording + hosted URL added after deployment - see
`HOW_TO_RUN.txt`)_

![DEMO HERE](demo_video_summarizer.gif)

**APP LINK:** _[CLICK HERE TO GO TO THE APP](https://object-tracking.streamlit.app/)_


## What is object tracking?

Object **detection** answers "what's in this frame, and where?" -
independently, frame by frame. Run it on frame 1, then frame 2, and nothing
connects the person you found in frame 1 to the person you find in frame 2,
even if it's obviously the same person one step to the left. Every frame is
a clean slate.

Object **tracking** adds identity and continuity on top of detection: each
object gets a track ID the first time it's seen, and a tracking algorithm
tries to keep that same ID attached to that same object in every later
frame - by predicting roughly where it should be next (motion) and matching
that prediction against the new frame's detections. That's what makes
questions like "how many *different* people walked through this video?" or
"did this car ever stop moving?" answerable at all - a per-frame detector
has no concept of "this is the same one as before" to answer them with.

**Detection vs. tracking, concretely, from this project's own output:**
`pedestrians_cctv.mp4` (15s, 150 frames) produces **19 unique tracked
objects** (`person=15, truck=1, car=3` - see
`sample_outputs/tracking_results.md`). A per-frame detector run on the same
clip would instead report a detection count *per frame* (something like
"3-5 people found" on nearly every one of the 150 frames) with no way to
say whether frame 40's "3 people" are the same 3 people as frame 41's -
tracking is precisely the layer that answers that question, by keeping
`#7`, `#9`, `#13` (etc.) attached to the same three people the whole clip
(see the app's "Tracked Objects" table, which lists each ID's total
frames-visible - `#9` above is visible in 139 of the 150 frames, i.e. one
person, tracked almost the entire clip, not 139 separate detections).


## Which tracking algorithm did we use?

Both **ByteTrack** and **BoT-SORT** are wired up (Ultralytics ships both
built in, selectable via `model.track(tracker=...)`) - the mini-project
brief specifically asks for a tracker dropdown, so `app.py` exposes both
rather than picking one. **ByteTrack is the default.**

- **ByteTrack**: associates detections to existing tracks using a Kalman
  filter (motion prediction) + IoU matching - and, its key idea, does this
  in *two* passes: first match high-confidence boxes, then make a *second*
  pass matching the *remaining* tracks against *low*-confidence boxes
  before giving up on them. That second pass is specifically what recovers
  a track through a brief occlusion or motion blur (a real detection at low
  confidence beats no detection at all), without needing any appearance
  model.
- **BoT-SORT**: same motion-based association as above, optionally
  strengthened with an appearance (ReID) feature embedding for matching
  look-alike objects. Ultralytics' default `botsort.yaml` in this version
  ships with ReID **off** (`with_reid: False`) - so out of the box it's
  close to ByteTrack's motion-only approach plus a camera-motion
  compensation step, rather than the full appearance-based tracker BoT-SORT
  is capable of. That's measurable in our own results below: unique-ID
  counts and suspected-switch counts are nearly identical between the two
  trackers on every one of our 5 clips - the ReID advantage BoT-SORT is
  known for in the literature doesn't show up here because it isn't turned
  on.


## Real challenges faced (measured, not guessed)

All numbers below are the actual output of
`coding_practice/01_track_videos.py` and
`coding_practice/02_id_consistency_check.py`
(`sample_outputs/tracking_results.md`, `sample_outputs/id_consistency.md`) -
run once against all 5 videos x both trackers before writing this section.

- **Small, fast-moving objects churn IDs badly.** `sports_soccer.mp4` has
  exactly one ball in frame, on screen almost the entire 14s clip - but
  both trackers report **21 unique "sports ball" IDs** and **17 suspected
  ID switches**, by far the worst of any clip. The ball moves fast enough
  to blur, gets fully occluded by players' feet on nearly every touch, and
  YOLO's per-frame confidence on it swings widely - so the tracker
  regularly loses it and picks it back up as what looks like a brand-new
  object a few frames later. This is the single clearest real limitation
  we hit: motion-based tracking is much better suited to people/vehicles
  (large, slower, rarely fully hidden) than to a small ball changing
  direction every touch.
- **A generic COCO model produces spurious secondary-object classes in
  busy scenes.** The Tokyo crosswalk clip tracks `handbag=27` and
  `backpack=9` "objects" alongside `person=125-126`; the mall clip tracks
  `tv=5` and `umbrella=1`. Some of these are real (people do carry bags),
  but the counts are implausibly high for what's actually a fairly small
  crowd on screen at once - almost certainly store-window displays and
  background clutter getting misclassified and then *tracked* as if they
  were real, distinct, moving objects. Raising the confidence threshold in
  the app's sidebar cuts these down, but trades off against losing real
  low-confidence person detections in the same crowd - there's no
  threshold that eliminates one without risking the other.
- **BoT-SORT's default config doesn't beat ByteTrack here.** Suspected-
  switch counts are nearly identical per clip between the two trackers
  (e.g. `sports_soccer.mp4`: 17 vs. 17, exactly the same; `pedestrians_mall.mp4`:
  7 vs. 7) - see "Which tracking algorithm" above for why: Ultralytics'
  default `botsort.yaml` ships with ReID off, so its practical behavior
  here is close to ByteTrack's.
- **CPU-only tracking is noticeably slower than CPU-only detection.**
  These clips take 18-62s to process on this machine's CPU (no GPU) - i.e.
  roughly 3-4x slower than real-time, on top of what Day29 already measured
  for detection alone. The tracker's own association step (Kalman
  filtering, IoU/ReID matching) runs every frame in addition to YOLO
  inference, and that adds up. `app.py`'s 300-frame processing cap exists
  specifically because of this.


## The 5 sample videos

We deliberately used **only 2 of the brief's 4 suggested categories**
(people walking, sports) instead of all 4. Reasoning: the brief's actual
requirement is that IDs **not** change when objects cross paths or occlude
each other - that only gets meaningfully exercised by scenes with frequent
crossing/occlusion. Parking-lot footage (mostly parked cars) and
free-flowing highway traffic barely exercise a tracker at all; pedestrian
crowds and sports footage do, constantly - see the crosswalk/soccer numbers
above.

| File | Category | Scene | Source | Duration |
|---|---|---|---|---|
| `pedestrians_cctv.mp4` | People walking | CCTV, multiple pedestrians crossing | OpenCV `samples/data/vtest.avi` (BSD test data), trimmed | 15.0s |
| `pedestrians_crosswalk.mp4` | People walking | Busy Tokyo-style crosswalk, dense crowd | Pexels (Musa Ortac) | 18.7s |
| `pedestrians_mall.mp4` | People walking | Shopping mall, crowd walking | Pexels (khanhhoangminh) | 18.6s |
| `sports_soccer.mp4` | Sports | Soccer players training | Pexels (Tima Miroshnichenko) | 14.1s |
| `sports_basketball.mp4` | Sports | Basketball players on court | Pexels (Tima Miroshnichenko) | 15.0s |

Pexels clips are under the Pexels License (free for commercial/personal
use, no attribution required); the OpenCV clip is BSD-licensed sample test
data - both fine to commit into a public repo. Fetched by
`download_samples.py`.


## Full results

| Video | Tracker | Frames | Unique objects | Per-class | Time | Suspected ID switches |
|---|---|---|---|---|---|---|
| pedestrians_cctv.mp4 | ByteTrack | 150 | 19 | person=15, truck=1, car=3 | 20.9s | 2 |
| pedestrians_cctv.mp4 | BoT-SORT | 150 | 19 | truck=1, car=3, person=15 | 18.6s | 3 |
| pedestrians_crosswalk.mp4 | ByteTrack | 468 | 190 | car=21, person=126, traffic light=4, backpack=9, handbag=27, bicycle=2, truck=1 | 61.7s | 19 |
| pedestrians_crosswalk.mp4 | BoT-SORT | 468 | 189 | person=125, car=21, traffic light=4, backpack=9, handbag=27, bicycle=2, truck=1 | 57.2s | 18 |
| pedestrians_mall.mp4 | ByteTrack | 446 | 77 | person=66, handbag=4, tv=5, cell phone=1, umbrella=1 | 52.0s | 7 |
| pedestrians_mall.mp4 | BoT-SORT | 446 | 77 | person=66, handbag=4, tv=5, cell phone=1, umbrella=1 | 52.3s | 7 |
| sports_soccer.mp4 | ByteTrack | 352 | 64 | sports ball=21, person=42, baseball glove=1 | 48.0s | 17 |
| sports_soccer.mp4 | BoT-SORT | 352 | 64 | person=42, sports ball=21, baseball glove=1 | 52.9s | 17 |
| sports_basketball.mp4 | ByteTrack | 375 | 12 | person=2, sports ball=8, tennis racket=1, car=1 | 51.8s | 1 |
| sports_basketball.mp4 | BoT-SORT | 375 | 12 | person=2, sports ball=8, tennis racket=1, car=1 | 43.2s | 1 |

"Suspected ID switches" is a heuristic (see `coding_practice/02_id_consistency_check.py`
docstring) - **not** a formal MOTA/IDF1 score, since no ground-truth ID
annotations exist for these clips. It flags a new track ID appearing within
5 frames and within 8% of the frame's shorter side of where a same-class ID
was just lost - i.e. "this looks like the tracker relabeling an object it
briefly lost," which is the best proxy available without hand-annotating
every frame.

Annotated output videos for all 10 runs above are in `sample_outputs/`.


## The Streamlit app - Smart Object Tracking System

`app.py` - pick a sample video (or upload your own), choose ByteTrack or
BoT-SORT, tune confidence/IoU, run tracking, and get back: the annotated
video (ID-colored box + short motion trail + live unique-count badge per
object), 4 summary metrics, a per-track-ID table (class, best confidence,
frames visible), a per-class bar chart, and a download button. Verified
working end-to-end in-browser with both trackers against the dense-crowd
crosswalk clip - see commit history for the exact numbers observed.

Run locally: `streamlit run app.py` (see `HOW_TO_RUN.txt`).


## Project layout

```
Day30/
├── app.py                          # Streamlit app - UI only
├── tracking.py                     # core module: model load, .track() wrapper,
│                                    #   ID-colored drawing, per-video/per-frame stats
├── download_samples.py             # fetches the 5 videos above
├── coding_practice/
│   ├── 01_track_videos.py          # runs tracking on all 5 videos x both trackers,
│   │                                #   saves annotated outputs, prints unique-ID counts
│   └── 02_id_consistency_check.py  # heuristic ID-switch counter (see "Full results")
├── sample_videos/                  # 5 short clips (committed)
├── sample_outputs/                 # 10 annotated output videos + 2 results tables
├── yolov8n.pt                      # pretrained COCO weights (gitignored - auto-downloaded
│                                    #   by ultralytics if missing, see .gitignore)
├── requirements.txt / runtime.txt
├── .streamlit/config.toml
└── HOW_TO_RUN.txt
```


## Setup & run

See [`HOW_TO_RUN.txt`](HOW_TO_RUN.txt) for full setup and run instructions.
