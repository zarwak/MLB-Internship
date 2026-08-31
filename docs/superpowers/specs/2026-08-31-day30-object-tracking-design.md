# Day 30 - Smart Object Tracking System (Design)

## Goal

Extend Day 27/29's YOLO detection work to **multi-object tracking**: assign a
stable ID to each detected object and keep that ID consistent as the object
moves, gets occluded by another object, or crosses paths with one - across a
handful of short real-world videos - then wrap it in a Streamlit app.

## Decisions made during brainstorming

- **Model**: pretrained `yolov8n.pt` (COCO). Rejected Day29's custom
  road-damage model - potholes/cracks don't move in a scene, so a dashcam
  clip of them is a weak demo of ID persistence through crossing/occlusion,
  which is exactly what this task needs to show.
- **Trackers**: support both Ultralytics built-in trackers - **ByteTrack**
  (`bytetrack.yaml`, default) and **BoT-SORT** (`botsort.yaml`) - selectable
  via a dropdown, per the brief's mini-project spec.
- **Video categories**: **People Walking + Sports** only, not all 4 example
  categories from the brief. Reasoning: the brief's actual requirement is
  "IDs shouldn't change when objects cross" - that only gets meaningfully
  exercised by scenes with frequent crossing/occlusion. Parking-lot footage
  (mostly parked cars) and free-flowing highway traffic barely exercise a
  tracker at all; pedestrian and sports scenes do, constantly.
- **Deployment**: same pattern as Day27/Day29 - build and verify locally,
  leave README/HOW_TO_RUN.txt placeholders for the live Streamlit Community
  Cloud URL, user deploys and fills it in afterward. No live deployment
  attempted in this session.

## Dataset - 5 sample videos (`sample_videos/`, fetched by `download_samples.py`)

| File | Category | Scene | Source | Duration |
|---|---|---|---|---|
| `pedestrians_cctv.mp4` | People walking | CCTV, multiple pedestrians crossing | OpenCV `samples/data/vtest.avi` (BSD test data), trimmed | 15s |
| `pedestrians_crosswalk.mp4` | People walking | Busy Tokyo-style crosswalk, dense crowd | Pexels (Musa Ortac) | 18.7s |
| `pedestrians_mall.mp4` | People walking | Shopping mall, crowd walking | Pexels (khanhhoangminh) | 18.6s |
| `sports_soccer.mp4` | Sports | Soccer players training | Pexels (Tima Miroshnichenko) | 14.1s |
| `sports_basketball.mp4` | Sports | Basketball players on court | Pexels (Tima Miroshnichenko) | 15.0s |

All fetched at Pexels' SD 960x540 encode (their own "Free Download" size
picker), Pexels License (free, no attribution required) / BSD (OpenCV).

## Architecture (mirrors Day27/Day29's split)

```
Day30/
├── app.py                          # Streamlit app - UI only
├── tracking.py                     # core module: model load, .track() wrapper,
│                                    #   ID-colored drawing, per-video/per-frame stats
├── download_samples.py             # DONE - fetches the 5 videos above
├── coding_practice/
│   ├── 01_track_videos.py          # runs tracking on all 5 videos x both trackers,
│   │                                #   saves annotated outputs, prints unique-ID counts
│   └── 02_id_consistency_check.py  # objective check: counts ID switches (a track
│                                    #   dying and a new ID appearing where the old
│                                    #   one should have continued) per video/tracker
├── sample_videos/                  # DONE - 5 clips, committed
├── sample_outputs/                 # annotated output videos + results table
├── yolov8n.pt                      # pretrained COCO weights (copied from Day27)
├── requirements.txt / runtime.txt
├── README.md / HOW_TO_RUN.txt
```

### `tracking.py`

- `load_model(name="yolov8n.pt")` - cached load, same fuse-under-lock pattern
  as Day29's `detection.py` (Streamlit can run concurrent sessions against
  one cached model).
- `TrackedObject` dataclass: `track_id, class_id, class_name, confidence, box`.
- `track_video(model, in_path, out_path, tracker, conf, iou, max_frames, progress_cb)`
  - calls `model.track(frame, persist=True, tracker=tracker, conf=conf, iou=iou)`
    frame-by-frame (persist=True is what keeps the tracker's internal state -
    and therefore IDs - alive across frames; without it every frame would
    restart tracking from scratch and every object would get a new ID
    immediately).
  - draws, per tracked box: a color keyed off **track ID** (not class ID) so
    one object keeps one color for its whole appearance in the clip, a
    `#{id} {class} {conf:.2f}` label, and a short fading trail of the
    object's last ~15 centroid positions (visually shows the tracker
    following motion, not just re-detecting per frame).
  - returns per-frame track lists + aggregate stats: total unique IDs seen,
    per-class unique-ID counts (a car that's tracked for 200 frames counts
    once, not 200 times - this is the key difference from Day29's
    per-frame detection counting).
- Video I/O reuses Day29's `imageio` + libx264 writer (browser-playable mp4;
  `cv2.VideoWriter` has no licensed H.264 encoder in this environment).

### `app.py` - "Smart Object Tracking System"

- Sidebar: source (sample dropdown / upload your own), **tracker dropdown**
  (ByteTrack / BoT-SORT), confidence slider, IoU slider.
- Main: run button -> progress bar -> annotated video player, metrics row
  (unique objects tracked, per-class breakdown, frames processed, processing
  time), a small table of every tracked object (ID, class, best confidence
  seen, frames visible), download button for the annotated mp4.
- Same `MAX_VIDEO_FRAMES` cap as Day29, so an uploaded video can't stall the
  free hosted tier.

### `coding_practice/01_track_videos.py`

Runs `track_video()` over all 5 sample videos x both trackers (10 runs
total), saves each annotated output into `sample_outputs/`, and prints a
summary table (video, tracker, unique IDs, processing time) - this is the
"Coding Practice" deliverable ("run tracking on at least 5 videos, display
unique IDs, count unique objects, save output videos").

### `coding_practice/02_id_consistency_check.py`

Objective, automatic check for the brief's "Make Sure" requirements (IDs
stay consistent, don't change on crossing). Approach: for each video/tracker
run, count **ID switches** - defined as: a track ID that stops appearing for
`>N` frames while a *new* ID appears in roughly the same location/class in
that gap (a heuristic proxy for "the tracker lost this object and started
calling it something else"). Reports this count per video/tracker as the
closest thing to ground-truth ID-stability evidence we can get without
manual annotation (no ground-truth ID labels exist for these clips, so this
is a heuristic, not a formal MOTA/IDF1 score - documented as such in the
README).

## README.md content plan

Answers the brief's required questions: what is object tracking, detection
vs. tracking, which algorithm we used (and why we support both), real
challenges hit while building this (e.g. ID switches under heavy occlusion
in the crosswalk/mall clips, tuning confidence threshold to avoid
flickering low-confidence boxes creating spurious short-lived IDs, CPU
video-processing speed) - written after actually running the trackers and
observing real behavior, not invented ahead of time (matching Day29's
"Challenges" section style, which documents real measured issues).

## Out of scope

- Live deployment (placeholder, per decision above).
- Screen recording (can't be produced by this session - README/HOW_TO_RUN.txt
  get a placeholder note, same as Day29's "Live demo" placeholder).
- A formal MOTA/IDF1 tracking-accuracy benchmark (no ground-truth ID
  annotations exist for these clips; the heuristic ID-switch check above is
  the practical substitute, documented as such).
