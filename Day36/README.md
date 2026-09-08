# Day 36 - Traffic Violation Detection System (YOLOv8n + ByteTrack)

Days 30-32 in this repo built up a stack: **tracking** (Day 30, a persistent
ID per object), **counting** (Day 31, tracking plus a line and a crossing
rule), **occupancy** (Day 32, tracking plus a region and an overlap rule).
Day 36 builds on the same tracking core and adds two more independent
rules on top: is this vehicle heading the **wrong way**, and has it entered
a **restricted zone**.

**Live demo:** _(recording + hosted URL added after deployment - see
`HOW_TO_RUN.txt`)_

**APP LINK:** _[add after deploying - see HOW_TO_RUN.txt]_


## How vehicle tracking works

`tracker.py` wraps Ultralytics' built-in tracker (`model.track(...,
persist=True)`, ByteTrack) - the same call Day30/31/32 use. `persist=True`
keeps the tracker's internal state alive across every frame of one video,
so a physical vehicle keeps one integer ID for as long as it stays visible,
instead of getting a brand-new ID every frame the way plain per-frame
detection would.

The one thing Day36 needs that earlier days didn't is a short **position
history** per track ID (`MotionHistory` in `tracker.py`): a rolling deque of
the last ~20 centroids. Day31 only ever needed "which side of a line is
this point on right now" - a single frame is enough for that. Knowing which
*direction* a vehicle is heading needs several frames of memory.

## How direction is calculated

`MotionHistory.direction_vector(track_id)` takes the **net** displacement
across the whole history window - oldest centroid to newest - not a
frame-to-frame delta. A single-frame delta is dominated by detection
jitter: a bounding box's exact edge can wobble a couple of pixels between
frames even for a car that isn't moving. The net vector over ~20 frames
reflects real motion instead. If that net displacement is under 6 pixels
(`MIN_DISPLACEMENT_PX`), the vehicle is treated as "not enough motion to
know" rather than given a noisy, near-random heading - this matters for
vehicles that just entered frame, or are stopped at a light.

The road's "normal" direction (`RoadDirection`) is one of 8 compass
directions - up/down/left/right and the 4 diagonals - stored as a unit
vector. 8-way rather than a simple horizontal/vertical split because a road
in an oblique aerial or elevated shot is very often not purely horizontal
or vertical - two of the highway-drone candidates dropped for this project
(see "Real challenges faced") ran diagonally across the frame, which is
exactly the case a horizontal/vertical-only scheme can't represent. The
angle between a vehicle's motion vector and the road's normal vector, via
the dot product, is the number everything else is judged against.

## How wrong-way vehicles are identified

A vehicle is wrong-way this frame if the angle between its motion vector
and the road's normal direction is at least `angle_threshold_deg` (default
130°) - well past 90° so a vehicle merely changing lanes or turning across
the road isn't misread as driving straight backwards. 130° (not a more
obvious 90-100°) is itself a measured choice - see "Real challenges faced"
for the false-positive rate a looser threshold produced on real footage.

That single-frame test is then **debounced**, the same idea Day32 used for
occupancy flips: each track keeps a rolling window of the last 14
wrong/right frames (`confirm_window`), and the violation is only confirmed
once at least 80% of that window agrees (`confirm_ratio`). A vehicle
braking hard, one bad detection, or a momentary jitter isn't enough on its
own to trigger a violation.

Once confirmed, a vehicle is added to a **sticky** set (`wrong_way_ids`) and
stays flagged - highlighted red - for the rest of the clip, and exactly one
`ViolationEvent` is recorded for it, ever. It is never re-evaluated.

An optional `direction_roi` (a rectangle, same fraction-of-frame convention
as the restricted zone) further restricts which vehicles the wrong-way rule
even looks at. This exists for one specific, real problem: on a **divided**
road, the two carriageways carry legitimately opposite traffic. Without a
way to say "only check vehicles in my carriageway," the entire oncoming
side would read as a mass wrong-way violation. `bangkok_boulevard`'s preset
scopes the check to the near carriageway (left of the physical median) for
exactly this reason - see "Real challenges faced" below for where this
does and doesn't fully solve the problem.

## How restricted zones are defined

`RegionBox` is an axis-aligned rectangle in **fraction-of-frame**
coordinates (0-1), the same resolution-independent convention Day31 used
for its `ROI` and Day32 used for its parking spaces - a zone calibrated
once looks right no matter what resolution the video is decoded or resized
at. A rectangle rather than a free polygon: it covers "mark a lane, a
shoulder, a median, a crosswalk" perfectly well and needs only 2 sliders
per axis in the UI, at the cost of not hugging a heavily oblique lane as
tightly as a full quadrilateral would (Day32 used quads for exactly that
reason on very oblique parking spaces; this project's zones are looser by
comparison and that trade felt worth it for a simpler control surface).

A violation is recorded on the **outside-to-inside transition** of a
vehicle's centroid, not on "the vehicle is currently inside" as an ongoing
state - see the next section for why that distinction matters.

## How duplicate violations are avoided

Two different mechanisms, one per rule, because the two rules have
different "shape":

- **Wrong-way** is a *sustained state* (a vehicle either is or isn't
  currently heading the wrong way), so it's deduplicated by a **sticky
  set** - `wrong_way_ids`. Once a track ID is in that set, `update()` skips
  it entirely; it can never trigger a second event.
- **Restricted zone** is a *transition* (entering matters, not idling
  inside), so it's deduplicated by **edge-triggering** - comparing this
  frame's inside/outside state to last frame's per track ID
  (`_zone_inside_prev`), the same technique Day31 used for line crossings.
  A vehicle that idles inside the zone for 200 frames counts once; a
  vehicle that leaves and comes back counts again, because that is a
  second, distinct entry.

Both are keyed by **track ID**, not by frame - the same reason Day31's
counting doesn't 3x-count one truck whose box is taller than the counting
line: a per-frame check with no memory would see the truck "inside" the
zone (or "wrong-way") on 30 consecutive frames and could record 30
violations for one real event.

## How violation statistics are generated

`ViolationState` (in `traffic_violation.py`) is the single source of truth,
updated one frame at a time and read from in three places (the CLI, the
batch script, and both Streamlit variants) so the numbers never disagree:

- **Total vehicles** / **vehicle type counts** - `id_to_class`, a dict of
  every track ID ever seen mapped to its class name, updated every frame a
  vehicle is visible. Counting unique **IDs**, not per-frame detections, is
  what makes a bus visible for 200 frames count once, not 200 times - same
  principle Day30 used for `class_counts`.
- **Violations by type / by class** - derived on demand from the flat
  `events: list[ViolationEvent]` log (`Counter` over `violation_type` or
  `class_name`) - one list is the only state that needs to stay correct,
  everything else is a view over it.
- **Timestamps** - each `ViolationEvent` carries `frame_idx` and
  `frame_idx / fps`, computed once at record time.
- **Traffic status** (`analytics.py`) - a simple threshold on
  `total_violations` (≤2 Normal, ≥6 High Violations, else Caution) - not a
  sophisticated model, just enough to turn a number into a glanceable
  banner for Variant 2's dashboard.

## Difference between the two demo variants

Both variants share the exact same detection/tracking/rules pipeline
(`process_video()` in `traffic_violation.py`) and differ only in what gets
drawn - same split Day32 used between its "Monitor" and "Analytics" looks.

**Variant 1 - Wrong-Way Detection** (`traffic_violation.py`'s
`render_variant1_frame`): a busy, full-frame "watch the road" view. Every
tracked vehicle gets a full box, an ID + class label, and a small arrow
along its own current heading. A fixed compass arrow in the top-left always
shows the configured normal direction for comparison. A confirmed
wrong-way vehicle turns red and is labeled "WRONG WAY"; the restricted zone
is a translucent orange-filled rectangle; a live counter badge (vehicles /
total violations / wrong-way / zone) sits top-left.

**Variant 2 - Traffic Violation Analytics** (`analytics.py`'s
`draw_analytics_frame`): the video itself is deliberately minimal - small
dots per vehicle (red = confirmed wrong-way, else its track colour) and a
thin zone outline, no boxes or arrows. The video is widened with a solid
dashboard panel docked to the right: total vehicles, total violations, a
Normal/Caution/High-Violations status banner, wrong-way/zone counts,
per-class counts, a trend sparkline, and a scrolling log of the most recent
violation events by vehicle ID and timestamp. This is meant to look like a
different *composition*, not just a recolour, of the same underlying data.

## The 6 sample videos

All 6 are free stock clips (Pixabay License - free for commercial and
personal use, no attribution required), picked to cover **different road
directions** and, more importantly, picked *after* finding out the hard
way which camera angles a plain COCO-pretrained YOLOv8n can and can't
detect vehicles in - see "Real challenges faced" below. Roughly a dozen
other candidates were downloaded and measured before landing on these 6;
the rejects are documented alongside the keepers below because *why* a
clip was dropped is more useful than a bare list of filenames.

| File | Scene | Normal direction | Notes |
|---|---|---|---|
| `madrid_intersection.mp4` | Street-level, multi-lane approach to a junction | down (toward camera) | Cleanest detections of the 6; full-frame direction check, no ROI needed |
| `hillside_street.mp4` | Street-level, two-way hill street, palm-lined | up (away from camera) | Real two-way traffic - see limitations below |
| `bangkok_boulevard.mp4` | Elevated fixed camera over a divided boulevard | down, scoped to the near carriageway | Demonstrates `direction_roi` on a genuinely divided road |
| `river_bridge_junction.mp4` | Aerial view of a river-town bridge + junction | right, scoped to the bridge | Restricted zone marks a cyclist lane vehicles shouldn't enter; needs a lower confidence slider (~0.15) - distant, small vehicles |
| `nhatrang_street.mp4` | Street-level, motorcycle-heavy Vietnamese street | up (away from camera) | Best vehicle-type diversity (car + motorcycle roughly 50/50); only 8s long, that's the clip's native length |
| `atlanta_commute.mp4` | Dusk drone shot of a commuter highway | up-right | Needs a lower confidence slider (~0.10); same real-world location as one of the rejected "train"-hallucination clips - see below for why this one works and that one didn't |

`download_samples.py` re-fetches and trims all 6 from their Pixabay CDN
URLs; the `*_config.json` presets (direction / zone / direction_roi per
video) are hand-calibrated against each clip's first frames and are
committed separately, same convention Day32 used for its `*_spaces.json`
layouts.

## Real challenges faced (measured, not guessed)

**Plain COCO YOLOv8n cannot see cars in a steep, elevated highway drone
shot - and confidently hallucinates a "train" instead.** The first two
candidate videos (both drone footage of a highway from a few hundred feet,
similar altitude/angle to a lot of "highway aerial" stock footage) were
measured to produce **zero** car/truck/bus/motorcycle detections even at
`conf=0.05`. What the model detected instead, at 0.39-0.83 confidence, was
the highway itself, classified as **"train"** - a long, straight, uniform
shape from directly above apparently reads as a train to a COCO-pretrained
detector more readily than the actual small, distant cars on it read as
cars. This is the same finding Day32 made for steep top-down parking-lot
footage (their fix: a custom CARPK-trained checkpoint). Trying Day32's
`vehicle_detection_aerial.pt` checkpoint here didn't transfer - it was
fine-tuned on nadir (straight-down) parking-lot imagery, not oblique
highway drone shots, and found at most 0-3 boxes on these clips. Given the
scope of this project, the fix was **not** to train a third detector, but
to swap those two videos out for street-level and moderately-elevated
fixed-camera footage instead - all 6 shipped samples were verified
(multiple sampled frames, `conf` from 0.10-0.25) to produce reliable
double-digit vehicle detections before being committed.

**The same real-world highway, filmed twice, gave opposite results - one
clip unusable, the other one of the best in the set.** `atlanta_commute`
and one of the rejected "train"-hallucinating clips are the same physical
overpass (same buildings, same shopping center, same highway geometry) -
just two different stock clips from the same location at different times
of day. The rejected one was flat midday light; `atlanta_commute` is warm,
low-angle dusk light with long shadows. At `conf=0.25` `atlanta_commute`
looks nearly as sparse as the rejected clip (2-3 vehicle boxes per frame);
dropping to `conf=0.10` reveals the model was finding most of the real
cars all along, just below the default cutoff (measured avg. 7
vehicles/frame across 12 sampled frames) - it never once hallucinated a
"train" on this clip. The likely reason: raking dusk light throws strong
shadows that separate each car's silhouette from the road surface, where
flat midday light lets a distant car blend into the pavement it's a similar
shade of grey as. Camera angle alone didn't predict detectability here;
lighting did.

**A few other rejected candidates, briefly:** a "tilt-shift"/miniature-effect
London street clip (selective blur is an intentional cinematic effect, not
something a detector should have to see through); three separate night
"hyperlapse" clips (Seoul, Taipei, an unnamed vertical clip) where a long
per-frame exposure turns every moving car into a light streak with no car
shape left to detect; a straight-down nadir highway clip that, like the
elevated "train" clips, produced high-confidence detections for the wrong
classes entirely ("tv", "suitcase", "microwave" for what are obviously
car-shaped patches of road); and a heavily backlit sunrise highway shot
silhouetted against the sun, too low-contrast for any detection at all.
None of these are exotic failure modes - they're exactly the kind of
"looks fine to a human, unusable for a detector" footage a real deployment
would also have to screen out before trusting a camera feed.

**A rectangular `direction_roi` cannot cleanly separate two lanes that
converge toward one vanishing point.** `bangkok_boulevard`'s divided
boulevard has a physical median (a hedge row), so "left of the median" is
a clean, real boundary between carriageways - the ROI works exactly as
intended there. `hillside_street` has no physical median: it's a normal
two-way street filmed head-on, so the near (oncoming) and far
(same-direction) lanes' bounding boxes overlap in x-position near the
vanishing point and only diverge close to the frame edges. An early
version scoped `hillside_street`'s ROI to a vertical strip (x 0.30-0.65)
expecting that to isolate the uphill lane; inspecting the annotated output
showed a wrong-way-flagged car and an unflagged bus sitting almost
side-by-side in nearly the same x-range - proof the rectangle wasn't
separating anything. The shipped config drops the ROI for this video
entirely and documents the honest consequence: genuine oncoming traffic on
this clip will often be flagged wrong-way, because distinguishing "legally
in the oncoming lane" from "actually driving against a one-way street"
needs lane-level geometry (e.g. a fitted lane polygon) that a single
rectangle can't express. `hillside_street`'s debounce is tightened further
than the project default specifically because of this (135° threshold, 85%
of the last 14 frames, vs. the default 130°/80%/14) - it still filters out
single-frame noise; it does not, and cannot, filter out sustained, genuine
oncoming motion.

**`river_bridge_junction`'s vehicles are small and distant enough that the
default confidence threshold under-detects them.** At `conf=0.25` (this
project's default, same as Day31) the clip averaged under 3 detected
vehicles per sampled frame; dropping to `conf=0.15` (in the app's sidebar
slider) raised that to over 5. Same root cause Day32 documented for its
own aerial samples: camera distance costs the detector confidence before
it costs it accuracy.

**`model.track(..., persist=True)` persists ACROSS calls, not just across
frames of one call - which silently corrupted multi-video batch runs.**
The app and `coding_practice/02_batch_all_videos.py` both reuse one cached
`YOLO` instance across every sample video, for the same reason Day29-32
cache their models (loading + fusing a checkpoint is expensive; a
Streamlit session or a batch script shouldn't pay that cost per video).
The first version of the batch script did exactly that and the result was
video #2 in the run picking up wherever video #1's ByteTracker left off -
its very first detected vehicle was labeled "#124" instead of "#1", and
in principle a stale track from the end of one video could theoretically
(if unlikely) match a detection at the start of an unrelated one. The fix,
`_reset_tracker()` in `traffic_violation.py`, calls the tracker's own
`.reset()` (clears its active/lost track state) and `.reset_id()` (clears
its ID counter) at the start of every `process_video()` call - both
methods exist on Ultralytics' `BYTETracker` for exactly this purpose, they
just aren't invoked automatically by `model.track()`.

## Limitations

- Direction and zone checks operate on a vehicle's **centroid**, not its
  full footprint - a large truck can be judged "in the zone" a frame or two
  before its front bumper visually crosses the line.
- The wrong-way rule has no notion of lanes, only a direction vector and an
  optional rectangular region - see `hillside_street` above for where that
  falls short.
- A parked or idling vehicle can occasionally still trip the wrong-way
  rule. `nhatrang_street`'s curb is lined with parked motorbikes, and a
  couple of them accumulate just enough apparent centroid drift over the
  14-frame confirmation window - from being nudged, partially occluded by
  a passing pedestrian, or simple detection-box jitter on a cluttered
  background - to read as sustained motion in one direction. Excluding the
  parked-bike curb from `direction_roi` didn't fix it (the flagged
  vehicles were within the driving lane's own x-range, not the curb), which
  says the false positives aren't coming from the parked cluster
  specifically - they're the general cost of judging "moving vs. not" from
  centroid displacement alone, with no independent check for continuous,
  monotonic motion (as opposed to noise that happens to net out in one
  direction over a handful of frames).
- All 6 sample presets are hand-calibrated for their specific camera angle
  and framing; a new video (via "Upload your own" in the app) needs its own
  direction + zone set through the sidebar sliders, same as Day31/32's ROI
  and space layouts did.
- Processing is capped at 300 frames (~12s at 25fps) on the hosted app so
  one upload can't stall it for other users - see `MAX_VIDEO_FRAMES` in
  `app.py`.


## Results

Generated by `coding_practice/02_batch_all_videos.py` (also in
`outputs/violation_results.md`):

| Video | Direction | Vehicles | Violations | Wrong-way | Zone | Frames |
|---|---|---|---|---|---|---|
| atlanta_commute | up-right | 14 | 2 | 0 | 2 | 300 |
| bangkok_boulevard | down | 78-92* | 21 | 0 | 21 | 300 |
| hillside_street | up | 19-21* | 6-9* | 2 | 4-5* | 300 |
| madrid_intersection | down | 45-78* | 0 | 0 | 0 | 300 |
| nhatrang_street | up | 24 | 13 | 2 | 11 | 201 |
| river_bridge_junction | right | 25 | 19 | 3 | 16 | 350 |

\* ByteTrack occasionally assigns and immediately drops an ID for a single
noisy detection (a spurious box that appears for one frame and never
tracks); a re-run can differ by a handful of vehicles even with identical
settings, since it comes down to which of those single-frame flickers the
detector happens to produce. `bangkok_boulevard`'s own vehicle IDs run
particularly high (into the 200s) for the same reason, amplified by that
scene's small, fast, closely-spaced vehicles - this doesn't affect which
violations get recorded, only the ID numbers printed on screen, since
`total_vehicles` counts distinct IDs that were actually seen, not the raw
ID value.

`madrid_intersection` recording **zero** violations is a feature, not a
gap: its traffic genuinely all flows the configured direction and never
enters the marked zone in this clip, and the stricter debounce (see "Real
challenges faced") is correctly not manufacturing a violation out of
normal driving. `atlanta_commute` is a similarly clean, mostly-quiet run
(2 zone entries from ordinary lane traffic passing through the marked
lane, 0 wrong-way). `bangkok_boulevard`, `hillside_street`,
`nhatrang_street`, and `river_bridge_junction` each demonstrate the
violation-drawing/dashboard path with real triggers - `nhatrang_street`
has the most (13, on the fewest frames of any clip - 201 vs. 300+ for the
others), partly genuine zone entries from real motorcycle/car traffic and
partly the parked-vehicle edge case documented under "Limitations".


## Project layout

```
Day36/
├── app.py                    Streamlit app (both demo variants)
├── tracker.py                YOLO + ByteTrack detection/tracking + motion history
├── traffic_violation.py      Direction/zone rules, Variant 1 drawing, process_video()
├── analytics.py              Variant 2 dashboard + traffic-status classification
├── download_samples.py       Re-fetches the 6 sample videos from Pixabay
├── requirements.txt
├── README.md
├── HOW_TO_RUN.txt
├── sample_videos/            6 clips + their *_config.json presets
├── sample_outputs/           coding_practice/01's single-video output
├── outputs/
│   ├── variant-1/             Wrong-Way Detection output, all 6 videos
│   ├── variant-2/             Traffic Violation Analytics output, all 6 videos
│   └── violation_results.md
├── screenshots/
└── coding_practice/
    ├── 01_traffic_violation.py   the brief's literal checklist
    └── 02_batch_all_videos.py    both variants x all 6 videos + results table
```


## Setup & run

See `HOW_TO_RUN.txt` for exact commands (install, CLI usage, the coding
practice scripts, running the app locally, and deploying it).
