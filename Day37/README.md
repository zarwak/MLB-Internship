# Day 37 - People Counting & Crowd Analysis

This project uses YOLOv8 + ByteTrack for person detection and tracking. It focuses on:

- People detection in video
- Persistent track IDs for each person
- Live people count in each frame
- Bounding boxes and confidence scores
- Optional counting line and ROI filtering
- Saving annotated output video
- Smart people counting dashboard in the Streamlit app

## What this project includes

- `people_counter.py`: reusable detection, tracking, and counting logic
- `app.py`: Streamlit mini project for uploading an image or video
- `coding_practice/01_people_counting.py`: single-file demo for a video
- `coding_practice/02_batch_people_count.py`: batch processing across multiple clips
- `requirements.txt`: project dependencies

## Setup

From the repo root:

1. Open the Day 37 folder.
2. Create a virtual environment if needed.
3. Install dependencies:

   pip install -r requirements.txt

4. Run the app:

   streamlit run app.py

## Example command line usage

python coding_practice/01_people_counting.py --video path/to/video.mp4 --out output/people_counted.mp4

## Project goals

The system should:

- detect people in each frame using YOLO
- keep the same track ID for the same person across frames
- count total visible people per frame
- draw boxes, IDs, and confidence scores
- display the live count on the processed video
- save the processed output to disk
- optionally count entries/exits using a counting line
- optionally count only people inside a region of interest

## How the counting algorithm works

The logic is built around a simple tracking-based pipeline:

1. Frame-by-frame detection
   - For each video frame, the model runs YOLOv8n with the `person` class only.
   - Each detected person returns a bounding box, confidence score, and a class label.
   - Detections with low confidence are ignored.

2. Persistent identity tracking
   - We call `model.track(..., persist=True, tracker="bytetrack.yaml")`.
   - This keeps the tracker state alive across frames, so the same person keeps the same track ID while visible.
   - This is essential because a person seen for 30 frames should count once, not 30 times.

3. Person centroid extraction
   - For each detected box, we compute the center point of the person:
     - centroid x = (x1 + x2) / 2
     - centroid y = (y1 + y2) / 2
   - This centroid is used for counting, ROI checks, and line crossing detection.

4. Counting people in the current frame
   - The current frame count is simply the number of tracked people in that frame.
   - `len(people)` gives the number of people visible now.

5. Total people seen across the video
   - We maintain a set of all track IDs seen so far.
   - At the end of processing, `len(seen_ids)` gives the total number of unique people detected during the whole video.
   - This avoids a double count when the same person stays visible for many frames.

6. Counting line logic
   - If a counting line is enabled, each person's centroid is compared to the line position.
   - For a horizontal line, we compare the person's y-coordinate against the line y-position.
   - For a vertical line, we compare the x-coordinate against the line x-position.
   - If the person's side relative to the line changes from one frame to the next, it is counted as a crossing.

7. ROI logic
   - If an ROI is enabled, the code checks whether the centroid falls inside the selected rectangle.
   - Only people whose centroid is inside the ROI are counted for the ROI-specific total.

8. Annotated output
   - Each person gets a bounding box, ID, and confidence label.
   - The frame overlay displays:
     - total number of people in the frame
     - peak people count so far
     - total unique people seen
     - optional ROI count and line-crossing count

9. Output saving
   - The processed video is written to disk as an annotated MP4.
   - This gives you a final output video plus a summary count for analysis.

### Why this works well

The key idea is counting by track ID rather than per-frame detection.
A person may appear in 100 frames, but if they keep the same track ID, they are still one person. This is what makes the system reliable and avoids inflated totals.

## Notes

- The app uses the COCO-pretrained YOLOv8n model, which includes the person class.
- For best results, use videos with clear camera framing and visible people.
- If you want to test with your own dataset, add 5+ videos of malls, hallways, streets, or campuses.

## Recommended dataset sources

- Pexels
- Pixabay
- Your own recorded footage
- Office, hallway, campus, or shopping scene videos
