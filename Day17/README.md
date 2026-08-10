# Day 17 - Image Transformations & Image Enhancement

This is my Day 17 work. The goal was simple: learn how to fix up images using
OpenCV, before feeding them into any AI model. Things like straightening a
tilted photo, cleaning up noise, and making text easier to read.

## What's in this folder

- **coding_practice/** - small practice scripts, one file per topic. This is
  where I learned each technique on its own, in isolation.
- **image_enhancement_tool/** - the real mini project. You give it a messy,
  tilted photo of a document and it hands back a clean, straightened,
  readable version. It also has a Streamlit web app so anyone can try it
  without touching code.
- **Challenge Task/** - I took 5 tilted document photos and ran them through
  the tool, so you can see the original, the straightened version, and the
  final cleaned-up version for each one.

Each folder has its own `HOW_TO_RUN.txt` with step by step instructions.

## The transformations I learned

- **Translation** - just sliding the whole image left/right or up/down.
  Nothing fancy, mostly used to shift or align images.
- **Rotation** - spinning the image around its center by an angle. Useful
  for fixing a sideways photo.
- **Scaling** - making the image bigger or smaller. Every AI model expects
  a certain image size, so this is used all the time.
- **Affine Transform** - a mix of moving, rotating, and slightly skewing an
  image, but straight lines stay straight and parallel lines stay parallel.
  I used 3 points to describe the change.
- **Perspective Transform** - the big one. This can take a photo taken at
  an angle (like a phone photo of paper lying on a table) and make it look
  like it was scanned flat. I used 4 corner points for this one instead of 3.

## Why each enhancement technique matters

- **Brightness Adjustment** - fixes a photo that's too dark or too washed out.
- **Contrast Adjustment** - makes the difference between light and dark
  areas stronger, so text stands out more from the background.
- **Gaussian Blur** - softly smooths the whole image. Good for general noise.
- **Median Blur** - great at removing random black/white dot noise
  ("salt and pepper" noise) without blurring everything else too much.
- **Bilateral Filter** - smooths out noise but tries to keep edges (like the
  edges of letters) sharp. Slower, but nicer results on text.
- **Sharpening** - makes edges and text look crisper. Basically the opposite
  of blurring.

In the real mini project, I ended up using bilateral filter for noise
removal (best for text) and a technique called CLAHE for contrast, which
is a smarter version of plain contrast adjustment - it fixes contrast in
small local areas instead of the whole image at once, so it doesn't blow
out bright spots or crush dark shadows.

## Which one made the biggest difference

Perspective transform, by far. A tilted, skewed photo is basically unusable
for OCR or reading text off it, no matter how much you sharpen or brighten
it. Once the page is straightened out, everything else (denoising,
contrast, sharpening) just makes an already-usable image look nicer. Fixing
the tilt is what makes the image usable in the first place.

## Challenges I ran into

- **Figuring out the corner order for perspective transform.** You have to
  always list the 4 corners in the same order (top-left, top-right,
  bottom-right, bottom-left), otherwise the image comes out flipped or
  twisted. Took some trial and error to get this consistent.
- **CLAHE (contrast) only works on grayscale images.** If I wanted to keep
  color, I had to convert to a different color format (LAB), apply CLAHE to
  just the brightness channel, then convert back. Wasn't obvious at first.
- **Scripts saving files in the wrong folder.** Early on, my scripts used
  paths like `"images"` and `"outputs"` which are relative to wherever you
  run the script from, not where the script file actually lives. So results
  kept landing in the wrong folder depending on how I ran things. Fixed it
  by always building the path from the script's own location.
- **Deploying to Streamlit Cloud broke because of a space in a folder name.**
  My project folder was called "Image Enhancement Tool" (with spaces), and
  Streamlit Cloud couldn't correctly find the `requirements.txt` file
  because of the space in the path. Renamed the folder to
  `image_enhancement_tool` (no spaces) and it worked.
- **No real dataset yet.** Since I didn't have real tilted document photos
  on hand, I wrote small scripts to generate fake ones (a plain page with
  text, warped to look tilted) just so I had something to test against.
  Real photos can just be dropped into the input folders instead.

## Links

- Streamlit app: [add your public app link here]
- GitHub repo: [add your repo link here]
- Screen recording: [add your recording link here]
