# Day 16 — OpenCV Fundamentals & Image Processing Toolkit

## Overview

This project covers the fundamentals of **Computer Vision** using **OpenCV**, including image representation, manipulation, and a full-featured **Streamlit** web application for interactive image processing.

## Project Structure

```
Day16/
├── app.py                  # Main Streamlit app (polished UI)
├── sample_images/          # Input sample images (5 categories)
│   ├── landscape.jpg
│   ├── person.jpg
│   ├── vehicle.jpg
│   ├── document.jpg
│   └── object.jpg
├── output_images/          # All processed output images
├── coding_practice/        # Individual practice programs
│   ├── 01_image_info.py
│   ├── 02_grayscale_conversion.py
│   ├── 03_resize_image.py
│   ├── 04_crop_image.py
│   ├── 05_rotate_image.py
│   ├── 06_flip_image.py
│   ├── 07_draw_shapes.py
│   ├── utils.py
│   ├── run_all.py
│   └── challenge_task.py
├── toolkit/                # Streamlit Image Processing Toolkit (alternative)
│   └── image_toolkit.py
└── README.md
```

## Coding Practice Programs

Each program in `coding_practice/` demonstrates a core OpenCV operation:

| Program | Description |
|---------|-------------|
| `01_image_info.py` | Reads an image and displays dimensions, channels, file size, data type, and pixel range |
| `02_grayscale_conversion.py` | Converts color to grayscale using `cvtColor()` and `imread(IMREAD_GRAYSCALE)` |
| `03_resize_image.py` | Resizes to 50%, 25%, 300×300, 800×600, 200% with appropriate interpolation |
| `04_crop_image.py` | Crops 4 quadrants + center region, saves collage |
| `05_rotate_image.py` | Rotates by 90° CW, 90° CCW, 180°, 270° using `cv2.rotate()` and `warpAffine()` |
| `06_flip_image.py` | Flips horizontally, vertically, and both; saves labeled collage |
| `07_draw_shapes.py` | Draws rectangle, circle, line, triangle, pentagon, and custom text |

### Running the Practice Programs

```bash
# Run all programs at once
python coding_practice/run_all.py

# Run a specific program (defaults to first sample image)
python coding_practice/01_image_info.py

# Run with a specific image
python coding_practice/01_image_info.py sample_images/landscape.jpg
```

## Image Processing Toolkit (Streamlit App)

A menu-driven web application built with **Streamlit** and **OpenCV** that allows users to interactively process images.

### Features

**Core Operations:**
- **Load an image** — Upload your own or select from sample images
- **Convert to grayscale** — Using `cv2.cvtColor()` with `COLOR_BGR2GRAY`
- **Resize image** — Specify custom width and height
- **Rotate image** — Any angle from -360° to 360°
- **Flip image** — Horizontal, vertical, or both
- **Crop image** — Select region via sliders
- **Draw shapes** — Rectangle, circle, line, polygon with custom colors
- **Add custom text** — Custom text with font size, color, and position
- **Save the processed image** — Save to `output_images/`

**Bonus Features:**
- **Adjust brightness and contrast** — Sliders for fine-tuning
- **BGR vs RGB comparison** — Side-by-side comparison with explanation
- **Side-by-side display** — Original and processed images shown together

### Running the Toolkit

**Main entry point (polished UI):**
```bash
streamlit run app.py
```

**Alternative (in toolkit/ folder):**
```bash
streamlit run toolkit/image_toolkit.py
```

## Key Concepts

### BGR vs RGB

OpenCV uses **BGR** (Blue-Green-Red) as its default color channel order, while most other libraries (matplotlib, PIL, web browsers) use **RGB** (Red-Green-Blue).

- **BGR**: OpenCV's `cv2.imread()` reads images in BGR format. When using `cv2.imshow()`, the image is expected in BGR.
- **RGB**: The standard color format used by most display libraries. When displaying with `matplotlib.pyplot.imshow()` or `PIL.Image`, the image must be in RGB format.

**Why the difference matters:** If you read an image with OpenCV (BGR) and display it with matplotlib (RGB) without conversion, the red and blue channels will be swapped, producing a color-shifted image. Always use `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` when switching between OpenCV and other libraries.

### Grayscale Images

A grayscale image has only one channel (intensity values from 0 to 255), as opposed to a color image which has 3 channels (BGR). Grayscale images are used because:

1. **Reduced complexity** — Single channel is easier to process and analyze
2. **Lower memory usage** — 1/3 the memory of a color image
3. **Focus on structure** — Many CV algorithms (edge detection, thresholding, feature detection) work on intensity values, not color
4. **Faster computation** — Fewer pixels to process means faster algorithms

OpenCV converts BGR to grayscale using the formula:
```
Gray = 0.299 * R + 0.587 * G + 0.114 * B
```

### OpenCV Functions Used

| Function | Purpose |
|----------|---------|
| `cv2.imread()` | Read an image from file |
| `cv2.imshow()` | Display an image in a window |
| `cv2.imwrite()` | Save an image to file |
| `cv2.cvtColor()` | Convert color space (BGR↔Gray, BGR↔RGB) |
| `cv2.resize()` | Resize an image |
| `cv2.rotate()` | Rotate by 90° multiples |
| `cv2.warpAffine()` | Apply affine transformation (rotation, translation) |
| `cv2.getRotationMatrix2D()` | Get 2D rotation matrix |
| `cv2.flip()` | Flip an image |
| `cv2.rectangle()` | Draw a rectangle |
| `cv2.circle()` | Draw a circle |
| `cv2.line()` | Draw a line |
| `cv2.polylines()` | Draw polygon outline |
| `cv2.fillPoly()` | Draw filled polygon |
| `cv2.putText()` | Add text to image |
| `cv2.waitKey()` | Wait for keyboard input |
| `cv2.destroyAllWindows()` | Close all OpenCV windows |

## Challenges Faced and Solutions

1. **Headless environment (no display)** — `cv2.imshow()` fails in environments without a display. Solved by creating a `safe_imshow()` wrapper in `utils.py` that catches the error and prints a message instead.

2. **Collage dimension mismatch** — When creating side-by-side collages, images of different sizes cause `np.hstack()` to fail. Solved by resizing all collage images to a uniform 300×300 for display.

3. **Grayscale conversion methods differ** — `cv2.cvtColor()` and `cv2.imread(IMREAD_GRAYSCALE)` produce slightly different results due to different internal algorithms. This is expected behavior and documented in the code.

4. **Streamlit color picker returns hex** — Streamlit's `color_picker` returns hex strings (e.g., `#FF0000`), but OpenCV expects BGR tuples. Solved with a `hex_to_bgr()` helper function.

5. **File path handling** — Scripts need to work from any directory. Solved by using `os.path.dirname(__file__)` to construct relative paths.

## Challenge Task

All operations from the toolkit have been applied to 5 different images (landscape, person, vehicle, document, object). Results are organized in `output_images/` with descriptive filenames.

## Environment

- **Python**: 3.13.2
- **OpenCV**: 4.10.0
- **NumPy**: 2.2.4
- **Streamlit**: 1.45.0

## How to Run

### Coding Practice Programs
```bash
# Run all practice programs
python coding_practice/run_all.py

# Run a specific program
python coding_practice/01_image_info.py [image_path]
```

### Challenge Task
```bash
# Apply all operations on all 5 sample images
python coding_practice/challenge_task.py
```

### Streamlit Image Processing Toolkit
```bash
# Launch the web application (polished UI)
streamlit run app.py

# Alternative (in toolkit/ folder)
streamlit run toolkit/image_toolkit.py
```
