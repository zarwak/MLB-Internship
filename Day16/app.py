"""
Image Processing Toolkit — Streamlit App
=========================================
A polished, professional image processing application built with OpenCV and Streamlit.

Features:
- Load an image (upload or select from samples)
- Convert to grayscale
- Resize image
- Rotate image
- Flip image
- Crop image
- Draw shapes (rectangle, circle, line, polygon)
- Add custom text
- Save the processed image

Bonus Features:
- Adjust brightness and contrast
- Convert to RGB and compare with BGR
- Display original and processed images side by side

Usage:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Page Configuration (must be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Image Processing Toolkit",
    page_icon="🖼️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS — Professional dark theme with accent colors
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0e1117; }
    
    /* Sidebar styling */
    .css-1d391kg { background-color: #1e222d; }
    
    /* Headers */
    h1 { color: #00d4ff; font-weight: 700; }
    h2 { color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 5px; }
    h3 { color: #ff6b6b; }
    
    /* Info boxes */
    .stAlert { background-color: #1e222d; border: 1px solid #00d4ff; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
    }
    
    /* Success messages */
    .stSuccess { background-color: #1a3a1a; color: #4ade80; }
    
    /* Image captions */
    .css-1v0mb3v { color: #a0a0a0; }
    
    /* Radio and select boxes */
    .stRadio > div { color: #ffffff; }
    
    /* Sidebar radio */
    .sidebar .stRadio > div > label { color: #ffffff; }
    
    /* Dividers */
    hr { border-color: #333; }
    
    /* Metric cards */
    .metric-card {
        background: #1e222d;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def load_sample_images():
    """Return a sorted list of sample image filenames."""
    if not os.path.exists(SAMPLE_DIR):
        return []
    return sorted([
        f for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])


def read_image(path):
    """Read an image from a file path and return it as a NumPy array (BGR)."""
    img = cv2.imread(path)
    if img is None:
        st.error(f"Could not read image: {path}")
        return None
    return img


def image_to_rgb(img):
    """Convert a BGR image to RGB for Streamlit display."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hex_to_bgr(hex_color):
    """Convert a hex color string (e.g., '#FF0000') to a BGR tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def display_side_by_side(original, processed, label="Processed"):
    """Display original and processed images side by side."""
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_to_rgb(original), caption="Original", use_container_width=True)
    with col2:
        st.image(image_to_rgb(processed), caption=label, use_container_width=True)


# ---------------------------------------------------------------------------
# Operation Functions
# ---------------------------------------------------------------------------
def op_grayscale(img):
    """Convert image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def op_resize(img, width, height):
    """Resize image to specified dimensions."""
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def op_rotate(img, angle):
    """Rotate image by a given angle (degrees)."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))


def op_flip(img, direction):
    """Flip image horizontally, vertically, or both."""
    flip_map = {"Horizontal": 1, "Vertical": 0, "Both": -1}
    return cv2.flip(img, flip_map[direction])


def op_crop(img, x, y, width, height):
    """Crop a region from the image."""
    h, w = img.shape[:2]
    x2 = min(x + width, w)
    y2 = min(y + height, h)
    return img[y:y2, x:x2]


def op_brightness_contrast(img, brightness, contrast):
    """Adjust brightness and contrast of an image."""
    result = img.copy().astype(np.float32)
    result = result * (contrast / 100.0 + 1.0)
    result = result + brightness
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def op_bgr_to_rgb(img):
    """Convert BGR to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def op_draw_shapes(img, shapes):
    """Draw shapes on the image."""
    canvas = img.copy()
    for shape in shapes:
        if shape["type"] == "Rectangle":
            cv2.rectangle(canvas, shape["pt1"], shape["pt2"],
                          shape["color"], shape["thickness"])
        elif shape["type"] == "Circle":
            cv2.circle(canvas, shape["center"], shape["radius"],
                       shape["color"], shape["thickness"])
        elif shape["type"] == "Line":
            cv2.line(canvas, shape["pt1"], shape["pt2"],
                     shape["color"], shape["thickness"])
        elif shape["type"] == "Polygon":
            pts = np.array(shape["points"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], shape["is_closed"],
                          shape["color"], shape["thickness"])
    return canvas


def op_add_text(img, text, position, font_scale, color, thickness):
    """Add custom text to the image."""
    canvas = img.copy()
    cv2.putText(canvas, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return canvas


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

# Header
st.markdown("""
<div style="text-align: center; padding: 20px 0; border-bottom: 2px solid #00d4ff; margin-bottom: 20px;">
    <h1 style="color: #00d4ff; margin: 0;">🖼️ Image Processing Toolkit</h1>
    <p style="color: #a0a0a0; margin: 5px 0 0 0; font-size: 16px;">
        A professional image processing application built with OpenCV & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Image Loading
# ---------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="color: #00d4ff; font-size: 18px; font-weight: 600; margin-bottom: 15px;">📁 Load Image</div>
""", unsafe_allow_html=True)

source = st.sidebar.radio("Image Source", ["📤 Upload Image", "🖼️ Sample Images"])

original_img = None
selected = None

if source == "📤 Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if original_img is None:
            st.sidebar.error("Could not read the uploaded image.")
else:
    sample_images = load_sample_images()
    if sample_images:
        selected = st.sidebar.selectbox("Select a sample image", sample_images)
        if selected:
            original_img = read_image(os.path.join(SAMPLE_DIR, selected))
    else:
        st.sidebar.warning("No sample images found in `sample_images/` folder.")

# ---------------------------------------------------------------------------
# Display Original Image
# ---------------------------------------------------------------------------
if original_img is not None:
    h, w = original_img.shape[:2]
    channels = original_img.shape[2] if len(original_img.shape) == 3 else 1

    # Image info card
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #00d4ff; margin-top: 0;">📊 Image Information</h3>
        <p style="color: #a0a0a0; margin: 5px 0;">
            <strong>Dimensions:</strong> {w} × {h} pixels |
            <strong>Channels:</strong> {channels} |
            <strong>Data type:</strong> {original_img.dtype}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Original Image")
    st.image(image_to_rgb(original_img), use_container_width=True)

    # ---------------------------------------------------------------------------
    # Operations Menu
    # ---------------------------------------------------------------------------
    st.sidebar.markdown("""
    <div style="color: #00d4ff; font-size: 18px; font-weight: 600; margin-bottom: 15px; margin-top: 20px;">🔧 Operations</div>
    """, unsafe_allow_html=True)

    operation = st.sidebar.selectbox(
        "Choose an operation",
        [
            "--- Select an operation ---",
            "🎨 Convert to Grayscale",
            "📏 Resize Image",
            "🔄 Rotate Image",
            "🔀 Flip Image",
            "✂️ Crop Image",
            "🔷 Draw Shapes",
            "📝 Add Custom Text",
            "☀️ Adjust Brightness & Contrast",
            "🎨 BGR vs RGB Comparison",
        ],
    )

    processed_img = original_img.copy()
    operation_applied = False

    # ---------------------------------------------------------------------------
    # Operation: Grayscale
    # ---------------------------------------------------------------------------
    if operation == "🎨 Convert to Grayscale":
        st.subheader("Convert to Grayscale")
        method = st.radio("Method", ["cv2.cvtColor (BGR2GRAY)", "cv2.imread (IMREAD_GRAYSCALE)"])
        processed_img = op_grayscale(original_img)
        operation_applied = True
        display_side_by_side(original_img, processed_img, "Grayscale")

    # ---------------------------------------------------------------------------
    # Operation: Resize
    # ---------------------------------------------------------------------------
    elif operation == "📏 Resize Image":
        st.subheader("Resize Image")
        col1, col2 = st.columns(2)
        with col1:
            new_width = st.number_input("Width", min_value=1, value=w, step=10)
        with col2:
            new_height = st.number_input("Height", min_value=1, value=h, step=10)
        processed_img = op_resize(original_img, new_width, new_height)
        operation_applied = True
        display_side_by_side(original_img, processed_img, f"Resized ({new_width}×{new_height})")

    # ---------------------------------------------------------------------------
    # Operation: Rotate
    # ---------------------------------------------------------------------------
    elif operation == "🔄 Rotate Image":
        st.subheader("Rotate Image")
        angle = st.slider("Rotation Angle (degrees)", -360, 360, 0, step=15)
        if angle != 0:
            processed_img = op_rotate(original_img, angle)
            operation_applied = True
            display_side_by_side(original_img, processed_img, f"Rotated {angle}°")
        else:
            st.info("Set an angle to rotate the image.")

    # ---------------------------------------------------------------------------
    # Operation: Flip
    # ---------------------------------------------------------------------------
    elif operation == "🔀 Flip Image":
        st.subheader("Flip Image")
        direction = st.radio("Flip Direction", ["Horizontal", "Vertical", "Both"])
        processed_img = op_flip(original_img, direction)
        operation_applied = True
        display_side_by_side(original_img, processed_img, f"Flipped {direction}")

    # ---------------------------------------------------------------------------
    # Operation: Crop
    # ---------------------------------------------------------------------------
    elif operation == "✂️ Crop Image":
        st.subheader("Crop Image")
        st.markdown("Select the region to crop:")
        col1, col2 = st.columns(2)
        with col1:
            x = st.slider("X (left)", 0, w - 1, 0)
            width = st.slider("Width", 1, w - x, w // 2)
        with col2:
            y = st.slider("Y (top)", 0, h - 1, 0)
            height = st.slider("Height", 1, h - y, h // 2)
        processed_img = op_crop(original_img, x, y, width, height)
        operation_applied = True
        display_side_by_side(original_img, processed_img, f"Cropped ({width}×{height})")

    # ---------------------------------------------------------------------------
    # Operation: Draw Shapes
    # ---------------------------------------------------------------------------
    elif operation == "🔷 Draw Shapes":
        st.subheader("Draw Shapes")
        st.markdown("Configure the shapes to draw on the image:")

        shapes = []

        # Rectangle
        if st.checkbox("Draw Rectangle", value=True):
            col1, col2 = st.columns(2)
            with col1:
                rx1 = st.number_input("Rect X1", 0, w, 20)
                ry1 = st.number_input("Rect Y1", 0, h, 20)
            with col2:
                rx2 = st.number_input("Rect X2", 0, w, 150)
                ry2 = st.number_input("Rect Y2", 0, h, 120)
            r_color = st.color_picker("Rectangle Color", "#00FF00")
            r_thickness = st.slider("Rectangle Thickness", 1, 10, 3)
            shapes.append({
                "type": "Rectangle",
                "pt1": (rx1, ry1), "pt2": (rx2, ry2),
                "color": hex_to_bgr(r_color), "thickness": r_thickness,
            })

        # Circle
        if st.checkbox("Draw Circle", value=True):
            cx = st.number_input("Circle Center X", 0, w, w - 100)
            cy = st.number_input("Circle Center Y", 0, h, 80)
            radius = st.number_input("Circle Radius", 1, min(w, h) // 2, 50)
            c_color = st.color_picker("Circle Color", "#FF0000")
            c_thickness = st.slider("Circle Thickness", 1, 10, 3)
            shapes.append({
                "type": "Circle",
                "center": (cx, cy), "radius": radius,
                "color": hex_to_bgr(c_color), "thickness": c_thickness,
            })

        # Line
        if st.checkbox("Draw Line", value=True):
            col1, col2 = st.columns(2)
            with col1:
                lx1 = st.number_input("Line X1", 0, w, 20)
                ly1 = st.number_input("Line Y1", 0, h, 20)
            with col2:
                lx2 = st.number_input("Line X2", 0, w, w - 20)
                ly2 = st.number_input("Line Y2", 0, h, h - 20)
            l_color = st.color_picker("Line Color", "#0000FF")
            l_thickness = st.slider("Line Thickness", 1, 10, 2)
            shapes.append({
                "type": "Line",
                "pt1": (lx1, ly1), "pt2": (lx2, ly2),
                "color": hex_to_bgr(l_color), "thickness": l_thickness,
            })

        # Polygon
        if st.checkbox("Draw Polygon"):
            st.markdown("Enter polygon points (x, y) as comma-separated values:")
            default_pts = f"{w//2},100,{w//2-80},200,{w//2+80},200"
            pts_str = st.text_input("Polygon Points", value=default_pts)
            try:
                coords = [int(x) for x in pts_str.split(",")]
                points = [(coords[i], coords[i + 1]) for i in range(0, len(coords) - 1, 2)]
                p_color = st.color_picker("Polygon Color", "#FFFF00")
                p_thickness = st.slider("Polygon Thickness", 1, 10, 3)
                shapes.append({
                    "type": "Polygon",
                    "points": points,
                    "is_closed": True,
                    "color": hex_to_bgr(p_color), "thickness": p_thickness,
                })
            except (ValueError, IndexError):
                st.warning("Invalid polygon points format. Use: x1,y1,x2,y2,...")

        if shapes:
            processed_img = op_draw_shapes(original_img, shapes)
            operation_applied = True
            display_side_by_side(original_img, processed_img, "Shapes Drawn")
        else:
            st.info("Enable at least one shape to draw.")

    # ---------------------------------------------------------------------------
    # Operation: Add Text
    # ---------------------------------------------------------------------------
    elif operation == "📝 Add Custom Text":
        st.subheader("Add Custom Text")
        text = st.text_input("Text to add", value="ML-Bench Day 16")
        col1, col2 = st.columns(2)
        with col1:
            text_x = st.number_input("X Position", 0, w, 20)
            text_y = st.number_input("Y Position", 0, h, h - 20)
        with col2:
            font_scale = st.slider("Font Scale", 0.1, 3.0, 0.8, 0.1)
            thickness = st.slider("Thickness", 1, 10, 2)
        text_color = st.color_picker("Text Color", "#FFFFFF")
        processed_img = op_add_text(
            original_img, text, (text_x, text_y),
            font_scale, hex_to_bgr(text_color), thickness
        )
        operation_applied = True
        display_side_by_side(original_img, processed_img, "Text Added")

    # ---------------------------------------------------------------------------
    # Operation: Brightness & Contrast
    # ---------------------------------------------------------------------------
    elif operation == "☀️ Adjust Brightness & Contrast":
        st.subheader("Adjust Brightness & Contrast")
        col1, col2 = st.columns(2)
        with col1:
            brightness = st.slider("Brightness", -100, 100, 0)
        with col2:
            contrast = st.slider("Contrast", -100, 100, 0)
        processed_img = op_brightness_contrast(original_img, brightness, contrast)
        operation_applied = True
        display_side_by_side(original_img, processed_img,
                             f"Brightness: {brightness}, Contrast: {contrast}")

    # ---------------------------------------------------------------------------
    # Operation: BGR vs RGB Comparison
    # ---------------------------------------------------------------------------
    elif operation == "🎨 BGR vs RGB Comparison":
        st.subheader("BGR vs RGB Comparison")
        st.markdown("""
        <div class="metric-card">
        <p style="color: #a0a0a0;">
        <strong>BGR (Blue-Green-Red)</strong> is the default color format used by OpenCV.<br>
        <strong>RGB (Red-Green-Blue)</strong> is the standard format used by most display libraries.
        </p>
        <p style="color: #a0a0a0;">
        When displaying an image with <code>cv2.imshow()</code>, OpenCV expects BGR.<br>
        When displaying with <code>matplotlib</code> or <code>PIL</code>, RGB is expected.<br>
        Swapping the channels incorrectly will produce color-shifted images.
        </p>
        </div>
        """, unsafe_allow_html=True)
        rgb_img = op_bgr_to_rgb(original_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_to_rgb(original_img), caption="BGR (OpenCV default)",
                     use_container_width=True)
        with col2:
            st.image(rgb_img, caption="RGB (Standard)", use_container_width=True)
        st.info("Note: The images look identical here because Streamlit handles the conversion. "
                "The difference matters when using matplotlib or saving with PIL.")
        processed_img = rgb_img
        operation_applied = True

    # ---------------------------------------------------------------------------
    # Save Processed Image
    # ---------------------------------------------------------------------------
    if operation_applied and operation != "--- Select an operation ---":
        st.markdown("---")
        st.subheader("💾 Save Processed Image")
        img_name = selected if source == "🖼️ Sample Images" else "uploaded_image"
        default_name = f"{os.path.splitext(os.path.basename(img_name))[0]}_processed"
        save_name = st.text_input("Filename (without extension)", value=default_name)
        if st.button("💾 Save Processed Image"):
            save_path = os.path.join(OUTPUT_DIR, f"{save_name}.jpg")
            img_to_save = processed_img
            if operation == "🎨 BGR vs RGB Comparison":
                img_to_save = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img_to_save)
            st.success(f"✅ Image saved to: `{save_path}`")

else:
    st.markdown("""
    <div class="metric-card" style="text-align: center; padding: 40px;">
        <h2 style="color: #00d4ff;">👈 Please load an image from the sidebar</h2>
        <p style="color: #a0a0a0; font-size: 16px;">
            Select an image source (Upload or Sample Images) from the sidebar,<br>
            then choose an operation to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Available Operations:
    - **🎨 Convert to Grayscale** — Convert color image to grayscale
    - **📏 Resize Image** — Change image dimensions
    - **🔄 Rotate Image** — Rotate by any angle
    - **🔀 Flip Image** — Horizontal, vertical, or both
    - **✂️ Crop Image** — Select a region to crop
    - **🔷 Draw Shapes** — Rectangle, circle, line, polygon
    - **📝 Add Custom Text** — Add text with custom font, size, and color
    - **☀️ Adjust Brightness & Contrast** — Fine-tune image appearance
    - **🎨 BGR vs RGB Comparison** — Compare color channel orders
    """)
