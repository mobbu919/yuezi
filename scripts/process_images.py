#!/usr/bin/env python3
"""
process_images.py — Automated painting detection, cropping, and framing.

For each raw photo in resource/yuezi_art/:
  1. Detect the painting rectangle (largest rectangular contour)
  2. Perspective-correct and crop to just the painting
  3. Add a clean white matte border (simulating a white frame)
  4. Save the processed image to src/assets/images/
  5. Detect Chinese/English text annotations and log them

Usage:
  python3 scripts/process_images.py

Output:
  - Updated images in src/assets/images/art-XX.jpg
  - Updated src/data/gallery.js with descriptions
  - text_annotations/ folder with extracted text regions (if any)
"""

import cv2
import numpy as np
import os
import re

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCE_DIR = os.path.join(PROJECT_ROOT, 'resource', 'yuezi_art')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'src', 'assets', 'images')
GALLERY_DATA = os.path.join(PROJECT_ROOT, 'src', 'data', 'gallery.js')
TEXT_DIR     = os.path.join(PROJECT_ROOT, 'scripts', 'text_annotations')

MAX_DIM = 1600
WHITE_BORDER_FRAC = 0.015


# ── Rectangle Detection ───────────────────────────────────────────────

def find_painting_contour(img_bgr):
    """
    Find the painting rectangle within a photo.

    Uses two strategies (Otsu thresholding and Canny edge detection)
    and picks the best detection. Returns (4 corner points, coverage).
    """
    h, w = img_bgr.shape[:2]

    # Downscale for faster processing
    scale = min(800 / max(h, w), 1.0)
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sh, sw = gray.shape

    best_box = None
    best_cov = 0

    # ── Strategy 1: Otsu thresholding ──
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    for t in [thresh, cv2.bitwise_not(thresh)]:
        contours, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        cov = area / (sh * sw)
        if 0.15 < cov < 0.98 and cov > best_cov:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            best_box = box / scale
            best_cov = cov

    # ── Strategy 2: Canny edge detection ──
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = np.median(blur)

    for sigma in [0.3, 0.5, 0.8, 1.0, 1.5]:
        low = int(max(0, sigma * med))
        high = int(min(255, 2.5 * med))
        edges = cv2.Canny(blur, low, high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box_cov = cv2.contourArea(box) / (sh * sw)
            if 0.15 < box_cov < 0.98 and box_cov > best_cov:
                best_box = box / scale
                best_cov = box_cov

    if best_box is not None:
        if best_cov < 0.35:
            return None, best_cov
        return best_box.astype(np.float32), best_cov
    return None, 0.0


def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def crop_and_rectify(img_rgb, pts):
    """
    Given four corner points, perform a perspective transform to get
    a rectified (front-facing) view of the painting.
    """
    if pts is None:
        return img_rgb

    pts = order_points(pts)
    (tl, tr, br, bl) = pts

    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bot))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img_rgb, M, (max_width, max_height),
                                  flags=cv2.INTER_LANCZOS4)
    return warped


# ── White Border ──────────────────────────────────────────────────────

def add_white_border(img_rgb):
    """
    Add a clean white matte border around the painting, simulating a
    white picture frame / mat.
    """
    h, w = img_rgb.shape[:2]
    border = max(int(max(h, w) * WHITE_BORDER_FRAC), 15)

    result = cv2.copyMakeBorder(img_rgb, border, border, border, border,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Subtle inner shadow line for depth
    h2, w2 = result.shape[:2]
    cv2.rectangle(result, (border - 1, border - 1),
                  (w2 - border + 1, h2 - border + 1),
                  (200, 200, 200), 1)

    return result


# ── Resize ────────────────────────────────────────────────────────────

def resize_to_max(img_rgb, max_dim):
    """Resize so the longest side is max_dim, preserving aspect ratio."""
    h, w = img_rgb.shape[:2]
    if max(h, w) <= max_dim:
        return img_rgb
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


# ── Text Detection ────────────────────────────────────────────────────

def has_text_in_margins(img_rgb, margin_frac=0.15):
    """
    Check if the margins (especially bottom) contain text-like patterns.
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    margin_h = int(h * margin_frac)
    bottom_margin = gray[h - margin_h:, :]

    thresh = cv2.adaptiveThreshold(bottom_margin, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    text_like = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        area = cw * ch
        margin_area = margin_h * w

        if area > 0.005 * margin_area and aspect < 6 and ch > 8 and cw > 10:
            text_like += 1

    return text_like > 8


def extract_text_region(img_rgb):
    """Extract the bottom portion of the image where text annotations live."""
    h, w = img_rgb.shape[:2]
    text_h = int(h * 0.2)
    return img_rgb[h - text_h:, :, :]


def ocr_text(img_rgb):
    """
    Attempt OCR on the image to extract text.
    Tries pytesseract if available.
    """
    try:
        import pytesseract
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(thresh, lang='eng').strip()
        if text and len(text) > 3:
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        try:
            langs = pytesseract.get_languages()
            if 'chi_sim' in langs:
                text_cn = pytesseract.image_to_string(thresh, lang='chi_sim+eng').strip()
                if text_cn and len(text_cn) > 3:
                    text_cn = re.sub(r'\s+', ' ', text_cn).strip()
                    return text_cn
        except:
            pass

        return ""
    except ImportError:
        return ""
    except Exception as e:
        print(f"         OCR error: {e}")
        return ""


# ── Gallery.js Update ────────────────────────────────────────────────

def save_text_annotations(text_images, text_descriptions):
    """Save detected text annotations to a report file for manual review."""
    report_path = os.path.join(TEXT_DIR, 'annotations_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Text Annotations Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        if not text_images:
            f.write("No text annotations detected.\n")
            print("\n  No text annotations to report")
            return
        
        f.write(f"Total images with detected text: {len(text_images)}\n\n")
        for art_id_num, fn in sorted(text_images):
            aid = f"art-{art_id_num:02d}"
            desc = text_descriptions.get(aid, "")
            f.write(f"{aid} ({fn}):\n")
            if desc:
                f.write(f"  OCR text: {desc}\n")
            else:
                f.write(f"  OCR failed - see text_annotations/{aid}_text.jpg\n")
            f.write("\n")
    
    print(f"\n  Saved text annotations report to text_annotations/annotations_report.txt")


def escape_js_string(s):
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('\r', '\\r')
    s = s.replace('\t', '\\t')
    return s


# ── Main ──────────────────────────────────────────────────────────────

def process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)

    res_files = sorted([
        f for f in os.listdir(RESOURCE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        and not f.startswith('.')
    ])

    print(f"Found {len(res_files)} resource images.\n")

    text_descriptions = {}
    text_images = []

    for idx, filename in enumerate(res_files, 1):
        art_id = f"art-{idx:02d}"
        input_path = os.path.join(RESOURCE_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"{art_id}.jpg")

        print(f"[{art_id}] Processing: {filename} ...", end=" ")

        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            print("FAILED (cannot read)")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_bgr.shape[:2]

        # Step 1: Detect painting rectangle
        painting_pts, coverage = find_painting_contour(img_bgr)

        if painting_pts is not None:
            cropped = crop_and_rectify(img_rgb, painting_pts)
            print(f"detected ({coverage:.0%} coverage),", end=" ")
        else:
            cropped = img_rgb.copy()
            print(f"full-image,", end=" ")

        # Step 2: Add white matte border
        framed = add_white_border(cropped)

        # Step 3: Resize for web
        final = resize_to_max(framed, MAX_DIM)

        # Step 4: Save as JPEG
        final_bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, final_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        fh, fw = final.shape[:2]
        print(f"saved ({fw}x{fh})")

        # Step 5: Check for text
        if has_text_in_margins(img_rgb):
            text_images.append((idx, filename))
            print(f"         Text detected in margins")
            text = ocr_text(img_rgb)
            if text:
                text_descriptions[art_id] = text
                print(f"         OCR: \"{text[:80]}\"")
            else:
                tr = extract_text_region(img_rgb)
                text_out = os.path.join(TEXT_DIR, f"{art_id}_text.jpg")
                cv2.imwrite(text_out, cv2.cvtColor(tr, cv2.COLOR_RGB2BGR))
                print(f"         Text region saved to text_annotations/")

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Total images: {len(res_files)}")
    print(f"  Images with detected text: {len(text_images)}")
    if text_images:
        print(f"  Text images:")
        for art_id_num, fn in text_images:
            aid = f"art-{art_id_num:02d}"
            desc = text_descriptions.get(aid, "OCR failed")
            print(f"    {aid} ({fn}): {desc[:60]}")

    save_text_annotations(text_images, text_descriptions)


if __name__ == '__main__':
    process_all()
