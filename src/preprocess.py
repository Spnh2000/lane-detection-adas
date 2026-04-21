import cv2
import numpy as np

# ── 1. REGION OF INTEREST ──────────────────────────────────────────────────────
def region_of_interest(img):
    """Keep only the lower trapezoid where lanes appear."""
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.45), int(h * 0.6)),
        (int(w * 0.55), int(h * 0.6)),
        (int(w * 0.95), h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

# ── 2. COLOR FILTERING ─────────────────────────────────────────────────────────
def filter_lane_colors(img):
    """Isolate white and yellow lane markings."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # White lanes
    white_mask = cv2.inRange(hsv,
        np.array([0,   0,   200]),
        np.array([180, 30,  255]))

    # Yellow lanes
    yellow_mask = cv2.inRange(hsv,
        np.array([15,  80,  80]),
        np.array([35,  255, 255]))

    combined = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(img, img, mask=combined)

# ── 3. EDGE DETECTION ──────────────────────────────────────────────────────────
def detect_edges(img):
    """Grayscale → Gaussian blur → Canny edges."""
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blur, threshold1=50, threshold2=150)
    return edges

# ── 4. PERSPECTIVE WARP ────────────────────────────────────────────────────────
def warp_perspective(img):
    """Bird's-eye view transform for cleaner lane fitting."""
    h, w = img.shape[:2]
    src = np.float32([
        [int(w * 0.45), int(h * 0.6)],
        [int(w * 0.55), int(h * 0.6)],
        [int(w * 0.95), h],
        [int(w * 0.05), h],
    ])
    dst = np.float32([
        [int(w * 0.2), 0],
        [int(w * 0.8), 0],
        [int(w * 0.8), h],
        [int(w * 0.2), h],
    ])
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped, M, Minv

# ── 5. FULL PIPELINE ───────────────────────────────────────────────────────────
def preprocess(img):
    """Run the full classical CV pipeline on one frame."""
    color_filtered = filter_lane_colors(img)
    edges          = detect_edges(color_filtered)
    roi            = region_of_interest(edges)
    warped, M, Minv = warp_perspective(img)          # also return warp matrices
    warped_edges   = region_of_interest(detect_edges(filter_lane_colors(warped)))
    return {
        "color_filtered": color_filtered,
        "edges":          edges,
        "roi":            roi,
        "warped":         warped,
        "warped_edges":   warped_edges,
        "M":              M,
        "Minv":           Minv,
    }