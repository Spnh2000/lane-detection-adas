import cv2
import torch
import numpy as np
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model      import LaneNet
from preprocess import warp_perspective

# ── CONFIG ─────────────────────────────────────────────────────────────────────
IMG_SIZE  = (256, 512)   # H, W  — must match training
THRESHOLD = 0.5          # sigmoid threshold for lane/no-lane
MEAN      = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD       = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── HELPERS ────────────────────────────────────────────────────────────────────
def preprocess_frame(frame):
    """BGR frame → normalised tensor (1, 3, H, W)."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE[1], IMG_SIZE[0]))
    norm  = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor

def overlay_mask(frame, mask, color=(0, 255, 100), alpha=0.45):
    """Blend a binary lane mask back onto the original frame."""
    h, w  = frame.shape[:2]
    mask_resized = cv2.resize(mask, (w, h))
    colored      = np.zeros_like(frame)
    colored[mask_resized > THRESHOLD] = color
    return cv2.addWeighted(frame, 1.0, colored, alpha, 0)

def fit_lane_lines(mask, frame):
    """Fit polynomials to left/right lanes and draw them."""
    h, w   = mask.shape
    binary = (mask > THRESHOLD).astype(np.uint8)

    # Split into left / right halves
    left_pts  = np.argwhere(binary[:, :w//2])
    right_pts = np.argwhere(binary[:, w//2:])
    right_pts[:, 1] += w // 2   # shift x back to full-width coords

    out = frame.copy()

    for pts, color in [(left_pts, (0, 200, 255)), (right_pts, (0, 200, 255))]:
        if len(pts) < 50:        # not enough points → skip
            continue
        ys = pts[:, 0]
        xs = pts[:, 1]
        try:
            coeffs   = np.polyfit(ys, xs, deg=2)
            plot_ys  = np.linspace(h * 0.5, h - 1, 80).astype(int)
            plot_xs  = np.polyval(coeffs, plot_ys).astype(int)
            pts_draw = np.column_stack([plot_xs, plot_ys])
            cv2.polylines(out, [pts_draw], False, color, thickness=4)
        except np.linalg.LinAlgError:
            pass

    return out

def draw_hud(frame, fps):
    """Minimal ADAS heads-up display."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (200, 55), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}",      (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, "Lane Detection ADAS",  (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame

# ── MAIN INFERENCE LOOP ────────────────────────────────────────────────────────
def run(video_path, weights_path, output_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model = LaneNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Loaded weights: {weights_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    import time
    prev_time = time.time()

    print("Running inference — press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Inference ──────────────────────────────────────────────────────────
        tensor = preprocess_frame(frame).to(device)
        with torch.no_grad():
            logits = model(tensor)                  # (1,1,H,W)
        mask = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H,W) 0-1

        # ── Visualise ──────────────────────────────────────────────────────────
        result = overlay_mask(frame, mask)
        result = fit_lane_lines(mask, result)

        now      = time.time()
        cur_fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        result   = draw_hud(result, cur_fps)

        if writer:
            writer.write(result)

        cv2.imshow("Lane Detection ADAS", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved output video → {output_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   required=True,  help="Path to input video")
    parser.add_argument("--weights", required=True,  help="Path to .pth weights")
    parser.add_argument("--output",  default=None,   help="Save output video path")
    args = parser.parse_args()
    run(args.video, args.weights, args.output)