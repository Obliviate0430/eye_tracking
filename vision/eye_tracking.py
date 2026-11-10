
import time

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. Camera initialization
# Auto-detect the ZED stereo camera: wide
# aspect ratio + non-black image.
# ==========================================


def find_stereo_camera(max_index=5):
    """Auto-detect the ZED stereo camera: wide aspect ratio + non-black image."""
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            continue

        # Read some frames to let auto exposure settle
        frame = None
        for _ in range(10):
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = None
                break

        if frame is None:
            cap.release()
            continue

        h, w = frame.shape[:2]
        brightness = np.mean(frame)

        if w / h > 2.5 and brightness > 20:
            print(f"Using Camera {idx}")
            print("Resolution:", frame.shape)
            print("Brightness:", brightness)
            return cap

        cap.release()

    raise RuntimeError("No stereo camera found")


cap = find_stereo_camera()

# ZED WVGA (1344x376) gives too few pixels per face for MediaPipe's
# internal detector. Request HD720 side-by-side: 2560x720 total,
# 1280x720 per eye.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
if not ret or frame is None:
    raise RuntimeError("Cannot read from stereo camera")

fh, fw = frame.shape[:2]
print(f"Requested 2560x720, actual: {fw}x{fh}")
if fw < 2560:
    print("Warning: camera did not accept 720p, faces may be too small to detect")


# ==========================================
# 2. Model initialization
# NOTE: MediaPipe's C library cannot open
# paths containing non-ASCII characters,
# so the model must stay under C:\mediapipe_models.
# ==========================================

MODEL_PATH = Path(
    r"C:\mediapipe_models\face_landmarker.task"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.4
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)

print("FaceLandmarker initialized successfully!")

# YOLO person detector (stage 1).
# MediaPipe's internal face detector needs the face to fill >=20% of
# the image width; the ZED's wide FOV puts it at ~15%, below the
# detection limit. So: YOLO finds the person -> head ROI crop ->
# FaceLandmarker runs on the enlarged crop.
yolo = YOLO("yolo11n.pt")

# Warm up YOLO so the first live frame is not slow
_ = yolo.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)
print("YOLO initialized successfully!")


# ==========================================
# 3. Landmark indices
# ==========================================

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

EYE_CORNERS = [33, 133, 362, 263]


# ==========================================
# 4. Head ROI from YOLO person box
# ==========================================

def head_roi_from_person_box(box, eye_w):
    """Return (x0, y0, size) square ROI around the head from a person
    box in full-frame coords, or None if it does not overlap [0, eye_w]."""
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(eye_w, x2)
    if x2 - x1 < 50:
        return None

    pw, ph = x2 - x1, y2 - y1
    hc_x = (x1 + x2) / 2
    hc_y = y1 + ph * 0.14
    size = max(pw * 0.7, ph * 0.34)

    return int(hc_x - size / 2), int(hc_y - size / 2), int(size)


# ==========================================
# 5. Face landmark detection on one eye image
# ==========================================

def detect_eye_landmarks(eye_image, x_offset=0):
    """Detect face landmarks in one eye image. Returns list of
    (x, y) points in display-frame coords, or None."""

    # Try YOLO head ROI first
    roi_box = None
    for b in person_boxes:
        roi = head_roi_from_person_box(b, eye_image.shape[1])
        if roi is None:
            continue
        x0, y0, size = roi
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(eye_image.shape[1], x0 + size)
        y1 = min(eye_image.shape[0], y0 + size)
        crop = eye_image[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb)
        )
        result = landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            ch, cw = crop.shape[:2]
            # map crop coords -> eye coords -> display coords
            points = []
            for p in landmarks:
                points.append((
                    int(p.x * cw) + x0 + x_offset,
                    int(p.y * ch) + y0
                ))
            return points

    # Fallback: try the full eye image (works when the face is close)
    rgb = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb)
    )
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        eh, ew = eye_image.shape[:2]
        points = []
        for p in landmarks:
            points.append((
                int(p.x * ew) + x_offset,
                int(p.y * eh)
            ))
        return points

    return None


def draw_eye(display, points, offset_x=0):
    """Draw eye corners and iris centers from landmark points."""

    # Eye corners
    for idx in EYE_CORNERS:
        x, y = points[idx]
        cv2.circle(display, (x, y), 4, (255, 0, 0), -1)

    # Iris centers
    for iris_indices in [LEFT_IRIS, RIGHT_IRIS]:
        if max(iris_indices) >= len(points):
            continue
        pts = np.array([points[i] for i in iris_indices])
        center = pts.mean(axis=0).astype(int)
        cv2.circle(display, tuple(center), 6, (0, 255, 0), -1)


# ==========================================
# 6. Real-time processing
# ==========================================

cv2.namedWindow(
    "ZED Eye Tracking",
    cv2.WINDOW_NORMAL
)

frame_count = 0
fps_t0 = time.time()
fps_display = 0.0

try:

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to read frame")
            break

        frame_count += 1

        # Keep original frame for display
        display_frame = frame.copy()

        h, w = frame.shape[:2]

        # FPS
        now = time.time()
        if now - fps_t0 > 1.0:
            fps_display = frame_count / (now - fps_t0) if frame_count else 0
            # reset counters relative to this window
            frame_count_window = 0
            fps_t0 = now
            frame_count = 0

        # Stage 1: YOLO person detection on the full stereo frame
        yolo_result = yolo.predict(
            frame,
            classes=[0],
            conf=0.3,
            imgsz=480,
            verbose=False
        )
        person_boxes = [
            b.xyxy[0].tolist() for b in yolo_result[0].boxes
        ]

        # Debug info every 60 frames
        if frame_count % 60 == 0:
            print(
                f"Frame {frame_count} | "
                f"persons: {len(person_boxes)} | "
                f"FPS: {fps_display:.1f}"
            )

        detected_points = None

        # Split side-by-side stereo image and process each eye
        if w / h > 2.5:

            mid = w // 2

            left_image = frame[:, :mid]
            right_image = frame[:, mid:]

            points_left = detect_eye_landmarks(
                left_image,
                x_offset=0
            )

            points_right = detect_eye_landmarks(
                right_image,
                x_offset=mid
            )

            if points_left is not None:
                draw_eye(display_frame, points_left, offset_x=0)
            if points_right is not None:
                draw_eye(display_frame, points_right, offset_x=mid)

            detected = (
                points_left is not None or points_right is not None
            )

        else:
            detected = False

        # Display detection status
        if detected:
            status = "Face detected"
            color = (0, 255, 0)
        else:
            status = "No face detected"
            color = (0, 0, 255)

        cv2.putText(
            display_frame,
            f"{status} | {fps_display:.1f} FPS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        cv2.imshow(
            "ZED Eye Tracking",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

finally:

    cap.release()
    landmarker.close()

    cv2.destroyAllWindows()
