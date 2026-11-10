"""Shared gaze pipeline: DPI setup, ZED stereo camera, two-stage
(YOLO person -> head ROI -> MediaPipe FaceLandmarker) gaze feature
extraction, One Euro filter and ridge regression utilities.

Feature vector (19 dims):
    [L_h, L_v, R_h, R_v,
     L_h^2, L_v^2, L_h*L_v, R_h^2, R_v^2, R_h*R_v,
     R00, R01, R02, R10, R11, R12, R20, R21, R22]
where L/R eye vectors are the iris position relative to the eye corners
(normalized), the quadratic terms capture the nonlinear gaze-angle
response at screen edges, and R** are entries of the head rotation matrix.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ASCII-only path: MediaPipe's C library cannot open non-ASCII paths
MODEL_PATH = Path(r"C:\mediapipe_models\face_landmarker.task")

# Landmark indices for the 478-point FaceLandmarker (with iris tracking)
RIGHT_IRIS = [468, 469, 470, 471, 472]     # person's right eye: center + ring
LEFT_IRIS = [473, 474, 475, 476, 477]      # person's left eye: center + ring
RIGHT_EYE_CORNERS = (33, 133)              # (outer, inner)
LEFT_EYE_CORNERS = (263, 362)              # (outer, inner)
LEFT_EAR_PTS = (33, 160, 158, 133, 153, 144)
RIGHT_EAR_PTS = (362, 385, 387, 263, 373, 380)
EAR_BLINK_THR = 0.18

ROT_IDX = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
           (2, 0), (2, 1), (2, 2)]


def set_dpi_awareness():
    """Make pixel coords 1:1 with physical pixels (needed on Windows)."""
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class StereoCamera:
    """Auto-detect the ZED stereo camera (wide aspect + non-black image)
    and open it at HD720 side-by-side (2560x720)."""

    def __init__(self, warmup_frames=15):
        self.cap = None
        self.index = None
        self._open()

        # 720p gives large enough eyes for landmark precision
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame = None
        for _ in range(warmup_frames):
            ret, frame = self.cap.read()
        if frame is None:
            raise RuntimeError("Cannot read from stereo camera")
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        if self.width < 2560:
            print(f"Warning: requested 2560x720, actual {self.width}x{self.height}")

    def _open(self):
        for idx in range(5):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue

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
            if w / h > 2.5 and np.mean(frame) > 20:
                self.cap = cap
                self.index = idx
                print(f"Using Camera {idx} ({w}x{h})")
                return
            cap.release()
        raise RuntimeError("No stereo camera found")

    def read(self):
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()


def head_roi_from_person_box(box, eye_w):
    """Square ROI around the head from a person box (full-frame coords).
    Person boxes put the head at the top ~28% of the box."""
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(eye_w), x2)
    if x2 - x1 < 50:
        return None
    pw, ph = x2 - x1, y2 - y1
    hc_x = (x1 + x2) / 2
    hc_y = y1 + ph * 0.14
    size = max(pw * 0.7, ph * 0.34)
    return int(max(0, hc_x - size / 2)), int(max(0, hc_y - size / 2)), int(size)


def _eye_vector(iris_pts, corner_out, corner_in):
    """Iris offset relative to the eye axis: (h, v) normalized by eye width.
    h: position along outer->inner axis, v: perpendicular offset."""
    iris = np.mean(iris_pts, axis=0)
    a = np.asarray(corner_out, dtype=float)
    b = np.asarray(corner_in, dtype=float)
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-3:
        return None
    e = ab / L
    n = np.array([-e[1], e[0]])
    d = iris - a
    h = float(np.dot(d, e) / L)
    v = float(np.dot(d, n) / L)
    return h, v


def _ear(pts, ear_indices):
    """Eye Aspect Ratio from 6 landmark points."""
    p = [np.asarray(pts[i], dtype=float) for i in ear_indices]
    # vertical pairs: (1,5), (2,4); horizontal: (0,3)
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    hz = np.linalg.norm(p[0] - p[3])
    if hz < 1e-3:
        return None
    return (v1 + v2) / (2.0 * hz)


class GazePipeline:
    """Two-stage gaze feature extraction from a stereo frame.

    Uses the LEFT camera view only (a full face is visible there);
    falls back to the RIGHT view when the left view fails.
    """

    def __init__(self, project_dir, yolo_imgsz=480):
        self.project_dir = Path(project_dir)
        self.yolo_imgsz = yolo_imgsz
        self.person_boxes = []

        self.yolo = YOLO(str(self.project_dir / "yolo11n.pt"))
        _ = self.yolo.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self):
        self.landmarker.close()

    # ---- landmarks -------------------------------------------------

    def _detect_landmarks(self, eye_image):
        """Return (points, rot_matrix) or (None, None). points is a list of
        (x, y) tuples in eye_image coords."""

        def run(img):
            rgb = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB)
            mp_image = mp_image_create(rgb)
            result = self.landmarker.detect(mp_image)
            if not result.face_landmarks:
                return None, None
            lm = result.face_landmarks[0]
            mat = None
            if result.facial_transformation_matrixes:
                mat = np.asarray(
                    result.facial_transformation_matrixes[0], dtype=float
                ).reshape(4, 4)
            hh, ww = img.shape[:2]
            return [(p.x * ww, p.y * hh) for p in lm], mat

        # Stage 1: YOLO head ROI crop
        for box in self.person_boxes:
            roi = head_roi_from_person_box(box, eye_image.shape[1])
            if roi is None:
                continue
            x0, y0, size = roi
            x0, y0 = max(0, x0), max(0, y0)
            x1 = min(eye_image.shape[1], x0 + size)
            y1 = min(eye_image.shape[0], y0 + size)
            crop = eye_image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            pts, mat = run(crop)
            if pts is not None:
                pts = [(x + x0, y + y0) for (x, y) in pts]
                return pts, mat

        # Stage 2: full eye image fallback
        pts, mat = run(eye_image)
        return pts, mat

    # ---- main entry ------------------------------------------------

    def process(self, frame):
        """Extract gaze features from a stereo frame.
        Returns dict with keys:
            ok, features (np.ndarray 13,), blink (bool),
            ear (earL, earR), iris (L, R) in full-frame coords,
            corners (dict), fps_hint
        """
        h, w = frame.shape[:2]
        mid = w // 2

        t0 = time.time()
        yolo_result = self.yolo.predict(
            frame, classes=[0], conf=0.3,
            imgsz=self.yolo_imgsz, verbose=False)
        self.person_boxes = [b.xyxy[0].tolist() for b in yolo_result[0].boxes]

        out = {
            "ok": False, "features": None, "blink": False,
            "ear": (None, None), "iris": (None, None), "corners": None,
        }

        for side in ("L", "R"):
            eye_img = frame[:, :mid] if side == "L" else frame[:, mid:]
            pts, mat = self._detect_landmarks(eye_img)
            if pts is None or mat is None:
                continue

            if len(pts) < 478:
                continue

            ear_l = _ear(pts, LEFT_EAR_PTS)
            ear_r = _ear(pts, RIGHT_EAR_PTS)
            out["ear"] = (ear_l, ear_r)
            blink = False
            if ear_l is not None and ear_l < EAR_BLINK_THR:
                blink = True
            if ear_r is not None and ear_r < EAR_BLINK_THR:
                blink = True
            out["blink"] = blink

            vec_r = _eye_vector([pts[i] for i in RIGHT_IRIS],
                                pts[RIGHT_EYE_CORNERS[0]],
                                pts[RIGHT_EYE_CORNERS[1]])
            vec_l = _eye_vector([pts[i] for i in LEFT_IRIS],
                                pts[LEFT_EYE_CORNERS[0]],
                                pts[LEFT_EYE_CORNERS[1]])
            if vec_r is None or vec_l is None:
                continue

            rot = [mat[i, j] for (i, j) in ROT_IDX]
            feats = np.array(
                [vec_l[0], vec_l[1], vec_r[0], vec_r[1]]
                + [vec_l[0] ** 2, vec_l[1] ** 2, vec_l[0] * vec_l[1]]
                + [vec_r[0] ** 2, vec_r[1] ** 2, vec_r[0] * vec_r[1]]
                + rot,
                dtype=np.float64)

            x_off = 0 if side == "L" else mid
            iris_l = np.mean([pts[i] for i in LEFT_IRIS], axis=0)
            iris_r = np.mean([pts[i] for i in RIGHT_IRIS], axis=0)
            out["iris"] = ((iris_l[0] + x_off, iris_l[1]),
                           (iris_r[0] + x_off, iris_r[1]))
            out["corners"] = {
                "L_outer": (pts[LEFT_EYE_CORNERS[0]][0] + x_off, pts[LEFT_EYE_CORNERS[0]][1]),
                "L_inner": (pts[LEFT_EYE_CORNERS[1]][0] + x_off, pts[LEFT_EYE_CORNERS[1]][1]),
                "R_outer": (pts[RIGHT_EYE_CORNERS[0]][0] + x_off, pts[RIGHT_EYE_CORNERS[0]][1]),
                "R_inner": (pts[RIGHT_EYE_CORNERS[1]][0] + x_off, pts[RIGHT_EYE_CORNERS[1]][1]),
            }
            out["ok"] = True
            out["features"] = feats
            out["side_used"] = side
            break

        out["infer_ms"] = (time.time() - t0) * 1000
        return out


_mp_image_cache = None


def mp_image_create(rgb):
    global _mp_image_cache
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


class OneEuro:
    """One Euro filter for low-jitter, low-lag 1D/2D signals."""

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t=None):
        x = np.asarray(x, dtype=float)
        if self.t_prev is None:
            self.t_prev = t if t is not None else time.time()
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        now = t if t is not None else time.time()
        dt = max(1e-3, now - self.t_prev)

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = now
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


# ------------------------------------------------------------------
# Ridge regression (numpy only, no sklearn dependency)
# ------------------------------------------------------------------

def ridge_fit(X, y, alpha):
    """Centered ridge regression. X: (n, d), y: (n, 2). Returns model dict."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_mean = X.mean(axis=0)
    y_mean = y.mean(axis=0)
    Xc = X - X_mean
    yc = y - y_mean
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ yc)
    return {"X_mean": X_mean, "y_mean": y_mean, "W": W, "alpha": alpha}


def ridge_predict(model, x):
    x = np.asarray(x, dtype=float)
    return (x - model["X_mean"]) @ model["W"] + model["y_mean"]
