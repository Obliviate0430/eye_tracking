"""
九宫格测试
按空格开始calibration 然后格子会随机亮起 注视格子来标定
标定完成后进行测试

可能用笔记本的webcam的时候需要改一下camera编号
"""

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from gaze_core import (
    set_dpi_awareness, StereoCamera, GazePipeline, OneEuro,
    ridge_fit, ridge_predict,
)

PROJECT_DIR = Path(__file__).parent
SW, SH = 1920, 1080          # primary screen resolution
MARGIN = 70
GAP = 14
GRID = 3                     # test grid
CALIB_GRID = 5               # calibration grid (denser than test grid)
CALIB_COUNTDOWN = 0.8        # cell dims, shows number
CALIB_COLLECT = 1.2          # cell bright, samples collected
TEST_TRIALS = 12
TEST_SETTLE = 0.5            # saccade time, not counted
TEST_MEASURE = 1.7           # prediction window
TEST_FLASH = 0.7             # verdict flash
ALPHAS = [0.05, 0.2, 1.0, 5.0, 20.0]

BG = (32, 32, 32)
GRID_LINE = (90, 90, 90)
DIM_FILL = (70, 70, 45)
LIT_FILL = (40, 200, 240)        # BGR: bright yellow
PRED_FILL = (200, 120, 20)       # BGR: blue-ish
HIT_FILL = (60, 190, 60)
MISS_FILL = (40, 40, 200)
WHITE = (240, 240, 240)
GREEN = (80, 220, 80)
RED = (80, 80, 240)


def _cell_w(grid):
    return (SW - 2 * MARGIN - GAP * (grid - 1)) / grid


def _cell_h(grid):
    return (SH - 2 * MARGIN - GAP * (grid - 1)) / grid


def cell_rect(i, grid=GRID):
    row, col = divmod(i, grid)
    x0 = MARGIN + col * (_cell_w(grid) + GAP)
    y0 = MARGIN + row * (_cell_h(grid) + GAP)
    return int(x0), int(y0), int(x0 + _cell_w(grid)), int(y0 + _cell_h(grid))


def cell_center(i, grid=GRID):
    x0, y0, x1, y1 = cell_rect(i, grid)
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def px_to_cell(x, y, grid=GRID):
    col = int(np.clip((x - MARGIN) / (_cell_w(grid) + GAP), 0, grid - 1))
    row = int(np.clip((y - MARGIN) / (_cell_h(grid) + GAP), 0, grid - 1))
    return row * grid + col


def new_canvas(grid=GRID):
    cv_ = np.full((SH, SW, 3), BG, dtype=np.uint8)
    for i in range(grid * grid):
        x0, y0, x1, y1 = cell_rect(i, grid)
        cv2.rectangle(cv_, (x0, y0), (x1, y1), GRID_LINE, 2)
    return cv_


def fill_cell(cv_, i, color, border=None, thickness=2, grid=GRID):
    x0, y0, x1, y1 = cell_rect(i, grid)
    cv2.rectangle(cv_, (x0, y0), (x1, y1), color, -1)
    if border is not None:
        cv2.rectangle(cv_, (x0, y0), (x1, y1), border, thickness)


def label_cell(cv_, i, text, color=(30, 30, 30), scale=5.0, thick=8, grid=GRID):
    cx, cy = cell_center(i, grid)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(cv_, text, (int(cx - tw / 2), int(cy + th / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def put_lines(cv_, lines, org=(30, 60), color=WHITE, scale=0.8, thick=2, step=34):
    x, y = org
    for ln in lines:
        cv2.putText(cv_, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thick, cv2.LINE_AA)
        y += step


class App:
    def __init__(self):
        set_dpi_awareness()
        cv2.namedWindow("GAZE GRID", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GAZE GRID", SW, SH)
        cv2.moveWindow("GAZE GRID", 0, 0)
        cv2.setWindowProperty("GAZE GRID", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.imshow("GAZE GRID", new_canvas())
        cv2.waitKey(30)

        put_splash(new_canvas(), [
            "GAZE GRID - 3x3 precision test",
            "Loading models (ZED camera, YOLO, MediaPipe)... ~15 s",
        ])
        cv2.imshow("GAZE GRID", new_canvas())
        cv2.waitKey(10)

        self.cam = StereoCamera()
        self.pipe = GazePipeline(PROJECT_DIR)
        self.euro = OneEuro(min_cutoff=1.2, beta=0.008, d_cutoff=1.0)

        self._set_phase("INTRO")
        self.fps = 0.0
        self._fps_n = 0
        self._fps_t = time.time()

        self.calib_order = []
        self.calib_i = 0
        self.calib_feats = []      # list of per-cell feature lists
        self.model = None
        self.loocv = None

        self.trial = None
        self.results = []
        self.last_trial_flash = None

    # -------------------------------------------------------------- loop

    def run(self):
        while True:
            ret, frame = self.cam.read()
            if not ret or frame is None:
                print("camera read failed")
                continue

            pipe = self.pipe.process(frame)
            self._update_fps()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if self.phase == "INTRO":
                self._intro(key)
            elif self.phase == "CALIB":
                self._calib(pipe, key)
            elif self.phase == "REPORT":
                self._report_screen(key)
            elif self.phase == "TEST":
                self._test(pipe, key)
            elif self.phase == "FINAL":
                self._final(key)

            cv2.imshow("GAZE GRID", self._canvas)

        self.cam.release()
        self.pipe.close()
        cv2.destroyAllWindows()

    # -------------------------------------------------------------- util

    def _update_fps(self):
        self._fps_n += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self.fps = self._fps_n / (now - self._fps_t)
            self._fps_n = 0
            self._fps_t = now

    def _start_calib(self):
        n_pts = CALIB_GRID * CALIB_GRID
        self.calib_order = random.sample(range(n_pts), n_pts)
        self.calib_i = 0
        self.calib_feats = [[] for _ in range(n_pts)]
        self.model = None
        self.loocv = None
        self._set_phase("CALIB")

    def _set_phase(self, name):
        self.phase = name
        self.t_phase = time.time()
        self._canvas = new_canvas()

    # -------------------------------------------------------------- intro

    def _intro(self, key):
        if key == ord(' '):
            self._start_calib()
            return
        put_lines(self._canvas, [
            "3x3 GAZE GRID PRECISION TEST",
            "",
            "Step 1  Calibration: a point brightens - LOOK at it",
            f"         until it moves on. 25 points (5x5), "
            f"~{int(25 * (CALIB_COUNTDOWN + CALIB_COLLECT))} s total.",
            "Step 2  Test: cells light up randomly - keep looking",
            "         at the bright one. The BLUE cell = prediction.",
            "         GREEN border = hit, RED = miss.",
            "",
            "Sit naturally, keep your head still, don't move the camera.",
            "",
            "PRESS SPACE to start calibration",
            "Q quit",
        ])
        put_lines(self._canvas, [f"FPS {self.fps:.0f}   camera {self.cam.width}x{self.cam.height}"],
                  org=(30, SH - 30), color=(150, 150, 150), scale=0.6)

    # -------------------------------------------------------------- calib

    def _calib(self, pipe, key):
        if key == ord('r'):
            self._start_calib()
            return

        n_pts = CALIB_GRID * CALIB_GRID
        cell = self.calib_order[self.calib_i]
        t_in = time.time() - self.t_phase

        self._canvas = new_canvas(CALIB_GRID)
        # previous points stay dimly filled to show progress
        for j in range(self.calib_i):
            fill_cell(self._canvas, self.calib_order[j], DIM_FILL,
                      grid=CALIB_GRID)

        if t_in < CALIB_COUNTDOWN:
            fill_cell(self._canvas, cell, (110, 110, 70), grid=CALIB_GRID)
            label_cell(self._canvas, cell, str(self.calib_i + 1),
                       scale=1.6, thick=3, grid=CALIB_GRID)
            put_lines(self._canvas, [
                f"CALIBRATION  {self.calib_i + 1}/{n_pts}",
                "Get ready - look at the numbered point ...",
            ])
        else:
            fill_cell(self._canvas, cell, LIT_FILL, grid=CALIB_GRID)
            label_cell(self._canvas, cell, str(self.calib_i + 1),
                       scale=1.6, thick=3, grid=CALIB_GRID)
            put_lines(self._canvas, [
                f"CALIBRATION  {self.calib_i + 1}/{n_pts}",
                "LOOK at the bright point",
            ])
            feats = pipe["features"]
            if pipe["ok"] and not pipe["blink"] and feats is not None:
                self.calib_feats[cell].append(feats)

            if t_in >= CALIB_COUNTDOWN + CALIB_COLLECT:
                n = len(self.calib_feats[cell])
                if n < 4:
                    print(f"point {cell}: only {n} valid frames, retrying")
                    self.t_phase = time.time()   # redo same point
                    return
                self.calib_i += 1
                self.t_phase = time.time()
                if self.calib_i >= len(self.calib_order):
                    self._fit_and_report()

    def _fit_and_report(self):
        n_pts = CALIB_GRID * CALIB_GRID
        X = []
        y = []
        for i in range(n_pts):
            # median feature per dim over all valid frames of the point
            F = np.asarray(self.calib_feats[i])
            med = np.median(F, axis=0)
            X.append(med)
            y.append(cell_center(i, CALIB_GRID))
        X = np.asarray(X)
        y = np.asarray(y)

        # LOOCV: leave-one-cell-out error for each alpha
        best_alpha, best_err = None, None
        table = []
        for a in ALPHAS:
            errs = []
            for i in range(len(X)):
                mask = np.ones(len(X), dtype=bool)
                mask[i] = False
                m = ridge_fit(X[mask], y[mask], a)
                p = ridge_predict(m, X[i])
                errs.append(np.linalg.norm(p - y[i]))
            mean_err = float(np.mean(errs))
            table.append((a, mean_err))
            if best_err is None or mean_err < best_err:
                best_err, best_alpha = mean_err, a

        self.model = ridge_fit(X, y, best_alpha)
        self.loocv = {
            "alpha": best_alpha,
            "mean_err_px": best_err,
            "mean_err_pct": best_err / SW * 100,
            "per_cell_err_px": [
                float(np.linalg.norm(
                    ridge_predict(ridge_fit(
                        X[np.ones(len(X), dtype=bool) & (np.arange(len(X)) != i)],
                        y[np.arange(len(X)) != i], best_alpha), X[i]) - y[i]))
                for i in range(len(X))
            ],
            "alpha_table": table,
        }

        print("\n=== CALIBRATION REPORT ===")
        print(f"best alpha: {best_alpha}")
        for a, e in table:
            print(f"  alpha={a:<6} LOOCV mean err = {e:7.1f} px ({e/SW*100:.2f}% of screen width)")
        errs = self.loocv["per_cell_err_px"]
        for r in range(CALIB_GRID):
            parts = [f"r{r + 1}c{c + 1}:{errs[r * CALIB_GRID + c]:5.0f}"
                     for c in range(CALIB_GRID)]
            print("  " + "  ".join(parts) + "   (px)")
        corner_idx = [0, CALIB_GRID - 1, n_pts - CALIB_GRID, n_pts - 1]
        corner_err = float(np.mean([errs[i] for i in corner_idx]))
        mid_idx = [(CALIB_GRID // 2) * CALIB_GRID + 1 + CALIB_GRID // 2]
        for c in range(1, CALIB_GRID - 1):
            mid_idx.append((CALIB_GRID // 2) * CALIB_GRID + c)
        mid_err = float(np.mean([errs[i] for i in mid_idx]))
        print(f"  midpoint avg {mid_err:.0f} px vs corner avg {corner_err:.0f} px"
              f"  (corner/mid ratio {corner_err / max(mid_err, 1):.1f}x)")
        verdict = ("GOOD (<96 px) - grid test should feel accurate"
                   if best_err < 96 else
                   "MEDIOCRE/POOR (>=96 px) - expectations lowered")
        print(f"verdict: {best_err:.1f} px avg -> {verdict}\n")

        self._save_calibration()
        self._set_phase("REPORT")

    def _save_calibration(self):
        data = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "screen": [SW, SH],
            "grid": GRID,
            "calib_grid": CALIB_GRID,
            "model": {
                "alpha": self.model["alpha"],
                "X_mean": self.model["X_mean"].tolist(),
                "y_mean": self.model["y_mean"].tolist(),
                "W": self.model["W"].tolist(),
            },
            "loocv": self.loocv,
            "features_per_point_median": [
                np.median(np.asarray(self.calib_feats[i]), axis=0).tolist()
                for i in range(CALIB_GRID * CALIB_GRID)
            ],
        }
        path = PROJECT_DIR / "gaze_calibration.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"calibration saved -> {path}")

    def _report_screen(self, key):
        if key == ord(' '):
            self._start_test()
            return
        if key == ord('r'):
            self._start_calib()
            return
        loo = self.loocv
        rows = [f"alpha {loo['alpha']}"]
        for a, e in loo["alpha_table"]:
            rows.append(f"  alpha={a:<5}  {e:6.1f} px")
        rows.append("")
        rows.append(f"LOOCV mean error: {loo['mean_err_px']:.1f} px"
                    f"  ({loo['mean_err_pct']:.2f}% of screen width)")
        rows.append("  <96 px (5%)  -> good")
        rows.append("  96-154 px    -> playable")
        rows.append("  >154 px (8%) -> needs work")
        rows.append("")
        rows.append("PRESS SPACE to start the grid test")
        rows.append("R recalibrate   Q quit")
        put_lines(self._canvas, rows)
        fill_cell(self._canvas, 4, DIM_FILL)
        label_cell(self._canvas, 4, "GO", color=(220, 220, 120), scale=2.0)

    # -------------------------------------------------------------- test

    def _start_test(self):
        self.results = []
        self.trial = None
        self._next_trial()
        self._set_phase("TEST")
        self.trial_t0 = time.time()

    def _next_trial(self):
        prev = self.trial["cell"] if self.trial else -1
        cell = prev
        while cell == prev:
            cell = random.randrange(GRID * GRID)
        self.trial = {
            "cell": cell, "votes": {}, "errs": [], "n": 0,
            "verdict": None, "pred": None, "t0": time.time(),
        }

    def _test(self, pipe, key):
        if key == ord('r'):
            self._start_calib()
            return
        if len(self.results) >= TEST_TRIALS:
            self._finish()
            return

        tr = self.trial
        t_in = time.time() - tr["t0"]

        self._canvas = new_canvas()

        # live gaze dot (filtered prediction)
        if self.model is not None and pipe["ok"] and not pipe["blink"] \
                and pipe["features"] is not None:
            pt = ridge_predict(self.model, pipe["features"])
            pt = self.euro.filter(pt)
            if 0 <= pt[0] < SW and 0 <= pt[1] < SH:
                cv2.circle(self._canvas, (int(pt[0]), int(pt[1])), 10,
                           GREEN, 2)
                cv2.circle(self._canvas, (int(pt[0]), int(pt[1])), 3,
                           GREEN, -1)

        hud = [f"TRIAL {len(self.results) + 1}/{TEST_TRIALS}   "
               f"hits {sum(1 for r in self.results if r['verdict'] == 'hit')}"
               f"/{len(self.results)}"]
        done = False

        if t_in < TEST_SETTLE:
            fill_cell(self._canvas, tr["cell"], LIT_FILL)
            hud.append("LOOK at the bright cell")
        elif t_in < TEST_SETTLE + TEST_MEASURE:
            fill_cell(self._canvas, tr["cell"], LIT_FILL)
            hud.append("LOOK at the bright cell (measuring)")

            if pipe["ok"] and not pipe["blink"] and pipe["features"] is not None:
                pt = ridge_predict(self.model, pipe["features"])
                pt = self.euro.filter(pt)
                if 0 <= pt[0] < SW and 0 <= pt[1] < SH:
                    c = px_to_cell(pt[0], pt[1])
                    tr["votes"][c] = tr["votes"].get(c, 0) + 1
                    cx, cy = cell_center(tr["cell"])
                    tr["errs"].append(float(np.hypot(pt[0] - cx, pt[1] - cy)))
                    tr["n"] += 1
                    # live prediction highlight
                    if c != tr["cell"]:
                        fill_cell(self._canvas, c, PRED_FILL)
            else:
                self.euro.reset()
        elif t_in < TEST_SETTLE + TEST_MEASURE + TEST_FLASH:
            # verdict
            if tr["verdict"] is None:
                if tr["n"] >= 5:
                    pred = max(tr["votes"], key=tr["votes"].get)
                    tr["pred"] = int(pred)
                    tr["verdict"] = "hit" if pred == tr["cell"] else "miss"
                    tr["mean_err"] = float(np.mean(tr["errs"])) if tr["errs"] else None
                else:
                    tr["verdict"] = "nodata"
                    tr["pred"] = None
                    tr["mean_err"] = None
                self.results.append(tr)
                self._log_trial(tr)

            if tr["verdict"] == "hit":
                fill_cell(self._canvas, tr["cell"], HIT_FILL, border=GREEN, thickness=6)
                hud.append("HIT")
            elif tr["verdict"] == "miss":
                fill_cell(self._canvas, tr["cell"], LIT_FILL)
                fill_cell(self._canvas, tr["pred"], MISS_FILL, border=RED, thickness=6)
                hud.append(f"MISS - predicted cell {tr['pred'] + 1}")
            else:
                hud.append("NO DETECTION (face lost?)")
            err_txt = f"{tr['mean_err']:.0f} px" if tr["mean_err"] is not None else "-"
            hud.append(f"positional error: {err_txt}")
        else:
            done = True

        if done:
            self.euro.reset()
            self._next_trial()
            self.trial_t0 = time.time()

        put_lines(self._canvas, hud)
        if self.results:
            acc = 100.0 * sum(1 for r in self.results if r["verdict"] == "hit") / len(self.results)
            put_lines(self._canvas, [f"accuracy so far: {acc:.0f}%"],
                      org=(30, SH - 30), color=(150, 150, 150), scale=0.7)

    def _log_trial(self, tr):
        err = tr["mean_err"]
        print(f"trial {len(self.results):2d}: look={tr['cell']} "
              f"pred={tr['pred']} -> {tr['verdict'].upper():7s} "
              f"err={err if err is None else round(err)}px "
              f"frames={tr['n']}")

    def _fmt_results(self):
        valid = [r for r in self.results if r["verdict"] in ("hit", "miss")]
        hits = [r for r in valid if r["verdict"] == "hit"]
        errs = [r["mean_err"] for r in valid if r["mean_err"] is not None]
        acc = 100.0 * len(hits) / len(valid) if valid else 0.0
        mean_e = float(np.mean(errs)) if errs else None
        med_e = float(np.median(errs)) if errs else None
        # per-cell accuracy and confusion
        conf = np.zeros((9, 9), dtype=int)
        for r in valid:
            true_c, pred_c = r["cell"], (r["pred"] if r["pred"] is not None else -1)
            if pred_c is not None and pred_c >= 0:
                conf[true_c, pred_c] += 1
        per_cell = {}
        for c in range(9):
            rs = [r for r in valid if r["cell"] == c]
            if rs:
                per_cell[c] = 100.0 * sum(1 for r in rs if r["verdict"] == "hit") / len(rs)
        return acc, mean_e, med_e, conf, per_cell, len(valid)

    def _finish(self):
        acc, mean_e, med_e, conf, per_cell, n_valid = self._fmt_results()
        print("\n=== GRID TEST REPORT ===")
        print(f"valid trials: {n_valid}/{TEST_TRIALS}")
        print(f"cell accuracy: {acc:.1f}%")
        if mean_e is not None:
            print(f"positional error: mean {mean_e:.0f} px | median {med_e:.0f} px"
                  f" | ({mean_e / SW * 100:.2f}% of screen width)")
        print("confusion matrix (rows=true, cols=predicted; labels 1..9):")
        print("      " + " ".join(f"{j + 1:4d}" for j in range(9)))
        for i in range(9):
            print(f"  {i + 1}:  " + " ".join(f"{conf[i, j]:4d}" for j in range(9)))
        for c, a in per_cell.items():
            print(f"  cell {c} (r{1 + c // GRID}c{1 + c % GRID}): {a:.0f}%")

        report = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_trials": len(self.results),
            "n_valid": n_valid,
            "accuracy_pct": acc,
            "mean_pos_err_px": mean_e,
            "median_pos_err_px": med_e,
            "mean_pos_err_pct": (mean_e / SW * 100) if mean_e is not None else None,
            "confusion": conf.tolist(),
            "per_cell_accuracy": per_cell,
            "trials": [
                {"look": r["cell"], "pred": r["pred"], "verdict": r["verdict"],
                 "mean_err_px": r["mean_err"], "frames": r["n"]}
                for r in self.results
            ],
            "fps": self.fps,
        }
        path = PROJECT_DIR / "gaze_test_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report saved -> {path}\n")

        self._final_summary = (acc, mean_e, med_e, n_valid)
        self._set_phase("FINAL")

    def _final(self, key):
        acc, mean_e, med_e, n_valid = self._final_summary
        rows = [
            "GRID TEST FINISHED",
            "",
            f"valid trials: {n_valid}",
            f"cell accuracy: {acc:.1f}%",
        ]
        if mean_e is not None:
            rows += [
                f"positional error: mean {mean_e:.0f} px  (median {med_e:.0f} px)",
                f"                 = {mean_e / SW * 100:.2f}% of screen width",
            ]
        rows += ["", " SPACE run test again",
                 " R recalibrate",
                 " Q quit"]
        put_lines(self._canvas, rows)
        if key == ord(' '):
            self._start_test()
        elif key == ord('r'):
            self._start_calib()


def put_splash(cv_, lines):
    put_lines(cv_, lines, org=(60, 200), scale=1.0)


if __name__ == "__main__":
    App().run()
