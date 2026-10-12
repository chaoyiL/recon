"""Frame-local grid initialization and persistent video IDs, independent of 3D pose."""

from dataclasses import dataclass
from itertools import combinations
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import markers as detector


def detect_candidates(gray):
    roi = detector.find_inner_roi(gray)
    centers, _, _ = detector.detect_cross_centers(gray, roi)
    return centers, roi


def select_complete_grid(centers, rows, cols, direction="auto"):
    """Reject up to two outliers only if one complete grid is clearly supported.

    This is initialization/acquisition, not a per-video-frame ID sorter.
    Rejects missing/ambiguous grids instead of fabricating points.
    """
    centers = np.asarray(centers, float)
    expected = rows * cols
    extra = len(centers) - expected
    if extra < 0 or extra > 2:
        raise ValueError(
            f"Need a complete {expected}-point grid; observed {len(centers)} candidates"
        )
    template = np.array([(c, r) for r in range(rows) for c in range(cols)], float)
    centered = (template - template.mean(0)) / np.ptp(template, axis=0)
    basis = np.column_stack(
        [centered[:, 0] ** i * centered[:, 1] ** j for i in range(4) for j in range(4)]
    )
    inverse = np.linalg.pinv(basis)
    leverage = np.sum(basis * inverse.T, axis=1)
    hypotheses = []
    for dropped in combinations(range(len(centers)), extra):
        kept = np.delete(np.arange(len(centers)), dropped)
        try:
            ordered = detector.sort_grid_points(centers[kept], rows, cols, direction)[:, 1:]
        except RuntimeError:
            continue
        h, _ = cv2.findHomography(template, ordered, 0)
        if h is None:
            continue
        distances = np.linalg.norm(ordered[:, None] - ordered[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        pitch = float(np.median(distances.min(axis=1)))
        # A smooth image warp allows a bent first frame; homography is only a degeneracy check.
        warp_error = np.linalg.norm(ordered - basis @ (inverse @ ordered), axis=1) / np.maximum(
            1 - leverage, 0.1
        )
        warp_rms = float(np.sqrt(np.mean(warp_error**2)))
        if extra == 0 or (warp_rms < 0.15 * pitch and warp_error.max() < 0.6 * pitch):
            hypotheses.append((warp_rms, ordered.copy(), list(dropped)))
    hypotheses.sort(key=lambda v: v[0])
    if not hypotheses:
        raise ValueError("No geometrically consistent complete grid; inspect the marker detections")
    if len(hypotheses) > 1 and hypotheses[1][0] - hypotheses[0][0] < 0.15:
        raise ValueError("Ambiguous outlier removal: competing grid assignments")
    rms, ordered, dropped = hypotheses[0]
    return np.ascontiguousarray(ordered), {
        "candidate_count": len(centers),
        "rejected_candidates": centers[dropped].tolist(),
        "grid_warp_crossvalidation_rms_px": rms,
    }


def require_regular_template(ids, xy):
    rows, cols = len(np.unique(xy[:, 1])), len(np.unique(xy[:, 0]))
    expected = np.array([(x, y) for y in np.unique(xy[:, 1]) for x in np.unique(xy[:, 0])])
    if (
        rows < 2
        or cols < 2
        or not np.array_equal(ids, np.arange(rows * cols))
        or not np.array_equal(xy, expected)
    ):
        raise ValueError("Grid acquisition requires a complete row-major rectangular template")
    return rows, cols


@dataclass
class TrackingSeed:
    ids: np.ndarray
    pixels: np.ndarray
    gray: np.ndarray

    @classmethod
    def build(cls, frame, ids, xy, direction="auto"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        rows, cols = require_regular_template(ids, xy)
        centers, roi = detect_candidates(gray)
        pixels, meta = select_complete_grid(centers, rows, cols, direction)
        return cls(ids.copy(), pixels, gray), roi, meta


class MarkerTracker:
    """Persistent IDs using forward/backward LK plus one-to-one marker matching.

    Missing points remain invalid. Nearby reacquisition uses a motion prediction
    supported by already tracked IDs; never renumbers a newly detected grid.
    """

    def __init__(self, seed, fb_threshold=1.0, gate_fraction=0.35):
        self.seed = seed
        self.previous_gray = seed.gray.copy()
        self.positions = seed.pixels.copy()
        self.valid = np.ones(len(self.positions), bool)
        self.fb_threshold = fb_threshold
        self.gate_fraction = gate_fraction
        self.failure_count = 0

    def update(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.shape != self.seed.gray.shape:
            raise ValueError("Frame resolution changed; reinitialize with matching calibration")
        n = len(self.positions)
        valid = np.zeros(n, bool)
        confidence = np.zeros(n, float)
        pixels = np.full((n, 2), np.nan)
        fb = np.full(n, np.inf)
        # Established IDs give a moving search window; do not rediscover the border every frame.
        lo = np.maximum(np.floor(self.positions.min(0) - 30), 0).astype(int)
        hi = np.minimum(np.ceil(self.positions.max(0) + 30), [gray.shape[1], gray.shape[0]]).astype(
            int
        )
        roi = (*lo, *(hi - lo))
        candidates, _, _ = detector.detect_cross_centers(gray, roi)
        active = np.flatnonzero(self.valid)
        predicted = self.positions.copy()
        reliable = np.zeros(n, bool)
        distances = np.linalg.norm(self.positions[:, None] - self.positions[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        pitch = float(np.median(distances.min(axis=1)))
        if len(active):
            p = np.ascontiguousarray(self.positions[active, None, :], dtype=np.float32)
            q, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                p,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if q is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.previous_gray, q, None, winSize=(21, 21), maxLevel=3
                )
                if back is not None:
                    fb[active] = np.linalg.norm(back.reshape(-1, 2) - p.reshape(-1, 2), axis=1)
                    finite = np.isfinite(q.reshape(-1, 2)).all(1)
                    predicted[active] = np.where(
                        finite[:, None], q.reshape(-1, 2), self.positions[active]
                    )
                    reliable[active] = (
                        (status.ravel() > 0)
                        & (back_status.ravel() > 0)
                        & (fb[active] < self.fb_threshold)
                        & finite
                    )
        gate = self.gate_fraction * pitch
        used = set()
        good = np.flatnonzero(reliable)
        if len(good) and len(candidates):
            costs = np.linalg.norm(predicted[good, None] - candidates[None], axis=2)
            # Dummy columns permit unmatched IDs, rather than forcing bad matches.
            rr, cc = linear_sum_assignment(
                np.column_stack((costs, np.full((len(good), len(good)), gate)))
            )
            for r, c in zip(rr, cc):
                if c < len(candidates) and costs[r, c] < gate:
                    idx = good[r]
                    pixels[idx] = candidates[c]
                    valid[idx] = True
                    used.add(int(c))
                    confidence[idx] = max(0.1, 1 - fb[idx] / self.fb_threshold) * max(
                        0.1, 1 - costs[r, c] / gate
                    )
        # Predict missing IDs from trusted motion; accept only actual, unambiguous detections.
        trusted = np.flatnonzero(valid)
        if len(trusted) >= 8:
            h, inliers = cv2.findHomography(
                self.positions[trusted], pixels[trusted], cv2.RANSAC, 2.0
            )
            if (
                h is not None
                and np.isfinite(h).all()
                and inliers is not None
                and inliers.sum() >= 8
            ):
                inferred = cv2.perspectiveTransform(self.positions[:, None, :], h).reshape(-1, 2)
                lost = np.flatnonzero(~valid)
                available = np.array([j for j in range(len(candidates)) if j not in used], int)
                if len(lost) and len(available):
                    costs = np.linalg.norm(
                        inferred[lost, None] - candidates[available][None], axis=2
                    )
                    rr, cc = linear_sum_assignment(
                        np.column_stack((costs, np.full((len(lost), len(lost)), gate * 0.7)))
                    )
                    for r, c in zip(rr, cc):
                        alternatives = np.sort(costs[r])
                        if (
                            c < len(available)
                            and costs[r, c] < gate * 0.7
                            and (
                                len(alternatives) < 2
                                or alternatives[1] - alternatives[0] > gate * 0.25
                            )
                        ):
                            idx = lost[r]
                            pixels[idx] = candidates[available[c]]
                            valid[idx] = True
                            confidence[idx] = 0.35
                use_prediction = (~valid) & np.isfinite(inferred).all(axis=1)
                predicted[use_prediction] = inferred[use_prediction]
        # Advance only with enough evidence; a lost frame does not silently reset IDs.
        if valid.sum() >= 8:
            self.positions = np.where(valid[:, None], pixels, predicted)
            self.previous_gray = gray.copy()
            self.valid = valid.copy()
            self.failure_count = 0
        else:
            self.failure_count += 1
        return {
            "ids": self.seed.ids.copy(),
            "pixels": pixels,
            "valid": valid,
            "confidence": confidence,
            "fb_error_px": fb,
            "candidate_count": len(candidates),
            "roi": roi,
            "status": "ok" if valid.all() else ("partial" if valid.sum() >= 8 else "lost"),
            "consecutive_lost_frames": self.failure_count,
        }
