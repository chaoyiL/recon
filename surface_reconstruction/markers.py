"""Cross detection and conservative grid acquisition; no standalone entry."""

import cv2
import numpy as np

LOCAL_BLOCK_SIZE = 15
LOCAL_THRESHOLD_C = 15
MIN_PEAK_CONTRAST = 60
MIN_AREA = 3
MAX_AREA = 300
MIN_WIDTH = 2
MAX_WIDTH = 30
MIN_HEIGHT = 2
MAX_HEIGHT = 30


def _detect_components(gray, roi_rect):
    """Local threshold; no fixed inset and no erosion of thin cross arms."""
    x, y, w, h = map(int, roi_rect)
    if x < 0 or y < 0 or w < 3 or h < 3 or x + w > gray.shape[1] or y + h > gray.shape[0]:
        raise ValueError("ROI must be a positive rectangle inside the image")
    # Compute on the full image so the ROI boundary cannot affect thresholding.
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        LOCAL_BLOCK_SIZE,
        LOCAL_THRESHOLD_C,
    )
    contrast = cv2.morphologyEx(
        gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    crop = binary[y : y + h, x : x + w].copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    centers, debug = [], []
    for label in range(1, n):
        left, top, width, height, area = map(int, stats[label])
        if not (
            MIN_AREA <= area <= MAX_AREA
            and MIN_WIDTH <= width <= MAX_WIDTH
            and MIN_HEIGHT <= height <= MAX_HEIGHT
        ):
            continue
        # Partial border fragments can look like small crosses after ROI clipping.
        if left == 0 or top == 0 or left + width >= w or top + height >= h:
            continue
        if not 0.15 <= width / height <= 6.0:
            continue
        component = labels[top : top + height, left : left + width] == label
        patch = contrast[y + top : y + top + height, x + left : x + left + width]
        peak = float(patch[component].max())
        if peak < MIN_PEAK_CONTRAST:
            continue
        yy, xx = np.nonzero(component)
        # Local darkness weights reduce the effect of nonuniform illumination.
        weights = patch[component].astype(float)
        cx = x + left + float(np.average(xx, weights=weights))
        cy = y + top + float(np.average(yy, weights=weights))
        centers.append([cx, cy])
        debug.append(
            dict(area=area, width=width, height=height, peak_contrast=peak, center=(cx, cy))
        )
    return np.asarray(centers, dtype=float).reshape(-1, 2), crop, debug


def find_inner_roi(gray):
    """Find the framed paper interior even when glare breaks a global mask.

    Closing is applied only to the frame search, never to mark localization.
    A bounding rectangle retains marks in shadows that indent the white contour.
    """
    local = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 7
    )
    global_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    candidates = {}
    for binary in (local, global_mask):
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for idx, contour in enumerate(contours):
            parent = hierarchy[0, idx, 3]
            area = cv2.contourArea(contour)
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            if parent < 0 or not 1000 < area < 0.8 * gray.size or min(w, h) < 35:
                continue
            if cv2.contourArea(contours[parent]) > 8 * area:
                continue
            if rect not in candidates:
                centers, _, _ = _detect_components(gray, rect)
                if len(centers) >= 8:
                    candidates[rect] = (len(centers), -area)
    if not candidates:
        raise RuntimeError(
            "Paper border not found; show the complete marked region for initialization"
        )
    return max(candidates, key=candidates.get)


def detect_cross_centers(gray, roi_rect):
    """Return observed centres, binary ROI and component diagnostics."""
    return _detect_components(gray, roi_rect)


def sort_grid_points(centers, rows, cols, row_direction="auto"):
    """Assign a complete grid without assuming that the paper is upright.

    Tests both PCA axes using row separation and cell orientation. Auto uses
    positive image x/y as the direction of increasing template rows; identical
    crosses cannot resolve a physical 180-degree rotation. No points are invented.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (rows * cols, 2) or not np.isfinite(centers).all():
        raise RuntimeError(
            f"检测到 {len(centers)} 个 mark，模板要求 {rows*cols} 个；请查看候选点图"
        )
    if len(np.unique(centers, axis=0)) != len(centers):
        raise RuntimeError("检测点存在重复坐标")
    directions = {
        "down": np.array([0.0, 1.0]),
        "up": np.array([0.0, -1.0]),
        "right": np.array([1.0, 0.0]),
        "left": np.array([-1.0, 0.0]),
    }
    if row_direction != "auto" and row_direction not in directions:
        raise ValueError("row_direction must be auto/down/up/right/left")
    _, _, axes = np.linalg.svd(centers - centers.mean(0), full_matrices=False)
    candidates = []
    for vertical in axes:
        vertical = vertical.copy()
        if row_direction == "auto":
            vertical *= 1 if vertical[np.argmax(np.abs(vertical))] >= 0 else -1
        else:
            alignment = vertical @ directions[row_direction]
            if abs(alignment) < np.cos(np.pi / 4):
                continue
            vertical *= 1 if alignment >= 0 else -1
        horizontal = np.array([vertical[1], -vertical[0]])
        order = np.argsort(centers @ vertical).reshape(rows, cols)
        groups = centers[order] @ vertical
        pitches = np.diff(groups.mean(axis=1))
        if len(pitches) and (
            np.min(pitches) <= 0 or np.max(np.ptp(groups, axis=1) / np.median(pitches)) > 0.65
        ):
            continue
        for r in range(rows):
            order[r] = order[r, np.argsort(centers[order[r]] @ horizontal)]
        grid = centers[order]
        column_edges = np.diff(grid, axis=1)
        row_edges = np.diff(grid, axis=0)
        if rows > 1 and cols > 1:
            # Both triangles of each cell must retain a consistent orientation.
            cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
            if np.any(cross(column_edges[:-1], row_edges[:, :-1]) <= 0) or np.any(
                cross(column_edges[1:], row_edges[:, 1:]) <= 0
            ):
                continue
        # Compare dispersion of each row across the candidate row axis.
        score = float(np.mean(np.ptp(groups, axis=1)) / (np.median(pitches) if len(pitches) else 1))
        candidates.append((score, order.ravel()))
    if not candidates:
        raise RuntimeError(
            "网格编号不可靠：请检查漏检、误检并重新拍摄清晰网格；强折叠不可直接按位置编号"
        )
    _, order = min(candidates, key=lambda item: item[0])
    return np.column_stack((np.arange(len(centers)), centers[order]))
