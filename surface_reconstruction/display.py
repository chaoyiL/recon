"""Fast CPU surface rendering and explicit observed/inferred marker overlays."""

import cv2
import numpy as np
from surface import local_coordinates


def marker_overlay(frame, pixels, valid, status, segment=0):
    canvas = frame.copy()
    for idx in np.flatnonzero(valid):
        u, v = np.rint(pixels[idx]).astype(int)
        cv2.circle(canvas, (u, v), 3, (0, 210, 0), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(idx),
            (u + 4, v - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 80, 240),
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (245, 245, 245), -1)
    cv2.putText(
        canvas,
        f"{status} | observed {valid.sum()}/{len(valid)} | segment {segment}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "IDs after a reset have a new local origin; Esc: exit, R: reset",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return canvas


class SurfaceView:
    def __init__(self, model, width=640, height=480):
        self.model = model
        self.width = width
        self.height = height
        xx, yy = np.meshgrid(
            np.linspace(model.xy[:, 0].min(), model.xy[:, 0].max(), 17),
            np.linspace(model.xy[:, 1].min(), model.xy[:, 1].max(), 29),
        )
        self.uv = np.column_stack((xx.ravel(), yy.ravel()))
        self.basis = model.basis(self.uv)
        grid = np.arange(len(self.uv)).reshape(29, 17)
        self.faces = np.column_stack(
            (
                grid[:-1, :-1].ravel(),
                grid[:-1, 1:].ravel(),
                grid[1:, 1:].ravel(),
                grid[1:, :-1].ravel(),
            )
        )
        self.rotation = cv2.Rodrigues(np.array([0.95, -0.35, -0.25]))[0]
        self.scale = min(width, height) * 0.72 / max(np.ptp(model.xy, axis=0))

    def render(self, coeff, info, fps=0.0):
        canvas = np.full((self.height, self.width, 3), 247, np.uint8)
        if coeff is None:
            cv2.putText(
                canvas,
                "No current 3D estimate",
                (35, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (60, 60, 200),
                2,
            )
            cv2.putText(
                canvas,
                str(info.get("status", "waiting"))[:65],
                (15, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 40, 40),
                1,
            )
            return canvas
        points = self.model.B @ coeff
        _, rotation, origin = local_coordinates(points, self.model.xy)
        local = (self.basis @ coeff - origin) @ rotation.T
        view = local @ self.rotation.T
        pixels = np.rint(
            view[:, :2] * [self.scale, -self.scale] + [self.width * 0.5, self.height * 0.55]
        ).astype(np.int32)
        heights = local[:, 2]
        limit = max(np.ptp(self.model.xy, axis=0).max() * 0.25, 1.0)
        colorindex = np.uint8(np.clip((heights / limit + 1) * 127.5, 0, 255))
        colors = cv2.applyColorMap(colorindex.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
        for face in self.faces[np.argsort(view[self.faces, 2].mean(1))]:
            color = tuple(map(int, colors[face].mean(0)))
            cv2.fillConvexPoly(canvas, pixels[face], color, cv2.LINE_AA)
        grid = pixels.reshape(29, 17, 2)
        for line in list(grid[::4]) + list(grid[:, ::4].transpose(1, 0, 2)):
            cv2.polylines(canvas, [np.ascontiguousarray(line)], False, (75, 75, 75), 1, cv2.LINE_AA)
        axis_start = np.array([75, self.height - 65])
        for axis, label, color in [
            (0, "X", (40, 40, 220)),
            (1, "Y", (30, 160, 30)),
            (2, "Z", (220, 50, 30)),
        ]:
            delta = self.rotation[:, axis][:2] * [38, -38]
            end = np.rint(axis_start + delta).astype(int)
            cv2.arrowedLine(canvas, tuple(axis_start), tuple(end), color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas, label, tuple(end + [3, -2]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        lines = [
            "Shape with best-fit rigid pose removed (mm)",
            f"{info['status']} | reprojection {info['reprojection_rms_px']:.2f} px | metric {100*info['metric_max_error']:.1f}%",
            f"Processing throughput: {fps:.1f} FPS",
            f"Local Z: {heights.min():.2f} to {heights.max():.2f} mm; +Z = template X cross Y",
        ]
        for k, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (12, 22 + k * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            "Interpolated surface; hidden marks are model predictions",
            (12, self.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def save_obj(self, path, coeff):
        vertices = self.basis @ coeff
        with open(path, "w", encoding="utf-8") as f:
            f.write("# camera XYZ, mm; bicubic interpolation, not measured between marks\n")
            for v in vertices:
                f.write("v " + " ".join(map(str, v)) + "\n")
            for a, b, c, d in self.faces + 1:
                f.write(f"f {a} {b} {c}\nf {a} {c} {d}\n")
