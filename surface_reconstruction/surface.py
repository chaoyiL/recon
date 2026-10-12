"""Smooth metric shape-from-template in mm, with free camera-relative pose.

A bicubic parametric patch is fit to image observations and differential
isometry. This is a numerical approximation, not a reproduction of a paper.
Temporal priors follow a newly estimated rigid pose, never a fixed camera plane.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from math import factorial


class SurfaceModel:
    def __init__(self, xy, camera, distortion, settings=None):
        self.xy = np.asarray(xy, float)
        self.camera = np.asarray(camera, float)
        self.distortion = np.asarray(distortion, float)
        self.options = dict(
            pixel_sigma=0.5,
            metric_sigma=0.015,
            temporal_sigma_mm=3.0,
            max_nfev=35,
            initialization_nfev=70,
            max_reprojection_px=1.5,
            max_metric_error=0.08,
        )
        self.options.update(settings or {})
        for key, value in self.options.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Solver setting {key} must be finite and positive")
        for key in ("max_nfev", "initialization_nfev"):
            if int(self.options[key]) != self.options[key]:
                raise ValueError(f"{key} must be an integer")
        self.center = self.xy.mean(0)
        self.half = np.ptp(self.xy, axis=0) / 2
        if np.any(self.half <= 0) or not np.isfinite(self.xy).all():
            raise ValueError("Template must span two physical dimensions")
        self.powers = [(i, j) for i in range(4) for j in range(4)]
        self.B = self.basis(self.xy)
        xx, yy = np.meshgrid(
            np.linspace(self.xy[:, 0].min(), self.xy[:, 0].max(), 7),
            np.linspace(self.xy[:, 1].min(), self.xy[:, 1].max(), 11),
        )
        samples = np.column_stack((xx.ravel(), yy.ravel()))
        self.Dx, self.Dy = self.basis(samples, 1, 0), self.basis(samples, 0, 1)
        self.dense_basis = self.basis(samples)
        self.previous = None
        self.alternative = None

    def basis(self, xy, dx=0, dy=0):
        u = (np.asarray(xy) - self.center) / self.half
        return np.column_stack(
            [
                (
                    (
                        factorial(i)
                        / factorial(i - dx)
                        * factorial(j)
                        / factorial(j - dy)
                        * u[:, 0] ** (i - dx)
                        * u[:, 1] ** (j - dy)
                        / self.half[0] ** dx
                        / self.half[1] ** dy
                    )
                    if i >= dx and j >= dy
                    else np.zeros(len(u))
                )
                for i, j in self.powers
            ]
        )

    def reset(self):
        self.previous = self.alternative = None

    def pose(self, points, pixels, visible):
        objects = np.ascontiguousarray(points[visible], dtype=float)
        image = np.ascontiguousarray(pixels[visible], dtype=float)
        ok, r, t = cv2.solvePnP(
            objects, image, self.camera, self.distortion, flags=cv2.SOLVEPNP_SQPNP
        )
        if not ok:
            raise ValueError("No valid pose initialization")
        r, t = cv2.solvePnPRefineLM(objects, image, self.camera, self.distortion, r, t)
        moved = points @ cv2.Rodrigues(r)[0].T + t.ravel()
        if np.min(moved[:, 2]) <= 1:
            raise ValueError("Pose crosses the camera plane")
        return moved

    def seeds(self, pixels, visible):
        xy = self.xy - self.center
        flat = np.column_stack((xy, np.zeros(len(xy))))
        shapes = [flat]
        # Both curvature signs and several bending axes; no flat initial state assumed.
        for angle in (0.0, np.pi / 2, np.pi / 4):
            axis = np.array([np.cos(angle), np.sin(angle)])
            along = xy @ axis
            perpendicular = xy @ np.array([-axis[1], axis[0]])
            radius = max(np.ptp(along) / 1.5, 1.0)
            for sign in (-1.0, 1.0):
                bent = radius * np.sin(along / radius)
                xy_bent = bent[:, None] * axis + perpendicular[:, None] * np.array(
                    [-axis[1], axis[0]]
                )
                shapes.append(
                    np.column_stack((xy_bent, sign * radius * (1 - np.cos(along / radius))))
                )
        output = []
        for shape in shapes:
            try:
                output.append(
                    np.linalg.lstsq(self.B, self.pose(shape, pixels, visible), rcond=None)[0]
                )
            except (ValueError, cv2.error):
                pass
        if not output:
            raise ValueError("No positive-depth initial shape hypothesis")
        return output

    def residual_function(self, pixels, visible, confidence, prior):
        opt = self.options
        B = self.B[visible]
        normalized = cv2.undistortPoints(
            np.ascontiguousarray(pixels[visible, None, :]), self.camera, self.distortion
        ).reshape(-1, 2)
        focal = np.array([self.camera[0, 0], self.camera[1, 1]])
        weight = (
            np.sqrt(np.maximum(confidence[visible], 0.15))[:, None] * focal / opt["pixel_sigma"]
        )
        # Weak higher-order coefficient regularization removes unsupported wrinkles.
        reg = np.array([0.015 if i + j >= 3 else 0.0 for i, j in self.powers])

        def evaluate(vector, jac=False):
            coeff = vector.reshape(16, 3)
            Q = B @ coeff
            z = np.maximum(Q[:, 2], 0.1)
            U, V = self.Dx @ coeff, self.Dy @ coeff
            if not jac:
                terms = [
                    ((Q[:, :2] / z[:, None] - normalized) * weight).ravel(),
                    (np.sum(U * U, axis=1) - 1) / 2 / opt["metric_sigma"],
                    (np.sum(V * V, axis=1) - 1) / 2 / opt["metric_sigma"],
                    np.sum(U * V, axis=1) / opt["metric_sigma"],
                    (reg[:, None] * coeff).ravel(),
                    np.maximum(1 - self.dense_basis @ coeff[:, 2], 0) * 10,
                ]
                if prior is not None:
                    terms.append(((self.B @ coeff - prior) / opt["temporal_sigma_mm"]).ravel())
                return np.concatenate(terms)
            J = np.zeros((len(B), 2, 16, 3))
            J[:, 0, :, 0] = B / z[:, None] * weight[:, 0, None]
            J[:, 1, :, 1] = B / z[:, None] * weight[:, 1, None]
            J[:, 0, :, 2] = -B * (Q[:, 0] / z**2 * weight[:, 0])[:, None]
            J[:, 1, :, 2] = -B * (Q[:, 1] / z**2 * weight[:, 1])[:, None]
            terms = [
                J.reshape(-1, 48),
                (self.Dx[:, :, None] * U[:, None, :] / opt["metric_sigma"]).reshape(-1, 48),
                (self.Dy[:, :, None] * V[:, None, :] / opt["metric_sigma"]).reshape(-1, 48),
                (
                    (self.Dx[:, :, None] * V[:, None, :] + self.Dy[:, :, None] * U[:, None, :])
                    / opt["metric_sigma"]
                ).reshape(-1, 48),
                np.diag(np.repeat(reg, 3)),
            ]
            positive = np.zeros((len(self.dense_basis), 16, 3))
            positive[:, :, 2] = (
                -10 * self.dense_basis * ((self.dense_basis @ coeff[:, 2]) < 1)[:, None]
            )
            terms.append(positive.reshape(-1, 48))
            if prior is not None:
                terms.append(
                    np.einsum("ij,ab->iajb", self.B, np.eye(3)).reshape(-1, 48)
                    / opt["temporal_sigma_mm"]
                )
            return np.vstack(terms)

        return evaluate

    def fit(self, pixels, visible=None, confidence=None, allow_trim=True):
        pixels = np.asarray(pixels, float)
        if pixels.shape != (len(self.xy), 2):
            raise ValueError("Expected one observation slot per template ID")
        visible = (
            np.isfinite(pixels).all(1)
            if visible is None
            else np.asarray(visible, bool) & np.isfinite(pixels).all(1)
        )
        if visible.sum() < max(12, int(np.ceil(0.65 * len(self.xy)))):
            raise ValueError("Too few observed marks for a reliable surface fit")
        if np.any(np.ptp(self.xy[visible], axis=0) < 0.75 * np.ptp(self.xy, axis=0)):
            raise ValueError("Observed marks do not cover the surface")
        confidence = np.ones(len(pixels)) if confidence is None else np.asarray(confidence, float)
        initializing = self.previous is None
        prior = None
        if initializing:
            seeds = self.seeds(pixels, visible)
        else:
            prior = self.pose(self.B @ self.previous, pixels, visible)
            seeds = [np.linalg.lstsq(self.B, prior, rcond=None)[0]]
            if self.alternative is not None:
                alternate = self.pose(self.B @ self.alternative, pixels, visible)
                seeds.append(np.linalg.lstsq(self.B, alternate, rcond=None)[0])
        scoring = self.residual_function(pixels, visible, confidence, None)
        results = []
        preferred = None
        for index, seed in enumerate(seeds):
            # Each hypothesis has its own pose-aligned prior; do not suppress alternatives toward the winner.
            own_prior = self.B @ seed if prior is not None else None
            evaluate = self.residual_function(pixels, visible, confidence, own_prior)
            result = least_squares(
                evaluate,
                seed.ravel(),
                jac=lambda x: evaluate(x, True),
                method="trf",
                loss="soft_l1",
                f_scale=2.0,
                max_nfev=self.options["initialization_nfev" if initializing else "max_nfev"],
                ftol=1e-5,
                xtol=1e-5,
                gtol=1e-5,
            )
            if np.isfinite(result.x).all():
                residual = scoring(result.x)
                result.cost = float(4 * np.sum(np.sqrt(1 + (residual / 2) ** 2) - 1))
                results.append(result)
                if index == 0:
                    preferred = result
        if not results:
            raise ValueError("No finite shape fit")
        results.sort(key=lambda r: r.cost)
        best = results[0]
        if prior is not None and preferred is not None and preferred.cost <= best.cost * 1.05 + 0.1:
            best = preferred
        coeff = best.x.reshape(16, 3)
        points = self.B @ coeff
        projected = cv2.projectPoints(
            points, np.zeros(3), np.zeros(3), self.camera, self.distortion
        )[0].reshape(-1, 2)
        errors = np.linalg.norm(projected[visible] - pixels[visible], axis=1)
        U, V = self.Dx @ coeff, self.Dy @ coeff
        metric = np.column_stack(
            (np.linalg.norm(U, axis=1) - 1, np.linalg.norm(V, axis=1) - 1, np.sum(U * V, axis=1))
        )
        rms = float(np.sqrt(np.mean(errors**2)))
        metric_max = float(np.max(np.abs(metric)))
        if allow_trim and rms > self.options["max_reprojection_px"]:
            median = np.median(errors)
            threshold = max(2.5, median + 3.0 * 1.4826 * np.median(np.abs(errors - median)))
            bad = np.flatnonzero(visible)[errors > threshold]
            if 0 < len(bad) <= 0.2 * visible.sum() and visible.sum() - len(bad) >= 32:
                kept = visible.copy()
                kept[bad] = False
                q, c, info = self.fit(pixels, kept, confidence, allow_trim=False)
                info["rejected_observation_ids"] = bad.tolist()
                return q, c, info
        if (
            np.min(self.dense_basis @ coeff[:, 2]) <= 1
            or rms > self.options["max_reprojection_px"]
            or metric_max > self.options["max_metric_error"]
        ):
            raise ValueError(f"Fit rejected: reprojection={rms:.2f}px, metric={metric_max:.3f}")
        ambiguous = False
        separation = 0.0
        alternate = None
        for candidate in results:
            if candidate is best:
                continue
            candidate_coeff = candidate.x.reshape(16, 3)
            other = self.B @ candidate_coeff
            separation = float(
                np.sqrt(np.mean(np.sum((align(other, points) - points) ** 2, axis=1)))
            )
            if separation > 0.5 and candidate.cost <= best.cost * 1.3 + 1.0:
                ambiguous = True
                alternate = candidate_coeff.copy()
                break
        # Keep the competing branch through the next frames; ambiguity is never a truth certificate.
        self.previous = coeff.copy()
        self.alternative = alternate
        return (
            points,
            coeff,
            dict(
                reprojection_rms_px=rms,
                metric_max_error=metric_max,
                metric_rms_error=float(np.sqrt(np.mean(metric**2))),
                status=(
                    "ambiguous"
                    if ambiguous
                    else ("estimated" if best.success else "budget_limited")
                ),
                alternative_shape_rms_mm=separation if ambiguous else None,
                initializing=initializing,
                iterations=int(best.nfev),
                optimizer_converged=bool(best.success),
                visible_count=int(visible.sum()),
                cost=float(best.cost),
            ),
        )


def align(source, target):
    """Proper rigid least-squares alignment, never a scale or reflection fit."""
    a, b = source - source.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    sign = np.eye(3)
    sign[2, 2] = np.linalg.det(u @ vt)
    return a @ (u @ sign @ vt) + target.mean(0)


def local_coordinates(points, xy):
    """Remove a best-fit rigid pose; +Z follows template X cross Y, not camera-facing."""
    flat = np.column_stack((xy - xy.mean(0), np.zeros(len(xy))))
    centered = points - points.mean(0)
    u, _, vt = np.linalg.svd(flat.T @ centered)
    sign = np.eye(3)
    sign[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ sign @ vt
    return centered @ rotation.T, rotation, points.mean(0)
