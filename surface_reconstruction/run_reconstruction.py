"""Single entry: camera/video/still reconstruction without a reference frame."""

import os

# Tiny least-squares systems are much faster without oversubscribed BLAS threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
import threading
import cv2
import numpy as np
from surface import SurfaceModel, local_coordinates
from tracking import TrackingSeed, MarkerTracker, require_regular_template
from display import SurfaceView, marker_overlay

ROOT = Path(__file__).resolve().parent


def load_inputs(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    if len(config["image_size"]) != 2 or any(
        not isinstance(v, int) or v <= 0 for v in config["image_size"]
    ):
        raise ValueError("image_size must contain positive integer width and height")
    base = Path(config_path).resolve().parent
    with np.load(base / config["camera_file"]) as data:
        camera = data["camera_matrix"].astype(float)
        distortion = data["dist_coeffs"].astype(float)
    if (
        camera.shape != (3, 3)
        or not np.isfinite(camera).all()
        or not np.isfinite(distortion).all()
        or min(camera[0, 0], camera[1, 1]) <= 0
    ):
        raise ValueError("Invalid camera calibration")
    data = np.genfromtxt(
        base / config["template_file"], delimiter=",", names=True, encoding="utf-8-sig"
    )
    ids = data["id"].astype(int)
    xy = np.column_stack((data["x_mm"], data["y_mm"]))
    require_regular_template(ids, xy)
    if not np.isfinite(xy).all():
        raise ValueError("Invalid physical template")
    return config, ids, xy, camera, distortion


class LatestCamera:
    """Drain camera continuously; processing always takes the latest available frame."""

    def __init__(self, capture):
        self.capture = capture
        self.condition = threading.Condition()
        self.stopped = False
        self.done = False
        self.latest = None
        self.index = 0
        self.consumed = 0
        self.worker = threading.Thread(target=self._read, daemon=True)
        self.worker.start()

    def _read(self):
        try:
            while not self.stopped:
                ok, frame = self.capture.read()
                if not ok:
                    break
                with self.condition:
                    self.index += 1
                    self.latest = (self.index, time.perf_counter(), frame)
                    self.condition.notify_all()
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def read(self):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.done or (self.latest is not None and self.latest[0] > self.consumed),
                timeout=5.0,
            )
            if not ready or self.latest is None or self.latest[0] <= self.consumed:
                return None
            packet = self.latest
            self.consumed = packet[0]
            return packet

    def close(self):
        self.stopped = True
        self.worker.join(timeout=1.0)
        self.capture.release()
        self.worker.join(timeout=1.0)


def unique_output(base):
    """Keep previous results; never mix a new run with stale successful estimates."""
    base = Path(base)
    if base.exists():
        base = base.with_name(base.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    base.mkdir(parents=True)
    return base


def make_writer(path, fps, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise OSError(f"Cannot encode video: {path}")
    return writer


def save_points(writer, index, timestamp, segment, result, points, xy):
    local = (
        local_coordinates(points, xy)[0] if points is not None else np.full((len(xy), 3), np.nan)
    )
    xyz = points if points is not None else local
    for k in range(len(xy)):
        observed = bool(result["valid"][k])
        source = (
            ("observed_fit" if observed else "model_inferred")
            if points is not None
            else "unavailable"
        )
        row = [
            index,
            timestamp,
            segment,
            k,
            int(observed),
            source,
            float(result["confidence"][k]),
            *result["pixels"][k],
            *xyz[k],
            *local[k],
        ]
        writer.writerow(
            ["" if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for v in row]
        )


def run(args):
    config, ids, xy, camera, distortion = load_inputs(args.config)
    source = (
        args.image
        if args.image is not None
        else (args.video if args.video is not None else str(config["camera_index"]))
    )
    is_image = args.image is not None
    is_camera = not is_image and str(source).isdigit()
    expected = tuple(config["image_size"])
    model = SurfaceModel(xy, camera, distortion, config.get("solver"))
    viewer = SurfaceView(model, *expected)
    basename = Path(source).stem if not is_camera else "camera_" + str(source)
    output = unique_output(args.out or ROOT / "output" / basename)
    marker_dir = ROOT / "marker_detection" / output.name
    marker_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(
        source=str(source),
        frames=0,
        reconstructed_frames=0,
        fully_observed_frames=0,
        ambiguous_frames=0,
        failed_frames=0,
        segments=0,
        complete=False,
        solver="bicubic differential isometry + image observations + pose-aligned temporal prior",
        reference_required=False,
        image_size=list(expected),
        settings=config,
        coordinate_system="camera XYZ mm; local shape uses a proper best-fit rigid alignment to the template",
        id_convention="row-major, per segment; identical marks have an unresolved physical 180-degree symmetry",
        accuracy_note="Reprojection and metric consistency are diagnostics, not measured 3D accuracy.",
        timing_note="Wall throughput includes capture, tracking, solve, render, encoding, CSV and UI. Not sensor-to-display latency.",
        camera_frames_dropped=0,
    )
    capture = None
    latest = None
    video_writer = None
    marker_writer = None
    tracker = None
    segment = 0
    lost = 0
    shape_failed = 0
    fps_smooth = 0.0
    count = 0
    metrics = []
    start = time.perf_counter()
    last_index = 0
    try:
        if is_image:
            still = cv2.imread(str(source))
            if still is None:
                raise ValueError(f"Cannot read image: {source}")
            fps = 1.0
        else:
            capture = cv2.VideoCapture(int(source) if is_camera else str(source))
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {source}")
            if is_camera:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, expected[0])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, expected[1])
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            summary["input_fps"] = fps
            if is_camera:
                latest = LatestCamera(capture)
        with (output / ("reconstructed_points.csv" if is_image else "frame_points.csv")).open(
            "w", newline="", encoding="utf-8"
        ) as pf, (output / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as mf:
            pw = csv.writer(pf)
            mw = csv.writer(mf)
            pw.writerow(
                [
                    "frame",
                    "time_s",
                    "segment",
                    "id",
                    "observed",
                    "point_source",
                    "confidence",
                    "u_px",
                    "v_px",
                    "X_mm",
                    "Y_mm",
                    "Z_mm",
                    "X_local_mm",
                    "Y_local_mm",
                    "Z_local_mm",
                ]
            )
            mw.writerow(
                [
                    "frame",
                    "time_s",
                    "segment",
                    "observed",
                    "status",
                    "reprojection_rms_px",
                    "metric_max_error",
                    "processing_ms",
                    "error",
                ]
            )
            while args.max_frames is None or count < args.max_frames:
                frame_start = time.perf_counter()
                if is_image:
                    if count:
                        break
                    frame = still
                    timestamp = 0.0
                    frame_index = 0
                elif latest is not None:
                    packet = latest.read()
                    if packet is None:
                        break
                    frame_index, acquired, frame = packet
                    timestamp = acquired - start
                    summary["camera_frames_dropped"] += max(0, frame_index - last_index - 1)
                    last_index = frame_index
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame_index = count
                    timestamp = count / fps
                if (frame.shape[1], frame.shape[0]) != expected:
                    raise ValueError(
                        f"Calibration is for {expected}; input is {(frame.shape[1],frame.shape[0])}. Recalibrate or use the matching capture mode."
                    )
                result = dict(
                    pixels=np.full((len(ids), 2), np.nan),
                    valid=np.zeros(len(ids), bool),
                    confidence=np.zeros(len(ids)),
                )
                info = {"status": "waiting_for_grid"}
                error = ""
                points = None
                coeff = None
                try:
                    if tracker is None:
                        seed, _, _ = TrackingSeed.build(frame, ids, xy, args.row_direction)
                        tracker = MarkerTracker(seed)
                        model.reset()
                        segment += 1
                        lost = 0
                        shape_failed = 0
                        result = dict(
                            pixels=seed.pixels.copy(),
                            valid=np.ones(len(ids), bool),
                            confidence=np.ones(len(ids)),
                        )
                    else:
                        result = tracker.update(frame)
                    lost = lost + 1 if result["valid"].sum() < 12 else 0
                    try:
                        points, coeff, info = model.fit(
                            result["pixels"], result["valid"], result["confidence"]
                        )
                        rejected = info.get("rejected_observation_ids", [])
                        if rejected:
                            result["valid"][rejected] = False
                            result["pixels"][rejected] = np.nan
                            result["confidence"][rejected] = 0.0
                            tracker.valid[rejected] = False
                            predicted = cv2.projectPoints(
                                points, np.zeros(3), np.zeros(3), camera, distortion
                            )[0].reshape(-1, 2)
                            tracker.positions[rejected] = predicted[rejected]
                        shape_failed = 0
                    except (ValueError, RuntimeError, cv2.error) as exc:
                        shape_failed += 1
                        info = {
                            "status": (
                                "insufficient_observations"
                                if result["valid"].sum() < 32
                                else "fit_rejected"
                            )
                        }
                        error = str(exc)
                        # Re-estimate shape from current observations, without carrying a stale depth prior.
                        if shape_failed >= 3:
                            tracker = None
                            model.reset()
                    if lost >= 5:
                        tracker = None
                        model.reset()
                except (ValueError, RuntimeError, cv2.error) as exc:
                    error = str(exc)
                    tracker = None
                    model.reset()
                overlay = marker_overlay(
                    frame, result["pixels"], result["valid"], info["status"], segment
                )
                shape = viewer.render(coeff, info, fps_smooth)
                combined = np.hstack((overlay, shape))
                save_points(pw, frame_index, timestamp, segment, result, points, xy)
                if is_image:
                    if not cv2.imwrite(str(marker_dir / "detected_markers.png"), overlay):
                        raise OSError("Cannot save marker image")
                    if coeff is not None:
                        if not cv2.imwrite(str(output / "surface.png"), combined):
                            raise OSError("Cannot save surface image")
                        viewer.save_obj(output / "surface.obj", coeff)
                    summary["image_diagnostics"] = dict(info, error=error)
                elif not args.no_record:
                    if video_writer is None:
                        video_writer = make_writer(
                            output / "reconstruction.avi", fps, (expected[0] * 2, expected[1])
                        )
                        marker_writer = make_writer(
                            marker_dir / "detected_markers.avi", fps, expected
                        )
                        summary["recording_note"] = (
                            "Video encodes processed frames at input FPS; use time_s in CSV for timing, especially for live capture with dropped frames."
                        )
                    video_writer.write(combined)
                    marker_writer.write(overlay)
                key = -1
                if args.preview or (is_camera and not args.no_preview):
                    cv2.imshow("Surface reconstruction | Esc: exit | R: reacquire", combined)
                    key = cv2.waitKey(1) & 255
                elapsed = time.perf_counter() - frame_start
                fps_smooth = (
                    1 / max(elapsed, 1e-9)
                    if count == 0
                    else 0.9 * fps_smooth + 0.1 / max(elapsed, 1e-9)
                )
                mw.writerow(
                    [
                        frame_index,
                        timestamp,
                        segment,
                        int(result["valid"].sum()),
                        info["status"],
                        info.get("reprojection_rms_px", ""),
                        info.get("metric_max_error", ""),
                        elapsed * 1000,
                        error,
                    ]
                )
                if count % 30 == 0:
                    pf.flush()
                    mf.flush()
                count += 1
                summary["frames"] = count
                summary["segments"] = segment
                summary["reconstructed_frames"] += int(coeff is not None)
                summary["fully_observed_frames"] += int(result["valid"].all())
                summary["ambiguous_frames"] += int(info["status"] == "ambiguous")
                summary["failed_frames"] += int(coeff is None)
                if coeff is not None:
                    metrics.append([info["reprojection_rms_px"], info["metric_max_error"]])
                if count % 120 == 0:
                    print(
                        f"{count} frames | 3D {summary['reconstructed_frames']} | {fps_smooth:.1f} FPS | {info['status']}",
                        flush=True,
                    )
                if key == 27:
                    break
                if key in (ord("r"), ord("R")):
                    tracker = None
                    model.reset()
            if count == 0:
                raise ValueError("No decodable frames")
            summary["complete"] = True
    except Exception as exc:
        summary["error"] = str(exc)
        raise
    finally:
        if latest is not None:
            latest.close()
        elif capture is not None:
            capture.release()
        if video_writer is not None:
            video_writer.release()
        if marker_writer is not None:
            marker_writer.release()
        if args.preview or (is_camera and not args.no_preview):
            cv2.destroyAllWindows()
        duration = time.perf_counter() - start
        summary["wall_seconds"] = duration
        summary["wall_throughput_fps"] = count / max(duration, 1e-9)
        if metrics:
            summary["median_reprojection_rms_px"] = float(np.median(np.array(metrics)[:, 0]))
            summary["median_metric_max_error"] = float(np.median(np.array(metrics)[:, 1]))
        (output / "diagnostics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "frames",
                    "reconstructed_frames",
                    "fully_observed_frames",
                    "ambiguous_frames",
                    "segments",
                    "wall_throughput_fps",
                )
            },
            indent=2,
        )
    )
    print(f"Results: {output}\nMarkers: {marker_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="Video file or camera index (default from config)")
    source.add_argument("--image", help="Single image; no reference image required")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-frames", type=int)
    ui = parser.add_mutually_exclusive_group()
    ui.add_argument("--preview", action="store_true", help="Show marker and live 3D views")
    ui.add_argument("--no-preview", action="store_true", help="Disable camera display")
    parser.add_argument(
        "--no-record", action="store_true", help="Save CSV/diagnostics but no videos"
    )
    parser.add_argument(
        "--row-direction", choices=["auto", "down", "up", "left", "right"], default="auto"
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
