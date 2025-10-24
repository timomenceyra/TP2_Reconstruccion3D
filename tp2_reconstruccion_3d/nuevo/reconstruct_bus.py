"""
Utilities to generate a 3D point cloud of the red bus using the CRE Stereo model.

The main entry point is `reconstruct_red_bus`, which:
  * Rectifies the requested stereo pair.
  * Runs the CRE Stereo disparity model (re-using the weights in `models/`).
  * Converts disparity to metric depth.
  * Builds a color-aware mask that keeps the bus region.
  * Saves the filtered point cloud as a `.ply` file that can be viewed in Meshlab.
"""

from __future__ import annotations

import argparse
import functools
import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

# Ensure we can import helpers that live alongside this file.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from disparity.methods import Calibration, Config, InputPair  # type: ignore  # noqa: E402
from utils import compute_depth, rectify_stereo_pair  # type: ignore  # noqa: E402

if TYPE_CHECKING:
    from disparity.method_cre_stereo import CREStereo  # pragma: no cover


def _default_models_path() -> Path:
    return (THIS_DIR / "models").resolve()


@functools.lru_cache(maxsize=1)
def _load_calibration_assets(root: Path) -> Tuple[dict, dict]:
    calib_path = root / "data/pkls/stereo_calibration.pkl"
    maps_path = root / "data/pkls/stereo_maps.pkl"
    with open(calib_path, "rb") as f:
        calib = pickle.load(f)
    with open(maps_path, "rb") as f:
        maps = pickle.load(f)
    return calib, maps


@functools.lru_cache(maxsize=1)
def _load_cre_model(models_path: Path) -> "CREStereo":
    from disparity.method_cre_stereo import CREStereo  # import locally to avoid loading onnxruntime too early

    config = Config(models_path=models_path)
    model = CREStereo(config)
    # Dataset images are 1280x720 after rectification.
    model.parameters["Shape"].set_value("1280x720")
    return model


def _build_calibration(calib: dict, height: int, width: int) -> Calibration:
    baseline_mm = float(np.linalg.norm(calib["T"]))
    fx = float(calib["left_K"][0, 0])
    fy = float(calib["left_K"][1, 1])
    cx = float(calib["left_K"][0, 2])
    cy = float(calib["left_K"][1, 2])
    return Calibration(
        width=width,
        height=height,
        baseline_meters=baseline_mm / 1000.0,
        fx=fx,
        fy=fy,
        cx0=cx,
        cx1=cx,
        cy=cy,
        depth_range=[0.1, 80.0],
        left_image_rect_normalized=[0, 0, 1, 1],
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component to reduce background leakage."""
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    filtered = (labels == biggest)
    return filtered


def _red_bus_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 80, 50], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([165, 70, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    mask = _largest_component(mask > 0)
    return mask


def _build_mask(
    depth_map_mm: np.ndarray,
    color_bgr: np.ndarray,
    depth_limits_mm: Optional[Tuple[float, float]],
    use_red_mask: bool,
    roi: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    mask = np.isfinite(depth_map_mm)
    if depth_limits_mm is not None:
        near, far = depth_limits_mm
        if near is not None:
            mask &= depth_map_mm >= near
        if far is not None:
            mask &= depth_map_mm <= far
    if use_red_mask:
        mask &= _red_bus_mask(color_bgr)
    if roi is not None:
        y0, y1, x0, x1 = roi
        h, w = depth_map_mm.shape
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        roi_mask = np.zeros_like(mask, dtype=bool)
        roi_mask[y0:y1, x0:x1] = True
        mask &= roi_mask
    mask &= depth_map_mm > 0
    return mask


def _depth_to_points(depth_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth_mm.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_mm
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]
    points = np.stack((X, Y, Z), axis=-1)
    return points


def _save_ply(path: Path, points_m: np.ndarray, colors_rgb: np.ndarray, mask: np.ndarray) -> Path:
    pts = points_m[mask]
    cols = colors_rgb[mask]
    if pts.size == 0:
        raise RuntimeError("Mask removed all points; adjust the depth range or ROI.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as ply:
        ply.write("ply\nformat ascii 1.0\n")
        ply.write(f"element vertex {len(pts)}\n")
        ply.write("property float x\nproperty float y\nproperty float z\n")
        ply.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        ply.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, cols.astype(np.uint8)):
            ply.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    return path


def reconstruct_red_bus(
    idx: int,
    output_path: Optional[Path] = None,
    *,
    base_dir: Optional[Path] = None,
    models_path: Optional[Path] = None,
    depth_limits_mm: Optional[Tuple[float, float]] = (1500.0, 8000.0),
    use_red_mask: bool = True,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Path:
    """
    Generate a filtered point cloud for the red bus.

    Args:
        idx: Stereo pair index (matches `left_{idx}.jpg` / `right_{idx}.jpg`).
        output_path: Optional `.ply` destination. Defaults to `../out/clouds_cre/bus_{idx}.ply`.
        base_dir: Repository root (`tp2_reconstruccion_3d/nuevo`). Auto-detected by default.
        models_path: Directory that holds CRE Stereo ONNX weights. Defaults to `base_dir / models`.
        depth_limits_mm: Near/Far clipping in millimetres for the mask (set `None` to keep all).
        use_red_mask: Enable the HSV-based red mask to isolate the bus.
        roi: Optional `(y0, y1, x0, x1)` bounding box to keep (pixels in rectified image).
    """
    base_dir = base_dir or THIS_DIR
    models_path = models_path or _default_models_path()
    output_path = output_path or (base_dir / "../out/clouds_cre" / f"bus_{idx}.ply")
    output_path = output_path.resolve()

    calib, maps = _load_calibration_assets(base_dir)
    cre_model = _load_cre_model(models_path)

    left_path = base_dir / f"data/captures/left_{idx}.jpg"
    right_path = base_dir / f"data/captures/right_{idx}.jpg"
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError(f"Stereo pair {idx} is missing ({left_path} / {right_path}).")

    rect_left, rect_right = rectify_stereo_pair(str(left_path), str(right_path), maps)
    rect_left_gray = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    rect_right_gray = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    height, width = rect_left_gray.shape

    calibration = _build_calibration(calib, height=height, width=width)
    pair = InputPair(left_image=rect_left_gray, right_image=rect_right_gray, calibration=calibration)
    disparity_output = cre_model.compute_disparity(pair)
    disparity_px = disparity_output.disparity_pixels.astype(np.float32)

    baseline_mm = float(np.linalg.norm(calib["T"]))
    fx = float(calib["left_K"][0, 0])
    depth_map = compute_depth(disparity_px, fx, baseline_mm, default=np.nan)

    mask = _build_mask(depth_map, rect_left, depth_limits_mm, use_red_mask, roi)
    colors_rgb = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB)
    points_mm = _depth_to_points(depth_map, calib["left_K"])
    points_m = points_mm / 1000.0

    return _save_ply(output_path, points_m, colors_rgb, mask)


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct the red bus as a colored point cloud.")
    parser.add_argument("--idx", type=int, default=27, help="Stereo pair index to process.")
    parser.add_argument("--output", type=Path, help="Target .ply path (defaults to ../out/clouds_cre/bus_{idx}.ply).")
    parser.add_argument("--depth-min", type=float, default=None, help="Near clip (millimetres).")
    parser.add_argument("--depth-max", type=float, default=None, help="Far clip (millimetres).")
    parser.add_argument("--no-red-mask", action="store_true", help="Disable the red HSV mask.")
    parser.add_argument("--no-depth-filter", action="store_true", help="Keep the full depth range.")
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("y0", "y1", "x0", "x1"),
        help="Optional bounding box to keep (in rectified pixel coordinates).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = _parse_args(argv)
    default_limits = (1500.0, 8000.0)
    if args.no_depth_filter:
        depth_limits = None
    elif args.depth_min is None and args.depth_max is None:
        depth_limits = default_limits
    else:
        near = args.depth_min if args.depth_min is not None else default_limits[0]
        far = args.depth_max if args.depth_max is not None else default_limits[1]
        depth_limits = (near, far)
    ply_path = reconstruct_red_bus(
        idx=args.idx,
        output_path=args.output,
        depth_limits_mm=depth_limits,
        use_red_mask=not args.no_red_mask,
        roi=tuple(args.roi) if args.roi else None,
    )
    print(f"Saved point cloud to {ply_path}")


if __name__ == "__main__":
    main()
