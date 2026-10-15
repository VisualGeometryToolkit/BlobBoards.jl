"""detect_markers.py -- the AprilTag and ArUco baselines of the BMVC 2026 evaluation.

A faithful port of BlobBoards-Evaluation 88bca9f (evaluation_pipe/pose_extractor.py,
evaluation_pipe/pose_estimators.py): same detectors, same settings, same
undistortion, same PnP, same frame alignment, same per-image export and overlay.
What changed is only how it is driven: every setting comes from the config of
record (configs/baselines.toml), the camera is the camera of record in
OpenCV's convention (written by scripts/detect_baselines.jl, which runs this
driver), and every output goes under the keyed run directory the caller names.

    pixi run --manifest-path bmvc2026/pixi.toml -e baselines python \
        bmvc2026/scripts/baselines/detect_markers.py --config <baselines.toml> \
        --camera <camera_opencv.json> --session <data_root()/absolute_pose_experiments/<set>> \
        --exports <dir> --val <dir>

Writes, per system (apriltags, aruco):
    <exports>/poses/<system>/<stem>.json                 the per-image export
    <exports>/poses/calibration/<system>/<stem>.json     on the calibration subset
    <exports>/occlusion/<LL>/poses/<system>/<stem>.json  on each occlusion level
    <val>/<system>/, <val>/calibration/<system>/,
    <val>/../occlusion/<LL>/val/<system>/               <stem>_val.png overlays
    <exports>/_environment.json                          interpreter and library versions
"""

import argparse
import glob
import json
import os
import sys
import threading
import time
import tomllib

import cv2 as cv
import numpy as np


# --- the camera --------------------------------------------------------------

def read_camera(path, cfg):
    """K and the distortion vector from the camera in OpenCV's convention (fx fy
    cx cy in OpenCV's pixel frame, the coefficients by name; scripts/
    detect_baselines.jl writes it from the camera of record), as the upstream
    reader returned them."""
    with open(path) as f:
        d = json.load(f)
    origin = d.get("image_origin")
    if origin is not None and origin != "opencv":
        raise ValueError(f"{path}: image_origin {origin!r}; the baselines take OpenCV's pixel frame")
    dtype = np.dtype(cfg["camera"]["dtype"])
    K = np.array([[d["fx"], 0.0, d["cx"]], [0.0, d["fy"], d["cy"]], [0.0, 0.0, 1.0]], dtype=dtype)
    names = cfg["camera"]["distortion"]
    unknown = [k for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tilt_x", "tilt_y")
               if k in d and k not in names and d[k] != 0]
    if unknown:
        raise ValueError(f"{path}: nonzero coefficients {unknown} outside the configured vector {names}")
    dist = np.array([float(d.get(k, 0.0)) for k in names], dtype=dtype)
    return K, dist


# --- the detectors (pose_estimators.py) ---------------------------------------

_APRILTAG_LOCK = threading.Lock()
_APRILTAG_DETECTOR = None


def detect_apriltags(image, intrinsics, cfg):
    from dt_apriltags import Detector
    global _APRILTAG_DETECTOR
    a = cfg["apriltags"]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    camera_params = (intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
    with _APRILTAG_LOCK:
        if _APRILTAG_DETECTOR is None:
            _APRILTAG_DETECTOR = Detector(
                searchpath=["apriltags"], families=a["families"], nthreads=a["nthreads"],
                quad_decimate=a["quad_decimate"], quad_sigma=a["quad_sigma"],
                refine_edges=a["refine_edges"], decode_sharpening=a["decode_sharpening"], debug=a["debug"],
            )
        start = time.perf_counter()
        try:
            results = _APRILTAG_DETECTOR.detect(gray, estimate_tag_pose=False, camera_params=camera_params)
        except Exception as e:  # upstream: a failed detect is an empty frame
            print(f"!! Error occurred while detecting tags: {e}", file=sys.stderr)
            results = []
        elapsed = (time.perf_counter() - start) * 1000.0
    ids, corners = [], []
    for r in results:
        c = np.array([r.corners[0], r.corners[1], r.corners[2], r.corners[3]], dtype=np.float32).reshape(4, 2)
        ids.append(r.tag_id)
        corners.append(c)
    return [str(i).zfill(3) for i in ids], corners, elapsed


def detect_aruco(image, intrinsics, cfg):
    a = cfg["aruco"]
    aruco_dict = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, a["dictionary"]))
    parameters = cv.aruco.DetectorParameters()
    parameters.useAruco3Detection = bool(a["use_aruco3_detection"])
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    start = time.perf_counter()
    corners_list, ids, _ = detector.detectMarkers(image)
    ids = ids.reshape(-1) if ids is not None else np.array([])
    corners_list = [c.reshape(4, 2) for c in corners_list]
    elapsed = (time.perf_counter() - start) * 1000.0
    return [str(i).zfill(3) for i in ids], corners_list, elapsed


DETECTORS = {"apriltags": detect_apriltags, "aruco": detect_aruco}


def pose_from_corners(ids_corners_sizes, K, cfg):
    flags = getattr(cv, cfg["pnp"]["flags"])
    ids, Rs, ts = [], [], []
    start = time.perf_counter()
    for (bid, corners, size_mm) in ids_corners_sizes:
        h = size_mm / 2.0
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
        ok, rvec, tvec = cv.solvePnP(obj, corners, K, None, flags=flags)
        if ok:
            R, _ = cv.Rodrigues(rvec)
            ids.append(str(bid).zfill(3))
            Rs.append(R.astype(np.float32))
            ts.append(tvec.squeeze().astype(np.float32))
    return ids, Rs, ts, (time.perf_counter() - start) * 1000.0


# The AprilTag frame re-expressed as x right, y up, z into the scene (pose_extractor.py).
_RZ90_INT = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
_RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
_RY180 = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
_RZ180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
_R_ALIGN = _RZ180 @ _RY180 @ _RZ90


# --- boards ------------------------------------------------------------------

def marker_sizes(board_dir, digits):
    """Tag side (mm) per board id: the detectable pattern at the print density."""
    sizes = {}
    for path in glob.glob(f"{board_dir}/*.json"):
        with open(path) as f:
            b = json.load(f)
        bid = b["hashes"]["config"]
        dpi = b["preamble"]["board_config"]["print_density"]["value"]
        px = b["preamble"]["pattern_config"]["detectable_pattern_resolution"]["value"]
        sizes[bid.zfill(3)] = round(px / dpi * 25.4, digits)
    return sizes


# --- outputs (pose_extractor.py) ---------------------------------------------

def write_export(poses, K, image_name, system, elapsed, out_dir):
    data = {"image": image_name, "board_type": system, "undist_K": K.tolist(),
            "elapsed_time_ms": elapsed, "poses": poses}
    with open(os.path.join(out_dir, os.path.splitext(image_name)[0] + ".json"), "w+") as f:
        json.dump(data, f, indent=4)


def write_overlay(poses, image, stem, out_dir, K, axis_fraction):
    for pose in poses:
        t = np.array(pose["trans"], dtype=np.float32)
        R = np.array(pose["rot_mat"], dtype=np.float32)
        s = pose["size_mm"] * axis_fraction
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        o, x, y, z = ((T @ np.array(p))[:3] for p in ([0., 0., 0., 1.], [s, 0., 0., 1.], [0., s, 0., 1.], [0., 0., s, 1.]))

        def proj(p):
            q = K @ p
            return (int(q[0] / q[2]), int(q[1] / q[2]))

        o2 = proj(o)
        cv.line(image, o2, proj(x), color=(0, 0, 255), thickness=3)
        cv.line(image, o2, proj(y), color=(0, 255, 0), thickness=3)
        cv.line(image, o2, proj(z), color=(255, 0, 0), thickness=3)
        cv.putText(image, f"{pose['board_id']}", o2, cv.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255))
        cv.circle(image, o2, 5, (255, 255, 255), thickness=-1)
    cv.imwrite(os.path.join(out_dir, f"{stem}_val.png"), image)


def run_subset(system, image_dir, board_dir, out_dir, val_dir, K, dist, cfg, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    exts = set(cfg["subsets"]["image_extensions"])
    images = sorted(
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and os.path.splitext(f)[1].lower() in exts
    )
    if limit is not None:
        images = images[:limit]
    sizes = marker_sizes(board_dir, cfg["pnp"]["size_digits"])
    detect = DETECTORS[system]
    for path in images:
        image = cv.imread(path)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        K_new = K.copy()
        image = cv.undistort(image, K, dist, None, K_new)
        ids, corners, t_detect = detect(image, K_new, cfg)
        known = [(ids[i], corners[i], sizes[ids[i]]) for i in range(len(ids)) if ids[i] in sizes]
        est_ids, Rs, ts, t_pnp = pose_from_corners(known, K_new, cfg)
        poses = []
        for i, bid in enumerate(est_ids):
            R = Rs[i]
            if system == "apriltags" and cfg["apriltags"]["align"]:
                R = R @ _RZ90_INT
                R = R @ _R_ALIGN
            poses.append({"board_id": bid, "rot_mat": R.tolist(), "trans": ts[i].flatten().tolist(), "size_mm": sizes[bid]})
        name = os.path.basename(path)
        write_export(poses, K_new, name, system, t_detect + t_pnp, out_dir)
        write_overlay(poses, image, os.path.splitext(name)[0], val_dir, K_new, cfg["val"]["axis_fraction"])
        print(f"[{system}] {name}: {len(poses)} poses", file=sys.stderr)
    return len(images)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--session", required=True, help="data_root()/absolute_pose_experiments/<set>")
    ap.add_argument("--exports", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--systems", default="apriltags,aruco")
    ap.add_argument("--parts", default="main,calibration,occlusion")
    ap.add_argument("--limit", type=int, default=None, help="first N images per subset (checks only)")
    args = ap.parse_args(argv)
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    K, dist = read_camera(args.camera, cfg)
    parts = set(args.parts.split(","))
    os.makedirs(args.exports, exist_ok=True)
    from importlib.metadata import version
    with open(os.path.join(args.exports, "_environment.json"), "w") as f:
        json.dump({"python": sys.version, "numpy": np.__version__, "opencv-python": version("opencv-python"),
                   "dt-apriltags": version("dt-apriltags"), "cv2": cv.__version__}, f, indent=4)
    for system in args.systems.split(","):
        sub = cfg["subsets"][system]
        boards = os.path.join(args.session, "boards", sub["boards"])
        if "main" in parts:
            run_subset(system, os.path.join(args.session, "images", sub["images"]), boards,
                       os.path.join(args.exports, "poses", system), os.path.join(args.val, system), K, dist, cfg, args.limit)
        if "calibration" in parts:
            run_subset(system, os.path.join(args.session, "images", sub["calibration"]), boards,
                       os.path.join(args.exports, "poses", "calibration", system),
                       os.path.join(args.val, "calibration", system), K, dist, cfg, args.limit)
        if "occlusion" in parts:
            for level in cfg["occlusion"]["levels"]:
                occ = os.path.join(args.session, "occlusion", level)
                run_subset(system, os.path.join(occ, "images", sub["images"]), boards,
                           os.path.join(args.exports, "occlusion", level, "poses", system),
                           os.path.join(os.path.dirname(os.path.normpath(args.val)), "occlusion", level, "val", system),
                           K, dist, cfg, args.limit)


if __name__ == "__main__":
    main()
