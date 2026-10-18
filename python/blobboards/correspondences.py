"""The 2D-3D correspondences of a BlobBoards run, in the shape OpenCV and MATLAB take.

A run directory carries ``observations.json`` (vgio/target_points2d/v1: per frame,
the image, the sensor and each measured point's pixel and reference into a target's
point list, in the pixel frame the file declares — the canonical COLMAP frame,
top-left pixel centre at (0.5, 0.5)) and ``targets_<k>.json`` (vgio/rig/v4: each
member target's points in its own frame — a ``points`` target's ``points``, a
``planar`` target's ``nominal`` points — in the file's declared ``length_unit``; a
planar target named only by its design reference carries no points here). This
module resolves the references into per-frame point arrays — object points
converted to millimetres from that declared unit, as VisualGeometryToolkitIo's
``correspondences`` does — and moves the pixels into the requested convention,
once, so no script redoes the half-pixel arithmetic:

    opencv  top-left pixel centre at (0, 0)   cv2.calibrateCamera
    matlab  top-left pixel centre at (1, 1)   estimateCameraParameters
    colmap  as stored
"""
import glob
import json
import os

import numpy as np

CANONICAL = "colmap"
SHIFT = {"colmap": 0.0, "opencv": -0.5, "matlab": 0.5, "julia": 0.5}
# Millimetres per unit, for the LENGTH units VisualGeometryToolkitIo's unit table spells
# (``_UNITS`` in contract.jl). ``px`` is in that table too but is not a length, so
# a target whose points declare it is refused, as a missing or unknown unit is.
MM_PER_UNIT = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "um": 1.0e-3, "inch": 25.4}


def _points_mm(path, unit, tgt):
    """A target's points as an N×2 or N×3 array in millimetres, from the file's declared ``length_unit``."""
    if unit not in MM_PER_UNIT:
        raise ValueError(f"{path}: length_unit {unit!r} is not a length unit; "
                         f"expected one of {sorted(MM_PER_UNIT)}")
    return np.asarray(tgt, dtype=np.float64) * MM_PER_UNIT[unit]


def _target_points(body):
    """A rig/v4 target body's own points (a list), or None for a design named only by reference."""
    if body.get("type") == "points":
        return body["points"]
    if body.get("type") == "planar":
        return body.get("nominal")
    raise ValueError(f"unknown target type {body.get('type')!r}")


def load_correspondences(run_dir, convention="opencv"):
    """Per frame: ``t``, ``image``, ``keys`` (target, index), ``object_points`` N×3 in mm, ``image_points`` N×2."""
    if convention not in SHIFT:
        raise ValueError(f"unknown pixel convention {convention!r}; one of {sorted(SHIFT)}")
    obs = json.load(open(os.path.join(run_dir, "observations.json")))
    if obs.get("schema") != "vgio/target_points2d/v1":
        stamp = obs.get("schema")
        hint = (" — a retired v1 file: migrate it with BlobBoards.migrate_pixel_convention(path)"
                if stamp == "vgio/observations/v1" else
                " — a retired kind: re-export it from its producer" if stamp == "vgio/observations/v2" else "")
        raise ValueError(f"{run_dir}: observations.json is {stamp!r}, not vgio/target_points2d/v1{hint}")
    frame = obs.get("pixel_frame")
    if frame not in SHIFT:
        raise ValueError(f"{run_dir}: observations.json declares pixel_frame {frame!r}; one of {sorted(SHIFT)}")
    base = int(obs.get("point_index_base", 1))
    paths = sorted(glob.glob(os.path.join(run_dir, "targets_*.json")))
    if len(paths) != len(obs.get("targets", [])):
        raise ValueError(f"{run_dir}: observations.json resolves against {len(obs.get('targets', []))} "
                         f"targets file(s), the run holds {len(paths)}")
    points = {}
    for path in paths:
        t = json.load(open(path))
        if t.get("schema") != "vgio/rig/v4":
            raise ValueError(f"{path}: not a vgio/rig/v4 file")
        for tgt in t.get("targets", []):
            own = _target_points(tgt["target"])
            if own is None:
                continue
            pts = _points_mm(path, t.get("length_unit"), own)
            if pts.ndim != 2:
                raise ValueError(f"{path}: target {tgt['id']} points are not a point list")
            if pts.shape[1] == 2:
                pts = np.hstack([pts, np.zeros((len(pts), 1))])
            if tgt["id"] in points and not np.allclose(points[tgt["id"]], pts):
                raise ValueError(f"target {tgt['id']} carries different points in two targets files")
            points[tgt["id"]] = pts
    if not points:
        raise ValueError(f"{run_dir}: no targets_*.json with points")
    shift = SHIFT[convention] - SHIFT[frame]
    frames = []
    for f in obs["frames"]:
        keys = [(o["target"], int(o["index"])) for o in f["points"]]
        for t, i in keys:
            if t not in points or not 0 <= i - base < len(points[t]):
                raise ValueError(f"{run_dir}: frame {f['t']}: (target {t}, index {i}) does not resolve in the targets")
        X = np.array([points[t][i - base] for t, i in keys], dtype=np.float64).reshape(-1, 3)
        U = np.array([o["xy"] for o in f["points"]], dtype=np.float64).reshape(-1, 2) + shift
        frames.append({"t": int(f["t"]), "image": f.get("image", ""), "keys": keys,
                       "object_points": X, "image_points": U})
    return frames


def image_size(run_dir):
    d = json.load(open(os.path.join(run_dir, "camera_mocap.json")))
    if d.get("schema") != "vgio/cameras/v3":
        raise ValueError(f"{run_dir}: camera_mocap.json is {d.get('schema')!r}, not vgio/cameras/v3")
    c = d["cameras"][0]
    return int(c["width"]), int(c["height"])


OPENCV_MODELS = ("default", "rational", "thin_prism")


def calibrate_opencv(run_dir, model="default", fix_aspect=False, min_points=6):
    """``cv2.calibrateCamera`` on the run's correspondences. ``model``: ``default`` (k1 k2 p1 p2 k3),
    ``rational`` (+ k4 k5 k6) or ``thin_prism`` (+ s1..s4). Two focal lengths, as the AprilGrid
    baselines were fitted, unless ``fix_aspect`` is True. OpenCV is handed its own pixels: the
    observations' declared frame converted to OpenCV's (top-left pixel centre at (0, 0)) by
    ``load_correspondences`` — COLMAP's xy − 0.5 — never the stored numbers as they are.

    Refuses a frame whose points span more than one target: each target's points are in that
    target's own frame, and cv2 takes one planar object per view, so such a frame would hand it
    one object made of two coordinate frames. Returns the camera (OpenCV frame), the per-frame
    poses (rvec, tvec in millimetres) and each posed frame's image and target."""
    frames = [f for f in load_correspondences(run_dir, "opencv") if len(f["object_points"]) >= min_points]
    if not frames:
        raise ValueError(f"{run_dir}: no frame with {min_points} or more correspondences")
    for f in frames:
        ts = sorted({t for t, _ in f["keys"]})
        if len(ts) > 1:
            raise ValueError(f"{run_dir}: frame {f['t']} ({f['image']}) sees {len(ts)} targets {ts}; "
                             "cv2.calibrateCamera takes one planar target per view — calibrate on "
                             "single-target frames")
    if model not in OPENCV_MODELS:
        raise ValueError(f"unknown model {model!r}; one of {sorted(OPENCV_MODELS)}")
    import cv2

    obj = [f["object_points"].astype(np.float32) for f in frames]
    img = [f["image_points"].astype(np.float32) for f in frames]
    w, h = image_size(run_dir)
    flags = 0
    if model == "rational":
        flags |= cv2.CALIB_RATIONAL_MODEL
    elif model == "thin_prism":
        flags |= cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
    K0 = None
    if fix_aspect:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
        K0 = np.array([[w, 0.0, w / 2], [0.0, w, h / 2], [0.0, 0.0, 1.0]])
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, (w, h), K0, None, flags=flags)
    return {"rms": float(rms), "K": K, "dist": dist.ravel(), "image_size": (w, h),
            "points": int(sum(len(o) for o in obj)), "source": f"cv2.calibrateCamera {cv2.__version__}",
            "poses": [{"image": f["image"], "target": f["keys"][0][0], "rvec": rv.ravel(), "tvec": tv.ravel()}
                      for f, rv, tv in zip(frames, rvecs, tvecs)]}


# --- native output (cameras v3, transforms v2), field for field what VisualGeometryToolkitIo.migrate_opencv writes -----------------



def _rotation_from_rvec(rvec):
    """Rodrigues: an axis-angle vector (OpenCV's rvec) → its 3×3 rotation."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(r))
    if th < 1e-15:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * Kx + (1.0 - np.cos(th)) * (Kx @ Kx)


def _quat_wxyz(R):
    """A rotation's unit Hamilton quaternion (w, x, y, z), w ≥ 0 (VisualGeometryToolkitIo's sign)."""
    m = np.asarray(R, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = 2.0 * np.sqrt(1.0 + t)
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def opencv_lens_record(dist):
    """The vgio/cameras/v3 lens record of OpenCV's distortion vector ``dist`` = (k1, k2, p1, p2[, k3[,
    k4, k5, k6[, s1, s2, s3, s4[, tx, ty]]]]) — 4, 5, 8, 12 or 14 entries, the terms that length
    evaluates, zeros included: VisualGeometryToolkitIo's ``opencv_lens`` written out (a perspective lens whose
    radial factor, tangential and thin-prism terms add at one point, the tilt after)."""
    v = [float(x) for x in np.asarray(dist, dtype=np.float64).ravel()]
    n = len(v)
    if n not in (4, 5, 8, 12, 14):
        raise ValueError(f"an OpenCV distortion vector has 4, 5, 8, 12 or 14 entries; got {n}")
    k = v[0:2] if n == 4 else [v[0], v[1], v[4]]
    d = v[5:8] if n >= 8 else []
    terms = [{"radial": [k, d]}, {"tangential": [v[2], v[3]]}]
    if n >= 12:
        terms.append({"thin_prism": [v[8:10], v[10:12]]})
    if n == 14:
        return {"perspective": [], "chain": [{"add": terms}, {"tilt": v[12:14]}]}
    return {"perspective": [], "add": terms}


def write_native_calibration(out_dir, stem, K, dist, poses, image_size, provenance):
    """Write an OpenCV calibration as the two native files VisualGeometryToolkitIo's ``migrate_opencv``
    writes: ``<stem>_cameras.json`` (vgio/cameras/v3, one camera named ``stem``, the lens record of
    OpenCV's model for ``dist``, ``opencv_lens_record``) and ``<stem>_poses.json`` (vgio/transforms/v3, one
    record per posed image, ``target:<target> → view:<image>``, millimetres).

    ``K`` and ``dist`` are OpenCV's, in OpenCV's pixel frame (top-left pixel centre at (0, 0)); the
    camera is written in the canonical COLMAP frame the file declares, so ``cx, cy`` move by +0.5.
    ``poses``: dicts with ``image``, ``target``, ``rvec`` and ``tvec`` (millimetres). Returns the two
    paths; refuses to overwrite either."""
    w, h = int(image_size[0]), int(image_size[1])
    camera = {"name": stem, "width": w, "height": h,
              "K": {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]) + 0.5,
                    "cy": float(K[1][2]) + 0.5, "skew": float(K[0][1])},
              "lens": opencv_lens_record(dist)}
    cams = {"schema": "vgio/cameras/v3", "pixel_frame": "colmap", "cameras": [camera], "provenance": provenance}
    records = []
    for p in poses:
        q = _quat_wxyz(_rotation_from_rvec(p["rvec"]))
        records.append({"from": f"target:{p['target']}", "to": f"view:{p['image']}",
                        "rotation": [float(x) for x in q],
                        "translation": [float(x) for x in np.asarray(p["tvec"], dtype=np.float64).ravel()]})
    if len({r["to"] for r in records}) != len(records):
        raise ValueError("write_native_calibration: an image is posed twice")
    trans = {"schema": "vgio/transforms/v3", "length_unit": "mm",
             "rotation_encoding": "quaternion_wxyz_hamilton", "transform_convention": "x_to = R * x_from + t",
             "transforms": records,
             "provenance": {k: v for k, v in provenance.items() if k not in ("reprojection_error", "error_statistic")}}
    os.makedirs(out_dir, exist_ok=True)
    paths = (os.path.join(out_dir, f"{stem}_cameras.json"), os.path.join(out_dir, f"{stem}_poses.json"))
    for path in paths:
        if os.path.exists(path):
            raise FileExistsError(f"{path} exists; refusing to overwrite it")
    for path, body in zip(paths, (cams, trans)):
        with open(path, "w") as f:
            json.dump(body, f, indent=1)
    return paths


def write_opencv_calibration(run_dir, model, out_dir, **kw):
    """Calibrate the run with OpenCV (``calibrate_opencv``) and write it natively as
    ``calibration_opencv_<model>_cameras.json`` + ``_poses.json`` in ``out_dir`` — the files
    ``VisualGeometryToolkitIo.migrate_opencv`` makes of a third-party ``calibration_opencv_<model>.json``,
    so a BlobBoards fit and a migrated one are read the same way."""
    r = calibrate_opencv(run_dir, model, **kw)
    prov = {"source": "opencv", "fitted_by": r["source"], "fitted_on": os.path.join(os.path.abspath(run_dir), "observations.json"),
            "model": model, "reprojection_error": r["rms"], "error_statistic": "rms"}
    cams, poses = write_native_calibration(out_dir, f"calibration_opencv_{model}", r["K"], r["dist"], r["poses"],
                                           r["image_size"], prov)
    return {"cameras": cams, "poses": poses, "rms": r["rms"], "K": r["K"], "n_posed": len(r["poses"])}


def write_matlab_correspondences(run_dir, out_path):
    """The run's correspondences for MATLAB's ``estimateCameraParameters`` (or any calibrator), as
    JSON (``jsondecode`` reads it): per frame the image name, ``object_points`` N×3 in millimetres
    and ``image_points`` N×2 in MATLAB's frame (top-left pixel centre at (1, 1)). The
    calibration that comes back is expected beside it as ``calibration_matlab_<model>.json`` in
    the shape of the capture's MATLAB files: ``fx fy cx cy r1 r2 r3 t1 t2 mean_error poses``."""
    frames = load_correspondences(run_dir, "matlab")
    body = {"pixel_frame": "matlab", "unit": "mm",
            "frames": [{"t": f["t"], "image": f["image"], "object_points": f["object_points"].tolist(),
                        "image_points": f["image_points"].tolist()} for f in frames]}
    with open(out_path, "w") as f:
        json.dump(body, f)
    return len(frames)


def save_matlab(run_dir, out_path):
    """Write ``worldPoints`` M×2 and ``imagePoints`` M×2×numImages (1-based pixel centres) for
    ``estimateCameraParameters``, which needs the SAME points in every image: the points seen in
    all frames are kept, the rest dropped. Returns (M, numImages)."""
    from scipy.io import savemat

    frames = load_correspondences(run_dir, "matlab")
    common = set(frames[0]["keys"])
    for f in frames[1:]:
        common &= set(f["keys"])
    keys = sorted(common)
    if not keys:
        raise ValueError(f"{run_dir}: no point is seen in every frame")
    index = [{k: i for i, k in enumerate(f["keys"])} for f in frames]
    world = np.array([frames[0]["object_points"][index[0][k]][:2] for k in keys])
    image = np.stack([np.array([f["image_points"][ix[k]] for k in keys]) for f, ix in zip(frames, index)], axis=2)
    savemat(out_path, {"worldPoints": world, "imagePoints": image, "images": [f["image"] for f in frames]})
    return len(keys), len(frames)


def _main(argv):
    import argparse

    ap = argparse.ArgumentParser(description="BlobBoards correspondences → OpenCV / MATLAB")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("opencv", help="cv2.calibrateCamera on a run, written natively as "
                       "calibration_opencv_<model>_cameras.json + _poses.json (vgio cameras v3, transforms v2)")
    a.add_argument("run_dir"); a.add_argument("model", choices=sorted(OPENCV_MODELS)); a.add_argument("out_dir")
    a.add_argument("--fix-aspect", action="store_true")
    m = sub.add_parser("matlab", help="the run's correspondences for MATLAB, as JSON")
    m.add_argument("run_dir"); m.add_argument("out")
    args = ap.parse_args(argv)
    if args.cmd == "opencv":
        d = write_opencv_calibration(args.run_dir, args.model, args.out_dir, fix_aspect=args.fix_aspect)
        print(f"{d['cameras']} + {os.path.basename(d['poses'])}: rms={d['rms']:.3f} px, {d['n_posed']} posed frames")
    else:
        n = write_matlab_correspondences(args.run_dir, args.out)
        print(f"{args.out}: {n} frames")


if __name__ == "__main__":
    import sys

    _main(sys.argv[1:])
