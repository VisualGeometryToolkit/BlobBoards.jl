"""The 2D-3D correspondences of a BlobBoards run, in the shape OpenCV and MATLAB take.

A run directory carries ``observations.json`` (``vgio/observations``: per frame, the image,
the sensor and each observed point's pixel and reference into a target's point list) and
``targets_<k>.json`` (``vgio/rig``: the target definitions those references resolve in).
Resolving the references into point arrays and moving the pixels into a calibrator's
convention is VisualGeometryToolkitIO's one export, ``save_correspondences``
(``export/correspondences/v2``: one row per frame and target, object points in that target's
own frame, in millimetres — on its surface, so a calibrated target's measured points, not its
design), and an OpenCV fit becomes the native camera store and poses through its one
converter, ``convert_opencv_calibration``; this module calls both through Julia and reads
their files, so no script redoes the half-pixel arithmetic:

    opencv  top-left pixel centre at (0, 0)   cv2.calibrateCamera
    matlab  top-left pixel centre at (1, 1)   estimateCameraParameters
    colmap  as stored (0.5, 0.5)
"""
import json
import os
import tempfile

import numpy as np

CAMERAS_MAJOR = "vgio/cameras/v6."


def _jl():
    import blobboards

    blobboards._init()
    return blobboards._jl


def load_correspondences(run_dir, convention="opencv"):
    """Per (frame, target): ``t`` (the frame index), ``image``, ``sensor``, ``target``,
    ``object_points`` N×3 in millimetres (the target's own frame) and ``image_points`` N×2 in
    ``convention`` (a pixel-frame name VisualGeometryToolkitIO knows — ``opencv``, ``colmap``, ``julia`` — or
    ``matlab``; any other is refused there) — IO's ``save_correspondences`` export of the run, read back."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "correspondences.json")
        _jl().BlobBoards._correspondences_python(os.path.abspath(run_dir), convention, out)
        d = json.load(open(out))
    if d["units"]["length"] != "mm":
        raise ValueError(f"{run_dir}: the export's object points are in {d['units']['length']!r}, expected mm")
    return [{"t": int(r["frame_index"]), "image": r["image"], "sensor": r["sensor"], "target": r["target"],
             "object_points": np.asarray(r["object_points"], dtype=np.float64).reshape(-1, 3),
             "image_points": np.asarray(r["image_points"], dtype=np.float64).reshape(-1, 2)} for r in d["rows"]]


def image_size(run_dir):
    """The (width, height) of the run's camera (``camera_mocap.json``, a ``vgio/cameras`` store of one camera)."""
    d = json.load(open(os.path.join(run_dir, "camera_mocap.json")))
    if not str(d.get("schema", "")).startswith(CAMERAS_MAJOR):
        raise ValueError(f"{run_dir}: camera_mocap.json is {d.get('schema')!r}, not {CAMERAS_MAJOR}x (an earlier store: "
                         "migrate it with VisualGeometryToolkitIO.vgio_migrate)")
    c = d["cameras"][0]["image_size"]
    return int(c["width"]), int(c["height"])


OPENCV_MODELS = ("default", "rational", "thin_prism")
# OpenCV's distortion vector, by name, in its order (the names an OpenCV export states).
OPENCV_COEFFICIENTS = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tilt_x", "tilt_y")


def calibrate_opencv(run_dir, model="default", fix_aspect=False, min_points=6):
    """``cv2.calibrateCamera`` on the run's correspondences. ``model``: ``default`` (k1 k2 p1 p2 k3),
    ``rational`` (+ k4 k5 k6) or ``thin_prism`` (+ s1..s4). Two focal lengths, as the AprilGrid
    baselines were fitted, unless ``fix_aspect`` is True. OpenCV is handed its own pixels: the
    observations' declared frame converted to OpenCV's (top-left pixel centre at (0, 0)) by
    ``load_correspondences`` — COLMAP's xy − 0.5 — never the stored numbers as they are.

    Each view is one (frame, target) row of the export: a target's points are in that target's own
    frame, and cv2 takes one planar object per view. The views must all be of one target (its poses
    are written as that target's). Returns the camera (OpenCV frame), the per-view poses (rvec,
    tvec in millimetres) and each posed view's image and target."""
    frames = [f for f in load_correspondences(run_dir, "opencv") if len(f["object_points"]) >= min_points]
    if not frames:
        raise ValueError(f"{run_dir}: no frame with {min_points} or more correspondences")
    ts = sorted({f["target"] for f in frames})
    if len(ts) > 1:
        raise ValueError(f"{run_dir}: the views see {len(ts)} targets {ts}; one calibration's poses are of one "
                         "target — calibrate on one target's views")
    _require_planar(run_dir, frames, "cv2.calibrateCamera (without an intrinsic guess)")
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
            "poses": [{"image": f["image"], "target": f["target"], "rvec": rv.ravel(), "tvec": tv.ravel()}
                      for f, rv, tv in zip(frames, rvecs, tvecs)]}


def _require_planar(run_dir, frames, who):
    """A calibrated target's measured points may leave its plane (a height field); a planar calibrator takes z = 0 only."""
    for f in frames:
        z = np.abs(f["object_points"][:, 2]).max(initial=0.0)
        if z > 0.0:
            raise ValueError(f"{run_dir}: frame {f['t']} ({f['image']}), target {f['target']}: its points leave the target's "
                             f"plane (|z| up to {z} mm, a calibrated surface); {who} takes planar points only")


def write_native_calibration(out_dir, stem, K, dist, poses, image_size, provenance):
    """Write an OpenCV calibration natively: the OpenCV export ``<stem>.json`` (``fx fy cx cy``, the
    distortion coefficients by name, ``rms``, one ``{rvec, tvec}`` per posed image, ``tvec`` in
    millimetres), converted by VisualGeometryToolkitIO's ``convert_opencv_calibration`` into
    ``out_dir``: the ``<stem>_cameras.json`` store (one camera named ``stem``, in the canonical COLMAP
    frame) and ``<stem>_poses.json`` (``target:<target> → view:<image>``). The export itself is a
    staging file (nothing is left in ``out_dir`` when the conversion refuses).

    ``K`` and ``dist`` are OpenCV's, in OpenCV's pixel frame (top-left pixel centre at (0, 0)), ``dist``
    of 4, 5, 8, 12 or 14 entries. ``poses``: dicts with ``image``, ``target`` (one target for all) and
    ``rvec``, ``tvec`` (millimetres). ``provenance["reprojection_error"]`` (required) is the fit's rms in
    pixels, which the camera store records. Returns the two native paths; refuses to overwrite."""
    if "reprojection_error" not in provenance:
        raise ValueError("write_native_calibration: provenance states the fit's `reprojection_error` (its rms, px)")
    w, h = int(image_size[0]), int(image_size[1])
    v = [float(x) for x in np.asarray(dist, dtype=np.float64).ravel()]
    if len(v) not in (4, 5, 8, 12, 14):
        raise ValueError(f"an OpenCV distortion vector has 4, 5, 8, 12 or 14 entries; got {len(v)}")
    if float(K[0][1]) != 0.0:
        raise ValueError("an OpenCV export has no skew; K carries one")
    targets = sorted({p["target"] for p in poses})
    if len(targets) != 1:
        raise ValueError(f"write_native_calibration: the poses are of one target; got {targets}")
    images = [p["image"] for p in poses]
    if len(set(images)) != len(images):
        raise ValueError("write_native_calibration: an image is posed twice")
    export = {"fx": float(K[0][0]), "fy": float(K[1][1]), "cx": float(K[0][2]), "cy": float(K[1][2]),
              **dict(zip(OPENCV_COEFFICIENTS, v)),
              "rms": float(provenance["reprojection_error"]), "tvec_unit": "mm", "images": images,
              "poses": [{"rvec": [float(x) for x in np.asarray(p["rvec"], dtype=np.float64).ravel()],
                         "tvec": [float(x) for x in np.asarray(p["tvec"], dtype=np.float64).ravel()]} for p in poses]}
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{stem}.json")   # the converter names its files after the export's stem
        with open(path, "w") as f:
            json.dump(export, f, indent=1)
        cams, poses_path = _jl().BlobBoards._convert_opencv_python(path, images, w, h, targets[0], os.path.abspath(out_dir))
    return str(cams), str(poses_path)


def write_opencv_calibration(run_dir, model, out_dir, **kw):
    """Calibrate the run with OpenCV (``calibrate_opencv``) and write it natively
    (``write_native_calibration``) as ``calibration_opencv_<model>.json`` with its
    ``_cameras.json`` + ``_poses.json`` in ``out_dir``."""
    r = calibrate_opencv(run_dir, model, **kw)
    cams, poses = write_native_calibration(out_dir, f"calibration_opencv_{model}", r["K"], r["dist"], r["poses"],
                                           r["image_size"], {"reprojection_error": r["rms"]})
    return {"cameras": cams, "poses": poses, "rms": r["rms"], "K": r["K"], "n_posed": len(r["poses"])}


def write_matlab_correspondences(run_dir, out_path):
    """The run's correspondences for MATLAB's ``estimateCameraParameters`` (or any calibrator), as
    JSON (``jsondecode`` reads it): per frame the image name, ``object_points`` N×3 in millimetres
    and ``image_points`` N×2 in MATLAB's frame (top-left pixel centre at (1, 1)). The
    calibration that comes back is expected beside it as ``calibration_matlab_<model>.json`` in
    the shape of the capture's MATLAB files: ``fx fy cx cy r1 r2 r3 t1 t2 mean_error poses``."""
    frames = load_correspondences(run_dir, "matlab")
    body = {"pixel_frame": "matlab", "unit": "mm",
            "frames": [{"t": f["t"], "image": f["image"], "target": f["target"], "object_points": f["object_points"].tolist(),
                        "image_points": f["image_points"].tolist()} for f in frames]}
    with open(out_path, "w") as f:
        json.dump(body, f)
    return len(frames)


def save_matlab(run_dir, out_path):
    """Write ``worldPoints`` M×2 and ``imagePoints`` M×2×numImages (1-based pixel centres) for
    ``estimateCameraParameters``, which needs the SAME points in every image, of one planar
    target: the points (by their position on the target) seen in all of its views are kept, the
    rest dropped. Returns (M, numImages)."""
    from scipy.io import savemat

    frames = load_correspondences(run_dir, "matlab")
    ts = sorted({f["target"] for f in frames})
    if len(ts) != 1:
        raise ValueError(f"{run_dir}: MATLAB takes one planar target; the views see {ts}")
    _require_planar(run_dir, frames, "estimateCameraParameters (worldPoints M×2)")
    index = [{tuple(X): i for i, X in enumerate(f["object_points"].tolist())} for f in frames]
    common = set(index[0])
    for ix in index[1:]:
        common &= set(ix)
    keys = sorted(common)
    if not keys:
        raise ValueError(f"{run_dir}: no point is seen in every frame")
    world = np.array([k[:2] for k in keys])
    image = np.stack([np.array([f["image_points"][ix[k]] for k in keys]) for f, ix in zip(frames, index)], axis=2)
    savemat(out_path, {"worldPoints": world, "imagePoints": image, "images": [f["image"] for f in frames]})
    return len(keys), len(frames)


def _main(argv):
    import argparse

    ap = argparse.ArgumentParser(description="BlobBoards correspondences → OpenCV / MATLAB")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("opencv", help="cv2.calibrateCamera on a run, written natively as "
                       "calibration_opencv_<model>_cameras.json + _poses.json (VisualGeometryToolkitIO's convert_opencv_calibration)")
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
