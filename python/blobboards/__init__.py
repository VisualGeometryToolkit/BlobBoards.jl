"""
BlobBoards — multi-scale blob calibration boards: make them, print them, find them.

This package wraps the Julia package BlobBoards.jl through juliacall, one
function per Julia entry point, with NumPy arrays and dataclasses in place of
Julia types. The first call starts Julia (seconds to minutes on a fresh
machine); later calls are fast.

Examples
--------
>>> from blobboards import blob_board, blob_pattern, short_side
>>> board = blob_board("A5", blob_sizes=(4.0, 0.2 * short_side), seed=0x1234)
>>> board.blobs.shape[1]
3
>>> pattern = blob_pattern(400, 300, blob_sizes=(12, None), seed=0x1234)
>>> pattern.pattern.shape
(300, 400)
"""

__version__ = "0.1.0"

from dataclasses import dataclass
from typing import Optional
import logging
import os
import numpy as np

_log = logging.getLogger("blobboards")

# =============================================================================
# Blob sizes: `blob_sizes=(min, max)`, printed disc diameters
# =============================================================================

@dataclass(frozen=True)
class ShortSideFraction:
    """
    A blob size given as a fraction of the pattern's short side.

    What ``fraction * short_side`` builds; valid as the max of ``blob_sizes``,
    a disc diameter like any other size. Mirrors Julia's ``ShortSideFraction``.

    Attributes
    ----------
    fraction : float
        The fraction, in (0, 1).

    Raises
    ------
    ValueError
        If ``fraction`` is not in (0, 1).

    Examples
    --------
    >>> from blobboards import short_side
    >>> (0.2 * short_side).fraction
    0.2
    """
    fraction: float
    _blobboards_short_side = True   # the tag the Julia side recognizes

    def __post_init__(self):
        if not 0 < self.fraction < 1:
            raise ValueError(f"a short_side fraction must be in (0, 1), got {self.fraction}")


class _ShortSide:
    """`0.2 * short_side` is 20% of the pattern's short side (mirrors Julia's `short_side`)."""

    def __rmul__(self, fraction):
        return ShortSideFraction(float(fraction))

    def __repr__(self):
        return "short_side"


short_side = _ShortSide()

# =============================================================================
# Dataclasses mirroring Julia types
# =============================================================================

@dataclass
class PatternConfig:
    """
    How a pattern was generated — mirrors Julia's ``PatternConfig``.

    Attributes
    ----------
    width, height : int or float
        Pattern size, in ``unit`` (whole pixels, an int, for a pattern).
    unit : str
        The unit of the sizes and σ as stored: "px" for a pattern, "pt" for a
        board (laid out on a fixed 0.01 pt grid; a board written before that,
        schema v3, is in "px" at its own density).
    min_scale, max_scale : float
        Smallest and largest blob σ in ``unit``, as stored (a disc's diameter is
        ``2 * cutoff * σ``).
    seed : int
        The placement seed.
    scales_per_octave : int
        Rungs per octave of the scale ladder.
    max_trials : int
        Placement attempts per blob.
    alpha : float
        Scale-sampling bias (Julia's ``α``).
    cutoff : float
        Disc radius in units of σ.
    raster_aligned : bool
        Whether blob centres sit on pixel centres (a board's sit on its 0.01 pt grid instead).
    polarity : str
        "dark" or "light".
    """
    width: float              # whole pixels (an int) in "px"; points on a v4 board
    height: float
    min_scale: float          # smallest σ, in `unit` (as stored)
    max_scale: float          # largest σ, in `unit` (as stored: the top rung of the scale ladder)
    seed: int
    scales_per_octave: int
    max_trials: int
    alpha: float
    cutoff: float
    raster_aligned: bool
    polarity: str
    unit: str = "px"


@dataclass
class PatternResult:
    """
    What ``blob_pattern`` returns — Julia's ``(pattern, blobs, config)``.

    Attributes
    ----------
    pattern : numpy.ndarray
        (H, W) grey image, values in [0, 1].
    blobs : numpy.ndarray
        (N, 3) array of ``[x, y, σ]`` in pixels.
    config : PatternConfig
        The generation parameters.
    config_hash : str
        SHA-256 of the canonical config: the pattern's identity.
    """
    pattern: np.ndarray      # (H, W) grayscale image
    blobs: np.ndarray        # (N, 3) array of [x, y, σ]
    config: PatternConfig
    config_hash: str         # SHA-256 of the canonical config (the pattern's identity)


@dataclass
class BoardConfig:
    """
    The board's physical layout — mirrors Julia's ``BoardConfig``.

    Attributes
    ----------
    canvas_width_mm, canvas_height_mm : float
        The page.
    board_width_mm, board_height_mm : float
        The board on the page (centred): the page less twice the margin.
    margin_width_mm, margin_height_mm : float
        The white margin left and right of the board, and above and below it.
    ruler_width_mm, corner_size_mm : float
        The ruler bands' width and the corner cells' side (the QR code, the uid
        characters); 0 when that mark is off.
    frame_padding_mm : float
        The white inset of the marks' content.
    major_tick_spacing_mm, minor_tick_spacing_mm : float
        Ruler tick intervals.
    major_tick_width_mm, minor_tick_width_mm : float
        Each tick's full drawn length across the frame band.
    """
    canvas_width_mm: float
    canvas_height_mm: float
    ruler_width_mm: float
    corner_size_mm: float
    frame_padding_mm: float
    board_width_mm: float = float("nan")
    board_height_mm: float = float("nan")
    margin_width_mm: float = float("nan")
    margin_height_mm: float = float("nan")
    major_tick_spacing_mm: float = float("nan")
    minor_tick_spacing_mm: float = float("nan")
    major_tick_width_mm: float = float("nan")
    minor_tick_width_mm: float = float("nan")


@dataclass
class BoardPreamble:
    """
    A board's full configuration — mirrors Julia's ``BoardPreamble``.

    Attributes
    ----------
    board_config : BoardConfig
        The physical layout.
    pattern_config : PatternConfig
        The blob pattern.
    """
    board_config: BoardConfig
    pattern_config: PatternConfig


@dataclass
class BoardResult:
    """
    A board — what ``blob_board``, ``load_board`` and ``render_board`` return;
    mirrors Julia's ``BoardBundle``.

    Attributes
    ----------
    blobs : numpy.ndarray
        (N, 3) array of ``[x, y, σ]`` in pixels, pattern frame, at ``density_dpi``.
    density_dpi : float
        The density (dots per inch) of the pixel coordinates in ``blobs``: the
        ``density`` given to the call, else the board's own raster density —
        the whole-number dpi at which its smallest disc is at least 16 pixels
        across (a schema v3 board: the density it was printed at). A board's
        layout has no density.
    preamble : BoardPreamble
        The configuration.
    config_hash : str
        SHA-256 identity; the board's uid is its first 3 hex characters. For a
        loaded or re-rendered board this is the stored identity.
    """
    blobs: np.ndarray
    density_dpi: float
    preamble: BoardPreamble
    config_hash: str

    @property
    def uid(self) -> str:
        """The 3-character uid printed in the board's corners."""
        return self.config_hash[:3]


@dataclass
class DiscDetection:
    """A detected blob with location, scale, and response.

    Mirrors Julia's DiscDetection structure with center, σ, response, polarity.
    """
    x: float
    y: float
    sigma: float
    response: float
    polarity: str  # "dark" or "light"

    @property
    def center(self) -> np.ndarray:
        """Blob center as (2,) array [x, y]. Mirrors Julia's det.center."""
        return np.array([self.x, self.y])

    @property
    def σ(self) -> float:
        """Scale parameter (sigma). Mirrors Julia's det.σ."""
        return self.sigma

    def to_array(self) -> np.ndarray:
        """Return [x, y, sigma] as (3,) array."""
        return np.array([self.x, self.y, self.sigma])

    def to_circle(self, cutoff: float = 2.0):
        """Return (x, y, radius) tuple for plotting."""
        return (self.x, self.y, cutoff * self.sigma)


@dataclass
class DetectionResult:
    """Result from detect_blobs."""
    detections: list

    def __len__(self):
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def __getitem__(self, idx):
        return self.detections[idx]

    def to_array(self) -> np.ndarray:
        """Return (N, 3) array of [x, y, sigma]."""
        if len(self.detections) == 0:
            return np.zeros((0, 3))
        return np.array([[d.x, d.y, d.sigma] for d in self.detections])

    def to_array_full(self) -> np.ndarray:
        """Return (N, 5) array of [x, y, sigma, response, polarity_code].

        polarity_code: 0 = dark, 1 = light
        """
        if len(self.detections) == 0:
            return np.zeros((0, 5))
        return np.array([
            [d.x, d.y, d.sigma, d.response, 0 if d.polarity == "dark" else 1]
            for d in self.detections
        ])

    def dark(self) -> "DetectionResult":
        """Return only dark blob detections."""
        return DetectionResult([d for d in self.detections if d.polarity == "dark"])

    def light(self) -> "DetectionResult":
        """Return only light blob detections."""
        return DetectionResult([d for d in self.detections if d.polarity == "light"])


# =============================================================================
# Julia interface
# =============================================================================

_jl = None
_initialized = False
_gpu = None   # None: not probed yet; else (usable: bool, reason: str)


def _init():
    """Initialize Julia and load BlobBoards. Called automatically on first use."""
    global _jl, _initialized
    if _initialized:
        return

    import os
    # Julia must handle its own signals: with juliacall's default (Python
    # handles them), Julia's GC safepoints and CUDA's runtime crash the process
    # with a segfault — measured here on the first GPU gallery build. Threads
    # let board loading/rasterizing run in parallel. Both only take effect if
    # juliacall has not been imported yet; a caller's own settings win.
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")
    from juliacall import Main as jl
    _jl = jl

    # Activate BlobBoards.jl's companion environment (authoring/Project.toml):
    # it dev's the checkout and lists PythonCall (BlobBoardsPythonExt, the
    # Python wrappers), a weak dependency the root project cannot load itself;
    # the old `Pkg.add("PythonCall")` here rewrote the tracked root
    # Project.toml. Board authoring is part of BlobBoards itself (2026-10-14).
    this_file = os.path.abspath(__file__)
    package_dir = os.path.dirname(this_file)  # blobboards/
    python_dir = os.path.dirname(package_dir)  # python/
    project_root = os.path.dirname(python_dir)  # BlobBoards.jl/
    authoring_env = os.path.join(project_root, "authoring")

    _jl.seval("import Pkg")
    _jl.Pkg.activate(authoring_env, io=_jl.devnull)

    # Ensure the registries are present and the environment is instantiated.
    # Without this, a fresh depot (e.g. /root/.julia in a clean container)
    # has the workspace unresolved and `using BlobBoards` fails with cryptic
    # precompile errors. No Manifest is committed (the repository is one Pkg
    # workspace), so the first instantiate RESOLVES it, and the stack packages
    # are in CauRegistry until they are registered in General: both are
    # needed. First import on a fresh box pays the download/precompile cost
    # once; subsequent imports no-op. Nothing here writes a Project.toml.
    _jl.seval("""
        let
            names = [r.name for r in Pkg.Registry.reachable_registries()]
            "General" in names || Pkg.Registry.add("General")
            "CauRegistry" in names ||
                Pkg.Registry.add(url = "https://github.com/prittjam/CauRegistry.jl")
            Pkg.instantiate(io=devnull)
        end
    """)

    _jl.seval("using BlobBoards")

    _initialized = True


def _probe_gpu():
    """Load CUDA + cuDNN into the Julia session (once) and report whether a GPU is usable.

    Lazy on purpose: board generation never needs CUDA, so it is loaded only
    when detection first asks for a device. CUDA and cuDNN are dependencies of
    the authoring environment, with the CUDA 12.9 runtime pinned there (ONNX
    Runtime's CUDA provider takes only 12.x); loading them activates
    BlobBoardsCUDAExt and ONNX Runtime's CUDA provider. On a machine without a
    usable NVIDIA GPU (macOS, no driver, CUDA_VISIBLE_DEVICES="") they load but
    `CUDA.functional()` is false.
    """
    global _gpu
    if _gpu is not None:
        return _gpu
    _init()
    try:
        _jl.seval("using CUDA, cuDNN")
        usable = bool(_jl.seval("CUDA.functional()"))
        reason = "" if usable else "CUDA.functional() is false (no NVIDIA GPU, no driver, or none visible)"
    except Exception as err:   # e.g. CUDA failed to load on this platform
        usable, reason = False, f"CUDA could not be loaded: {err}"
    _gpu = (usable, reason)
    return _gpu


def device(requested: str = "auto") -> str:
    """
    The device detection (``find_boards``, ``track_boards``) runs on.

    Parameters
    ----------
    requested : str, default "auto"
        "auto": "cuda" when a usable NVIDIA GPU is present, else "cpu".
        "cuda": "cuda", or an error when there is none. "cpu": "cpu".

    Returns
    -------
    str
        "cuda" or "cpu".

    Raises
    ------
    ValueError
        If ``requested`` is not "auto", "cuda" or "cpu".
    RuntimeError
        If ``requested="cuda"`` and no GPU is usable.

    Notes
    -----
    Loads CUDA into the Julia session (once) to check; board generation never
    does. ``CUDA_VISIBLE_DEVICES`` (set before Python starts) picks the GPU.

    Examples
    --------
    >>> import blobboards
    >>> blobboards.device() in ("cuda", "cpu")
    True
    """
    return _resolve_device(requested)


def _resolve_device(requested):
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError(f'device must be "auto", "cuda" or "cpu", got {requested!r}')
    if requested == "cpu":
        return "cpu"
    usable, reason = _probe_gpu()
    if usable:
        return "cuda"
    if requested == "cuda":
        raise RuntimeError(f'device="cuda" but no usable GPU: {reason}. Use device="cpu" (or "auto").')
    return "cpu"


def _pattern_config(pc) -> PatternConfig:
    size = int if str(pc.unit) == "px" else float  # a pattern's pixels are whole
    return PatternConfig(
        width=size(pc.width),
        height=size(pc.height),
        unit=str(pc.unit),
        min_scale=float(pc.min_scale),
        max_scale=float(pc.max_scale),
        seed=int(pc.seed),
        scales_per_octave=int(pc.scales_per_octave),
        max_trials=int(pc.max_trials),
        alpha=float(pc.alpha),
        cutoff=float(pc.cutoff),
        raster_aligned=bool(pc.raster_aligned),
        polarity=str(pc.polarity),
    )


def blob_pattern(width, height, *, blob_sizes=None, seed=None, dir=None,
                 scales_per_octave=None,
                 max_trials=None, alpha=None, cutoff=None,
                 raster_aligned=None, polarity=None) -> PatternResult:
    """
    Generate a blob pattern in pixels (no frame) — Julia's ``blob_pattern``.

    Parameters
    ----------
    width : int
        Pattern width in pixels.
    height : int
        Pattern height in pixels.
    blob_sizes : (min, max), optional
        Smallest and largest disc DIAMETER (2·cutoff·σ) in pixels (bare numbers)
        or Pint quantities in px. `max` may be `0.2 * short_side` (a fraction of
        the short side) or None (default: 20% of the short side). Default min:
        8 px at the default cutoff (σ = 2 px, the floor). Stored as σ in px; min
        and max are both exact.
    seed : int, optional
        Random seed for reproducibility.
    dir : str, optional
        Directory to save output files (PNG + JSON).
    scales_per_octave : int, optional
        Number of scales per octave (default: 3).
    max_trials : int, optional
        Maximum placement attempts (default: 5000).
    alpha : float, optional
        Scale-sampling bias, Julia's ``α``: how blob counts fall with size (default: 1.2).
    cutoff : float, optional
        Gaussian truncation radius in units of σ (default and recommended: 2.0).
    raster_aligned : bool, optional
        Align blob centers to pixel centers (default: True; False samples
        continuous centers). A board's centres are on its 0.01 pt layout grid instead.
    polarity : str, optional
        Blob polarity: "dark" or "light" (default: "dark").

    Returns
    -------
    PatternResult
        The image, the blobs ``[x, y, σ]`` (px), the config and its hash.

    Raises
    ------
    juliacall.JuliaError
        For invalid sizes (a min below the 8 px floor at the default cutoff, a
        max that does not fit the pattern) or a removed keyword.

    Examples
    --------
    >>> from blobboards import blob_pattern, short_side
    >>> r = blob_pattern(400, 300, blob_sizes=(12, 0.2 * short_side), seed=0x1234)
    >>> r.pattern.shape, r.blobs.shape[1]
    ((300, 400), 3)
    """
    _init()

    # Only what the caller gave crosses to Julia (Python's `alpha` is Julia's `α`).
    given = dict(
        seed=seed, dir=dir, blob_sizes=None if blob_sizes is None else tuple(blob_sizes),
        scales_per_octave=scales_per_octave, max_trials=max_trials, α=alpha,
        cutoff=cutoff, raster_aligned=raster_aligned, polarity=polarity,
    )
    kwargs = {k: v for k, v in given.items() if v is not None}

    result = _jl.BlobBoards.blob_pattern_python(
        width, height,
        **kwargs
    )

    # Convert to dataclass
    return PatternResult(
        pattern=np.asarray(result.pattern),
        blobs=np.asarray(result.blobs),
        config=_pattern_config(result.config),
        config_hash=str(result.config_hash),
    )


def suggest_blob_sizes(density, extent) -> tuple:
    """
    The ``blob_sizes`` of a board for a printer's density — Julia's
    ``suggest_blob_sizes``: the smallest and largest printed disc diameters,
    in millimetres.

    - smallest: the disc that prints its profile well at ``density`` — 3 mm at
      1200 dpi, inversely proportional to the density (6 mm at 600 dpi, 12 mm
      at 300). A printer makes grey from halftone cells, each a fixed number of
      printer dots, so the smallest disc whose profile spans enough of them
      grows as the density drops.
    - largest: 20% of the board's shorter side, but never more than 64 mm.

    A board has no density of its own and ``blob_sizes`` no default: this is
    how the printer enters a board's design,
    ``blob_board("A4", blob_sizes=suggest_blob_sizes(600, "A4"))``. Either
    end may also be given directly (the max as ``0.2 * short_side``: that
    fraction of the pattern's short side, no cap).

    Parameters
    ----------
    density : int or float
        The printer's density in dots per inch.
    extent : str or (width, height)
        A paper name (``"A4"``, ``"Letter"``, ...: the board ``blob_board`` makes
        on that paper, the paper less 10 mm on every side), or the board's
        width and height — millimetres as bare numbers, or Pint lengths.

    Returns
    -------
    (float, float)
        The smallest and largest diameters in millimetres.

    Examples
    --------
    >>> from blobboards import suggest_blob_sizes
    >>> suggest_blob_sizes(600, "A4")
    (6.0, 38.0)
    >>> suggest_blob_sizes(1200, (300, 200))
    (3.0, 40.0)
    """
    _init()
    if isinstance(extent, str):
        jl_extent = extent
    else:
        if len(extent) != 2:
            raise ValueError(f"extent must be a paper name or a (width, height) pair, got {extent!r}")
        w, h = (float(x.to("mm").magnitude) if hasattr(x, "to") else float(x) for x in extent)
        jl_extent = _jl.seval("(w, h) -> (w * BlobBoards.mm, h * BlobBoards.mm)")(w, h)
    sizes = _jl.seval("""let V = BlobBoards.VisualGeometryToolkitCore
                             (d, e) -> Tuple(Float64(V.ustrip(V.mm, s)) for s in BlobBoards.suggest_blob_sizes(d * V.dpi, e))
                         end""")(float(density), jl_extent)
    return (float(sizes[0]), float(sizes[1]))


def blob_board(paper, *, blob_sizes, seed=None,
               corner_size=None, ruler_width=None, frame_padding=None, margin=None, board_size=None,
               major_tick_spacing=None, minor_tick_spacing=None,
               major_tick_width=None, minor_tick_width=None,
               dir=None, format=None, density=None, icc_profile=None, id_label=None,
               shading_function=None, bleed=None,
               scales_per_octave=None,
               max_trials=None, alpha=None, cutoff=None,
               polarity=None,
               image_origin="colmap") -> BoardResult:
    """
    Generate a printable calibration board — Julia's ``blob_board``.

    Every keyword mirrors the Julia keyword of the same name (``alpha`` is
    Julia's ``α``); bare numbers are millimetres, and Pint quantities are
    accepted for every length. A board is laid out in physical units and has no
    pixel density: the same sheet prints on any printer. A density belongs to a
    raster output (``format="tiff"`` or ``"png"`` with ``density``), and the
    printer a board is sized for enters through ``blob_sizes``, with
    ``suggest_blob_sizes(density, extent)``. Where Julia names the output with a value
    (``Pdf()``, ``Tiff(density)``, ``Png(density)``), Python names it with
    ``format``, ``density``, ``icc_profile``, ``shading_function`` and ``bleed``. Removed
    keywords (``pdf_standard``, ``print_density``, ``blob_measure``,
    ``frame_style``, ``frame_width``, ``raster_aligned``) and the former
    positional ``dpi`` are not parameters, so giving one is a ``TypeError``.

    Parameters
    ----------
    paper : str or tuple
        A paper name (a key of ``PRINT_SIZES``: "A4", "Letter", …) or
        ``(width, height)`` in mm.
    blob_sizes : (min, max)
        Required. Smallest and largest printed disc DIAMETER (2·cutoff·σ) in
        millimeters (bare numbers) or Pint lengths (e.g.
        `(1 * ureg.mm, 12 * ureg.mm)`, inches work too). There is no default
        and both ends must be given; `None` for either is an error. The
        smallest disc is a choice about the printer: `suggest_blob_sizes(density, paper)`
        gives the one that prints well (3 mm at 1200 dpi, 6 mm at 600, 12 mm
        at 300) and the largest, 20% of the board's shorter side, at most 64 mm.
        Either end may be given directly; the max also as `0.2 * short_side`
        (exactly that fraction of the pattern's short side, no cap). Stored as σ in points.
    seed : int, optional
        Random seed for reproducibility.
    corner_size : float or pint.Quantity, optional
        The corner cells' side in mm (default: 10 mm): the QR code top-left, a
        uid character in each other corner; blobs keep 1 mm clear of them. 0
        turns them off. A Pint length works too.
    ruler_width : float or pint.Quantity, optional
        The ruler bands' width in mm (default: 10 mm), along each side between
        the corner cells. 0 turns the ruler off.
    frame_padding : float or pint.Quantity, optional
        White inset of the marks' content (ticks, QR code, uid) in mm (default: 0).
    major_tick_spacing, minor_tick_spacing : float or pint.Quantity, optional
        Ruler tick spacings in mm (while the ruler is on); defaults 10 mm / 5 mm.
    major_tick_width, minor_tick_width : float or pint.Quantity, optional
        Tick lengths in mm (while the ruler is on): the full drawn length across
        the ruler band, centred in it. Defaults: major_tick_width = the padded
        band (ruler_width − 2·frame_padding), minor_tick_width = half of
        major_tick_width; 0 < minor_tick_width ≤ major_tick_width ≤ the band.
    margin : tuple, optional
        The white margin around the board, which is centred on the paper, as
        (width, height) in mm: left and right of it, above and below. Default
        (10, 10). The board is the paper less twice the margin. The paper is
        the finished sheet: for a print shop that trims, keep the margin and
        give ``bleed``.
    board_size : tuple, optional
        The board instead, as (width, height) in mm; its margin is half the
        paper's excess. ``margin`` and ``board_size`` are mutually exclusive.
    dir : str, optional
        Directory to save output files (the board file + JSON). Without it no
        file is written.
    format : str, optional
        Output format: "pdf" (default: the PDF/X-4 print master, which has no
        density), "tiff" (the sheet converted at ``density``: LZW, resolution
        tags, ICC) or "png" (the same conversion, for digital use). "tiff" and
        "png" require ``density``.
    density : float or Pint quantity, optional
        Dots per inch. Required for "tiff" and "png": the density of the image,
        a whole number, at which the board's smallest blob must be at least
        2 pixels per σ. For a PDF it does not affect the file. In every case it
        is the density of the returned pixel blobs; default: the board's own
        raster density, the whole-number dpi at which its smallest disc is at
        least 16 pixels across (``density_dpi``).
    icc_profile : str, optional
        The OutputIntent / embedded profile for PDF and TIFF: "linear" (default,
        linear-reflectance grey) or the path of a grey ICC profile. Not for "png".
    id_label : bool, optional
        Print the board's identity (file stem, board code, BlobBoards
        version) in small type under the board, in the sheet margin
        (default: True; skipped with a warning when there is no margin).
    shading_function : str, optional
        The colour function of the blobs' radial shadings (always vector, one
        shading per blob, evaluated by the printer at its own resolution):
        "calculator" (default: the exact profile, a PostScript calculator
        function) or "sampled" (a sampled
        table, for a RIP that mishandles calculator functions). PDF only: "tiff"
        and "png" are the PDF converted at ``density``.
    bleed : float or pint.Quantity, optional
        A print shop's trim allowance in mm on every side (default: 0), PDF
        only — Julia's ``Pdf(bleed=…)``. The paper stays the finished sheet and
        the PDF's TrimBox; the page (MediaBox = BleedBox) grows by ``bleed`` on
        each side, a white strip the shop cuts off. The board, its uid and file
        name do not change; the JSON records the bleed, so a rebuild prints the
        same page. A Pint length works too.
    scales_per_octave : int, optional
        Number of scales per octave (default: 3).
    max_trials : int, optional
        Maximum placement attempts (default: 5000).
    alpha : float, optional
        Scale-sampling bias, Julia's ``α``: how blob counts fall with size (default: 1.2).
    cutoff : float, optional
        Gaussian truncation radius in units of σ (default and recommended: 2.0).
    polarity : str, optional
        Blob polarity: "dark" (dark blobs on light background, default) or
        "light" (light blobs on dark background for negative/inverted boards).
    image_origin : str
        Pixel coordinate convention for the returned blobs:
        "colmap" (first pixel centre at (0.5, 0.5); default), "opencv" ((0, 0))
        or "julia" ((1, 1)).

    Returns
    -------
    BoardResult
        Blobs (pixels at ``density``, else at the board's raster density), the
        preamble and the board's identity.

    Raises
    ------
    TypeError
        For a missing ``blob_sizes``, or a keyword that is not a parameter
        (including removed ones).
    juliacall.JuliaError
        For invalid values, checked before placement: sizes (including a
        ``None`` min or max), tick widths
        (``0 < minor ≤ major ≤ ruler_width − 2·frame_padding``), an unknown
        ``format``, "tiff" or "png" without ``density`` (or at a density that
        gives the smallest blob fewer than 2 pixels per σ),
        ``shading_function`` for a raster format, ``icc_profile`` for PNG, or
        an output keyword without ``dir``.

    Examples
    --------
    >>> from blobboards import blob_board, suggest_blob_sizes, short_side
    >>> b = blob_board("A5", blob_sizes=(12, 0.2 * short_side), seed=0xBEEF)
    >>> b.preamble.board_config.ruler_width_mm, len(b.uid)
    (10.0, 3)
    >>> b = blob_board("A4", blob_sizes=suggest_blob_sizes(600, "A4"), seed=0xBEEF)   # a 600 dpi printer: 6 mm … 38 mm
    >>> len(b.uid)
    3
    """
    _init()

    # Only what the caller gave crosses to Julia, which applies its own defaults
    # (configs/authoring.toml). Python's `alpha` is Julia's `α`.
    given = dict(
        seed=seed, ruler_width=ruler_width, corner_size=corner_size,
        frame_padding=frame_padding, margin=margin, board_size=board_size,
        dir=dir, format=format, density=density, icc_profile=icc_profile,
        id_label=id_label, shading_function=shading_function, bleed=bleed,
        scales_per_octave=scales_per_octave, max_trials=max_trials, α=alpha,
        cutoff=cutoff, polarity=polarity,
        major_tick_spacing=major_tick_spacing, minor_tick_spacing=minor_tick_spacing,
        major_tick_width=major_tick_width, minor_tick_width=minor_tick_width,
    )
    kwargs = {'blob_sizes': tuple(blob_sizes)}
    kwargs.update({k: v for k, v in given.items() if v is not None})

    result = _jl.BlobBoards.blob_board_python(
        paper,
        image_origin=image_origin,
        **kwargs
    )
    return _board_result(result)


def _board_result(result) -> BoardResult:
    bc = result.board_config
    board_config = BoardConfig(
        canvas_width_mm=float(bc.canvas_width_mm),
        canvas_height_mm=float(bc.canvas_height_mm),
        ruler_width_mm=float(bc.ruler_width_mm),
        corner_size_mm=float(bc.corner_size_mm),
        frame_padding_mm=float(bc.frame_padding_mm),
        board_width_mm=float(bc.board_width_mm),
        board_height_mm=float(bc.board_height_mm),
        margin_width_mm=float(bc.margin_width_mm),
        margin_height_mm=float(bc.margin_height_mm),
        major_tick_spacing_mm=float(bc.major_tick_spacing_mm),
        minor_tick_spacing_mm=float(bc.minor_tick_spacing_mm),
        major_tick_width_mm=float(bc.major_tick_width_mm),
        minor_tick_width_mm=float(bc.minor_tick_width_mm),
    )
    return BoardResult(
        blobs=np.asarray(result.blobs),
        density_dpi=float(result.density_dpi),
        preamble=BoardPreamble(board_config=board_config,
                               pattern_config=_pattern_config(result.pattern_config)),
        config_hash=str(result.config_hash),
    )


def load_board(json_path, *, image_origin="colmap", density=None) -> BoardResult:
    """
    Load a board from its JSON sidecar — Julia's ``load_board`` / ``load_bundle``.

    Every schema BlobBoards ever wrote is read (older ones are migrated in
    memory); the file is not rewritten. The identity is the stored one.

    Parameters
    ----------
    json_path : str
        The board's ``blob_board_<uid>.json``.
    image_origin : str, default "colmap"
        Pixel convention of the returned blobs: "colmap", "opencv" or "julia".
    density : float or Pint quantity, optional
        The density (dpi) the returned pixel blobs are at. Default: that of the
        board's image — a raster beside the JSON at the density its sidecar
        records, else the board's own raster density, the whole-number dpi at
        which its smallest disc is at least 16 pixels across. A schema v3 board
        takes only its own density.

    Returns
    -------
    BoardResult
        The board, with its stored ``config_hash``; the density used is
        ``density_dpi``.

    Raises
    ------
    juliacall.JuliaError
        If the file is not a board bundle or its schema is unknown, or
        ``density`` disagrees with a v3 board's or a stored raster's own.

    Examples
    --------
    >>> import os
    >>> from blobboards import load_board, example_data
    >>> d = example_data("boards")
    >>> b = load_board(os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith(".json"))[0]))
    >>> len(b.config_hash)
    64
    """
    _init()
    return _board_result(_jl.BlobBoards.load_board_python(
        str(json_path), image_origin=image_origin, density=density))


def render_board(json_path, *, dir, format=None, density=None, icc_profile=None,
                 id_label=False, shading_function=None, bleed=None, image_origin="colmap") -> BoardResult:
    """
    Re-render a stored board without re-running placement — Julia's ``render_board``.

    The pattern comes from the recorded blobs, the uid and file name from the
    stored identity; the QR carries the board code (the board's definition, from
    which a print alone rebuilds it), or the identity for a board stored before it.

    Parameters
    ----------
    json_path : str
        The board's JSON sidecar.
    dir : str
        Output directory (created if missing).
    format : str, optional
        "pdf" (default: PDF/X-4), "tiff" or "png"; "tiff" and "png" require
        ``density``.
    density : float or Pint quantity, optional
        Dots per inch. Required for "tiff" and "png": the density of the image,
        a whole number giving the smallest blob at least 2 pixels per σ (a
        schema v3 board converts only at its own density,
        ``load_board(json_path).density_dpi``).
        For a PDF it does not affect the file. In every case it is the density
        of the returned pixel blobs; default as for ``load_board``.
    icc_profile : str, optional
        "linear" (default) or a grey ICC profile path; not for PNG.
    id_label : bool, default False
        Print the identity line under the board (boards in circulation have none).
    shading_function : str, optional
        "calculator" (default) or "sampled" (PDF only; see ``blob_board``).
    bleed : float or pint.Quantity, optional
        A print shop's trim allowance in mm on every side (default: 0), PDF
        only; see ``blob_board``.
    image_origin : str, default "colmap"
        Pixel convention of the returned blobs.

    Returns
    -------
    BoardResult
        The board, with its stored ``config_hash``.

    Raises
    ------
    juliacall.JuliaError
        For an invalid output option (see ``blob_board``).

    Examples
    --------
    >>> import os, tempfile
    >>> from blobboards import load_board, render_board, example_data
    >>> d = example_data("boards")
    >>> j = os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith(".json"))[0])
    >>> out = tempfile.mkdtemp()
    >>> dpi = load_board(j).density_dpi
    >>> b = render_board(j, dir=out, format="png", density=dpi)
    >>> any(f.endswith(".png") for f in os.listdir(out))
    True
    """
    _init()
    return _board_result(_jl.BlobBoards.render_board_python(
        str(json_path), dir=str(dir), format=format, density=density,
        icc_profile=icc_profile, id_label=bool(id_label), shading_function=shading_function,
        bleed=bleed, image_origin=image_origin))


def example_data(name) -> str:
    """
    The directory of a bundled example dataset — Julia's ``BlobBoards.example_data``.

    Downloaded on first use from the package's artifact catalog.

    Parameters
    ----------
    name : str
        "boards" (50 board definitions), "still" (an image and its camera) or
        "video" (a short clip and its camera).

    Returns
    -------
    str
        The directory.

    Examples
    --------
    >>> import os
    >>> from blobboards import example_data
    >>> os.path.isdir(example_data("boards"))
    True
    """
    if name not in ("boards", "still", "video"):
        raise ValueError(f'name must be "boards", "still" or "video", got {name!r}')
    _init()
    return str(_jl.seval(f"BlobBoards.example_data(:{name})"))


def descriptor_network_path(config=None) -> str:
    """
    The descriptor network a run configuration names — Julia's ``descriptor_network_path``.

    Parameters
    ----------
    config : str, optional
        A run configuration TOML; default: the package's ``configs/default.toml``.
        A network named as an artifact is downloaded on first use.

    Returns
    -------
    str
        The ONNX file's path.

    Examples
    --------
    >>> from blobboards import descriptor_network_path
    >>> descriptor_network_path().endswith(".onnx")
    True
    """
    _init()
    if config is None:
        return str(_jl.BlobBoards.descriptor_network_path())
    return str(_jl.BlobBoards.descriptor_network_path(str(config)))


def __getattr__(name):
    # PRINT_SIZES: read from Julia on first access (module attribute, PEP 562)
    if name == "PRINT_SIZES":
        _init()
        sizes = _jl.seval("""let V = BlobBoards.VisualGeometryToolkitCore
                                 Dict(k => (Float64(V.ustrip(V.mm, v.width)), Float64(V.ustrip(V.mm, v.height)))
                                      for (k, v) in BlobBoards.PRINT_SIZES)
                             end""")
        val = {str(k): (float(v[0]), float(v[1])) for k, v in dict(sizes).items()}
        globals()["PRINT_SIZES"] = val
        return val
    raise AttributeError(f"module 'blobboards' has no attribute {name!r}")


def blobs(json_path, *, units="px", reference_frame="pattern", image_origin="colmap", density=None) -> np.ndarray:
    """
    A bundle's blobs in a chosen frame and unit — Julia's ``blobs(bundle, frame; units)``.

    Parameters
    ----------
    json_path : str
        A board or pattern JSON sidecar.
    units : str, default "px"
        "px", "mm" or "inch" (a pattern supports only "px").
    reference_frame : str, default "pattern"
        For a board: "pattern", "board", "canvas" (pixel frames, y down) or
        "world" (the board's centred, y-up plane in mm — what a homography from
        ``find_boards`` maps to image pixels).
    image_origin : str, default "colmap"
        Pixel convention for ``units="px"``: "colmap" (first pixel centre at
        (0.5, 0.5)), "opencv" ((0, 0)) or "julia" ((1, 1)).
    density : float or Pint quantity, optional
        For a board with ``units="px"``: the density the pixels are at (dpi, e.g.
        300). A stored v3 board's own by default; required for a v4 board, whose
        layout has no density. Not used for mm,
        inch or "world", nor for a pattern (its pixels are its own).

    Returns
    -------
    numpy.ndarray
        (N, 3) array of ``[x, y, σ]``.

    Examples
    --------
    >>> import os
    >>> from blobboards import blobs, example_data
    >>> d = example_data("boards")
    >>> j = os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith(".json"))[0])
    >>> blobs(j, reference_frame="world", units="mm").shape[1]
    3
    """
    _init()

    return np.asarray(_jl.BlobBoards.blobs_python(
        json_path,
        units=units,
        reference_frame=reference_frame,
        image_origin=image_origin,
        density=density,
    ))


def detect_blobs(
    img: np.ndarray,
    *,
    peak_threshold: float = 0.01,
    edge_threshold: float = 5.0,
    polarity: str = "dark",
    image_origin: str = "opencv",
) -> DetectionResult:
    """
    Detect blobs in an image using scale-space analysis.

    Parameters
    ----------
    img : ndarray
        Input image. Can be:
        - (H, W) grayscale float or uint8
        - (H, W, 3) RGB uint8
    peak_threshold : float
        Minimum response magnitude (default: 0.01, = Julia DetectConfig()).
        Lower values detect more blobs.
    edge_threshold : float
        Maximum edge score - lower is stricter (default: 5.0, = Julia DetectConfig()).
        Filters out edge-like responses.
    polarity : str
        Which blob types to detect:
        - "dark": Dark blobs on light background (default)
        - "light": Light blobs on dark background
        - "none": No polarity filtering (detect all)
    image_origin : str
        Output coordinate convention:
        - "opencv": (0, 0) at top-left pixel center (default, matches NumPy)
        - "colmap": (0.5, 0.5) at top-left pixel center
        - "julia": (1, 1) at top-left pixel center

    Returns
    -------
    DetectionResult
        Container with list of DiscDetection objects.
        Use .to_array() for (N, 3) NumPy array of [x, y, sigma].
        Use .dark() or .light() to filter by polarity.

    Examples
    --------
    >>> import numpy as np
    >>> from blobboards import detect_blobs
    >>> img = np.random.rand(480, 640)  # grayscale image
    >>> result = detect_blobs(img, peak_threshold=0.01)
    >>> print(f"Found {len(result)} blobs")
    >>> blobs = result.to_array()  # (N, 3) array of [x, y, sigma]

    >>> # Detect all blobs (no polarity filtering)
    >>> result = detect_blobs(img, polarity="none")
    >>> dark_blobs = result.dark()
    >>> light_blobs = result.light()
    """
    _init()

    # Ensure image is in correct format
    img = np.asarray(img)

    result = _jl.BlobBoards.detect_blobs_python(
        img,
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
        polarity=polarity,
        image_origin=image_origin,
    )

    # Convert tuples to DiscDetection objects
    detections = [
        DiscDetection(
            x=float(d[0]),
            y=float(d[1]),
            sigma=float(d[2]),
            response=float(d[3]),
            polarity=str(d[4]),
        )
        for d in result
    ]

    return DetectionResult(detections=detections)


# =============================================================================
# Track loading  (writing lives in Julia: `build_tracks_for_sequences`)
# =============================================================================

@dataclass
class TrackData:
    """Tracks loaded from HDF5. Pure Python — no Julia required."""
    patches: np.ndarray              # (N, P, P) float32
    track_ids: np.ndarray            # (N,) int32
    frame_ids: np.ndarray            # (N,) int32
    track_lengths: np.ndarray        # (n_tracks,) int32
    untracked: np.ndarray            # (M, P, P) float32
    untracked_frame_ids: np.ndarray  # (M,) int32
    absolute_homographies: Optional[np.ndarray] = None  # (n_frames, 3, 3) float64



def load_tracks(path: str, sequence: str = None) -> TrackData:
    """
    Load tracks from a .tracks file (HDF5 format). Pure Python -- no Julia required.

    The file contains one or more sequence groups (one per video). By default
    all sequences are loaded and concatenated; supply ``sequence="name"`` to
    load a single one.

    Parameters
    ----------
    path : str
        Path to .tracks file written by `build_tracks_for_sequences`.
    sequence : str, optional
        Sequence name to load. If None (default), loads all sequences.

    Returns
    -------
    TrackData
        Dataclass with patches, track_ids, frame_ids, track_lengths,
        untracked, untracked_frame_ids.
    """
    import h5py

    all_patches = []
    all_track_ids = []
    all_frame_ids = []
    all_track_lengths = []
    all_untracked = []
    all_untracked_fids = []

    with h5py.File(path, 'r') as f:
        seqs = f['sequences']
        seq_names = [sequence] if sequence is not None else list(seqs.keys())
        track_id_offset = 0

        for name in seq_names:
            sg = seqs[name]
            g = sg['tracks']
            # HDF5.jl reverses dims on write, so h5py reads (N, P, P) directly
            patches = np.array(g['patches'], dtype=np.float32)
            track_ids = np.array(g['track_id'], dtype=np.int32) + track_id_offset
            frame_ids = np.array(g['frame_id'], dtype=np.int32)
            track_lengths = np.array(g['track_lengths'], dtype=np.int32)

            all_patches.append(patches)
            all_track_ids.append(track_ids)
            all_frame_ids.append(frame_ids)
            all_track_lengths.append(track_lengths)
            track_id_offset += int(g['n_tracks'][()])

            if 'untracked' in sg:
                ug = sg['untracked']
                all_untracked.append(np.array(ug['patches'], dtype=np.float32))
                if 'frame_id' in ug:
                    all_untracked_fids.append(np.array(ug['frame_id'], dtype=np.int32))
                else:
                    all_untracked_fids.append(np.zeros(len(all_untracked[-1]), dtype=np.int32))

            if 'absolute_homographies' in g:
                # Optional absolute homographies (n_frames, 3, 3)
                abs_h = np.stack(
                    list(map(
                        lambda x: np.array([*x]).reshape(3, 3).transpose(),
                        [g["absolute_homographies"][i][0] for i in range(len(g["absolute_homographies"]))]
                    )),
                    axis=0
                ).astype(np.float64)
                absolute_homographies = abs_h
            else:
                absolute_homographies = None

    patches = np.concatenate(all_patches) if all_patches else np.zeros((0, 64, 64), dtype=np.float32)
    track_ids = np.concatenate(all_track_ids) if all_track_ids else np.zeros(0, dtype=np.int32)
    frame_ids = np.concatenate(all_frame_ids) if all_frame_ids else np.zeros(0, dtype=np.int32)
    track_lengths = np.concatenate(all_track_lengths) if all_track_lengths else np.zeros(0, dtype=np.int32)
    untracked = np.concatenate(all_untracked) if all_untracked else np.zeros((0, patches.shape[1], patches.shape[2]), dtype=np.float32)
    untracked_fids = np.concatenate(all_untracked_fids) if all_untracked_fids else np.zeros(0, dtype=np.int32)

    return TrackData(
        patches=patches,
        track_ids=track_ids,
        frame_ids=frame_ids,
        track_lengths=track_lengths,
        untracked=untracked,
        untracked_frame_ids=untracked_fids,
        absolute_homographies=absolute_homographies,
    )


def plot_tracks(data, track_indices=None, max_tracks=20, max_cols=None):
    """
    Visualize tracks as a grid: one row per track, one column per observation.

    Tracks are sorted longest-first. Pure Python using matplotlib.

    Parameters
    ----------
    data : TrackData or str
        TrackData object or path to .tracks file.
    track_indices : array-like, optional
        Which track IDs (1-based, matching Julia) to show.
        Default: longest max_tracks tracks.
    max_tracks : int
        Maximum number of tracks to display (default: 20).
    max_cols : int, optional
        Maximum columns (observations) to show per track. Default: all.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if isinstance(data, str):
        data = load_tracks(data)

    unique_tracks = np.unique(data.track_ids)

    if track_indices is not None:
        unique_tracks = np.array([t for t in track_indices if t in unique_tracks])

    # Compute length of each track and sort longest-first
    track_lens = np.array([int((data.track_ids == tid).sum()) for tid in unique_tracks])
    order = np.argsort(-track_lens)
    unique_tracks = unique_tracks[order]

    if len(unique_tracks) > max_tracks:
        unique_tracks = unique_tracks[:max_tracks]

    n_rows = len(unique_tracks)
    if n_rows == 0:
        raise ValueError("No tracks to plot")

    # Find max track length for grid sizing
    n_cols = max(int((data.track_ids == tid).sum()) for tid in unique_tracks)
    if max_cols is not None:
        n_cols = min(n_cols, max_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.0, n_rows * 1.0),
                             squeeze=False)

    for row, tid in enumerate(unique_tracks):
        mask = data.track_ids == tid
        patches = data.patches[mask]
        frames = data.frame_ids[mask]

        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(patches):
                ax.imshow(patches[col].T[::-1], cmap='gray', vmin=0, vmax=1)
                ax.set_title(str(frames[col]), fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.tight_layout(pad=0.3)
    return fig

def find_boards(board_paths, model_path, img, *, gallery_path,
                camera_json=None, device="auto",
                min_inliers=None, outlier_halfwidth=None,
                verify_similarity=None, min_appearance_support=None,
                image_origin="opencv"):
    """
    Find known boards in an image and estimate each one's homography or pose —
    Julia's ``find_boards(board_feature_dict, descriptor_network, img)``.

    The gallery and the network lead, the image follows. The network, the
    gallery's features and the bundles are built once per process and reused.

    Parameters
    ----------
    board_paths : list[str]
        Board-bundle JSON paths (the gallery of candidate boards).
    model_path : str
        ONNX descriptor model path.
    img : np.ndarray
        Grayscale image, (H, W), uint8 or float in [0, 1].
    gallery_path : str
        JLD2 feature-cache path (created on first use).
    camera_json : str, optional
        Camera-calibration JSON → calibrated P3P path. Omit for the
        homography path.
    device : str
        "auto" (default): the GPU when a usable NVIDIA GPU is present, else the
        CPU, with a warning logged on the "blobboards" logger. "cuda": the GPU,
        or `RuntimeError` when there is none. "cpu": the CPU. The detection
        stages and the descriptor network (ONNX Runtime's CUDA or CPU
        provider) run on the same device. `blobboards.device()` resolves it
        without running detection; each result also says where it ran.
    min_inliers, outlier_halfwidth : optional
        Geometry gates (defaults from the Julia side).
    verify_similarity, min_appearance_support : optional
        Appearance-verification knobs — the mandatory per-inlier descriptor
        gate (defaults 0.6 / 0.5 on the Julia side).
    image_origin : str
        Pixel convention of the returned homography: "opencv" (default, as
        `detect_blobs`: top-left pixel centre at (0, 0)), "colmap" ((0.5, 0.5))
        or "julia" ((1, 1)).

    Returns
    -------
    list[dict]
        One dict per verified board: `uid`, `model`, `quality`, `n_inliers`,
        `n_csponds`, `weights` (np.ndarray), `device` ("cuda" or "cpu") and
        `execution_provider` (the network's ONNX Runtime provider, "cuda" or
        "cpu"). `model` is the 3×3 homography H
        from the board's world plane — centred, y up, in millimetres — to
        pixels in `image_origin`'s convention; on the pose path (`camera_json`)
        it is `{"R": 3×3, "t": 3-vector}`, the board → camera pose in
        millimetres, which carries no pixel convention.

    Raises
    ------
    TypeError
        For arguments in the retired order or of the wrong shape.
    ValueError
        For an unknown ``device``.
    RuntimeError
        For ``device="cuda"`` without a usable GPU.

    Examples
    --------
    >>> import os
    >>> import numpy as np
    >>> from blobboards import find_boards, example_data, descriptor_network_path
    >>> d = example_data("boards")
    >>> paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    >>> found = find_boards(paths, descriptor_network_path(), np.zeros((480, 640)),
    ...                     gallery_path="gallery_doc.jld2")   # a blank image: no boards
    >>> found
    []
    """
    # The image moved LAST (the gallery and the network lead). Refuse a call in
    # the retired `(img, board_paths, model_path)` order here, before Julia
    # starts: it would otherwise stringify a list as the model path.
    if isinstance(board_paths, (str, bytes)) or getattr(board_paths, "ndim", 1) != 1:
        raise TypeError("find_boards(board_paths, model_path, img, ...): board_paths "
                        "must be a sequence of board-bundle JSON paths")
    if not isinstance(model_path, (str, os.PathLike)):
        raise TypeError("find_boards(board_paths, model_path, img, ...): model_path "
                        f"must be a path, got {type(model_path).__name__}")
    img = np.asarray(img)
    if img.ndim != 2:
        raise TypeError("find_boards(board_paths, model_path, img, ...): img must be "
                        f"a 2-D grayscale array, got ndim={img.ndim}")
    _init()
    dev = _detection_device(device, "find_boards")
    kwargs = {"gallery_path": gallery_path, "device": dev, "image_origin": str(image_origin)}
    if camera_json is not None:
        kwargs["camera_json"] = camera_json
    if min_inliers is not None:
        kwargs["min_inliers"] = int(min_inliers)
    if outlier_halfwidth is not None:
        kwargs["outlier_halfwidth"] = float(outlier_halfwidth)
    if verify_similarity is not None:
        kwargs["verify_similarity"] = float(verify_similarity)
    if min_appearance_support is not None:
        kwargs["min_appearance_support"] = float(min_appearance_support)
    result = _jl.BlobBoards.find_boards_python(list(board_paths), str(model_path),
                                               img, **kwargs)
    out = []
    for b in result:
        d = dict(b)
        d["weights"] = np.asarray(d["weights"])
        d["model"] = _py_model(d["model"])
        out.append(d)
    return out


def _py_model(m):
    return {str(k): np.asarray(v) for k, v in dict(m).items()} if hasattr(m, "keys") else np.asarray(m)


def _detection_device(device, caller):
    dev = _resolve_device(device)
    if device == "auto" and dev == "cpu":
        _log.warning("blobboards.%s: no usable GPU (%s); running on the CPU, "
                     "which is about 10x slower", caller, _probe_gpu()[1])
    _log.info("blobboards.%s: device=%s", caller, dev)
    return dev


def track_boards(board_paths, model_path, frames, *, gallery_path, camera_json=None,
                 device="auto", image_origin="opencv"):
    """
    Track boards through a video or a sequence of frames — Julia's ``track_boards``.

    Each board is carried from frame to frame by motion-guided association and
    a per-frame fit; full discovery (``find_boards``) runs on keyframes.

    Parameters
    ----------
    board_paths : list[str]
        Board-bundle JSON paths (the gallery).
    model_path : str
        ONNX descriptor model path (``descriptor_network_path()``).
    frames : str or iterable of numpy.ndarray
        A video file path (streamed), or grey frames (H, W).
    gallery_path : str
        JLD2 feature-cache path (created on first use).
    camera_json : str, optional
        Camera-calibration JSON → poses instead of homographies.
    device : str, default "auto"
        "auto", "cuda" or "cpu", as for ``find_boards``.
    image_origin : str, default "opencv"
        Pixel convention of returned homographies: "opencv", "colmap" or "julia".

    Returns
    -------
    dict[str, list[dict]]
        Per board uid, one dict per frame it was found in: ``frame`` (1-based,
        as Julia), ``model`` (3×3 homography, or ``{"R", "t"}`` with a camera),
        ``quality``, ``n_inliers``, ``source`` ("track", "init" or "detect"),
        ``device`` and ``execution_provider``.

    Raises
    ------
    ValueError
        For an unknown ``device``.
    RuntimeError
        For ``device="cuda"`` without a usable GPU.

    Examples
    --------
    >>> import os
    >>> import numpy as np
    >>> from blobboards import track_boards, example_data, descriptor_network_path
    >>> d = example_data("boards")
    >>> paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    >>> tracks = track_boards(paths, descriptor_network_path(), [np.zeros((480, 640))] * 2,
    ...                       gallery_path="gallery_doc.jld2")   # blank frames: no boards
    >>> tracks
    {}
    """
    if isinstance(board_paths, (str, bytes)):
        raise TypeError("track_boards(board_paths, model_path, frames, ...): board_paths "
                        "must be a sequence of board-bundle JSON paths")
    _init()
    dev = _detection_device(device, "track_boards")
    if not isinstance(frames, (str, os.PathLike)):
        frames = [np.asarray(f) for f in frames]
    else:
        frames = str(frames)
    kwargs = {"gallery_path": gallery_path, "device": dev, "image_origin": str(image_origin)}
    if camera_json is not None:
        kwargs["camera_json"] = camera_json
    result = _jl.BlobBoards.track_boards_python(list(board_paths), str(model_path), frames, **kwargs)
    out = {}
    for uid, dets in dict(result).items():
        rows = []
        for det in dets:
            r = dict(det)
            r["model"] = _py_model(r["model"])
            rows.append(r)
        out[str(uid)] = rows
    return out


# =============================================================================
# Public surface — one-to-one with the Julia entry points. Everything else in
# this module (detect_blobs, load_tracks, ...) is advanced/unstable API.
# =============================================================================
__all__ = [
    "blob_board", "blob_pattern",      # generation
    "short_side", "ShortSideFraction", # `0.2 * short_side`: a blob size relative to the short side
    "suggest_blob_sizes",                   # a board's blob_sizes (mm) for a printer's dpi and the board's extent
    "load_board", "render_board",      # stored boards: load, re-render
    "blobs",                           # bundle accessor (frames, units)
    "find_boards", "track_boards",     # detection: stills, video
    "device",                          # where detection runs: "cuda" or "cpu"
    "example_data", "descriptor_network_path", "PRINT_SIZES",
    "BoardConfig", "BoardPreamble", "BoardResult", "PatternConfig", "PatternResult",
]
# detect_blobs / track utilities remain importable as advanced API but are not
# part of the supported surface.
