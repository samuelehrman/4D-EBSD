"""
band_analysis.py
----------------
Interactive band (rectangle ROI) analysis for pairs of experimental and
simulated 4D-EBSD Kikuchi patterns.

Usage (oimapi conda env, run as Administrator):
    python band_analysis.py

Steps:
  1. Enter two pattern indices and press Load.
  2. Click once on a pattern to place the centerline start point.
  3. Click again to place the end point – the rectangle appears immediately.
  4. Drag either cyan endpoint dot to adjust the line position; update the
     Width field to change the rectangle width.
  5. Press Analyze to extract the band data and show cross-band profiles.

Mode 1 – Shared     : one rectangle is drawn and propagates to all four patterns.
Mode 2 – Independent: draw a rectangle on Experimental 1 (propagates to
                      Simulated 1) and a separate one on Experimental 2
                      (propagates to Simulated 2).

Profile convention:
  For a rectangle that is L pixels long and W pixels wide, the band array has
  shape (L, W).  Summing along axis-0 (the centerline direction) collapses each
  column of W values into one number, giving a profile of length W.
  The x-axis of the profile is centred at 0 (the drawn centerline).
"""

import os
import sys
import threading

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.ndimage import map_coordinates
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sim_compare import (
    request_simulated,
    load_experimental,
    _normalize,
    _match_sizes,
    _to_2d,
    _check_admin,
)
from ANG import Ang
from UP2 import UP2

# ---------------------------------------------------------------------------
# Channel metadata (mirrors viewer.py)
# ---------------------------------------------------------------------------
CHANNEL_NAMES = [
    "phi1 (Euler)", "PHI (Euler)", "phi2 (Euler)",
    "X Position", "Y Position",
    "IQ (Image Quality)", "CI (Confidence Index)",
    "Phase Index", "SEM Signal", "Fit",
]
CHANNEL_CMAPS = [
    "hsv", "hsv", "hsv",
    "viridis", "viridis",
    "gray", "viridis", "tab10", "gray", "plasma",
]
DEFAULT_CHANNEL = 6  # CI


# ---------------------------------------------------------------------------
# Rectangle state
# ---------------------------------------------------------------------------

class RectState:
    """Holds the two centerline endpoints and width that define a rectangle."""

    def __init__(self):
        self.p1: tuple | None = None   # (col, row) in image data coordinates
        self.p2: tuple | None = None
        self.width: float = 20.0

    @property
    def complete(self) -> bool:
        return self.p1 is not None and self.p2 is not None

    def get_corners(self) -> np.ndarray | None:
        """Return (4, 2) array of (col, row) rectangle corners, or None."""
        if not self.complete:
            return None
        p1, p2 = np.array(self.p1, float), np.array(self.p2, float)
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1e-6:
            return None
        along = d / length
        perp = np.array([-along[1], along[0]])
        hw = self.width / 2.0
        return np.array([
            p1 + hw * perp,
            p1 - hw * perp,
            p2 - hw * perp,
            p2 + hw * perp,
        ])


# ---------------------------------------------------------------------------
# Band extraction
# ---------------------------------------------------------------------------

def extract_band(image: np.ndarray, p1, p2, width: float) -> np.ndarray:
    """
    Sample a rotated rectangular region from *image*.

    Parameters
    ----------
    image : 2-D float array (row, col indexing)
    p1, p2 : (col, row) endpoints of the rectangle centerline
    width : total width of the rectangle in pixels

    Returns
    -------
    band : ndarray, shape (length_px, width_px)
        axis-0 runs along the centerline  – sum over this axis to get profile
        axis-1 runs across the centerline – this is the width dimension
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1:
        return np.zeros((1, max(1, int(round(width)))))

    along = d / length
    perp  = np.array([-along[1], along[0]])

    length_px = max(1, int(round(length)))
    width_px  = max(1, int(round(width)))

    t_along = np.linspace(0.0, length, length_px)           # (L,)
    t_perp  = np.linspace(-width / 2.0, width / 2.0, width_px)  # (W,)

    t_a, t_p = np.meshgrid(t_along, t_perp, indexing='ij')  # (L, W) each

    col_coords = p1[0] + t_a * along[0] + t_p * perp[0]
    row_coords = p1[1] + t_a * along[1] + t_p * perp[1]

    return map_coordinates(
        image,
        [row_coords, col_coords],
        order=1,
        mode='constant',
        cval=0.0,
    )   # shape (length_px, width_px)


# ---------------------------------------------------------------------------
# Mode 3 – Line-scan window
# ---------------------------------------------------------------------------

class Mode3Window:
    """
    Standalone Toplevel for Mode 3.

    Workflow:
      1. Load ANG + UP2 files.
      2. Draw a two-click scan line on the EBSD map.
      3. The middle pattern (exp + sim) is loaded automatically.
      4. Draw a band rectangle on the pattern pair.
      5. Press Analyze – 10 (configurable) evenly-spaced patterns along the
         line are each processed with the same rectangle and their cross-band
         intensity profiles are plotted together.
    """

    _PAT_LABELS = ["Experimental (middle)", "Simulated (middle)"]

    def __init__(self, parent_root: tk.Tk):
        self.win = tk.Toplevel(parent_root)
        self.win.title("Mode 3 – Line Scan")
        self.win.minsize(1200, 650)

        # ANG / UP2
        self.ang: Ang | None = None
        self.up2: UP2 | None = None
        self.ang_data: np.ndarray | None = None
        self._map_colorbar = None

        # Map display controls
        self._vmin_var  = tk.StringVar()
        self._vmax_var  = tk.StringVar()
        self.scale_mode = tk.StringVar(value="Linear")

        # Map scan line (imshow col, row coords)
        self.map_p1: tuple | None = None
        self.map_p2: tuple | None = None
        self._map_draw_phase = 'idle'
        self._map_drag_target: str | None = None
        self._map_line_artist = None
        self._map_endpt_artist = None
        self._map_prev_line = None
        self._map_mid_marker = None

        # Middle pattern
        self.mid_exp: np.ndarray | None = None
        self.mid_sim: np.ndarray | None = None
        self.mid_pat_idx: int | None = None

        # Rectangle on pattern pair
        self.rect = RectState()
        self._poly_patches  = [None, None]
        self._cline_artists = [None, None]
        self._endpt_artists = [None, None]
        self._prev_lines_pat = [None, None]
        self._pat_draw_phase = 'idle'
        self._drag_target: str | None = None

        self._build_ui()
        self._connect_events()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        ctrl = ttk.Frame(self.win, padding=4)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(ctrl, text="Load ANG", command=self._load_ang).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Load UP2", command=self._load_up2).pack(side=tk.LEFT, padx=4)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(ctrl, text="Channel:").pack(side=tk.LEFT)
        self.channel_combo = ttk.Combobox(ctrl, values=CHANNEL_NAMES, width=22, state="disabled")
        self.channel_combo.current(DEFAULT_CHANNEL)
        self.channel_combo.pack(side=tk.LEFT, padx=4)
        self.channel_combo.bind("<<ComboboxSelected>>", lambda _: self._update_map())

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(ctrl, text="Width (px):").pack(side=tk.LEFT)
        self.width_var = tk.StringVar(value="20")
        ttk.Entry(ctrl, textvariable=self.width_var, width=6).pack(side=tk.LEFT, padx=2)
        self.width_var.trace_add('write', lambda *_: self._on_width_change())

        ttk.Label(ctrl, text="  N patterns:").pack(side=tk.LEFT, padx=(8, 0))
        self.npatterns_var = tk.StringVar(value="10")
        ttk.Entry(ctrl, textvariable=self.npatterns_var, width=4).pack(side=tk.LEFT, padx=2)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(ctrl, text="Scale:").pack(side=tk.LEFT)
        scale_combo = ttk.Combobox(
            ctrl, textvariable=self.scale_mode,
            values=["Linear", "Log"], width=8, state="readonly",
        )
        scale_combo.pack(side=tk.LEFT, padx=2)
        scale_combo.bind("<<ComboboxSelected>>", lambda _: self._update_map())

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(ctrl, text="Reset Line",  command=self._reset_map_line).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Reset Rect",  command=self._reset_rect).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Analyze",     command=self._analyze).pack(side=tk.LEFT, padx=4)

        # Bounds bar
        bounds_bar = ttk.Frame(self.win, padding=(4, 0, 4, 2))
        bounds_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bounds_bar, text="Map bounds:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(bounds_bar, text="Min:").pack(side=tk.LEFT)
        vmin_entry = ttk.Entry(bounds_bar, textvariable=self._vmin_var, width=10)
        vmin_entry.pack(side=tk.LEFT, padx=(2, 8))
        vmin_entry.bind("<Return>",   lambda _: self._update_map())
        vmin_entry.bind("<FocusOut>", lambda _: self._update_map())
        ttk.Label(bounds_bar, text="Max:").pack(side=tk.LEFT)
        vmax_entry = ttk.Entry(bounds_bar, textvariable=self._vmax_var, width=10)
        vmax_entry.pack(side=tk.LEFT, padx=(2, 8))
        vmax_entry.bind("<Return>",   lambda _: self._update_map())
        vmax_entry.bind("<FocusOut>", lambda _: self._update_map())
        ttk.Button(bounds_bar, text="Reset", command=self._reset_bounds).pack(side=tk.LEFT)

        # Status bar
        self.status_var = tk.StringVar(value="Load ANG and UP2 files to begin.")
        ttk.Label(self.win, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

        # PanedWindow: left = map, right = pattern pair
        paned = ttk.PanedWindow(self.win, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: EBSD map
        map_frame = ttk.LabelFrame(paned, text="EBSD Map  (draw scan line here)", padding=2)
        paned.add(map_frame, weight=1)

        self.map_fig = Figure(figsize=(5, 5), constrained_layout=True)
        self.map_ax  = self.map_fig.add_subplot(111)
        self.map_ax.set_axis_off()
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=map_frame)
        self.map_canvas.draw()
        self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        map_nav = NavigationToolbar2Tk(self.map_canvas, map_frame)
        map_nav.update()

        # Right: pattern pair
        pat_frame = ttk.LabelFrame(paned, text="Middle pattern pair  (draw band rectangle here)", padding=2)
        paned.add(pat_frame, weight=2)

        self.pat_fig  = Figure(figsize=(8, 4), constrained_layout=True)
        self.pat_axes = [self.pat_fig.add_subplot(1, 2, i + 1) for i in range(2)]
        for ax, lbl in zip(self.pat_axes, self._PAT_LABELS):
            ax.set_title(lbl, fontsize=10)
            ax.set_axis_off()
        self.pat_canvas = FigureCanvasTkAgg(self.pat_fig, master=pat_frame)
        self.pat_canvas.draw()
        self.pat_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _connect_events(self):
        self.map_canvas.mpl_connect('button_press_event',   self._map_on_press)
        self.map_canvas.mpl_connect('motion_notify_event',  self._map_on_motion)
        self.map_canvas.mpl_connect('button_release_event', self._map_on_release)

        self.pat_canvas.mpl_connect('button_press_event',   self._pat_on_press)
        self.pat_canvas.mpl_connect('motion_notify_event',  self._pat_on_motion)
        self.pat_canvas.mpl_connect('button_release_event', self._pat_on_release)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_ang(self):
        path = filedialog.askopenfilename(
            parent=self.win, title="Open ANG File",
            filetypes=[("ANG files", "*.ang"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.status_var.set("Parsing ANG file…")
            self.win.update_idletasks()
            self.ang = Ang(path)
            self.ang_data = self.ang.generate_np_array()
            self.channel_combo.config(state="readonly")
            self._update_map()
            self.status_var.set(
                f"ANG: {os.path.basename(path)}  {self.ang.nrows}×{self.ang.ncols}  |  "
                "Click the map to draw the scan line."
            )
        except Exception as exc:
            messagebox.showerror("ANG Load Error", str(exc), parent=self.win)

    def _load_up2(self):
        path = filedialog.askopenfilename(
            parent=self.win, title="Open UP2 File",
            filetypes=[("UP2 files", "*.up2"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.up2 = UP2(path)
            self.status_var.set(
                self.status_var.get() + f"  |  UP2: {os.path.basename(path)}"
            )
        except Exception as exc:
            messagebox.showerror("UP2 Load Error", str(exc), parent=self.win)

    # ------------------------------------------------------------------
    # Map display
    # ------------------------------------------------------------------

    def _reset_bounds(self):
        self._vmin_var.set("")
        self._vmax_var.set("")
        self._update_map()

    def _update_map(self):
        if self.ang_data is None:
            return
        ch      = self.channel_combo.current()
        data    = self.ang_data[:, :, ch]
        cmap    = CHANNEL_CMAPS[ch]
        use_log = self.scale_mode.get() == "Log"

        try:
            manual_vmin = float(self._vmin_var.get()) if self._vmin_var.get().strip() else None
        except ValueError:
            manual_vmin = None
        try:
            manual_vmax = float(self._vmax_var.get()) if self._vmax_var.get().strip() else None
        except ValueError:
            manual_vmax = None

        if use_log:
            positive = data[np.isfinite(data) & (data > 0)]
            if positive.size == 0:
                use_log = False
                vmin, vmax = manual_vmin, manual_vmax
            else:
                auto_vmin = float(positive.min())
                auto_vmax = float(positive.max())
                vmin = manual_vmin if (manual_vmin is not None and manual_vmin > 0) else auto_vmin
                vmax = manual_vmax if manual_vmax is not None else auto_vmax
                if vmin == vmax:
                    vmax = vmin * 1.001
                data = np.ma.masked_less_equal(data, 0)
        else:
            vmin, vmax = manual_vmin, manual_vmax

        self.map_ax.cla()
        if use_log:
            im = self.map_ax.imshow(
                data, cmap=cmap, origin='upper', interpolation='nearest',
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
        else:
            im = self.map_ax.imshow(
                data, cmap=cmap, origin='upper', interpolation='nearest',
                vmin=vmin, vmax=vmax,
            )
        self.map_ax.set_title(
            f"{CHANNEL_NAMES[ch]} (log)" if use_log else CHANNEL_NAMES[ch],
            fontsize=9,
        )
        self.map_ax.set_xlabel("Column")
        self.map_ax.set_ylabel("Row")

        if self._map_colorbar is None:
            self._map_colorbar = self.map_fig.colorbar(im, ax=self.map_ax, fraction=0.046, pad=0.04)
        else:
            self._map_colorbar.update_normal(im)

        # Reset artist references after cla()
        self._map_line_artist  = None
        self._map_endpt_artist = None
        self._map_mid_marker   = None
        self._redraw_map_line()
        self.map_canvas.draw()

    def _redraw_map_line(self):
        for attr in ('_map_line_artist', '_map_endpt_artist', '_map_mid_marker'):
            a = getattr(self, attr, None)
            if a is not None:
                try:
                    a.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        if self.map_p1 is not None and self.map_p2 is None:
            sc = self.map_ax.scatter(
                [self.map_p1[0]], [self.map_p1[1]],
                color='yellow', s=50, zorder=6,
            )
            self._map_endpt_artist = sc

        elif self.map_p1 is not None and self.map_p2 is not None:
            (ln,) = self.map_ax.plot(
                [self.map_p1[0], self.map_p2[0]],
                [self.map_p1[1], self.map_p2[1]],
                color='yellow', linewidth=2.0, linestyle='-', zorder=5,
            )
            self._map_line_artist = ln

            sc = self.map_ax.scatter(
                [self.map_p1[0], self.map_p2[0]],
                [self.map_p1[1], self.map_p2[1]],
                color='yellow', s=50, zorder=6,
            )
            self._map_endpt_artist = sc

            mx = (self.map_p1[0] + self.map_p2[0]) / 2.0
            my = (self.map_p1[1] + self.map_p2[1]) / 2.0
            (mid,) = self.map_ax.plot(
                mx, my, marker='*', color='red',
                markersize=10, linestyle='none', zorder=7,
            )
            self._map_mid_marker = mid

    # ------------------------------------------------------------------
    # Map line drawing
    # ------------------------------------------------------------------

    def _map_on_press(self, event):
        if event.button != 1 or event.inaxes is not self.map_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        xy = (event.xdata, event.ydata)

        # Try to drag an existing endpoint
        if self.map_p1 is not None and self.map_p2 is not None and self._map_draw_phase == 'idle':
            try:
                bb   = self.map_ax.get_window_extent()
                xl, xr = self.map_ax.get_xlim()
                yb, yt = self.map_ax.get_ylim()
                tol_x  = 8.0 * abs(xr - xl) / max(bb.width,  1)
                tol_y  = 8.0 * abs(yt - yb) / max(bb.height, 1)
            except Exception:
                tol_x = tol_y = 8.0
            click = np.array(xy)
            for name, pt in (('map_p1', np.array(self.map_p1)), ('map_p2', np.array(self.map_p2))):
                dist = np.hypot(
                    (click[0] - pt[0]) / tol_x,
                    (click[1] - pt[1]) / tol_y,
                )
                if dist <= 1.0:
                    self._map_drag_target = name
                    return

        if self._map_draw_phase == 'idle':
            self.map_p1 = xy
            self.map_p2 = None
            self._map_draw_phase = 'wait_p2'
            self._redraw_map_line()
            self.map_canvas.draw_idle()
            self.status_var.set("Click again to place the end of the scan line.")

        elif self._map_draw_phase == 'wait_p2':
            self.map_p2 = xy
            self._map_draw_phase = 'idle'
            self._clear_map_prev_line()
            self._redraw_map_line()
            self.map_canvas.draw_idle()
            self._load_middle_pattern()

    def _map_on_motion(self, event):
        # Drag existing endpoint
        tgt = self._map_drag_target
        if tgt and event.xdata is not None and event.ydata is not None:
            setattr(self, tgt, (event.xdata, event.ydata))
            self._redraw_map_line()
            self.map_canvas.draw_idle()
            return

        # Preview second endpoint
        if self._map_draw_phase == 'wait_p2' and self.map_p1 is not None:
            if self._map_prev_line is not None:
                try:
                    self._map_prev_line.remove()
                except Exception:
                    pass
                self._map_prev_line = None
            if event.inaxes is self.map_ax and event.xdata is not None:
                (ln,) = self.map_ax.plot(
                    [self.map_p1[0], event.xdata],
                    [self.map_p1[1], event.ydata],
                    color='yellow', linewidth=1.5, linestyle='--', zorder=5,
                )
                self._map_prev_line = ln
            self.map_canvas.draw_idle()

    def _map_on_release(self, event):
        if self._map_drag_target:
            self._map_drag_target = None
            if self.map_p1 is not None and self.map_p2 is not None:
                self._load_middle_pattern()

    def _clear_map_prev_line(self):
        if self._map_prev_line is not None:
            try:
                self._map_prev_line.remove()
            except Exception:
                pass
            self._map_prev_line = None

    def _reset_map_line(self):
        self.map_p1 = None
        self.map_p2 = None
        self._map_draw_phase   = 'idle'
        self._map_drag_target  = None
        self._clear_map_prev_line()
        self._redraw_map_line()
        self.map_canvas.draw_idle()
        self.mid_exp = None
        self.mid_sim = None
        self.mid_pat_idx = None
        self._reset_rect()

    # ------------------------------------------------------------------
    # Middle-pattern helpers
    # ------------------------------------------------------------------

    def _get_n_patterns(self) -> int:
        try:
            return max(2, int(self.npatterns_var.get()))
        except ValueError:
            return 10

    def _map_line_pattern_indices(self, n: int) -> list:
        """Return n evenly-spaced pattern indices along the drawn map line."""
        if self.ang is None or self.map_p1 is None or self.map_p2 is None:
            return []
        ncols = self.ang.ncols
        nrows = self.ang.nrows
        x1, y1 = self.map_p1
        x2, y2 = self.map_p2
        ts   = np.linspace(0.0, 1.0, n)
        cols = np.clip(np.round(x1 + ts * (x2 - x1)).astype(int), 0, ncols - 1)
        rows = np.clip(np.round(y1 + ts * (y2 - y1)).astype(int), 0, nrows - 1)
        return [int(r) * ncols + int(c) for r, c in zip(rows, cols)]

    def _load_middle_pattern(self):
        if self.ang is None or self.up2 is None:
            self.status_var.set(
                "Line drawn. Load ANG and UP2 files to load the middle pattern."
            )
            return
        n       = self._get_n_patterns()
        indices = self._map_line_pattern_indices(n)
        if not indices:
            return
        mid_idx = indices[n // 2]
        self.status_var.set(f"Loading middle pattern #{mid_idx}…")
        self.win.update_idletasks()

        def _worker(idx=mid_idx):
            try:
                exp = _to_2d(self.up2.read_pattern(idx, process=True))
                sim = _to_2d(request_simulated(idx))
                exp, sim = _match_sizes(exp, sim)
                exp = _normalize(exp)
                sim = _normalize(sim)
                self.win.after(0, lambda: self._on_middle_loaded(idx, exp, sim, None))
            except Exception as exc:
                self.win.after(0, lambda m=str(exc): self._on_middle_loaded(idx, None, None, m))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_middle_loaded(self, idx, exp, sim, error):
        if error:
            messagebox.showerror("Load Error", error, parent=self.win)
            self.status_var.set(f"Failed to load pattern #{idx}: {error}")
            return

        self.mid_exp     = exp
        self.mid_sim     = sim
        self.mid_pat_idx = idx

        for ax, pat, lbl in zip(self.pat_axes, [exp, sim], self._PAT_LABELS):
            ax.cla()
            ax.imshow(pat, cmap='gray', origin='upper')
            ax.set_title(f"{lbl}  (#{idx})", fontsize=10)
            ax.set_axis_off()

        # Reset rectangle overlay references
        self._poly_patches   = [None, None]
        self._cline_artists  = [None, None]
        self._endpt_artists  = [None, None]
        self._prev_lines_pat = [None, None]
        self.rect = RectState()
        self.rect.width = self._current_width()
        self.pat_canvas.draw()

        self.status_var.set(
            f"Middle pattern #{idx} loaded. "
            "Draw a rectangle on either pattern to define the band."
        )

    # ------------------------------------------------------------------
    # Rectangle drawing on pattern pair
    # ------------------------------------------------------------------

    def _current_width(self) -> float:
        try:
            return max(1.0, float(self.width_var.get()))
        except ValueError:
            return 20.0

    def _on_width_change(self):
        self.rect.width = self._current_width()
        self._redraw_rect()
        self.pat_canvas.draw_idle()

    def _reset_rect(self):
        self.rect = RectState()
        self._pat_draw_phase = 'idle'
        self._drag_target    = None
        self._clear_pat_prev_lines()
        for lst in (self._poly_patches, self._cline_artists, self._endpt_artists):
            for i in range(2):
                if lst[i] is not None:
                    try:
                        lst[i].remove()
                    except Exception:
                        pass
                    lst[i] = None
        self._poly_patches   = [None, None]
        self._cline_artists  = [None, None]
        self._endpt_artists  = [None, None]
        self._prev_lines_pat = [None, None]
        self.pat_canvas.draw_idle()

    def _remove_pat_artists(self, i: int):
        for lst in (self._poly_patches, self._cline_artists, self._endpt_artists):
            if lst[i] is not None:
                try:
                    lst[i].remove()
                except Exception:
                    pass
                lst[i] = None

    def _redraw_rect(self):
        for i in range(2):
            self._remove_pat_artists(i)
        self.rect.width = self._current_width()

        for i in range(2):
            ax = self.pat_axes[i]
            if not self.rect.complete:
                if self.rect.p1 is not None:
                    sc = ax.scatter(
                        [self.rect.p1[0]], [self.rect.p1[1]],
                        color='cyan', s=40, zorder=6,
                    )
                    self._endpt_artists[i] = sc
                continue

            corners = self.rect.get_corners()
            if corners is None:
                continue

            poly = MplPolygon(
                corners, closed=True,
                linewidth=1.5, edgecolor='cyan',
                facecolor=(0.0, 1.0, 1.0, 0.08),
                zorder=4,
            )
            ax.add_patch(poly)
            self._poly_patches[i] = poly

            (line,) = ax.plot(
                [self.rect.p1[0], self.rect.p2[0]],
                [self.rect.p1[1], self.rect.p2[1]],
                color='cyan', linewidth=1.0, linestyle='--', zorder=5,
            )
            self._cline_artists[i] = line

            sc = ax.scatter(
                [self.rect.p1[0], self.rect.p2[0]],
                [self.rect.p1[1], self.rect.p2[1]],
                color='cyan', s=40, zorder=6,
            )
            self._endpt_artists[i] = sc

    def _pat_ax_index(self, event) -> int | None:
        for i, ax in enumerate(self.pat_axes):
            if event.inaxes is ax:
                return i
        return None

    def _pat_drag_tolerance(self, ax_idx: int) -> tuple:
        ax = self.pat_axes[ax_idx]
        try:
            bb   = ax.get_window_extent()
            xl, xr = ax.get_xlim()
            yb, yt = ax.get_ylim()
            tol_x  = 8.0 * abs(xr - xl) / max(bb.width,  1)
            tol_y  = 8.0 * abs(yt - yb) / max(bb.height, 1)
            return tol_x, tol_y
        except Exception:
            return 8.0, 8.0

    def _pat_on_press(self, event):
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        ax_idx = self._pat_ax_index(event)
        if ax_idx is None:
            return
        xy = (event.xdata, event.ydata)

        # Try to drag existing endpoint
        if self.rect.complete and self._pat_draw_phase == 'idle':
            tol_x, tol_y = self._pat_drag_tolerance(ax_idx)
            click = np.array(xy)
            for name, pt in (('p1', np.array(self.rect.p1)), ('p2', np.array(self.rect.p2))):
                dist = np.hypot(
                    (click[0] - pt[0]) / tol_x,
                    (click[1] - pt[1]) / tol_y,
                )
                if dist <= 1.0:
                    self._drag_target = name
                    return

        if self._pat_draw_phase == 'idle':
            self.rect.p1 = xy
            self.rect.p2 = None
            self.rect.width = self._current_width()
            self._pat_draw_phase = 'wait_p2'
            self._redraw_rect()
            self.pat_canvas.draw_idle()
            self.status_var.set("Click to place the second endpoint of the centerline.")

        elif self._pat_draw_phase == 'wait_p2':
            self.rect.p2 = xy
            self._pat_draw_phase = 'idle'
            self._clear_pat_prev_lines()
            self._redraw_rect()
            self.pat_canvas.draw_idle()
            self.status_var.set(
                "Rectangle drawn. Drag endpoints to adjust. Press Analyze to run line scan."
            )

    def _pat_on_motion(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Drag
        if self._drag_target:
            setattr(self.rect, self._drag_target, (event.xdata, event.ydata))
            self._clear_pat_prev_lines()
            self._redraw_rect()
            self.pat_canvas.draw_idle()
            return

        # Preview
        if self._pat_draw_phase == 'wait_p2' and self.rect.p1 is not None:
            for i in range(2):
                if self._prev_lines_pat[i] is not None:
                    try:
                        self._prev_lines_pat[i].remove()
                    except Exception:
                        pass
                    self._prev_lines_pat[i] = None
                if event.inaxes is self.pat_axes[i] or True:
                    (ln,) = self.pat_axes[i].plot(
                        [self.rect.p1[0], event.xdata],
                        [self.rect.p1[1], event.ydata],
                        color='yellow', linewidth=1.0, linestyle='-', zorder=5,
                    )
                    self._prev_lines_pat[i] = ln
            self.pat_canvas.draw_idle()

    def _pat_on_release(self, event):
        if self._drag_target:
            self._drag_target = None

    def _clear_pat_prev_lines(self):
        for i in range(2):
            if self._prev_lines_pat[i] is not None:
                try:
                    self._prev_lines_pat[i].remove()
                except Exception:
                    pass
                self._prev_lines_pat[i] = None

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze(self):
        if self.ang is None or self.up2 is None:
            messagebox.showinfo("Not ready", "Load ANG and UP2 files first.", parent=self.win)
            return
        if self.map_p1 is None or self.map_p2 is None:
            messagebox.showinfo("Not ready", "Draw a scan line on the EBSD map first.", parent=self.win)
            return
        if not self.rect.complete:
            messagebox.showinfo("Not ready", "Draw a rectangle on the middle pattern first.", parent=self.win)
            return

        n       = self._get_n_patterns()
        indices = self._map_line_pattern_indices(n)

        self.status_var.set(f"Analyzing {n} patterns… please wait.")
        self.win.update_idletasks()

        rect_p1    = self.rect.p1
        rect_p2    = self.rect.p2
        rect_width = self.rect.width
        mid_shape  = self.mid_exp.shape if self.mid_exp is not None else None
        up2        = self.up2

        def _worker():
            from skimage.transform import resize as sk_resize
            profiles     = []
            sim_profiles = []
            labels       = []
            errors       = []
            for idx in indices:
                try:
                    exp = _to_2d(up2.read_pattern(idx, process=True))
                    exp = _normalize(exp)
                    if mid_shape is not None and exp.shape != mid_shape:
                        exp = sk_resize(exp, mid_shape, anti_aliasing=True,
                                        preserve_range=True).astype(np.float32)
                    band    = extract_band(exp, rect_p1, rect_p2, rect_width)
                    profile = band.sum(axis=0)
                    profiles.append(profile)
                    labels.append(f"#{idx}")

                    try:
                        sim = _to_2d(request_simulated(idx))
                        sim = _normalize(sim)
                        if mid_shape is not None and sim.shape != mid_shape:
                            sim = sk_resize(sim, mid_shape, anti_aliasing=True,
                                            preserve_range=True).astype(np.float32)
                        sim_band    = extract_band(sim, rect_p1, rect_p2, rect_width)
                        sim_profile = sim_band.sum(axis=0)
                        sim_profiles.append(sim_profile)
                    except Exception as sim_exc:
                        sim_profiles.append(None)
                        errors.append(f"Sim #{idx}: {sim_exc}")

                except Exception as exc:
                    errors.append(f"Pattern #{idx}: {exc}")

            self.win.after(0, lambda: self._show_profiles(
                profiles, sim_profiles, labels, errors, rect_width
            ))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_profiles(self, profiles, sim_profiles, labels, errors, width):
        if errors:
            print("\nMode 3 analysis warnings:")
            for e in errors:
                print(f"  {e}")

        if not profiles:
            messagebox.showinfo("No data", "No profiles could be extracted.", parent=self.win)
            return

        win = tk.Toplevel(self.win)
        win.title("Line-scan band profiles")
        win.geometry("800x480")

        pfig = Figure(figsize=(8, 4.5), constrained_layout=True)
        pax  = pfig.add_subplot(111)

        n      = len(profiles)
        cmap_v = matplotlib.colormaps.get_cmap('turbo')
        for k, (prof, lbl) in enumerate(zip(profiles, labels)):
            xc    = np.linspace(-width / 2.0, width / 2.0, prof.size)
            color = cmap_v(k / max(n - 1, 1))
            pax.plot(xc, prof, label=f"{lbl} exp", color=color, linewidth=1.2)
            if k < len(sim_profiles) and sim_profiles[k] is not None:
                sim_xc = np.linspace(-width / 2.0, width / 2.0, sim_profiles[k].size)
                pax.plot(sim_xc, sim_profiles[k],
                         linestyle='--', color=color, linewidth=1.0,
                         label=f"{lbl} sim")

        pax.set_xlabel("Distance from centerline (px)")
        pax.set_ylabel("Summed intensity (along centerline)")
        pax.set_title(f"Line-scan profiles  ({n} patterns along drawn line)")
        pax.axvline(0, color='gray', linewidth=0.8, linestyle='--', zorder=0)
        pax.legend(fontsize=7, ncol=max(1, n // 3))
        pax.grid(True, alpha=0.3)

        pcanvas = FigureCanvasTkAgg(pfig, master=win)
        pcanvas.draw()
        pcanvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var.set(f"Line scan complete — {n} profiles plotted.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class BandAnalysisApp:
    _LABELS = ["Experimental 1", "Simulated 1", "Experimental 2", "Simulated 2"]
    _COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Band Analysis — 4D-EBSD")
        self.root.minsize(900, 700)

        self.patterns: list = [None, None, None, None]   # exp1, sim1, exp2, sim2
        self.mode = tk.IntVar(value=1)
        self._mode3_win: Mode3Window | None = None

        # Rectangle states
        self.rect_shared = RectState()   # Mode 1
        self.rect1 = RectState()         # Mode 2 – axes 0 & 1
        self.rect2 = RectState()         # Mode 2 – axes 2 & 3

        # Drawing / drag state
        self._draw_phase: str = 'idle'          # 'idle' | 'wait_p2'
        self._active_rect: RectState | None = None
        self._drag_target: str | None = None    # 'p1' | 'p2'
        self._drag_rect: RectState | None = None

        # Per-axis overlay artists
        self._poly_patches:  list = [None] * 4
        self._cline_artists: list = [None] * 4
        self._endpt_artists: list = [None] * 4
        self._prev_lines:    list = [None] * 4  # motion-preview Line2D

        self._build_ui()
        self._connect_events()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Control bar -------------------------------------------
        ctrl = ttk.Frame(self.root, padding=4)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="Pattern 1:").pack(side=tk.LEFT)
        self.pat1_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.pat1_var, width=8).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl, text="Pattern 2:").pack(side=tk.LEFT, padx=(8, 0))
        self.pat2_var = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.pat2_var, width=8).pack(side=tk.LEFT, padx=2)

        ttk.Button(ctrl, text="Load", command=self._load_patterns).pack(side=tk.LEFT, padx=6)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(ctrl, text="Mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(ctrl, text="1 – Shared", variable=self.mode, value=1,
                        command=self._on_mode_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(ctrl, text="2 – Independent", variable=self.mode, value=2,
                        command=self._on_mode_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(ctrl, text="3 – Line Scan", variable=self.mode, value=3,
                        command=self._on_mode_change).pack(side=tk.LEFT, padx=2)

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(ctrl, text="Width (px):").pack(side=tk.LEFT)
        self.width_var = tk.StringVar(value="20")
        ttk.Entry(ctrl, textvariable=self.width_var, width=6).pack(side=tk.LEFT, padx=2)
        self.width_var.trace_add('write', lambda *_: self._on_width_change())

        ttk.Separator(ctrl, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Button(ctrl, text="Reset", command=self._reset_rects).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Analyze", command=self._analyze).pack(side=tk.LEFT, padx=4)

        # ---- Status bar --------------------------------------------
        self.status_var = tk.StringVar(value="Enter pattern indices and press Load.")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

        # ---- 2×2 figure --------------------------------------------
        self.fig = Figure(figsize=(10, 8), constrained_layout=True)
        self.axes = [self.fig.add_subplot(2, 2, i + 1) for i in range(4)]
        for ax, lbl in zip(self.axes, self._LABELS):
            ax.set_title(lbl, fontsize=10)
            ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Event binding
    # ------------------------------------------------------------------

    def _connect_events(self):
        self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

    # ------------------------------------------------------------------
    # Pattern loading
    # ------------------------------------------------------------------

    def _load_patterns(self):
        try:
            idx1 = int(self.pat1_var.get())
            idx2 = int(self.pat2_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Pattern indices must be integers.")
            return

        self.status_var.set("Loading patterns… please wait.")
        self.root.update()

        def _worker():
            results = [None, None, None, None]
            errors = []
            try:
                e1 = _to_2d(load_experimental(idx1))
                s1 = _to_2d(request_simulated(idx1))
                e1, s1 = _match_sizes(e1, s1)
                results[0] = _normalize(e1)
                results[1] = _normalize(s1)
            except Exception as exc:
                errors.append(f"Pair 1: {exc}")
            try:
                e2 = _to_2d(load_experimental(idx2))
                s2 = _to_2d(request_simulated(idx2))
                e2, s2 = _match_sizes(e2, s2)
                results[2] = _normalize(e2)
                results[3] = _normalize(s2)
            except Exception as exc:
                errors.append(f"Pair 2: {exc}")

            self.root.after(
                0, lambda: self._on_patterns_loaded(results, errors, idx1, idx2)
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_patterns_loaded(self, results, errors, idx1, idx2):
        self.patterns = results
        self._reset_rects()    # clears state and redraws
        self._redraw_patterns()
        if errors:
            msg = "Loaded with errors:\n" + "\n".join(errors)
            self.status_var.set(msg)
            messagebox.showwarning("Load warnings", msg)
        else:
            self.status_var.set(
                f"Loaded patterns {idx1} & {idx2}.  "
                "Click on a pattern to start drawing the centerline."
            )

    def _redraw_patterns(self):
        for i, (ax, lbl) in enumerate(zip(self.axes, self._LABELS)):
            ax.cla()
            ax.set_title(lbl, fontsize=10)
            ax.set_axis_off()
            if self.patterns[i] is not None:
                ax.imshow(self.patterns[i], cmap='gray', origin='upper')

        # Reset artist tracking after cla()
        self._poly_patches  = [None] * 4
        self._cline_artists = [None] * 4
        self._endpt_artists = [None] * 4
        self._prev_lines    = [None] * 4

        self._redraw_rects()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Rectangle state helpers
    # ------------------------------------------------------------------

    def _get_rect_for_ax(self, ax_idx: int) -> RectState:
        if self.mode.get() == 1:
            return self.rect_shared
        return self.rect1 if ax_idx in (0, 1) else self.rect2

    def _axes_for_rect(self, rect: RectState) -> list:
        if self.mode.get() == 1:
            return [0, 1, 2, 3]
        return [0, 1] if rect is self.rect1 else [2, 3]

    def _current_width(self) -> float:
        try:
            return max(1.0, float(self.width_var.get()))
        except ValueError:
            return 20.0

    def _can_initiate_draw_on(self, ax_idx: int) -> bool:
        """Mode 1: any subplot. Mode 2: only experimental axes (0, 2)."""
        return True if self.mode.get() == 1 else ax_idx in (0, 2)

    def _reset_rects(self):
        self.rect_shared = RectState()
        self.rect1 = RectState()
        self.rect2 = RectState()
        self._draw_phase  = 'idle'
        self._active_rect = None
        self._drag_target = None
        self._drag_rect   = None
        self._clear_preview_lines()
        self._redraw_rects()
        self.canvas.draw_idle()

    def _redraw_rects(self):
        """Remove all overlay artists and redraw from current rect state."""
        for i in range(4):
            self._remove_artists_for_ax(i)
        rects = (
            [self.rect_shared] if self.mode.get() == 1
            else [self.rect1, self.rect2]
        )
        for rect in rects:
            rect.width = self._current_width()
            self._draw_rect_on_axes(rect)

    def _remove_artists_for_ax(self, i: int):
        for lst in (self._poly_patches, self._cline_artists, self._endpt_artists):
            if lst[i] is not None:
                try:
                    lst[i].remove()
                except Exception:
                    pass
                lst[i] = None

    def _draw_rect_on_axes(self, rect: RectState):
        """Draw (or refresh) the rectangle overlay for *rect* on its axes."""
        for i in self._axes_for_rect(rect):
            self._remove_artists_for_ax(i)
            ax = self.axes[i]

            if not rect.complete:
                # Draw the p1 marker if the first point has been placed
                if rect.p1 is not None:
                    sc = ax.scatter(
                        [rect.p1[0]], [rect.p1[1]],
                        color='cyan', s=40, zorder=6,
                    )
                    self._endpt_artists[i] = sc
                continue

            corners = rect.get_corners()
            if corners is None:
                continue

            poly = MplPolygon(
                corners, closed=True,
                linewidth=1.5, edgecolor='cyan',
                facecolor=(0.0, 1.0, 1.0, 0.08),
                zorder=4,
            )
            ax.add_patch(poly)
            self._poly_patches[i] = poly

            (line,) = ax.plot(
                [rect.p1[0], rect.p2[0]], [rect.p1[1], rect.p2[1]],
                color='cyan', linewidth=1.0, linestyle='--', zorder=5,
            )
            self._cline_artists[i] = line

            sc = ax.scatter(
                [rect.p1[0], rect.p2[0]], [rect.p1[1], rect.p2[1]],
                color='cyan', s=40, zorder=6,
            )
            self._endpt_artists[i] = sc

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _ax_index(self, event) -> int | None:
        for i, ax in enumerate(self.axes):
            if event.inaxes is ax:
                return i
        return None

    def _endpoint_drag_tolerance(self, ax_idx: int) -> tuple:
        """Return (tol_x, tol_y) in data units corresponding to ~8 display px."""
        ax = self.axes[ax_idx]
        try:
            bb = ax.get_window_extent()
            xl, xr = ax.get_xlim()
            yb, yt = ax.get_ylim()
            tol_x = 8.0 * abs(xr - xl) / max(bb.width,  1)
            tol_y = 8.0 * abs(yt - yb) / max(bb.height, 1)
            return tol_x, tol_y
        except Exception:
            return 8.0, 8.0

    def _on_press(self, event):
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        ax_idx = self._ax_index(event)
        if ax_idx is None:
            return

        xy   = (event.xdata, event.ydata)
        rect = self._get_rect_for_ax(ax_idx)

        # ---- Try to drag an existing endpoint -----------------------
        if rect.complete and self._draw_phase == 'idle':
            tol_x, tol_y = self._endpoint_drag_tolerance(ax_idx)
            click = np.array(xy)
            for name, pt in (('p1', np.array(rect.p1)), ('p2', np.array(rect.p2))):
                dist = np.hypot(
                    (click[0] - pt[0]) / tol_x,
                    (click[1] - pt[1]) / tol_y,
                )
                if dist <= 1.0:
                    self._drag_target = name
                    self._drag_rect   = rect
                    return

        # ---- Drawing ------------------------------------------------
        if not self._can_initiate_draw_on(ax_idx):
            return

        if self._draw_phase == 'idle':
            # Abandon any previously incomplete drawing on another rect
            if self._active_rect is not None and self._active_rect is not rect:
                self._active_rect.p1 = None
                self._draw_rect_on_axes(self._active_rect)

            rect.p1 = xy
            rect.p2 = None
            rect.width = self._current_width()
            self._draw_phase  = 'wait_p2'
            self._active_rect = rect
            self._draw_rect_on_axes(rect)
            self.canvas.draw_idle()
            self.status_var.set("Click to place the second endpoint of the centerline.")

        elif self._draw_phase == 'wait_p2':
            if self._active_rect is rect:
                # Place p2 – rectangle is now complete
                rect.p2       = xy
                self._draw_phase  = 'idle'
                self._active_rect = None
                self._clear_preview_lines()
                self._draw_rect_on_axes(rect)
                self.canvas.draw_idle()
                self.status_var.set(
                    "Rectangle drawn. Drag the cyan endpoints to adjust the line. "
                    "Update the Width field to change the band width."
                )
            else:
                # User clicked on a different rect's axis – abandon old, start new
                if self._active_rect is not None:
                    self._active_rect.p1 = None
                    self._draw_rect_on_axes(self._active_rect)
                rect.p1 = xy
                rect.p2 = None
                rect.width = self._current_width()
                self._active_rect = rect
                self._clear_preview_lines()
                self._draw_rect_on_axes(rect)
                self.canvas.draw_idle()
                self.status_var.set("Click to place the second endpoint of the centerline.")

    def _on_motion(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # ---- Drag mode ----------------------------------------------
        if self._drag_target and self._drag_rect:
            setattr(self._drag_rect, self._drag_target, (event.xdata, event.ydata))
            self._clear_preview_lines()
            self._draw_rect_on_axes(self._drag_rect)
            self.canvas.draw_idle()
            return

        # ---- Preview line while placing second endpoint -------------
        if self._draw_phase == 'wait_p2' and self._active_rect is not None:
            rect = self._active_rect
            for i in self._axes_for_rect(rect):
                if self._prev_lines[i] is not None:
                    try:
                        self._prev_lines[i].remove()
                    except Exception:
                        pass
                    self._prev_lines[i] = None

                if rect.p1 is not None:
                    (ln,) = self.axes[i].plot(
                        [rect.p1[0], event.xdata],
                        [rect.p1[1], event.ydata],
                        color='yellow', linewidth=1.0, linestyle='-', zorder=5,
                    )
                    self._prev_lines[i] = ln

            self.canvas.draw_idle()

    def _on_release(self, event):
        if self._drag_target:
            self._drag_target = None
            self._drag_rect   = None

    def _clear_preview_lines(self):
        for i in range(4):
            if self._prev_lines[i] is not None:
                try:
                    self._prev_lines[i].remove()
                except Exception:
                    pass
                self._prev_lines[i] = None

    # ------------------------------------------------------------------
    # Width / mode callbacks
    # ------------------------------------------------------------------

    def _on_width_change(self):
        w = self._current_width()
        for rect in (self.rect_shared, self.rect1, self.rect2):
            rect.width = w
        self._redraw_rects()
        self.canvas.draw_idle()

    def _on_mode_change(self):
        if self.mode.get() == 3:
            if self._mode3_win is None or not self._mode3_win.win.winfo_exists():
                self._mode3_win = Mode3Window(self.root)
            else:
                self._mode3_win.win.lift()
        else:
            self._reset_rects()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyze(self):
        mode = self.mode.get()
        if mode == 1:
            rects = [self.rect_shared] * 4
        else:
            rects = [self.rect1, self.rect1, self.rect2, self.rect2]

        if not any(r.complete for r in set(rects)):
            messagebox.showinfo("No rectangle", "Draw a rectangle first.")
            return

        profiles   = []
        x_axes     = []
        valid_lbls = []
        valid_cols = []

        for i in range(4):
            rect = rects[i]
            pat  = self.patterns[i]
            if pat is None or not rect.complete:
                continue

            band = extract_band(pat, rect.p1, rect.p2, rect.width)

            # Print the Experimental 1 band array to console for inspection
            if i == 0:
                print("\n=== Extracted band — Experimental 1 ===")
                print(f"Shape: {band.shape}  "
                      f"(axis-0 = along centerline [{band.shape[0]} px], "
                      f"axis-1 = across centerline [{band.shape[1]} px])")
                print(band)
                print("========================================\n")

            # Sum along the centerline direction (axis-0) → profile of length width_px
            profile = band.sum(axis=0)
            x_ax    = np.linspace(-rect.width / 2.0, rect.width / 2.0, profile.size)

            profiles.append(profile)
            x_axes.append(x_ax)
            valid_lbls.append(self._LABELS[i])
            valid_cols.append(self._COLORS[i])

        if not profiles:
            messagebox.showinfo("No data", "No patterns loaded or no complete rectangles.")
            return

        # ---- Profile window ----------------------------------------
        win = tk.Toplevel(self.root)
        win.title("Cross-band intensity profiles")
        win.geometry("700x420")

        pfig = Figure(figsize=(7, 4), constrained_layout=True)
        pax  = pfig.add_subplot(111)

        for prof, xc, lbl, col in zip(profiles, x_axes, valid_lbls, valid_cols):
            pax.plot(xc, prof, label=lbl, color=col)

        pax.set_xlabel("Distance from centerline (px)")
        pax.set_ylabel("Summed intensity (along centerline)")
        pax.set_title("Cross-band intensity profiles")
        pax.axvline(0, color='gray', linewidth=0.8, linestyle='--', zorder=0)
        pax.legend()
        pax.grid(True, alpha=0.3)

        pcanvas = FigureCanvasTkAgg(pfig, master=win)
        pcanvas.draw()
        pcanvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var.set(
            "Analysis complete — see profile window and console for band array."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _check_admin()
    root = tk.Tk()
    app = BandAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
