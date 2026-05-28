"""
Interactive EBSD Viewer
-----------------------
Displays an EBSD map from an .ang file and shows Kikuchi patterns
from a .up2 file when the user clicks on the map.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ANG import Ang
from UP2 import UP2

# Column index mapping (matches ANG column_headers order)
CHANNEL_NAMES = [
    "phi1 (Euler)",
    "PHI (Euler)",
    "phi2 (Euler)",
    "X Position",
    "Y Position",
    "IQ (Image Quality)",
    "CI (Confidence Index)",
    "Phase Index",
    "SEM Signal",
    "Fit",
]
CHANNEL_CMAPS = [
    "hsv", "hsv", "hsv",  # Euler angles
    "viridis", "viridis",  # positions
    "gray",                # IQ
    "viridis",             # CI
    "tab10",               # Phase
    "gray",                # SEM
    "plasma",              # Fit
]
DEFAULT_CHANNEL = 6  # CI


class EBSDViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("4D-EBSD Kikuchi Pattern Viewer")
        self.root.minsize(1100, 600)

        self.ang: Ang | None = None
        self.up2: UP2 | None = None
        self.data: np.ndarray | None = None   # shape (nrows, ncols, 10)
        self.current_channel = tk.IntVar(value=DEFAULT_CHANNEL)
        self.last_click_info = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ---- Top toolbar ----
        toolbar = ttk.Frame(self.root, padding=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load ANG File", command=self._load_ang).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Load UP2 File", command=self._load_up2).pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="  Channel:").pack(side=tk.LEFT)
        self.channel_combo = ttk.Combobox(
            toolbar,
            values=CHANNEL_NAMES,
            width=22,
            state="disabled",
        )
        self.channel_combo.current(DEFAULT_CHANNEL)
        self.channel_combo.pack(side=tk.LEFT, padx=4)
        self.channel_combo.bind("<<ComboboxSelected>>", self._on_channel_change)

        ttk.Button(toolbar, text="Save Pattern Info", command=self._save_info).pack(side=tk.RIGHT, padx=4)

        # ---- Status bar ----
        self.status_var = tk.StringVar(value="Load an ANG file to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ---- Main content area ----
        content = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: map panel
        left_frame = ttk.LabelFrame(content, text="EBSD Map", padding=2)
        content.add(left_frame, weight=1)

        self.map_fig = Figure(figsize=(5, 5), tight_layout=True)
        self.map_ax = self.map_fig.add_subplot(111)
        self.map_ax.set_axis_off()
        self.map_canvas = FigureCanvasTkAgg(self.map_fig, master=left_frame)
        self.map_canvas.draw()
        self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        map_toolbar = NavigationToolbar2Tk(self.map_canvas, left_frame)
        map_toolbar.update()
        self.map_canvas.mpl_connect("button_press_event", self._on_map_click)

        # Right: pattern + info panel
        right_frame = ttk.Frame(content, padding=2)
        content.add(right_frame, weight=1)

        pattern_frame = ttk.LabelFrame(right_frame, text="Kikuchi Pattern", padding=2)
        pattern_frame.pack(fill=tk.BOTH, expand=True)

        self.pat_fig = Figure(figsize=(4, 4), tight_layout=True)
        self.pat_ax = self.pat_fig.add_subplot(111)
        self.pat_ax.set_axis_off()
        self.pat_canvas = FigureCanvasTkAgg(self.pat_fig, master=pattern_frame)
        self.pat_canvas.draw()
        self.pat_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Info box
        info_frame = ttk.LabelFrame(right_frame, text="Point Information", padding=6)
        info_frame.pack(fill=tk.X, pady=(4, 0))

        labels = ["Pattern #:", "Row:", "Column:", "X (µm):", "Y (µm):", "CI:", "IQ:", "Phase:"]
        self.info_vars = {lbl: tk.StringVar(value="—") for lbl in labels}
        for i, lbl in enumerate(labels):
            ttk.Label(info_frame, text=lbl, width=12, anchor=tk.E).grid(row=i // 2, column=(i % 2) * 2, sticky=tk.E, padx=(4, 2))
            ttk.Label(info_frame, textvariable=self.info_vars[lbl], anchor=tk.W).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky=tk.W)

        # Crosshair marker reference
        self._marker = None

    # ------------------------------------------------------------------
    # File Loading
    # ------------------------------------------------------------------

    def _load_ang(self):
        path = filedialog.askopenfilename(
            title="Open ANG File",
            filetypes=[("ANG files", "*.ang"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.status_var.set("Parsing ANG file…")
            self.root.update_idletasks()
            self.ang = Ang(path)
            self.data = self.ang.generate_np_array()
            self.channel_combo.config(state="readonly")
            self._update_map()
            self.status_var.set(
                f"Loaded: {os.path.basename(path)}  —  "
                f"{self.ang.nrows} rows × {self.ang.ncols} cols  |  "
                f"Step: {self.ang.xstep} × {self.ang.ystep} µm"
            )
        except Exception as exc:
            messagebox.showerror("ANG Load Error", str(exc))
            self.status_var.set("Failed to load ANG file.")

    def _load_up2(self):
        path = filedialog.askopenfilename(
            title="Open UP2 File",
            filetypes=[("UP2 files", "*.up2"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.up2 = UP2(path)
            self.status_var.set(
                self.status_var.get().split("|")[0].strip()
                + f"  |  UP2: {os.path.basename(path)} ({self.up2.nPatterns} patterns, {self.up2.patshape})"
            )
        except Exception as exc:
            messagebox.showerror("UP2 Load Error", str(exc))

    # ------------------------------------------------------------------
    # Map Display
    # ------------------------------------------------------------------

    def _update_map(self):
        if self.data is None:
            return

        ch = self.channel_combo.current()
        channel_data = self.data[:, :, ch]
        cmap = CHANNEL_CMAPS[ch]

        self.map_ax.cla()
        im = self.map_ax.imshow(
            channel_data,
            cmap=cmap,
            origin="upper",
            interpolation="nearest",
        )
        self.map_ax.set_title(CHANNEL_NAMES[ch], fontsize=10)
        self.map_ax.set_xlabel("Column")
        self.map_ax.set_ylabel("Row")

        # Colorbar: replace if already present
        if hasattr(self, "_colorbar") and self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
        self._colorbar = self.map_fig.colorbar(im, ax=self.map_ax, fraction=0.046, pad=0.04)

        self._marker = None
        self.map_canvas.draw()

    def _on_channel_change(self, _event=None):
        self._update_map()

    # ------------------------------------------------------------------
    # Click Handling
    # ------------------------------------------------------------------

    def _on_map_click(self, event):
        if event.inaxes != self.map_ax:
            return
        if self.data is None:
            return
        if event.button != 1:  # left click only
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))

        nrows, ncols = self.data.shape[:2]
        col = max(0, min(col, ncols - 1))
        row = max(0, min(row, nrows - 1))

        pattern_number = row * ncols + col
        point = self.data[row, col]

        # Update info panel
        self.info_vars["Pattern #:"].set(str(pattern_number))
        self.info_vars["Row:"].set(str(row))
        self.info_vars["Column:"].set(str(col))
        self.info_vars["X (µm):"].set(f"{point[3]:.4f}")
        self.info_vars["Y (µm):"].set(f"{point[4]:.4f}")
        self.info_vars["CI:"].set(f"{point[6]:.4f}")
        self.info_vars["IQ:"].set(f"{point[5]:.2f}")
        self.info_vars["Phase:"].set(str(int(point[7])))

        self.last_click_info = {
            "pattern_number": pattern_number,
            "row": row,
            "col": col,
            "x_um": float(point[3]),
            "y_um": float(point[4]),
            "CI": float(point[6]),
            "IQ": float(point[5]),
            "phase": int(point[7]),
            "phi1": float(point[0]),
            "PHI": float(point[1]),
            "phi2": float(point[2]),
            "fit": float(point[9]),
        }

        # Draw crosshair marker on map
        if self._marker is not None:
            try:
                self._marker.remove()
            except Exception:
                pass
        (self._marker,) = self.map_ax.plot(
            col, row, marker="+", color="red", markersize=14, markeredgewidth=1.5, linestyle="none"
        )
        self.map_canvas.draw()

        # Display Kikuchi pattern
        self._show_pattern(pattern_number, row, col)

    def _show_pattern(self, pattern_number: int, row: int, col: int):
        self.pat_ax.cla()

        if self.up2 is None:
            self.pat_ax.text(
                0.5, 0.5,
                "Load a UP2 file\nto view patterns",
                ha="center", va="center",
                transform=self.pat_ax.transAxes,
                fontsize=11, color="gray",
            )
            self.pat_ax.set_axis_off()
            self.pat_canvas.draw()
            return

        if pattern_number >= self.up2.nPatterns:
            self.pat_ax.text(
                0.5, 0.5,
                f"Pattern #{pattern_number}\nout of range\n(UP2 has {self.up2.nPatterns} patterns)",
                ha="center", va="center",
                transform=self.pat_ax.transAxes,
                fontsize=10, color="red",
            )
            self.pat_ax.set_axis_off()
            self.pat_canvas.draw()
            return

        try:
            pat = self.up2.read_pattern(pattern_number, process=True)
        except Exception as exc:
            messagebox.showerror("Pattern Read Error", str(exc))
            return

        self.pat_ax.imshow(pat, cmap="gray", origin="upper", interpolation="nearest")
        self.pat_ax.set_title(
            f"Pattern #{pattern_number}  (row={row}, col={col})",
            fontsize=9,
        )
        self.pat_ax.set_axis_off()
        self.pat_canvas.draw()

    # ------------------------------------------------------------------
    # Save Info
    # ------------------------------------------------------------------

    def _save_info(self):
        if not self.last_click_info:
            messagebox.showinfo("Nothing to save", "Click on the map first to select a point.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Pattern Info",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        info = self.last_click_info
        lines = [
            "EBSD Point Information",
            "======================",
            f"Pattern #   : {info['pattern_number']}",
            f"Row         : {info['row']}",
            f"Column      : {info['col']}",
            f"X position  : {info['x_um']:.4f} µm",
            f"Y position  : {info['y_um']:.4f} µm",
            f"CI          : {info['CI']:.4f}",
            f"IQ          : {info['IQ']:.4f}",
            f"Phase       : {info['phase']}",
            f"phi1        : {info['phi1']:.6f} rad",
            f"PHI         : {info['PHI']:.6f} rad",
            f"phi2        : {info['phi2']:.6f} rad",
            f"Fit         : {info['fit']:.4f}",
        ]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

        messagebox.showinfo("Saved", f"Pattern info saved to:\n{path}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    root = tk.Tk()
    app = EBSDViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
