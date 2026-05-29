"""
main.py
-------
4D-EBSD Interactive Viewer with Simulated Pattern Comparison.

Opens the EBSD map viewer (viewer.py).  When the user clicks a point:
  - The experimental Kikuchi pattern is shown in the main window as usual.
  - A new pop-up window opens showing:
        Left:   Experimental pattern   (from .up2)
        Centre: Simulated pattern      (from OIM API via 4DEBSD_interactor.py)
        Right:  Absolute % difference  (pixel-wise)

The OIM simulation is requested in a background thread so the main UI
stays responsive while it runs.

Run from an Administrator terminal with the OIMapi conda env active:
    python main.py
"""

import os
import sys
import threading

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from viewer import EBSDViewer
from sim_compare import (
    request_simulated,
    _normalize,
    _match_sizes,
    _to_2d,
    _check_admin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pixel-wise percent difference (exp - sim) / ((exp + sim) / 2) * 100."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return (a - b) / ((a + b) / 2.0) * 100.0


# ---------------------------------------------------------------------------
# Extended viewer
# ---------------------------------------------------------------------------

class Main4DEBSDViewer(EBSDViewer):
    """
    Inherits all map/pattern behaviour from EBSDViewer and adds a per-click
    comparison window that shows the experimental pattern, the OIM-simulated
    pattern, and their pixel-wise percent difference.
    """

    def _show_pattern(self, pattern_number: int, row: int, col: int):
        # Draw the experimental pattern in the main window (parent behaviour).
        super()._show_pattern(pattern_number, row, col)

        # Open a comparison window immediately with a "loading" placeholder,
        # then fill it once the OIM simulation finishes in a background thread.
        win = self._open_comparison_placeholder(pattern_number, row, col)

        def _worker(pn=pattern_number, r=row, c=col, w=win):
            try:
                sim = request_simulated(pn)
                self.root.after(
                    0,
                    lambda s=sim: self._populate_comparison(w, pn, r, c, s, error=None),
                )
            except Exception as exc:
                msg = str(exc)
                self.root.after(
                    0,
                    lambda m=msg: self._populate_comparison(w, pn, r, c, sim=None, error=m),
                )

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Comparison window
    # ------------------------------------------------------------------

    def _open_comparison_placeholder(self, pattern_number: int, row: int, col: int):
        """Create the comparison Toplevel with 'loading' placeholders."""
        win = tk.Toplevel(self.root)
        win.title(f"Pattern #{pattern_number}  (row={row}, col={col})  —  Comparison")
        win.geometry("950x370")

        fig = Figure(figsize=(9.5, 3.5), constrained_layout=True)
        for i, label in enumerate(["Experimental", "Simulated", "% Difference"]):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.set_title(label, fontsize=10)
            ax.text(
                0.5, 0.5,
                "Requesting simulation…" if i == 1 else "Loading…",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=10, color="gray",
            )
            ax.set_axis_off()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Status label at the bottom
        status = tk.StringVar(value="Waiting for OIM simulation…")
        ttk.Label(win, textvariable=status, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X
        )

        win._fig    = fig
        win._canvas = canvas
        win._status = status
        return win

    def _populate_comparison(
        self,
        win,
        pattern_number: int,
        row: int,
        col: int,
        sim,
        error: str | None,
    ):
        """Called on the main thread once simulation is ready (or has failed)."""
        if not win.winfo_exists():
            return

        fig = win._fig
        fig.clear()

        # ---- Error path ------------------------------------------------
        if error is not None:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f"Simulation failed:\n\n{error}",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=9, color="red",
            )
            ax.set_axis_off()
            win._status.set("Simulation failed — see error above.")
            win._canvas.draw()
            return

        if self.up2 is None:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No UP2 file loaded.\nLoad a UP2 file in the main window first.",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=10, color="gray",
            )
            ax.set_axis_off()
            win._canvas.draw()
            return

        # ---- Load experimental pattern --------------------------------
        try:
            exp = _to_2d(self.up2.read_pattern(pattern_number, process=True))
        except Exception as exc:
            ax = fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f"Could not load experimental pattern:\n{exc}",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=9, color="red",
            )
            ax.set_axis_off()
            win._canvas.draw()
            return

        # ---- Align sizes, normalise, compute difference ---------------
        sim = _to_2d(sim)
        exp_orig_shape = exp.shape
        sim_orig_shape = sim.shape
        exp, sim = _match_sizes(exp, sim)
        exp = _normalize(exp)
        sim = _normalize(sim)
        diff = _pct_diff(exp, sim)
        mean_diff = float(diff.mean())

        # ---- Plot ------------------------------------------------------
        ax_exp  = fig.add_subplot(1, 3, 1)
        ax_sim  = fig.add_subplot(1, 3, 2)
        ax_diff = fig.add_subplot(1, 3, 3)

        ax_exp.imshow(exp, cmap="gray", origin="upper")
        ax_exp.set_title(f"Experimental  (#{pattern_number})", fontsize=9)
        ax_exp.set_axis_off()
        if exp.shape != exp_orig_shape:
            ax_exp.text(0.5, -0.04,
                        f"Original: {exp_orig_shape[1]}x{exp_orig_shape[0]}  ->  {exp.shape[1]}x{exp.shape[0]}",
                        ha="center", va="top", transform=ax_exp.transAxes, fontsize=7, color="gray")

        ax_sim.imshow(sim, cmap="gray", origin="upper")
        ax_sim.set_title(f"Simulated  (#{pattern_number})", fontsize=9)
        ax_sim.set_axis_off()
        if sim.shape != sim_orig_shape:
            ax_sim.text(0.5, -0.04,
                        f"Original: {sim_orig_shape[1]}x{sim_orig_shape[0]}  ->  {sim.shape[1]}x{sim.shape[0]}",
                        ha="center", va="top", transform=ax_sim.transAxes, fontsize=7, color="gray")

        im = ax_diff.imshow(diff, cmap="RdBu_r", origin="upper", vmin=-200, vmax=200)
        ax_diff.set_title(f"% Difference  (mean = {mean_diff:.1f} %)", fontsize=9)
        ax_diff.set_axis_off()
        fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04, label="%")

        win._status.set(
            f"Pattern #{pattern_number}  |  row={row}, col={col}  |  "
            f"Mean |exp - sim| = {mean_diff:.1f} %"
        )
        win._canvas.draw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _check_admin()
    root = tk.Tk()
    app = Main4DEBSDViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
