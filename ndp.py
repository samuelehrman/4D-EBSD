"""
ndp.py
------
Compute and display a Normalized Dot Product (NDP) map comparing
experimental Kikuchi patterns (from a .up2 file) with OIM-API-simulated
patterns across an entire EBSD scan.

    NDP(a, b) = dot(a_flat, b_flat) / (|a_flat| * |b_flat|)

Steps:
  1. Select a .ang scan file via file dialog.
  2. Select a .up2 pattern file via file dialog.
  3. Click "Run NDP Map" – a background process simulates every pattern
     and computes the NDP value.
  4. A progress bar shows per-pattern progress.
  5. The finished NDP map is displayed in an embedded matplotlib figure.

Requirements:
  * Must be run from an Administrator terminal (OIM API requirement).
  * C:\\Users\\User\\Desktop\\OIMpy\\4DEBSD_interactor.py must be up-to-date
    (re-copy from script_to_paste.py if needed – it now contains the
    batch_ndp mode).
"""

import sys
import os
import json
import subprocess
import threading
import ctypes

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------------------------------------------------------------------------
# Paths – edit to match your environment
# ---------------------------------------------------------------------------
OIM_SCRIPT_DIR = r"C:\Users\User\Desktop\OIMpy"
OIM_SCRIPT     = "4DEBSD_interactor.py"
PHASE_PATH     = r"E:\MPC-Share\Sam\Code\4D-EBSD\Data\GaN_hex_8kV.oem"

_HERE         = os.path.dirname(os.path.abspath(__file__))
_REQUEST_PATH = os.path.join(_HERE, "_ndp_request.json")
_RESULT_PATH  = os.path.join(_HERE, "_ndp_result.npy")

# ---------------------------------------------------------------------------
# Admin check
# ---------------------------------------------------------------------------

def _check_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class NDPApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("NDP Map Generator")
        root.resizable(True, True)

        if not _check_admin():
            messagebox.showwarning(
                "Not Administrator",
                "The OIM API requires an Administrator terminal.\n"
                "Please restart from an elevated (Run as administrator) terminal.",
            )

        frm = ttk.Frame(root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # .ang row
        ttk.Label(frm, text=".ang file:").grid(
            row=0, column=0, sticky="w", padx=(0, 4))
        self._ang_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self._ang_var, width=55).grid(
            row=0, column=1, sticky="ew")
        ttk.Button(frm, text="Browse...", command=self._pick_ang).grid(
            row=0, column=2, padx=(4, 0))

        # .up2 row
        ttk.Label(frm, text=".up2 file:").grid(
            row=1, column=0, sticky="w", padx=(0, 4), pady=(4, 0))
        self._up2_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self._up2_var, width=55).grid(
            row=1, column=1, sticky="ew", pady=(4, 0))
        ttk.Button(frm, text="Browse...", command=self._pick_up2).grid(
            row=1, column=2, padx=(4, 0), pady=(4, 0))

        # Run button
        self._run_btn = ttk.Button(frm, text="Run NDP Map", command=self._run)
        self._run_btn.grid(row=2, column=0, columnspan=3, pady=8)

        # Progress bar
        self._prog_var = tk.IntVar(value=0)
        self._prog_bar = ttk.Progressbar(
            frm, variable=self._prog_var, maximum=100, length=500)
        self._prog_bar.grid(row=3, column=0, columnspan=3, sticky="ew")

        # Status label
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(frm, textvariable=self._status_var, foreground="gray").grid(
            row=4, column=0, columnspan=3, pady=(2, 6))

        # Bounds controls (active once a map is loaded)
        self._vmin_var = tk.StringVar()
        self._vmax_var = tk.StringVar()
        self._ndp_map  = None

        bounds_frm = ttk.Frame(frm)
        bounds_frm.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Label(bounds_frm, text="Map bounds:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(bounds_frm, text="Min:").pack(side=tk.LEFT)
        vmin_entry = ttk.Entry(bounds_frm, textvariable=self._vmin_var, width=10)
        vmin_entry.pack(side=tk.LEFT, padx=(2, 8))
        vmin_entry.bind("<Return>", self._redraw_ndp_map)
        vmin_entry.bind("<FocusOut>", self._redraw_ndp_map)
        ttk.Label(bounds_frm, text="Max:").pack(side=tk.LEFT)
        vmax_entry = ttk.Entry(bounds_frm, textvariable=self._vmax_var, width=10)
        vmax_entry.pack(side=tk.LEFT, padx=(2, 8))
        vmax_entry.bind("<Return>", self._redraw_ndp_map)
        vmax_entry.bind("<FocusOut>", self._redraw_ndp_map)
        ttk.Button(bounds_frm, text="Reset", command=self._reset_ndp_bounds).pack(side=tk.LEFT)

        # Matplotlib canvas (NDP map displayed here)
        self._fig      = plt.Figure(figsize=(5, 5), constrained_layout=True)
        self._ax       = self._fig.add_subplot(1, 1, 1)
        self._ax.set_axis_off()
        self._colorbar = None
        self._canvas   = FigureCanvasTkAgg(self._fig, master=frm)
        self._canvas.get_tk_widget().grid(
            row=6, column=0, columnspan=3, sticky="nsew")

        frm.rowconfigure(6, weight=1)
        frm.columnconfigure(1, weight=1)

    # ---- File pickers -----------------------------------------------------

    def _pick_ang(self):
        p = filedialog.askopenfilename(
            title="Select .ang scan file",
            filetypes=[("ANG files", "*.ang"), ("All files", "*.*")],
        )
        if p:
            self._ang_var.set(p)

    def _pick_up2(self):
        p = filedialog.askopenfilename(
            title="Select .up2 pattern file",
            filetypes=[("UP2 files", "*.up2"), ("All files", "*.*")],
        )
        if p:
            self._up2_var.set(p)

    # ---- Run --------------------------------------------------------------

    def _run(self):
        ang = self._ang_var.get().strip()
        up2 = self._up2_var.get().strip()
        if not ang or not up2:
            messagebox.showerror(
                "Missing files", "Please select both a .ang and a .up2 file.")
            return

        self._run_btn.config(state="disabled")
        self._status_var.set("Reading scan dimensions from .ang file...")
        self._prog_var.set(0)
        self.root.update_idletasks()

        # Read scan dimensions from the ANG file (fast, no UP2 I/O)
        sys.path.insert(0, _HERE)
        from ANG import Ang
        ang_data = Ang(ang).generate_np_array()
        nrows, ncols = ang_data.shape[:2]
        N = nrows * ncols

        self._prog_bar.config(maximum=N)
        self._status_var.set(
            f"Scan: {nrows} x {ncols}  ({N} patterns).  Starting OIM simulation...")

        # Write the batch request JSON
        request = {
            "mode":        "batch_ndp",
            "osc_path":    ang,
            "up2_path":    up2,
            "phase_path":  PHASE_PATH,
            "output_path": _RESULT_PATH,
            "repo_path":   _HERE,
        }
        with open(_REQUEST_PATH, "w") as f:
            json.dump(request, f, indent=2)

        def _worker(total=N, nr=nrows, nc=ncols):
            script_full = os.path.join(OIM_SCRIPT_DIR, OIM_SCRIPT)
            try:
                proc = subprocess.Popen(
                    [sys.executable, script_full, _REQUEST_PATH],
                    cwd=OIM_SCRIPT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                # Stream stdout – parse PROGRESS: lines to drive the bar
                for line in proc.stdout:
                    line = line.strip()
                    if line.startswith("PROGRESS:"):
                        try:
                            done = int(line.split(":")[1].strip().split("/")[0])
                            self.root.after(
                                0, lambda v=done, t=total: self._set_progress(v, t))
                        except (IndexError, ValueError):
                            pass
                proc.wait()
                stderr_text = proc.stderr.read()
                if proc.returncode != 0:
                    self.root.after(
                        0, lambda e=stderr_text: self._on_error(e))
                else:
                    self.root.after(0, lambda: self._on_done(nr, nc))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self._on_error(e))

        threading.Thread(target=_worker, daemon=True).start()

    # ---- Callbacks (always called on the Tk main thread) -----------------

    def _set_progress(self, done: int, total: int):
        self._prog_var.set(done)
        self._status_var.set(f"Pattern {done} / {total}...")

    def _on_error(self, msg: str):
        self._run_btn.config(state="normal")
        self._status_var.set("Error – see dialog.")
        messagebox.showerror("OIM Error", (msg or "Unknown error.")[:2000])

    def _on_done(self, nrows: int, ncols: int):
        ndp_flat = np.load(_RESULT_PATH)
        self._ndp_map = ndp_flat.reshape(nrows, ncols)

        # Reset to default NDP range (0–1) on each new map
        self._vmin_var.set("")
        self._vmax_var.set("")
        self._redraw_ndp_map()

        mean_ndp = float(self._ndp_map.mean())
        self._status_var.set(
            f"Done!   {nrows} x {ncols}   Mean NDP = {mean_ndp:.4f}")
        self._run_btn.config(state="normal")

    def _redraw_ndp_map(self, _event=None):
        if self._ndp_map is None:
            return
        try:
            vmin = float(self._vmin_var.get()) if self._vmin_var.get().strip() else 0.0
        except ValueError:
            vmin = 0.0
        try:
            vmax = float(self._vmax_var.get()) if self._vmax_var.get().strip() else 1.0
        except ValueError:
            vmax = 1.0

        self._ax.clear()
        self._ax.set_axis_on()
        im = self._ax.imshow(
            self._ndp_map, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
        self._ax.set_title("NDP Map  (Experimental vs Simulated)", fontsize=10)
        self._ax.set_axis_off()

        if self._colorbar is None:
            self._colorbar = self._fig.colorbar(
                im, ax=self._ax, fraction=0.046, pad=0.04, label="NDP")
        else:
            self._colorbar.update_normal(im)

        self._canvas.draw()

    def _reset_ndp_bounds(self):
        self._vmin_var.set("")
        self._vmax_var.set("")
        self._redraw_ndp_map()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    NDPApp(root)
    root.mainloop()
