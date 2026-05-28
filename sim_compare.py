"""
sim_compare.py
--------------
Compare an experimental Kikuchi pattern (from a .up2 file) with its
simulated counterpart generated via the OIM API.

Because oimpy can only run from its own working directory, this script
delegates simulation to script_to_paste.py by:
  1. Writing a JSON request file.
  2. Running script_to_paste.py as a subprocess with the correct cwd.
  3. Loading the .npy result that script_to_paste.py writes back.

Usage:
    python sim_compare.py [pattern_index]

    pattern_index defaults to 0 if not supplied.
"""

import sys
import os
import json
import subprocess
import ctypes

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# ---------------------------------------------------------------------------
# Paths – edit these to match your environment
# ---------------------------------------------------------------------------
OIM_SCRIPT_DIR = r"C:\Users\User\Desktop\OIMpy"
OIM_SCRIPT     = "4DEBSD_interactor.py"

OSC_PATH   = r"E:\MPC-Share\Sam\Code\4D-EBSD\Data\20240508_27238_flipX_Rescan_with_PC_of_Marc.ang"
UP2_PATH   = r"E:\MPC-Share\Sam\Code\4D-EBSD\Data\20240508_27238_256x256_flipX.up2"
PHASE_PATH = r"E:\MPC-Share\Sam\Code\4D-EBSD\Data\GaN_hex_8kV.oem"

# Temp files used for inter-process communication (created next to this script)
_HERE         = os.path.dirname(os.path.abspath(__file__))
_REQUEST_PATH = os.path.join(_HERE, "_sim_request.json")
_RESULT_PATH  = os.path.join(_HERE, "_sim_result.npy")

# Add repo root to path so we can use our own UP2 reader
sys.path.insert(0, _HERE)
from UP2 import UP2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr, dtype=np.float32)


def _match_sizes(a: np.ndarray, b: np.ndarray):
    """Shrink the larger of the two images so both have the same shape."""
    th = min(a.shape[0], b.shape[0])
    tw = min(a.shape[1], b.shape[1])
    if a.shape != (th, tw):
        a = resize(a, (th, tw), anti_aliasing=True, preserve_range=True).astype(np.float32)
    if b.shape != (th, tw):
        b = resize(b, (th, tw), anti_aliasing=True, preserve_range=True).astype(np.float32)
    return a, b


def _to_2d(arr) -> np.ndarray:
    """Coerce any ndarray-like to float32 2D."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_experimental(pattern_idx: int) -> np.ndarray:
    """Read and process the experimental pattern directly from the .up2 file."""
    up2 = UP2(UP2_PATH)
    pat = up2.read_pattern(pattern_idx, process=True)
    return _to_2d(pat)


def _check_admin():
    """Warn if not running as administrator (OIM API requires it)."""
    try:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        is_admin = False
    if not is_admin:
        print(
            "WARNING: This script must be run from an Administrator terminal.\n"
            "  Right-click your terminal / Anaconda Prompt and choose\n"
            "  'Run as administrator', then try again.",
            file=sys.stderr,
        )


def request_simulated(pattern_idx: int) -> np.ndarray:
    """
    Ask script_to_paste.py (running in the OIM API directory) to simulate
    the pattern and return it as a 2D float32 array.
    """
    request = {
        "pattern_idx": pattern_idx,
        "osc_path":    OSC_PATH,
        "up2_path":    UP2_PATH,
        "phase_path":  PHASE_PATH,
        "output_path": _RESULT_PATH,
    }
    with open(_REQUEST_PATH, "w") as f:
        json.dump(request, f, indent=2)

    script_full_path = os.path.join(OIM_SCRIPT_DIR, OIM_SCRIPT)
    print(f"Calling OIM API script: {script_full_path}")
    # Re-use the current Python interpreter (already in the oimapi conda env)
    # and set cwd to the OIMpy directory — identical to running the script
    # directly from that folder, which is what makes oimpy's DLLs load.
    result = subprocess.run(
        [sys.executable, script_full_path, _REQUEST_PATH],
        cwd=OIM_SCRIPT_DIR,
        capture_output=True,
        text=True,
    )

    # Always print OIM output so errors are visible.
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"OIM script exited with code {result.returncode}. "
            "Check stderr above for details."
        )

    return _to_2d(np.load(_RESULT_PATH))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _check_admin()
    pattern_idx = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    print(f"Pattern index: {pattern_idx}")

    print("Loading experimental pattern…")
    exp = load_experimental(pattern_idx)

    print("Requesting simulated pattern via OIM API…")
    sim = request_simulated(pattern_idx)

    # Make both images the same size (shrink the larger one).
    exp_orig_shape = exp.shape
    sim_orig_shape = sim.shape
    exp, sim = _match_sizes(exp, sim)

    # Normalise to [0, 1] for display.
    exp = _normalize(exp)
    sim = _normalize(sim)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

    axs[0].imshow(exp, cmap="gray", origin="upper")
    axs[0].set_title(f"Experimental  (#{pattern_idx})", fontsize=10)
    axs[0].set_axis_off()
    if exp.shape != exp_orig_shape:
        axs[0].text(0.5, -0.04,
                    f"Original: {exp_orig_shape[1]}x{exp_orig_shape[0]}  ->  {exp.shape[1]}x{exp.shape[0]}",
                    ha="center", va="top", transform=axs[0].transAxes, fontsize=7, color="gray")

    axs[1].imshow(sim, cmap="gray", origin="upper")
    axs[1].set_title(f"Simulated  (#{pattern_idx})", fontsize=10)
    axs[1].set_axis_off()
    if sim.shape != sim_orig_shape:
        axs[1].text(0.5, -0.04,
                    f"Original: {sim_orig_shape[1]}x{sim_orig_shape[0]}  ->  {sim.shape[1]}x{sim.shape[0]}",
                    ha="center", va="top", transform=axs[1].transAxes, fontsize=7, color="gray")

    fig.suptitle("Kikuchi Pattern Comparison", fontsize=12)
    plt.show()


if __name__ == "__main__":
    main()
