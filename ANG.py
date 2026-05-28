"""
Class to handle .ang files for EBSD data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt


class Ang:
    """
    Class to easily parse .ang files.

    Inputs:
        ang_path (str): Path to the .ang file
    """

    def __init__(self, ang_path):
        self.ang_path = ang_path
        self.header = {}
        self.column_headers = []
        self._data = None
        self._parse_header()

    def _parse_header(self):
        """Read header metadata from the .ang file."""
        with open(self.ang_path, "r") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                line = line.lstrip("#").strip()
                if line.startswith("XSTEP:"):
                    self.header["xstep"] = float(line.split(":")[1].strip())
                elif line.startswith("YSTEP:"):
                    self.header["ystep"] = float(line.split(":")[1].strip())
                elif line.startswith("NCOLS_ODD:"):
                    self.header["ncols"] = int(line.split(":")[1].strip())
                elif line.startswith("NROWS:"):
                    self.header["nrows"] = int(line.split(":")[1].strip())
                elif line.startswith("COLUMN_HEADERS:"):
                    self.column_headers = [c.strip() for c in line.split(":")[1].split(",")]

        self.nrows = self.header.get("nrows")
        self.ncols = self.header.get("ncols")
        self.xstep = self.header.get("xstep")
        self.ystep = self.header.get("ystep")

    def generate_np_array(self, save_path=None):
        """
        Parse the data columns of the .ang file and return them as a numpy array.

        The returned array has shape (N_pixels, N_columns) where columns are:
        phi1, PHI, phi2, x, y, IQ, CI, Phase index, SEM, Fit

        Args:
            save_path (str, optional): Full file path to save the array as a .npy file.

        Returns:
            np.ndarray: Array of shape (N_pixels, N_columns).
        """
        rows = []
        in_data = False
        with open(self.ang_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    if "HEADER: End" in line:
                        in_data = True
                    continue
                if in_data:
                    values = line.split()
                    if values:
                        rows.append([float(v) for v in values])

        nrows = self.header["nrows"]
        ncols = self.header["ncols"]
        data = np.array(rows).reshape(nrows, ncols, 10)
        self._data = data

        if save_path is not None:
            np.save(save_path, data)

        return data