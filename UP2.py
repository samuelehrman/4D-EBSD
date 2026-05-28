import os
import struct

from skimage import io
from scipy import ndimage
import numpy as np


class UP2:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.i = 0
        self.start_byte = np.int64(16)
        self.header()
        self.set_processing()

    def __len__(self):
        return self.nPatterns

    def __getitem__(self, i):
        return self.read_pattern(i, process=True)

    def __str__(self):
        return (
            f"UP2 file: {self.path}\n"
            f"Patterns: {self.nPatterns}\n"
            f"Pattern shape: {self.patshape}\n"
            f"File size: {self.filesize}\n"
            f"Bits per pixel: {self.bitsPerPixel}\n"
        )

    def __repr__(self):
        return self.__str__()

    def header(self):
            chunk_size = 4
            # Read the 4 header integers (16 bytes total)
            tmp = self.read(16)
            # Unpack as 4 integers (little-endian)
            header_vals = struct.unpack("<iiii", tmp)
            
            v = header_vals[0]      # vers
            w = header_vals[1]      # width
            h = header_vals[2]      # height
            ds = header_vals[3]     # dStart
            
            # Mirror the macro's validation and version-specific swap
            if w < 1 or h < 1 or w > 5000 or h > 5000 or ds < 0:
                if v > 2:
                    # In newer versions, the first two ints are actually Width and Height
                    # and the data starts much earlier (offset 8)
                    self.patshape = (v, w) # width, height
                    self.start_byte = np.int64(8)
                else:
                    raise ValueError("Not a valid UP2 file or unsupported header format.")
            else:
                self.patshape = (w, h)
                self.start_byte = np.int64(ds)

            self.bitsPerPixel = 16 # Since it's a UP2
            sizeBytes = os.path.getsize(self.path) - self.start_byte
            self.filesize = f"{round(sizeBytes / 1e6, 1)} MB"
            
            bytesPerPixel = 2
            self.nPatterns = int((sizeBytes / bytesPerPixel) / (self.patshape[0] * self.patshape[1]))
            self.pattern_bytes = np.int64(self.patshape[0] * self.patshape[1] * 2)

    def set_processing(
        self,
        low_pass_sigma: float = 0.0,
        high_pass_sigma: float = 0.0,
        truncate_std_scale: float = 0.0,
    ):
        """Set the parameters for processing the patterns.
        Values of 0.0 will skip the step.

        Args:
            low_pass_sigma (float): The sigma for the low pass filter. Roughly 1% of the image size works well.
            high_pass_sigma (float): The sigma for the high pass filter. Roughly 20% of the image size works well.
            truncate_std_scale (float): The number of standard deviations to truncate. 3.0 is a good value.
        """
        self.low_pass_sigma = low_pass_sigma
        self.high_pass_sigma = high_pass_sigma
        self.truncate_std_scale = truncate_std_scale

    def read(self, chunks, i=None):
        """Read the next `chunks` bytes from the file. If `i` is not None, read from the current position."""
        if i is None:
            i = self.i
        with open(self.path, "rb") as upFile:
            upFile.seek(i)
            data = upFile.read(chunks)
        self.i += chunks
        return data

    def read_pattern(self, i, process=False):
        # Read in the patterns
        seek_pos = np.int64(self.start_byte + np.int64(i) * self.pattern_bytes)
        buffer = self.read(chunks=self.pattern_bytes, i=seek_pos)
        pat = np.frombuffer(buffer, dtype=np.uint16).reshape((self.patshape[1], self.patshape[0]))
        if process:
            pat = self.process_pattern(pat)
        return pat

    def read_patterns(self, idx=-1, process=False):
        if type(idx) == int:
            if idx != -1:
                return self.read_pattern(idx)
            else:
                idx = range(self.nPatterns)
        else:
            idx = np.asarray(idx)

        # Read in the patterns
        in_shape = idx.shape + self.patshape
        idx = idx.flatten()
        if process:
            pats = np.zeros(idx.shape + self.patshape, dtype=np.float32)
        else:
            pats = np.zeros(idx.shape + self.patshape, dtype=np.uint16)
        for i in range(idx.shape[0]):
            pats[i] = self.read_pattern(idx[i], process)
        return pats.reshape(in_shape)

    def process_pattern(
        self,
        img: np.ndarray,
    ) -> np.ndarray:
        """Cleans patterns by equalizing the histogram and normalizing.
        Applies a bandpass filter to the patterns and truncates the extreme values.
        Images will be in the range [0, 1].

        Args:
            img (np.ndarray): The patterns to clean. (H, W)
            low_pass_sigma (float): The sigma for the low pass filter.
            high_pass_sigma (float): The sigma for the high pass filter.
            truncate_std_scale (float): The number of standard deviations to truncate.
        Returns:
            np.ndarray: The cleaned patterns. (N, H, W)"""

        # Correct dtype
        img = img.astype(np.float32)

        # Normalize
        img = (img - img.min()) / (img.max() - img.min())

        # Low pass filter
        if self.low_pass_sigma > 0:
            img = ndimage.gaussian_filter(img, self.low_pass_sigma)

        # High pass filter
        if self.high_pass_sigma > 0:
            background = ndimage.gaussian_filter(img, self.high_pass_sigma)
            img = img - background

        # Truncate step
        if self.truncate_std_scale > 0:
            mean, std = img.mean(), img.std()
            img = np.clip(
                img,
                mean - self.truncate_std_scale * std,
                mean + self.truncate_std_scale * std,
            )

        # Re normalize
        img = (img - img.min()) / (img.max() - img.min())

        return img


if __name__ == "__main__":
    up2_path = r"C:\Users\samue\Box\Tribeam\Dislocation-Density-V1\Data\20240320_27061_256x256.up2"
    up2 = UP2(up2_path)
    print(up2.__repr__())
    pat = up2.read_pattern(30000, process=True)
    pat = np.around(255 * (pat - pat.min()) / (pat.max() - pat.min())).astype(np.uint8)
    print(pat.shape)
    # print(type(pat))
    # print(np.max(pat), np.min(pat))
    io.imsave("pattern.png", pat)
    # print(pat)
