from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# Load FITS file
fits_file = "AIA20231211_085808_lev0_0131.fits" # provided in a zip file

with fits.open(fits_file) as hdul:
    data = hdul[0].data
    header = hdul[0].header

# Ensure 2D
if data.ndim > 2:
    data = data[0]

# Image shape
H, W = data.shape
h_mid, w_mid = H // 2, W // 2

# Full image median
global_median = np.nanmedian(data)
print(f'Global median: {global_median:.2f}')

# Copy full image
M = np.copy(data)

# Define quadrants
quads = [
    (slice(0, h_mid), slice(0, w_mid)),     # Q1: top-left
    (slice(0, h_mid), slice(w_mid, W)),     # Q2: top-right
    (slice(h_mid, H), slice(0, w_mid)),     # Q3: bottom-left
    (slice(h_mid, H), slice(w_mid, W))      # Q4: bottom-right
]

# Apply: quadrant_image = M - median(quadrant)
corrected = np.zeros_like(data)

for i, (ys, xs) in enumerate(quads):
    quad = data[ys, xs]
    quad_median = np.nanmedian(quad)
    corrected_quad = M[ys, xs] - (quad_median - global_median)
    corrected[ys, xs] = corrected_quad
    print(f'Q{i+1} median: {quad_median:.2f} → shifted by {(quad_median - global_median):.2f}')

# Clip to 1st–99th percentiles for display
p1, p99 = np.percentile(corrected[np.isfinite(corrected)], [1, 99])
norm_image = np.clip(corrected, p1, p99)
norm_image = (norm_image - p1) / (p99 - p1)

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(norm_image, origin='lower', cmap='gray', vmin=0, vmax=1)
plt.colorbar(label='Normalized Value')
plt.title('Quadrant-Corrected FITS Image')

plt.subplot(1, 2, 2)
plt.hist(norm_image.flatten(), bins=200, color='orange', alpha=0.7)
plt.xlabel('Normalized Pixel Value')
plt.ylabel('Count')
plt.title('Histogram after Median Shift + Normalization')

plt.tight_layout()
plt.show()
