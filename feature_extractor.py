"""Original PlantDoctor AI 1632-feature extractor from notebook cell 7.

IMPORTANT: OpenCV resize uses the default INTER_LINEAR interpolation.
Do not replace with INTER_AREA: a different abandoned notebook cell used it.
This function has not yet been tested against the saved .npy files in this environment.
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_features(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read: {path}")

    img = cv2.resize(img, (64, 64))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    features = []

    # HSV histograms
    for channel, maximum in [(0, 180), (1, 256), (2, 256)]:
        hist = cv2.calcHist(
            [hsv], [channel], None,
            [16], [0, maximum]
        ).ravel()
        hist /= hist.sum() + 1e-8
        features.extend(hist)

    # RGB statistics
    for c in range(3):
        values = rgb[:, :, c]
        features.extend([
            values.mean() / 255,
            values.std() / 255
        ])

    # LBP texture
    lbp = local_binary_pattern(
        gray, P=8, R=1, method="uniform"
    )
    hist, _ = np.histogram(
        lbp.ravel(), bins=np.arange(11)
    )
    hist = hist / (hist.sum() + 1e-8)
    features.extend(hist)

    # HOG
    h = hog(
        gray,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    features.extend(h)

    return np.asarray(features, dtype=np.float32)

