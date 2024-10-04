from step import step
from run_flow import run_flow
import time
from typing import Any
import cv2
import numpy as np
from data_types import ColorImage, ImageMask, RGB
from scipy.signal import find_peaks

@step
def estimate_k_from_histogram(image_obj: ColorImage, bins=8):
    image = image_obj.value
    # Calculate histogram for all three channels together
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # Flatten the histogram for peak detection
    hist_flattened = hist.flatten()
    # Find peaks in the flattened histogram
    peaks, _ = find_peaks(hist_flattened, height=32, threshold=5, distance=5, prominence=10)
    # Estimate k as the number of peaks
    k = len(peaks)
    return k

@step
def color_quantization(image_obj: ColorImage, k: int):
    """
    Perform color quantization on an image using K-means clustering.

    This function reduces the number of colors in an image by clustering similar colors
    into a specified number of clusters. It then creates a dictionary mapping each
    quantized color to its corresponding mask in the original image.

    Args:
        image_obj (ColorImage): The input image object.
        k (int): The number of clusters (quantized colors) to reduce the image to.

    Returns:
        dict: A dictionary mapping each quantized color (as a tuple) to its corresponding ImageMask.
    """
    image = image_obj.value
    Z = image.reshape((-1, 3))
    Z = Z.astype(np.float32)

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    best_labels: Any = None
    _, label, center = cv2.kmeans(Z, k, best_labels, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to uint8 and reshape to original image shape
    center = center.astype(np.uint8)

    # Create a mask for each quantized color
    masks = {}
    for i in range(k):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[label.reshape(image.shape[:2]) == i] = 255
        masks[RGB(*center[i])] = ImageMask(mask)

    return masks

@step
def load_image():
    image = cv2.imread('quantized_mountain.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return ColorImage(image)

@step
def demo():
    image = load_image()
    k = estimate_k_from_histogram(image)
    quantized_image = color_quantization(image, k)

def worker_thread():
    demo()

if __name__ == "__main__":
    run_flow(worker_thread)
