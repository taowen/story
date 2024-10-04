from step import step
from run_flow import run_flow
import time
from typing import Any
import cv2
import numpy as np
from data_types import ColorImage, ImageMask
from scipy.signal import find_peaks

@step
def estimate_k_from_histogram(image_obj: ColorImage, bins=8):
    image = image_obj.value
    # Calculate histogram for all three channels together
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # Flatten the histogram for peak detection
    hist_flattened = hist.flatten()
    # Find peaks in the flattened histogram
    peaks, _ = find_peaks(hist_flattened, height=16)
    # Estimate k as the number of peaks
    k = len(peaks)
    return k

@step
def color_quantization(image_obj: ColorImage, k: int):
    image = image_obj.value
    data = image.reshape((-1, 3))
    data = data.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    best_labels: Any = None
    ret, label, center = cv2.kmeans(data, k, best_labels, criteria, 10, cv2.KMEANS_PP_CENTERS)
    center = center.astype(np.uint8)
    res = center[label.flatten()]
    quantized_image = res.reshape(image.shape)
    return ColorImage(quantized_image)

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
