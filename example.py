from step import step
from run_flow import run_flow
import time
from typing import Any
import cv2
import numpy as np
from data_types import ColorImage, ImageMask
from scipy.signal import find_peaks

@step
def estimate_k_from_histogram(image_obj: ColorImage, bins=32):
    image = image_obj.value
    # Calculate the histogram for each color channel
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])
    
    # Find peaks in the histogram
    peaks_r, _ = find_peaks(hist_r.flatten(), height=0)
    peaks_g, _ = find_peaks(hist_g.flatten(), height=0)
    peaks_b, _ = find_peaks(hist_b.flatten(), height=0)
    
    # Estimate k as the sum of the number of peaks in each channel
    k = len(peaks_r) + len(peaks_g) + len(peaks_b)
    
    return k

@step
def color_quantization(image_obj: ColorImage, k: int):
    image = image_obj.value
    # Step 1: Reshape the image to a 2D array of pixels
    data = image.reshape((-1, 3))
    
    # Convert to float32
    data = data.astype(np.float32)
    
    # Step 2: Define the criteria for k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Step 3: Apply k-means clustering
    best_labels: Any = None
    ret, label, center = cv2.kmeans(data, k, best_labels, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Step 4: Convert center back to 8-bit values
    center = center.astype(np.uint8)
    
    # Step 5: Replace each pixel's color with the centroid of the cluster it belongs to
    res = center[label.flatten()]
    
    # Step 6: Reshape the result to the original image shape
    quantized_image = res.reshape(image.shape)
    
    return ColorImage(quantized_image)

@step
def load_image():
    # Step 7: Load the image
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
