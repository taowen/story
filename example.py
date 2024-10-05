# example.py

from step import step
from run_flow import run_flow
import time
from typing import Any
import cv2
import numpy as np
from data_types import ColorImage, ImageMask, RGB, SVG, ConvexHull
from scipy.signal import find_peaks
import xml.etree.ElementTree as ET

@step
def estimate_k_from_histogram(image_obj: ColorImage, bins=8):
    """
    Estimate the number of dominant colors in an image using histogram analysis.
    Args:
        image_obj (ColorImage): Input image object
        bins (int): Number of bins for histogram calculation
    Returns:
        int: Estimated number of dominant colors
    """
    image = image_obj.value  # shape: (height, width, 3)
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

    Args:
        image_obj (ColorImage): The input image object
        k (int): The number of clusters (quantized colors) to reduce the image to

    Returns:
        dict: A dictionary mapping each quantized color (as RGB) to its corresponding ImageMask
    """
    image = image_obj.value  # shape: (height, width, 3)
    Z = image.reshape((-1, 3))
    Z = Z.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    best_labels: Any = None
    _, label, center = cv2.kmeans(Z, k, best_labels, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = center.astype(np.uint8)

    masks = {}
    for i in range(k):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[label.reshape(image.shape[:2]) == i] = 255
        masks[RGB(*center[i])] = ImageMask(mask)

    return masks
@step
def load_image():
    """
    Load an image from file.

    Returns:
        ColorImage: Loaded image object
    """
    image = cv2.imread('quantized_mountain.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return ColorImage(image)

@step
def compute_convex_hull(mask: ImageMask):
    """
    Compute the convex hull of the largest contour in a binary mask.

    Args:
        mask (ImageMask): Input binary mask

    Returns:
        numpy.ndarray: Convex hull points as a 2D array of shape (N, 1, 2),
                       where N is the number of hull points. Each point is
                       represented as [x, y] coordinates.
    """
    contours, _ = cv2.findContours(mask.value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        return ConvexHull(hull, mask.value.shape[:2])
    else:
        raise Exception('unexpected')

@step
def get_largest_component(mask_color: RGB, mask: ImageMask):
    """
    Extract the largest connected component from a binary mask.

    Args:
        mask_color (RGB): Color associated with the mask
        mask (ImageMask): Input binary mask

    Returns:
        ImageMask: Mask containing only the largest connected component
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.value, connectivity=8)
    
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_component = np.zeros_like(mask.value)
        largest_component[labels == largest_label] = 255
        return ImageMask(largest_component)
    else:
        return ImageMask(np.zeros_like(mask.value))

@step
def create_svg_from_hull(hull_obj: ConvexHull, image_shape: tuple):
    """
    Create an SVG path from the convex hull points.

    Args:
        hull (numpy.ndarray): Convex hull points as a 2D array of shape (N, 1, 2)
        image_shape (tuple): Shape of the original image (height, width)

    Returns:
        str: SVG string representation of the path
    """
    hull = hull_obj.points
    # Create the root SVG element
    svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': str(image_shape[1]),
        'height': str(image_shape[0]),
    })

    # Create the path element
    path = ET.SubElement(svg, 'path', {
        'fill': 'none',
        'stroke': 'black',
        'stroke-width': '2',
    })

    # Create the path data
    path_data = f"M {hull[0][0][0]},{hull[0][0][1]}"
    for point in hull[1:]:
        path_data += f" L {point[0][0]},{point[0][1]}"
    path_data += " Z"

    path.set('d', path_data)

    # Convert the SVG to a string
    return SVG(ET.tostring(svg, encoding='unicode'))

@step
def demo():
    """
    Demonstrate the image processing pipeline.
    """
    image = load_image()
    k = estimate_k_from_histogram(image)
    masks = color_quantization(image, k)
    largest_mask_color, largest_mask = max(masks.items(), key=lambda item: cv2.countNonZero(item[1].value))
    del masks[largest_mask_color]
    largest_mask_color, largest_mask = max(masks.items(), key=lambda item: cv2.countNonZero(item[1].value))
    largest_component = get_largest_component(largest_mask_color, largest_mask)
    convex_hull = compute_convex_hull(largest_component)
    svg_string = create_svg_from_hull(convex_hull, image.value.shape[:2])
    

def worker_thread():
    demo()

if __name__ == "__main__":
    run_flow(worker_thread)
