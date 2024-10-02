import cv2
import numpy as np
from step import step

@step
def quantize_color(image):
    """Quantize the colors of an image using k-means clustering.
    Args:
        image: cv2 image object

    Returns:
        A dictionary where keys are RGB tuples representing the quantized colors
        and values are the input image masked with the corresponding color. 
    """
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 8  # Number of clusters (colors)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    color_map = {}

    for i in range(k):
        # Create a mask for pixels belonging to the current cluster
        mask = (labels.flatten() == i).astype(np.uint8) * 255
        mask = mask.reshape(image.shape[:2])

        # Use the mask for bitwise_and
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert color from BGR to RGB for the key
        color_key = tuple(int(c) for c in centers[i][::-1])  
        color_map[color_key] = masked_image

    return color_map
