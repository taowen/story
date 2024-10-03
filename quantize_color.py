import cv2
import numpy as np
from typing import Dict
from data_types import ColorImage, RGB, ImageMask
import typing
from step import step

@step
def quantize_color(image_obj: ColorImage) -> Dict[RGB, ImageMask]:
    """Quantize the colors of an image using cv2 k-means clustering.
    """
    image = image_obj.value
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3)).astype(np.float32)

    # Define criteria for k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Number of clusters (colors)
    k = 8
    init_labels: typing.Any = None
    _, labels, centers = cv2.kmeans(pixels, k, init_labels, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = centers.astype(np.uint8)
    # Map each pixel to its cluster center
    quantized_image = centers[labels.flatten()]
    # Reshape back to the original image shape
    quantized_image = quantized_image.reshape(image.shape)

    # Create a dictionary mapping RGB values to ImageMasks
    color_masks = {}
    for i in range(k):
        color = RGB(centers[i][2], centers[i][1], centers[i][0]) # Note: OpenCV uses BGR order
        mask = np.where((quantized_image == centers[i]).all(axis=2), 255, 0).astype(np.uint8)
        color_masks[color] = ImageMask(mask)

    return color_masks
