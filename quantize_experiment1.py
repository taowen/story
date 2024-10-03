from step import step
import numpy as np
import cv2
from data_types import ColorImage
import typing
from visualize_steps import visualize_steps

import cv2
import numpy as np
from typing import Dict
from data_types import ColorImage, RGB, ImageMask
import typing
from step import step

@step
def find_clusters(image_obj: ColorImage) -> Dict[RGB, ImageMask]:
    """cluster the colors of an image using cv2 k-means clustering.
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
        color = RGB(centers[i][0], centers[i][1], centers[i][2])
        mask = np.where((quantized_image == centers[i]).all(axis=2), 255, 0).astype(np.uint8)
        mask_obj = ImageMask(mask)
        color_masks[color] = mask_obj
    return color_masks

@step
def quantize_color(image_obj: ColorImage) -> Dict[RGB, ImageMask]:
    """Quantize the colors of an image using cv2 k-means clustering.
    """
    # Create a dictionary mapping RGB values to ImageMasks
    color_masks = find_clusters(image_obj)
    for mask in color_masks.values():
        get_dominant_color_in_mask(image_obj, mask)
    return color_masks

@step
def get_dominant_color_in_mask(image_obj: ColorImage, mask_obj: ImageMask) -> tuple[RGB, ImageMask]:
    """Given an image and a mask, return the most popular color within that mask."""
    image = image_obj.value
    mask = mask_obj.value

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked_image[np.where(mask != 0)].reshape((-1, 3))
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant_color = unique_colors[np.argmax(counts)]
    dominant_color_mask = np.all(image == dominant_color, axis=-1).astype(np.uint8) * 255
    return RGB(dominant_color[0], dominant_color[1], dominant_color[2]), ImageMask(dominant_color_mask)

@step
def load_image():
    image = cv2.imread('quantized_mountain.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ColorImage(image)
    return image

@step
def demo():
    image = load_image()
    quantize_color(image)

def main():
    if visualize_steps():
        demo()

if __name__ == "__main__":
    main()
