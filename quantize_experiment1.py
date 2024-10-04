from step import step
import numpy as np
import cv2
from data_types import ColorImage
import typing
from visualize_steps import visualize_steps
import typing
import cv2
import numpy as np
from typing import Dict
from data_types import ColorImage, RGB, ImageMask, Contours
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
    k = 5
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
def find_biggest_color_layer(image_obj: ColorImage):
    """Quantize the colors of an image using cv2 k-means clustering.
    """
    # Create a dictionary mapping RGB values to ImageMasks
    biggest_color_layer = None
    for color, mask in find_clusters(image_obj).items():
        color_layer = find_components(image_obj, color, mask)
        if biggest_color_layer is None:
            biggest_color_layer = color_layer
        elif color_layer['max_component_area'] > biggest_color_layer['max_component_area']:
            biggest_color_layer = color_layer
    return biggest_color_layer

@step
def find_components(image_obj: ColorImage, color: RGB, mask_obj: ImageMask):
    """Find connected components in the mask and return their statistics."""
    # Use connected components with stats to get labels and statistics
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_obj.value, connectivity=8)

    # The first component (label 0) is the background, so we ignore it
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    component_masks = []
    valid_component_indices = []

    # Create masks for all connected components except the background
    for i in range(1, stats.shape[0]):
        if component_areas[i - 1] >= 10:  # Filter components by area
            component_mask = (labels == i).astype(np.uint8) * 255
            component_masks.append(ImageMask(component_mask))
            valid_component_indices.append(i)

    # Filter the component areas and masks
    filtered_component_areas = [component_areas[i - 1] for i in valid_component_indices]

    # Find the largest connected component among the valid ones
    if len(filtered_component_areas) == 0:
        max_component_area = 0
    else:
        max_component_index = np.argmax(filtered_component_areas)
        max_component_area = filtered_component_areas[max_component_index]

    return {
        "color": color,
        "mask": mask_obj,
        "components": component_masks,  # all valid components masks
        "component_areas": filtered_component_areas,  # all valid components areas
        "max_component_area": max_component_area
    }


@step
def load_image():
    image = cv2.imread('cartoon.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ColorImage(image)
    return image

@step
def pick_null_color(image_obj: ColorImage) -> RGB:
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image_obj.value, cv2.COLOR_RGB2HSV)
    # Create a mask for the entire image
    mask = np.ones((hsv_image.shape[0], hsv_image.shape[1]), dtype=np.uint8) * 255
    # Apply morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Find the average color of the masked region
    avg_hsv = cv2.mean(hsv_image, mask=mask)[:3]
    # Ensure the color is not present in the image
    avg_hsv = np.array([(avg_hsv[0] + 90) % 180, avg_hsv[1], avg_hsv[2]], dtype=np.uint8)
    # Reshape to a 1x1x3 array to represent a single pixel in HSV
    avg_hsv = avg_hsv.reshape(1, 1, 3)

    # Convert HSV to RGB
    avg_rgb = cv2.cvtColor(avg_hsv, cv2.COLOR_HSV2RGB)
    r, g, b = avg_rgb[0, 0]
    return RGB(r, g, b)


@step
def demo():
    image = load_image()
    null_color = pick_null_color(image)
    find_biggest_color_layer(image)

def main():
    if visualize_steps():
        demo()

if __name__ == "__main__":
    main()
