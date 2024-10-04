from step import step
import cv2
from data_types import ColorImage, RGB, ImageMask
from visualize_steps import visualize_steps
from collections import Counter
import numpy as np

@step
def load_image():
    image = cv2.imread('cartoon.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ColorImage(image)
    return image

@step
def list_colors(image: ColorImage):
    # Flatten the image array
    flattened_image = image.value.reshape((-1, 3))

    # Convert RGB tuples to strings for use as dictionary keys
    color_counts = Counter([tuple(color) for color in flattened_image])

    # Sort colors by count in descending order
    sorted_colors = sorted(color_counts.items(), key=lambda item: item[1], reverse=True)

    # Create a list of RGB objects
    total_pixels = len(flattened_image)
    rgb_colors = [(RGB(r, g, b), count / total_pixels) for (r, g, b), count in sorted_colors]

    return rgb_colors


@step
def cut_layers(image: ColorImage, colors: list[tuple[RGB, float]]):
    for color, ratio in colors:
        if ratio > 0.001:
            cut_layer_of_color(image, color)

@step
def cut_layer_of_color(image: ColorImage, color: RGB):
    # Create a mask for the specified color
    mask = (image.value[:, :, 0] == color.r) & \
           (image.value[:, :, 1] == color.g) & \
           (image.value[:, :, 2] == color.b)

    # Find contours of connected components in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours: list[cv2.typing.MatLike] = contours

    # Assuming you have an image loaded as 'original_image'
    image_with_contours = image.value.copy()  # Create a copy to draw on

    # Draw contours 
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Green outlines, 2 pixels thick

    return ColorImage(image_with_contours), len(contours)

@step
def demo():
    image = load_image()
    colors = list_colors(image)
    cut_layers(image, colors)

def main():
    if visualize_steps():
        demo()

if __name__ == "__main__":
    main()

