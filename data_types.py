# data_types.py

import numpy.typing
import streamlit as st
import cv2
import numpy as np
import streamlit.components.v1 as components

current_key = 1

def next_key():
    global current_key
    current_key += 1
    return current_key

class RGB:
    """
    Represents an RGB color.

    Attributes:
        r (int): Red component (0-255)
        g (int): Green component (0-255)
        b (int): Blue component (0-255)
    """
    def __init__(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b

    def visualize(self):
        st.color_picker("RGB Color", f"#{self.r:02x}{self.g:02x}{self.b:02x}", key=next_key())

    def __str__(self):
        return f"RGB({self.r}, {self.g}, {self.b})"
    
class ColorImage:
    """
    Represents a color image.

    Attributes:
        value (numpy.ndarray): 3D array of shape (height, width, 3) representing the RGB image
    """
    def __init__(self, value: numpy.typing.NDArray):
        self.value = value  # shape: (height, width, 3)
    def visualize(self):
        st.image(self.value)

    def __str__(self):
        return f'ColorImage(shape={self.value.shape})'
class ImageMask:
    """
    Represents a binary image mask.

    Attributes:
        value (numpy.ndarray): 2D array of shape (height, width) representing the binary mask
    """
    def __init__(self, value: numpy.typing.NDArray):
        self.value = value  # shape: (height, width)
    def visualize(self):
        st.image(self.value)

    def __str__(self):
        return f'ImageMask(shape={self.value.shape})'
class Contours:
    """
    Represents contours in an image.

    Attributes:
        image (numpy.ndarray): 3D array of shape (height, width, 3) representing the original image
        contours (list): List of contours, each contour is a numpy array of shape (n_points, 1, 2)
        contour_areas (list): List of contour areas
    """
    def __init__(self, image: numpy.typing.NDArray, contours: list[cv2.typing.MatLike]):
        self.image = image.copy()  # shape: (height, width, 3)
        self.contours = contours  # each contour shape: (n_points, 1, 2)
        self.contour_areas = [cv2.contourArea(contour) for contour in contours]

    def visualize(self):
        cv2.drawContours(self.image, self.contours, -1, (0, 255, 0), 3)
        st.image(self.image)
        st.write(self.contour_areas)

    def __str__(self):
        return f'Contours[{len(self.contours)}]'
    
class SVG:
    """
    Represents an SVG image.

    Attributes:
        value (str): String representation of the SVG
    """
    def __init__(self, value: str):
        self.value = value

    def visualize(self):
        components.html(self.value, width=600, height=400)

    def __str__(self):
        return f'SVG(length={len(self.value)})'
    
class ConvexHull:
    """
    Represents a convex hull of points.

    Attributes:
        points (numpy.ndarray): 2D array of shape (N, 1, 2) representing the convex hull points
        image_shape (tuple): Shape of the original image (height, width)
    """
    def __init__(self, points: np.ndarray, image_shape: tuple):
        self.points = points  # shape: (N, 1, 2)
        self.image_shape = image_shape

    def visualize(self):
        # Create a blank image
        img = np.zeros((*self.image_shape, 3), dtype=np.uint8)
        
        # Draw the convex hull
        cv2.drawContours(img, [self.points], 0, (0, 255, 0), 2)
        
        # Display the image
        st.image(img, channels="RGB", use_column_width=True)

    def __str__(self):
        return f'ConvexHull(points={len(self.points)}, image_shape={self.image_shape})'