import numpy.typing
import streamlit as st
import cv2
import numpy as np

current_key = 1

def next_key():
    global current_key
    current_key += 1
    return current_key

class RGB:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def visualize(self):
        st.color_picker("RGB Color", f"#{self.r:02x}{self.g:02x}{self.b:02x}", key=next_key())

    def __str__(self):
        return f"RGB({self.r}, {self.g}, {self.b})"
    
class ColorImage:
    def __init__(self, value: numpy.typing.NDArray):
        self.value = value

    def visualize(self):
        st.image(self.value)

    def __str__(self):
        return '`ColorImage`'
    
class ImageMask:
    def __init__(self, value: numpy.typing.NDArray):
        self.value = value

    def visualize(self):
        st.image(self.value)

    def __str__(self):
        return '`ImageMask`'

class Contours:
    def __init__(self, image: numpy.typing.NDArray, contours: list[cv2.typing.MatLike]):
        self.image = image.copy()
        self.contours = contours
        self.contour_areas = [cv2.contourArea(contour) for contour in contours]

    def visualize(self):
        cv2.drawContours(self.image, self.contours, -1, (0, 255, 0), 3)
        st.image(self.image)
        st.write(self.contour_areas)


    def __str__(self):
        return f'Contours[{len(self.contours)}]'
