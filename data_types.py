import numpy.typing
import streamlit as st

class RGB:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def visualize(self):
        st.color_picker("RGB Color", f"#{self.r:02x}{self.g:02x}{self.b:02x}")

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
