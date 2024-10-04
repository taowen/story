from step import step
import numpy as np
import cv2
from data_types import ColorImage, ImageMask
import typing
from visualize_steps import visualize_steps
import typing
import cv2
import numpy as np
from typing import Dict
from data_types import ColorImage, RGB, ImageMask, Contours
import typing
from step import step
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import time

@step
def load_image():
    image = cv2.imread('quantized_mountain.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ColorImage(image)
    return image

@step
def segment_image_optics(image: ColorImage) -> ImageMask:
    # Flatten the image to a 2D array of pixels (rows, cols, 3) -> (rows * cols, 3)
    print('!!! enter')
    time.sleep(100)
    print('!!! exit')

@step
def demo():
    image = load_image()
    mask = segment_image_optics(image)

def main():
    if visualize_steps():
        import threading
        def run_demo():
            for i in range(10):
                print('!!!', i)
                time.sleep(10)
        thread = threading.Thread(target=run_demo)
        thread.start()
        demo()

if __name__ == "__main__":
    main()
