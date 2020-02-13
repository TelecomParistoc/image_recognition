import numpy as np
from cv2 import aruco
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

def get_orientation(nb_images=5):
    images_batch = []
    for i in range(nb_images):
        _, frame = capture.read()
        images_batch.append(frame)

    for image in images_batch:
        tag_decision = tag_function(image)
        circle_decision = circle_function(image)

        #Decision process



def tag_function(image):
    pass

def circle_function(image):
    pass
