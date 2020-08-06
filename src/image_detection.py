#pylint:disable=E1101,W0621
import numpy as np
import cv2

def get_orientation(nb_images=5):
    images_batch = []
    for i in range(nb_images):
        _, frame = cv2.capture.read()
        images_batch.append(frame)

    for image in images_batch:
        tag_decision = tag_function(image)
        circle_decision = circle_function(image)

        #Decision process



def tag_function(image):
    pass

def circle_function(image):
    pass
