#pylint:disable=E1101,W0621
import numpy as np
import cv2
from method1 import get_orientation as get_orientation_m1
from method2 import get_orientation as get_orientation_m2

#TODO: finalize function
def get_orientation(nb_images=5):
    images_batch = []
    cap = cv2.VideoCapture(0)
    for i in range(nb_images):
        _, frame = cap.read()
        images_batch.append(frame)
        #TODO: sleep ?

    for image in images_batch:
        tag_decision = get_orientation_m1(image)
        circle_decision = get_orientation_m2(image)
        #TODO: Decision process

if __name__ == "__main__":
    path = './tests/img/test2.jpg'
    print(get_orientation_m1(path))
    print(get_orientation_m2(path))