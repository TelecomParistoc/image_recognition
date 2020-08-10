#pylint:disable=E1101,W0621
import time
import numpy as np
import cv2
from method1 import get_orientation as get_orientation_m1
from method2 import get_orientation as get_orientation_m2

def most_frequent(List):
    """https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
    :param List: list to analyse
    :type List: list
    """
    dict = {}
    count, itm = 0, ''
    for item in reversed(List):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count:
            count, itm = dict[item], item
    return itm

def get_orientation(nb_images : int = 5) -> int:
    """Get the orientation of the compass

    :param nb_images: nb of image to use, defaults to 5
    :type nb_images: int, optional
    :return: orientation (1 = NORTH / 0 = SOUTH / -1 = NO RESULT)
    :rtype: int
    """
    #Take the images
    images_batch = []
    cap = cv2.VideoCapture(0)
    for _ in range(nb_images):
        _, frame = cap.read()
        images_batch.append(frame)
        time.sleep(0.1)
    
    #Test : images_batch = ["./tests/img/test0.jpg" for _ in range(5)]

    #Analyse them
    decisions = []
    for image in images_batch:
        tag_decision = get_orientation_m1(image)
        if tag_decision != -1:
            decisions.append(tag_decision)
        else:
            circle_decision = get_orientation_m2(image)
            decisions.append(circle_decision)

    return most_frequent(decisions)

if __name__ == "__main__":
    PATH = './tests/img/test0.jpg'
    print(get_orientation_m1(PATH))
    print(get_orientation_m2(PATH))
    print(get_orientation(get_orientation))
