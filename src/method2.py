#pylint:disable=E1101,W0621
from typing import Union
import random as rng
import numpy as np
import numpy.linalg as alg
import cv2

rng.seed(12345)
DEBUG = 1
VECTEURS = 1
MIN_PIXEL = 3000

def get_orientation(input_image : Union[str, np.ndarray]) -> int:
    """Get the orientation from an image

    :param input: path of the image taken by the camera OR frame imported via imread
    :type input: Union[str, np.ndarray]
    :return: orientation (1 = NORTH / 0 = SOUTH)
    :rtype: int
    """
    #Load image
    if isinstance(input_image, str):
        img = cv2.imread(input_image)
    else:
        img = input_image

    img = cv2.resize(img, (360, 640), interpolation = cv2.INTER_AREA)
    height, width = img.shape[0], img.shape[1]

    #Blur
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)

    #Filter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_v = np.array([0, 0, 0])
    upper_v = np.array([255, 255, 120])
    mask = cv2.inRange(hsv, lower_v, upper_v)

    #Generate connected components, then calculate the angle
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    angle = get_orientation_from_data_final(centroids, num, stats, labels, height, width)
    return angle

def normalize(vecteur):
    norme = alg.norm(vecteur)
    if norme == 0:
        return vecteur
    return vecteur / norme


def eigen_elements(matrice):
    eigen_results = alg.eig(matrice)
    valeur_propre_1, valeur_propre_2, vecteur_propre_1, vecteur_propre_2 = eigen_results[0][0], eigen_results[0][1], eigen_results[1][0], eigen_results[1][1]
    vecteur_propre_1_normalized, vecteur_propre_2_normalized = (vecteur_propre_1), (vecteur_propre_2)
    if valeur_propre_2 > valeur_propre_1:
        return vecteur_propre_2_normalized, vecteur_propre_1_normalized
    return vecteur_propre_1_normalized, vecteur_propre_2_normalized


def get_inertia_matrix(moments):
    '''Permet d'obtenir la matrice d'inertie à partir des moments fournis par opencv dans un dictionnaire'''

    a_value = moments["mu20"]
    b_value = moments["mu02"]
    f_value = moments["mu11"]
    matrice_inertie = np.array([[a_value, -f_value], [-f_value, b_value]])
    return matrice_inertie


def get_distance_2_center(x_value, y_value, height, width):
    return (x_value - width / 2)**2 + (y_value - height / 2)**2


def get_orientation_from_data(centroids, num, stats, labels, height, width, img, image_tronquee) -> int:

    sorted_tuples = []

    #Stats of len num : num connected components detected by cv
    for i in range(1, num):     #The first one is the whole image
        #If the number of pixels is too low, just ignore
        if stats[i][-1] < MIN_PIXEL:
            continue
        distance = get_distance_2_center(centroids[i, 0], centroids[i, 1], height, width)
        sorted_tuples.append((i, distance))

    #Get the tuples (index of connected components, distance to center), now sorted by increasing distance
    sorted_tuples.sort(key=lambda x: x[1])
    sorted_tuples = np.array(sorted_tuples)

    #If no element, some parameters have to be tuned ... (Masking step)
    if len(sorted_tuples)==0:
        return np.NaN

    #Get the list of indexes, still sorted by increasing distance to center
    index_list = sorted_tuples[:, 0]
    index_list = index_list.astype(int)

    #For loop for test purpose (displaying multiple proposals, tune the MIN_PIXEL parameter zb)
    angle = 0
    for i in index_list[0:1]:
        #Get the mask with the coresponding connected component
        object_img = np.uint8(labels == i)

        contours, _ = cv2.findContours(object_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the convex hull object for each contour
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)

        # Draw contours + hull results
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(img, hull_list, -1, color, 5)

        #Display the masked image, debug purpose
        if DEBUG:
            object_img2 = cv2.bitwise_and(image_tronquee, image_tronquee, mask=object_img)
            cv2.imshow('imageTronquee', object_img2)
            cv2.waitKey(1000)

        #Get the moments and build the inertial matrix from it, and recover center coord of the connected component
        moment = cv2.moments(hull_list[0])
        if moment["m00"] != 0:
            x = int(moment["m10"] / moment["m00"])
            y = int(moment["m01"] / moment["m00"])
        matrice_inertie = get_inertia_matrix(moment)

        #Eigenelements are calculated, they allow us to recover the angle
        vecteur_propre_1_normalized, vecteur_propre_2_normalized = eigen_elements(matrice_inertie)

        theta = np.arctan2(vecteur_propre_2_normalized[1],vecteur_propre_2_normalized[0]) * 180 / np.pi
        angle = theta
        if vecteur_propre_2_normalized[1] > 0:
            aa = "SOUTH"
        else:
            aa = "NORTH"
        #Display the eigenvectors superposed on the image
        if VECTEURS:
            cv2.arrowedLine(img, (int(x), int(y)), (int(x + vecteur_propre_1_normalized[0]*100), int(y + vecteur_propre_1_normalized[1]*100)),  (0, 255, 255), 5)        
            cv2.arrowedLine(img, (int(x), int(y)), (int(x + vecteur_propre_2_normalized[0]*50), int(y + vecteur_propre_2_normalized[1]*50)),  (0, 255, 255), 5)
            cv2.putText(img, aa, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
            cv2.imshow('Debug vecteurs', img)
            cv2.waitKey(10000)

    return(aa)
    
def get_orientation_from_data_final(centroids, num, stats, labels, height, width) -> int:

    sorted_tuples = []

    #Stats of len num : num connected components detected by cv
    for i in range(1, num):     #The first one is the whole image
        #If the number of pixels is too low, just ignore
        if stats[i][-1] < MIN_PIXEL:
            continue
        distance = get_distance_2_center(centroids[i, 0], centroids[i, 1], height, width)
        sorted_tuples.append((i, distance))

    #Get the tuples (index of connected components, distance to center), now sorted by increasing distance
    sorted_tuples.sort(key=lambda x: x[1])
    sorted_tuples = np.array(sorted_tuples)

    #If no element, some parameters have to be tuned ... (Masking step)
    if len(sorted_tuples)==0:
        return np.NaN

    #Get the list of indexes, still sorted by increasing distance to center
    index_list = sorted_tuples[:, 0]
    index_list = index_list.astype(int)

    #For loop for test purpose (displaying multiple proposals, tune the MIN_PIXEL parameter zb)
    angle = 0
    for i in index_list[0:1]:
        #Get the mask with the coresponding connected component
        object_img = np.uint8(labels == i)

        contours, _ = cv2.findContours(object_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the convex hull object for each contour
        hull_list = []
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            hull_list.append(hull)

        #Get the moments and build the inertial matrix from it, and recover center coord of the connected component
        moment = cv2.moments(hull_list[0])
        if moment["m00"] != 0:
            x = int(moment["m10"] / moment["m00"])
            y = int(moment["m01"] / moment["m00"])
        matrice_inertie = get_inertia_matrix(moment)

        #Eigenelements are calculated, they allow us to recover the angle
        vecteur_propre_1_normalized, vecteur_propre_2_normalized = eigen_elements(matrice_inertie)

        if vecteur_propre_2_normalized[1] > 0:
            return 0
        else:
            return 1

if __name__ == "__main__":
    """Debug purpose script"""
    PATH = "tests/img/test4.jpg"
    print(get_orientation(PATH))
    #Load image
    img = cv2.imread(PATH)
    img = cv2.resize(img, (360, 640), interpolation=cv2.INTER_AREA)
    height, width = img.shape[0], img.shape[1]

    #Blur
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)

    #Filter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_v = np.array([0, 0, 0])
    upper_v = np.array([255, 255, 120])
    mask = cv2.inRange(hsv, lower_v, upper_v)
    aa = cv2.bitwise_not(img)

    #Display masked area (to tune the mask)
    image_tronquee = cv2.bitwise_and(aa, aa, mask=mask)
    cv2.imshow("Masque", image_tronquee)
    cv2.waitKey(1000)

    #Generate connected components, then calculate the angle
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    angle = get_orientation_from_data(centroids, num, stats, labels, height, width, img, image_tronquee)
    print(angle)
