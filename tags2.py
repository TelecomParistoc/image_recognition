import numpy as np
import numpy.linalg as alg
import cv2
import time
#TODO: fix debug
DEBUG = 0
VECTEURS = 1

def normalize(vecteur):
    norme = alg.norm(vecteur)
    if norme == 0:
        return vecteur
    return vecteur / norme


def eigen_elements(matrice):
    eigen_results = alg.eig(matrice)
    valeur_propre_1, valeur_propre_2, vecteur_propre_1, vecteur_propre_2 = eigen_results[0][0], eigen_results[0][1], eigen_results[1][0], eigen_results[1][1]
    vecteur_propre_1_normalized, vecteur_propre_2_normalized = normalize(vecteur_propre_1), normalize(vecteur_propre_2)
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


def get_distance_2_center(x_value, y_value):
    return (x_value - width / 2)**2 + (y_value - height / 2)**2


def get_angle_list(centroids, num, stats, labels):
    sorted_tuples = []
    for i in range(1, num):
        distance = get_distance_2_center(centroids[i, 0], centroids[i, 1])
        sorted_tuples.append((i, distance))
    #Permet de trier la liste selon le 2ème paramètre
    sorted_tuples.sort(key=lambda x: x[1])
    
    sorted_tuples = np.array(sorted_tuples)
    index_list = sorted_tuples[:, 0]
    index_list = index_list.astype(int)

    angle = 0
    for i in index_list:
        object_img = np.uint8(labels == i)
        if DEBUG:
            object_img2 = cv2.bitwise_and(image_tronquee, image_tronquee, mask=object_img)
            cv2.imshow('imageTronquee',object_img2)
            cv2.waitKey(2000)
        moment = cv2.moments(object_img, True)
        if moment["m00"] != 0:
            x = int(moment["m10"] / moment["m00"])
            y = int(moment["m01"] / moment["m00"])
        matrice_inertie = get_inertia_matrix(moment)
        vecteur_propre_1_normalized, vecteur_propre_2_normalized = eigen_elements(matrice_inertie)

        theta = np.arctan2(vecteur_propre_1_normalized[1],vecteur_propre_1_normalized[0]) * 180 / np.pi
        if VECTEURS:
            #TODO: couleur vecteurs
            cv2.arrowedLine(mask, (int(x), int(y)), (int(x + vecteur_propre_1_normalized[0]*100), int(y+vecteur_propre_1_normalized[1]*100)),  (0, 255, 255), 10)        
            cv2.arrowedLine(mask, (int(x), int(y)), (int(x + vecteur_propre_2_normalized[0]*50), int(y+vecteur_propre_2_normalized[1]*50)),  (0, 255, 255), 10)
            cv2.imshow('imageTronquee',mask)
            cv2.waitKey(5000)
        angle = theta
    return(angle)
    

'''
capture = cv2.VideoCapture(0)
while capture.isOpened():
'''
img = cv2.imread("test55.jpg")
height, width = img.shape[0], img.shape[1]

#Downsampling
W = width/2
imgScale = W/width
newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
img = cv2.resize(img,(int(newX),int(newY)))

#Floutage
kernel = np.ones((5,5),np.float32)/25
img = cv2.filter2D(img,-1,kernel)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_v = np.array([0, 0, 0])
upper_v = np.array([255, 255, 45])
mask = cv2.inRange(hsv, lower_v, upper_v)
#image_tronquee = cv2.bitwise_and(img, img, mask=mask)    
num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
angle = get_angle_list(centroids, num, stats, labels)
print(angle)