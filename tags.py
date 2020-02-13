import numpy as np
from cv2 import aruco
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

"""
fig = plt.figure()
nx = 4
ny = 3
for i in range(1, nx*ny+1):
    ax = fig.add_subplot(ny,nx, i)
    img = aruco.drawMarker(aruco_dict,i, 700)
    plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")


plt.savefig("arkers.pdf")
plt.show()

"""
capture = cv2.VideoCapture(0)

while 1:
    #read_flag, frame = capture.read()
    frame = cv2.imread("./test55.jpg")
    #Floutage
    kernel = np.ones((8,8),np.float32)/64
    frame = cv2.filter2D(frame,-1,kernel)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(100, 0, 240))
    for corner in corners:
        point1 = tuple(corner[0][0])
        point2 = tuple(corner[0][1])
        cv2.arrowedLine(frame, point1, point2, color=(255, 0, 0), thickness=4)
        x, y = np.array(point1)-np.array(point2)
        theta = np.arctan2(x, y)*180/np.pi
        print(theta)
    cv2.imshow("Lol", frame)
    cv2.waitKey(1)