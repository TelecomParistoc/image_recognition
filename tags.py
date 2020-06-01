import numpy as np
from cv2 import aruco
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_orientation(path : str) -> int:
    """Get the orientation from a saved image

    :param path: path of the image taken by the camera
    :type path: str
    :return: orientation (1 = NORTH / 0 = SOUTH)
    :rtype: int
    """

    frame = cv2.imread(path)
    frame = cv2.resize(frame, (360, 640), interpolation = cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Change the DICT to 6x6 if doesnt work
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if len(corners) != 0:
        corner = corners[0]
        point1 = tuple(corner[0][1])
        point2 = tuple(corner[0][2])
        _, y = np.array(point2)-np.array(point1)
        if y < 0:
            return 0
        else:
            return 1

if __name__ == "__main__":
    path = "./test4.jpg"
    print(get_orientation(path))
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (360, 640), interpolation = cv2.INTER_AREA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Change the DICT to 6x6 if doesnt work ...
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame, corners, ids)
    aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(100, 0, 240))
    if len(corners) != 0:
        corner = corners[0]
        point1 = tuple(corner[0][1])
        point2 = tuple(corner[0][2])
        cv2.arrowedLine(frame, point1, point2, color=(255, 0, 0), thickness=4)
        _, y = np.array(point2)-np.array(point1)
        if y < 0:
            aa = "SOUTH"
        else:
            aa = "NORTH"
        cv2.putText(frame, aa, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    else:
        corner = rejectedImgPoints[1]
        point1 = tuple(corner[0][1])
        point2 = tuple(corner[0][2])
        cv2.arrowedLine(frame, point1, point2, color=(255, 0, 0), thickness=4)
        _, y = np.array(point2)-np.array(point1)
        if y < 0:
            aa = "SOUTH(R)"
        else:
            aa = "NORTH(R)"
        cv2.putText(frame, aa, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    cv2.imshow("Lol", frame)
    cv2.waitKey(1000)
    input()