#pylint:disable=E1101,W0621
from typing import Union
import numpy as np
from cv2 import aruco
import cv2

def get_orientation(input_image : Union[str, np.ndarray]) -> int:
    """Get the orientation from an image

    :param input: path of the image taken by the camera OR frame imported via imread
    :type input: Union[str, np.ndarray]
    :return: orientation (1 = NORTH / 0 = SOUTH)
    :rtype: int
    """

    if isinstance(input_image, str):
        frame = cv2.imread(input_image)
    else:
        frame = input_image

    frame = cv2.resize(frame, (360, 640), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Change the DICT to 6x6 if doesnt work
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    parameters =  aruco.DetectorParameters_create()
    corners, _, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
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
    """Debug purpose code"""
    path = "./tests/img/test4.jpg"
    #Result of main code
    print(get_orientation(path))
    #Debug code
    """ Piece to take a photo, just in case
    cam = cv2.VideoCapture(0)
    retval, frame = cam.read()
    """
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (360, 640), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Change the DICT to 6x6 if it doesnt work ...
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
    #Detector
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    #Corners : aruco detected
    #Rejected : detected but rejected
    #Print these detected tags
    aruco.drawDetectedMarkers(frame, corners, ids)
    aruco.drawDetectedMarkers(frame, rejectedImgPoints, borderColor=(100, 0, 240))
    #Take the first detected (shouldnt be a problem), and print the detection
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
        cv2.putText(frame, aa, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
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
        cv2.putText(frame, aa, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    cv2.imshow("Lol", frame)
    cv2.waitKey(1000)
    input()