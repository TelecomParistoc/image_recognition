# Explanation to this methods

- Method 1 : Aruco tag
The technique is simple : use the aruco tag in the middle of the compass.
The tag detection is a builtin opencv function, which is nice.

<img src="../doc/ill_m1.jpg" width="350">

- Method 2 : Compass color
To be simple, we use the compass black and white colors to determine its orientation. If the black part is up, then the compass indicates the north.
An homemade script calculates the inertia matrix of the black part and deduces the orientation.

<img src="../doc/ill_m2.jpg" width="350">


TODO: tests

TODO: calibration of second one ?