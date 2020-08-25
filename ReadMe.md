# Orientation detector

For the Coupe de Robotique, we need to determine the orientation of a compass.

## Description

Here is a description of the compass :

<img src="doc/gir.png" width="350">

This program uses 2 different methods to determine the orientation :
- the first one uses the aruco tag in the middle
- the first one uses the black half-circle

Details can be found in the src folder.

## Setup for Rpi

1. Install pip if necessary
```
sudo apt install python3-pip -y
```

2. Install the python requirements
```
pip3 install - requirements.txt
```

3. Install other dependencies
```
sudo apt install libatlas-base-dev
sudo apt install libhdf5-dev
sudo apt install libhdf5-serial-dev
sudo apt install libcblas-dev
sudo apt install libatlas-base-dev
sudo apt install libjasper-dev 
sudo apt install libqtgui4 
sudo apt install libqt4-test
sudo apt install libilmbase23
sudo apt install libopenexr23
sudo apt install libavcodec-extra58
sudo apt install libavformat58
sudo apt install libswscale5
sudo apt install libharfbuzz0b
```

## Usage :

The scripts are in **Python 3**.

To put in zshrc :
```
export LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1
```

TODO: usage & install