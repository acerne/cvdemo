# Computer vision demo

A small Python project that shows a demonstration of common image processing methods used in computer vision.  
Based on OpenCV image processing library.

## Prerequisite installation
:memo: _Ubuntu-based_

OpenCV library and Python3 essentials.
```bash
sudo apt update
sudo apt install libopencv-dev python3-opencv python3-pip
```
Libraries used in Python scripts
```bash
pip3 install numpy
pip3 install matplotlib
pip3 install argparse
```

## Execution

To show the results
```bash
python3 filters/average.py
```

To save result images without showing
```bash
python3 filters/average.py -s
```