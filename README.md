## Project Description

The goal of this project is to identify and locate the safe door handle for subsequent robot automatic grabbing and opening. 

The project is divided into two parts: 

1. Door handle recognition: Based on yolov5 method to perform initial recognition of door handle. 
2. Keyhole and door handle positioning: Based on RGB-D obtained depth information and contour recognition and LSD line detection algorithm to obtain three-dimensional pose of keyhole and door handle.

![reslut](https://raw.githubusercontent.com/Ning030/git_test/master/result.png)

## Project Structure

The project is structured as follows:

- `color_result` -> Keyhole detection result
- `depth` -> depth raw data
- `images`  -> rgb raw data
- `include`  -> has the header files

- `src` -> has the source files
- `weights` -> yolov5 weight file

- CMakeLists.txt -> It is the CMake file for the project



## How to Run the project

1、First of all, we need to configure our pytorch：https://pytorch.org/cppdocs/installing.html

2、Then, we need to modify the file path in main.cpp.

## Dependencies

- Ubuntu 20.04
- OpenCV 3.4.12
- LibTorch 1.6.0
- Eigen



## References

1. https://github.com/ultralytics/yolov5
2. [Question about the code in non_max_suppression](https://github.com/ultralytics/yolov5/issues/422)
3. https://github.com/walktree/libtorch-yolov3
4. https://pytorch.org/cppdocs/index.html
5. https://github.com/pytorch/vision

