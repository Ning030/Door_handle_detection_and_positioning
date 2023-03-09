cmake_minimum_required(VERSION 3.5.1)
project(libtorch-yolov5)

set(CMAKE_CXX_STANDARD 14)
# It prevents the decay to C++98 when the compiler does not support C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# It disables the use of compiler-specific extensions
# e.g. -std=c++14 rather than -std=gnu++14
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


# Try to find OpenCV
# set(OpenCV_DIR ....)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
MESSAGE("OpenCV version:"  ${OpenCV_VERSION})
MESSAGE("OpenCV INCLUDE DIRS:"  ${OpenCV_INCLUDE_DIRS})
MESSAGE("OpenCV LIBRARY DIRS:"  ${OpenCV_LIB_DIRS})



set(Torch_DIR /home/ning/libtorch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCE_FILES src/*.cpp)

add_executable(yolo ${SOURCE_FILES})

target_link_libraries (
        yolo
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)
