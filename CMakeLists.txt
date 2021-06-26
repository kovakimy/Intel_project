cmake_minimum_required(VERSION 3.0.0)
project(AreaIntrusionDetection)
find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(${PROJECT_NAME} src/ExtendedTracker src/ReidDescriptor.cpp src/main.cpp src/ObjectDetector.cpp src/ReidNetwork.cpp src/ObjectTracker.cpp src/Context.cpp src/Drawer.cpp src/LineCrossesAndAreaIntrusionDetection.cpp
include/ReidNetwork.hpp include/ExtendedTracker include/ReidDescriptor.hpp include/ObjectTracker.hpp include/ObjectDetector.hpp include/Context.hpp include/Drawer.hpp include/LineCrossesAndAreaIntrusionDetection.hpp include/Kalman.hpp )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES})
