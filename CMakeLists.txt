cmake_minimum_required(VERSION 3.0.0)
project(AreaIntrusionDetection)
find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(${PROJECT_NAME} src/main.cpp src/detector.cpp src/reid_network.cpp src/ObjectTracker.cpp src/Context.cpp src/Drawer.cpp src/LineCrossesAndAreaIntrusionDetection.cpp
include/reid_network.hpp include/ObjectTracker.hpp include/detector.hpp include/Context.hpp include/Drawer.hpp include/LineCrossesAndAreaIntrusionDetection.hpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES})
