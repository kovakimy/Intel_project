cmake_minimum_required(VERSION 3.0.0)
project(AreaIntrusionDetection)

find_package(ngraph          REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV     	 REQUIRED)

add_executable(${PROJECT_NAME} src/ExtendedTracker src/ReidDescriptor.cpp src/main.cpp src/ObjectDetector.cpp src/ReidNetwork.cpp src/Context.cpp src/Drawer.cpp src/LineCrossesAndAreaIntrusionDetection.cpp
include/ReidNetwork.hpp include/ExtendedTracker include/ReidDescriptor.hpp include/ObjectDetector.hpp include/Context.hpp include/Drawer.hpp include/LineCrossesAndAreaIntrusionDetection.hpp include/Kalman.hpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src
										   PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES})
message("TEST" ${OpenCV_DIR})
add_custom_command( TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${OpenCV_DIR}"/bin/Debug/opencv_world453d.dll"
		${AreaIntrusionDetection_BINARY_DIR}"/Debug")