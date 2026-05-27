cmake_minimum_required(VERSION 3.8)
project(weld_scanner)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(cv_bridge REQUIRED)

# Ép CMake dùng bản OpenCV hệ thống (bản đầy đủ)
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

message(STATUS "FIND OpenCV VERSION: ${OpenCV_VERSION}")
message(STATUS "FIND OpenCV LIBS: ${OpenCV_LIBRARIES}")

set(HIK_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(SCANNER_LIB_DIR "${PROJECT_SOURCE_DIR}/libs")

include_directories(
  ${HIK_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${SCANNER_LIB_DIR})

add_executable(scanner_node src/scanner_node.cpp)

# Link cực mạnh: iRoVi trước, OpenCV sau, và ép thêm các module lẻ
target_link_libraries(scanner_node
  libiRoVi_3D_scanner_core.so
  libiRoVi_FrameGrabber.so
  libiRoVi_ProjectorConnector.so
  libiRoVi_SL3DReconstruction.so
  ${OpenCV_LIBRARIES}
  opencv_calib3d
  opencv_core
  opencv_imgproc
  opencv_highgui
)

ament_target_dependencies(scanner_node
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
  cv_bridge
)

install(TARGETS
  scanner_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
