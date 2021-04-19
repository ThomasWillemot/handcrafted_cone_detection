cmake_minimum_required(VERSION 3.0.2)

project(handcrafted_cone_detection)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge #was commented out
  sensor_msgs
  geometry_msgs
  nav_msgs
  image_transport
  message_generation
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system signals thread)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS}
                           ${catkin_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
)
## Add the needed messages
add_message_files(
  FILES
  ConeImgLoc.msg
)

## Generate services in the 'srv' folder
add_service_files(
    FILES
    SendRelCor.srv
)


## Generate added messages and services with any dependencies listed here
generate_messages(
    DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
)

catkin_package(
   LIBRARIES handcrafted_cone_detection
   CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

include_directories(
 ${catkin_INCLUDE_DIRS}
 )

catkin_install_python(PROGRAMS
  src/waypoint_extractor.py
  src/waypoint_estimator_nn.py
  src/camera_visualiser.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
