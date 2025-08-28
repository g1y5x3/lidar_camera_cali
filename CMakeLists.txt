cmake_minimum_required(VERSION 3.8)
project(lidar_camera_cali)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(apriltag REQUIRED)

add_executable(lidar_camera_cali_node src/lidar_camera_cali.cpp)

target_include_directories(lidar_camera_cali_node PUBLIC
  ${apriltag_INCLUDE_DIRS}
)

ament_target_dependencies(lidar_camera_cali_node
  rclcpp
  sensor_msgs
  cv_bridge
)

target_link_libraries(lidar_camera_cali_node
  ${OpenCV_LIBRARIES}
  ${apriltag_LIBRARIES}
)

install(TARGETS
  lidar_camera_cali_node
  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
