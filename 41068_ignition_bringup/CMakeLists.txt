cmake_minimum_required(VERSION 3.8)
project(41068_ignition_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ros_ign_interfaces REQUIRED)

# Install the package resources. "weights" is added for the trained YOLO model.
install(
  DIRECTORY config launch models urdf_husky worlds urdf_parrot weights
  DESTINATION share/${PROJECT_NAME}
)

# Install executable Python nodes.
install(
  PROGRAMS
    scripts/dynamic_world_demo.py
    scripts/basic_autonomy_demo.py
    scripts/animal_yolo_detector.py
    scripts/thermal_candidate_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

# Imported by the perception executables. Keeping this as a regular file avoids
# exposing it as a ros2 run entry point.
install(
  FILES
    scripts/perception_common.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
