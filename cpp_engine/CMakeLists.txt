cmake_minimum_required(VERSION 3.10)
project(AquaEye_Degradation_Engine)

set(CMAKE_CXX_STANDARD 17)

# --- THE FIX: Point deeper to the specific vc16/lib folder ---
# This bypasses the version check that is failing.
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(degradation_engine main.cpp DegradationEngine.cpp)
target_link_libraries(degradation_engine ${OpenCV_LIBS})