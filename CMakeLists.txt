cmake_minimum_required(VERSION 3.10)
project(BundleAdjustment)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(CURL REQUIRED)
find_package(TIFF REQUIRED)
#find_package(GDAL REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/Libs/Eigen)

# Add executable
add_executable(bundle_adjustment bundle_adjustment.cpp)

# Link libraries
target_link_libraries(bundle_adjustment PRIVATE 
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES}
    ${CURL_LIBRARIES} 
    ${TIFF_LIBRARIES} 
#    ${GDAL_LIBRARIES}
)
