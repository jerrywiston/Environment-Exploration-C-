cmake_minimum_required(VERSION 2.8)

project(GridSLAM)
find_package(OpenMP REQUIRED)
find_package(Python 3.6 COMPONENTS Development REQUIRED)
message(Error ${PYTHON_INCLUDE_DIRS})
add_subdirectory(pybind11)

add_definitions(-DUSE_DOUBLE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#add_definitions(-DWITH_OPENCV)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(NO_OUTPUT_DIRS)

file(GLOB_RECURSE SOURCE src/*.cpp)
file(GLOB_RECURSE HEADER inc/*.h src/*.h)

add_library(GridSLAM STATIC ${SOURCE} ${HEADER})
target_include_directories(GridSLAM PUBLIC inc)
target_link_libraries(GridSLAM ${CONAN_LIBS})
target_link_libraries(GridSLAM PRIVATE OpenMP::OpenMP_CXX)

project(gslam)
pybind11_add_module(gslam python/gslam.cpp)
target_link_libraries(gslam GridSLAM)
