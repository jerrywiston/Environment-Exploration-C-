cmake_minimum_required(VERSION 2.8)

project(GridSLAM)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

file(GLOB_RECURSE SOURCE src/*.cpp)
file(GLOB_RECURSE HEADER inc/*.h src/*.h)

add_executable(GridSLAM ${SOURCE} ${HEADER})
target_include_directories(GridSLAM PUBLIC inc)
target_link_libraries(GridSLAM ${CONAN_LIBS})
