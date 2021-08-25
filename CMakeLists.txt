cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sold_impl)

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

add_compile_options(-std=c++14)

set(CMAKE_PREFIX_PATH "/home/buenos/slam/project/pytorch_cpp_tutorial/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV 3.2.0 REQUIRED)
if(NOT OpenCV_FOUND)
   message(FATAL_ERROR "OpenCV == 3.2.0 not found.")
endif()

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
