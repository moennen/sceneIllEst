CMAKE_MINIMUM_REQUIRED(VERSION 3.1) 

PROJECT(filterImgLst)
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

SET(LOCAL_DIR /usr/local CACHE PATH "System local path")

SET(BOOST_DIR ${LOCAL_DIR} CACHE PATH "Boost path")
SET(EIGEN_DIR ${LOCAL_DIR} CACHE PATH "Eigen path")
SET(OPENCV_DIR ${LOCAL_DIR} CACHE PATH "OpenCV path")

SET(FLUTILS_SRC_DIR ${LOCAL_DIR} CACHE PATH "flutils source path")

INCLUDE_DIRECTORIES(
  ${LOCAL_DIR}/include
  ${BOOST_DIR}/include
  ${EIGEN_DIR}
  ${OPENCV_DIR}/include
  ${FLUTILS_DIR}/include
  .
)

LINK_DIRECTORIES(
  ${LOCAL_DIR}/lib
  ${LOCAL_DIR}/lib64
  ${BOOST_DIR}/lib
  ${OPENCV_DIR}/lib64
  ${FLUTILS_DIR}/lib
)

ADD_EXECUTABLE( filterImgLst main.cpp )

TARGET_LINK_LIBRARIES( filterImgLst
	                   boost_thread boost_system boost_filesystem
	                   flutils opencv_world )