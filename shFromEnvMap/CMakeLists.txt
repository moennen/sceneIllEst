# CMAKE FILE : shFromEnvMap/moennen/2017-12-07

CMAKE_MINIMUM_REQUIRED(VERSION 2.6) 

PROJECT(shFromEnvMap)
ADD_DEFINITIONS( -DCOMPILER_GCC -std=gnu++0x -O3 )
SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

SET(MOENNEN_LOCAL /mnt/p4/favila/moennen/local)
SET(MOENNEN_WKSPCE /mnt/p4/avila/moennen_wkspce)
SET(M_INC ${MOENNEN_LOCAL}/include)
SET(M_LIBS ${MOENNEN_LOCAL}/lib)
SET(M_LIBS64 ${MOENNEN_LOCAL}/lib64)
SET(CUDA /usr/local/cuda-8.0)

SET(PROJROOT ../.)
SET(UTILS ${PROJROOT}/utils/)
SET(EXTERNALS ${PROJROOT}/externals)
SET(SPH ${EXTERNALS}/sh)


INCLUDE_DIRECTORIES(
	${MOENNEN_WKSPCE}
	${M_INC}
	${M_INC}/GL
	${M_INC}/GLFW
	${M_INC}/eigen3
	${CUDA}/include
	${PROJROOT}
	${EXTERNALS}
)
LINK_DIRECTORIES(
	${M_LIBS}
	${M_LIBS64}
	${CUDA}/lib64
)

ADD_EXECUTABLE( shFromEnvMap shFromEnvMap_main.cpp 
                ${SPH}/spherical_harmonics.cc)

TARGET_LINK_LIBRARIES( shFromEnvMap
	boost_thread boost_system boost_filesystem leveldb opencv_world )

