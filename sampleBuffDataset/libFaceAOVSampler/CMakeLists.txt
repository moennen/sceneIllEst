# CMAKE FILE : sampleEnvMapShDataset/moennen/2017-12-20

CMAKE_MINIMUM_REQUIRED(VERSION 2.6) 

PROJECT(FaceAOVSampler)
ADD_DEFINITIONS( -DCOMPILER_GCC -std=gnu++0x -O3 )
SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# PREFORMANCES
#ADD_DEFINITIONS( -DHOP_ENABLED )

#SET(MOENNEN_LOCAL /usr/local)
#SET(MOENNEN_WKSPCE /home/moennen)
SET(MOENNEN_LOCAL /mnt/p4/favila/moennen/local)
SET(MOENNEN_WKSPCE /mnt/p4/avila/moennen_wkspce/sceneIllEst)
SET(M_INC ${MOENNEN_LOCAL}/include)
SET(M_LIBS ${MOENNEN_LOCAL}/lib)
SET(M_LIBS64 ${MOENNEN_LOCAL}/lib64)

SET(PROJROOT ../../.)
SET(UTILS ${PROJROOT}/utils/)

INCLUDE_DIRECTORIES(
	${MOENNEN_WKSPCE}
	${M_INC}
	${M_INC}/eigen3
	${PROJROOT}
)

LINK_DIRECTORIES(
	${M_LIBS}
	${M_LIBS64}
)

ADD_LIBRARY(FaceAOVSampler SHARED libSampler.cpp
				${UTILS}/cv_utils.C )

SET_PROPERTY(TARGET FaceAOVSampler PROPERTY POSITION_INDEPENDENT_CODE ON)

TARGET_LINK_LIBRARIES( FaceAOVSampler
	                    boost_thread boost_system boost_filesystem
	                    opencv_world
	                    rt dl )

