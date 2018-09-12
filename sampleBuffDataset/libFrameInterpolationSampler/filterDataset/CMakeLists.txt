# CMAKE FILE : filterDataset/moennen/2018-09-07

CMAKE_MINIMUM_REQUIRED(VERSION 2.6) 

PROJECT(filterDataset)
ADD_DEFINITIONS( -DCOMPILER_GCC -std=gnu++0x )
SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(MOENNEN_LOCAL /mnt/p4/favila/moennen/local)
SET(MOENNEN_WKSPCE /mnt/p4/avila/moennen_wkspce)
SET(M_INC ${MOENNEN_LOCAL}/include)
SET(M_LIBS ${MOENNEN_LOCAL}/lib)
SET(M_LIBS64 ${MOENNEN_LOCAL}/lib64)
SET(CUDA /usr/local/cuda)

SET(PROJROOT ../../../.)
SET(UTILS ${PROJROOT}/utils/)

INCLUDE_DIRECTORIES(
	${MOENNEN_WKSPCE}
	${M_INC}
	${M_INC}/GL
	${M_INC}/eigen3
   ${PROJROOT}
)
LINK_DIRECTORIES(
	${M_LIBS}
	${M_LIBS64}
)

ADD_EXECUTABLE(filterDataset main.cpp ${UTILS}/cv_utils.C )

TARGET_LINK_LIBRARIES( filterDataset boost_system boost_filesystem opencv_world )

