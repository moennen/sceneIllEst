# CMAKE FILE : flowMapCreation/moennen/2018-10-02

CMAKE_MINIMUM_REQUIRED(VERSION 2.6) 

PROJECT(flowMapCreation)
SET(CMAKE_CXX_STANDARD 14)
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
SET(MOENNEN_WKSPCE /mnt/p4/avila/moennen_wkspce)
SET(M_INC ${MOENNEN_LOCAL}/include)
SET(M_LIBS ${MOENNEN_LOCAL}/lib)
SET(M_LIBS64 ${MOENNEN_LOCAL}/lib64)

SET(PROJROOT ../.)
SET(UTILS ${PROJROOT}/utils/)
SET(EXTERNALS ${PROJROOT}/externals)

INCLUDE_DIRECTORIES(
	${PROJROOT}
   ${EXTERNALS}
	${M_INC}
	${M_INC}/eigen3
   ${M_INC}/varOpticalFlowCL
)
LINK_DIRECTORIES(
	${M_LIBS}
	${M_LIBS64}
)

ADD_EXECUTABLE(genIntegratedFlowMap  genIntegratedFlowMap.cpp ${UTILS}/cv_utils.C )
TARGET_LINK_LIBRARIES( genIntegratedFlowMap boost_system boost_filesystem VarOpticalFlowCL opencv_world )
