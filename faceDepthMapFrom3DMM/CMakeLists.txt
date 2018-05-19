# CMAKE FILE : depthMapCreation/moennen/2017-12-19

CMAKE_MINIMUM_REQUIRED(VERSION 2.6) 

PROJECT(faceDepthMapFrom3DMM)
ADD_DEFINITIONS( -DCOMPILER_GCC -std=c++11 )
SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# GL_UTILS IMPORT/EXPORT
ADD_DEFINITIONS( -DWITH_ASSIMP )

# PREFORMANCES
#ADD_DEFINITIONS( -DHOP_ENABLED )

SET(MOENNEN_LOCAL /usr/local)
SET(MOENNEN_WKSPCE /home/moennen)
#SET(MOENNEN_LOCAL /mnt/p4/favila/moennen/local)
#SET(MOENNEN_WKSPCE /mnt/p4/avila/moennen_wkspce)
SET(CUDA /usr/local/cuda)
SET(M_INC ${MOENNEN_LOCAL}/include)
SET(M_LIBS ${MOENNEN_LOCAL}/lib)
SET(M_LIBS64 ${MOENNEN_LOCAL}/lib64)

SET(PROJROOT ../.)
SET(UTILS ${PROJROOT}/utils/)
SET(EXTERNALS ${PROJROOT}/externals)
SET(FACE ${EXTERNALS}/face)
SET(FACEMM ${FACE}/MMFaceModel)
SET(FACEMM_PRIV ${FACEMM}/MM_Restricted)


INCLUDE_DIRECTORIES(
	${MOENNEN_WKSPCE}/sceneIllEst
	${M_INC}
	${M_INC}/GL
	${M_INC}/eigen3
	${PROJROOT}
	${EXTERNALS}
)
LINK_DIRECTORIES(
	${M_LIBS}
	${M_LIBS64}
	${CUDA}/lib64
)

ADD_EXECUTABLE(createFaceMap faceDepthMapFrom3D_main.cpp 
	            ${FACE}/faceDetector.C 
	            ${FACE}/bFaceMModel.C
	            ${FACEMM}/face3dMLModelTracker.C
	            ${FACEMM_PRIV}/FileHandler.cpp
					${FACEMM_PRIV}/FileLoader.cpp
					${FACEMM_PRIV}/FileWriter.cpp
					${FACEMM_PRIV}/MultilinearModel.cpp
					${FACEMM_PRIV}/MultilinearModelHandler.cpp
	            ${UTILS}/gl_utils.C)

TARGET_LINK_LIBRARIES( createFaceMap GLEW GL SDL2 dlib openblas assimp opencv_world )
