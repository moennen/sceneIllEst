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

#SET(MOENNEN_LOCAL /usr/local)
#SET(MOENNEN_WKSPCE /home/moennen)
SET(MOENNEN_LOCAL /mnt/p4/favila/moennen/local)
SET(MOENNEN_WKSPCE /mnt/p4/avila/moennen_wkspce)
SET(CUDA /usr/local/cuda)
SET(M_INC ${MOENNEN_LOCAL}/include)
SET(M_LIBS ${MOENNEN_LOCAL}/lib)
SET(M_LIBS64 ${MOENNEN_LOCAL}/lib64)

SET(PROJROOT ../.)
SET(UTILS ${PROJROOT}/utils/)
SET(EXTERNALS ${PROJROOT}/externals)
SET(FACE ${EXTERNALS}/face)
SET(FACEMM /mnt/p4/flameBGD/MMFaceModel)

INCLUDE_DIRECTORIES(
	${MOENNEN_WKSPCE}/sceneIllEst
	${M_INC}
	${M_INC}/GL
	${M_INC}/eigen3
	${PROJROOT}
	${EXTERNALS}
	${FACEMM}
)
LINK_DIRECTORIES(
	${M_LIBS}
	${M_LIBS64}
	${CUDA}/lib64
)

ADD_EXECUTABLE(createFaceMap faceDepthMapFrom3D_main.cpp 
	            ${FACE}/faceDetector.C 
	            ${FACE}/bFaceMModel.C
	            ${FACE}/beFaceMModel.C
	            ${FACE}/beFaceModelInstance.C
	            ${FACE}/face3dMLModelTracker.C
	            ${FACEMM}/MM_Restricted/FileHandler.cpp
					${FACEMM}/MM_Restricted/FileLoader.cpp
					${FACEMM}/MM_Restricted/FileWriter.cpp
					${FACEMM}/MM_Restricted/MultilinearModel.cpp
					${FACEMM}/MM_Restricted/MultilinearModelHandler.cpp
	            ${UTILS}/gl_utils.C)

TARGET_LINK_LIBRARIES( createFaceMap GLEW GL SDL2 cudnn cusolver curand cudart cublas dlib openblas assimp opencv_world )

ADD_EXECUTABLE(createFaceMapsFromBEInst faceMapsFromBEInst_main.cpp
	            ${FACE}/beFaceMModel.C
	            ${FACE}/beFaceModelInstance.C
	            ${UTILS}/gl_utils.C
	            ${UTILS}/cv_utils.C)

TARGET_LINK_LIBRARIES( createFaceMapsFromBEInst GLEW GL SDL2 assimp opencv_world )

ADD_EXECUTABLE(renderFaceMaps renderFaceMaps.cpp 
	            ${FACE}/beFaceMModel.C
	            ${UTILS}/gl_utils.C
	            ${UTILS}/cv_utils.C)

TARGET_LINK_LIBRARIES( renderFaceMaps GLEW GL SDL2 assimp opencv_world boost_system boost_filesystem )
