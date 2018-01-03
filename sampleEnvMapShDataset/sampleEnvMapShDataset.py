from ctypes import *
import numpy as np

library_path = "/home/moennen/sceneIllEst/sampleEnvMapShDataset/libsampleEnvMapShDataset.so"

class EnvMapShDataset(object):

    def __init__(self, dataPath, shOrder):

	self.__idx = 0 
        
	self.library=CDLL(library_path)
        
        self.__setDataPath = self.library.initEnvMapShDataSampler
        self.__setDataPath.restype = c_int
        self.__setDataPath.argtypes = [c_int,c_char_p,c_int]
        if self.__setDataPath(self.__idx, dataPath,shOrder) != 0 :
         raise NameError("Bad dataset")

	self.__getNbCameraParams = self.library.getEnvMapShNbCamParams
	self.__getNbCameraParams.restype = c_int
	self.__getNbCameraParams.argtypes = [c_int]
	self.nbCameraParams = self.__getNbCameraParams(self.__idx)
	
	self.__getNbShCoeffs = self.library.getEnvMapShNbCoeffs
	self.__getNbShCoeffs.restype = c_int
	self.__getNbShCoeffs.argtypes = [c_int]
	self.nbShCoeffs = self.__getNbShCoeffs(self.__idx)
        
        self.__sampleData = self.library.getEnvMapShDataSample
        self.__sampleData.restype = c_int 
        self.__sampleData.argtypes = [c_int, c_int, POINTER(c_float), POINTER(c_float), c_int, c_int, POINTER(c_float) ]


    def sampleData( self, dims ):
        
        imgData = np.zeros(dims[0]*dims[1]*dims[2]*3,np.float32)
	camData = np.zeros(dims[0]*self.nbCameraParams, np.float32) 
	shCoeffsData = np.zeros(dims[0]*self.nbShCoeffs*3, np.float32)
        err = self.__sampleData(self.__idx,dims[0],shCoeffsData.ctypes.data_as(POINTER(c_float)),camData.ctypes.data_as(POINTER(c_float)),dims[1],dims[2],imgData.ctypes.data_as(POINTER(c_float)))
        
        return [np.reshape(imgData,(dims[0],dims[1],dims[2],3)),np.reshape(shCoeffsData,(dims[0],self.nbShCoeffs*3)),np.reshape(camData,(dims[0],self.nbCameraParams))]  


 
