from ctypes import *
import numpy as np

# library_path =
# "/home/moennen/sceneIllEst/sampleImageShDataset/libsampleImageShDataset.so"
library_path = "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleImageShDataset/libsampleImageShDataset.so"


class GenBuffersDataset(object):

    __library = 0
    __setDataPath = 0
    __sampleData = 0

    __idx = 0

    @staticmethod
    def init(library_path):
       
       __library = CDLL(library_path)

       __setDataPath = library.initBuffersDataSampler
       __setDataPath.restype = c_int
       __setDataPath.argtypes = [c_int, c_char_p, c_char_p, c_int, c_int, c_int]
    
       __sampleData = library.getBuffersDataSample
       __sampleData.restype = c_int
       __sampleData.argtypes = [c_int, c_int, POINTER(
        c_float), POINTER(c_float), c_int, c_int, POINTER(c_float)]

    def __init__(self, datasetPath, dataPath, seed):

        self.__idx = GenBuffersDataset.__idx
        GenBuffersDataset.__idx += 1

        if GenBuffersDataset.__setDataPath(self.__idx, datasetPath, dataPath, seed) != 0:
            raise NameError("Bad dataset")

        self.nbCameraParams = ImageShDataset.__getNbCameraParams(self.__idx)

        self.nbShCoeffs = ImageShDataset.__getNbShCoeffs(self.__idx)

    def getSample(self, dims):

        imgData = np.zeros(dims[0]*dims[1]*dims[2]*3, np.float32)
        camData = np.zeros(dims[0]*self.nbCameraParams, np.float32)
        shCoeffsData = np.zeros(dims[0]*self.nbShCoeffs*3, np.float32)
        err = ImageShDataset.__sampleData(self.__idx, dims[0], shCoeffsData.ctypes.data_as(POINTER(c_float)), camData.ctypes.data_as(
            POINTER(c_float)), dims[2], dims[1], imgData.ctypes.data_as(POINTER(c_float)))

        if err != 0 :
            raise NameError("Incompatible sample")

        return [np.reshape(imgData, (dims[0], dims[1], dims[2], 3)), np.reshape(shCoeffsData, (dims[0], self.nbShCoeffs*3)),
                np.reshape(camData, (dims[0], self.nbCameraParams))]
