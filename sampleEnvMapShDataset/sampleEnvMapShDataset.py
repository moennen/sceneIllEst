from ctypes import *
import numpy as np

# library_path =
# "/home/moennen/sceneIllEst/sampleEnvMapShDataset/libsampleEnvMapShDataset.so"
library_path = "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleEnvMapShDataset/libsampleEnvMapShDataset.so"


class EnvMapShDataset(object):

    library = CDLL(library_path)

    __computeNbShCoeffs = library.getNbShCoeffs
    __computeNbShCoeffs.restype = c_int
    __computeNbShCoeffs.argtypes = [c_int]

    __loadImg = library.getImgFromFile
    __loadImg.restype = c_int
    __loadImg.argtypes = [c_char_p, POINTER(c_float), c_int, c_int, c_int]

    __generateEnvMap = library.getEnvMapFromCoeffs
    __generateEnvMap.restype = c_int
    __generateEnvMap.argtypes = [c_int, POINTER(
        c_float), POINTER(c_float), c_int, c_int]

    __setDataPath = library.initEnvMapShDataSampler
    __setDataPath.restype = c_int
    __setDataPath.argtypes = [c_int, c_char_p, c_char_p, c_int, c_int, c_int]

    __getNbCameraParams = library.getEnvMapShNbCamParams
    __getNbCameraParams.restype = c_int
    __getNbCameraParams.argtypes = [c_int]

    __getNbShCoeffs = library.getEnvMapShNbCoeffs
    __getNbShCoeffs.restype = c_int
    __getNbShCoeffs.argtypes = [c_int]

    __sampleData = library.getEnvMapShDataSample
    __sampleData.restype = c_int
    __sampleData.argtypes = [c_int, c_int, POINTER(
        c_float), POINTER(c_float), c_int, c_int, POINTER(c_float)]

    __idx = 0

    @staticmethod
    def nbShCoeffs(shOrder):
        return EnvMapShDataset.__computeNbShCoeffs(shOrder)

    @staticmethod
    def loadImg(imgFilename, sz, linearCS):
        imgData = np.zeros(sz[0]*sz[1]*3, np.float32)
        err = EnvMapShDataset.__loadImg(c_char_p(
            imgFilename), imgData.ctypes.data_as(POINTER(c_float)), sz[1], sz[0], linearCS)
        return np.reshape(imgData, (1, sz[0], sz[1], 3))

    @staticmethod
    def generateEnvMap(shOrder, shCoeffsData, sz):
        imgData = np.zeros(sz[0]*sz[1]*3, np.float32)
        err = EnvMapShDataset.__generateEnvMap(
            shOrder,  shCoeffsData.ctypes.data_as(POINTER(c_float)), imgData.ctypes.data_as(POINTER(c_float)), sz[1], sz[0])
        return np.reshape(imgData, (1, sz[0], sz[1], 3))

    def __init__(self, dataPath, imgPath, shOrder, seed, linearCS):

        self.__idx = EnvMapShDataset.__idx
        EnvMapShDataset.__idx += 1

        if EnvMapShDataset.__setDataPath(self.__idx, dataPath, imgPath, shOrder, seed, linearCS) != 0:
            raise NameError("Bad dataset")

        self.nbCameraParams = EnvMapShDataset.__getNbCameraParams(self.__idx)

        self.nbShCoeffs = EnvMapShDataset.__getNbShCoeffs(self.__idx)

    def sampleData(self, dims):

        imgData = np.zeros(dims[0]*dims[1]*dims[2]*3, np.float32)
        camData = np.zeros(dims[0]*self.nbCameraParams, np.float32)
        shCoeffsData = np.zeros(dims[0]*self.nbShCoeffs*3, np.float32)
        err = EnvMapShDataset.__sampleData(self.__idx, dims[0], shCoeffsData.ctypes.data_as(POINTER(c_float)), camData.ctypes.data_as(
            POINTER(c_float)), dims[2], dims[1], imgData.ctypes.data_as(POINTER(c_float)))

        return [np.reshape(imgData, (dims[0], dims[1], dims[2], 3)), np.reshape(shCoeffsData, (dims[0], self.nbShCoeffs*3)),
                np.reshape(camData, (dims[0], self.nbCameraParams))]
