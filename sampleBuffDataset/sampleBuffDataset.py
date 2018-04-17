from ctypes import *
import numpy as np


class BufferDataSamplerLibrary(object):

    def __init__(self, library_path):
        self.library = CDLL(library_path)

        self.initDataset = self.library.initBuffersDataSampler
        self.initDataset.restype = c_int
        self.initDataset.argtypes = [c_int, c_char_p,
                                     c_char_p, c_int, POINTER(c_float), c_int]

        self.sampleData = self.library.getBuffersDataSample
        self.sampleData.restype = c_int
        self.sampleData.argtypes = [c_int, POINTER(c_float)]

        self.getNbBuffers = self.library.getNbBuffers
        self.getNbBuffers.restype = c_int
        self.getNbBuffers.argtypes = [c_int]

        self.getBuffersDim = self.library.getBuffersDim
        self.getBuffersDim.restype = c_int
        self.getBuffersDim.argtypes = [c_int, POINTER(c_float)]

        self.idx = 0

    def getNewSamplerIdx(self):
        self.idx = self.idx+1
        return self.idx-1


class BufferDataSampler(object):

    def __init__(self, sampler_library, datasetPath, dataPath, params, seed):

        self.sampler_library = sampler_library
        self.idx = sampler_library.getNewSamplerIdx()

        if sampler_library.initDataset(self.idx, datasetPath, dataPath,
                                       params.size, params.ctypes.data_as(POINTER(c_float)), seed) != 0:
            raise NameError("Bad dataset")

        self.batchSz = int(params[0])

        nbBuffers = sampler_library.getNbBuffers(self.idx)
        buffersDim = np.zeros(nbBuffers*3, dtype=np.float32)
        sampler_library.getBuffersDim(self.idx,
                                      buffersDim.ctypes.data_as(POINTER(c_float)))
        buffersDim = buffersDim.astype(int)
        self.buffersDim = np.array([(self.batchSz, buffersDim[i*3], buffersDim[i*3+1], buffersDim[i*3+2])
                                    for i in range(nbBuffers)])
        print self.buffersDim

        buffersSz = [np.prod(self.buffersDim[i][:]) for i in range(nbBuffers)]
        self.buffersEnd = np.cumsum(buffersSz)
        self.buffersBegin = np.append(0,  self.buffersEnd[0:-1])

        self.buffersData = np.zeros(self.buffersEnd[-1], dtype=np.float32)

        print buffersSz
        print np.prod(self.buffersDim)
        print self.buffersBegin
        print self.buffersEnd

    def getDataBuffers(self):

        # sample
        if self.sampler_library.sampleData(self.idx, self.buffersData.ctypes.data_as(POINTER(c_float))) != 0:
            raise NameError("Sample data error")

        # reshape
        return [np.reshape(self.buffersData[self.buffersBegin[i]:self.buffersEnd[i]], self.buffersDim[i][:])
                for i in range(self.buffersDim.shape[0])]
