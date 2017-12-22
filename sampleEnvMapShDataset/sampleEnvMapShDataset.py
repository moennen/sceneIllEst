from ctypes import *


class EnvMapShDataset(object):

    def __init__(self, dataPath):

        self.library=CDLL("libsampleEnvMapShDataset.so")
        
        self.setDataPath = self.library.setEnvMapShDataset
        self.setDataPath.restype = c_int
        self.setDataPath.argtypes = [charptr]
        if self.setDataPath(dataPath) != 0 :
         raise NameError("Bad dataset")
        
        self.sampleData = self.library.sampleEnvMapShDataset
        self.sampleData.restype = 
        self.setDataPath.argtypes = []
        
   def sampleData( dims ):
         
        return self.sampleData(dims) 