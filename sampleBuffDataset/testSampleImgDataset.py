#!/usr/bin/python

import argparse
import time
import numpy as np
from scipy.misc import toimage
from sampleBuffDataset import *


def test(libPath, dataPath, rootPath):
    batchSz = 10
    rseed = int(time.time())

    samplerLib = BufferDataSamplerLibrary(libPath)
    sampler = BufferDataSampler(
        samplerLib, dataPath, rootPath, np.array([batchSz, 256,256, 0.15], dtype=np.float32), rseed)
    data = sampler.getDataBuffers()
    print len(data)
    for buff in data:
        print buff.shape
        toimage(buff[0]).show()
        toimage(buff[batchSz-1]).show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "libPath", help="path to the dataset sampler library")
    parser.add_argument(
        "dbPath", help="path to the dataset")
    parser.add_argument(
        "rootPath", help="root directory of the data")

    args = parser.parse_args()
    test(args.libPath, args.dbPath, args.rootPath)
