#!/usr/bin/python

import argparse
import time
import numpy as np
from scipy.misc import toimage
from sampleBuffDataset import *

import sys
sys.path.append('/mnt/p4/favila/moennen/local/lib/python2.7/site-packages')
import cv2 as cv


def testSupRes(libPath, dataPath, rootPath):
    batchSz = 10
    rseed = int(time.time())

    samplerLib = BufferDataSamplerLibrary(libPath)
    sampler = BufferDataSampler(
        samplerLib, dataPath, rootPath, np.array([batchSz, 256, 256, 0.15], dtype=np.float32), rseed)
    data = sampler.getDataBuffers()
    print len(data)
    for buff in data:
        print buff.shape
        toimage(buff[0]).show()
        toimage(buff[batchSz-1]).show()


def testFrameInterp(libPath, dataPath, rootPath):
    batchSz = 100
    rseed = int(time.time())

    samplerLib = BufferDataSamplerLibrary(libPath)
    sampler = BufferDataSampler(
        samplerLib, dataPath, rootPath, np.array([batchSz, 256, 256, 0.75, 0], dtype=np.float32), rseed)
    data = sampler.getDataBuffers()
    print len(data)
    for b in range(batchSz):
        i = 0
        for buff in data:
            imName = "TestSample#" + str(i)
            i += 1
            print buff.shape
            cv.imshow(imName, cv.cvtColor(buff[b], cv.COLOR_RGB2BGR))
        cv.waitKey(0)


def testDepthImg(libPath, dataPath, rootPath):
    batchSz = 3
    rseed = int(time.time())

    samplerLib = BufferDataSamplerLibrary(libPath)
    sampler = BufferDataSampler(
        samplerLib, dataPath, rootPath, np.array([batchSz, 256, 256], dtype=np.float32), rseed)
    data = sampler.getDataBuffers()
    print len(data)
    for b in range(batchSz):
        cv.imshow("Image", cv.cvtColor(data[0][b], cv.COLOR_RGB2BGR))
        cv.imshow("Depth", data[1][b])
        cv.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "libPath", help="path to the dataset sampler library")
    parser.add_argument(
        "dbPath", help="path to the dataset")
    parser.add_argument(
        "rootPath", help="root directory of the data")

    args = parser.parse_args()
    #testSupRes(args.libPath, args.dbPath, args.rootPath)
    #testFrameInterp(args.libPath, args.dbPath, args.rootPath)
    testDepthImg(args.libPath, args.dbPath, args.rootPath)
