#!/usr/bin/python

import argparse
import time
from scipy.misc import toimage
from sampleEnvMapShDataset import *


def test(dataPath, imgPath, shOrder):
    dims = [1, 128, 256]
    img = EnvMapShDataset.loadImg(imgPath, dims[1:3])
    toimage(img[0]).show()
    rseed = int(time.time())
    shDb = EnvMapShDataset(dataPath, shOrder, rseed)
    print shDb.nbShCoeffs
    imgs, coeffs, cparams = shDb.sampleData(dims)
    print cparams
    print coeffs[0]
    toimage(imgs[0]).show()
    envMap = EnvMapShDataset.generateEnvMap(shOrder, coeffs[0], dims[1:3])
    print envMap.shape
    toimage(envMap[0]).show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dbPath", help="path to the EnvMapDataset levelDb path")
    parser.add_argument(
        "imgPath", help="path to an image")
    parser.add_argument("order", help="shOrder")

    args = parser.parse_args()
    test(args.dbPath, args.imgPath, int(args.order))
