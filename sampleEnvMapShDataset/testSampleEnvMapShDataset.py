#!/usr/bin/python

import argparse
from scipy.misc import toimage
from sampleEnvMapShDataset import *


def test(dataPath, shOrder):
    shDb = EnvMapShDataset(dataPath, shOrder)
    print shDb.nbShCoeffs
    dims = [1, 128, 128]
    imgs, coeffs, cparams = shDb.sampleData(dims)
    print cparams
    print coeffs
    toimage(imgs[0]).show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dbPath", help="path to the EnvMapDataset levelDb path")
    args = parser.parse_args()
    test(args.dbPath, 4)
