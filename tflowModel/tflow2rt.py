#!/usr/bin/python
""" TFlow to TRT
"""

import uff

if __name__ == "__main__":

    #------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("inModelPath")
    parse.add_argument("outputNodes")
    parser.add_argument("outModelPath")

    args = parser.parse_args()

    uff.from_tensorflow_frozen_model(
        args.inModelPath, output_nodes=args.outputNodes, output_filename preprocessor=None)

    #------------------------------------------------------------------------------------------------

    #testDataset(args.imgRootDir, args.trainLstPath)

    #------------------------------------------------------------------------------------------------

    trainModel(args.modelPath, args.imgRootDir,
               args.trainLstPath, args.testLstPath)
