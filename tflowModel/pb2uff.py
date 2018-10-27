#!/usr/bin/python
""" Tensorflow Frozen PB model to UFF
"""
import argparse
import tensorflow as tf
import uff

if __name__ == "__main__":

    #------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("inModelPath")
    parser.add_argument("outModelPath")
    parser.add_argument("inputNodes")
    parser.add_argument("outputNodes")

    args = parser.parse_args()

    uff.from_tensorflow_frozen_model(
        args.inModelPath, args.outputNodes.split(), text=False, list_nodes=False,
        output_filename=args.outModelPath, input_nodes=args.inputNodes.split())
