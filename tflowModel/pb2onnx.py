#!/usr/bin/python
""" Tensorflow Frozen PB model to ONNX
"""
import argparse
import tensorflow as tf
from onnx_tf.frontend import tensorflow_graph_to_onnx_model


def pb2onnx(input_pb, output_nodes):

    with tf.gfile.GFile(input_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        onnx_model = tensorflow_graph_to_onnx_model(
            graph_def, output_nodes, ignore_unimplemented=True)

        return onnx_model


if __name__ == "__main__":

    #------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("inModelPath")
    parser.add_argument("inputNodes")
    parser.add_argument("outputNodes")
    parser.add_argument("outModelPath")

    args = parser.parse_args()

    onnx_model = pb2onnx(args.inModelPath, args.outputNodes)

    file = open(args.outModelPath, "wb")
    file.write(onnx_model.SerializeToString())
    file.close()
