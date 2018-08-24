#!/usr/bin/python
""" TFlow to TRT
"""
import os
import argparse
import tensorflow as tf
from onnx_tf.frontend import tensorflow_graph_to_onnx_model


def freeze_onnx_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        for n in tf.get_default_graph().as_graph_def().node:
            print(n.name)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            # The graph_def is used to retrieve the nodes
            tf.get_default_graph().as_graph_def(),
            # The output node names are used to select the usefull nodes
            output_node_names.split(",")
        )
        # remove unused training nodes
        output_graph_def = tf.graph_util.remove_training_nodes(
            output_graph_def)

        # convert to ONNX
        onnx_model = tensorflow_graph_to_onnx_model(output_graph_def,
                                                    output_node_names,
                                                    # opset=6,
                                                    ignore_unimplemented=True)

    return onnx_model


if __name__ == "__main__":

    #------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("inModelPath")
    parser.add_argument("inputNodes")
    parser.add_argument("outputNodes")
    parser.add_argument("outModelPath")

    args = parser.parse_args()

    onnxModel = freeze_onnx_graph(args.inModelPath, args.outputNodes)

    onnxFile = open(args.outModelPath, "wb")
    onnxFile.write(onnxModel.SerializeToString())
    onnxFile.close()
