#!/usr/bin/python
""" TFlow to TRT
"""
import os
import argparse
import tensorflow as tf
from tensorflow.python.tools import freeze_graph


def freeze_graph(model_dir, output_node_names):
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

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    sess_config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(graph=tf.Graph(), config=sess_config) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        tf.train.write_graph(sess.graph, '', '/tmp/graph.pb', as_text=False)

        output_graph_def = tf.get_default_graph().as_graph_def()

        for n in output_graph_def.node:
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
        # output_graph_def = tf.graph_util.remove_training_nodes(
        #    output_graph_def)

        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


if __name__ == "__main__":

    #------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("inModelPath")
    parser.add_argument("inputNodes")
    parser.add_argument("outputNodes")
    parser.add_argument("outModelPath")

    args = parser.parse_args()

    pbModel = freeze_graph(args.inModelPath, args.outputNodes)

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(args.outModelPath, "wb") as f:
        f.write(pbModel.SerializeToString())
    #

    # freeze_graph("", "", "", "", output_node_names=args.outputNodes)
