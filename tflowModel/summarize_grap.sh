#!/usr/bin/env bash

INGRAPH=$1

/mnt/p4/favila/moennen/local/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=$INGRAPH --print_structure=true