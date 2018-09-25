#!/usr/bin/env bash

INGRAPH=$1
INPUTS=$2
OUTPUTS=$3

/mnt/p4/favila/moennen/local/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=$INGRAPH \
--out_graph=$INGRAPH.opt \
--inputs=$INPUTS \
--outputs=$OUTPUTS \
--transforms='strip_unused_nodes(type=float, shape="1,299,299,3") remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms fuse_pad_and_conv fuse_resize_and_conv fuse_resize_pad_and_conv merge_duplicate_nodes sort_by_execution_order'
