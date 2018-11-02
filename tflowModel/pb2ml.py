#!/usr/bin/python
""" Tensorflow Frozen PB model to CoreML
"""
import argparse
import tensorflow as tf
import tfcoreml as tf_converter

if __name__ == "__main__":

   #------------------------------------------------------------------------------------------------

   # NB : coreml add :0 to input and output names

   parser = argparse.ArgumentParser()
   parser.add_argument("inModelPath")
   parser.add_argument("outModelPath")
   parser.add_argument("inputNodes")
   parser.add_argument("outputNodes")

   args = parser.parse_args()

   inputs_dict = {}
   for input_name in args.inputNodes.split():
      inputs_dict[input_name] = [1, 240, 320, 3]

   print inputs_dict

   tf_converter.convert(tf_model_path=args.inModelPath,
                        mlmodel_path=args.outModelPath,
                        output_feature_names=args.outputNodes.split(),
                        input_name_shape_dict=inputs_dict)
