#!/usr/bin/python  
""" EnvMapSh model #0000

---> input size : 192x108x3 
---> output : spherical harmonics coefficient up to the 4th order
---> convolutionnal architecture :
---> fully connected output layer
 
"""
#from __future__ import division, print_function, absolute_import
import argparse
import os
import sys
import tensorflow as tf
import itertools
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath('/home/moennen/sceneIllEst/sampleEnvMapShDataset'))
from sampleEnvMapShDataset import *
from tensorflow.contrib.data import Dataset, Iterator

# Parameters
learningRate = 0.001
numSteps = 5
batchSz = 6
shOrder = 4
imgSz = [192,108]
dropout = 0.25 # Dropout, probability to drop a unit

def conv_layer(x, filter_size, step):
    layer_w = tf.Variable(tf.random_normal(filter_size))
    layer_b = tf.Variable(tf.random_normal(filter_size[3]))
    layer = tf.nn.conv2d(x, layer_w, strides=[1, step, step, 1], padding='VALID')
    layer = tf.nn.bias_add(layer, layer_b)
    layer = tf.nn.relu(layer)
    return layer

def envMapShModel000(imgs, outputSz, dropout, training):

    with tf.variable_scope('EnvMapShModel0000', reuse=reuse):

        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

	# ----> 192x108x3	
        layer0=imgs
	# ----> 90x48x32
        layer1=conv_layer(layer0, [7, 7, 3, 32], 2)
	# ----> 41x20x64
        layer2=conv_layer(layer1, [5, 5, 32, 64], 2)
	# ----> 18x8x128
        layer3=conv_layer(layer2, [3, 3, 64, 128], 2)
	# ----> 18x8x128
        layer4=conv_layer(layer3, [3, 3, 128, 128], 1)
	# ----> 7x2x256
        layer5=conv_layer(layer4, [3, 3, 128, 256], 2)
	# ----> 1x1x512
        layer6=conv_layer(layer5, [3, 3, 256, 512], 2)

	#
	layer6f= tf.contrib.layers.flatten(layer6)
	layer7 = tf.layers.dense(layer6f, 1024)
        layer7d= tf.layers.dropout(layer7, rate=dropout, training=training)

        outputLayer = tf.layers.dense(layer7d, outputSz)

	return outputLayer

class EnvMapShDatasetIterator(object):

	def __init__(self,dbPath):

	   self.__envMapDb = EnvMapShDataset(dbPath, shOrder)
           self.__dims = [batchSz,imgSz[0], imgSz[1]] 		
		
	def genEnvMapSh(self):
	   for i in itertools.count(1):
	      imgs, coeffs, cparams = self.__envMapDb.sampleData(self.__dims)
	      yield (coeffs,imgs)
	
	def getIterator(self):
	   self.__data = Dataset.from_generator(self.genEnvMapSh, (tf.float32, tf.float32))
	   self.__it =   Iterator.from_structure(self.__data.output_types, self.__data.output_shapes)
           self.__elem = self.__it.get_next()
           self.__initIt = self.__it.make_initializer(self.__data)
	   return self.__elem, self.__initIt


def trainEnvMapShModel(modelPath, trainPath, testPath):
	
   trDbIt = EnvMapShDatasetIterator(trainPath)
   tr_elem, tr_initIt = trDbIt.getIterator()

   tsDbIt = EnvMapShDatasetIterator(testPath)
   ts_elem, ts_initIt = tsDbIt.getIterator()

   with tf.Session() as sess:

       # initialize the iterator on the training data
       sess.run(tr_initIt)

       # get each element of the training dataset until the end is reached
       for i in range(numSteps):
          
          try:
             elem = sess.run(tr_elem)
             print elem[0].shape
             print elem[1].shape

          except tf.errors.OutOfRangeError:
             print("End of training dataset.")
             break



if __name__== "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("modelPath", help="path to the trainedModel")
  parser.add_argument("trainDbPath", help="path to the Training EnvMapDataset levelDb path")
  parser.add_argument("testDbPath", help="path to the Testing EnvMapDataset levelDb path")
  args = parser.parse_args()

  trainEnvMapShModel(args.modelPath, args.trainDbPath, args.testDbPath)

