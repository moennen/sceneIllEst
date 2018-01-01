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
numSteps = 5
batchSz = 6
shOrder = 4
imgSz = [192,108]

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

class EnvMapShDatasetTF(object):

	def __init__(self,dbPath):

	   self.__envMapDb = EnvMapShDataset(dbPath, shOrder)
           self.__dims = [batchSz,imgSz[0], imgSz[1]] 		
	   self.data = Dataset.from_generator(self.genEnvMapSh, (tf.float32, tf.float32))
		
	def genEnvMapSh(self):
	   for i in itertools.count(1):
	      imgs, coeffs, cparams = self.__envMapDb.sampleData(self.__dims)
	      yield (coeffs,imgs)
	
	def getNbShCoeffs(self):
	   return self.__envMapDb.nbShCoeffs*3
       
	
def trainEnvMapShModel(modelPath, trainPath, testPath):
	
   trDs = EnvMapShDatasetTF(trainPath)
   tsDs = EnvMapShDatasetTF(testPath)
   
   nbShCoeffs = trDs.getNbShCoeffs()
   inputShape = [batchSz, imgSz[0], imgSz[1], 3]
   outputShape = [batchSz, nbShCoeffs]
   
   dsIt =  Iterator.from_structure(trDs.data.output_types, trDs.data.output_shapes)
   dsView = dsIt.get_next()

   trInit = dsIt.make_initializer(trDs)
   tsInit = dsIt.make_initializer(tsDs)

   # Input
   inputView = tf.placeholder(tf.float32, shape=inputShape, name="input_view")
   outputSh  = tf.placeholder(tf.float32, shape=outputShape, name="output_sh") 
   dropoutProb = tf.placeholder(tf.float32)  # dropout (keep probability)
   training = tf.placeholder(tf.bool)
   
   # Graph
   computedSh = envMapShModel0000(input_view,nbShCoeffs,dropoutProb,training)	

   # Optimizer
   cost = tf.reduce_mean(tf.square(tf.substract(computedSh,outputSh)))
   learningRate = tf.placeholder(tf.float32, shape=[])
   optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

   # Accuracy	
   accuracy = tf.reduce_mean(tf.square(tf.substract(model, pred)))

   # Params Initializer
   varInit = tf.contrib.layers.xavier_initializer()

   # Persistency
   persistency = tf.train.Saver()

   with tf.Session() as sess:

       # initialize params	
       sess.run(varInit)

       # initialize the iterator on the training data
       sess.run(trInit)

       # Restore model if needed
       persistency.restore(sess, modelPath)

       # get each element of the training dataset until the end is reached
       for step in range(numSteps):
          
          # Get the next training batch
          coeffs, imgs = sess.run(dsView)

          # Run optimization op (backprop)
          sess.run(optimizer, feed_dict={learningRate: 0.001,
                                         dropoutProb: dropout,
                                         inputView: imgs,
                                         outputSh: coeffs})

	  if step % logStep == 0:
	     
             persistency.save(sess, modelPath)  	


if __name__== "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("modelPath", help="path to the trainedModel")
  parser.add_argument("trainDbPath", help="path to the Training EnvMapDataset levelDb path")
  parser.add_argument("testDbPath", help="path to the Testing EnvMapDataset levelDb path")
  args = parser.parse_args()

  trainEnvMapShModel(args.modelPath, args.trainDbPath, args.testDbPath)

