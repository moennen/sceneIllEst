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
numSteps = 1000
logStep = 25
logTrSteps = 3
logTsSteps = 10
batchSz = 32
shOrder = 4
imgSz = [192,108]

def printVarTF(sess):
   tvars = tf.trainable_variables()
   for var in tvars:
      print var.name
      print var.eval(sess)

def conv_layer(x, filter_size, step):
    layer_w = tf.Variable(tf.random_normal(filter_size))
    layer_b = tf.Variable(tf.random_normal([filter_size[3]]))
    layer = tf.nn.conv2d(x, layer_w, strides=[1, step, step, 1], padding='VALID')
    layer = tf.nn.bias_add(layer, layer_b)
    layer = tf.nn.relu(layer)
    return layer

def envMapShModel0000(imgs, outputSz, dropout, training):

    with tf.variable_scope('EnvMapShModel0000'):

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
	layer7 = tf.layers.dense(layer6f, 1024, activation=tf.nn.relu)
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

   modelFilename = modelPath + "/tfData"
	
   trDs = EnvMapShDatasetTF(trainPath)
   tsDs = EnvMapShDatasetTF(testPath)
   
   nbShCoeffs = trDs.getNbShCoeffs()
   inputShape = [batchSz, imgSz[0], imgSz[1], 3]
   outputShape = [batchSz, nbShCoeffs]
   
   dsIt =  Iterator.from_structure(trDs.data.output_types, trDs.data.output_shapes)
   dsView = dsIt.get_next()

   trInit = dsIt.make_initializer(trDs.data)
   tsInit = dsIt.make_initializer(tsDs.data)

   # Input
   inputView = tf.placeholder(tf.float32, shape=inputShape, name="input_view")
   outputSh  = tf.placeholder(tf.float32, shape=outputShape, name="output_sh") 
   dropoutProb = tf.placeholder(tf.float32)  # dropout (keep probability)
   training = tf.placeholder(tf.bool)
   
   # Graph
   computedSh = envMapShModel0000(inputView,nbShCoeffs,dropoutProb,training)	

   # Optimizer
   cost = tf.reduce_mean(tf.square(tf.subtract(computedSh,outputSh)))
   accuracy = tf.sqrt(cost)
   learningRate = tf.placeholder(tf.float32, shape=[])
   optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

   # Params Initializer
   varInit = tf.global_variables_initializer()

   # Persistency
   persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3, 
                                filename=modelFilename)

   with tf.Session() as sess:

       # initialize params	
       sess.run(varInit)
        
       # Restore model if needed
       try:
          persistency.restore(sess, tf.train.latest_checkpoint(modelPath))
       except:
          print "Cannot load model:", sys.exc_info()[0]

       sess.run(trInit)
 
       # get each element of the training dataset until the end is reached
       for step in range(numSteps):

	  # initialize the iterator on the training data
   
          # Get the next training batch
          coeffs, imgs = sess.run(dsView)

          # Run optimization op (backprop)
          sess.run(optimizer, feed_dict={learningRate: 0.001,
                                         dropoutProb: 0.15,
                                         inputView: imgs,
                                         outputSh: coeffs,
                                         training: True})

	  # Log 
	  if step % logStep == 0:
	   
	     # Sample train accuracy
             sess.run(trInit)
             trAccuracy = 0
             for logTrStep in range(logTrSteps):
                coeffs, imgs = sess.run(dsView)
	        trAccuracy += sess.run(accuracy, feed_dict={dropoutProb: 0.0,
                                                            inputView: imgs,
                                                            outputSh:  coeffs,
                                                            training: False})  
            
             print("Log Train Accurarcy Sample" + str(step * batchSz) + " " 
                   + "{:.5f}".format(trAccuracy/logTrSteps))

	     # Sample test accuracy
             sess.run(tsInit)
             tsAccuracy = 0
             for logTsStep in range(logTsSteps):
                coeffs, imgs = sess.run(dsView)
	        tsAccuracy += sess.run(accuracy, feed_dict={dropoutProb: 0.0,
                                                            inputView: imgs,
                                                            outputSh:  coeffs,
                                                            training: False})  
            
             print("Log Test Accurarcy Sample" + str(step * batchSz) + " " 
                   + "{:.5f}".format(tsAccuracy/logTsSteps))

	
             # step  
             persistency.save(sess, modelFilename, global_step=step)
               	
             # reset the training iterator
             sess.run(trInit)


if __name__== "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("modelPath", help="path to the trainedModel")
  parser.add_argument("trainDbPath", help="path to the Training EnvMapDataset levelDb path")
  parser.add_argument("testDbPath", help="path to the Testing EnvMapDataset levelDb path")
  args = parser.parse_args()

  trainEnvMapShModel(args.modelPath, args.trainDbPath, args.testDbPath)

