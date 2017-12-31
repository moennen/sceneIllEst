#!/usr/bin/python  
'''
TESTS : envMapShDataset 
'''

import argparse
import os
import sys
import tensorflow as tf
import itertools
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath('/home/moennen/sceneIllEst/sampleEnvMapShDataset'))
from sampleEnvMapShDataset import *
from tensorflow.contrib.data import Dataset, Iterator


def test(dataPath, shOrder, batchSz, imgSz, epochSz):

   # global params
   dims = [batchSz, imgSz, imgSz] 

   # database 
   shDb = EnvMapShDataset(dataPath, shOrder)

   # generator function
   def genEnvMapSh():
      for i in itertools.count(1):
	imgs, coeffs, cparams = shDb.sampleData(dims)
	yield (coeffs,imgs)

   # tensorflow dataset
   tr_data = Dataset.from_generator(genEnvMapSh, (tf.float32, tf.float32))
   iterator = Iterator.from_structure(tr_data.output_types,
                                      tr_data.output_shapes)
   next_element = iterator.get_next()
   training_init_op = iterator.make_initializer(tr_data)

   with tf.Session() as sess:

       # initialize the iterator on the training data
       sess.run(training_init_op)

       # get each element of the training dataset until the end is reached
       for i in range(epochSz):

          try:
             elem = sess.run(next_element)
             print elem[0].shape
             print elem[1].shape

          except tf.errors.OutOfRangeError:
             print("End of training dataset.")
             break

if __name__== "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("dbPath", help="path to the EnvMapDataset levelDb path")
  args = parser.parse_args()
  test(args.dbPath,4,6,16,10)
