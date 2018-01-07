""" EnvMapSh model #0000

---> input size : 192x108x3
---> output : spherical harmonics coefficient up to the 4th order
---> convolutionnal architecture :
---> fully connected output layer

"""
# from __future__ import division, print_function, absolute_import
import argparse
import os
import sys
import tensorflow as tf
import itertools
from matplotlib import pyplot as plt
# sys.path.append(os.path.abspath('/home/moennen/sceneIllEst/sampleEnvMapShDataset'))
sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleEnvMapShDataset'))
from sampleEnvMapShDataset import *
from tensorflow.contrib.data import Dataset, Iterator

# Parameters
numSteps = 100000
logStep = 100
logTrSteps = 1
logTsSteps = 3
batchSz = 128
shOrder = 8
imgSz = [192, 108]

tf.logging.set_verbosity(tf.logging.INFO)


def printVarTF(sess):
    tvars = tf.trainable_variables()
    for var in tvars:
        print var.name
        print var.eval(sess)


def conv_layer(x, filter_size, step):
    # tf.random_normal(filter_size))
    initializer = tf.contrib.layers.xavier_initializer()
    layer_w = tf.Variable(initializer(filter_size))
    layer_b = tf.Variable(initializer([filter_size[3]]))
    layer = tf.nn.conv2d(x, layer_w, strides=[
                         1, step, step, 1], padding='VALID')
    layer = tf.nn.bias_add(layer, layer_b)
    layer = tf.nn.relu(layer)
    return layer


def envMapShModel0000(imgs, outputSz, dropout):

    with tf.variable_scope('EnvMapShModel0000'):

        # ----> 192x108x3
        layer0 = imgs
        # ----> 90x48x32
        layer1 = conv_layer(layer0, [7, 7, 3, 32], 2)
        # ----> 41x20x64
        layer2 = conv_layer(layer1, [5, 5, 32, 64], 2)
        # ----> 18x8x128
        layer3 = conv_layer(layer2, [3, 3, 64, 128], 2)
        # ----> 18x8x128
        layer4 = conv_layer(layer3, [3, 3, 128, 128], 1)
        # ----> 7x2x256
        layer5 = conv_layer(layer4, [3, 3, 128, 256], 2)
        # ----> 1x1x512
        layer6 = conv_layer(layer5, [3, 3, 256, 512], 2)

        #
        # layer6f = tf.contrib.layers.flatten(layer6)
        initializer = tf.contrib.layers.xavier_initializer()
        layer7 = tf.layers.dense(layer6, 1024, activation=tf.nn.relu, kernel_initializer=initializer,
                                 bias_initializer=initializer)
        layer7d = tf.layers.dropout(layer7, rate=dropout)

        outputLayer = tf.layers.dense(layer7d, outputSz, kernel_initializer=initializer,
                                      bias_initializer=initializer)

        return outputLayer


class EnvMapShDatasetTF(object):

    def __init__(self, dbPath):
        self.__envMapDb = EnvMapShDataset(dbPath, shOrder)
        self.__dims = [batchSz, imgSz[0], imgSz[1]]
        self.data = Dataset.from_generator(
            self.genEnvMapSh, (tf.float32, tf.float32))

    def genEnvMapSh(self):
        for i in itertools.count(1):
            imgs, coeffs, cparams = self.__envMapDb.sampleData(self.__dims)
            yield (coeffs, imgs)

    def getNbShCoeffs(self):
        return self.__envMapDb.nbShCoeffs*3


def trainEnvMapShModel(modelPath, trainPath, testPath):

    tbLogsPath = modelPath + "/tbLogs"
    modelFilename = modelPath + "/tfData"

    trDs = EnvMapShDatasetTF(trainPath)
    tsDs = EnvMapShDatasetTF(testPath)

    nbShCoeffs = trDs.getNbShCoeffs()
    inputShape = [batchSz, imgSz[0], imgSz[1], 3]
    outputShape = [batchSz, nbShCoeffs]

    dsIt = Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input
    inputView = tf.placeholder(tf.float32, shape=inputShape, name="input_view")
    outputSh = tf.placeholder(tf.float32, shape=outputShape, name="output_sh")
    dropoutProb = tf.placeholder(tf.float32)  # dropout (keep probability)
    training = tf.placeholder(tf.bool)

    # Test
    outputSh2 = tf.placeholder(
        tf.float32, shape=outputShape, name="output_sh2")
    outStd = tf.sqrt(tf.reduce_mean(
        tf.square(tf.subtract(outputSh2, outputSh))))

    # Graph
    computedSh = envMapShModel0000(inputView, nbShCoeffs, dropoutProb)

    # Optimizer
    cost = tf.reduce_mean(tf.square(tf.subtract(computedSh, outputSh)))
    accuracy = tf.sqrt(cost)
    globalStep = tf.Variable(0, trainable=False)
    learningRate = tf.train.polynomial_decay(0.001, globalStep, numSteps, 0.00000001,
                                             power=0.5)
    optEngine = tf.train.AdamOptimizer(
        learning_rate=learningRate, epsilon=0.01)
    optimizer = optEngine.minimize(cost, global_step=globalStep)

    # Persistency
    persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                 filename=modelFilename)

    # Initializing the board
    tf.summary.scalar("loss", cost)
    tf.summary.scalar("learning_rate", learningRate)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    grads = optEngine.compute_gradients(cost, tf.trainable_variables())
    for grad, var in grads:
        tf.summary.histogram(var.name + '/gradient', grad)
    merged_summary_op = tf.summary.merge_all()
    tf.summary.scalar("test_loss", cost)
    test_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(tbLogsPath,
                                           graph=tf.get_default_graph())

    # Params Initializer
    varInit = tf.global_variables_initializer()

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
            sess.run(optimizer, feed_dict={dropoutProb: 0.33,
                                           inputView: imgs,
                                           outputSh: coeffs,
                                           training: True})
            # Log
            if step % logStep == 0:

                # summary
                summary = sess.run(merged_summary_op, feed_dict={dropoutProb: 0.0,
                                                                 inputView: imgs,
                                                                 outputSh: coeffs})
                summary_writer.add_summary(summary, globalStep.eval(sess))

                # Sample train accuracy
                sess.run(trInit)
                trAccuracy = 0
                trStd = 0
                for logTrStep in range(logTrSteps):
                    coeffs, imgs = sess.run(dsView)
                    trAccuracy += sess.run(accuracy, feed_dict={dropoutProb: 0.0,
                                                                inputView: imgs,
                                                                outputSh:  coeffs})
                    coeffs2, imgs2 = sess.run(dsView)
                    trStd += sess.run(outStd, feed_dict={outputSh:  coeffs,
                                                         outputSh2: coeffs2})

                # Sample test accuracy
                sess.run(tsInit)
                tsAccuracy = 0
                for logTsStep in range(logTsSteps):
                    coeffs, imgs = sess.run(dsView)
                    tsAccuracy += sess.run(accuracy, feed_dict={dropoutProb: 0.0,
                                                                inputView: imgs,
                                                                outputSh:  coeffs})
                # summary
                summary = sess.run(test_summary_op, feed_dict={dropoutProb: 0.0,
                                                               inputView: imgs,
                                                               outputSh: coeffs})
                summary_writer.add_summary(summary, globalStep.eval(sess))

                print("{:08d}".format(globalStep.eval(sess)) +
                      " | lr = " + "{:.8f}".format(learningRate.eval()) +
                      " | trAcc  = " + "{:.5f}".format(trAccuracy/logTrSteps) +
                      " | tsAcc  = " + "{:.5f}".format(tsAccuracy/logTsSteps) +
                      " | trStd  = " + "{:.5f}".format(trStd/logTrSteps))

                # step
                persistency.save(sess, modelFilename, global_step=globalStep)

                # reset the training iterator
                sess.run(trInit)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("modelPath", help="path to the trainedModel")
    parser.add_argument(
        "trainDbPath", help="path to the Training EnvMapDataset levelDb path")
    parser.add_argument(
        "testDbPath", help="path to the Testing EnvMapDataset levelDb path")
    args = parser.parse_args()

    trainEnvMapShModel(args.modelPath, args.trainDbPath, args.testDbPath)
