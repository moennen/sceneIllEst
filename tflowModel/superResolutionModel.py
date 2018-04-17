#!/usr/bin/python
""" SuperResolution Model

"""
import argparse
import os
import sys
import time
import tensorflow as tf
import itertools
# from matplotlib import pyplot as plt
from scipy.misc import toimage
from PIL import Image
import OpenEXR
import Imath
import numpy as np

sys.path.append('/mnt/p4/favila/moennen/local/lib/python2.7/site-packages')
import cv2 as cv

sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/'))
from sampleBuffDataset import *

# Parameters
numSteps = 35000
logStep = 5
logTrSteps = 1
logTsSteps = 1

supResFactor = 0.75

batchSz = 128
imgSz = [256, 256]

# logging
tf.logging.set_verbosity(tf.logging.INFO)


#-----------------------------------------------------------------------------------------------------
# DATASET
#-----------------------------------------------------------------------------------------------------

class SupResDatasetTF(object):

    __lib = BufferDataSamplerLibrary(
        "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libSuperResolutionSampler/libSuperResolutionSampler.so")

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, scaleFactor, seed):
        params = np.array([batchSz, imgSz[0], imgSz[1],
                           scaleFactor], dtype=np.float32)
        self.__ds = BufferDataSampler(
            SupResDatasetTF.__lib, dbPath, imgRootDir, params, seed)
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            imgHD, imgLD = self.__ds.getDataBuffers()
            yield (imgHD, imgLD)

#-----------------------------------------------------------------------------------------------------
# UTILS
#-----------------------------------------------------------------------------------------------------


def loadImgPIL(img_name, imgSz, linearCS):

    im = Image.open(img_name)
    ratio = float(imgSz[1])/imgSz[0]
    imgRatio = float(im.size[0])/im.size[1]
    cw = (int(im.size[1]*ratio) if imgRatio > ratio else im.size[0])
    ow = (int((im.size[0]-cw)/2) if imgRatio > ratio else 0)
    ch = (int(im.size[0]/ratio) if imgRatio < ratio else im.size[1])
    oh = (int((im.size[1]-ch)/2) if imgRatio < ratio else 0)
    im = im.crop([ow, oh, ow+cw, oh+ch])
    im = im.resize([imgSz[1], imgSz[0]])
    im = np.array(im)
    im = im.astype(np.float32) / 255.0

    if linearCS == 1:
        im = (im <= 0.04045) * (im / 12.92) + (im > 0.04045) * \
            np.power((im + 0.055)/1.055, 2.4)

    return [im]


def printVarTF(sess):
    tvars = tf.trainable_variables()
    for var in tvars:
        print var.name
        print var.eval(sess)

#-----------------------------------------------------------------------------------------------------
# MODEL
#-----------------------------------------------------------------------------------------------------


def conv_layer(x, filter_size, step, scope, padding='VALID'):
    # tf.random_normal(filter_size))
    initializer = tf.contrib.layers.xavier_initializer()
    layer_w = tf.Variable(initializer(filter_size))
    layer_b = tf.Variable(initializer([filter_size[3]]))
    if step > 0:
        layer = tf.nn.conv2d(x, layer_w, [1, step, step, 1], padding)
    else:
        layer = tf.nn.atrous_conv2d(x, layer_w, abs(step), padding)
    layer = tf.nn.bias_add(layer, layer_b)
    layer = tf.nn.relu(layer, name=scope)
    return layer


def ms_conv_layer(x, filter_size, scope, padding='SAME'):

    ms_filter_size = [filter_size[0], filter_size[1],
                      filter_size[3], filter_size[3]]

    with tf.name_scope('ms_0') as scope:
        ms_0 = conv_layer(x, filter_size, 1, scope, padding)
    with tf.name_scope('ms_2') as scope:
        ms_2 = conv_layer(ms_2, ms_filter_size, -2, scope, padding)
    with tf.name_scope('ms_5') as scope:
        ms_5 = conv_layer(ms_2, ms_filter_size, -5, scope, padding)

    return tf.concat([ms_0, ms_2, ms_5], 3)


def supResBaseModel(imgs):

    with tf.variable_scope('SupResBaseModel'):

        # -----> preprocessing : put the pix values in [-1..1]
        with tf.name_scope('preprocess') as scope:
            img_mean = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32, shape=[
                1, 1, 1, 3], name='img_mean')
            layer = 2.0 * (imgs - img_mean)
        # layer0 = imgs

        # ---->
        with tf.name_scope('layer1') as scope:
            layer = ms_conv_layer(layer, [5, 5, 3, 8], scope)

        # ---->
        with tf.name_scope('layer2') as scope:
            layer = ms_conv_layer(layer, [5, 5, 24, 12], scope)

        # ---->
        with tf.name_scope('layer3') as scope:
            layer = ms_conv_layer(layer, [5, 5, 36, 18], scope)

        # ---->
        with tf.name_scope('layer4') as scope:
            layer = ms_conv_layer(layer, [5, 5, 54, 27], scope)

        # ---->
        with tf.name_scope('layer5') as scope:
            layer = ms_conv_layer(layer, [5, 5, 81, 46], scope)

        # ---->
        with tf.name_scope('layer6') as scope:
            layer = ms_conv_layer(layer, [5, 5, 138, 69], scope)

        # ---->
        with tf.name_scope('layer7') as scope:
            layer = conv_layer(layer, [3, 3, 207, 3], 1, scope, 'SAME')

        return layer


def supResModel(imgs):

    return supResBaseModel(imgs)

#-----------------------------------------------------------------------------------------------------
# UNIT TESTS
#-----------------------------------------------------------------------------------------------------


def tstDataset(imgRootDir, trainPath):

    # rseed = int(time.time())
    rseed = 20160704
    print "SEED : " + str(rseed)

    tf.set_random_seed(rseed)

    trDs = SupResDatasetTF(trainPath, imgRootDir, batchSz,
                           imgSz, supResFactor, rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)

    with tf.Session() as sess:

        sess.run(trInit)

        for step in range(1):

            imgHD, imgLD = sess.run(dsView)

            toimage(imgHD[0]).show()
            toimage(imgLD[0]).show()

            toimage(imgHD[-1]).show()
            toimage(imgLD[-1]).show()


#-----------------------------------------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------------------------------------


def trainSupResModel(modelPath, imgRootDir, trainPath, testPath):

    tbLogsPath = modelPath + "/tbLogs"
    modelFilename = modelPath + "/tfData"

    # rseed = int(time.time())
    rseed = 20160704
    print "SEED : " + str(rseed)

    tf.set_random_seed(rseed)

    trDs = SupResDatasetTF(trainPath, imgRootDir, batchSz,
                           imgSz, supResFactor, rseed)
    tsDs = SupResDatasetTF(testPath, imgRootDir, batchSz,
                           imgSz, supResFactor, rseed)

    inputShape = [batchSz, imgSz[0], imgSz[1], 3]
    outputShape = [batchSz, imgSz[0], imgSz[1], 3]

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input
    inputLD = tf.placeholder(
        tf.float32, shape=inputShape, name="input_ld")
    outputHD = tf.placeholder(
        tf.float32, shape=outputShape, name="output_hd")

    # Graph
    computedHDRes = supResModel(inputLD)

    outputHDRes = tf.subtract(outputHD, inputLD)

    # Optimizer

    # TODO : set a cost function that is adaptive according to the ground truth residual
    cost = tf.reduce_mean(tf.square(tf.subtract(computedHDRes, outputHDRes)))

    globalStep = tf.Variable(0, trainable=False)

    learningRate = tf.train.polynomial_decay(0.001, globalStep, numSteps, 0.0,
                                             power=0.7)

    optEngine = tf.train.AdamOptimizer(learning_rate=learningRate)
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

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}, allow_soft_placement=True, log_device_placement=False)) as sess:
        # with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(tbLogsPath, graph=sess.graph)

        # initialize params
        sess.run(varInit)
        sess.run(tf.local_variables_initializer())

        # Restore model if needed
        try:
            persistency.restore(sess, tf.train.latest_checkpoint(modelPath))
        except:
            print "ERROR Loading model @ ", sys.exc_info()[0]

        sess.run(trInit)

        # get each element of the training dataset until the end is reached
        for step in range(globalStep.eval(sess), numSteps):

            # Get the next training batch
            imgHD, imgLD = sess.run(dsView)

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={inputLD: imgLD, outputHD: imgHD})

            # Log
            if step % logStep == 0:

                # summary
                summary = sess.run(merged_summary_op, feed_dict={
                                   inputLD: imgLD, outputHD: imgHD})
                summary_writer.add_summary(summary, globalStep.eval(sess))

                # Sample train accuracy
                sess.run(trInit)
                trCost = 0
                for logTrStep in range(logTrSteps):
                    imgHD, imgLD = sess.run(dsView)
                    trC = sess.run(
                        cost, feed_dict={inputLD: imgLD, outputHD: imgHD})
                    trCost += trC

                # Sample test accuracy
                sess.run(tsInit)
                tsCost = 0
                for logTsStep in range(logTsSteps):
                    imgHD, imgLD = sess.run(dsView)
                    tsC = sess.run(
                        cost, feed_dict={inputLD: imgLD, outputHD: imgHD})
                    tsCost += tsC

                # summary
                summary = sess.run(test_summary_op, feed_dict={
                                   inputLD: imgLD, outputHD: imgHD})
                summary_writer.add_summary(summary, globalStep.eval(sess))

                print("{:08d}".format(globalStep.eval(sess)) +
                      " | lr = " + "{:.8f}".format(learningRate.eval()) +
                      " | trCost = " + "{:.5f}".format(trCost/logTrSteps) +
                      " | tsCost = " + "{:.5f}".format(tsCost/logTsSteps))

                # step
                persistency.save(sess, modelFilename, global_step=globalStep)

                # reset the training iterator
                sess.run(trInit)

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    #------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("modelPath", help="path to the trainedModel")

    parser.add_argument(
        "imgRootDir", help="root directory to the images in the datasets")

    parser.add_argument(
        "trainLstPath", help="path to the training dataset (list of images path relative to root dir)")

    parser.add_argument(
        "testLstPath", help="path to the testing dataset (list of images path relative to root dir)")

    args = parser.parse_args()

    #------------------------------------------------------------------------------------------------

    # tstDataset(args.imgRootDir, args.trainLstPath)

    trainSupResModel(args.modelPath, args.imgRootDir,
                     args.trainLstPath, args.testLstPath)
