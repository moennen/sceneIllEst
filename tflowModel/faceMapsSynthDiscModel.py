#!/usr/bin/python
""" Face Maps From Image Model

"""
import argparse
import os
import sys
import time
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.tools import freeze_graph
import itertools
import numpy as np
import random

from commonModel import *
from sampleBuffDataset import *

import cv2 as cv

#-----------------------------------------------------------------------------------------------------
# DATASET
#-----------------------------------------------------------------------------------------------------


class DatasetTF(object):

    __lib = BufferDataSamplerLibrary(
        "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libFaceAOVSampler/libFaceAOVSampler.so")

    def __init__(self, dbPathPos, dbPathNeg, imgRootDir, batchSz, imgSz, seed):
        self.batchSz = batchSz//2
        self.imgSz = imgSz
        params = np.array([self.batchSz, imgSz[0], imgSz[1]], dtype=np.float32)
        self.__dsPos = BufferDataSampler(
            DatasetTF.__lib, dbPathPos, imgRootDir, params, seed)
        self.__dsNeg = BufferDataSampler(
            DatasetTF.__lib, dbPathNeg, imgRootDir, params, seed)
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32, tf.float32, tf.int32))

    def sample(self):
        for i in itertools.count(1):
            currImgPos, currUVsPos, currDepthPos, currNormalsPos = self.__dsPos.getDataBuffers()
            currImgNeg, currUVsNeg, currDepthNeg, currNormalsNeg = self.__dsNeg.getDataBuffers()

            currUVs = tf.concat([currUVsPos, currUVsNeg], axis=0)
            currDepth = tf.concat([currDepthPos, currDepthNeg], axis=0)
            currNormals = tf.concat([currNormalsPos, currNormalsNeg], axis=0)

            currLabels = tf.concat(tf.constant(1, dtype=tf.int32, shape=[self.batchSz, self.imgSz[0], self.imgSz[1], 1]),
                                   tf.constant(0, dtype=tf.int32, shape=[self.batchSz, self.imgSz[0], self.imgSz[1], 1]), axis=0)

            yield (currUVs, currDepth, currNormals, currLabels)

#-----------------------------------------------------------------------------------------------------
# UNIT TESTS
#-----------------------------------------------------------------------------------------------------


def testDataset(imgRootDir, posPath, negPath):

    rseed = int(time.time())
    imgSz = [256, 256]

    tf.set_random_seed(rseed)
    random.seed(rseed)

    batchSz = 16

    trDs = DatasetTF(posPath, negPath, imgRootDir, batchSz, imgSz, rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)

    with tf.Session() as sess:

        sess.run(trInit)

        for step in range(100):

            currUVs, currDepth, currNormals, currLabels = sess.run(
                dsView)

            idx = random.randint(0, batchSz-1)

            cv.imshow('currDepth', currDepth[idx])

            u, v = cv.split(currUVs[idx])
            cv.imshow('currU', u)
            cv.imshow('currV', v)

            cv.imshow('currNormals', cv.cvtColor(
                currNormals[idx], cv.COLOR_RGB2BGR))

            cv.imshow('currLabels', currLabels)

            cv.waitKey(10)

#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class FaceMapsSynthDiscParams(Pix2PixParams):

    def __init__(self, modelPath, data_format, seed=int(time.time())):

        #
        #

        Pix2PixParams.__init__(self, modelPath, data_format, seed)

        self.numMaxSteps = 217500
        self.numSteps = 2175000
        self.backupStep = 250
        self.trlogStep = 250
        self.tslogStep = 250
        self.vallogStep = 250

        # dimensions
        self.imgSzTr = [296, 296]
        self.batchSz = 24

        # bn vs no bn
        self.useBatchNorm = True
        self.nbChannels = 64
        self.nbInChannels = 6
        self.nbOutputChannels = 1
        self.kernelSz = 5
        self.stridedEncoder = True
        # strided vs resize
        self.stridedDecoder = False
        self.inDispRange = np.array([[0, 1, 2]])
        self.outDispRange = np.array([[0, 1, 2], [3, 4, 5]])
        self.alphaData = 1.0
        self.alphaDisc = 0.0
        self.linearImg = False

        self.doClassOut = True

        # model
        self.model = pix2pix_gen_p
        # loss
        self.loss = pix2pix_classout_loss

        self.update()

#-----------------------------------------------------------------------------------------------------
# EXPORT
#-----------------------------------------------------------------------------------------------------


def saveModel(modelPath, asText, data_format):

    lp = FaceMapsSynthDiscParams(modelPath, data_format)
    lp.isTraining = False

    mdSuff = '-last.pb.txt' if asText else '-last.pb'

    inputsi = tf.placeholder(tf.float32, name="adsk_inFront")
    inputs = tf.multiply(tf.subtract(inputsi, 0.5), 2.0)

    with tf.variable_scope("generator"):
        outputs = lp.model(inputs, lp)

    outputsSz = outputs.get_shape()
    sliceSz = [outputsSz[0], outputsSz[1], outputsSz[2], 1]

    outputNames = "adsk_outNormals,adsk_outUVD"
    # outputNames = "adsk_outNormals"
    outputUVD, outputNormals = tf.split(
        outputs, [3, 3], axis=3 if data_format == 'NHWC' else 1)

    outputUVD = tf.multiply(tf.add(outputUVD, 1.0), 0.5, name="adsk_outUVD")
    outputNormals = tf.identity(outputNormals, name="adsk_outNormals")

    # Persistency
    persistency = tf.train.Saver(filename=lp.modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=sess_config) as sess:

        # initialize params
        sess.run(varInit)

        # Restore model if needed
        persistency.restore(sess, tf.train.latest_checkpoint(modelPath))

        tf.train.write_graph(
            sess.graph, '', lp.modelFilename + mdSuff, as_text=asText)

        freeze_graph.freeze_graph(
            lp.modelFilename + mdSuff,
            '',
            not asText,
            tf.train.latest_checkpoint(modelPath),
            outputNames,
            '', '', lp.modelFilename + mdSuff, True, '')

#-----------------------------------------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------------------------------------


def trainModel(modelPath, imgRootDir, trainPathPos, trainPathNeg, testPathPos, testPathNeg, data_format):

    lp = FaceMapsSynthDiscParams(modelPath, data_format)

    # Datasets / Iterators
    trDs = DatasetTF(trainPathPos, trainPathNeg, imgRootDir,
                     lp.batchSz, lp.imgSzTr, lp.rseed)
    tsDs = DatasetTF(testPathPos. testPathNeg, imgRootDir,
                     lp.batchSz, lp.imgSzTr, lp.rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input placeholders
    inUVi = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 2], name="input_uvs")
    inUV = preprocess(inUVi, True, data_format)
    inDepthi = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 1], name="input_depth")
    inDepth = preprocess(inDepthi, True, data_format)
    inNormali = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_normals")
    inNormal = preprocess(inNormali, False, data_format)

    inLabels = f.placeholder(
        tf.int32, shape=[lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 1], name="labels")

    inMaps = tf.concat([inUV, inDepth, inNormal],
                       3 if data_format == 'NHWC' else 1)

    # Optimizers
    [opts, loss, trSum, tsSum, valSum] = pix2pix_optimizer(
        inMaps, inLabels, lp)

    # Persistency
    persistency = tf.train.Saver(
        pad_step_number=True, max_to_keep=1, filename=lp.modelFilename)

    # Logger
    merged_summary_op = tf.summary.merge_all()

    # Params Initializer
    varInit = tf.global_variables_initializer()

    # Sessions options
    sess_config = tf.ConfigProto(device_count={'GPU': 1})
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        # with tf.Session() as sess:

        train_summary_writer = tf.summary.FileWriter(
            lp.tbLogsPath + "/Train", graph=sess.graph)
        test_summary_writer = tf.summary.FileWriter(lp.tbLogsPath + "/Test")
        val_summary_writer = tf.summary.FileWriter(lp.tbLogsPath + "/Val")

        # profiling
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        # initialize params
        sess.run(varInit)
        sess.run(tf.local_variables_initializer())

        # Restore model if needed
        try:
            persistency.restore(sess, tf.train.latest_checkpoint(modelPath))
        except:
            print "Cannot load model:", sys.exc_info()[0]

        sess.run(trInit)

        # get each element of the training dataset until the end is reached
        while lp.globalStep.eval(sess) < lp.numSteps:

            # Get the next training batch
            currUVs, currDepth, currNormals, currLabels = sess.run(
                dsView)

            trFeed = {lp.isTraining: True,
                      inUVi: currUVs,
                      inDepthi: currDepth,
                      inNormali: currNormals,
                      inLabels: currLabels}

            step = lp.globalStep.eval(sess) + 1

            # Run optimization
            if step % lp.trlogStep == 0:
                _, summary, _ = sess.run(
                    [opts, trSum, lp.globalStepInc], feed_dict=trFeed)
                train_summary_writer.add_summary(summary, step)
            else:
                sess.run([opts, lp.globalStepInc], feed_dict=trFeed)

            # if profile:
            #     # Create the Timeline object, and write it to a json
            #     tl = timeline.Timeline(run_metadata.step_stats)
            #     ctf = tl.generate_chrome_trace_format()
            #     with open('timeline.json', 'w') as f:
            #         f.write(ctf)

            # SUMMARIES

            if step % lp.trlogStep == 0:
                summary = sess.run(tsSum, feed_dict={lp.isTraining: False,
                                                     inUVi: currUVs,
                                                     inDepthi: currDepth,
                                                     inNormali: currNormals,
                                                     inLabels: currLabels})
                train_summary_writer.add_summary(summary, step)

            if step % lp.tslogStep == 0:

                sess.run(tsInit)
                currUVs, currDepth, currNormals, currLabels = sess.run(dsView)
                tsLoss, summary = sess.run([loss, tsSum], feed_dict={lp.isTraining: False,
                                                                     inUVi: currUVs,
                                                                     inDepthi: currDepth,
                                                                     inNormali: currNormals,
                                                                     inLabels: currLabels})

                test_summary_writer.add_summary(summary, step)

                print("{:08d}".format(step-1) +
                      " | lr = " + "{:.8f}".format(lp.learningRate.eval()) +
                      " | loss = " + "{:.5f}".format(tsLoss))

                # reset the training iterator
                sess.run(trInit)

            # PERSISTENCY
            if step % lp.backupStep == 0:
                persistency.save(sess, lp.modelFilename,
                                 global_step=lp.globalStep)



#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    #------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument("modelPath", help="path to the trainedModel")

    parser.add_argument(
        "imgRootDir", help="root directory to the images in the datasets")

    parser.add_argument(
        "trainPosLstPath", help="path to the training dataset (list of images path relative to root dir)")
    parser.add_argument(
        "trainNegLstPath", help="path to the training dataset (list of images path relative to root dir)")

    parser.add_argument(
        "testPosLstPath", help="path to the testing dataset (list of images path relative to root dir)")

    parser.add_argument(
        "testNegLstPath", help="path to the testing dataset (list of images path relative to root dir)")

    parser.add_argument("--nhwc", dest='nhwc',
                        default=False, action='store_true')

    args = parser.parse_args()

    data_format = 'NHWC' if args.nhwc else 'NCHW'

    #------------------------------------------------------------------------------------------------

    # testDataset(args.imgRootDir, args.trainPosLstPath, args.trainNegLstPath)

    #------------------------------------------------------------------------------------------------

    trainModel(args.modelPath, args.imgRootDir,
               args.trainPosLstPath, args.trainNegLstPath,
               args.testPosLstPath, args.testNegLstPath, data_format)

    #------------------------------------------------------------------------------------------------

    # saveModel(args.modelPath, False, data_format)
