#!/usr/bin/python
""" Frame Interpolation Model

"""
import argparse
import os
import sys
import time
import tensorflow as tf
from tensorflow.python.client import timeline
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
        "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libFrameInterpolationSampler/libFrameInterpolationSampler.so")

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, scaleFactor, blendInLDFreq, mode, linearCS, seed,
                 minPrevNextSqDiff=-1.0, maxPrevNextSqDiff=1000000.0):
        params = np.array([batchSz, imgSz[0], imgSz[1],
                           scaleFactor, blendInLDFreq, minPrevNextSqDiff, maxPrevNextSqDiff, mode,
                           1.0 if linearCS else 0.0], dtype=np.float32)
        self.__ds = BufferDataSampler(
            DatasetTF.__lib, dbPath, imgRootDir, params, seed)
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            currHD, currLD, blendSple, prevSple, nextSple = self.__ds.getDataBuffers()
            yield (currHD, currLD, blendSple, prevSple, nextSple)


def loadValidationData(dataPath, dataRootDir, dataSz, linearCS=False):

    prevIm = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3))
    nextIm = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3))
    blendIm = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3))

    n = 0

    # input
    with open(dataPath, 'r') as img_names_file:

        for data in img_names_file:

            if n >= dataSz[0]:
                break

            data = data.rstrip('\n').split()

            prevIm[n, :, :, :] = loadResizeImgPIL(dataRootDir + "/" +
                                                  data[0], [dataSz[1], dataSz[2]], linearCS)
            nextIm[n, :, :, :] = loadResizeImgPIL(dataRootDir + "/" +
                                                  data[2], [dataSz[1], dataSz[2]], linearCS)

            alpha = float(data[3])

            blendIm[n, :, :, :] = alpha * prevIm[n, :, :, :] + \
                (1.0 - alpha) * nextIm[n, :, :, :]

            n = n + 1

    return blendIm, prevIm, nextIm

#-----------------------------------------------------------------------------------------------------
# UNIT TESTS
#-----------------------------------------------------------------------------------------------------


def testDataset(imgRootDir, trainPath):

    rseed = int(time.time())
    imgSz = [256, 256]

    tf.set_random_seed(rseed)

    batchSz = 16

    trDs = DatasetTF(trainPath, imgRootDir, batchSz,
                     imgSz, 1.0, 1.0, 0, False, rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=sess_config) as sess:

        sess.run(trInit)

        for step in range(100):

            currHD, currLD, blendSple, prevSple, nextSple = sess.run(dsView)

            idx = random.randint(0, batchSz-1)

            cv.imshow('prevImg', cv.cvtColor(prevSple[idx], cv.COLOR_RGB2BGR))
            cv.imshow('nextImg', cv.cvtColor(nextSple[idx], cv.COLOR_RGB2BGR))
            cv.imshow('blendImg', cv.cvtColor(
                blendSple[idx], cv.COLOR_RGB2BGR))
            cv.imshow('currImg', cv.cvtColor(currHD[idx], cv.COLOR_RGB2BGR))
            cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class FrameInterpolationModelParams(Pix2PixParams):

    def __init__(self, modelPath, seed=int(time.time())):

        #
        # model 0 : 296x296x32 / charbonnier / resize / pix2pix_gen_p / bn
        #

        Pix2PixParams.__init__(self, modelPath, seed)

        self.numMaxSteps = 217500
        self.numSteps = 217500
        self.backupStep = 250
        self.trlogStep = 250
        self.tslogStep = 250
        self.vallogStep = 250

        self.imgSzTr = [296, 296]
        self.batchSz = 24

        # bn vs no bn
        self.useBatchNorm = True
        self.nbChannels = 32
        self.nbInChannels = 9
        self.nbOutputChannels = 3
        self.kernelSz = 5
        self.stridedEncoder = True
        # strided vs resize
        self.stridedDecoder = False
        self.inDispRange = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.outDispRange = np.array([[0, 1, 2]])
        self.alphaData = 1.0
        self.alphaDisc = 0.0
        self.linearImg = False

        self.model = pix2pix_gen_p
        self.inErr = tf.placeholder(tf.float32, shape=[
            self.batchSz, self.imgSzTr[0], self.imgSzTr[1], 1], name="input_mask")
        self.loss = loss_errw_charbonnier

        self.dsScaleFactor = 0.7
        self.dsBlendInLdFreq = 0.0
        self.dsMode = 0
        # minimum value of frame difference # default to 0.0001
        self.dsMinPrevNextSqDiff = 0.000075
        # maximum value ""                  # default to 0.0005
        self.dsMaxPrevNextSqDiff = 0.001

        self.update()

        def loss_errw_charbonnier(self, outputs, targets):

            outputs_sc = tf.log(tf.add(tf.multiply(outputs, 3.0), 4.0))
            targets_sc = tf.log(tf.add(tf.multiply(targets, 3.0), 4.0))

            diff = tf.multiply(tf.subtract(
                outputs_sc, targets_sc), self.inMask)

            nvalid_b = tf.add(tf.reduce_sum(self.inMask, axis=[1, 2]), EPS)

            log_scales = tf.divide(tf.reduce_sum(
                diff, axis=[1, 2], keepdims=True), nvalid_b)
            diff = tf.subtract(diff, log_scales)

            return tf.divide(tf.reduce_sum(tf.sqrt(EPS + tf.square(diff))), tf.reduce_sum(nvalid_b))


#-----------------------------------------------------------------------------------------------------
# VALIDATION
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgRootDir, imgLst, forceTrainingSize=True, maxSz=-1):

    lp = FrameInterpolationModelParams(modelPath)
    lp.isTraining = False

    evalSz = [1, 256, 256, 3]

    prevImgi = tf.placeholder(tf.float32, name="prev_img")
    prevImg = preprocess(prevImgi)
    nextImgi = tf.placeholder(tf.float32, name="next_img")
    nextImg = preprocess(nextImgi)
    interpFactor = tf.placeholder(tf.float32, name="interp_factor")
    blendImg = tf.add(tf.multiply(prevImg, interpFactor),
                      tf.multiply(nextImg, 1.0-interpFactor))

    inputs = tf.concat([blendImg, tf.subtract(
        blendImg, prevImg), tf.subtract(blendImg, nextImg)], axis=3)

    with tf.variable_scope("generator"):
        outputs = lp.model(inputs, lp)

    # Persistency
    persistency = tf.train.Saver(filename=lp.modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    sess_config = tf.ConfigProto(device_count={'GPU': 0})
    #sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:

        # initialize params
        sess.run(varInit)

        # Restore model if needed
        persistency.restore(sess, tf.train.latest_checkpoint(modelPath))

        # input
        with open(imgLst, 'r') as img_names_file:

            for data in img_names_file:

                data = data.rstrip('\n').split()

                alpha = float(data[3])

                if forceTrainingSize:
                    prevIm = [loadResizeImgPIL(
                        imgRootDir + "/" + data[0], [evalSz[1], evalSz[2]], lp.linearImg)]
                    nextIm = [loadResizeImgPIL(
                        imgRootDir + "/" + data[1], [evalSz[1], evalSz[2]], lp.linearImg)]
                else:
                    prevIm = [loadImgPIL(
                        imgRootDir + "/" + data[0], lp.linearImg)]
                    nextIm = [loadImgPIL(
                        imgRootDir + "/" + data[1], lp.linearImg)]

                currIm = sess.run(outputs, feed_dict={prevImgi: prevIm,
                                                      nextImgi: nextIm,
                                                      interpFactor: alpha})

                # show the sample
                cv.imshow('Prev', cv.cvtColor(prevIm[0], cv.COLOR_RGB2BGR))
                cv.imshow('Next', cv.cvtColor(nextIm[0], cv.COLOR_RGB2BGR))

                cv.imshow('Curr', cv.cvtColor(currIm[0], cv.COLOR_RGB2BGR))

                cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------------------------------------


def trainModel(modelPath, imgRootDir, trainPath, testPath, valPath):

    lp = FrameInterpolationModelParams(modelPath)

    # Datasets / Iterators
    trDs = DatasetTF(trainPath, imgRootDir, lp.batchSz,
                     lp.imgSzTr, lp.dsScaleFactor, lp.dsBlendInLdFreq, lp.dsMode, lp.linearImg, lp.rseed,
                     lp.dsMinPrevNextSqDiff, lp.dsMaxPrevNextSqDiff)
    tsDs = DatasetTF(testPath, imgRootDir, lp.batchSz,
                     lp.imgSzTr, lp.dsScaleFactor, lp.dsBlendInLdFreq, lp.dsMode, lp.linearImg, lp.rseed,
                     lp.dsMinPrevNextSqDiff, lp.dsMaxPrevNextSqDiff)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input placeholders
    sampleShape = [lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3]

    inBlendi = tf.placeholder(
        tf.float32, shape=sampleShape, name="input_blend")
    inBlend = preprocess(inBlendi)
    inPrevi = tf.placeholder(tf.float32, shape=sampleShape, name="input_prev")
    inPrev = tf.subtract(inBlend, preprocess(inPrevi))
    inNexti = tf.placeholder(tf.float32, shape=sampleShape, name="input_next")
    inNext = tf.subtract(inBlend, preprocess(inNexti))
    inCurri = tf.placeholder(tf.float32, shape=sampleShape, name="input_curr")
    inCurr = tf.subtract(inBlend, preprocess(inCurri))

    # Optimizers
    [opts, loss, trSum, tsSum, valSum] = pix2pix_optimizer(
        tf.concat([inBlend, inPrev, inNext], axis=3), inCurr, lp)

    # Validation
    valBlend, valPrev, valNext = loadValidationData(
        valPath, imgRootDir, sampleShape, lp.linearImg)

    # Persistency
    persistency = tf.train.Saver(
        pad_step_number=True, max_to_keep=3, filename=lp.modelFilename)

    # Logger
    merged_summary_op = tf.summary.merge_all()

    # Params Initializer
    varInit = tf.global_variables_initializer()

    # Session configuration
    sess_config = tf.ConfigProto(device_count={'GPU': 1})
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        # with tf.Session() as sess:

        train_summary_writer = tf.summary.FileWriter(
            lp.tbLogsPath + "/Train", graph=sess.graph)
        test_summary_writer = tf.summary.FileWriter(lp.tbLogsPath + "/Test")
        val_summary_writer = tf.summary.FileWriter(lp.tbLogsPath + "/Val")

        # profiling
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()

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
            currHD, currLD, blendSple, prevSple, nextSple = sess.run(dsView)

            trFeed = {lp.isTraining: True,
                      inCurri: currHD,
                      inPrevi: prevSple,
                      inNexti: nextSple,
                      inBlendi: blendSple}

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
                                                     inCurri: currHD,
                                                     inPrevi: prevSple,
                                                     inNexti: nextSple,
                                                     inBlendi: blendSple})
                train_summary_writer.add_summary(summary, step)

            if step % lp.tslogStep == 0:

                sess.run(tsInit)
                currHD, currLD, blendSple, prevSple, nextSple = sess.run(
                    dsView)
                tsLoss, summary = sess.run([loss, tsSum], feed_dict={lp.isTraining: False,
                                                                     inCurri: currHD,
                                                                     inPrevi: prevSple,
                                                                     inNexti: nextSple,
                                                                     inBlendi: blendSple})

                test_summary_writer.add_summary(summary, step)

                print("{:08d}".format(step-1) +
                      " | lr = " + "{:.8f}".format(lp.learningRate.eval()) +
                      " | loss = " + "{:.5f}".format(tsLoss))

                # reset the training iterator
                sess.run(trInit)

            # validation
            if step % lp.vallogStep == 0:

                summary = sess.run(valSum, feed_dict={lp.isTraining: False,
                                                      inCurri: valBlend,
                                                      inPrevi: valPrev,
                                                      inNexti: valNext,
                                                      inBlendi: valBlend})

                val_summary_writer.add_summary(summary, step)

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
        "trainLstPath", help="path to the training dataset (list of images path relative to root dir)")

    parser.add_argument(
        "testLstPath", help="path to the testing dataset (list of images path relative to root dir)")

    parser.add_argument(
        "valLstPath", help="path to the validation dataset (list of images path relative to root dir)")

    args = parser.parse_args()

    #------------------------------------------------------------------------------------------------

    testDataset(args.imgRootDir, args.trainLstPath)

    #------------------------------------------------------------------------------------------------

    # trainModel(args.modelPath, args.imgRootDir,
    #           args.trainLstPath, args.testLstPath, args.valLstPath)

    #------------------------------------------------------------------------------------------------

    # testModel(args.modelPath, args.imgRootDir, args.testLstPath, 100)

    #------------------------------------------------------------------------------------------------

    #evalModel(args.modelPath, args.imgRootDir, args.valLstPath)
