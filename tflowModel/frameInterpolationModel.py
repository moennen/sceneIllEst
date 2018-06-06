#!/usr/bin/python
""" Frame Interpolation Model

"""
import argparse
import os
import sys
import time
import tensorflow as tf
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

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, scaleFactor, blendInLDFreq, mode, seed,
                 minPrevNextSqDiff=-1.0, maxPrevNextSqDiff=1000000.0):
        params = np.array([batchSz, imgSz[0], imgSz[1],
                           scaleFactor, blendInLDFreq, minPrevNextSqDiff, maxPrevNextSqDiff, mode], dtype=np.float32)
        self.__ds = BufferDataSampler(
            DatasetTF.__lib, dbPath, imgRootDir, params, seed)
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            currHD, currLD, blendSple, prevSple, nextSple = self.__ds.getDataBuffers()
            yield (currHD, currLD, blendSple, prevSple, nextSple)

#-----------------------------------------------------------------------------------------------------
# MODEL
#-----------------------------------------------------------------------------------------------------


def model(imgs):

    return base_model(imgs)

#-----------------------------------------------------------------------------------------------------
# EVAL
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgRootDir, imgLst, minPyrSize):

    modelFilename = modelPath + "/tfData"

    inPrevi = tf.placeholder(tf.float32, name="input_prev")
    inPrev = preprocess(inPrevi)
    inNexti = tf.placeholder(tf.float32, name="input_next")
    inNext = preprocess(inNexti)

    interpFactor = tf.placeholder(tf.float32, name="interp_factor")

    inBlend = tf.add(tf.multiply(inPrev, interpFactor),
                     tf.multiply(inNext, 1.0-interpFactor))
    # Model
    outCurr = model(
        tf.concat([inBlend, inPrev, inNext], 3))

    # Persistency
    persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                 filename=modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session() as sess:

        # initialize params
        sess.run(varInit)

        # Restore model if needed
        persistency.restore(sess, tf.train.latest_checkpoint(modelPath))

        # input
        with open(imgLst, 'r') as img_names_file:

            for data in img_names_file:

                data = data.split()

                prevImg = loadImgPIL(imgRootDir + "/" + data[0], 1)
                nextImg = loadImgPIL(imgRootDir + "/" + data[2], 1)
                currImg = loadImgPIL(imgRootDir + "/" + data[1], 1)

                alpha = float(data[3].rstrip('\n'))

                print data[0], data[1], data[2], alpha

                estImg = sess.run(
                    outCurr, feed_dict={inPrevi: prevImg, inNext: nextImg,
                                        interpFactor: alpha})

                # show the sample
                cv.imshow('GTH', cv.cvtColor(
                    currImg[0], cv.COLOR_RGB2BGR))
                cv.imshow('EST', cv.cvtColor(estImg[0], cv.COLOR_RGB2BGR))
                cv.imshow('PRV', cv.cvtColor(
                    prevImg[0], cv.COLOR_RGB2BGR))
                cv.imshow('NXT', cv.cvtColor(
                    nextImg[0], cv.COLOR_RGB2BGR))
                cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# UNIT TESTS
#-----------------------------------------------------------------------------------------------------


def testDataset(imgRootDir, trainPath):

    rseed = int(time.time())
    print "SEED : " + str(rseed)

    tf.set_random_seed(rseed)

    batchSz = 16

    trDs = DatasetTF(trainPath, imgRootDir, batchSz,
                     imgSz, 0.7, 1.0, 0, rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)

    with tf.Session() as sess:

        sess.run(trInit)

        for step in range(100):

            currHD, currLD, blendSple, prevSple, nextSple = sess.run(dsView)

            idx = random.randint(0, batchSz-1)

            cv.imshow('currHD', cv.cvtColor(currHD[idx], cv.COLOR_RGB2BGR))
            cv.imshow('currLD', cv.cvtColor(currLD[idx], cv.COLOR_RGB2BGR))
            cv.imshow('blendSple', cv.cvtColor(
                blendSple[idx], cv.COLOR_RGB2BGR))
            cv.imshow('prevSple', cv.cvtColor(prevSple[idx], cv.COLOR_RGB2BGR))
            cv.imshow('nextSple', cv.cvtColor(nextSple[idx], cv.COLOR_RGB2BGR))

            cv.waitKey(300)

#-----------------------------------------------------------------------------------------------------
# TESTING
#-----------------------------------------------------------------------------------------------------


def testModel(modelPath, imgRootDir, testPath, nbTests):

    lp = LearningParams(modelPath, int(time.time()))
    baseN = 64

    # Datasets / Iterators
    tsDs = DatasetTF(testPath, imgRootDir, 1,
                     lp.imgSzTs, 0.7, 0.0, 0, lp.rseed)

    dsIt = tf.data.Iterator.from_structure(
        tsDs.data.output_types, tsDs.data.output_shapes)
    dsView = dsIt.get_next()
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input placeholders
    sampleShape = [1, lp.imgSzTs[0], lp.imgSzTs[1], 3]

    inPrevi = tf.placeholder(tf.float32, shape=sampleShape, name="input_prev")
    inPrev = preprocess(inPrevi)
    inNexti = tf.placeholder(tf.float32, shape=sampleShape, name="input_next")
    inNext = preprocess(inNexti)
    inCurri = tf.placeholder(tf.float32, shape=sampleShape, name="input_curr")
    inCurr = preprocess(inCurri)
    inBlendi = tf.placeholder(
        tf.float32, shape=sampleShape, name="input_blend")
    inBlend = preprocess(inBlendi)

    # Model
    outCurr = pix2pix_gen(
        tf.concat([inBlend, inPrev, inNext], 3), 3, baseN, 0.0, False)

    # Persistency
    persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                 filename=lp.modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        # initialize params
        sess.run(varInit)

        # initialize iterator
        sess.run(tsInit)

        # Restore model if needed
        persistency.restore(sess, tf.train.latest_checkpoint(modelPath))

        for step in range(nbTests):
            # Get the next training batch
            currHD, currLD, blendSple, prevSple, nextSple = sess.run(dsView)

            estSple = sess.run(outCurri, feed_dict={
                inCurri: currHD, inPrevi: prevSple, inNexti: nextSple, inBlendi: blendSple})

            cv.imshow('GT',  cv.cvtColor(currHD[0], cv.COLOR_RGB2BGR))
            cv.imshow('EST',  cv.cvtColor(estSple[0], cv.COLOR_RGB2BGR))
            cv.imshow('BLD', cv.cvtColor(blendSple[0], cv.COLOR_RGB2BGR))
            cv.imshow('PREV', cv.cvtColor(prevSple[0], cv.COLOR_RGB2BGR))
            cv.imshow('NXT', cv.cvtColor(nextSple[0], cv.COLOR_RGB2BGR))
            cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------------------------------------


def trainModel(modelPath, imgRootDir, trainPath, testPath):

    lp = LearningParams(modelPath, 20160704)
    lp.numSteps = 130000
    lp.tslogStep = 150
    lp.trlogStep = 150
    lp.backupStep = 250
    lp.imgSzTr = [64, 64]
    lp.batchSz = 64
    baseN = 32
    alpha_loss = 0.9
    minPrevNextSqDiff = 0.0001    # minimum value of frame difference # default to 0.0001
    maxPrevNextSqDiff = 0.0003    # maximum value ""                  # default to 0.0005
    lp.update()

    # Datasets / Iterators
    trDs = DatasetTF(trainPath, imgRootDir, lp.batchSz,
                     lp.imgSzTr, 0.7, 0.0, 0, lp.rseed, minPrevNextSqDiff, maxPrevNextSqDiff)
    tsDs = DatasetTF(testPath, imgRootDir, lp.batchSz,
                     lp.imgSzTr, 0.7, 0.0, 0, lp.rseed, minPrevNextSqDiff, maxPrevNextSqDiff)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input placeholders
    sampleShape = [lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3]

    inPrevi = tf.placeholder(tf.float32, shape=sampleShape, name="input_prev")
    inPrev = preprocess(inPrevi)
    inNexti = tf.placeholder(tf.float32, shape=sampleShape, name="input_next")
    inNext = preprocess(inNexti)
    inCurri = tf.placeholder(tf.float32, shape=sampleShape, name="input_curr")
    inCurr = preprocess(inCurri)
    inBlendi = tf.placeholder(
        tf.float32, shape=sampleShape, name="input_blend")
    inBlend = preprocess(inBlendi)

    # Model
    [optDisc, lossDisc, optGen, lossGen, trSum, tsSum] = pix2pix_optimizer(
        tf.concat([inBlend, inPrev, inNext], axis=3), inCurr,
        lp.learningRate, alpha_loss, lp.globalStep, lp.dropoutProb, lp.isTraining, baseN)

    # Persistency
    persistency = tf.train.Saver(
        pad_step_number=True, max_to_keep=1, filename=lp.modelFilename)

    # Logger
    merged_summary_op = tf.summary.merge_all()

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:

        train_summary_writer = tf.summary.FileWriter(
            lp.tbLogsPath + "/Train", graph=sess.graph)
        test_summary_writer = tf.summary.FileWriter(lp.tbLogsPath + "/Test")

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
        for step in range(lp.globalStep.eval(sess)+1, lp.numSteps+1):

            # Get the next training batch
            currHD, currLD, blendSple, prevSple, nextSple = sess.run(dsView)

            trFeed = {lp.dropoutProb: 0.01, lp.isTraining: True, inCurri: currHD,
                      inPrevi: prevSple, inNexti: nextSple, inBlendi: blendSple}

            # Run optimization
            if step % lp.trlogStep == 0:
                _, _, summary = sess.run(
                    [optDisc, optGen, trSum], feed_dict=trFeed)
                train_summary_writer.add_summary(
                    summary, lp.globalStep.eval(sess))
            else:
                sess.run([optDisc, optGen], feed_dict=trFeed)

            # SUMMARIES
            if step % lp.tslogStep == 0:

                    # Sample test accuracy
                sess.run(tsInit)
                currHD, currLD, blendSple, prevSple, nextSple = sess.run(
                    dsView)
                loss, summary = sess.run([lossGen, tsSum], feed_dict={lp.dropoutProb: 0.0,
                                                                      lp.isTraining: False,
                                                                      inCurri: currHD,
                                                                      inPrevi: prevSple,
                                                                      inNexti: nextSple,
                                                                      inBlendi: blendSple})

                test_summary_writer.add_summary(
                    summary, lp.globalStep.eval(sess))

                print("{:08d}".format(lp.globalStep.eval(sess)) +
                      " | lr = " + "{:.8f}".format(lp.learningRate.eval()) +
                      " | loss = " + "{:.5f}".format(loss))

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
        "trainLstPath", help="path to the training dataset (list of images path relative to root dir)")

    parser.add_argument(
        "testLstPath", help="path to the testing dataset (list of images path relative to root dir)")

    args = parser.parse_args()

    #------------------------------------------------------------------------------------------------

    # testDataset(args.imgRootDir, args.trainLstPath)

    #------------------------------------------------------------------------------------------------

    trainModel(args.modelPath, args.imgRootDir,
               args.trainLstPath, args.testLstPath)

    #------------------------------------------------------------------------------------------------

    # testModel(args.modelPath, args.imgRootDir, args.testLstPath, 100)

    #------------------------------------------------------------------------------------------------

    # evalModel(args.modelPath, args.imgRootDir, args.testLstPath, 64.0)
