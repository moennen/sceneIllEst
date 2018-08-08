#!/usr/bin/python
""" Depth Map From Image Model

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
        "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libDepthImgSampler/libDepthImgSampler.so")

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, seed):
        params = np.array([batchSz, imgSz[0], imgSz[1]], dtype=np.float32)
        self.__ds = BufferDataSampler(
            DatasetTF.__lib, dbPath, imgRootDir, params, seed)
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            currImg, currDepth = self.__ds.getDataBuffers()
            yield (currImg, currDepth)


def loadValidationData(dataPath, dataRootDir, dataSz):

    im = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3))
    depth = np.full((dataSz[0], dataSz[1], dataSz[2], 1), 0.5)

    n = 0

    # input
    with open(dataPath, 'r') as img_names_file:

        for data in img_names_file:

            if n >= dataSz[0]:
                break

            im[n, :, :, :] = loadResizeImgPIL(dataRootDir + "/" +
                                              data.rstrip('\n'), [dataSz[1], dataSz[2]], False)
            n = n + 1

    return im, depth


#-----------------------------------------------------------------------------------------------------
# MODEL
#-----------------------------------------------------------------------------------------------------


def model(imgs):

    return base_model(imgs)

#-----------------------------------------------------------------------------------------------------
# UNIT TESTS
#-----------------------------------------------------------------------------------------------------


def testDataset(imgRootDir, trainPath):

    rseed = int(time.time())
    imgSz = [128, 128]

    tf.set_random_seed(rseed)

    batchSz = 16

    trDs = DatasetTF(trainPath, imgRootDir, batchSz, imgSz, rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)

    with tf.Session() as sess:

        sess.run(trInit)

        for step in range(100):

            currImg, currDepth = sess.run(dsView)

            idx = random.randint(0, batchSz-1)

            cv.imshow('currImg', cv.cvtColor(currImg[idx], cv.COLOR_RGB2BGR))
            cv.imshow('currDepth', currDepth[idx])

            cv.waitKey(700)

#-----------------------------------------------------------------------------------------------------
# VALIDATION
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgRootDir, imgLst):

    modelFilename = modelPath + "/tfData"
    baseN = 64
    evalSz = [1, 256, 256, 3]

    inputsi = tf.placeholder(tf.float32, shape=evalSz, name="input")
    inputs = preprocess(inputsi)

    with tf.variable_scope("generator"):
        outputs = pix2pix_gen(inputs, 1, baseN, 0.0, False)

    # Persistency
    persistency = tf.train.Saver(filename=modelFilename)

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

                img = [loadResizeImgPIL(imgRootDir + "/" +
                                        data.rstrip('\n'), [evalSz[1], evalSz[2]], False)]

                depth = sess.run(outputs, feed_dict={inputsi: img})

                # show the sample
                cv.imshow('Input', cv.cvtColor(img[0], cv.COLOR_RGB2BGR))
                cv.imshow('Output', depth[0])
                cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------------------------------------


def trainModel(modelPath, imgRootDir, trainPath, testPath, valPath):

    lp = LearningParams(modelPath)  # , 20160704)
    lp.numSteps = 250000
    lp.backupStep = 300
    lp.imgSzTr = [128, 128]
    lp.batchSz = 128
    baseN = 64
    alpha_data = 1.0
    alpha_disc = 0.0
    lp.update()
    profile = True

    # Datasets / Iterators
    trDs = DatasetTF(trainPath, imgRootDir, lp.batchSz, lp.imgSzTr, lp.rseed)
    tsDs = DatasetTF(testPath, imgRootDir, lp.batchSz, lp.imgSzTr, lp.rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input placeholders
    inImgi = tf.placeholder(tf.float32, shape=[
                            lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_img")
    inImg = preprocess(inImgi)
    inDepthi = tf.placeholder(tf.float32, shape=[
                              lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 1], name="input_depth")
    inDepth = tf.multiply(tf.subtract(inDepthi, 0.5), 2.0)

    # Optimizers
    [opts, loss, trSum, tsSum] = pix2pix_optimizer(inImg, inDepth,
                                                   lp.learningRate, alpha_data, alpha_disc, lp.dropoutProb,
                                                   lp.isTraining, baseN, False,
                                                   np.array([[0, 1, 2]]),
                                                   np.array([[0, 0, 0]]))

    # Validation
    valImg, valDepth = loadValidationData(
        valPath, imgRootDir, [lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1]])

    # Persistency
    persistency = tf.train.Saver(
        pad_step_number=True, max_to_keep=3, filename=lp.modelFilename)

    # Logger
    merged_summary_op = tf.summary.merge_all()

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
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
            currImg, currDepth = sess.run(dsView)

            trFeed = {lp.dropoutProb: 0.01,
                      lp.isTraining: True,
                      inImgi: currImg,
                      inDepthi: currDepth}

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
                summary = sess.run(tsSum, feed_dict={lp.dropoutProb: 0.0,
                                                     lp.isTraining: False,
                                                     inImgi: currImg,
                                                     inDepthi: currDepth})
                train_summary_writer.add_summary(summary, step)

            if step % lp.tslogStep == 0:

                sess.run(tsInit)
                currImg, currDepth = sess.run(dsView)
                tsLoss, summary = sess.run([loss, tsSum], feed_dict={lp.dropoutProb: 0.0,
                                                                     lp.isTraining: False,
                                                                     inImgi: currImg,
                                                                     inDepthi: currDepth})

                test_summary_writer.add_summary(summary, step)

                print("{:08d}".format(step-1) +
                      " | lr = " + "{:.8f}".format(lp.learningRate.eval()) +
                      " | loss = " + "{:.5f}".format(tsLoss))

                # reset the training iterator
                sess.run(trInit)

            # validation
            if step % lp.vallogStep == 0:

                summary = sess.run(tsSum, feed_dict={lp.dropoutProb: 0.0,
                                                     lp.isTraining: False,
                                                     inImgi: valImg,
                                                     inDepthi: valDepth})

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

    #testDataset(args.imgRootDir, args.trainLstPath)

    #------------------------------------------------------------------------------------------------

    trainModel(args.modelPath, args.imgRootDir,
               args.trainLstPath, args.testLstPath, args.valLstPath)

    #------------------------------------------------------------------------------------------------

    # testModel(args.modelPath, args.imgRootDir, args.testLstPath, 100)

    #------------------------------------------------------------------------------------------------

    # evalModel(args.modelPath, args.imgRootDir, args.valLstPath)
