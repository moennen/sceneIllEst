#!/usr/bin/python
""" Face Maps From Image Model

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
        "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libFaceAOVSampler/libFaceAOVSampler.so")

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, seed):
        params = np.array([batchSz, imgSz[0], imgSz[1]], dtype=np.float32)
        self.__ds = BufferDataSampler(
            DatasetTF.__lib, dbPath, imgRootDir, params, seed)
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32, tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            currImg, currUVs, currDepth, currNormals = self.__ds.getDataBuffers()
            yield (currImg, currUVs, currDepth, currNormals)


def loadValidationData(dataPath, dataRootDir, dataSz):

    im = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3))
    uv = np.full((dataSz[0], dataSz[1], dataSz[2], 2), 0.5)
    depth = np.full((dataSz[0], dataSz[1], dataSz[2], 1), 0.5)
    # normals are in [-1.0,1.0]
    normal = np.full((dataSz[0], dataSz[1], dataSz[2], 3), 0)

    n = 0

    # input
    with open(dataPath, 'r') as img_names_file:

        for data in img_names_file:

            if n >= dataSz[0]:
                break

            im[n, :, :, :] = loadResizeImgPIL(dataRootDir + "/" +
                                              data.rstrip('\n'), [dataSz[1], dataSz[2]], False)
            n = n + 1

    return im, uv, depth, normal

#-----------------------------------------------------------------------------------------------------
# UNIT TESTS
#-----------------------------------------------------------------------------------------------------


def testDataset(imgRootDir, trainPath):

    rseed = int(time.time())
    imgSz = [256, 256]

    tf.set_random_seed(rseed)
    random.seed(rseed)

    batchSz = 16

    trDs = DatasetTF(trainPath, imgRootDir, batchSz, imgSz, rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)

    with tf.Session() as sess:

        sess.run(trInit)

        for step in range(100):

            currImg, currUVs, currDepth, currNormals = sess.run(dsView)

            idx = random.randint(0, batchSz-1)

            cv.imshow('currImg', cv.cvtColor(currImg[idx], cv.COLOR_RGB2BGR))

            cv.imshow('currDepth', currDepth[idx])

            u, v = cv.split(currUVs[idx])
            cv.imshow('currU', u)
            cv.imshow('currV', v)

            cv.imshow('currNormals', cv.cvtColor(
                currNormals[idx], cv.COLOR_RGB2BGR))

            cv.waitKey(700)

#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class FaceMapsModelParams(Pix2PixParams):

    def __init__(self, modelPath, seed=int(time.time())):

        #
        # model 0 : charbonnier / resize / pix2pix_gen_p / bn
        #

        seed = 0

        Pix2PixParams.__init__(self, modelPath, seed)

        self.numMaxSteps = 175000
        self.numSteps = 175000
        self.backupStep = 250
        self.trlogStep = 250
        self.tslogStep = 250
        self.vallogStep = 250

        self.imgSzTr = [256, 256]
        self.batchSz = 32

        # bn vs no bn
        self.useBatchNorm = True
        self.nbChannels = 32
        self.nbInChannels = 3
        self.nbOutputChannels = 6
        self.kernelSz = 5
        self.stridedEncoder = True
        # strided vs resize
        self.stridedDecoder = False
        self.inDispRange = np.array([[0, 1, 2]])
        self.outDispRange = np.array([[0, 1, 2], [3, 4, 5]])
        self.alphaData = 1.0
        self.alphaDisc = 0.0
        self.linearImg = False

        self.model = pix2pix_gen_p

        # loss
        self.loss = pix2pix_charbonnier_loss

        self.update()

#-----------------------------------------------------------------------------------------------------
# VALIDATION
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgRootDir, imgLst):

    lp = FaceMapsModelParams(modelPath)
    lp.isTraining = False

    evalSz = [1, 256, 256, 3]

    inputsi = tf.placeholder(tf.float32, shape=evalSz, name="input")
    inputs = preprocess(inputsi)

    with tf.variable_scope("generator"):
        outputs = pix2pix_gen(inputs, 1, baseN, 0.0, False)

    # Persistency
    persistency = tf.train.Saver(filename=lp.modelFilename)

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

                uvdn = sess.run(outputs, feed_dict={inputsi: img})

                # show the sample
                cv.imshow('Input', cv.cvtColor(img[0], cv.COLOR_RGB2BGR))
                cv.imshow('Output_UVD', uvdn[0, :, :, 0:3])
                cv.imshow('Output_DNorm', uvdn[0, :, :, 2:5])

                cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------------------------------------


def trainModel(modelPath, imgRootDir, trainPath, testPath, valPath):

    lp = FaceMapsModelParams(modelPath)

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
    inUVi = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 2], name="input_uvs")
    inUV = tf.multiply(tf.subtract(inUVi, 0.5), 2.0)
    inDepthi = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 1], name="input_depth")
    inDepth = tf.multiply(tf.subtract(inDepthi, 0.5), 2.0)
    inNormali = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_normals")
    inNormal = inNormali

    inMaps = tf.concat([inUV, inDepth, inNormal], 3)

    # Optimizers
    [opts, loss, trSum, tsSum, valSum] = pix2pix_optimizer(inImg, inMaps, lp)

    # Validation
    valImg, valUVs, valDepth, valNormals = loadValidationData(
        valPath, imgRootDir, [lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1]])

    # Persistency
    persistency = tf.train.Saver(
        pad_step_number=True, max_to_keep=1, filename=lp.modelFilename)

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
            currImg, currUVs, currDepth, currNormals = sess.run(dsView)

            trFeed = {lp.isTraining: True,
                      inImgi: currImg,
                      inUVi: currUVs,
                      inDepthi: currDepth,
                      inNormali: currNormals}

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
                                                     inImgi: currImg,
                                                     inUVi: currUVs,
                                                     inDepthi: currDepth,
                                                     inNormali: currNormals})
                train_summary_writer.add_summary(summary, step)

            if step % lp.tslogStep == 0:

                sess.run(tsInit)
                currImg, currUVs, currDepth, currNormals = sess.run(dsView)
                tsLoss, summary = sess.run([loss, tsSum], feed_dict={lp.isTraining: False,
                                                                     inImgi: currImg,
                                                                     inUVi: currUVs,
                                                                     inDepthi: currDepth,
                                                                     inNormali: currNormals})

                test_summary_writer.add_summary(summary, step)

                print("{:08d}".format(step-1) +
                      " | lr = " + "{:.8f}".format(lp.learningRate.eval()) +
                      " | loss = " + "{:.5f}".format(tsLoss))

                # reset the training iterator
                sess.run(trInit)

            # validation
            if step % lp.vallogStep == 0:

                summary = sess.run(valSum, feed_dict={lp.isTraining: False,
                                                      inImgi: valImg,
                                                      inUVi: valUVs,
                                                      inDepthi: valDepth,
                                                      inNormali: valNormals})

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
