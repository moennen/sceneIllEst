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
        "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libImageLstSampler/libImageLstSampler.so")

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, seed, disc=False):
        params = np.array([batchSz, imgSz[0], imgSz[1], 3], dtype=np.float32)
        self.__ds = BufferDataSampler(
            DatasetTF.__lib, dbPath, imgRootDir, params, seed)

        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            leftImg, rightImg, randImg = self.__ds.getDataBuffers()
            yield (leftImg, rightImg, randImg)


def loadValidationData(dataPath, dataRootDir, dataSz):

    lim = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3))
    rim = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3))

    n = 0

    # input
    with open(dataPath, 'r') as img_names_file:

        for data in img_names_file:

            data = data.rstrip('\n').split()

            if n >= dataSz[0]:
                break

            lim[n, :, :, :] = loadResizeImgPIL(
                dataRootDir + "/" + data[0], [dataSz[1], dataSz[2]], False)
            rim[n, :, :, :] = loadResizeImgPIL(
                dataRootDir + "/" + data[1], [dataSz[1], dataSz[2]], False)

            n = n + 1

    return lim, rim

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

            imLeft, imRight, imRand = sess.run(dsView)

            idx = random.randint(0, batchSz-1)

            cv.imshow('leftImg', cv.cvtColor(imLeft[idx], cv.COLOR_RGB2BGR))
            cv.imshow('rightImg', cv.cvtColor(imRight[idx], cv.COLOR_RGB2BGR))
            cv.imshow('randImg', cv.cvtColor(imRand[idx], cv.COLOR_RGB2BGR))

            cv.waitKey(10)

#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class ImDistMapModelParams(Pix2PixParams):

    def __init__(self, modelPath, data_format, seed=int(time.time())):

        #
        # exp0000 : 128x128x16x8 / charbonnier / strided / pix2pix_gen_p / no bn
        #
        #

        Pix2PixParams.__init__(self, modelPath, data_format, seed)

        self.numMaxSteps = 217500
        self.numSteps = 2175000
        self.backupStep = 150
        self.trlogStep = 150
        self.tslogStep = 150
        self.vallogStep = 150

        # dimensions
        self.imgSzTr = [128, 128]
        self.batchSz = 32

        # bn vs no bn
        self.useBatchNorm = False
        self.nbChannels = 32
        self.nbInChannels = 3
        self.nbOutputChannels = 3
        self.kernelSz = 5
        self.stridedEncoder = True
        # strided vs resize
        self.stridedDecoder = True
        self.inDispRange = np.array([0, 1, 2])
        self.outDispRange = np.array([0, 1, 2])
        self.alphaData = 1.0
        self.alphaDisc = 0.0

        # model
        self.model = pix2pix_gen_p
        # loss
        self.loss = self.lossDist3

        self.update()

    def getModelName(self):
        return "imDistMapModel"

    def lossDist3(self, imBatchLeft, imBatchRight, imBatchRand):
        return pix2pix_l2_loss(imBatchLeft, imBatchRight)

#-----------------------------------------------------------------------------------------------------
# VALIDATION
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgRootDir, imgLst, forceTrainingSize, maxSz, writeResults, data_format):

    lp = ImDistMapModelParams(modelPath, data_format)
    lp.isTraining = False

    evalSz = [1, 256, 256, 3]

    inputsi = tf.placeholder(tf.float32, name="image")
    inputs = preprocess(inputsi, True, data_format)

    with tf.variable_scope(lp.getModelName()):
        outputs = lp.model(inputs, lp)

    # Persistency
    persistency = tf.train.Saver(filename=lp.modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    sess_config = tf.ConfigProto(device_count={'GPU': 0})
    # sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:

        # initialize params
        sess.run(varInit)

        # Restore model if needed
        persistency.restore(sess, tf.train.latest_checkpoint(modelPath))

        # input
        with open(imgLst, 'r') as img_names_file:

            videoId = 0
            for data in img_names_file:

                data = data.rstrip('\n').split()

                imgName = imgRootDir + "/" + data[0]

                if forceTrainingSize:
                    img = [loadResizeImgPIL(
                        imgName, [evalSz[1], evalSz[2]], lp.linearImg)]
                else:
                    img = [loadImgPIL(imgName, lp.linearImg)]

                if maxSz > 0:
                    ds = min(
                        1.0, min(float(maxSz)/img[0].shape[0], float(maxSz)/img[0].shape[1]))
                    img[0] = cv.resize(img[0], dsize=(
                        0, 0), fx=ds, fy=ds, interpolation=cv.INTER_AREA)

                mapDist = sess.run(outputs, feed_dict={inputsi: img})

                imOut = (cv.cvtColor(
                    img[0], cv.COLOR_RGB2BGR)*255.0).astype(np.uint8)
                mapDistOut = cv.cvtColor(0.5*(mapDist+1.0), cv.COLOR_RGB2BGR)

                # show the sample
                cv.imshow('Input Image', imOut)
                cv.imshow('Output Distance Map', mapDistOut)

                if writeResults:

                    pathIm = imgRootDir + \
                        '/evalOutput/rgb_{:06d}.png'.format(videoId)
                    cv.imwrite(pathIm, imOut)

                    pathMapDist = imgRootDir + \
                        '/evalOutput/dist_map_{:06d}.png'.format(videoId)
                    cv.imwrite(pathMapDist, mapDistOut)

                cv.waitKey(10)

                videoId += 1

#-----------------------------------------------------------------------------------------------------
# EXPORT
#-----------------------------------------------------------------------------------------------------


def saveModel(modelPath, asText, data_format):

    lp = ImDistMapModelParams(modelPath, data_format)
    lp.isTraining = False

    mdSuff = '-last.pb.txt' if asText else '-last.pb'

    inputsi = tf.placeholder(tf.float32, name="adsk_inFront")
    inputs = tf.multiply(tf.subtract(inputsi, 0.5), 2.0)

    with tf.variable_scope(lp.getModelName()):
        outputs = lp.model(inputs, lp)

    outputsSz = outputs.get_shape()
    sliceSz = [outputsSz[0], outputsSz[1], outputsSz[2], 1]

    outputNames = "adsk_outDistMap"

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
# MODEL
#-----------------------------------------------------------------------------------------------------


def pix2pix_twins_optimizer(imBatchLeft, imBatchRight, imBatchRand, params):

    with tf.variable_scope(params.getModelName()) as modelVs:
        #    outBatchLeft = params.model(imBatchLeft, params)
        # with tf.variable_scope(modelVs, reuse=True):
        outBatchRight = params.model(imBatchRight, params)
    # with tf.variable_scope(modelVs, reuse=True):
    #    outBatchRand = params.model(imBatchRand, params)

    outBatchLeft = imBatchLeft
    outBatchRand = imBatchRand

    with tf.variable_scope(params.getModelName() + "_loss"):
        gen_loss_data = params.loss(outBatchLeft, outBatchRight, outBatchRand)

    loss = params.alphaData * gen_loss_data

    opt, tvars, grads_and_vars = getOptimizerData(loss, params)

    trSum = []
    addSummaryParams(trSum, params, tvars, grads_and_vars)
    trSum = tf.summary.merge(trSum, "Train")

    tsSum = []
    addSummaryScalar(tsSum, loss, "data", "loss")

    addSummaryImages(tsSum, "Images", params,
                     [imBatchLeft, outBatchLeft, imBatchRight,
                         outBatchRight, imBatchRand, outBatchRand],
                     [params.inDispRange, params.outDispRange, params.inDispRange, params.outDispRange, params.inDispRange, params.outDispRange])
    tsSum = tf.summary.merge(tsSum, "Test")

    valSum = []
    addSummaryImages(valSum, "Images", params,
                     [imBatchLeft, outBatchLeft, imBatchRight, outBatchRight],
                     [params.inDispRange, params.outDispRange, params.inDispRange, params.outDispRange])
    valSum = tf.summary.merge(valSum, "Val")

    return [opt, loss, trSum, tsSum, valSum]

#-----------------------------------------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------------------------------------


def trainModel(modelPath, imgRootDir, trainPath, testPath, valPath, data_format):

    lp = ImDistMapModelParams(modelPath, data_format)

    # Datasets / Iterators
    trDs = DatasetTF(trainPath, imgRootDir, lp.batchSz, lp.imgSzTr, lp.rseed)
    tsDs = DatasetTF(testPath, imgRootDir, lp.batchSz, lp.imgSzTr, lp.rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input placeholders
    imLeftInRaw = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_left_image")
    imLeftIn = preprocess(imLeftInRaw, True, data_format)

    imRightInRaw = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_right_image")
    imRightIn = preprocess(imRightInRaw, True, data_format)

    imRandInRaw = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_rand_image")
    imRandIn = preprocess(imRandInRaw, True, data_format)

    # Optimizers
    [opts, loss, trSum, tsSum, valSum] = pix2pix_twins_optimizer(
        imLeftIn, imRightIn, imRandIn, lp)

    # Validation
    imLeftVal, imRightVal = loadValidationData(
        valPath, imgRootDir, [lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1]])

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
            imLeftSple, imRightSple, imRandSple = sess.run(dsView)

            trFeed = {lp.isTraining: True,
                      imLeftInRaw: imLeftSple,
                      imRightInRaw: imRightSple,
                      imRandInRaw: imRandSple}

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
                                                     imLeftInRaw: imLeftSple,
                                                     imRightInRaw: imRightSple,
                                                     imRandInRaw: imRandSple})
                train_summary_writer.add_summary(summary, step)

            if step % lp.tslogStep == 0:

                sess.run(tsInit)
                imLeftSple, imRightSple, imRandSple = sess.run(dsView)
                tsLoss, summary = sess.run([loss, tsSum], feed_dict={lp.isTraining: False,
                                                                     imLeftInRaw: imLeftSple,
                                                                     imRightInRaw: imRightSple,
                                                                     imRandInRaw: imRandSple})

                test_summary_writer.add_summary(summary, step)

                print("{:08d}".format(step-1) +
                      " | lr = " + "{:.8f}".format(lp.learningRate.eval()) +
                      " | loss = " + "{:.5f}".format(tsLoss))

                # reset the training iterator
                sess.run(trInit)

            # validation
            if step % lp.vallogStep == 0:

                summary = sess.run(valSum, feed_dict={lp.isTraining: False,
                                                      imLeftInRaw: imLeftVal,
                                                      imRightInRaw: imRightVal})

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

    parser.add_argument("--nhwc", dest='nhwc',
                        default=False, action='store_true')

    args = parser.parse_args()

    data_format = 'NHWC' if args.nhwc else 'NCHW'

    #------------------------------------------------------------------------------------------------

    # testDataset(args.imgRootDir, args.trainLstPath)

    #------------------------------------------------------------------------------------------------

    trainModel(args.modelPath, args.imgRootDir,
               args.trainLstPath, args.testLstPath, args.valLstPath, data_format)

    #------------------------------------------------------------------------------------------------

    # testModel(args.modelPath, args.imgRootDir, args.testLstPath, 100, data_format)

    #------------------------------------------------------------------------------------------------

    # evalModel(args.modelPath, args.imgRootDir,
    #          args.valLstPath, False, 640, True, data_format)

    #------------------------------------------------------------------------------------------------

    # saveModel(args.modelPath, False, data_format)
