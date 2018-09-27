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

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, seed):
        params = np.array([batchSz, imgSz[0], imgSz[1]], dtype=np.float32)
        self.__ds = BufferDataSampler(
            DatasetTF.__lib, dbPath, imgRootDir, params, seed)
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            currImg, currUVD, currNorm, currId, currPos = self.__ds.getDataBuffers()
            yield (currImg, currUVD, currNorm, currId, currPos)


def loadValidationData(dataPath, dataRootDir, dataSz):

    im = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3), dtype=np.float32)
    pos = np.zeros((dataSz[0], dataSz[1], dataSz[2], 2), dtype=np.float32)

    # fill the positions
    arrPos = np.zeros((dataSz[1], dataSz[2], 2), dtype=np.float32)
    for x in range(dataSz[1]):
        for y in range(dataSz[2]):
            arrPos[x, y, 0] = (2.0*x/(dataSz[1]-1)) - 1.0
            arrPos[x, y, 1] = (2.0*y/(dataSz[2]-1)) - 1.0

    n = 0

    # input
    with open(dataPath, 'r') as img_names_file:

        for data in img_names_file:

            data = data.rstrip('\n').split()

            if n >= dataSz[0]:
                break

            pos[n, :, :, :] = arrPos
            im[n, :, :, :] = loadResizeImgPIL(
                dataRootDir + "/" + data[0], [dataSz[1], dataSz[2]], False)
            n = n + 1

    return im, pos

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

    arrId = np.full((imgSz[0], imgSz[1], 3), 0.0, dtype=np.float32)
    arrPos = np.full((imgSz[0], imgSz[1], 3), 0.0, dtype=np.float32)

    with tf.Session() as sess:

        sess.run(trInit)

        for step in range(100):

            currImg, currUVD, currNorm, currId, currPos = sess.run(dsView)

            idx = random.randint(0, batchSz-1)

            cv.imshow('currImg', cv.cvtColor(currImg[idx], cv.COLOR_RGB2BGR))
            cv.imshow('currUVD', cv.cvtColor(currUVD[idx], cv.COLOR_RGB2BGR))
            cv.imshow('currNormals', cv.cvtColor(
                currNorm[idx], cv.COLOR_RGB2BGR))

            arrId[:, :, 0:2] = (currId[idx] + 1.0) * 0.5
            cv.imshow('currIds', cv.cvtColor(arrId, cv.COLOR_RGB2BGR))

            arrPos[:, :, 0:2] = (currPos[idx] + 1.0) * 0.5
            cv.imshow('currPos', cv.cvtColor(arrPos, cv.COLOR_RGB2BGR))

            cv.waitKey(50)

#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class FaceMapsModelParams(Pix2PixParams):

    def __init__(self, modelPath, data_format, seed=int(time.time())):

        #
        # exp0000 : 256x256x32 / charbonnier / resize / pix2pix_gen_p / bn / d00
        # exp0001 : 296x296x32 / charbonnier / resize / pix2pix_gen_p / bn / d01
        # exp0002 : 256x256x32 / charbonnier / deconv / pix2pix_gen_p / bn / d01
        # exp0003 : 256x256x32x32 / charbonnier / deconv / pix2pix_gen_p / bn / d02
        #
        # -- New Model with IM / UVD / NORM / ID / POS
        #
        # exp0005 : 256x256x32x36 / charbonnier / deconv / pix2pix_gen_p / bn / d02
        # - transition to new pix2pix framework
        # exp0006 : 256x256x32x36 / charbonnier / deconv / pix2pix_gen_p / bn / d02
        #

        Pix2PixParams.__init__(self, modelPath, data_format, seed)

        self.numMaxSteps = 217500
        self.numSteps = 2175000
        self.backupStep = 250
        self.trlogStep = 250
        self.tslogStep = 250
        self.vallogStep = 250

        # dimensions
        self.imgSzTr = [320, 320]
        self.batchSz = 32

        # bn vs no bn
        self.useBatchNorm = True
        self.nbChannels = 36
        self.nbInChannels = 5
        self.nbOutputChannels = 8
        self.kernelSz = 5
        self.stridedEncoder = True
        # strided vs resize
        self.stridedDecoder = False
        self.inDispRange = np.array([[0, 1, 2], [3, 4, 4]])
        self.outDispRange = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 7]])
        self.alphaData = 1.0
        self.alphaDisc = 0.0
        self.linearImg = False

        # model
        self.model = pix2pix_gen_p
        # loss
        self.loss = self.loss_maps

        self.update()

    def loss_maps(self, batchOutput, batchTargets):

        batchTargetsUV, batchTargetsD, batchTargetsNorm, batchTargetsID = tf.split(
            batchTargets, [2, 1, 3, 2], axis=3 if self.data_format == 'NHWC' else 1)
        batchOutputUV, batchOutputD, batchOutputNorm, batchOutputID = tf.split(
            batchOutput, [2, 1, 3, 2], axis=3 if self.data_format == 'NHWC' else 1)

        return [pix2pix_charbonnier_loss(batchTargetsUV, batchOutputUV),
                pix2pix_charbonnier_loss(batchTargetsD, batchOutputD),
                pix2pix_charbonnier_loss(batchTargetsNorm, batchOutputNorm),
                pix2pix_charbonnier_loss(batchTargetsID, batchOutputID)]

    def optimizer(self, batchInput, batchTargets):

        with tf.variable_scope(self.getModelName()) as modelVs:
            batchOutput = self.model(batchInput, self)

        with tf.variable_scope(self.getModelName() + "_loss"):
            loss_uv, loss_d, loss_norm, loss_id = self.loss(
                batchTargets, batchOutput)

        loss = self.alphaData * 0.25 * (loss_uv + loss_d + loss_norm + loss_id)

        depends = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS) if self.useBatchNorm else []
        opt, tvars, grads_and_vars = getOptimizerData(loss, depends, self)

        trSum = []
        addSummaryParams(trSum, self, tvars, grads_and_vars)
        trSum = tf.summary.merge(trSum, "Train")

        tsSum = []
        addSummaryScalar(tsSum, loss, "data", "loss")
        addSummaryScalar(tsSum, loss_uv, "data", "loss_uv")
        addSummaryScalar(tsSum, loss_d, "data", "loss_d")
        addSummaryScalar(tsSum, loss_norm, "data", "loss_norm")
        addSummaryScalar(tsSum, loss_id, "data", "loss_id")

        addSummaryImages(tsSum, "Images", self,
                         [batchInput, batchInput, batchTargets, batchTargets,
                             batchTargets, batchOutput, batchOutput, batchOutput],
                         [[0, 1, 2], [3, 4, 4], [0, 1, 2], [3, 4, 5], [6, 7, 7], [0, 1, 2], [3, 4, 5], [6, 7, 7]])
        tsSum = tf.summary.merge(tsSum, "Test")

        valSum = []
        addSummaryImages(valSum, "Images", self,
                         [batchInput, batchInput, batchOutput,
                             batchOutput, batchOutput],
                         [[0, 1, 2], [3, 4, 4], [0, 1, 2], [3, 4, 5], [6, 7, 7]])
        valSum = tf.summary.merge(valSum, "Val")

        return [opt, loss, trSum, tsSum, valSum]

#-----------------------------------------------------------------------------------------------------
# VALIDATION
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgRootDir, imgLst, forceTrainingSize, maxSz, writeResults, data_format):

    lp = FaceMapsModelParams(modelPath, data_format)
    lp.isTraining = False

    evalSz = [1, 256, 256, 3]

    inputsi = tf.placeholder(tf.float32, name="input")
    inputs = preprocess(inputsi, True, data_format)

    with tf.variable_scope("generator"):
        outputs = lp.model(inputs, lp)
        outputs = postprocess(outputs, False, data_format)

    # Persistency
    persistency = tf.train.Saver(filename=lp.modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    sess_config = tf.ConfigProto(device_count={'GPU': 1})
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

                uvdn = sess.run(outputs, feed_dict={inputsi: img})

                inputImg = (cv.cvtColor(
                    img[0], cv.COLOR_RGB2BGR)*255.0).astype(np.uint8)
                uvd = cv.cvtColor(
                    0.5*(uvdn[0, :, :, 0:3]+1.0), cv.COLOR_RGB2BGR)
                n = cv.cvtColor(uvdn[0, :, :, 3:6], cv.COLOR_RGB2BGR)

                # show the sample
                cv.imshow('Input', inputImg)
                cv.imshow('UVDepth', uvd)
                cv.imshow('Normals', 0.5*(n+1.0))

                if writeResults:

                    outUVDName = imgRootDir + \
                        '/evalOutput/uvd_{:06d}.exr'.format(videoId)
                    cv.imwrite(outUVDName, uvd)

                    outNormName = imgRootDir + \
                        '/evalOutput/norm_{:06d}.exr'.format(videoId)
                    cv.imwrite(outNormName, n)

                    outRGBName = imgRootDir + \
                        '/evalOutput/rgb_{:06d}.png'.format(videoId)
                    cv.imwrite(outRGBName, inputImg)

                cv.waitKey(10)

                videoId += 1

#-----------------------------------------------------------------------------------------------------
# EXPORT
#-----------------------------------------------------------------------------------------------------


def saveModel(modelPath, asText, data_format):

    lp = FaceMapsModelParams(modelPath, data_format)
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
    # outputUVD, outputNormals = tf.split(
    #    outputs, [3, 3], axis=3 if data_format == 'NHWC' else 1)

    if data_format == 'NHWC':
        uvd_size = tf.constant([-1, -1, -1, 3], dtype=tf.int32)
        uvd_begin = tf.constant([0, 0, 0, 0], dtype=tf.int32)
        norm_size = tf.constant([-1, -1, -1, 3], dtype=tf.int32)
        norm_begin = tf.constant([0, 0, 0, 3], dtype=tf.int32)
    else:
        uvd_size = tf.constant([-1, 3, -1, -1], dtype=tf.int32)
        uvd_begin = tf.constant([0, 0, 0, 0], dtype=tf.int32)
        norm_size = tf.constant([-1, 3, -1, -1], dtype=tf.int32)
        norm_begin = tf.constant([0, 3, 0, 0], dtype=tf.int32)

    outputUVD = tf.slice(outputs, uvd_begin, uvd_size)
    outputUVD = tf.multiply(tf.add(outputUVD, 1.0), 0.5, name="adsk_outUVD")
    # outputNormals = tf.identity(outputNormals, name="adsk_outNormals")

    outputNormals = tf.slice(
        outputs, norm_begin, norm_size, name="adsk_outNormals")

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


def trainModel(modelPath, imgRootDir, trainPath, testPath, valPath, data_format):

    lp = FaceMapsModelParams(modelPath, data_format)

    # Datasets / Iterators
    trDs = DatasetTF(trainPath, imgRootDir, lp.batchSz, lp.imgSzTr, lp.rseed)
    tsDs = DatasetTF(testPath, imgRootDir, lp.batchSz, lp.imgSzTr, lp.rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input placeholders

    batchImRaw = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_img")
    batchIm = preprocess(batchImRaw, True, data_format)
    batchPosRaw = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 2], name="input_pos")
    batchPos = preprocess(batchPosRaw, False, data_format)
    batchInput = tf.concat([batchIm, batchPos],
                           axis=3 if data_format == 'NHWC' else 1)

    batchUVDRaw = tf.placeholder(
        tf.float32, shape=[
            lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_uvd")
    batchUVD = preprocess(batchUVDRaw, True, data_format)
    batchNormRaw = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_normals")
    batchNorm = preprocess(batchNormRaw, False, data_format)
    batchIdRaw = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 2], name="input_id")
    batchId = preprocess(batchIdRaw, False, data_format)

    batchTargets = tf.concat([batchUVD, batchNorm, batchId],
                             axis=3 if data_format == 'NHWC' else 1)

    # Optimizers
    [opts, loss, trSum, tsSum, valSum] = lp.optimizer(batchInput, batchTargets)

    # Validation
    batchImVal, batchPosVal = loadValidationData(
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
            batchImSple, batchUVDSple, batchNormSple, batchIdSple, batchPosSple = sess.run(
                dsView)

            trFeed = {lp.isTraining: True,
                      batchImRaw: batchImSple,
                      batchUVDRaw: batchUVDSple,
                      batchNormRaw: batchNormSple,
                      batchIdRaw: batchIdSple,
                      batchPosRaw: batchPosSple}

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
                                                     batchImRaw: batchImSple,
                                                     batchUVDRaw: batchUVDSple,
                                                     batchNormRaw: batchNormSple,
                                                     batchIdRaw: batchIdSple,
                                                     batchPosRaw: batchPosSple})
                train_summary_writer.add_summary(summary, step)

            if step % lp.tslogStep == 0:

                sess.run(tsInit)
                batchImSple, batchUVDSple, batchNormSple, batchIdSple, batchPosSple = sess.run(
                    dsView)
                tsLoss, summary = sess.run([loss, tsSum], feed_dict={lp.isTraining: False,
                                                                     batchImRaw: batchImSple,
                                                                     batchUVDRaw: batchUVDSple,
                                                                     batchNormRaw: batchNormSple,
                                                                     batchIdRaw: batchIdSple,
                                                                     batchPosRaw: batchPosSple})

                test_summary_writer.add_summary(summary, step)

                print("{:08d}".format(step-1) +
                      " | lr = " + "{:.8f}".format(lp.learningRate.eval()) +
                      " | loss = " + "{:.5f}".format(tsLoss))

                # reset the training iterator
                sess.run(trInit)

            # validation
            if step % lp.vallogStep == 0:

                summary = sess.run(valSum, feed_dict={lp.isTraining: False,
                                                      batchImRaw: batchImVal,
                                                      batchPosRaw: batchPosVal})

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
