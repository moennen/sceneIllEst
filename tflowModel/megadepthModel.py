#!/usr/bin/python
""" Depth Map From Image Model

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
        "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libSparseDepthImgSampler/libSparseDepthImgSampler.so")

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, linearCS, rescale, seed):
        params = np.array([batchSz, imgSz[0], imgSz[1],
                           1.0 if linearCS else 0.0, 1.0 if rescale else 0.0], dtype=np.float32)
        self.__nds = 3
        self.__currds = 0
        self.__ds = [BufferDataSampler(
            DatasetTF.__lib, dbPath, imgRootDir, params, seed+1) for i in range(self.__nds)]
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            currImg, currDepth, currMask = self.__ds[self.__currds].getDataBuffers(
            )
            self.__currds = (self.__currds+1) % self.__nds
            yield (currImg, currDepth, currMask)


def loadValidationData(dataPath, dataRootDir, dataSz, linearCS=False):

    im = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3))
    depth = np.full((dataSz[0], dataSz[1], dataSz[2], 1), 0.5)

    n = 0

    # input
    with open(dataPath, 'r') as img_names_file:

        for data in img_names_file:

            if n >= dataSz[0]:
                break

            data = data.rstrip('\n').split()

            im[n, :, :, :] = loadResizeImgPIL(dataRootDir + "/" +
                                              data[0], [dataSz[1], dataSz[2]], linearCS)
            n = n + 1

    return im, depth

#-----------------------------------------------------------------------------------------------------
# UNIT TESTS
#-----------------------------------------------------------------------------------------------------


def testDataset(imgRootDir, trainPath):

    rseed = int(time.time())
    imgSz = [240, 320]

    tf.set_random_seed(rseed)
    random.seed(rseed)

    batchSz = 16

    trDs = DatasetTF(trainPath, imgRootDir, batchSz,
                     imgSz, False, True, rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)

    sess_config = tf.ConfigProto(device_count={'GPU': 0})

    with tf.Session(config=sess_config) as sess:

        sess.run(trInit)

        for step in range(100):

            currImg, currDepth, currMask = sess.run(dsView)

            idx = random.randint(0, batchSz-1)

            currTest = currImg
            currTest = increaseSize2x(currTest, 3, 'NHWC').eval()
            #currTest = tf.image.rgb_to_grayscale(currTest)
            #currTest = tf.square(filterLoG_3x3(currTest, 3, 'NHWC'))
            #currTest = currTest.eval()
            #cv.imshow('currTest', currTest[idx])
            cv.imshow('currTest', cv.cvtColor(currTest[idx], cv.COLOR_RGB2BGR))
            cv.imshow('currImg', cv.cvtColor(currImg[idx], cv.COLOR_RGB2BGR))

            currDepth = currDepth[idx]
            currDepthMin = np.amin(currDepth)
            currDepthMax = np.amax(currDepth)
            currDepth = (currDepth - currDepthMin) / \
                (currDepthMax-currDepthMin)

            print currDepthMin
            print currDepthMax

            cv.imshow('currDepth', cv.applyColorMap(
                (currDepth*255.0).astype(np.uint8), cv.COLORMAP_JET))

            cv.imshow('currMask', currMask[idx])

            cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class DepthPredictionModelParams(Pix2PixParams):

    def __init__(self, modelPath, data_format, seed=int(time.time())):

        #
        # exp0000 : 320x320x32x32 / pix2pix_gen_hg_0 / resize / lossScaleReg
        # exp0001 : 320x320x32 / pix2pix_hglass / lossScaleReg
        #
        seed = 0

        Pix2PixParams.__init__(self, modelPath, data_format, seed)

        self.numMaxSteps = 217500
        self.numSteps = 217500
        self.backupStep = 250
        self.trlogStep = 250
        self.tslogStep = 250
        self.vallogStep = 250

        self.imgSzTr = [240, 320]
        self.batchSz = 32

        # bn vs no bn
        self.useBatchNorm = True
        self.nbChannels = 32
        self.nbInChannels = 3
        self.nbOutputChannels = 1
        self.kernelSz = 5
        self.stridedEncoder = True
        # strided vs resize
        self.stridedDecoder = False
        self.inDispRange = np.array([[0, 1, 2]])
        self.outDispRange = np.array([[0, 0, 0]])
        self.alphaData = 1.0
        self.alphaDisc = 0.0
        self.alphaReg = 0.5
        self.linearImg = False
        self.rescaleImg = True

        # network arch function
        self.minimizeMemory = True
        self.doClassOut = True
        #self.model = pix2pix_gen_p
        #self.model = pix2pix_gen_hg_0
        self.model = pix2pix_hglass

        # loss function
        self.inMaski = tf.placeholder(tf.float32, shape=[
            self.batchSz, self.imgSzTr[0], self.imgSzTr[1], 1], name="input_mask")
        self.inMask = preprocess(self.inMaski, False, self.data_format)
        self.loss = self.loss_scale_reg

        self.doProfile = False

        self.wh_axis = [
            1, 2] if self.data_format == 'NHWC' else [2, 3]

        self.update()

    def meanstd_norm(self, outputs):
        # output depth map mean / stdDev normalization
        outputs_means = tf.reduce_mean(
            outputs, axis=self.wh_axis, keepdims=True)
        outputs_centered = tf.subtract(outputs, outputs_means)
        outputs_var = tf.reduce_mean(
            tf.square(outputs_centered), axis=self.wh_axis, keepdims=True)
        outputs_stdDev = tf.add(tf.sqrt(outputs_var), EPS)
        outputs_sc = tf.divide(outputs_centered, outputs_stdDev)

        return outputs_sc

    def megadepth_post(self, out):
        # output depth map mean / stdDev normalization
        outputs = tf.divide(1.0, tf.exp(out))
        outputs_maxs = tf.reduce_max(
            outputs, axis=self.wh_axis, keepdims=True)
        return tf.add(tf.multiply(tf.divide(outputs, outputs_maxs), 2.0), -1.0)

    def loss_scale_reg(self, outputs, targets):

        nvalid_b = tf.reduce_sum(self.inMask)

        diff = tf.multiply(tf.subtract(outputs, targets), self.inMask)

        data_loss_l2 = tf.divide(tf.reduce_sum(tf.square(diff)), nvalid_b)
        data_loss_scale_l2 = tf.square(
            tf.divide(tf.reduce_sum(diff), nvalid_b))

        data_loss = data_loss_l2 - data_loss_scale_l2

        reg_loss = self.loss_reg(outputs, targets)

        return data_loss, reg_loss

    def loss_reg(self, batchOutput, batchTargets):

        lossGrad = tf.constant(0.0)

        batchOutputResized = batchOutput
        batchTargetResized = batchTargets
        maskResized = self.inMask

        for i in range(1):

            batchErrGradX = l2(filterGradX_3x3(batchOutputResized, self.nbOutputChannels, self.data_format),
                               filterGradX_3x3(batchTargetResized, self.nbOutputChannels, self.data_format))
            batchErrGradY = l2(filterGradY_3x3(batchOutputResized, self.nbOutputChannels, self.data_format),
                               filterGradY_3x3(batchTargetResized,  self.nbOutputChannels, self.data_format))

            maskGradX = tf.math.floor(filterGradX_3x3(
                maskResized, 1, self.data_format))
            maskGradY = tf.math.floor(filterGradY_3x3(
                maskResized, 1, self.data_format))

            batchErrGrad = tf.divide(tf.reduce_sum(tf.multiply(batchErrGradX, maskGradX)), tf.reduce_sum(
                maskGradX)) + tf.divide(tf.reduce_sum(tf.multiply(batchErrGradY, maskGradY)), tf.reduce_sum(maskGradY))

            lossGrad = tf.add(lossGrad, batchErrGrad)

            batchOutputResized = reduceSize2x(
                batchOutputResized, self.nbOutputChannels, self.data_format)
            batchTargetResized = reduceSize2x(
                batchTargetResized, self.nbOutputChannels, self.data_format)
            maskResized = tf.math.floor(
                reduceSize2x(maskResized, 1, self.data_format))

        return lossGrad

    def optimizer(self, batchInput, batchTargets):

        with tf.variable_scope(self.getModelName()) as modelVs:
            batchOutput = self.model(batchInput, self)

        with tf.variable_scope(self.getModelName() + "_loss"):
            loss_data, loss_reg = self.loss_scale_reg(
                batchOutput, batchTargets)

        loss = loss_reg  # loss_data + self.alphaReg * loss_reg

        # dependencies for the batch normalization
        depends = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS) if self.useBatchNorm else []

        # optimizer
        opt, tvars, grads_and_vars = getOptimizerData(
            loss, depends, self, self.getModelName())

        # put summary on CPU to free some VRAM
        with tf.device('/cpu:0'):

            trSum = []
            addSummaryParams(trSum, self, tvars, grads_and_vars)
            trSum = tf.summary.merge(trSum, "Train")

            tsSum = []
            addSummaryScalar(tsSum, loss, "loss", "loss")
            addSummaryScalar(tsSum, loss_data, "loss", "data")
            addSummaryScalar(tsSum, loss_reg, "loss", "reg")

            addSummaryImages(tsSum, "Images", self,
                             [batchInput,  self.megadepth_post(
                                 batchTargets),  self.megadepth_post(batchOutput)],
                             [[0, 1, 2], [0, 0, 0], [0, 0, 0]])
            tsSum = tf.summary.merge(tsSum, "Test")

            valSum = []
            addSummaryImages(valSum, "Images", self,
                             [batchInput, self.megadepth_post(batchOutput)],
                             [[0, 1, 2], [0, 0, 0]])
            valSum = tf.summary.merge(valSum, "Val")

        return [opt, loss, trSum, tsSum, valSum]


#----------------------------------------------------------------------------------------------------
# VALIDATION
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgRootDir, imgLst, forceTrainingSize, maxSz, writeResults, data_format):

    lp = DepthPredictionModelParams(modelPath, data_format)
    lp.isTraining = False

    evalSz = [1, lp.imgSzTr[0], lp.imgSzTr[1], 3]

    inputsi = tf.placeholder(tf.float32, name="input")
    inputs = preprocess(inputsi, True, data_format)

    with tf.variable_scope("generator"):
        outputs = lp.model(inputs, lp)

    # Persistency
    persistency = tf.train.Saver(filename=lp.modelFilename)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    sess_config = tf.ConfigProto(device_count={'GPU': 0})
    # sess_config.gpu_options.allow_growth = True

    printSessionConfigProto(sess_config)

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

                depth = sess.run(outputs, feed_dict={inputsi: img})
                depth = postprocess(depth, True, data_format)
                # cv.normalize(depth[0], depth[0], 0, 1.0, cv.NORM_MINMAX)

                inputImg = (cv.cvtColor(
                    img[0], cv.COLOR_RGB2BGR)*255.0).astype(np.uint8)
                coloredDepth = cv.applyColorMap(
                    (depth[0] * 255.0).astype(np.uint8), cv.COLORMAP_JET)

                # show the sample
                cv.imshow('Input', inputImg)
                cv.imshow('Output', depth[0])  # coloredDepth)

                # write the results
                if writeResults:
                    outDName = imgRootDir + \
                        '/evalOutputMask/{:06d}_d.exr'.format(videoId)
                    cv.imwrite(outDName, depth[0])

                    outCDName = imgRootDir + \
                        '/evalOutputMask/{:06d}_cd.png'.format(videoId)
                    cv.imwrite(outCDName, coloredDepth)

                    outRGBName = imgRootDir + \
                        '/evalOutputMask/{:06d}_rgb.png'.format(videoId)
                    cv.imwrite(outRGBName, inputImg)

                cv.waitKey(0)

                videoId += 1

#-----------------------------------------------------------------------------------------------------
# EXPORT
#-----------------------------------------------------------------------------------------------------


def saveModel(modelPath, asText, data_format, convert_df):

    lp = DepthPredictionModelParams(modelPath, data_format)
    lp.isTraining = False

    mdSuff = '-last.pb.txt' if asText else '-last.pb'

    inputsi = tf.placeholder(tf.float32, name="adsk_inFront")
    if convert_df:
        inputs = preprocess(inputsi, True, data_format)
    else:
        inputs = tf.multiply(tf.subtract(inputsi, 0.5), 2.0)

    with tf.variable_scope("generator"):
        outputs = tf.exp(lp.model(inputs, lp))
        if convert_df:
            outputs = postprocess(outputs, False, data_format)

    outputNames = outputs.name
    outputNames = outputNames[:outputNames.find(":")]

    inputNames = inputsi.name
    inputNames = inputNames[:inputNames.find(":")]

    print "Exporting graph : " + inputNames + " -> " + outputNames + "  ( " + outputs.name + " )"

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
            outputNames,  # 'outFront',
            '', '', lp.modelFilename + mdSuff, True, '')


#-----------------------------------------------------------------------------------------------------
# TRAINING
#-----------------------------------------------------------------------------------------------------


def trainModel(modelPath, imgRootDir, trainPath, testPath, valPath, data_format):

    lp = DepthPredictionModelParams(modelPath, data_format)

    # Datasets / Iterators
    trDs = DatasetTF(trainPath, imgRootDir, lp.batchSz,
                     lp.imgSzTr, lp.linearImg, lp.rescaleImg, lp.rseed)
    tsDs = DatasetTF(testPath, imgRootDir, lp.batchSz,
                     lp.imgSzTr, lp.linearImg, lp.rescaleImg, lp.rseed)

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input placeholders
    inImgi = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_img")
    inImg = preprocess(inImgi, True, data_format)
    inDepthi = tf.placeholder(tf.float32, shape=[
        lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 1], name="input_depth")
    # inDepthi is meanStdDev normalized : no need to centered it
    inDepth = preprocess(inDepthi, False, data_format)

    # Optimizers
    [opts, loss, trSum, tsSum, valSum] = lp.optimizer(inImg, inDepth)

    # Validation
    valImg, valDepth = loadValidationData(
        valPath, imgRootDir, [lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1]], lp.linearImg)

    # Persistency
    persistency = tf.train.Saver(
        pad_step_number=True, max_to_keep=lp.modelNbToKeep, filename=lp.modelFilename)

    # Logger
    merged_summary_op = tf.summary.merge_all()

    # Params Initializer
    varInit = tf.global_variables_initializer()

    # Session configuration
    sess_config = tf.ConfigProto(device_count={'GPU': 1})
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:

        train_summary_writer = tf.summary.FileWriter(
            lp.tbLogsPath + "/Train", graph=sess.graph)
        test_summary_writer = tf.summary.FileWriter(lp.tbLogsPath + "/Test")
        val_summary_writer = tf.summary.FileWriter(lp.tbLogsPath + "/Val")

        # profiling
        if lp.doProfile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        # initialize params
        sess.run(varInit)
        sess.run(tf.local_variables_initializer())

        # Restore model if needed
        try:
            persistency.restore(sess, tf.train.latest_checkpoint(modelPath))
        except:
            print "Cannot load model:", sys.exc_info()[0]

        sess.run(trInit)

        step = lp.globalStep.eval(sess)

        # get each element of the training dataset until the end is reached
        while step < lp.numSteps:

            # print "------------------------------------------"

            # Get the next training batch
            # print "dsView"
            # start_ms = time.time()*1000.0
            currImg, currDepth, currMask = sess.run(
                dsView, options=run_options, run_metadata=run_metadata)
            # end_ms = time.time()*1000.0
            # print(end_ms-start_ms)

            if lp.doProfile:
                lp.profiler.update(timeline.Timeline(
                    run_metadata.step_stats).generate_chrome_trace_format(), "dsView")

            trFeed = {lp.isTraining: True,
                      inImgi: currImg,
                      inDepthi: currDepth,
                      lp.inMaski: currMask}

            step = step + 1

            # print "opts"
            # start_ms = time.time()*1000.0
            # Run optimization
            if step % lp.trlogStep == 0:
                _, summary, step = sess.run(
                    [opts, trSum, lp.globalStepInc], feed_dict=trFeed)
                train_summary_writer.add_summary(summary, step)
            else:
                _, step = sess.run([opts, lp.globalStepInc], feed_dict=trFeed,
                                   options=run_options, run_metadata=run_metadata)

                if lp.doProfile:
                    lp.profiler.update(timeline.Timeline(
                        run_metadata.step_stats).generate_chrome_trace_format(), "opts")

            # end_ms = time.time()*1000.0
            # print(end_ms-start_ms)

            # SUMMARIES

            if step % lp.trlogStep == 0:
                summary = sess.run(tsSum, feed_dict={lp.isTraining: False,
                                                     inImgi: currImg,
                                                     inDepthi: currDepth,
                                                     lp.inMaski: currMask})
                train_summary_writer.add_summary(summary, step)

            if step % lp.tslogStep == 0:

                sess.run(tsInit)
                currImg, currDepth, currMask = sess.run(dsView)
                tsLoss, summary = sess.run([loss, tsSum], feed_dict={lp.isTraining: False,
                                                                     inImgi: currImg,
                                                                     inDepthi: currDepth,
                                                                     lp.inMaski: currMask})

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
                                                      inDepthi: valDepth})

                val_summary_writer.add_summary(summary, step)

            # PERSISTENCY
            if step % lp.backupStep == 0:
                persistency.save(sess, lp.modelFilename,
                                 global_step=lp.globalStep)

        # WRITE PROFILER
        if lp.doProfile:
            lp.profiler.save()


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    #------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode", help="mode : test_ds / train / eval /save")

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

    parser.add_argument("--uff", dest='uff',
                        default=False, action='store_true')

    args = parser.parse_args()

    data_format = 'NHWC' if args.nhwc else 'NCHW'

    #------------------------------------------------------------------------------------------------
    if args.mode == 'test_ds':
        testDataset(args.imgRootDir, args.trainLstPath)

    #------------------------------------------------------------------------------------------------
    if args.mode == 'train':
        trainModel(args.modelPath, args.imgRootDir,
                   args.trainLstPath, args.testLstPath, args.valLstPath, data_format)

    #------------------------------------------------------------------------------------------------
    if args.mode == 'eval':
        evalModel(args.modelPath, args.imgRootDir,
                  args.valLstPath, True, 512, False, data_format)

    #------------------------------------------------------------------------------------------------
    if args.mode == 'save':
        saveModel(args.modelPath, False, data_format, not args.uff)
