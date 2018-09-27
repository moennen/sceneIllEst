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
        self.__ds = BufferDataSampler(
            DatasetTF.__lib, dbPath, imgRootDir, params, seed)
        self.data = tf.data.Dataset.from_generator(
            self.sample, (tf.float32, tf.float32, tf.float32))

    def sample(self):
        for i in itertools.count(1):
            currImg, currDepth, currMask = self.__ds.getDataBuffers()
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
    imgSz = [256, 256]

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
        # model 0 : scale_l2 / resize / pix2pix_gen_p / bn
        # model 1 : scale_charbonnier / resize / pix2pix_gen_p / bn
        # model 2 : 296x296x3 / MeanStdDev_charbonnier_0.7 / resize / pix2pix_gen_p / bn
        # model 3 : 296x296x3 / loss_masked_meanstd_norm_charbonnier / stride / pix2pix_gen_p / bn
        # model 4 : 296x296x3 / loss_masked_meanstd_norm_charbonnier + disc / stride / pix2pix_gen_p / bn
        #
        # exp0005 : 296x296x3 / loss_masked_meanstd_norm_charbonnier / stride / pix2pix_gen_p / bn / md only
        #

        seed = 0

        Pix2PixParams.__init__(self, modelPath, data_format, seed)

        self.numMaxSteps = 217500
        self.numSteps = 217500
        self.backupStep = 250
        self.trlogStep = 250
        self.tslogStep = 250
        self.vallogStep = 250

        self.imgSzTr = [296, 296]
        self.batchSz = 48

        # bn vs no bn
        self.useBatchNorm = True
        self.nbChannels = 32
        self.nbInChannels = 3
        self.nbOutputChannels = 1
        self.kernelSz = 5
        self.stridedEncoder = True
        # strided vs resize
        self.stridedDecoder = True
        self.inDispRange = np.array([[0, 1, 2]])
        self.outDispRange = np.array([[0, 0, 0]])
        self.dispProcessOutputs = self.meanstd_norm
        self.alphaData = 1.0
        self.alphaDisc = 0.0
        self.linearImg = False
        self.rescaleImg = True

        # network arch function
        self.model = pix2pix_gen_p

        # loss function
        self.inMaski = tf.placeholder(tf.float32, shape=[
            self.batchSz, self.imgSzTr[0], self.imgSzTr[1], 1], name="input_mask")
        self.inMask = preprocess(self.inMaski, False, self.data_format)
        self.varLossAlpha = 0.00001
        self.regLoGLossAlpha = 0.1
        self.loss = self.loss_masked_meanstd_norm_charbonnier

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

    def loss_LoG(self, outputs, n):
        loss = 0
        outs = outputs
        for i in range(4):
            loss += tf.reduce_mean(tf.square(filterLoG_3x3(outs,
                                                           n, self.data_format)))
            outs = reduceSize2x(outs, n, self.data_format)
        return loss * 0.25

    def loss_masked_meanstd_norm_charbonnier(self, outputs, targets):

        outputs_sc = self.meanstd_norm(outputs)
        targets_sc = targets

        nvalid_b = tf.add(tf.reduce_sum(self.inMask, axis=self.wh_axis), EPS)
        diff = tf.multiply(tf.subtract(outputs_sc, targets_sc), self.inMask)

        data_loss = tf.reduce_mean(tf.divide(
            tf.reduce_sum(tf.sqrt(EPS + tf.square(diff)), axis=self.wh_axis), nvalid_b))

        reg_loss = self.loss_LoG(outputs_sc, self.nbOutputChannels)

        return data_loss + self.regLoGLossAlpha * reg_loss

    def loss_masked_logscale_charbonnier(self, outputs, targets):

        outputs_sc = tf.log(tf.add(tf.multiply(outputs, 3.0), 4.0))
        targets_sc = tf.log(tf.add(tf.multiply(targets, 3.0), 4.0))

        diff = tf.multiply(tf.subtract(outputs_sc, targets_sc), self.inMask)

        nvalid_b = tf.add(tf.reduce_sum(self.inMask, axis=self.wh_axis), EPS)

        log_scales = tf.divide(tf.reduce_sum(
            diff, axis=self.wh_axis, keepdims=True), nvalid_b)
        diff = tf.subtract(diff, log_scales)

        return tf.divide(tf.reduce_sum(tf.sqrt(EPS + tf.square(diff))), tf.reduce_sum(nvalid_b))

#-----------------------------------------------------------------------------------------------------
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


def saveModel(modelPath, asText, data_format):

    lp = DepthPredictionModelParams(modelPath, data_format)
    lp.isTraining = False

    mdSuff = '-last.pb.txt' if asText else '-last.pb'

    inputsi = tf.placeholder(tf.float32, name="adsk_inFront")
    inputs = tf.multiply(tf.subtract(inputsi, 0.5), 2.0)

    with tf.variable_scope("generator"):
        outputs = lp.model(inputs, lp)

    outputName = "adsk_outZ-Depth"
    outputs = tf.multiply(tf.add(outputs, 1.0), 0.5, name=outputName)

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
            outputName,  # 'outFront',
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
    [opts, loss, trSum, tsSum, valSum] = pix2pix_optimizer(inImg, inDepth, lp)

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

    #testDataset(args.imgRootDir, args.trainLstPath)

    #------------------------------------------------------------------------------------------------

    trainModel(args.modelPath, args.imgRootDir,
               args.trainLstPath, args.testLstPath, args.valLstPath, data_format)

    #------------------------------------------------------------------------------------------------

    # testModel(args.modelPath, args.imgRootDir, args.testLstPath, 100, data_format)

    #------------------------------------------------------------------------------------------------

    # evalModel(args.modelPath, args.imgRootDir, args.valLstPath, True, 512, False, data_format)

    #------------------------------------------------------------------------------------------------

    # saveModel(args.modelPath, False, data_format)
