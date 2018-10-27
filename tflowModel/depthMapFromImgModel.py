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
       "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libDepthImgSampler/libDepthImgSampler.so")

   def __init__(self, dbPath, imgRootDir, batchSz, imgSz, linearCS, seed):
      params = np.array([batchSz, imgSz[0], imgSz[1],
                         1.0 if linearCS else 0.0], dtype=np.float32)

      self.__nds = 2
      self.__currds = 0

      self.__ds = [BufferDataSampler(
          DatasetTF.__lib, dbPath, imgRootDir, params, seed+i) for i in range(self.__nds)]
      self.data = tf.data.Dataset.from_generator(
          self.sample, (tf.float32, tf.float32))

   def sample(self):
      for i in itertools.count(1):
         currImg, currDepth = self.__ds[self.__currds].getDataBuffers()
         self.__currds = (self.__currds+1) % self.__nds
         yield (currImg, currDepth)


def loadValidationData(dataPath, dataRootDir, dataSz, linearCS=False):

   im = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3))
   depth = np.full((dataSz[0], dataSz[1], dataSz[2], 1), 0.5)

   n = 0

   # input
   with open(dataPath, 'r') as img_names_file:

      for data in img_names_file:

         data = data.rstrip('\n').split()

         if n >= dataSz[0]:
            break

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

   batchSz = 16

   trDs = DatasetTF(trainPath, imgRootDir, batchSz, imgSz, False, rseed)

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
         cv.imshow('currDepth', cv.applyColorMap(
             (currDepth[idx] * 255.0).astype(np.uint8), cv.COLORMAP_JET))

         cv.waitKey(700)

#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class DepthPredictionModelParams(Pix2PixParams):

   def __init__(self, modelPath, data_format, seed=int(time.time())):

      #
      # model 0 : scale / resize / pix2pix_gen / no bn
      # model 1 : scale / strided / pix2pix_ires / no bn
      # model 2 : scale / resize / pix2pix_gen / bn
      # model 3 : scale / resize / pix2pix_ires / bn
      # model 4 : scale / resize / pix2pix_gen_p / bn
      # model 5 : noscale / resize / pix2pix_gen_p / bn
      # model 6 : scale / strided / pix2pix_gen_p / bn
      #
      # exp00007 : 32x240x320x32 / strided / pix2pix_gen_p, bn / charb + reg
      # exp00008 : 32x240x320x32 / strided-s / pix2pix_gen_p, bn / l2 + reg-l2

      # seed = 0

      Pix2PixParams.__init__(self, modelPath, data_format, seed)

      self.numMaxSteps = 250000
      self.numSteps = 250000
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
      self.stridedDecoder = True
      self.inDispRange = np.array([[0, 1, 2]])
      self.outDispRange = np.array([[0, 0, 0]])
      self.alphaData = 1.0
      self.alphaReg = 0.125
      self.alphaDisc = 0.0
      self.linearImg = False

      # network arch function
      self.minimizeMemory = False
      self.model = pix2pix_gen_p

      self.update()

   def loss(self, batchOutput, batchTargets):

      return charbonnier_loss(batchOutput, batchTargets), self.loss_reg(batchOutput, batchTargets)

   def loss_reg(self, batchOutput, batchTargets):

      lossGrad = tf.constant(0.0)

      axisH = 1 if self.data_format == 'NHWC' else 2

      batchOutputResized = batchOutput
      batchTargetResized = batchTargets

      for i in range(4):

         lossGradX = l2_loss(filterGradX_3x3(batchOutputResized, self.nbOutputChannels, self.data_format),
                             filterGradX_3x3(batchTargetResized, self.nbOutputChannels, self.data_format))
         lossGradY = l2_loss(filterGradY_3x3(batchOutputResized, self.nbOutputChannels, self.data_format),
                             filterGradY_3x3(batchTargetResized,  self.nbOutputChannels, self.data_format))

         lossGrad = tf.add(lossGrad, tf.add(lossGradX, lossGradY))

         batchOutputResized = reduceSize2x(
             batchOutputResized, self.nbOutputChannels, self.data_format)
         batchTargetResized = reduceSize2x(
             batchTargetResized, self.nbOutputChannels, self.data_format)

      return lossGrad

   def optimizer(self, batchInput, batchTargets):

      with tf.device('/gpu:*'):
         with tf.variable_scope(self.getModelName()) as modelVs:
            batchOutput = self.model(batchInput, self)

         with tf.variable_scope(self.getModelName() + "_loss"):
            loss_data, loss_reg = self.loss(batchOutput, batchTargets)
            loss = self.alphaData * loss_data + self.alphaReg * loss_reg

         # dependencies for the batch normalization
         depends = tf.get_collection(
             tf.GraphKeys.UPDATE_OPS) if self.useBatchNorm else []

         # optimizer
         opt, tvars, grads_and_vars = getOptimizerData(
             loss, depends, self, self.getModelName())

      # put summary on CPU to free some VRAM
      with tf.device('/cpu:*'):

         trSum = []
         addSummaryParams(trSum, self, tvars, grads_and_vars)
         trSum = tf.summary.merge(trSum, "Train")

         tsSum = []
         addSummaryScalar(tsSum, loss, "loss", "loss")
         addSummaryScalar(tsSum, loss_data, "loss", "data")
         addSummaryScalar(tsSum, loss_reg, "loss", "reg")

         addSummaryImages(tsSum, "Images", self,
                          [batchInput, batchTargets, batchOutput],
                          [[0, 1, 2], [0, 0, 0], [0, 0, 0]])
         tsSum = tf.summary.merge(tsSum, "Test")

         valSum = []
         addSummaryImages(valSum, "Images", self,
                          [batchInput, batchOutput],
                          [[0, 1, 2], [0, 0, 0]])
         valSum = tf.summary.merge(valSum, "Val")

      return [opt, loss, trSum, tsSum, valSum]

#-----------------------------------------------------------------------------------------------------
# VALIDATION
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgRootDir, imgLst, forceTrainingSize=True, maxSz=-1):

   lp = DepthPredictionModelParams(modelPath)
   lp.isTraining = False

   evalSz = [1, lp.imgSzTr[0], lp.imgSzTr[1], 3]

   inputsi = tf.placeholder(tf.float32, name="input")
   inputs = preprocess(inputsi)

   with tf.variable_scope("generator"):
      outputs = lp.model(inputs, lp)

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

         videoId = 0
         for data in img_names_file:

            imgName = imgRootDir + "/" + data.rstrip('\n')
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
            cv.normalize(depth[0], depth[0], 0, 1.0, cv.NORM_MINMAX)

            inputImg = (cv.cvtColor(
                img[0], cv.COLOR_RGB2BGR)*255.0).astype(np.uint8)
            coloredDepth = cv.applyColorMap(
                (depth[0] * 255.0).astype(np.uint8), cv.COLORMAP_JET)

            # show the sample
            cv.imshow('Input', inputImg)
            cv.imshow('Output', coloredDepth)

            outDName = imgRootDir + \
                '/evalOutput/{:06d}_d.exr'.format(videoId)
            cv.imwrite(outDName, depth[0])

            outCDName = imgRootDir + \
                '/evalOutput/{:06d}_cd.png'.format(videoId)
            cv.imwrite(outCDName, coloredDepth)

            outRGBName = imgRootDir + \
                '/evalOutput/{:06d}_rgb.png'.format(videoId)
            cv.imwrite(outRGBName, inputImg)

            cv.waitKey(500)

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
      outputs = lp.model(inputs, lp)
      if lp.doClassOut:
         outputs = tf.nn.tanh(outputs)
      if convert_df:
         outputs = postprocess(outputs, False, data_format)

   outputNames = outputs.name
   outputNames = outputNames[:outputNames.find(":")]

   inputNames = inputsi.name
   inputNames = inputNames[:inputNames.find(":")]

   print "-------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
   print "Exporting graph input : " + inputNames
   print "Exporting graph output : " + outputNames + "  (" + outputs.name + ")"
   print "------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

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

   lp = DepthPredictionModelParams(modelPath, data_format)

   # Datasets / Iterators
   trDs = DatasetTF(trainPath, imgRootDir, lp.batchSz,
                    lp.imgSzTr, lp.linearImg, lp.rseed)
   tsDs = DatasetTF(testPath, imgRootDir, lp.batchSz,
                    lp.imgSzTr, lp.linearImg, lp.rseed)

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
   inDepth = preprocess(inDepthi, True, data_format)

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
   sess_config = tf.ConfigProto()  # device_count={'GPU': 2})
   # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
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
         currImg, currDepth = sess.run(dsView)

         trFeed = {lp.isTraining: True,
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
            summary = sess.run(tsSum, feed_dict={lp.isTraining: False,
                                                 inImgi: currImg,
                                                 inDepthi: currDepth})
            train_summary_writer.add_summary(summary, step)

         if step % lp.tslogStep == 0:

            sess.run(tsInit)
            currImg, currDepth = sess.run(dsView)
            tsLoss, summary = sess.run([loss, tsSum], feed_dict={lp.isTraining: False,
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

            summary = sess.run(valSum, feed_dict={lp.isTraining: False,
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
                args.valLstPath, False, 640, True, data_format)

   #------------------------------------------------------------------------------------------------
   if args.mode == 'save':
      saveModel(args.modelPath, False, data_format, not args.uff)
