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
import math
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
       "/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/libSemSegSampler/libSemSegSampler.so")

   def __init__(self, dbPath, imgRootDir, batchSz, imgSz, linearCS, rescale, mapping, seed):
      params = np.array([batchSz, imgSz[0], imgSz[1],
                         1.0 if linearCS else 0.0,
                         1.0 if rescale else 0.0, mapping],
                        dtype=np.float32)
      self.__ds = BufferDataSampler(
          DatasetTF.__lib, dbPath, imgRootDir, params, seed)
      self.data = tf.data.Dataset.from_generator(
          self.sample, (tf.float32, tf.float32))

   def sample(self):
      for i in itertools.count(1):
         currImg, currLabels, currMask = self.__ds.getDataBuffers()
         yield (currImg, currLabels)


def loadValidationData(dataPath, dataRootDir, dataSz, linearCS):

   im = np.zeros((dataSz[0], dataSz[1], dataSz[2], 3))
   labels = np.zeros((dataSz[0], dataSz[1], dataSz[2], 1))

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

   return im, labels

#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class SemSegModelParams(Pix2PixParams):

   def __init__(self, modelPath, data_format, seed=int(time.time())):

      #
      # model 0 : resize / pix2pix_gen_p / bn / mapping#-1
      # model 1 : resize / pix2pix_gen_p / bn / mapping#0
      # model 2 : stided / pix2pix_gen_p / bn / mapping#-1
      #
      # exp0003 : 296x296x24x32 / resize / pix2pix_gen_p / bn / mapping#1
      #
      # exp0005 : 320x240x16x32 / pix2pix_hglass, bn / mapping#0
      #
      # exp0006 : 320x240x32x32 / pix2pix_gen_p, bn / mapping#0
      #
      # exp0007 : 32x240x320x32 / pix2pix_gen_p, bn / mapping#2 / classout_loss_with_unlabeled
      # exp0007 : 32x240x320x32 / pix2pix_gen_p, bn / mapping#2 / classout_loss_reg_with_unlabeled

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
      self.nbOutputChannels = 3
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
      self.dsRescale = True
      # Mapping
      self.dsMapping = 2

      self.minimizeMemory = False
      self.model = pix2pix_gen_p

      # loss
      self.doClassOut = True
      self.loss = self.classout_loss_reg_with_unlabeled

      self.update()

   def labels_from_targets(self, targets):
      return tf.maximum(tf.subtract(targets, 1), 0)

   def classout_loss_reg_with_unlabeled(self, outputs, targets):

      target_unlabeled_mask = tf.to_float(tf.minimum(targets, 1))
      target_labels = tf.maximum(tf.subtract(targets, 1), 0)

      cel = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.squeeze(target_labels), logits=outputs)

      loss = tf.divide(tf.reduce_sum(tf.multiply(cel, tf.squeeze(target_unlabeled_mask))),
                       tf.reduce_sum(target_unlabeled_mask))

      lossGrad = tf.constant(0.0)

      axisD = 3 if self.data_format == 'NHWC' else 1

      batchOutputResized = outputs

      batchTargetResized = tf.to_float(targets)
      bts = []
      for j in range(self.nbOutputChannels):
         bts.append(tf.abs(tf.sign(tf.subtract(batchTargetResized, j+1))))
      batchTargetResized = tf.concat(bts, axis=axisD)

      batchMaskResized = target_unlabeled_mask

      n = 4
      for i in range(n):

         batchErrGradX = l2(filterGradX_3x3(batchOutputResized, self.nbOutputChannels, self.data_format),
                            filterGradX_3x3(batchTargetResized, self.nbOutputChannels, self.data_format))
         batchErrGradY = l2(filterGradY_3x3(batchOutputResized, self.nbOutputChannels, self.data_format),
                            filterGradY_3x3(batchTargetResized,  self.nbOutputChannels, self.data_format))

         valid = tf.reduce_sum(batchMaskResized)
         lossGrad += tf.divide(tf.reduce_sum(tf.multiply(batchErrGradX, batchMaskResized)), valid)
         lossGrad += tf.divide(tf.reduce_sum(tf.multiply(batchErrGradY, batchMaskResized)), valid)

         if i < n-1:
            batchOutputResized = reduceSize2x(
                batchOutputResized, self.nbOutputChannels, self.data_format)
            batchTargetResized = reduceSize2xNN(
                batchTargetResized, self.nbOutputChannels, self.data_format)
            batchMaskResized = reduceSize2xNN(
                batchMaskResized, 1, self.data_format)

      return loss, lossGrad

   def classout_loss(self, outputs, targets):

      return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=tf.squeeze(targets), logits=outputs))

   def classout_loss_binary(self, outputs, targets):

      outputs_pos = tf.sigmoid(outputs)
      targets_f = tf.to_float(targets)

      return tf.reduce_mean(-1.0*targets_f*tf.log(outputs_pos) + (targets_f-1.0)*tf.log(1.0-outputs_pos))

   def optimizer(self, batchInput, batchTargets):

      with tf.device('/gpu:*'):
         with tf.variable_scope(self.getModelName()) as modelVs:
            batchOutput = self.model(batchInput, self)

         with tf.variable_scope(self.getModelName() + "_loss"):
            loss_data, loss_reg = self.loss(batchOutput, batchTargets)
            loss = self.alphaData * loss_data + self.alphaReg * loss_reg

      with tf.device('/gpu:*'):

         # dependencies for the batch normalization
         depends = tf.get_collection(
             tf.GraphKeys.UPDATE_OPS) if self.useBatchNorm else []

         # optimizer
         opt, tvars, grads_and_vars = getOptimizerData(
             loss, depends, self, self.getModelName())

      # put summary on CPU to free some VRAM
      with tf.device('/cpu:*'):

         batchTargetSple = tf.multiply(
             tf.subtract(tf.to_float(self.labels_from_targets(batchTargets))/self.nbOutputChannels, 0.5), 2.0)
         if self.nbOutputChannels > 1:
            argmax_axis = 3 if self.data_format == 'NHWC' else 1
            batchOutputSple = tf.multiply(
                tf.subtract(tf.to_float(tf.argmax(batchOutput,
                                                  axis=argmax_axis))/self.nbOutputChannels, 0.5), 2.0)
            batchOutputSple = tf.expand_dims(
                batchOutputSple, axis=argmax_axis)
         else:
            batchOutputSple = batchOutput

         trSum = []
         addSummaryParams(trSum, self, tvars, grads_and_vars)
         trSum = tf.summary.merge(trSum, "Train")

         tsSum = []
         addSummaryScalar(tsSum, loss, "loss", "loss")
         addSummaryScalar(tsSum, loss, "loss", "loss_data")
         addSummaryScalar(tsSum, loss, "loss", "loss_reg")

         addSummaryImages(tsSum, "Images", self,
                          [batchInput, batchTargetSple, batchOutputSple],
                          [[0, 1, 2], [0, 0, 0], [0, 0, 0]])
         tsSum = tf.summary.merge(tsSum, "Test")

         valSum = []
         addSummaryImages(valSum, "Images", self,
                          [batchInput, batchOutputSple],
                          [[0, 1, 2], [0, 0, 0]])
         valSum = tf.summary.merge(valSum, "Val")

      return [opt, loss, trSum, tsSum, valSum]

#-----------------------------------------------------------------------------------------------------
# UNIT TESTS
#-----------------------------------------------------------------------------------------------------


def testDataset(imgRootDir, trainPath):

   lp = SemSegModelParams("", "NHWC")

   trDs = DatasetTF(trainPath, imgRootDir, lp.batchSz,
                    lp.imgSzTr, lp.linearImg, lp.dsRescale, lp.dsMapping, lp.rseed)

   dsIt = tf.data.Iterator.from_structure(
       trDs.data.output_types, trDs.data.output_shapes)
   dsView = dsIt.get_next()

   trInit = dsIt.make_initializer(trDs.data)

   with tf.Session() as sess:

      sess.run(trInit)

      for step in range(100):

         currImg, currLabels = sess.run(dsView)

         idx = random.randint(0, lp.batchSz-1)

         cv.imshow('currImg', cv.cvtColor(currImg[idx], cv.COLOR_RGB2BGR))
         cv.imshow('currLabelsGL', currLabels[idx])
         cv.imshow('currLabels', cv.applyColorMap(
             ((255.0 / (lp.nbOutputChannels+1)) * currLabels[idx]).astype(np.uint8), cv.COLORMAP_JET))

         cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# VALIDATION
#-----------------------------------------------------------------------------------------------------


def evalModel(modelPath, imgRootDir, imgLst):

   lp = SemSegModelParams(modelPath, data_format)
   lp.isTraining = False

   evalSz = [1, 620, 480, 3]

   inputsi = tf.placeholder(tf.float32, shape=evalSz, name="input")
   inputs = preprocess(inputsi, True, data_format)

   with tf.variable_scope("generator"):
      outputs = pix2pix_gen(inputs, lp)
      outputs = postprocess(outputs, False, data_format)

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

         for data in img_names_file:

            img = [loadResizeImgPIL(imgRootDir + "/" +
                                    data.rstrip('\n'), [evalSz[1], evalSz[2]], lp.linearImg)]

            labels = sess.run(outputs, feed_dict={inputsi: img})

            # show the sample
            cv.imshow('Input', cv.cvtColor(img[0], cv.COLOR_RGB2BGR))
            cv.imshow('Output', cv.applyColorMap(
                labels.astype(np.uint8), cv.COLORMAP_JETdepth[0]))
            cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# EXPORT
#-----------------------------------------------------------------------------------------------------


def saveModel(modelPath, asText, data_format, convert_df):

   lp = SemSegModelParams(modelPath, data_format)
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
   print "Exporting graph : " + inputNames + " -> " + outputNames + "  ( " + outputs.name + " )"
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

   lp = SemSegModelParams(modelPath, data_format)

   # Datasets / Iterators
   trDs = DatasetTF(trainPath, imgRootDir, lp.batchSz,
                    lp.imgSzTr, lp.linearImg, lp.dsRescale, lp.dsMapping, lp.rseed)
   tsDs = DatasetTF(testPath, imgRootDir, lp.batchSz,
                    lp.imgSzTr, lp.linearImg, lp.dsRescale, lp.dsMapping, lp.rseed)

   dsIt = tf.data.Iterator.from_structure(
       trDs.data.output_types, trDs.data.output_shapes)
   dsView = dsIt.get_next()

   trInit = dsIt.make_initializer(trDs.data)
   tsInit = dsIt.make_initializer(tsDs.data)

   # Input placeholders
   inImgi = tf.placeholder(tf.float32, shape=[
       lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_img")
   inImg = preprocess(inImgi, True, data_format)
   inLabelsi = tf.placeholder(tf.float32, shape=[
       lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 1], name="input_depth")
   inLabels = tf.to_int32(preprocess(inLabelsi, False, data_format))

   # Optimizers
   [opts, loss, trSum, tsSum, valSum] = lp.optimizer(inImg, inLabels)

   # Validation
   valImg, valLabels = loadValidationData(
       valPath, imgRootDir, [lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1]], lp.linearImg)

   # Persistency
   persistency = tf.train.Saver(
       pad_step_number=True, max_to_keep=3, filename=lp.modelFilename)

   # Logger
   merged_summary_op = tf.summary.merge_all()

   # Params Initializer
   varInit = tf.global_variables_initializer()

   # Session configuration
   sess_config = tf.ConfigProto()
   # sess_config = tf.ConfigProto(device_count={'GPU': 1})
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
         currImg, currLabels = sess.run(dsView)

         trFeed = {lp.isTraining: True,
                   inImgi: currImg,
                   inLabelsi: currLabels}

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
                                                 inLabelsi: currLabels})
            train_summary_writer.add_summary(summary, step)

         if step % lp.tslogStep == 0:

            sess.run(tsInit)
            currImg, currLabels = sess.run(dsView)
            tsLoss, summary = sess.run([loss, tsSum], feed_dict={lp.isTraining: False,
                                                                 inImgi: currImg,
                                                                 inLabelsi: currLabels})

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
                                                  inLabelsi: valLabels})

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
