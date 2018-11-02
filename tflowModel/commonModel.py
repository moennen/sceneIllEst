""" common for tflowModel
"""

import sys
import os
import time
import json
import tensorflow as tf
# from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import memory_saving_gradients

#-----------------------------------------------------------------------------------------------------
# Set common custom includes
#
sys.path.append('/mnt/p4/favila/moennen/local/lib/python2.7/site-packages')
sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/'))

#-----------------------------------------------------------------------------------------------------
# Set common global
#
EPS = 1e-12

#-----------------------------------------------------------------------------------------------------
# Set tensorflow logging
#
tf.logging.set_verbosity(tf.logging.FATAL)

#-----------------------------------------------------------------------------------------------------
# Tensorflow Utils
#


def printSessionConfigProto(sess_config):

   serialized = sess_config.SerializeToString()
   result = ["0x"+c.encode('hex') for c in serialized]
   print(result)


def printVarTF(sess):

   tvars = tf.trainable_variables()
   for var in tvars:
      print var.name
      print var.eval(sess)

#-----------------------------------------------------------------------------------------------------
# Images Utils
#


# def showImgs(batch, img_depths):

#     n_imgs = len(img_depths)

#     if np.sum(img_depths) != batch.shape[3]:
#         raise ValueError()

#     batch_im = np.zeros(
#         (batch.shape[0] * batch.shape[1], n_imgs * batch.shape[2], 3))

#     for b in range(batch.shape[0]):

#         n_offset = 0
#         for n in range(n_imgs):

#             im_d = img_depths[n]
#             im = batch[b, :, :, n_offset:n_offset + im_d]

#             if im_d > 3:
#                 gray = np.mean(im, axis=2)
#                 im = np.stack([gray, gray, gray], 2)

#             batch_im[b * batch.shape[1]:(b + 1) * batch.shape[1],
#                      n * batch.shape[2]:(n + 1) * batch.shape[2], 0:im_d] = im

#             n_offset += im_d

#     plt.imshow(batch_im)
#     plt.show()


def writeExrRGB(img, output_filename):
   print img.shape
   rpix = img[:, :, 0].astype(np.float16).tostring()
   gpix = img[:, :, 1].astype(np.float16).tostring()
   bpix = img[:, :, 2].astype(np.float16).tostring()
   HEADER = OpenEXR.Header(img.shape[1], img.shape[0])
   half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
   HEADER['channels'] = dict([(c, half_chan) for c in "RGB"])
   exr = OpenEXR.OutputFile(output_filename, HEADER)
   exr.writePixels({'R': rpix, 'G': gpix, 'B': bpix})
   exr.close()


def loadImgPIL(img_name, linearCS):

   im = Image.open(img_name)
   im = np.array(im)
   im = im.astype(np.float32) / 255.0

   if linearCS == 1:
      im = (im <= 0.04045) * (im / 12.92) + (im > 0.04045) * \
          np.power((im + 0.055)/1.055, 2.4)

   return im


def loadResizeImgPIL(img_name, imgSz, linearCS):

   im = Image.open(img_name)
   ratio = float(imgSz[1])/imgSz[0]
   imgRatio = float(im.size[0])/im.size[1]
   cw = (int(im.size[1]*ratio) if imgRatio > ratio else im.size[0])
   ow = (int((im.size[0]-cw)/2) if imgRatio > ratio else 0)
   ch = (int(im.size[0]/ratio) if imgRatio < ratio else im.size[1])
   oh = (int((im.size[1]-ch)/2) if imgRatio < ratio else 0)
   im = im.crop([ow, oh, ow+cw, oh+ch])
   im = im.resize([imgSz[1], imgSz[0]])
   im = np.array(im)
   im = im.astype(np.float32) / 255.0

   if linearCS == 1:
      im = (im <= 0.04045) * (im / 12.92) + (im > 0.04045) * \
          np.power((im + 0.055)/1.055, 2.4)

   return im

#-----------------------------------------------------------------------------------------------------
# PROFILER
#-----------------------------------------------------------------------------------------------------


class Profiler:

   def __init__(self, tl_path):
      self._tl_path = tl_path
      self._timeline_dict = {}

   def update(self, chrome_trace, event_id):
      # convert crome trace to python dict
      chrome_trace_dict = json.loads(chrome_trace)
      # for first run store full trace
      if event_id not in self._timeline_dict:
         self._timeline_dict[event_id] = chrome_trace_dict
      # for other - update only time consumption, not definitions
      else:
         for event in chrome_trace_dict['traceEvents']:
            # events time consumption started with 'ts' prefix
            if 'ts' in event:
               self._timeline_dict[event_id]['traceEvents'].append(event)

   def save(self):
      try:
         i = 0
         for event_id in self._timeline_dict:
            with open(self._tl_path + '_02_step_%d.json' % i, 'w') as f:
               # with open(self._tl_path + '_step_{:02d}.json'.format(i), 'w') as f:
               json.dump(self._timeline_dict[event_id], f)
            i += 1
      except:
         print "Cannot save profiler data :", sys.exc_info()[0]


#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class LearningParams:

   def __init__(self, modelPath, data_format='NHWC', seed=int(time.time())):

      self.numMaxSteps = 175000
      self.numSteps = self.numMaxSteps
      self.trlogStep = 150
      self.tslogStep = 150
      self.vallogStep = 150
      self.backupStep = 1500

      self.batchSz = 64
      self.imgSzTr = [64, 64]
      self.imgSzTs = [256, 256]

      self.linearImg = False

      self.globalStep = tf.Variable(0, trainable=False, name='globalStep')
      self.globalStepInc = tf.assign_add(self.globalStep, 1)

      self.baseLearningRate = 0.0005
      self.learningRate = tf.train.polynomial_decay(self.baseLearningRate, self.globalStep, self.numSteps, 0.0,
                                                    power=0.7)

      self.isTraining = tf.placeholder(tf.bool)
      self.data_format = data_format

      self.tbLogsPath = modelPath + "/tbLogs"
      self.modelFilename = modelPath + "/tfData"
      self.modelNbToKeep = 7

      self.minimizeMemory = False

      self.doProfile = False
      self.profiler = None

      self.doExtSummary = True

      self.dispProcessOutputs = None

      self.rseed = seed

      tf.set_random_seed(self.rseed)

   def update(self):

      if self.doProfile:
         self.profiler = Profiler(self.modelFilename + "-tl")

      self.numMaxSteps = max(self.numMaxSteps, self.numSteps)
      self.learningRate = tf.train.polynomial_decay(self.baseLearningRate, self.globalStep, self.numMaxSteps, 0.0,
                                                    power=0.7)
      tf.set_random_seed(self.rseed)


#-----------------------------------------------------------------------------------------------------
# PREPARE INPUT
#-----------------------------------------------------------------------------------------------------

def preprocess(ins, center, data_format):

   outs = tf.transpose(
       ins, [0, 3, 1, 2]) if data_format == 'NCHW' else ins
   if center:
      outs = tf.multiply(tf.subtract(outs, 0.5), 2.0)
   return outs


def postprocess(ins, center, data_format):

   outs = tf.transpose(
       ins, [0, 2, 3, 1]) if data_format == 'NCHW' else ins
   if center:
      outs = tf.multiply(tf.add(outs, 1.0), 0.5)
   return outs


#-----------------------------------------------------------------------------------------------------
# IMG2IMG MODELS
#-----------------------------------------------------------------------------------------------------


class Pix2PixParams(LearningParams):

   def __init__(self, modelPath, data_format='NHWC', seed=int(time.time())):

      LearningParams.__init__(self, modelPath, data_format, seed)

      self.useBatchNorm = True

      self.nbChannels = 64
      self.nbInChannels = 3
      self.nbOutputChannels = 3
      self.kernelSz = 4
      self.stridedEncoder = True
      self.stridedDecoder = True

      self.inDispRange = []
      self.outDispRange = []

      self.alphaData = 1.0
      self.alphaDisc = 0.0

      self.prepare_disc = prepareDisc

      self.disc_model_targets = pix2pix_disc
      self.disc_model_outputs = self.disc_model_targets
      self.disc_loss_targets = pix2pix_disc_loss_targets
      self.disc_loss_outputs = pix2pix_disc_loss_outputs

      self.doClassOut = False

   def getModelName(self):
      return "generator"

   def doDisc(self):
      return self.alphaDisc > 0.0

   def getDiscModelName(self):
      return "discriminator"


#
# n = number of output channels
# ks = kernel size
# ss = stride size
# bn = do batch norm
#


def increaseSize2x(inputs, i_n, data_format):

   kernels = np.zeros((2, 2, i_n, i_n), dtype=np.float32)
   v = 1.0
   for i in range(i_n):
      kernels[0, 0, i, i] = v
      kernels[0, 1, i, i] = v
      kernels[1, 0, i, i] = v
      kernels[1, 1, i, i] = v

   w = tf.constant(kernels)
   strides = [1, 2, 2, 1] if data_format == 'NHWC' else [1, 1, 2, 2]
   ref_shape = tf.shape(inputs)
   if data_format == 'NHWC':
      ref_shape = [ref_shape[0], 2*ref_shape[1], 2*ref_shape[2], i_n]
   else:
      ref_shape = [ref_shape[0], i_n, 2*ref_shape[2], 2*ref_shape[3]]

   x = tf.nn.conv2d_transpose(
       inputs, w, ref_shape, strides=strides, padding='SAME', data_format=data_format)

   return x


def increaseSize2xWithRef(inputs, ref, i_n, data_format):

   kernels = np.zeros((2, 2, i_n, i_n), dtype=np.float32)
   v = 1.0
   for i in range(i_n):
      kernels[0, 0, i, i] = v
      kernels[0, 1, i, i] = v
      kernels[1, 0, i, i] = v
      kernels[1, 1, i, i] = v

   w = tf.constant(kernels)
   strides = [1, 2, 2, 1] if data_format == 'NHWC' else [1, 1, 2, 2]
   ref_shape = tf.shape(ref)
   if data_format == 'NHWC':
      ref_shape = [ref_shape[0], ref_shape[1], ref_shape[2], i_n]
   else:
      ref_shape = [ref_shape[0], i_n, ref_shape[2], ref_shape[3]]

   x = tf.nn.conv2d_transpose(
       inputs, w, ref_shape, strides=strides, padding='SAME', data_format=data_format)

   return x


def filter_kernel(inputs, i_n, data_format, kernel, ks, strides=[1, 1, 1, 1]):

   w = tf.expand_dims(tf.concat([kernel for i in range(i_n)], axis=2), axis=3)
   pad_size = ks//2
   if data_format == 'NHWC':
      pad_mat = np.array([[0, 0], [pad_size, pad_size],
                          [pad_size, pad_size], [0, 0]])
   else:
      pad_mat = np.array(
          [[0, 0], [0, 0], [pad_size, pad_size], [pad_size, pad_size]])
   x = tf.pad(inputs, pad_mat)
   x = tf.nn.depthwise_conv2d(
       x, w, strides=strides, padding='VALID', data_format=data_format)

   return x


def reduceSize2x(inputs, i_n, data_format):

   kernel = [[[0.125], [0.25], [0.125]], [
       [0.25], [0.5], [0.25]], [[0.125], [0.25], [0.125]]]
   kernel = tf.divide(tf.constant(kernel), 2.0)
   strides = [1, 2, 2, 1] if data_format == 'NHWC' else [1, 1, 2, 2]
   return filter_kernel(inputs, i_n, data_format, kernel, 3, strides)


def reduceSize2xNN(inputs, i_n, data_format):

   kernel = [[[1.0]]]
   strides = [1, 2, 2, 1] if data_format == 'NHWC' else [1, 1, 2, 2]
   return filter_kernel(inputs, i_n, data_format, kernel, 1, strides)


def filterLoG_5x5(inputs, i_n, data_format):

   kernel = [[[0.0], [0.0], [0.3125], [0.0], [0.0]], [[0.0], [0.03125], [0.0625], [0.03125], [0.0]], [[0.03125], [0.0625], [-0.5], [0.0625], [0.03125]],
             [[0.0], [0.03125], [0.0625], [0.03125], [0.0]], [[0.0], [0.0], [0.3125], [0.0], [0.0]]]
   kernel = tf.constant(kernel)
   return filter_kernel(inputs, i_n, data_format, kernel, 5)


def filterLoG_3x3(inputs, i_n, data_format):

   kernel = [[[0.125], [0.125], [0.125]], [
       [0.125], [-1.0], [0.125]], [[0.125], [0.125], [0.125]]]
   kernel = tf.divide(tf.constant(kernel), 2.0)

   return filter_kernel(inputs, i_n, data_format, kernel, 3)


def filterGradX_3x3(inputs, i_n, data_format):

   kernel = [[[0.125], [0.0], [-0.125]], [
       [0.25], [0.0], [-0.25]], [[0.125], [0.0], [-0.125]]]

   return filter_kernel(inputs, i_n, data_format, tf.constant(kernel), 3)


def filterGradY_3x3(inputs, i_n, data_format):

   kernel = [[[0.125], [0.25], [0.125]], [
       [0.0], [0.0], [0.0]], [[-0.125], [-0.25], [-0.125]]]

   return filter_kernel(inputs, i_n, data_format, tf.constant(kernel), 3)


def pix2pix_conv(inputs, i_n, o_n, ks, ss, bias, data_format):

   initializer = tf.contrib.layers.xavier_initializer()

   shape = [ks, ks, i_n, o_n]
   w = tf.get_variable(
       "w", shape, initializer=initializer, trainable=True)
   pad_size = ks//2
   if data_format == 'NHWC':
      pad_mat = np.array([[0, 0], [pad_size, pad_size],
                          [pad_size, pad_size], [0, 0]])
      strides = [1, ss, ss, 1]
   else:
      pad_mat = np.array(
          [[0, 0], [0, 0], [pad_size, pad_size], [pad_size, pad_size]])
      strides = [1, 1, ss, ss]
   x = tf.pad(inputs, pad_mat)
   x = tf.nn.conv2d(x, w, strides=strides,
                    data_format=data_format, padding='VALID')
   if bias:
      x = tf.nn.bias_add(x, tf.get_variable(
          "b", [o_n], initializer=initializer, trainable=True), data_format)

   return x


def pix2pix_deconv(inputs, ref, i_n, o_n, ks, ss, bias, data_format):

   initializer = tf.contrib.layers.xavier_initializer()

   shape = [ks, ks, o_n, i_n]
   strides = [1, ss, ss, 1] if data_format == 'NHWC' else [1, 1, ss, ss]
   w = tf.get_variable(
       "deconv_w", shape, initializer=initializer, trainable=True)
   ref_shape = tf.shape(ref)
   if data_format == 'NHWC':
      ref_shape = [ref_shape[0], ref_shape[1], ref_shape[2], o_n]
   else:
      ref_shape = [ref_shape[0], o_n, ref_shape[2], ref_shape[3]]
   x = tf.nn.conv2d_transpose(
       inputs, w, ref_shape, strides=strides, padding='SAME', data_format=data_format)

   if bias:
      x = tf.nn.bias_add(x, tf.get_variable(
          "deconv_b", [o_n], initializer=initializer, trainable=True), data_format)

   return x


def pix2pix_cdeconv(inputs, ref, i_n, o_n, ks, ss, bias, data_format):

   if ss == 2:
      x = tf.nn.relu(pix2pix_deconv(inputs, ref, i_n, o_n, 2, ss, True, data_format))
      return pix2pix_conv(x, o_n, o_n, ks, 1, bias, data_format)
      #    x = increaseSize2xWithRef(inputs, ref, i_n, data_format)
      #    return pix2pix_conv(x, i_n, o_n, ks, 1, bias, data_format)

   else:
      # force the kernel size to be a multiplier of the stride
      return pix2pix_deconv(inputs, ref, i_n, o_n, ks, ss, bias, data_format)


def pix2pix_conv_bn(inputs, i_n, o_n, ks, ss, strided, bn, train, data_format):

   if strided:
      layer = pix2pix_conv(inputs, i_n, o_n, ks, ss, not bn, data_format)

   else:
      layer = pix2pix_conv(
          inputs, i_n, o_n, ks, 1, not bn, data_format)
      if data_format == 'NCHW':
         layer = tf.transpose(layer, [0, 2, 3, 1])
      layer_shape = tf.shape(layer)
      layer = tf.image.resize_area(
          layer, [layer_shape[1] / ss, layer_shape[2] / ss])
      if data_format == 'NCHW':
         layer = tf.transpose(layer, [0, 3, 1, 2])

   if bn:
      layer = tf.layers.batch_normalization(layer,
                                            axis=3 if data_format == 'NHWC' else 1,
                                            momentum=0.1,
                                            epsilon=1e-5,
                                            center=True,
                                            scale=False,
                                            fused=True,
                                            training=train)
   return layer


def pix2pix_deconv_bn(inputs, ref, i_n, o_n, ks, ss, strided, bn, train, data_format):

   # image resize does not exist in NCHW
   if strided:
      layer = pix2pix_cdeconv(inputs, ref, i_n, o_n,
                              ks, ss, not bn, data_format)
      # layer = tf.layers.conv2d_transpose(inputs, o_n,
      #                                    kernel_size=ks,
      #                                    strides=(ss, ss),
      #                                    padding="same",
      #                                    data_format='channels_last' if data_format == 'NHWC' else 'channels_first',
      #                                    use_bias=not bn,
      #                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
      # # should pad the layer tensor to be able to concat it with the ref tensor
      # if ss > 1:
      #     layer = tf.slice(layer, [0, 0, 0, 0], tf.shape(ref))
      # #layer = pix2pix_conv(layer, i_n, o_n, ks, 1, not bn, data_format)

   else:
      layer_shape = tf.shape(ref)
      if data_format == 'NCHW':
         inputs = tf.transpose(inputs, [0, 2, 3, 1])
         layer_wh = [layer_shape[2], layer_shape[3]]
      else:
         layer_wh = [layer_shape[1], layer_shape[2]]
      layer = tf.image.resize_bilinear(inputs, layer_wh)
      if data_format == 'NCHW':
         layer = tf.transpose(layer, [0, 3, 1, 2])

      layer = pix2pix_conv(layer, i_n, o_n, ks, 1, not bn, data_format)

   if bn:
      layer = tf.layers.batch_normalization(layer,
                                            axis=3 if data_format == 'NHWC' else 1,
                                            momentum=0.1,
                                            epsilon=1e-5,
                                            center=True,
                                            scale=False,
                                            fused=True,
                                            training=train)

   return layer


def pix2pix_encoder_bn(inputs, i_n, o_n, ks, ss, strided, bn, name, train, data_format):

   with tf.variable_scope(name):

      layer = pix2pix_conv_bn(inputs, i_n, o_n, ks,
                              ss, strided, bn, train, data_format)
      layer = tf.nn.relu(layer)

      return layer


def pix2pix_decoder_skip(inputs, skip, i_n, o_n, ks, ss, strided, bn, name, train, data_format):

   with tf.variable_scope(name):

      layer = pix2pix_deconv_bn(
          inputs, skip, i_n, o_n, ks, ss, strided, bn, train, data_format)
      layer = tf.nn.relu(layer)

      layer = tf.concat(
          [skip, layer], axis=3 if data_format == 'NHWC' else 1)

      return layer


def pix2pix_hglass_inception(input_layer, nIn, nOut, nInt, ks, bn, name, train, data_format):

   layer = pix2pix_encoder_bn(
       input_layer, nIn, nOut, 1, 1, True, bn, name+"_00", train, data_format)

   layers = [layer]

   for i in range(len(ks)):
      layer = pix2pix_encoder_bn(
          input_layer, nIn, nInt, 1, 1, True, bn, name+"_0"+str(i+1), train, data_format)
      layer = pix2pix_encoder_bn(
          layer, nInt, nOut, ks[i], 1, True, bn, name+"_1"+str(i+1), train, data_format)
      layers.append(layer)

   return tf.concat(layers, axis=3 if data_format == 'NHWC' else 1, name=name)


def pix2px_hglass_upsample(input_layer, ref_layer, nIn, data_format):

   layer_shape = tf.shape(ref_layer)
   inputs = input_layer
   if data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 2, 3, 1])
      layer_wh = [layer_shape[2], layer_shape[3]]
   else:
      layer_wh = [layer_shape[1], layer_shape[2]]
   layer = tf.image.resize_bilinear(inputs, layer_wh)
   if data_format == 'NCHW':
      layer = tf.transpose(layer, [0, 3, 1, 2])

   return tf.add(layer, ref_layer)


def pix2pix_hglass(imgs, params):

   nIn = params.nbInChannels
   nOut = params.nbOutputChannels
   data_format = params.data_format
   bn = params.useBatchNorm
   train = params.isTraining
   data_format_p = "channels_last" if data_format == "NHWC" else "channels_first"

   encoder_00 = pix2pix_encoder_bn(
       imgs, nIn, 128, 3, 1, True, False, "encoder_00", train, data_format)

   encoder_10 = tf.layers.max_pooling2d(
       encoder_00, [2, 2], [2, 2], padding="SAME", data_format=data_format_p)
   encoder_10 = pix2pix_hglass_inception(
       encoder_10, 128, 32, 32, [3, 5, 7], bn, "encoder_100", train, data_format)
   encoder_10 = pix2pix_hglass_inception(
       encoder_10, 128, 32, 32, [3, 5, 7], bn, "encoder_101", train, data_format)

   encoder_11 = pix2pix_hglass_inception(
       encoder_00, 128, 16, 64, [3, 7, 11], bn, "encoder_110", train, data_format)

   encoder_20 = tf.layers.max_pooling2d(
       encoder_10, [2, 2], [2, 2], padding="SAME", data_format=data_format_p)
   encoder_20 = pix2pix_hglass_inception(
       encoder_20, 128, 32, 32, [3, 5, 7], bn, "encoder_200", train, data_format)
   encoder_20 = pix2pix_hglass_inception(
       encoder_20, 128, 64, 32, [3, 5, 7], bn, "encoder_201", train, data_format)

   encoder_21 = pix2pix_hglass_inception(
       encoder_10, 128, 32, 32, [3, 5, 7], bn, "encoder_210", train, data_format)
   encoder_21 = pix2pix_hglass_inception(
       encoder_21, 128, 32, 64, [3, 7, 11], bn, "encoder_211", train, data_format)

   encoder_30 = tf.layers.max_pooling2d(
       encoder_20, [2, 2], [2, 2], padding="SAME", data_format=data_format_p)
   encoder_30 = pix2pix_hglass_inception(
       encoder_30, 256, 64, 32, [3, 5, 7], bn, "encoder_300", train, data_format)
   encoder_30 = pix2pix_hglass_inception(
       encoder_30, 256, 64, 32, [3, 5, 7], bn, "encoder_301", train, data_format)

   encoder_31 = pix2pix_hglass_inception(
       encoder_20, 256, 64, 32, [3, 5, 7], bn, "encoder_310", train, data_format)
   encoder_31 = pix2pix_hglass_inception(
       encoder_31, 256, 64, 64, [3, 7, 11], bn, "encoder_311", train, data_format)

   encoder_40 = tf.layers.max_pooling2d(
       encoder_30, [2, 2], [2, 2], padding="SAME", data_format=data_format_p)
   encoder_40 = pix2pix_hglass_inception(
       encoder_40, 256, 64, 32, [3, 5, 7], bn, "encoder_400", train, data_format)
   encoder_40 = pix2pix_hglass_inception(
       encoder_40, 256, 64, 32, [3, 5, 7], bn, "encoder_401", train, data_format)
   encoder_40 = pix2pix_hglass_inception(
       encoder_40, 256, 64, 32, [3, 5, 7], bn, "encoder_402", train, data_format)

   encoder_41 = pix2pix_hglass_inception(
       encoder_30, 256, 64, 32, [3, 5, 7], bn, "encoder_410", train, data_format)
   encoder_41 = pix2pix_hglass_inception(
       encoder_41, 256, 64, 32, [3, 5, 7], bn, "encoder_411", train, data_format)

   decoder_40 = pix2px_hglass_upsample(
       encoder_40, encoder_41, 256, data_format)
   decoder_40 = pix2pix_hglass_inception(
       decoder_40, 256, 64, 32, [3, 5, 7], bn, "decoder_400", train, data_format)
   decoder_40 = pix2pix_hglass_inception(
       decoder_40, 256, 64, 64, [3, 7, 11], bn, "decoder_401", train, data_format)

   decoder_30 = pix2px_hglass_upsample(
       decoder_40, encoder_31, 256, data_format)
   decoder_30 = pix2pix_hglass_inception(
       decoder_30, 256, 64, 32, [3, 5, 7], bn, "decoder_300", train, data_format)
   decoder_30 = pix2pix_hglass_inception(
       decoder_30, 256, 32, 32, [3, 5, 7], bn, "decoder_301", train, data_format)

   decoder_20 = pix2px_hglass_upsample(
       decoder_30, encoder_21, 128, data_format)
   decoder_20 = pix2pix_hglass_inception(
       decoder_20, 128, 32, 32, [3, 5, 7], bn, "decoder_200", train, data_format)
   decoder_20 = pix2pix_hglass_inception(
       decoder_20, 128, 16, 64, [3, 7, 11], bn, "decoder_201", train, data_format)

   decoder_10 = pix2px_hglass_upsample(
       decoder_20, encoder_11, 64, data_format)
   with tf.variable_scope("decoder_10"):
      decoder_10 = pix2pix_conv_bn(
          decoder_10, 64, nOut, 3, 1, True, False, train, data_format)

   return decoder_10


def pix2pix_hglass_d(imgs, params):

   nIn = params.nbInChannels
   nOut = params.nbOutputChannels
   data_format = params.data_format
   bn = params.useBatchNorm
   train = params.isTraining
   data_format_p = "channels_last" if data_format == "NHWC" else "channels_first"

   encoder_00 = pix2pix_encoder_bn(
       imgs, nIn, 128, 3, 1, True, False, "encoder_00", train, data_format)

   encoder_10 = tf.layers.average_pooling2d(
       encoder_00, [2, 2], [2, 2], padding="SAME", data_format=data_format_p)
   encoder_10 = pix2pix_hglass_inception(
       encoder_10, 128, 32, 32, [3, 5, 7], bn, "encoder_100", train, data_format)
   encoder_10 = pix2pix_hglass_inception(
       encoder_10, 128, 32, 32, [3, 5, 7], bn, "encoder_101", train, data_format)

   encoder_11 = pix2pix_hglass_inception(
       encoder_00, 128, 16, 64, [3, 7, 11], bn, "encoder_110", train, data_format)

   encoder_20 = tf.layers.average_pooling2d(
       encoder_10, [2, 2], [2, 2], padding="SAME", data_format=data_format_p)
   encoder_20 = pix2pix_hglass_inception(
       encoder_20, 128, 32, 32, [3, 5, 7], bn, "encoder_200", train, data_format)
   encoder_20 = pix2pix_hglass_inception(
       encoder_20, 128, 64, 32, [3, 5, 7], bn, "encoder_201", train, data_format)

   encoder_21 = pix2pix_hglass_inception(
       encoder_10, 128, 32, 32, [3, 5, 7], bn, "encoder_210", train, data_format)
   encoder_21 = pix2pix_hglass_inception(
       encoder_21, 128, 32, 64, [3, 7, 11], bn, "encoder_211", train, data_format)

   encoder_30 = tf.layers.average_pooling2d(
       encoder_20, [2, 2], [2, 2], padding="SAME", data_format=data_format_p)
   encoder_30 = pix2pix_hglass_inception(
       encoder_30, 256, 64, 32, [3, 5, 7], bn, "encoder_300", train, data_format)
   encoder_30 = pix2pix_hglass_inception(
       encoder_30, 256, 32, 32, [3, 5, 7], bn, "encoder_301", train, data_format)

   decoder_20 = pix2px_hglass_upsample(
       encoder_30, encoder_21, 128, data_format)
   decoder_20 = pix2pix_hglass_inception(
       decoder_20, 128, 32, 32, [3, 5, 7], bn, "decoder_200", train, data_format)
   decoder_20 = pix2pix_hglass_inception(
       decoder_20, 128, 16, 64, [3, 7, 11], bn, "decoder_201", train, data_format)

   decoder_10 = pix2px_hglass_upsample(
       decoder_20, encoder_11, 64, data_format)
   with tf.variable_scope("decoder_10"):
      decoder_10 = pix2pix_conv_bn(
          decoder_10, 64, nOut, 3, 1, True, False, train, data_format)

   return decoder_10


def pix2pix_gen_p(imgs, params):

   n = params.nbChannels
   nIn = params.nbInChannels
   nOut = params.nbOutputChannels
   data_format = params.data_format

   ks = params.kernelSz
   ess = params.stridedEncoder
   dss = params.stridedDecoder

   bn = params.useBatchNorm

   train = params.isTraining

   # encoder part

   # S x I --> S x N
   encoder_0 = pix2pix_encoder_bn(
       imgs, nIn, n, ks, 1, True, False, "encoder_0", train, data_format)
   # S x N --> S/2 x 2*N
   encoder_1 = pix2pix_encoder_bn(
       encoder_0, n, n*2, ks, 2, ess, bn, "encoder_1", train, data_format)
   # S/2 x 2*N --> S/4 x 4*N
   encoder_2 = pix2pix_encoder_bn(
       encoder_1, n*2, n*4, ks, 2, ess, bn, "encoder_2", train, data_format)
   # S/4 x 4*N --> S/8 x 4*N
   encoder_3 = pix2pix_encoder_bn(
       encoder_2, n*4, n*4, ks, 2, ess, bn, "encoder_3", train, data_format)
   # S/8 x 8*N --> S/16 x 8*N
   encoder_4 = pix2pix_encoder_bn(
       encoder_3, n*4, n*8, ks, 2, ess, bn, "encoder_4", train, data_format)
   # S/16 x 8*N --> S/32 x 8*N
   encoder_5 = pix2pix_encoder_bn(
       encoder_4, n*8, n*8, ks, 2, ess, bn, "encoder_5", train, data_format)
   # S/32 x 8*N --> S/64 x 4*N
   encoder_6 = pix2pix_encoder_bn(
       encoder_5, n*8, n*4, ks, 2, ess, bn, "encoder_6", train, data_format)
   # S/64 x 4*N --> S/64 x 4*N
   encoder_7 = pix2pix_encoder_bn(
       encoder_6, n*4, n*4, ks, 1, ess, bn, "encoder_7", train, data_format)

   # decoder with skip connections

   # S/64 x 4*N --> S/32 x 12*N
   decoder_6 = pix2pix_decoder_skip(
       encoder_7, encoder_5, n*4, n*4, ks, 2, dss, bn, "decoder_6", train, data_format)
   # S/32 x 12*N --> S/16 x 12*N
   decoder_5 = pix2pix_decoder_skip(
       decoder_6, encoder_4, n*12, n*4, ks, 2, dss, bn, "decoder_5", train, data_format)
   # S/16 x 12*N --> S/8 x 8*N
   decoder_4 = pix2pix_decoder_skip(
       decoder_5, encoder_3, n*12, n*4, ks, 2, dss, bn, "decoder_4", train, data_format)
   # S/8 x 8*N --> S/4 x 6*N
   decoder_3 = pix2pix_decoder_skip(
       decoder_4, encoder_2, n*8, n*2, ks, 2, dss, bn, "decoder_3", train, data_format)
   # S/4 x 6*N --> S/2 x 4*N
   decoder_2 = pix2pix_decoder_skip(
       decoder_3, encoder_1, n*6, n*2, ks, 2, dss, bn, "decoder_2", train, data_format)
   # S/2 x 4*N --> S x 2*N
   decoder_1 = pix2pix_decoder_skip(
       decoder_2, encoder_0, n*4, n, ks, 2, dss, bn, "decoder_1", train, data_format)
   # S x N --> S x I
   with tf.variable_scope("decoder_0"):
      output = pix2pix_conv_bn(decoder_1, n*2, nOut,
                               ks, 1, True, False, train, data_format)
      if not params.doClassOut:
         output = tf.nn.tanh(output)

   return output


def pix2pix_disc(gen_inputs, gen_outputs, params):

   inputs = tf.concat([gen_inputs, gen_outputs],
                      axis=3 if data_format == 'NHWC' else 1)

   nIn = params.nbInChannels + params.nbOutputChannels

   return pix2pix_disc_s(inputs, nIn, params)


def pix2pix_disc_s(inputs, nIn, params):

   train = params.isTraining

   n = params.nbChannels
   data_format = params.data_format

   ks = params.kernelSz
   ess = params.stridedEncoder
   bn = params.useBatchNorm

   # layer_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ndf]
   layer_1 = pix2pix_encoder_bn(
       inputs, nIn, n, ks, 1, True, False, "layer_1", train, data_format)
   # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
   layer_2 = pix2pix_encoder_bn(
       layer_1, n, n*2, ks, 2, ess, bn, "layer_2", train, data_format)
   # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
   layer_3 = pix2pix_encoder_bn(
       layer_2, n*2, n*4, ks, 1, True, bn, "layer_3", train, data_format)
   # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
   layer_4 = pix2pix_encoder_bn(
       layer_3, n*4, n*8, ks, 2, ess, bn, "layer_4", train, data_format)
   # layer_5: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
   layer_5 = pix2pix_encoder_bn(
       layer_4, n*8, n*8, ks, 1, True, bn, "layer_5", train, data_format)

   # layer_6
   with tf.variable_scope("layer_6"):
      layer_6 = pix2pix_conv(layer_5, n*8, 1, ks, ess, True, data_format)
      # layer_6 = tf.nn.sigmoid(layer_6)

   return layer_6


def pix2pix_disc_f(inputs, nIn, params, reuse):

   train = params.isTraining

   n = params.nbChannels
   data_format = params.data_format

   ks = params.kernelSz
   ess = params.stridedEncoder
   bn = params.useBatchNorm

   # S x I --> S x N
   encoder_0 = pix2pix_encoder_bn(
       inputs, nIn, n, ks, 1, True, False, "encoder_0", train, data_format)
   # S x N --> S/2 x 2*N
   encoder_1 = pix2pix_encoder_bn(
       encoder_0, n, n*2, ks, 2, ess, bn, "encoder_1", train, data_format)
   # S/2 x 2*N --> S/4 x 4*N
   encoder_2 = pix2pix_encoder_bn(
       encoder_1, n*2, n*4, ks, 2, ess, bn, "encoder_2", train, data_format)
   # S/4 x 4*N --> S/8 x 4*N
   encoder_3 = pix2pix_encoder_bn(
       encoder_2, n*4, n*4, ks, 2, ess, bn, "encoder_3", train, data_format)
   # S/8 x 8*N --> S/16 x 8*N
   encoder_4 = pix2pix_encoder_bn(
       encoder_3, n*4, n*8, ks, 2, ess, bn, "encoder_4", train, data_format)
   # S/16 x 8*N --> S/32 x 8*N
   encoder_5 = pix2pix_encoder_bn(
       encoder_4, n*8, n*8, ks, 2, ess, bn, "encoder_5", train, data_format)
   # S/32 x 8*N --> S/64 x 4*N
   encoder_6 = pix2pix_encoder_bn(
       encoder_5, n*8, n*4, ks, 2, ess, bn, "encoder_6", train, data_format)
   # S/64 x 4*N --> S/64 x 4*N
   encoder_7 = pix2pix_encoder_bn(
       encoder_6, n*4, n*4, ks, 1, ess, bn, "encoder_7", train, data_format)

   # last layer : dense with sigmoid
   layer = tf.layers.dense(tf.layers.flatten(encoder_7), 1,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           bias_initializer=tf.contrib.layers.xavier_initializer(),
                           reuse=reuse)
   layer = tf.sigmoid(layer)

   return layer


def l2(outputs, targets):
   return tf.square(tf.subtract(outputs, targets))


def l1(outputs, targets):
   return tf.abs(tf.subtract(outputs, targets))


def charbonnier(outputs, targets):
   return tf.sqrt(EPS + tf.square(tf.subtract(outputs, targets)))


def l2_loss(outputs, targets):
   return tf.reduce_mean(l2(outputs, targets))


def l1_loss(outputs, targets):
   return tf.reduce_mean(l1(outputs, targets))


def charbonnier_loss(outputs, targets):
   return tf.reduce_mean(charbonnier(outputs, targets))


def disc_loss(outputs, label):
   return tf.reduce_mean(
       tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(label,
                                                                  shape=outputs.shape), logits=outputs))


def pix2pix_logscale_l2_loss(outputs, targets):

   outputs_sc = tf.log(tf.add(tf.multiply(outputs, 3.0), 4.0))
   targets_sc = tf.log(tf.add(tf.multiply(targets, 3.0), 4.0))

   diff = tf.subtract(outputs_sc, targets_sc)

   return tf.reduce_mean(tf.square(diff)) - tf.square(tf.reduce_mean(diff))


def pix2pix_logscale_charbonnier_loss_nhwc(outputs, targets):

   outputs_sc = tf.log(tf.add(tf.multiply(outputs, 3.0), 4.0))
   targets_sc = tf.log(tf.add(tf.multiply(targets, 3.0), 4.0))

   diff = tf.subtract(outputs_sc, targets_sc)
   log_scales = tf.reduce_mean(diff, axis=[1, 2], keepdims=True)
   diff = tf.subtract(diff, log_scales)

   return tf.reduce_mean(tf.sqrt(EPS + tf.square(diff)))


def pix2pix_logscale_charbonnier_loss_nchw(outputs, targets):

   outputs_sc = tf.log(tf.add(tf.multiply(outputs, 3.0), 4.0))
   targets_sc = tf.log(tf.add(tf.multiply(targets, 3.0), 4.0))

   diff = tf.subtract(outputs_sc, targets_sc)
   log_scales = tf.reduce_mean(diff, axis=[2, 3], keepdims=True)
   diff = tf.subtract(diff, log_scales)

   return tf.reduce_mean(tf.sqrt(EPS + tf.square(diff)))


def pix2pix_classout_loss(outputs, targets):

   return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
       labels=tf.squeeze(targets), logits=outputs))


def pix2pix_disc_loss_outputs(outputs):
   return tf.reduce_mean(
       tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(0., shape=outputs.shape), logits=outputs))


def pix2pix_disc_loss_targets(targets):
   return tf.reduce_mean(
       tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(1., shape=targets.shape), logits=targets))


def prepareDisc(imgs, targets, outputs):
   return imgs, targets, imgs, outputs


def getOptimizerData(loss, depends, params, name):

   with tf.control_dependencies(depends):
      with tf.variable_scope(name + "_opt"):
         gen_tvars = [var for var in tf.trainable_variables(
         ) if var.name.startswith(name)]

         #gen_optim = tf.contrib.opt.AdamWOptimizer(params.learningRate)
         gen_optim = tf.train.AdamOptimizer(params.learningRate, 0.5, 0.999)

         if params.minimizeMemory:
            gen_grads = memory_saving_gradients.gradients(
                loss, gen_tvars, checkpoints='speed')
            gen_grads_and_vars = list(zip(gen_grads, gen_tvars))
         else:
            gen_grads_and_vars = gen_optim.compute_gradients(
                loss, var_list=gen_tvars)

         gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

   return gen_train, gen_tvars, gen_grads_and_vars


def pix2pix_optimizer(imgs, targets_in, params):

   optimizers = []
   depends = []

   gen_loss_data = 0

   targets = targets_in

   with tf.variable_scope(params.getModelName()):
      outputs = params.model(imgs, params)

   with tf.variable_scope("generator_loss"):
      gen_loss_data = params.loss(outputs, targets)

   gen_loss = params.alphaData * gen_loss_data

   disc = params.alphaDisc > 0.0

   if disc:

      disc_targets_img, disc_targets_pred, disc_outputs_img, disc_outputs_pred = params.prepare_disc(
          imgs, targets, outputs)

      with tf.variable_scope("discriminator"):
         disc_targets = params.disc_model_targets(
             disc_targets_img, disc_targets_pred, params)
         disc_outputs = params.disc_model_outputs(
             disc_outputs_img, disc_outputs_pred, params)

      disc_loss_targets = params.disc_loss_targets(disc_targets)
      disc_loss_outputs = params.disc_loss_outputs(disc_outputs)

      disc_loss = disc_loss_targets + disc_loss_outputs

      gen_loss = gen_loss + disc_loss * params.alphaDisc

      with tf.variable_scope("discriminator_train"):
         with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            disc_tvars = [var for var in tf.trainable_variables(
            ) if var.name.startswith("discriminator")]
            disc_optim = tf.train.AdamOptimizer(params.learningRate)
            disc_grads_and_vars = disc_optim.compute_gradients(
                disc_loss, var_list=disc_tvars)
            disc_train = disc_optim.apply_gradients(disc_grads_and_vars)

      depends.append(disc_train)

   with tf.variable_scope("generator_train"):
      depends = depends if len(depends) > 0 else tf.get_collection(
          tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(depends):
         gen_tvars = [var for var in tf.trainable_variables(
         ) if var.name.startswith("generator")]
         gen_optim = tf.train.AdamOptimizer(params.learningRate)
         gen_grads_and_vars = gen_optim.compute_gradients(
             gen_loss, var_list=gen_tvars)
         gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

   optimizers.append(gen_train)

   trSum = []
   tsSum = []
   valSum = []
   trSum.append(tf.summary.scalar("learningRate",
                                  params.learningRate, family="LearningParams"))
   tsSum.append(tf.summary.scalar(
       "generator_loss", gen_loss, family="lossGen"))
   tsSum.append(tf.summary.scalar("generator_loss_data",
                                  gen_loss_data, family="lossGen"))
   if (params.doExtSummary):
      for var in gen_tvars:
         trSum.append(tf.summary.histogram(var.name, var, family="varGen"))
      for grad, var in gen_grads_and_vars:
         trSum.append(tf.summary.histogram(
             var.name + '_gradient', grad, family="gradGen"))

   if disc:

      tsSum.append(tf.summary.scalar(
          "discriminator_loss", disc_loss, family="lossDisc"))
      tsSum.append(tf.summary.scalar(
          "discriminator_loss_gt", disc_loss_targets, family="lossDisc"))
      tsSum.append(tf.summary.scalar(
          "discriminator_loss_gen", disc_loss_outputs, family="lossDisc"))

      if (params.doExtSummary):
         for var in disc_tvars:
            trSum.append(tf.summary.histogram(
                var.name, var, family="varDisc"))
         for grad, var in disc_grads_and_vars:
            trSum.append(tf.summary.histogram(
                var.name + '_gradient', grad, family="gradDisc"))

      disc_targets_disp = disc_targets if params.data_format == 'NHWC' else tf.transpose(
          disc_targets, [0, 2, 3, 1])
      disc_outputs_disp = disc_outputs if params.data_format == 'NHWC' else tf.transpose(
          disc_outputs, [0, 2, 3, 1])

      disc_targets_out = tf.constant(
          1., shape=disc_targets_disp.shape) - tf.nn.sigmoid(disc_targets_disp)
      targetsSamplesDisc = tf.concat([[disc_targets_out[it, :, :, :]]
                                      for it in range(16)], axis=2)
      disc_outputs_out = tf.constant(
          1., shape=disc_outputs_disp.shape) - tf.nn.sigmoid(disc_outputs_disp)
      outputSamplesDisc = tf.concat([[disc_outputs_out[it, :, :, :]]
                                     for it in range(16)], axis=2)
      imgSamples = tf.concat([targetsSamplesDisc, outputSamplesDisc], axis=1)
      tsSum.append(tf.summary.image("samplesDisc", imgSamples))

   if params.doClassOut:
      targetsSamples = tf.multiply(
          tf.subtract(tf.to_float(targets)/params.nbOutputChannels, 0.5), 2.0)
      argmax_axis = 3 if params.data_format == 'NHWC' else 1
      outputSamples = tf.multiply(
          tf.subtract(tf.to_float(tf.argmax(outputs, axis=argmax_axis))/params.nbOutputChannels, 0.5), 2.0)
   else:
      targetsSamples = targets
      outputSamples = outputs

   imgSamples = []

   imgs_disp = imgs if params.data_format == 'NHWC' else tf.transpose(imgs, [
       0, 2, 3, 1])
   imgSz = imgs_disp.get_shape()
   sliceSz = [imgSz[0], imgSz[1], imgSz[2], 1]

   for i in range(params.inDispRange.shape[0]):
      inSamples = tf.concat(
          [tf.slice(imgs_disp, [0, 0, 0, it], sliceSz) for it in params.inDispRange[i]], axis=3)
      inSamples = tf.concat([[inSamples[it, :, :, 0:3]]
                             for it in range(16)], axis=2)
      imgSamples.append(tf.clip_by_value(inSamples, -1.0, 1.0))

   outputs_disp = params.dispProcessOutputs(
       outputSamples) if not params.dispProcessOutputs is None else outputSamples
   outputs_disp = outputs_disp if params.data_format == 'NHWC' else tf.transpose(
       outputs_disp, [0, 2, 3, 1])

   for i in range(params.outDispRange.shape[0]):
      outSamples = tf.concat(
          [tf.slice(outputs_disp, [0, 0, 0, it], sliceSz) for it in params.outDispRange[i]], axis=3)
      outSamples = tf.concat([[outSamples[it, :, :, 0:3]]
                              for it in range(16)], axis=2)
      imgSamples.append(tf.clip_by_value(outSamples, -1.0, 1.0))

   ioSamples = tf.concat(imgSamples, axis=1)

   valSum.append(tf.summary.image("out_samples", ioSamples))

   targets_disp = targetsSamples if params.data_format == 'NHWC' else tf.transpose(
       targetsSamples, [0, 2, 3, 1])

   for i in range(params.outDispRange.shape[0]):
      outSamples = tf.concat(
          [tf.slice(targets_disp, [0, 0, 0, it], sliceSz) for it in params.outDispRange[i]], axis=3)
      outSamples = tf.concat([[outSamples[it, :, :, 0:3]]
                              for it in range(16)], axis=2)
      imgSamples.append(tf.clip_by_value(outSamples, -1.0, 1.0))

   iotSamples = tf.concat(imgSamples, axis=1)

   tsSum.append(tf.summary.image("out_samples", iotSamples))

   trSum = tf.summary.merge(trSum, "Train")
   tsSum = tf.summary.merge(tsSum, "Test")
   valSum = tf.summary.merge(valSum, "Val")

   return [optimizers, gen_loss, trSum, tsSum, valSum]


def addSummaryParams(summary, params, gen_tvars, gen_grads_and_vars):

   summary.append(tf.summary.scalar(
       "learningRate", params.learningRate, family="Params"))

   if (params.doExtSummary):

      for var in gen_tvars:
         summary.append(tf.summary.histogram(
             var.name, var, family="varGen"))

      for grad, var in gen_grads_and_vars:
         summary.append(tf.summary.histogram(
             var.name + '_gradient', grad, family="gradGen"))


def addSummaryScalar(summary, var, groupname, varname):

   summary.append(tf.summary.scalar(varname, var, family=groupname))


def addSummaryImages(summary, name, params, imBatchLst, chnLst):

   imSummary = []

   nbIm = params.batchSz

   for l in range(len(imBatchLst)):

      im = postprocess(imBatchLst[l], False, params.data_format)
      chn = chnLst[l]

      szIm = im.get_shape()
      szSlice = [szIm[0], szIm[1], szIm[2], 1]

      imCat = tf.concat(
          [tf.slice(im, [0, 0, 0, it], szSlice) for it in chn], axis=3)
      imCat = tf.concat([[imCat[it, :, :, 0:3]]
                         for it in range(nbIm)], axis=2)

      imSummary.append(tf.clip_by_value(imCat, -1.0, 1.0))

   summary.append(tf.summary.image(name, tf.concat(imSummary, axis=1)))

#-----------------------------------------------------------------------------------------------------
# IMG2VECTOR MODELS
#-----------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------
# TRAINED VGG16
#-----------------------------------------------------------------------------------------------------


def loadVGG16(x, vggParamsFilename, maxLayerId, sess, toTrain=False):

   parameters = []
   layers = []
   currLayer = 0

   with tf.name_scope('vgg') as scope:

      # convert from -1..1 to 0..255
      mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[
          1, 1, 1, 3], name='img_mean')

      vggLayer = tf.multiply(tf.add(x, 1.0), 127.5) - mean

      # conv1_1
      with tf.name_scope('conv1_1') as scope:
         kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                  stddev=1e-1), name='weights')
         conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
         biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                              trainable=toTrain, name='biases')
         out = tf.nn.bias_add(conv, biases)
         vggLayer = tf.nn.relu(out, name=scope)
         parameters += [kernel, biases]
         layers += [vggLayer]
         currLayer += 1

      if (currLayer < maxLayerId):

         # conv1_2
         with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # pool1
         vggLayer = tf.nn.max_pool(vggLayer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool1')

         # conv2_1
         with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # conv2_2
         with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # pool2
         vggLayer = tf.nn.max_pool(vggLayer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool2')

         # conv3_1
         with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # conv3_2
         with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # conv3_3
         with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # pool3
         vggLayer = tf.nn.max_pool(vggLayer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool3')

         # conv4_1
         with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # conv4_2
         with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # conv4_3
         with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # pool4
         vggLayer = tf.nn.max_pool(vggLayer,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME',
                                   name='pool4')

         # conv5_1
         with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # conv5_2
         with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      if (currLayer < maxLayerId):

         # conv5_3
         with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(vggLayer, kernel, [
                1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=toTrain, name='biases')
            out = tf.nn.bias_add(conv, biases)
            vggLayer = tf.nn.relu(out, name=scope)
            parameters += [kernel, biases]
            layers += [vggLayer]
            currLayer += 1

      # set the parameters
      weights = np.load(vggParamsFilename)
      keys = sorted(weights.keys())
      for i, k in enumerate(keys):
         if i >= currLayer:
            break
         sess.run(parameters[i].assign(weights[k]))

      return layers
