""" common for tflowModel
"""

import sys
import os
import time
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

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
tf.logging.set_verbosity(tf.logging.INFO)

#-----------------------------------------------------------------------------------------------------
# Tensorflow Utils
#


def printVarTF(sess):
    tvars = tf.trainable_variables()
    for var in tvars:
        print var.name
        print var.eval(sess)

#-----------------------------------------------------------------------------------------------------
# Images Utils
#


def showImgs(batch, img_depths):

    n_imgs = len(img_depths)

    if np.sum(img_depths) != batch.shape[3]:
        raise ValueError()

    batch_im = np.zeros(
        (batch.shape[0] * batch.shape[1], n_imgs * batch.shape[2], 3))

    for b in range(batch.shape[0]):

        n_offset = 0
        for n in range(n_imgs):

            im_d = img_depths[n]
            im = batch[b, :, :, n_offset:n_offset + im_d]

            if im_d > 3:
                gray = np.mean(im, axis=2)
                im = np.stack([gray, gray, gray], 2)

            batch_im[b * batch.shape[1]:(b + 1) * batch.shape[1],
                     n * batch.shape[2]:(n + 1) * batch.shape[2], 0:im_d] = im

            n_offset += im_d

    plt.imshow(batch_im)
    plt.show()


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

    return [im]


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
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class LearningParams:

    def __init__(self, modelPath, seed=int(time.time())):

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

        self.baseLearningRate = 0.0001
        self.learningRate = tf.train.polynomial_decay(self.baseLearningRate, self.globalStep, self.numSteps, 0.0,
                                                      power=0.7)

        self.isTraining = tf.placeholder(tf.bool)

        self.tbLogsPath = modelPath + "/tbLogs"
        self.modelFilename = modelPath + "/tfData"
        self.modelNbToKeep = 3

        self.rseed = seed

        tf.set_random_seed(self.rseed)

    def update(self):

        self.numMaxSteps = max(self.numMaxSteps, self.numSteps)
        self.learningRate = tf.train.polynomial_decay(self.baseLearningRate, self.globalStep, self.numMaxSteps, 0.0,
                                                      power=0.7)
        tf.set_random_seed(self.rseed)


#-----------------------------------------------------------------------------------------------------
# OPERATORS
#-----------------------------------------------------------------------------------------------------


def preprocess(imgs):
    # -----> preprocessing : put the pix values in [-1..1]
    with tf.name_scope('preprocess') as scope:

        img_mean = tf.constant([0.5, 0.5, 0.5], dtype=tf.float32, shape=[
            1, 1, 1, 3], name='img_mean')

        return tf.multiply(tf.subtract(imgs, img_mean), 2.0)


def postprocess(imgs):
    # -----> preprocessing : put the pix values in [0..1]
    with tf.name_scope('postprocess') as scope:

        img_mean = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32, shape=[
            1, 1, 1, 3], name='img_mean')

        return tf.multiply(tf.add(imgs, img_mean), 0.5)


#-----------------------------------------------------------------------------------------------------
# IMG2IMG MODELS
#-----------------------------------------------------------------------------------------------------

class Pix2PixParams(LearningParams):

    def __init__(self, modelPath, seed=int(time.time())):

        LearningParams.__init__(self, modelPath, seed)

        self.useBatchNorm = True

        self.nbChannels = 64
        self.nbInChannels = 3
        self.nbOutputChannels = 3
        self.kernelSz = 4
        self.stridedEncoder = True
        self.stridedDecoder = True

        self.doLogNormOutputs = False

        self.inDispRange = []
        self.outDispRange = []

        self.alphaData = 1.0
        self.alphaDisc = 0.0

        self.doClassOut = False

#
# n = number of output channels
# ks = kernel size
# ss = stride size
# bn = do batch norm
#


def pix2pix_convo(inputs, i_n, o_n, ks, ss, bias):

    return tf.layers.conv2d(inputs, o_n, ks, ss,
                            padding='same',
                            use_bias=bias,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())


def pix2pix_conv(inputs, i_n, o_n, ks, ss, bias):

    initializer = tf.contrib.layers.xavier_initializer()

    shape = [ks, ks, i_n, o_n]
    w = tf.Variable(initializer(shape))
    pad_size = ks//2
    pad_mat = np.array([[0, 0], [pad_size, pad_size],
                        [pad_size, pad_size], [0, 0]])
    x = tf.pad(inputs, pad_mat)
    x = tf.nn.conv2d(x, w, strides=[1, ss, ss, 1], padding='VALID')
    if bias:
        x = tf.nn.bias_add(x, tf.Variable(initializer([o_n])))

    return x


def pix2pix_nconv(inputs, i_n, o_n, ks, ss):

    return tf.nn.relu(pix2pix_conv(inputs, i_n, o_n, ks, ss, True))


def pix2pix_nencoder(inputs, i_n, o_n, ks, ss, name):

    with tf.variable_scope(name):
        layer = pix2pix_nconv(inputs, i_n, o_n, ks, ss)
        layer = tf.nn.relu(layer)
        return layer


def pix2pix_ndecoder(inputs, skip, i_n, o_n, ks, no_skip, name):

    with tf.variable_scope(name):

        layer_shape = tf.shape(skip)
        layer = tf.image.resize_images(inputs,
                                       [layer_shape[1], layer_shape[2]])
        layer = pix2pix_nconv(layer, i_n, o_n, ks, 1)

        if not no_skip:
            layer = tf.concat([skip, layer], axis=3)

        return layer


def pix2pix_ngen(imgs, params):

    n = params.nbChannels
    nIn = params.nbInChannels
    nOut = params.nbOutputChannels

    ks = params.kernelSz
    ess = params.stridedEncoder
    dss = params.stridedDecoder

    bn = params.useBatchNorm

    train = params.isTraining

    # encoder part

    # S x I --> S x N
    encoder_0 = pix2pix_nencoder(imgs, nIn, n, 3, 1, "encoder_0")
    # S x N --> S/2 x 2*N
    encoder_1 = pix2pix_nencoder(encoder_0, n, n*2, ks, 2, "encoder_1")
    # S/2 x 2*N --> S/4 x 4*N
    encoder_2 = pix2pix_nencoder(encoder_1, n*2, n*4, ks, 2, "encoder_2")
    # S/4 x 4*N --> S/8 x 4*N
    encoder_3 = pix2pix_nencoder(encoder_2, n*4, n*4, ks, 2, "encoder_3")
    # S/8 x 8*N --> S/16 x 8*N
    encoder_4 = pix2pix_nencoder(encoder_3, n*4, n*8, ks, 2, "encoder_4")
    # S/16 x 8*N --> S/32 x 8*N
    encoder_5 = pix2pix_nencoder(encoder_4, n*8, n*8, ks, 2, "encoder_5")

    # decoder with skip connections

    # S/32 x 8*N --> S/16 x 12*N
    decoder_5 = pix2pix_ndecoder(
        encoder_5, encoder_4, n*8, n*4, ks, False, "decoder_5")
    # S/16 x 12*N --> S/8 x 8*N
    decoder_4 = pix2pix_ndecoder(
        decoder_5, encoder_3, n*12, n*4, ks, False, "decoder_4")
    # S/8 x 8*N --> S/4 x 6*N
    decoder_3 = pix2pix_ndecoder(
        decoder_4, encoder_2, n*8, n*2, ks, False, "decoder_3")
    # S/4 x 6*N --> S/2 x 4*N
    decoder_2 = pix2pix_ndecoder(
        decoder_3, encoder_1, n*6, n*2, ks, False, "decoder_2")
    # S/2 x 4*N --> S x 2*N
    decoder_1 = pix2pix_ndecoder(
        decoder_2, encoder_0, n*4, n, ks, False, "decoder_1")
    # S x N --> S x I
    with tf.variable_scope("decoder_0"):
        output = pix2pix_nconv(decoder_1, n*2, nOut, 3, 1)
        if not params.doClassOut:
            output = tf.nn.tanh(output)

    return output


def pix2pix_conv_bn(inputs, i_n, o_n, ks, ss, strided, bn, train):

    if strided:
        layer = pix2pix_conv(inputs, i_n, o_n, ks, ss, not bn)

    else:
        layer = layer = pix2pix_conv(inputs, i_n, o_n, ks, 1, not bn)
        layer_shape = tf.shape(layer)
        layer = tf.image.resize_area(
            layer, [layer_shape[1] / ss, layer_shape[2] / ss])

    if bn:
        layer = tf.layers.batch_normalization(layer,
                                              axis=3,
                                              momentum=0.1,
                                              epsilon=1e-5,
                                              center=True,
                                              scale=False,
                                              training=train)
    return layer


def pix2pix_deconv_bn(inputs, ref, i_n, o_n, ks, ss, strided, bn, train):

    if strided:
        layer = tf.layers.conv2d_transpose(inputs, o_n,
                                           kernel_size=ks,
                                           strides=(ss, ss),
                                           padding="same",
                                           use_bias=False,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
        # should pad the layer tensor to be able to concat it with the ref tensor
        if ss > 1:
            pad_mat = np.array([[0, 0], [0, ref.shape[1]-ref.shape[1]],
                                [0, ref.shape[2]-ref.shape[2]], [0, 0]])
            layer = tf.pad(layer, pad_mat)
            #layer = pix2pix_conv(layer, i_n, o_n, ks, 1, not bn)

    else:
        layer_shape = tf.shape(ref)
        layer = tf.image.resize_bilinear(inputs,
                                         [layer_shape[1], layer_shape[2]])
        layer = pix2pix_conv(layer, i_n, o_n, ks, 1, not bn)

    if bn:
        layer = tf.layers.batch_normalization(layer,
                                              axis=3,
                                              momentum=0.1,
                                              epsilon=1e-5,
                                              center=True,
                                              scale=False,
                                              training=train)

    return layer


def pix2pix_deception_bn(inputs, i_n, n, o_n, ss, strided, bn, name, train):

    with tf.variable_scope(name):

        c3 = pix2pix_conv_bn(inputs, i_n, n, 1, 1, True, bn, train)
        c3 = tf.nn.relu(c3)
        c3 = pix2pix_conv_bn(c3, n, o_n//2, 3, ss, strided, bn, train)
        c3 = tf.nn.relu(c3)

        c7 = pix2pix_conv_bn(inputs, i_n, n, 1, 1, True, bn, train)
        c7 = tf.nn.relu(c7)
        c7 = pix2pix_conv_bn(c7, n, o_n//2, 7, ss, strided, bn, train)
        c7 = tf.nn.relu(c7)

        layer = tf.concat([c3, c7], axis=3)

        return layer


def pix2pix_uception_bn(inputs, ref, i_n, n, o_n, ss, strided, bn, name, train):

    with tf.variable_scope(name):

        c3 = pix2pix_conv_bn(inputs, i_n, n, 1, 1, True, bn, train)
        c3 = tf.nn.relu(c3)
        c3 = pix2pix_deconv_bn(c3, ref, n, o_n//2, 3, ss, strided, bn, train)
        c3 = tf.nn.relu(c3)

        c7 = pix2pix_conv_bn(inputs, i_n, n, 1, 1, True, bn, train)
        c7 = tf.nn.relu(c7)
        c7 = pix2pix_deconv_bn(c7, ref, n, o_n//2, 7, ss, strided, bn, train)
        c7 = tf.nn.relu(c7)

        layer = tf.concat([c3, c7], axis=3)
        layer = tf.add(layer, ref)

        return layer


def pix2pix_encoder_bn(inputs, i_n, o_n, ks, ss, strided, bn, name, train):

    with tf.variable_scope(name):

        layer = pix2pix_conv_bn(inputs, i_n, o_n, ks, ss, strided, bn, train)
        layer = tf.nn.relu(layer)

        return layer


def pix2pix_decoder(inputs, ref, i_n, o_n, ks, ss, strided, bn, name, train):

    with tf.variable_scope(name):

        layer = pix2pix_deconv_bn(
            inputs, ref, i_n, o_n, ks, ss, strided, bn, train)
        layer = tf.nn.relu(layer)

        return layer


def pix2pix_decoder_skip(inputs, skip, i_n, o_n, ks, ss, strided, bn, name, train):

    with tf.variable_scope(name):

        layer = pix2pix_deconv_bn(
            inputs, skip, i_n, o_n, ks, ss, strided, bn, train)
        layer = tf.nn.relu(layer)

        layer = tf.concat([skip, layer], axis=3)

        return layer


def pix2pix_decoder_res(inputs, res, i_n, o_n, ks, ss, strided, bn, name, train):

    with tf.variable_scope(name):

        layer = pix2pix_deconv_bn(
            inputs, res, i_n, o_n, ks, ss, strided, bn, train)
        layer = tf.nn.relu(layer)

        layer = tf.add(layer, res)

        return layer


def pix2pix_gen(imgs, params):

    n = params.nbChannels
    nIn = params.nbInChannels
    nOut = params.nbOutputChannels

    ks = params.kernelSz
    ess = params.stridedEncoder
    dss = params.stridedDecoder

    bn = params.useBatchNorm

    train = params.isTraining

    # encoder part

    # S x I --> S x N
    encoder_0 = pix2pix_encoder_bn(
        imgs, nIn, n, 3, 1, True, False, "encoder_0", train)
    # S x N --> S/2 x 2*N
    encoder_1 = pix2pix_encoder_bn(
        encoder_0, n, n*2, ks, 2, ess, bn, "encoder_1", train)
    # S/2 x 2*N --> S/4 x 4*N
    encoder_2 = pix2pix_encoder_bn(
        encoder_1, n*2, n*4, ks, 2, ess, bn, "encoder_2", train)
    # S/4 x 4*N --> S/8 x 4*N
    encoder_3 = pix2pix_encoder_bn(
        encoder_2, n*4, n*4, ks, 2, ess, bn, "encoder_3", train)
    # S/8 x 8*N --> S/16 x 8*N
    encoder_4 = pix2pix_encoder_bn(
        encoder_3, n*4, n*8, ks, 2, ess, bn, "encoder_4", train)
    # S/16 x 8*N --> S/32 x 8*N
    encoder_5 = pix2pix_encoder_bn(
        encoder_4, n*8, n*8, ks, 2, ess, bn, "encoder_5", train)

    # decoder with skip connections

    # S/32 x 8*N --> S/16 x 12*N
    decoder_5 = pix2pix_decoder_skip(
        encoder_5, encoder_4, n*8, n*4, ks, 2, dss, bn, "decoder_5", train)
    # S/16 x 12*N --> S/8 x 8*N
    decoder_4 = pix2pix_decoder_skip(
        decoder_5, encoder_3, n*12, n*4, ks, 2, dss, bn, "decoder_4", train)
    # S/8 x 8*N --> S/4 x 6*N
    decoder_3 = pix2pix_decoder_skip(
        decoder_4, encoder_2, n*8, n*2, ks, 2, dss, bn, "decoder_3", train)
    # S/4 x 6*N --> S/2 x 4*N
    decoder_2 = pix2pix_decoder_skip(
        decoder_3, encoder_1, n*6, n*2, ks, 2, dss, bn, "decoder_2", train)
    # S/2 x 4*N --> S x 2*N
    decoder_1 = pix2pix_decoder_skip(
        decoder_2, encoder_0, n*4, n, ks, 2, dss, bn, "decoder_1", train)
    # S x N --> S x I
    with tf.variable_scope("decoder_0"):
        output = pix2pix_conv_bn(decoder_1, n*2, nOut,
                                 3, 1, True, False, train)
        if not params.doClassOut:
            output = tf.nn.tanh(output)

    return output


def pix2pix_res(imgs, params):

    n = params.nbChannels
    nIn = params.nbInChannels
    nOut = params.nbOutputChannels

    ks = params.kernelSz
    ess = params.stridedEncoder
    dss = params.stridedDecoder

    bn = params.useBatchNorm

    train = params.isTraining

    # encoder part

    # S x I --> S x N
    encoder_0 = pix2pix_encoder_bn(
        imgs, nIn, n, 3, 1, True, False, "encoder_0", train)
    # S x N --> S/2 x 2*N
    encoder_1 = pix2pix_encoder_bn(
        encoder_0, n, n*2, ks, 2, ess, bn, "encoder_1", train)
    # S/2 x 2*N --> S/4 x 4*N
    encoder_2 = pix2pix_encoder_bn(
        encoder_1, n*2, n*4, ks, 2, ess, bn, "encoder_2", train)
    # S/4 x 4*N --> S/8 x 4*N
    encoder_3 = pix2pix_encoder_bn(
        encoder_2, n*4, n*4, ks, 2, ess, bn, "encoder_3", train)
    # S/8 x 8*N --> S/16 x 8*N
    encoder_4 = pix2pix_encoder_bn(
        encoder_3, n*4, n*8, ks, 2, ess, bn, "encoder_4", train)
    # S/16 x 8*N --> S/32 x 8*N
    encoder_5 = pix2pix_encoder_bn(
        encoder_4, n*8, n*8, ks, 2, ess, bn, "encoder_5", train)

    # decoder with skip connections

    # S/32 x 8*N --> S/16 x 8*N
    decoder_5 = pix2pix_decoder_res(
        encoder_5, encoder_4, n*8, n*8, ks, 2, dss, bn, "decoder_5", train)
    # S/16 x 12*N --> S/8 x 4*N
    decoder_4 = pix2pix_decoder_res(
        decoder_5, encoder_3, n*8, n*4, ks, 2, dss, bn, "decoder_4", train)
    # S/8 x 8*N --> S/4 x 4*N
    decoder_3 = pix2pix_decoder_res(
        decoder_4, encoder_2, n*4, n*4, ks, 2, dss, bn, "decoder_3", train)
    # S/4 x 6*N --> S/2 x 2*N
    decoder_2 = pix2pix_decoder_res(
        decoder_3, encoder_1, n*4, n*2, ks, 2, dss, bn, "decoder_2", train)
    # S/2 x 4*N --> S x N
    decoder_1 = pix2pix_decoder_res(
        decoder_2, encoder_0, n*2, n, ks, 2, dss, bn, "decoder_1", train)
    # S x N --> S x I
    with tf.variable_scope("decoder_0"):
        output = pix2pix_conv_bn(decoder_1, n, nOut,
                                 3, 1, True, False, train)
        if not params.doClassOut:
            output = tf.nn.tanh(output)

    return output


def pix2pix_ires(imgs, params):

    n = params.nbChannels
    nIn = params.nbInChannels
    nOut = params.nbOutputChannels

    ks = params.kernelSz
    ess = params.stridedEncoder
    dss = params.stridedDecoder

    bn = params.useBatchNorm

    train = params.isTraining

    # encoder part

    # S x I --> S x 2*N
    encoder_0 = pix2pix_encoder_bn(
        imgs, nIn, n*2, 3, 1, True, False, "encoder_0", train)
    # S x N --> S/2 x 2*N
    encoder_1 = pix2pix_deception_bn(
        encoder_0, n*2, n, n*2, 2, ess, bn, "encoder_1", train)
    # S/2 x 2*N --> S/4 x 4*N
    encoder_2 = pix2pix_deception_bn(
        encoder_1, n*2, n, n*4, 2, ess, bn, "encoder_2", train)
    # S/4 x 4*N --> S/8 x 4*N
    encoder_3 = pix2pix_deception_bn(
        encoder_2, n*4, n, n*4, 2, ess, bn, "encoder_3", train)
    # S/8 x 4*N --> S/16 x 8*N
    encoder_4 = pix2pix_deception_bn(
        encoder_3, n*4, n, n*8, 2, ess, bn, "encoder_4", train)
    # S/16 x 8*N --> S/32 x 8*N
    encoder_5 = pix2pix_deception_bn(
        encoder_4, n*8, n, n*8, 2, ess, bn, "encoder_5", train)

    # decoder with skip connections

    # S/32 x 8*N --> S/16 x 8*N
    decoder_5 = pix2pix_uception_bn(
        encoder_5, encoder_4, n*8, n, n*8, 2, dss, bn, "decoder_5", train)
    # S/16 x 8*N --> S/8 x 4*N
    decoder_4 = pix2pix_uception_bn(
        decoder_5, encoder_3, n*8, n, n*4, 2, dss, bn, "decoder_4", train)
    # S/8 x 4*N --> S/4 x 4*N
    decoder_3 = pix2pix_uception_bn(
        decoder_4, encoder_2, n*4, n, n*4, 2, dss, bn, "decoder_3", train)
    # S/4 x 4*N --> S/2 x 2*N
    decoder_2 = pix2pix_uception_bn(
        decoder_3, encoder_1, n*4, n, n*2, 2, dss, bn, "decoder_2", train)
    # S/2 x 2*N --> S x 2*N
    decoder_1 = pix2pix_uception_bn(
        decoder_2, encoder_0, n*2, n, n*2, 2, dss, bn, "decoder_1", train)
    # S x N --> S x I
    with tf.variable_scope("decoder_0"):
        output = pix2pix_conv_bn(decoder_1, n*2, nOut,
                                 3, 1, True, False, train)
        if not params.doClassOut:
            output = tf.nn.tanh(output)

    return output


def pix2pix_gen_orig(imgs, params):

    nIn = params.nbChannels
    nOut = params.nbOutputChannels

    ks = params.kernelSz
    ess = params.stridedEncoder
    dss = params.stridedDecoder

    bn = params.useBatchNorm

    train = params.isTraining

    # encoder part

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, n]
    # no batch norm for the first layer : why ?
    encoder_1 = pix2pix_encoder_bn(
        imgs, nIn, ks, 2, ess, False, "encoder_1", train)
    # encoder_2: [batch, 128, 128, n] => [batch, 64, 64, n * 2]
    encoder_2 = pix2pix_encoder_bn(
        encoder_1, nIn*2, ks, 2, ess, bn, "encoder_2", train)
    # encoder_3: [batch, 64, 64, n * 2] => [batch, 32, 32, n * 4]
    encoder_3 = pix2pix_encoder_bn(
        encoder_2, nIn*4, ks, 2, ess, bn, "encoder_3", train)
    # encoder_4: [batch, 32, 32, n * 4] => [batch, 16, 16, n * 8]
    encoder_4 = pix2pix_encoder_bn(
        encoder_3, nIn*8, ks, 2, ess, bn, "encoder_4", train)
    # encoder_5: [batch, 16, 16, n * 8] => [batch, 8, 8, n * 8]
    encoder_5 = pix2pix_encoder_bn(
        encoder_4, nIn*8, ks, 2, ess, bn, "encoder_5", train)
    # encoder_6: [batch, 8, 8, n * 8] => [batch, 4, 4, n * 8]
    encoder_6 = pix2pix_encoder_bn(
        encoder_5, nIn*8, ks, 2, ess, bn, "encoder_6", train)
    # encoder_7: [batch, 4, 4, n * 8] => [batch, 2, 2, n * 8]
    encoder_7 = pix2pix_encoder_bn(
        encoder_6, nIn*8, ks, 1, True, bn, "encoder_7", train)
    # encoder_8: [batch, 2, 2, n * 8] => [batch, 1, 1, n * 8]
    # encoder_8 = pix2pix_encoder_bn(
    #    encoder_7, n*8, "encoder_8", train)

    # decoder with skip connections

    # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
    # decoder_8 = pix2pix_decoder_skip(
    #    encoder_8, encoder_7, n*8, "decoder_8", dropout, train)
    # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
    decoder_7 = pix2pix_decoder_skip(
        encoder_7, encoder_6, nIn*8, ks, 1, True, bn, False, "decoder_7", train)  # , params.convDropOut)
    # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
    decoder_6 = pix2pix_decoder_skip(
        decoder_7, encoder_5, nIn*8, ks, 2, dss, bn, False, "decoder_6", train)  # , params.convDropOut)
    # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
    decoder_5 = pix2pix_decoder_skip(
        decoder_6, encoder_4, nIn*4, ks, 2, dss, bn, False, "decoder_5", train)
    # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
    decoder_4 = pix2pix_decoder_skip(
        decoder_5, encoder_3, nIn*4, ks, 2, dss, bn, False, "decoder_4", train)
    # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
    decoder_3 = pix2pix_decoder_skip(
        decoder_4, encoder_2, nIn*2, ks, 2, dss, bn, False, "decoder_3", train)
    # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    decoder_2 = pix2pix_decoder_skip(
        decoder_3, encoder_1, nIn*2, ks, 2, dss, bn, False, "decoder_2", train)
    # decoder_1: output
    decoder_1 = pix2pix_decoder_skip(
        decoder_2, imgs, nIn, ks, 2, dss, bn, True, "decoder_1", train)

    with tf.variable_scope("gen_output"):
        output = pix2pix_conv_bn(decoder_1, nOut, ks, 1, True, False, train)
        if not lp.doClassOut:
            output = tf.nn.tanh(output)

    return output


def pix2pix_disc(gen_inputs, gen_outputs, params):

    train = params.isTraining

    n = params.nbChannels
    nIn = params.nbInChannels
    nOut = params.nbOutputChannels

    ks = params.kernelSz
    ess = params.stridedEncoder
    bn = params.useBatchNorm

    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):

        inputs = tf.concat([gen_inputs, gen_outputs], axis=3)

        # layer_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ndf]
        layer_1 = pix2pix_encoder_bn(
            inputs, nIn, n, ks, ess, False, "layer_1", train)
        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        layer_2 = pix2pix_encoder_bn(
            layer_1, n, n*2, ks, 2, ess, bn, "layer_2", train)
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        layer_3 = pix2pix_encoder_bn(
            layer_2, n*2, n*4, ks, 1, True, bn, "layer_3", train)
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        layer_4 = pix2pix_encoder_bn(
            layer_3, n*4, n*8, ks, 2, ess, bn, "layer_4", train)
        # layer_5: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        layer_5 = pix2pix_encoder_bn(
            layer_4, n*8, n*8, ks, 1, True, bn, "layer_5", train)

        # layer_6
        with tf.variable_scope("layer_6"):
            layer_6 = pix2pix_conv(layer_5, n*8, 1, ks, ess, True)
            # layer_6 = tf.nn.sigmoid(layer_6)

    return layer_6


def pix2pix_optimizer(imgs, targets_in, params):

    targets = targets_in

    optimizers = []
    depends = []

    gen_loss_data = 0

    with tf.variable_scope("generator"):
        outputs = params.model(imgs, params)
        if params.doLogNormOutputs:
            # scale from [-1..1] to [1..7] and apply log -> [0..2]
            outputs_sc = tf.log(tf.add(tf.multiply(outputs, 3.0), 4.0))
            targets_sc = tf.log(tf.add(tf.multiply(targets, 3.0), 4.0))
        else:
            outputs_sc = outputs
            targets_sc = targets

    if params.doClassOut:
        gen_loss_data = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.squeeze(targets_sc), logits=outputs_sc))
    else:
        gen_data_diff = tf.subtract(outputs_sc, targets_sc)

        # for log normalization add the mean scaling term
        if params.doLogNormOutputs:
            gen_data_log_scales = tf.expand_dims(tf.expand_dims(
                tf.reduce_mean(gen_data_diff, axis=[1, 2]), axis=1), axis=2)
            gen_data_diff = tf.subtract(gen_data_diff, gen_data_log_scales)

        gen_loss_data = tf.reduce_mean(tf.sqrt(EPS + tf.square(gen_data_diff)))

    gen_loss = params.alphaData * gen_loss_data

    disc = params.alphaDisc > 0.0

    if disc:

        with tf.name_scope("discriminator_gt"):
            disc_targets = pix2pix_disc(imgs, targets, params)

        with tf.name_scope("discriminator_gen"):
            disc_outputs = pix2pix_disc(imgs, outputs, params)

        # disc_loss_targets = tf.reduce_mean(-tf.log(disc_targets + EPS))
        disc_loss_targets = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(1., shape=disc_targets.shape), logits=disc_targets))
        # disc_loss_outputs = tf.reduce_mean(-tf.log(1 - disc_outputs + EPS))
        disc_loss_outputs = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(0., shape=disc_outputs.shape), logits=disc_outputs))
        disc_loss = 2.0 * disc_loss_targets + disc_loss_outputs

        # gen_loss_disc = tf.reduce_mean(-tf.log(disc_outputs + EPS))
        gen_loss_disc = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(1., shape=disc_outputs.shape), logits=disc_outputs))

        gen_loss = gen_loss + gen_loss_disc * params.alphaDisc

        with tf.name_scope("discriminator_train"):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                disc_tvars = [var for var in tf.trainable_variables(
                ) if var.name.startswith("discriminator")]
                disc_optim = tf.train.AdamOptimizer(params.learningRate)
                disc_grads_and_vars = disc_optim.compute_gradients(
                    disc_loss, var_list=disc_tvars)
                disc_train = disc_optim.apply_gradients(disc_grads_and_vars)

        depends.append(disc_train)

    with tf.name_scope("generator_train"):
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
    tsSum.append(tf.summary.scalar(
        "generator_loss", gen_loss, family="lossGen"))
    tsSum.append(tf.summary.scalar("generator_loss_data",
                                   gen_loss_data, family="lossGen"))
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
        tsSum.append(tf.summary.scalar("generator_loss_dis",
                                       gen_loss_disc, family="lossGen"))
        for var in disc_tvars:
            trSum.append(tf.summary.histogram(var.name, var, family="varDisc"))
        for grad, var in disc_grads_and_vars:
            trSum.append(tf.summary.histogram(
                var.name + '_gradient', grad, family="gradDisc"))

        disc_targets_out = tf.constant(
            1., shape=disc_targets.shape) - tf.nn.sigmoid(disc_targets)
        targetsSamplesDisc = tf.concat([[disc_targets_out[it, :, :, :]]
                                        for it in range(16)], axis=2)
        disc_outputs_out = tf.constant(
            1., shape=disc_outputs.shape) - tf.nn.sigmoid(disc_outputs)
        outputSamplesDisc = tf.concat([[disc_outputs_out[it, :, :, :]]
                                       for it in range(16)], axis=2)
        imgSamples = tf.concat([targetsSamplesDisc, outputSamplesDisc], axis=1)
        tsSum.append(tf.summary.image("samplesDisc", imgSamples))

    if params.doClassOut:
        targetsSamples = tf.multiply(
            tf.subtract(tf.to_float(targets)/params.nbOutputChannels, 0.5), 2.0)
        outputSamples = tf.multiply(
            tf.subtract(tf.to_float(tf.expand_dims(tf.argmax(
                outputs, axis=3), axis=3))/params.nbOutputChannels, 0.5), 2.0)
    else:
        targetsSamples = targets
        outputSamples = outputs

    imgSamples = []

    imgSz = imgs.get_shape()
    sliceSz = [imgSz[0], imgSz[1], imgSz[2], 1]

    for i in range(params.inDispRange.shape[0]):
        inSamples = tf.concat(
            [tf.slice(imgs, [0, 0, 0, it], sliceSz) for it in params.inDispRange[i]], axis=3)
        inSamples = tf.concat([[inSamples[it, :, :, 0:3]]
                               for it in range(16)], axis=2)
        imgSamples.append(inSamples)

    for i in range(params.outDispRange.shape[0]):
        outSamples = tf.concat(
            [tf.slice(outputSamples, [0, 0, 0, it], sliceSz) for it in params.outDispRange[i]], axis=3)
        outSamples = tf.concat([[outSamples[it, :, :, 0:3]]
                                for it in range(16)], axis=2)
        imgSamples.append(outSamples)

    ioSamples = tf.concat(imgSamples, axis=1)

    valSum.append(tf.summary.image("out_samples", ioSamples))

    for i in range(params.outDispRange.shape[0]):
        outSamples = tf.concat(
            [tf.slice(targetsSamples, [0, 0, 0, it], sliceSz) for it in params.outDispRange[i]], axis=3)
        outSamples = tf.concat([[outSamples[it, :, :, 0:3]]
                                for it in range(16)], axis=2)
        imgSamples.append(outSamples)

    iotSamples = tf.concat(imgSamples, axis=1)

    tsSum.append(tf.summary.image("out_samples", iotSamples))

    trSum = tf.summary.merge(trSum, "Train")
    tsSum = tf.summary.merge(tsSum, "Test")
    valSum = tf.summary.merge(valSum, "Val")

    return [optimizers, gen_loss, trSum, tsSum, valSum]

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
