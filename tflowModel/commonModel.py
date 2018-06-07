""" common for tflowModel
"""

import sys
import os
import time
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

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

#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------


class LearningParams:

    def __init__(self, modelPath, seed=int(time.time())):

        self.numSteps = 75000
        self.trlogStep = 75
        self.tslogStep = 150
        self.backupStep = 1500

        self.batchSz = 64
        self.imgSzTr = [64, 64]
        self.imgSzTs = [256, 256]

        self.globalStep = tf.Variable(0, trainable=False)

        self.baseLearningRate = 0.0001
        self.learningRate = tf.train.polynomial_decay(self.baseLearningRate, self.globalStep, self.numSteps, 0.0,
                                                      power=0.7)
        self.dropoutProb = tf.placeholder(
            tf.float32)  # dropout (keep probability)

        self.isTraining = tf.placeholder(tf.bool)

        self.tbLogsPath = modelPath + "/tbLogs"
        self.modelFilename = modelPath + "/tfData"

        self.rseed = seed

        tf.set_random_seed(self.rseed)

    def update(self):

        self.learningRate = tf.train.polynomial_decay(self.baseLearningRate, self.globalStep, self.numSteps, 0.0,
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

def pix2pix_encoder_bn(inputs, n, ks, ss, name, train):

    with tf.variable_scope(name):

        layer = tf.layers.conv2d(
            inputs, n, ks, ss, padding='same', use_bias=False,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        layer = tf.layers.batch_normalization(layer,
                                              axis=3,
                                              momentum=0.1,
                                              epsilon=1e-5,
                                              center=True,
                                              scale=False,
                                              training=train)
        layer = tf.nn.leaky_relu(layer)

        return layer


def pix2pix_decoder_skip(inputs, skip, n, ks, ss, name, dropout, train):

    with tf.variable_scope(name):

        layer = tf.layers.conv2d_transpose(inputs, n, kernel_size=ks, strides=(ss, ss),
                                           padding="same", use_bias=False,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())

        layer = tf.layers.batch_normalization(layer,
                                              axis=3,
                                              momentum=0.1,
                                              epsilon=1e-5,
                                              center=True,
                                              scale=False,
                                              training=train)
        layer = tf.nn.relu(layer)

        layer = tf.nn.dropout(layer, keep_prob=1.0 - dropout)

        layer = tf.concat([skip, layer], axis=3)

        return layer


def pix2pix_gen(imgs, output_channels, n, dropout, train):

    xinit = tf.contrib.layers.xavier_initializer()

    # encoder part

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, n]
    with tf.variable_scope("encoder_1"):

        encoder_1 = tf.layers.conv2d(
            imgs, n, 4, 2, padding='same', use_bias=True,
            bias_initializer=xinit, kernel_initializer=xinit)

        encoder_1 = tf.nn.leaky_relu(encoder_1)

    # encoder_2: [batch, 128, 128, n] => [batch, 64, 64, n * 2]
    encoder_2 = pix2pix_encoder_bn(
        encoder_1, n*2, 4, 2, "encoder_2", train)
    # encoder_3: [batch, 64, 64, n * 2] => [batch, 32, 32, n * 4]
    encoder_3 = pix2pix_encoder_bn(
        encoder_2, n*4, 4, 2, "encoder_3", train)
    # encoder_4: [batch, 32, 32, n * 4] => [batch, 16, 16, n * 8]
    encoder_4 = pix2pix_encoder_bn(
        encoder_3, n*8, 4, 2, "encoder_4", train)
    # encoder_5: [batch, 16, 16, n * 8] => [batch, 8, 8, n * 8]
    encoder_5 = pix2pix_encoder_bn(
        encoder_4, n*8, 4, 2, "encoder_5", train)
    # encoder_6: [batch, 8, 8, n * 8] => [batch, 4, 4, n * 8]
    encoder_6 = pix2pix_encoder_bn(
        encoder_5, n*8, 4, 2, "encoder_6", train)
    # encoder_7: [batch, 4, 4, n * 8] => [batch, 2, 2, n * 8]
    encoder_7 = pix2pix_encoder_bn(
        encoder_6, n*8, 4, 1, "encoder_7", train)
    # encoder_8: [batch, 2, 2, n * 8] => [batch, 1, 1, n * 8]
    # encoder_8 = pix2pix_encoder_bn(
    #    encoder_7, n*8, "encoder_8", train)

    # decoder with skip connections

    # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
    # decoder_8 = pix2pix_decoder_skip(
    #    encoder_8, encoder_7, n*8, "decoder_8", dropout, train)
    # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
    decoder_7 = pix2pix_decoder_skip(
        encoder_7, encoder_6, n*8, 4, 1, "decoder_7", dropout, train)
    # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
    decoder_6 = pix2pix_decoder_skip(
        decoder_7, encoder_5, n*8, 4, 2, "decoder_6", dropout, train)
    # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
    decoder_5 = pix2pix_decoder_skip(
        decoder_6, encoder_4, n*8, 4, 2, "decoder_5", 0.0, train)
    # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
    decoder_4 = pix2pix_decoder_skip(
        decoder_5, encoder_3, n*4, 4, 2, "decoder_4", 0.0, train)
    # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
    decoder_3 = pix2pix_decoder_skip(
        decoder_4, encoder_2, n*2, 4, 2, "decoder_3", 0.0, train)
    # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    decoder_2 = pix2pix_decoder_skip(
        decoder_3, encoder_1, n, 4, 2, "decoder_2", 0.0, train)

    # decoder_1:
    with tf.variable_scope("decoder_1"):

        decoder_1 = tf.layers.conv2d_transpose(decoder_2, output_channels, kernel_size=4, strides=(
            2, 2), padding="same", use_bias=True, kernel_initializer=xinit)

        decoder_1 = tf.nn.tanh(decoder_1)

    return decoder_1


def pix2pix_layer_bn(inputs, n, stride, name, train, xinit):

    with tf.variable_scope(name):

        layer = tf.layers.conv2d(
            inputs, n, 4, stride, padding='valid', use_bias=False,
            kernel_initializer=xinit)

        layer = tf.layers.batch_normalization(layer,
                                              axis=3,
                                              momentum=0.1,
                                              epsilon=1e-5,
                                              center=True,
                                              scale=False,
                                              training=train)
        layer = tf.nn.leaky_relu(layer)

        return layer


def pix2pix_disc(gen_inputs, gen_outputs, n, train):

    xinit = tf.contrib.layers.xavier_initializer()

    inputs = tf.concat([gen_inputs, gen_outputs], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):

        layer_1 = tf.layers.conv2d(
            inputs, n, 4, 2, padding='valid', use_bias=True,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        layer_1 = tf.nn.leaky_relu(layer_1)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    layer_2 = pix2pix_layer_bn(layer_1, n*2, 2, "layer_2", train, xinit)
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    layer_3 = pix2pix_layer_bn(layer_2, n*4, 1, "layer_3", train, xinit)
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    layer_4 = pix2pix_layer_bn(layer_3, n*8, 1, "layer_4", train, xinit)
    # layer_5: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    layer_5 = pix2pix_layer_bn(layer_4, n*8, 1, "layer_5", train, xinit)

    # layer_6
    with tf.variable_scope("layer_6"):

        layer_6 = tf.layers.conv2d(
            layer_5, 1, 4, 2, padding='valid', use_bias=True,
            kernel_initializer=xinit)

        layer_6 = tf.nn.sigmoid(layer_6)

    return layer_6


def pix2pix_optimizer(imgs, targets, learning_rate, alpha_data, alpha_disc, global_step, dropout, train, n=64):

    output_channels = int(targets.get_shape()[-1])
    optimizers = []
    depends = []

    with tf.variable_scope("generator"):
        outputs = pix2pix_gen(imgs, output_channels, n, dropout, train)

    gen_loss_data = tf.reduce_mean(tf.sqrt(EPS + tf.square(targets - outputs)))
    gen_loss = alpha_data * gen_loss_data

    disc = alpha_disc > 0.0

    if disc:

        with tf.variable_scope("discriminator"):
            disc_targets = pix2pix_disc(imgs, targets, n, train)

        with tf.variable_scope("discriminator", reuse=True):
            disc_outputs = pix2pix_disc(imgs, outputs, n, train)

        disc_loss_targets = tf.reduce_mean(-tf.log(disc_targets + EPS))
        disc_loss_outputs = tf.reduce_mean(-tf.log(1 - disc_outputs + EPS))
        disc_loss = disc_loss_targets + disc_loss_outputs

        gen_loss_disc = tf.reduce_mean(-tf.log(disc_outputs + EPS))
        gen_loss = gen_loss + gen_loss_disc * alpha_disc

        with tf.name_scope("discriminator_train"):
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                disc_tvars = [var for var in tf.trainable_variables(
                ) if var.name.startswith("discriminator")]
                disc_optim = tf.train.AdamOptimizer(learning_rate)
                disc_grads_and_vars = disc_optim.compute_gradients(
                    disc_loss, var_list=disc_tvars)
                disc_train = disc_optim.apply_gradients(
                    disc_grads_and_vars, global_step=global_step)

        optimizers.append(disc_train)
        depends.append(disc_train)

    with tf.name_scope("generator_train"):
        depends = depends if len(depends) > 0 else tf.get_collection(
            tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(depends):
            gen_tvars = [var for var in tf.trainable_variables(
            ) if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(learning_rate)
            gen_grads_and_vars = gen_optim.compute_gradients(
                gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(
                gen_grads_and_vars, global_step=global_step)
    optimizers.append(gen_train)

    trSum = []
    trSum.append(tf.summary.scalar(
        "generator_loss", gen_loss, family="lossGen"))
    trSum.append(tf.summary.scalar("generator_loss_data",
                                   gen_loss_data, family="lossGen"))
    tsSum = [trSum[0], trSum[1]]
    for var in gen_tvars:
        trSum.append(tf.summary.histogram(var.name, var, family="varGen"))
    for grad, var in gen_grads_and_vars:
        trSum.append(tf.summary.histogram(
            var.name + '_gradient', grad, family="gradGen"))

    if disc:

        trSum.append(tf.summary.scalar(
            "discriminator_loss", disc_loss, family="lossDisc"))
        tsSum.append(trSum[-1])
        trSum.append(tf.summary.scalar(
            "discriminator_loss_gt", disc_loss_targets, family="lossDisc"))
        tsSum.append(trSum[-1])
        trSum.append(tf.summary.scalar(
            "discriminator_loss_gen", disc_loss_outputs, family="lossDisc"))
        tsSum.append(trSum[-1])
        trSum.append(tf.summary.scalar("generator_loss_dis",
                                       gen_loss_disc, family="lossGen"))
        tsSum.append(trSum[-1])

        for var in disc_tvars:
            trSum.append(tf.summary.histogram(var.name, var, family="varDisc"))
        for grad, var in disc_grads_and_vars:
            trSum.append(tf.summary.histogram(
                var.name + '_gradient', grad, family="gradDisc"))

    targetsSamples = tf.concat([[targets[it, :, :, :]]
                                for it in range(16)], axis=2)
    outputSamples = tf.concat([[outputs[it, :, :, :]]
                               for it in range(16)], axis=2)
    imgSamples = tf.concat([targetsSamples, outputSamples], axis=1)
    tsSum.append(tf.summary.image("samples", imgSamples))

    trSum = tf.summary.merge(trSum, "Train")
    tsSum = tf.summary.merge(tsSum, "Test")

    return [optimizers, gen_loss, trSum, tsSum]

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
