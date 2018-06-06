#!/usr/bin/python
""" EnvMapSh Model

"""
import argparse
import os
import sys
import time
import tensorflow as tf
import itertools
import numpy as np
import random

from commonModel import *

# sys.path.append('/mnt/p4/favila/moennen/local/lib/python2.7/site-packages')
import cv2 as cv

sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleEnvMapShDataset'))
from sampleEnvMapShDataset import *

#-----------------------------------------------------------------------------------------------------
# GLOBAL PARAMS
#-----------------------------------------------------------------------------------------------------

shOrder = 4

shCoeffsMean8 = [0.902903, 0.875094, 0.896489, -0.162082, -0.252774, -0.406744, 0.00337894, 0.00172557, 0.00311243, 0.000909537, 0.000695117, -4.90672e-05, -0.00128019, -0.00102835, -0.000301714, 0.002949, 0.00374928, 0.00226501, -0.00952863, -0.0208408, -0.0508576, -0.000182751, -0.000320935, -0.000347464, -0.0185872, -0.0366991, -0.0886874, -0.0410145, -0.0560474, -0.0794013, -0.000163254, -4.07888e-05, -2.79235e-05, -0.0313884, -0.0427534, -0.0605482, 0.00334514, 0.00201432, 0.000809579, -7.04794e-05, -0.000115991, -
                 0.000284364, 0.00531681, 0.00525261, 0.00461419, -0.00017454, -0.000219352, -0.000313192, -0.000728616, -0.000451431, 5.66945e-05, -0.000715351, -0.000834947, -0.00157328, -0.000165865, -2.45795e-05, 0.000226656, -0.000148126, 0.000878715, 0.00207265, -0.00715132, -0.0107122, -0.0205379, -6.46675e-05, -8.40424e-05, -0.000117739, -0.0194614, -0.0248351, -0.0394562, 0.000269648, 0.000203064, 1.16736e-05, -0.0216183, -0.0284543, -0.0478178, -0.0137995, -0.0180241, -0.0228328, -6.30815e-06, -5.03176e-05, -0.000133641]

shCoeffsStd8 = [0.396919, 0.3877, 0.446635, 0.329931, 0.31224, 0.353571, 0.142876, 0.142675, 0.152221, 0.131147, 0.129519, 0.135236, 0.135648, 0.129224, 0.124978, 0.140774, 0.135626, 0.13652, 0.137738, 0.133197, 0.138377, 0.0921217, 0.0883874, 0.0877252, 0.186394, 0.181423, 0.192694, 0.12275, 0.121771, 0.141144, 0.0720004, 0.0689185, 0.0716978, 0.106684, 0.103984, 0.117957, 0.0773083, 0.0739887, 0.0748107, 0.0610265, 0.0578222,
                0.0592729, 0.0835063, 0.0797521, 0.079296, 0.0892349, 0.08521, 0.0832717, 0.0643495, 0.060952, 0.0640418, 0.0604189, 0.058079, 0.0623505, 0.0511543, 0.04894, 0.0520293, 0.0628388, 0.0605339, 0.0650797, 0.0653313, 0.063549, 0.0674159, 0.0475521, 0.0454968, 0.0472838, 0.0756618, 0.073626, 0.0784726, 0.051918, 0.0494255, 0.0514637, 0.0911271, 0.0892432, 0.0961533, 0.0650854, 0.0648895, 0.0739945, 0.0417161, 0.0398853, 0.0426519]


#-----------------------------------------------------------------------------------------------------
# DATASET
#-----------------------------------------------------------------------------------------------------


class DatasetTF(object):

    def __init__(self, dbPath, imgRootDir, batchSz, imgSz, seed):
        self.__envMapDb = EnvMapShDataset(
            dbPath, imgRootDir, shOrder, seed, True)
        self.__dims = [batchSz, imgSz[0], imgSz[1]]
        self.data = tf.data.Dataset.from_generator(
            self.genEnvMapSh, (tf.float32, tf.float32))

    def genEnvMapSh(self):
        for i in itertools.count(1):
            imgs, coeffs, cparams = self.__envMapDb.sampleData(self.__dims)

            yield (coeffs, imgs)

    def getNbShCoeffs(self):
        return self.__envMapDb.nbShCoeffs*3

#-----------------------------------------------------------------------------------------------------
# Test


def testDataset(imgRootDir, dsPath):

    rseed = int(time.time())
    tf.set_random_seed(rseed)

    batchSz = 8

    ds = DatasetTF(dsPath, imgRootDir, batchSz, [128, 128], rseed)
    dsIt = tf.data.Iterator.from_structure(
        ds.data.output_types, ds.data.output_shapes)
    dsView = dsIt.get_next()

    dsInit = dsIt.make_initializer(ds.data)

    with tf.Session() as sess:

        sess.run(dsInit)

        for step in range(100):

            # get the data
            coeffs, imgs = sess.run(dsView)

            idx = random.randint(0, batchSz-1)

            # generate env map from coeffs
            envMap = EnvMapShDataset. generateEnvMap(
                shOrder, coeffs[idx], [128, 256])

            cv.imshow('img', cv.cvtColor(imgs[idx], cv.COLOR_RGB2BGR))
            cv.imshow('envMap', cv.cvtColor(envMap[0], cv.COLOR_RGB2BGR))

            cv.waitKey(0)

#-----------------------------------------------------------------------------------------------------
# MODEL
#-----------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------
# Basic model : used for testing purposes


def base_model(imgs, output_channels, n, dropout, train):

    xinit = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope('layer_1'):
        layer_1 = tf.layers.conv2d(imgs, n, [7, 7], 2, padding='same',
                                   bias_initializer=xinit, kernel_initializer=xinit, activation=tf.nn.relu)

    with tf.variable_scope('layer_2'):
        layer_2 = tf.layers.conv2d(layer_1, n*2, [5, 5], 2, padding='same',
                                   bias_initializer=xinit, kernel_initializer=xinit, activation=tf.nn.relu)

    with tf.variable_scope('layer_3'):
        layer_3 = tf.layers.conv2d(layer_2, n*4, [3, 3], 2, padding='same',
                                   bias_initializer=xinit, kernel_initializer=xinit, activation=tf.nn.relu)

    layer_4 = tf.layers.flatten(layer_3)
    with tf.variable_scope("layer_4"):
        layer_4 = tf.layers.dense(layer_4, n*8, activation=tf.nn.relu, kernel_initializer=xinit,
                                  bias_initializer=xinit)
        layer_4 = tf.nn.dropout(layer_4, keep_prob=1.0 - dropout)

    with tf.variable_scope("layer_5"):
        layer_5 = tf.layers.dense(layer_4, output_channels, activation=None, kernel_initializer=xinit,
                                  bias_initializer=xinit)

    return layer_5


def base_optimizer(imgs, targets, learning_rate, global_step, dropout, n=32):

    output_channels = int(targets.get_shape()[-1])

    with tf.variable_scope("base"):
        outputs = base_model(imgs, output_channels, n, dropout, True)

    loss_l2 = tf.reduce_mean(tf.square(tf.subtract(targets, outputs)))

    base_optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
    base_train = base_optim.minimize(loss_l2, global_step=global_step)

    tf.summary.scalar("loss_l2", loss_l2)

    return [base_train, loss_l2, outputs]

#-----------------------------------------------------------------------------------------------------
# Model


def pix2vec_model(imgs, output_channels, n, dropout, train):

    xinit = tf.contrib.layers.xavier_initializer()

    # encoder_1: [batch, 256, 256, in_channels] => [batch,  , n]
    with tf.variable_scope("encoder_1"):

        encoder_1 = tf.layers.conv2d(
            imgs, n, 4, 2, padding='same', use_bias=True,
            bias_initializer=xinit, kernel_initializer=xinit)

        encoder_1 = tf.nn.leaky_relu(encoder_1)

    # encoder_2: [batch, 128, 128, n] => [batch, 64, 64, n * 2]
    encoder_2 = pix2pix_encoder_bn(
        encoder_1, n*2, "encoder_2", train)
    # encoder_3: [batch, 64, 64, n * 2] => [batch, 32, 32, n * 4]
    encoder_3 = pix2pix_encoder_bn(
        encoder_2, n*4, "encoder_3", train)
    # encoder_4: [batch, 32, 32, n * 4] => [batch, 16, 16, n * 8]
    encoder_4 = pix2pix_encoder_bn(
        encoder_3, n*8, "encoder_4", train)
    # encoder_5: [batch, 16, 16, n * 8] => [batch, 8, 8, n * 8]
    encoder_5 = pix2pix_encoder_bn(
        encoder_4, n*8, "encoder_5", train)
    # encoder_6: [batch, 8, 8, n * 8] => [batch, 4, 4, n * 8]
    encoder_6 = pix2pix_encoder_bn(
        encoder_5, n*8, "encoder_6", train)
    # encoder_7: [batch, 4, 4, n * 8] => [batch, 2, 2, n * 8]
    encoder_7 = pix2pix_encoder_bn(
        encoder_6, n*8, "encoder_7", train)
    # encoder_8: [batch, 2, 2, n * 8] = > [batch, 1, 1, n * 8]
    encoder_8 = pix2pix_encoder_bn(
        encoder_7, n*8, "encoder_8", train)

    # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
    decoder_8 = pix2pix_decoder_skip(
        encoder_8, encoder_7, n*8, "decoder_8", dropout, train)
    # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
    decoder_7 = pix2pix_decoder_skip(
        decoder_8, encoder_6, n*8, "decoder_7", dropout, train)
    # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
    decoder_6 = pix2pix_decoder_skip(
        decoder_7, encoder_5, n*8, "decoder_6", dropout, train)

    # dense
    vectorizer_0 = tf.layers.flatten(decoder_6)
    with tf.variable_scope("vectorizer_1"):
        vectorizer_1 = tf.layers.dense(vectorizer_0, n*8*2, activation=tf.nn.relu, kernel_initializer=xinit,
                                       bias_initializer=xinit)
        vectorizer_1 = tf.nn.dropout(vectorizer_1, keep_prob=1.0 - dropout)
    with tf.variable_scope("vectorizer_2"):
        vectorizer_2 = tf.layers.dense(vectorizer_1, n*8*2, activation=tf.nn.relu, kernel_initializer=xinit,
                                       bias_initializer=xinit)
    with tf.variable_scope("vectorizer_3"):
        vectorizer_3 = tf.layers.dense(
            vectorizer_2, output_channels, kernel_initializer=xinit, bias_initializer=xinit)

    return vectorizer_3


def pix2vec_optimizer(imgs, targets, learning_rate, global_step, dropout, n=64, ):

    output_channels = int(targets.get_shape()[-1])

    with tf.variable_scope("pix2vec"):
        outputs = pix2vec_model(imgs, output_channels, n, dropout, True)

    loss = tf.reduce_mean(tf.square(targets - outputs))

    with tf.name_scope("pix2vec_train"):
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            p2v_tvars = [var for var in tf.trainable_variables(
            ) if var.name.startswith("pix2vec")]
            p2v_optim = tf.train.AdamOptimizer(learning_rate)
            p2v_grads_and_vars = p2v_optim.compute_gradients(
                loss, var_list=p2v_tvars)
            p2v_train = p2v_optim.apply_gradients(
                p2v_grads_and_vars, global_step=global_step)

    tf.summary.scalar("loss", loss)
    for var in p2v_tvars:
        tf.summary.histogram(var.name, var)
    for grad, var in p2v_grads_and_vars:
        tf.summary.histogram(var.name + '_gradient', grad)

    return [p2v_train, loss, outputs]

#-----------------------------------------------------------------------------------------------------
# TRAIN
#-----------------------------------------------------------------------------------------------------


def trainModel(modelPath, imgRootDir, trainPath, testPath):

    lp = LearningParams(modelPath, 20160704)
    lp.numSteps = 100000
    lp.imgSzTr = [256, 256]
    lp.batchSz = 32

    trDs = DatasetTF(trainPath, imgRootDir,
                     lp.batchSz, lp.imgSzTr, lp.rseed)
    tsDs = DatasetTF(testPath, imgRootDir,
                     lp.batchSz, lp.imgSzTr, lp.rseed)

    nbShCoeffs = trDs.getNbShCoeffs()

    dsIt = tf.data.Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input
    inputViews = tf.placeholder(
        tf.float32, shape=[lp.batchSz, lp.imgSzTr[0], lp.imgSzTr[1], 3], name="input_views")
    inputViews = preprocess(inputViews)

    # targets
    targetSh = tf.placeholder(
        tf.float32, shape=[lp.batchSz, nbShCoeffs], name="target_sh")
    shMean = tf.constant(shCoeffsMean8[0:nbShCoeffs], dtype=tf.float32, shape=[
        1, nbShCoeffs], name='shCoeffs_mean')
    shStd = tf.constant(shCoeffsStd8[0:nbShCoeffs], dtype=tf.float32, shape=[
        1, nbShCoeffs], name='shCoeffs_mean')
    targetSh = (targetSh-shMean) / shStd

    optimizer, loss, _ = pix2vec_optimizer(
        inputViews, targetSh, lp.learningRate, lp.globalStep, lp.dropoutProb, 32)

    # Persistency
    persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                 filename=lp.modelFilename)

    # Logger
    merged_summary_op = tf.summary.merge_all()

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:

        train_summary_writer = tf.summary.FileWriter(
            lp.tbLogsPath + "/Train", graph=sess.graph)
        test_summary_writer = tf.summary.FileWriter(lp.tbLogsPath + "/Test")

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
        for step in range(lp.globalStep.eval(sess), lp.numSteps):

            # initialize the iterator on the training data

            # Get the next training batch
            coeffs, imgs = sess.run(dsView)

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={lp.dropoutProb: 0.01,
                                           inputViews: imgs,
                                           targetSh: coeffs})
            # Log
            if step % lp.logStep == 0:

                # summary
                summary = sess.run(merged_summary_op, feed_dict={lp.dropoutProb: 0.0,
                                                                 inputViews: imgs,
                                                                 targetSh: coeffs})
                train_summary_writer.add_summary(
                    summary, lp.globalStep.eval(sess))

                # Sample train accuracy
                sess.run(trInit)
                coeffs, imgs = sess.run(dsView)
                trLoss = sess.run(loss, feed_dict={lp.dropoutProb: 0.0,
                                                   inputViews: imgs,
                                                   targetSh:  coeffs})
                # Sample test accuracy
                sess.run(tsInit)
                coeffs, imgs = sess.run(dsView)
                tsLoss = sess.run(loss, feed_dict={lp.dropoutProb: 0.0,
                                                   inputViews: imgs,
                                                   targetSh:  coeffs})
                # summary
                summary = sess.run(merged_summary_op, feed_dict={lp.dropoutProb: 0.0,
                                                                 inputViews: imgs,
                                                                 targetSh: coeffs})
                test_summary_writer.add_summary(
                    summary, lp.globalStep.eval(sess))

                print("{:08d}".format(lp.globalStep.eval(sess)) +
                      " | lr = " + "{:.8f}".format(lp.learningRate.eval()) +
                      " | trLoss  = " + "{:.5f}".format(trLoss) +
                      " | tsLoss  = " + "{:.5f}".format(tsLoss))

                # step
                persistency.save(sess, lp.modelFilename,
                                 global_step=lp.globalStep)

                # reset the training iterator
                sess.run(trInit)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("modelPath", help="path to the trainedModel")
    parser.add_argument(
        "imgRootDir", help="root directory to the images in the levelDb databases")
    parser.add_argument(
        "trainDbPath", help="path to the Training EnvMapDataset levelDb path")
    parser.add_argument(
        "testDbPath", help="path to the Testing EnvMapDataset levelDb path")
    args = parser.parse_args()

    # testDataset(args.imgRootDir, args.trainDbPath)

    trainModel(args.modelPath, args.imgRootDir, args.trainDbPath,
               args.testDbPath)
