""" EnvMapSh model #0000

---> input size : 192x108x3
---> output : spherical harmonics coefficient up to the 4th order
---> convolutionnal architecture :
---> fully connected output layer

"""
# from __future__ import division, print_function, absolute_import
import argparse
import os
import sys
import tensorflow as tf
import itertools
from matplotlib import pyplot as plt
from scipy.misc import toimage
# sys.path.append(os.path.abspath('/home/moennen/sceneIllEst/sampleEnvMapShDataset'))
sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleEnvMapShDataset'))
from sampleEnvMapShDataset import *
from tensorflow.contrib.data import Dataset, Iterator

# Parameters
numSteps = 150000
logStep = 100
logTrSteps = 1
logTsSteps = 1
batchSz = 128
shOrder = 4
imgSz = [108, 192]

tf.logging.set_verbosity(tf.logging.INFO)


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


def printVarTF(sess):
    tvars = tf.trainable_variables()
    for var in tvars:
        print var.name
        print var.eval(sess)


def conv_layer(x, filter_size, step, scope, padding='VALID'):
    # tf.random_normal(filter_size))
    initializer = tf.contrib.layers.xavier_initializer()
    layer_w = tf.Variable(initializer(filter_size))
    layer_b = tf.Variable(initializer([filter_size[3]]))
    layer = tf.nn.conv2d(x, layer_w, strides=[
                         1, step, step, 1], padding=padding)
    layer = tf.nn.bias_add(layer, layer_b)
    layer = tf.nn.relu(layer, name=scope)
    return layer


def envMapShModel0000(imgs, outputSz, dropout):

    with tf.variable_scope('EnvMapShModel00001'):

        # ----> 192x108x3
        layer0 = imgs
        # ----> 90x48x32
        with tf.name_scope('layer1_1') as scope:
            layer1 = conv_layer(layer0, [7, 7, 3, 32], 2, scope)
        # ----> 41x20x64
        with tf.name_scope('layer2_1') as scope:
            layer2 = conv_layer(layer1, [5, 5, 32, 64], 2, scope)
        with tf.name_scope('layer2_2') as scope:
            layer2 = conv_layer(layer2, [5, 5, 64, 96], 1, scope, 'SAME')
        # with tf.name_scope('layer2_2') as scope:
        #    layer2 = conv_layer(layer2, [5, 5, 64, 96], 1, scope)
        # ----> 18x8x128
        with tf.name_scope('layer3_1') as scope:
            layer3 = conv_layer(layer2, [3, 3, 96, 128], 2, scope)
        with tf.name_scope('layer3_2') as scope:
            layer3 = conv_layer(layer3, [3, 3, 128, 256], 1, scope, 'SAME')
        # ----> 18x8x128
        with tf.name_scope('layer4_1') as scope:
            layer4 = conv_layer(layer3, [3, 3, 256, 256], 1, scope)
        # ----> 7x2x256
        with tf.name_scope('layer5_1') as scope:
            layer5 = conv_layer(layer4, [3, 3, 256, 512], 2, scope)
        # ----> 1x1x512
        with tf.name_scope('layer6_1') as scope:
            layer6 = conv_layer(layer5, [3, 3, 512, 1024], 2, scope)

        #
        layer6f = tf.contrib.layers.flatten(layer6)
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.name_scope('layer7_1') as scope:
            layer7 = tf.layers.dense(layer6f, 2048, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer7_2') as scope:
            layer7 = tf.layers.dense(layer7d, 1024, activation=tf.nn.relu, kernel_initializer=initializer,
                                     bias_initializer=initializer, name=scope)
            layer7d = tf.layers.dropout(layer7, rate=dropout)
        with tf.name_scope('layer8_1') as scope:
            outputLayer = tf.layers.dense(layer7d, outputSz, kernel_initializer=initializer,
                                          bias_initializer=initializer, name=scope)

        return outputLayer


class EnvMapShDatasetTF(object):

    def __init__(self, dbPath):
        self.__envMapDb = EnvMapShDataset(dbPath, shOrder)
        self.__dims = [batchSz, imgSz[0], imgSz[1]]
        self.data = Dataset.from_generator(
            self.genEnvMapSh, (tf.float32, tf.float32))

    def genEnvMapSh(self):
        for i in itertools.count(1):
            imgs, coeffs, cparams = self.__envMapDb.sampleData(self.__dims)
            yield (coeffs, imgs)

    def getNbShCoeffs(self):
        return self.__envMapDb.nbShCoeffs*3


def trainEnvMapShModel(modelPath, trainPath, testPath):

    tbLogsPath = modelPath + "/tbLogs"
    modelFilename = modelPath + "/tfData"

    trDs = EnvMapShDatasetTF(trainPath)
    tsDs = EnvMapShDatasetTF(testPath)

    nbShCoeffs = trDs.getNbShCoeffs()
    inputShape = [batchSz, imgSz[0], imgSz[1], 3]
    outputShape = [batchSz, nbShCoeffs]

    dsIt = Iterator.from_structure(
        trDs.data.output_types, trDs.data.output_shapes)
    dsView = dsIt.get_next()

    trInit = dsIt.make_initializer(trDs.data)
    tsInit = dsIt.make_initializer(tsDs.data)

    # Input
    inputView = tf.placeholder(tf.float32, shape=inputShape, name="input_view")
    outputSh = tf.placeholder(tf.float32, shape=outputShape, name="output_sh")
    dropoutProb = tf.placeholder(tf.float32)  # dropout (keep probability)
    training = tf.placeholder(tf.bool)

    # Test
    # outputSh2 = tf.placeholder(
    #    tf.float32, shape=outputShape, name="output_sh2")
    # outStd = tf.sqrt(tf.reduce_mean(
    #    tf.square(tf.subtract(outputSh2, outputSh))))

    # Graph
    computedSh = envMapShModel0000(inputView, nbShCoeffs, dropoutProb)

    # Optimizer
    costAll = tf.reduce_mean(
        tf.square(tf.subtract(computedSh, outputSh)), 0)
    accuracyAll = tf.sqrt(costAll)
    cost = tf.reduce_mean(costAll)
    accuracy = tf.reduce_mean(accuracyAll)
    globalStep = tf.Variable(0, trainable=False)
    learningRate = tf.train.polynomial_decay(0.001, globalStep, numSteps, 0.0,
                                             power=0.5)
    optEngine = tf.train.AdamOptimizer(learning_rate=learningRate)
    optimizer = optEngine.minimize(cost, global_step=globalStep)

    # Persistency
    persistency = tf.train.Saver(pad_step_number=True, keep_checkpoint_every_n_hours=3,
                                 filename=modelFilename)

    # Initializing the board
    tf.summary.scalar("loss", cost)
    tf.summary.scalar("learning_rate", learningRate)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    grads = optEngine.compute_gradients(cost, tf.trainable_variables())
    for grad, var in grads:
        tf.summary.histogram(var.name + '/gradient', grad)
    merged_summary_op = tf.summary.merge_all()
    tf.summary.scalar("test_loss", cost)
    test_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(tbLogsPath,
                                           graph=tf.get_default_graph())

    # Metrics : Mean
    meanOutputSh, meanOutputShUpdateOp = tf.contrib.metrics.streaming_mean_tensor(
        outputSh)

    currentMeanOutputSh = tf.placeholder(
        tf.float32, shape=outputShape, name="output_sh")
    outStdAll = tf.reshape(tf.reduce_mean(
        currentMeanOutputSh, 0), [1, nbShCoeffs])
    outStdAll = tf.tile(outStdAll, [batchSz, 1])
    outStdAll = tf.sqrt(tf.reduce_mean(
        tf.square(tf.subtract(outStdAll, outputSh)), 0))
    outStd = tf.reduce_mean(outStdAll)

    # Params Initializer
    varInit = tf.global_variables_initializer()

    with tf.Session() as sess:

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
        for step in range(numSteps):

            # initialize the iterator on the training data

            # Get the next training batch
            coeffs, imgs = sess.run(dsView)

            # Run optimization op (backprop)
            _, currMean = sess.run([optimizer, meanOutputShUpdateOp], feed_dict={dropoutProb: 0.2,
                                                                                 inputView: imgs,
                                                                                 outputSh: coeffs,
                                                                                 training: True})
            # Log
            if step % logStep == 0:

                # summary
                summary = sess.run(merged_summary_op, feed_dict={dropoutProb: 0.0,
                                                                 inputView: imgs,
                                                                 outputSh: coeffs})
                summary_writer.add_summary(summary, globalStep.eval(sess))

                # Sample train accuracy
                sess.run(trInit)
                trAccuracy = 0
                trStd = 0
                for logTrStep in range(logTrSteps):
                    coeffs, imgs = sess.run(dsView)
                    trAcc = sess.run(accuracy, feed_dict={dropoutProb: 0.0,
                                                          inputView: imgs,
                                                          outputSh:  coeffs})
                    trAccuracy += trAcc
                    #trAccuracyAll += trAccAll
                    # coeffs2, imgs2 = sess.run(dsView)
                    # trStd += sess.run(outStd, feed_dict={outputSh:  coeffs,
                    #                                     outputSh2: coeffs2})

                # print "------------------------------------------"
                # print imgs
                # print "------------------------------------------"
                # print coeffs
                # print "------------------------------------------"
                # print computedSh.eval(feed_dict={dropoutProb: 0.0,
                #                                 inputView: imgs,
                #                                 outputSh:  coeffs})
                # print "------------------------------------------"

            # Sample test accuracy
                sess.run(tsInit)
                tsAccuracy = 0
                stdAccuracy = 0
                for logTsStep in range(logTsSteps):
                    coeffs, imgs = sess.run(dsView)
                    tsAcc, stdAcc = sess.run([accuracy, outStd], feed_dict={dropoutProb: 0.0,
                                                                            inputView: imgs,
                                                                            outputSh:  coeffs,
                                                                            currentMeanOutputSh: currMean})
                    tsAccuracy += tsAcc
                    stdAccuracy += stdAcc

                # summary
                summary = sess.run(test_summary_op, feed_dict={dropoutProb: 0.0,
                                                               inputView: imgs,
                                                               outputSh: coeffs})
                summary_writer.add_summary(summary, globalStep.eval(sess))

                print("{:08d}".format(globalStep.eval(sess)) +
                      " | lr = " + "{:.8f}".format(learningRate.eval()) +
                      " | trAcc  = " + "{:.5f}".format(trAccuracy/logTrSteps) +
                      " | tsAcc  = " + "{:.5f}".format(tsAccuracy/logTsSteps) +
                      " | stdAcc  = " + "{:.5f}".format(stdAccuracy/logTsSteps))
                #print(" trAcc :")
                # print trAccAll

                # step
                persistency.save(sess, modelFilename, global_step=globalStep)

                # reset the training iterator
                sess.run(trInit)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("modelPath", help="path to the trainedModel")
    parser.add_argument(
        "trainDbPath", help="path to the Training EnvMapDataset levelDb path")
    parser.add_argument(
        "testDbPath", help="path to the Testing EnvMapDataset levelDb path")
    args = parser.parse_args()

    trainEnvMapShModel(args.modelPath, args.trainDbPath, args.testDbPath)
