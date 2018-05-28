""" common for tflowModel
"""

import sys
import tensorflow as tf

#-----------------------------------------------------------------------------------------------------
# Set common custom includes
#
sys.path.append('/mnt/p4/favila/moennen/local/lib/python2.7/site-packages')
sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/sceneIllEst/sampleBuffDataset/'))

#-----------------------------------------------------------------------------------------------------
# Set tensorflow logging
#
tf.logging.set_verbosity(tf.logging.INFO)


#-----------------------------------------------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------------------------------------------
class LearningParams:

	def __init__(selef):
		
		self.numSteps = 35000
		self.logStep = 150
		self.logShowFirstTraining = False

		self.batchSize = 64
		self.imgSzTr = [64,64]
		self.imgSzTs = [256,256]

		self.baseLearningRate = 0.00005
		

#-----------------------------------------------------------------------------------------------------
# OPERATORS
#-----------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------
# IMG2IMG MODELS
#-----------------------------------------------------------------------------------------------------


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

