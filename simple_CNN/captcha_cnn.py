from __future__ import division
# all tensorflow api is accessible through this
import tensorflow as tf
# to visualize the resutls
import matplotlib.pyplot as plt
import os
import numpy as np
import time
from readers import image_reader

train_X, train_Y = image_reader.load_training_dataset()
test_X, test_Y = image_reader.load_testing_dataset()

X_input = tf.placeholder(tf.float32, [None, 216 * 128])
X = tf.reshape(X_input, shape=[-1, 216, 128, 1])
Y_ = tf.placeholder(tf.float32, [None, 5 * 36])

learning_rate = 0.1

training_iters = 100  # 128*5000
display_step = 100


def create_fully_connected_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def create_conv_weight(patch_height, patch_width, input_channel, output_channel):
    initial = tf.truncated_normal(shape=[patch_height, patch_width, input_channel, output_channel], stddev=0.1)
    return tf.Variable(initial)


def create_bias(shape):
    initial = 0.1 * tf.random_normal(shape=shape)
    return tf.Variable(initial)


def create_strides(batch_step, height_step, width_step, channel_step):
    return [batch_step, height_step, width_step, channel_step]


def create_conv_layer(input, W, strides, padding='SAME'):
    return tf.nn.conv2d(input, W, strides, padding)


def apply_max_pool(x, ksize, strides, padding='SAME'):
    return tf.nn.max_pool(x, ksize, strides, padding)


# create first conv layer, with 3 input channel of orig. image, 4 output channels, stride of 1*1 and padding =SAME

keep_prob = tf.placeholder(tf.float32)

W1 = create_conv_weight(3, 3, 1, 32)
B1 = create_bias([32])
strides1 = create_strides(1, 1, 1, 1)
Y1 = tf.nn.relu(create_conv_layer(X, W1, strides1, padding="SAME") + B1)
Y1 = apply_max_pool(Y1, [1, 2, 2, 1], [1, 2, 2, 1])
Y1 = tf.nn.dropout(Y1, keep_prob=keep_prob)

W2 = create_conv_weight(3, 3, 32, 64)
B2 = create_bias([64])
strides2 = create_strides(1, 1, 1, 1)
Y2 = tf.nn.relu(create_conv_layer(Y1, W2, strides2, padding="SAME") + B2)
Y2 = apply_max_pool(Y2, [1, 2, 2, 1], [1, 2, 2, 1])
Y2 = tf.nn.dropout(Y2, keep_prob=keep_prob)

W3 = create_conv_weight(3, 3, 64, 64)
B3 = create_bias([64])
strides3 = create_strides(1, 1, 1, 1)
Y3 = tf.nn.relu(create_conv_layer(Y2, W3, strides3, padding="SAME") + B3)
Y3 = apply_max_pool(Y3, [1, 2, 2, 1], [1, 2, 2, 1])
Y3 = tf.nn.dropout(Y3, keep_prob=keep_prob)

# keep_prob = tf.placeholder(tf.float32)

Y3 = tf.reshape(Y3, [-1, 27 * 16 * 64])

W4 = create_fully_connected_weight([27 * 16 * 64, 1024])
B4 = create_bias([1024])
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y4 = tf.nn.dropout(Y4, keep_prob=keep_prob)

W5 = create_fully_connected_weight([1024, 5 * 36])
B5 = create_bias([5 * 36])
Ylogits = tf.matmul(Y4, W5) + B5

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(Ylogits, Y_)
loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# prediction
pred = tf.reshape(Ylogits, [-1, 5, 36])
Ytrue = tf.reshape(Y_, [-1, 5, 36])
correct_prediction = tf.equal(tf.argmax(pred, 2), tf.argmax(Ytrue, 2))

# Define the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_classes = 36
batch_size = 100
n_epochs = 20
n_batches_train = int(train_Y.shape[0] // batch_size)
print "number of batches: %d" % (n_batches_train)


def all_batches_run_train(n_batches, data=None, labels=None):
    sum_all_batches_loss = 0
    sum_all_batches_acc = 0
    sum_n_samples = 0

    for b in xrange(n_batches):

        offset = b * batch_size
        batch_data = data[offset: offset + batch_size, :]
        n_samples = batch_data.shape[0]

        batch_labels = labels[offset: offset + batch_size]
        # batch_labels = (np.arange(n_classes) == batch_labels[:, None]).astype(np.float32)
        # print np.shape(batch_data)
        # print np.shape(batch_labels)
        feed_dict = {X_input: batch_data, Y_: batch_labels, keep_prob: 0.75}
        _, loss_value, a = sess.run([train_step, loss, accuracy], feed_dict=feed_dict)
        sum_all_batches_loss += loss_value * n_samples
        sum_all_batches_acc += a * n_samples
        sum_n_samples += n_samples
        if (n_samples != batch_size):
            print('n_samples =%d' % n_samples)
    print "sum of samples trained %d" % (sum_n_samples)
    return (sum_all_batches_loss / sum_n_samples, sum_all_batches_acc / sum_n_samples)


def run_test(data=None, labels=None):
    assert (data.shape[0] == labels.shape[0])
    batch_size_test = 10000
    # labels = (np.arange(n_classes) == labels[:, None]).astype(np.float32)
    feed_dict = {X_input: data, Y_: labels, keep_prob: 1}
    test_a = sess.run([accuracy], feed_dict=feed_dict)
    return test_a


i = 1

train_ac = []
train_loss = []
test_ac = []
for e in xrange(n_epochs):
    start_time = time.time()
    n_data = train_X.shape[0]
    # print n_data
    perm = np.random.permutation(n_data)
    train_X = train_X[perm, :]
    train_Y = train_Y[perm]
    mean_loss_per_sample_train, accuracy_per_sample_train = all_batches_run_train(n_batches_train, data=train_X,
                                                                                  labels=train_Y)
    test_a = run_test(data=test_X, labels=test_Y)
    print "loss after epoch %d = %f: " % (i, mean_loss_per_sample_train)
    print "train accuracy after epoch %d = %f: " % (i, accuracy_per_sample_train)
    print "test accuracy after epoch %d = %f: " % (i, test_a[0])

    i = i + 1
    train_ac.append(accuracy_per_sample_train)
    train_loss.append(mean_loss_per_sample_train)
    test_ac.append(test_a[0])

print('done training')

plt.title("Training Accuracy over epochs")
plt.plot(train_ac, label="Training Accuracy")
plt.plot(test_ac, label="Test Accuracy")
plt.xlabel("epoch")
plt.legend(loc=4)
plt.grid(True)
plt.show()

plt.title("Training loss over epochs")
plt.plot(train_loss, label="Training Loss")
plt.xlabel("epoch")
plt.grid(True)
plt.show()

test_a = run_test(data=test_X, labels=test_Y)
print('done testing')
print("Testing Accuracy " + str(test_a[0]))
