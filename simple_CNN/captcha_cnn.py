from __future__ import division
# all tensorflow api is accessible through this
import tensorflow as tf
# to visualize the resutls
import matplotlib.pyplot as plt
import numpy as np
import time
from readers import image_reader
from readers import label_util

# train_X, train_Y = image_reader.load_training_dataset()
test_X, test_Y = image_reader.load_testing_dataset()

X_input = tf.placeholder(tf.float32, [None, 160 * 80])
X = tf.reshape(X_input, shape=[-1, 160, 80, 1])
Y_ = tf.placeholder(tf.float32, [None, 5 * image_reader.NUM_CLASSES])

learning_rate = 0.01

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

W1 = create_conv_weight(5, 5, 1, 32)
B1 = create_bias([32])
strides1 = create_strides(1, 1, 1, 1)
Y1 = tf.nn.relu(create_conv_layer(X, W1, strides1, padding="SAME") + B1)
Y1 = apply_max_pool(Y1, [1, 2, 2, 1], [1, 2, 2, 1])
Y1 = tf.nn.dropout(Y1, keep_prob=keep_prob)

W2 = create_conv_weight(5, 5, 32, 64)
B2 = create_bias([64])
strides2 = create_strides(1, 1, 1, 1)
Y2 = tf.nn.relu(create_conv_layer(Y1, W2, strides2, padding="SAME") + B2)
Y2 = apply_max_pool(Y2, [1, 2, 2, 1], [1, 2, 2, 1])
Y2 = tf.nn.dropout(Y2, keep_prob=keep_prob)

W3 = create_conv_weight(5, 5, 64, 128)
B3 = create_bias([128])
strides3 = create_strides(1, 1, 1, 1)
Y3 = tf.nn.relu(create_conv_layer(Y2, W3, strides3, padding="SAME") + B3)
Y3 = apply_max_pool(Y3, [1, 2, 2, 1], [1, 2, 2, 1])
Y3 = tf.nn.dropout(Y3, keep_prob=keep_prob)

W4 = create_conv_weight(5, 5, 128, 128)
B4 = create_bias([128])
strides4 = create_strides(1, 1, 1, 1)
Y4 = tf.nn.relu(create_conv_layer(Y3, W4, strides4, padding="SAME") + B4)
Y4 = apply_max_pool(Y4, [1, 2, 2, 1], [1, 2, 2, 1])
Y4 = tf.nn.dropout(Y4, keep_prob=keep_prob)

# keep_prob = tf.placeholder(tf.float32)

Y4 = tf.reshape(Y4, [-1, 10 * 5 * 128])

W5 = create_fully_connected_weight([10 * 5 * 128, 1024])
B5 = create_bias([1024])
Y5 = tf.nn.relu(tf.matmul(Y4, W5) + B5)
Y5 = tf.nn.dropout(Y5, keep_prob=keep_prob)

W6 = create_fully_connected_weight([1024, 5 * image_reader.NUM_CLASSES])
B6 = create_bias([5 * image_reader.NUM_CLASSES])
Ylogits = tf.matmul(Y5, W6) + B6

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(Ylogits, Y_)
loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# prediction
predictions = tf.reshape(Ylogits, [-1, 5, image_reader.NUM_CLASSES])
Ytrue = tf.reshape(Y_, [-1, 5, image_reader.NUM_CLASSES])
correct_prediction = tf.equal(tf.argmax(predictions, 2), tf.argmax(Ytrue, 2))

# Define the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_classes = image_reader.NUM_CLASSES
batch_size = 100
n_epochs = 20
n_batches_train = int(image_reader.training_dataset_length() // batch_size)
print "number of batches: %d" % (n_batches_train)


def all_batches_run_train(n_batches, data=None, labels=None):
    sum_all_batches_loss = 0
    sum_all_batches_acc = 0
    sum_n_samples = 0

    for b in xrange(n_batches):

        offset = b * batch_size
        batch_data, batch_labels = image_reader.load_dataset("../imgs/", offset, offset + batch_size)
        # batch_data = data[offset: offset + batch_size, :]
        n_samples = batch_data.shape[0]

        # batch_labels = labels[offset: offset + batch_size]
        # batch_labels = (np.arange(n_classes) == batch_labels[:, None]).astype(np.float32)
        # print np.shape(batch_data)
        # print np.shape(batch_labels)

        feed_dict = {X_input: batch_data, Y_: batch_labels, keep_prob: 0.50}
        _, loss_value, a = sess.run([train_step, loss, accuracy], feed_dict=feed_dict)
        sum_all_batches_loss += loss_value * n_samples
        sum_all_batches_acc += a * n_samples
        sum_n_samples += n_samples
        if (n_samples != batch_size):
            print('n_samples =%d' % n_samples)

    print "sum of samples trained %d" % (sum_n_samples)
    return (sum_all_batches_loss / sum_n_samples, sum_all_batches_acc / sum_n_samples)


def test_epochs(data=None, labels=None):
    # assert (data.shape[0] == labels.shape[0])
    # # labels = (np.arange(n_classes) == labels[:, None]).astype(np.float32)
    # feed_dict = {X_input: data, Y_: labels, keep_prob: 1}
    # test_a = sess.run([accuracy], feed_dict=feed_dict)
    return 1




def test_and_evaluate(data=None, labels=None):
    assert (data.shape[0] == labels.shape[0])
    feed_dict = {X_input: data, Y_: labels, keep_prob: 1}
    test_results = sess.run([predictions,correct_prediction,accuracy], feed_dict=feed_dict)
    test_preds = test_results[0]
    label_util.compare_predictions(test_results[0],labels)
    print "printing correct predictions"
    print  test_results[1]
    #print test_results[1].shape
    return test_results

i = 1

train_ac = []
train_loss = []
test_ac = []
for e in xrange(n_epochs):
    start_time = time.time()
    n_data = image_reader.training_dataset_length()
    # n_data = train_X.shape[0]
    # print n_data
    perm = np.random.permutation(n_data)
    # train_X = train_X[perm, :]
    # train_Y = train_Y[perm]
    mean_loss_per_sample_train, accuracy_per_sample_train = all_batches_run_train(n_batches_train)
    #test_a = test_epochs(data=test_X, labels=test_Y)
    print "loss after epoch %d = %f: " % (i, mean_loss_per_sample_train)
    print "train accuracy after epoch %d = %f: " % (i, accuracy_per_sample_train)
    #print "test accuracy after epoch %d = %f: " % (i, test_a[0])
    print "-----------------------------------\n"
    i = i + 1
    train_ac.append(accuracy_per_sample_train)
    train_loss.append(mean_loss_per_sample_train)
#    test_ac.append(test_a[0])



print('done training')

plt.title("Training Accuracy over epochs")
plt.plot(train_ac, label="Training Accuracy")
#plt.plot(test_ac, label="Test Accuracy")
plt.xlabel("epoch")
plt.legend(loc=4)
plt.grid(True)
plt.show()

plt.title("Training loss over epochs")
plt.plot(train_loss, label="Training Loss")
plt.xlabel("epoch")
plt.grid(True)
plt.show()

test_results = test_and_evaluate(data=test_X, labels=test_Y)

print('done testing')
print("Test Accuracy " + str(test_results[2]))

