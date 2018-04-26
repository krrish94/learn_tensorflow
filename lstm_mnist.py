# Tutorial from: https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/

import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data

# Read in MNIST data with labels as one-hot encodings
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Declare constants

# LSTM is unrolled through 28 time steps
time_steps = 28
# Number of hidden LSTM units
num_units = 128
# Each input row to the LSTM has 28 pixels
n_input = 28
# Learning rate
learning_rate = 0.001
# Number of classes
n_classes=10
# Batch size
batch_size=128

# Define placeholders

# Outputs
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

# Inputs
x = tf.placeholder("float", [None, time_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Reshape x from shape [batch_size, time_steps, n_input] to
# 'time_steps' number of [batch_size, n_input] tensors
input = tf.unstack(x, time_steps, axis = 1)

# Define network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias = 1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype = tf.float32)
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

# Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
# Optimization
opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# Model evaluation
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Init variables
init = tf.global_variables_initializer()

# Run session
with tf.Session() as sess:
	sess.run(init)
	iter = 1
	while iter < 800:
		batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
		batch_x = batch_x.reshape((batch_size, time_steps, n_input))
		sess.run(opt, feed_dict = {x: batch_x, y: batch_y})

		if iter % 10 == 0:
			acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
			los = sess.run(loss, feed_dict = {x:batch_x, y: batch_y})
			print('Iter: ', iter, 'Acc: ', acc, 'Loss: ', los)
		iter += 1

	# Compute test accuracy
	test_data = mnist.test.images[:128].reshape((-1, time_steps, n_input))
	test_label = mnist.test.labels[:128]
	print('Test Acc: ', sess.run(accuracy, feed_dict = {x: test_data, y: test_label}))
