# Tutorial from: https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


# Declare constants

# LSTM is unrolled through 10 time steps
seq_len = 10
# Number of hidden LSTM units
num_units = 8
# Size of each input
n_input = 1
# Learning rate
learning_rate = 0.1
# Number of classes
n_classes = 2
# Batch size
batch_size = 2


# Synthesize data
np.random.seed(1234)
num_tokens = 10
dataset_size = 2
data = np.zeros((dataset_size, seq_len))
label = np.zeros((dataset_size, seq_len, n_classes))
for i in range(dataset_size):
	# Generate a random permutation of all tokens. 
	# Throw in a random translation of all tokens.
	tmp = np.random.permutation(num_tokens) + np.random.randint(240)
	coin_filp = np.random.randint(0,2)
	if coin_filp == 0:
		# Add a repetition
		# Index of number to repeat
		tmpIdx_src = np.random.randint(len(tmp))
		# Where to repeat that number
		tmpIdx_dst = np.random.randint(len(tmp))
		while tmpIdx_dst == tmpIdx_src:
			tmpIdx_dst = np.random.randint(len(tmp))
		tmp[tmpIdx_dst] = tmp[tmpIdx_src]
		label[i,tmpIdx_src,1] = 1.0
		label[i,tmpIdx_dst,1] = 1.0
	data[i,:] = tmp
	print(data[i,:])
	print(label[i,:])



# Define placeholders

# Outputs
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

# Inputs
x = tf.placeholder("float", [batch_size, seq_len, n_input])
y = tf.placeholder("float", [batch_size, seq_len, n_classes])

# Reshape x from shape [batch_size, seq_len, n_input] to
# 'seq_len' number of [batch_size, n_input] tensors
input = tf.unstack(x, seq_len, axis = 1)

# Define network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias = 1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype = tf.float32)
# Reshape outputs to batch_size * seq_len x num_units
outputs_reshaped = tf.reshape(outputs, [batch_size * seq_len, num_units])
prediction = tf.matmul(outputs_reshaped, out_weights) + out_bias

# Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
# Optimization
opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# Model evaluation
# correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Init variables
init = tf.global_variables_initializer()

# Run session
with tf.Session() as sess:
	sess.run(init)
	iter = 1
	while iter < 2:
		# batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
		# batch_x = batch_x.reshape((batch_size, seq_len, n_input))
		batch_x = data[0:batch_size,:]
		batch_x = np.expand_dims(batch_x, -1)
		print(batch_x, batch_x.shape)
		batch_y = label[0:batch_size,:,:]
		print(batch_y, batch_y.shape)
		# batch_y = np.reshape(batch_y, (batch_size * seq_len,-1))
		net_out = sess.run([outputs, outputs_reshaped, prediction, loss, opt], feed_dict = {x: batch_x, y: batch_y})

		print(net_out[0][0].shape, net_out[1].shape)
		# if iter % 10 == 0:
		# 	acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
		# 	los = sess.run(loss, feed_dict = {x:batch_x, y: batch_y})
		# 	print('Iter: ', iter, 'Acc: ', acc, 'Loss: ', los)
		iter += 1

	# # Compute test accuracy
	# test_data = mnist.test.images[:128].reshape((-1, seq_len, n_input))
	# test_label = mnist.test.labels[:128]
	# print('Test Acc: ', sess.run(accuracy, feed_dict = {x: test_data, y: test_label}))
