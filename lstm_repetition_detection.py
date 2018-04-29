# Tutorial from: https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


# Declare constants

# LSTM is unrolled through 10 time steps
seq_len = 10
# Number of hidden LSTM units
num_units = 10
# Size of each input
n_input = 1
# Learning rate
learning_rate = 0.0001
# Beta (for ADAM)
beta = 0.9
# Momentum
momentum = 0.099
# Number of classes
n_classes = 2
# Batch size
batch_size = 10
# Train/Test split
train_split = 0.8
# Number of epochs
num_epochs = 200
# Flag to check if loss has been stepped down
stepFlag = False
# Class weights (for 0->no-repetition vs 1->repetition)
class_weights = tf.constant([0.01, 0.99])
# class_weights = np.ones((batch_size * seq_len,n_classes))
# class_weights[:,0] = 0.1*class_weights[:,0]
# class_weights[:,1] = 0.1*class_weights[:,1]
# class_weights_tensor = tf.constant(class_weights, dtype = tf.float32)


# Synthesize data
np.random.seed(12345)
num_tokens = 10
dataset_size = 20
data = np.zeros((dataset_size, seq_len))
label = np.zeros((dataset_size, seq_len, n_classes))
label = np.concatenate((np.ones((dataset_size, seq_len, 1)), np.zeros((dataset_size, seq_len, 1))), axis = -1)
print(label.shape)
for i in range(dataset_size):
	# Generate a random permutation of all tokens. 
	# Throw in a random translation of all tokens.
	tmp = 0.01 * np.random.permutation(num_tokens) #+ np.random.randint(50)
	coin_filp = np.random.randint(0,2)
	if coin_filp == 0 or coin_filp == 1:
		# Add a repetition
		# Index of number to repeat
		tmpIdx_src = np.random.randint(len(tmp))
		# Where to repeat that number
		tmpIdx_dst = np.random.randint(len(tmp))
		while tmpIdx_dst == tmpIdx_src:
			tmpIdx_dst = np.random.randint(len(tmp))
		tmp[tmpIdx_dst] = tmp[tmpIdx_src]
		# label[i,tmpIdx_src,1] = 1.0
		if tmpIdx_src > tmpIdx_dst:
			tmpvar = tmpIdx_dst
			tmpIdx_dst = tmpIdx_src
			tmpIdx_src = tmpvar
		label[i,tmpIdx_dst,0] = 0.0
		label[i,tmpIdx_dst,1] = 1.0
		label[i,tmpIdx_src,0] = 1.0
		label[i,tmpIdx_src,1] = 0.0
	data[i,:] = tmp
	# print(data[i,:])
	# print(label[i,:])


# More variable definitions
num_iters = int(np.floor(dataset_size / batch_size))
train_iters = int(train_split * num_iters)
test_iters = num_iters - train_iters


# Define placeholders

# Outputs
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

# Inputs
x = tf.placeholder("float", [None, seq_len, n_input])
y = tf.placeholder("float", [None, seq_len, n_classes])

# Reshape x from shape [batch_size, seq_len, n_input] to
# 'seq_len' number of [batch_size, n_input] tensors
input = tf.unstack(x, seq_len, axis = 1)

# Define network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias = 1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype = tf.float32)
outputs = tf.transpose(outputs, [1,0,2])
# Reshape outputs to batch_size * seq_len x num_units
# outputs_reshaped = tf.reshape(outputs, [batch_size * seq_len, num_units])
outputs_reshaped = tf.reshape(outputs, [-1, num_units])
prediction = tf.matmul(outputs_reshaped, out_weights) + out_bias
weighted_prediction = tf.multiply(prediction, class_weights)

# Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = weighted_prediction, labels = y))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
# Optimization
# opt = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta).minimize(loss)
opt = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = momentum).minimize(loss)

# Model evaluation
# correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
soft = tf.nn.softmax(prediction)
# # Set the idx of the max location to 1 and the other label to 0
# hard = tf.where(tf.equal(tf.reduce_max(soft, axis = 1, keep_dims = True), soft), \
# 	tf.constant(1.0, shape = soft.shape), \
# 	tf.constant(0.0, shape = soft.shape))

# Init variables
init = tf.global_variables_initializer()

# Run session
with tf.Session() as sess:
	sess.run(init)
	epoch = 1
	while epoch < num_epochs:

		shuffledOrder = np.random.permutation(dataset_size)

		# if epoch > 7 and not stepFlag:
		# 	learning_rate = learning_rate / 10
		# 	stepFlag = True

		iter = 1
		while iter < train_iters:
			# # batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
			# # batch_x = batch_x.reshape((batch_size, seq_len, n_input))
			# startIdx = (iter-1)*batch_size
			# endIdx = iter*batch_size
			# batch_x = data[startIdx:endIdx,:]
			# batch_x = np.expand_dims(batch_x, -1)
			# # print(batch_x, batch_x.shape)
			# batch_y = label[startIdx:endIdx,:,:]
			# # print(batch_y, batch_y.shape)
			# # batch_y = np.reshape(batch_y, (batch_size * seq_len,-1))

			curIterInds = shuffledOrder[(iter-1)*batch_size:iter*batch_size]
			batch_x = data[curIterInds,:]
			batch_x = np.expand_dims(batch_x, -1)
			batch_y = label[curIterInds,:,:]

			net_out = sess.run([outputs, outputs_reshaped, prediction, loss, opt], \
				feed_dict = {x: batch_x, y: batch_y})

			# print('Pred:', net_out[2], net_out[2].shape)
			# print('Soft:', net_out[5])
			# print('Label:', batch_y, batch_y.shape)
			if iter % 10 == 0:
				# acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
				# los = sess.run(loss, feed_dict = {x:batch_x, y: batch_y})
				# print('Iter: ', iter, 'Acc: ', acc, 'Loss: ', los)

				# Non-Repeat Accuracy
				tmp_out = np.transpose(batch_y, [1,0,2])
				tmp_out = np.reshape(batch_y, (batch_size*seq_len, n_classes))
				print('Epoch: ', epoch, 'Loss:', np.sum(np.abs(net_out[2] - tmp_out)))
			
			iter += 1

		while iter < train_iters + test_iters:
			# batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
			# batch_x = batch_x.reshape((batch_size, seq_len, n_input))
			startIdx = (iter-1)*batch_size
			endIdx = iter*batch_size
			batch_x = data[startIdx:endIdx,:]
			batch_x = np.expand_dims(batch_x, -1)
			# print(batch_x, batch_x.shape)
			batch_y = label[startIdx:endIdx,:,:]
			# print(batch_y, batch_y.shape)
			# batch_y = np.reshape(batch_y, (batch_size * seq_len,-1))
			net_out = sess.run([prediction], feed_dict = {x: batch_x, y: batch_y})

			# print('Pred:', net_out[2], net_out[2].shape)
			# print('Soft:', net_out[5])
			# print('Label:', batch_y, batch_y.shape)
			if iter % 10 == 0:
				# acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
				# los = sess.run(loss, feed_dict = {x:batch_x, y: batch_y})
				# print('Iter: ', iter, 'Acc: ', acc, 'Loss: ', los)

				# Non-Repeat Accuracy
				tmp_out = np.transpose(batch_y, [1,0,2])
				tmp_out = np.reshape(batch_y, (batch_size*seq_len, n_classes))
				print('Epoch: ', epoch, 'Test Loss:', np.sum(np.abs(net_out[0] - tmp_out)))
			
			iter += 1

		epoch += 1

	# Number of repetitions detected
	iter = train_iters
	while iter < train_iters + test_iters:
		startIdx = (iter-1)*batch_size
		endIdx = iter*batch_size
		batch_x = data[startIdx:endIdx,:]
		batch_x = np.expand_dims(batch_x, -1)
		batch_y = label[startIdx:endIdx,:,:]
		net_out = sess.run([prediction], feed_dict = {x: batch_x, y: batch_y})

		if iter % 10 == 0:
			tmp_out = np.transpose(batch_y, [1,0,2])
			tmp_out = np.reshape(batch_y, (batch_size*seq_len, n_classes))
			print('Final Test Loss:', np.sum(np.abs(net_out[0] - tmp_out)))
			# net_out_hard = np.where(np.equal(np.maximum.reduce(net_out[0], axis = 0), net_out[0]), \
			# 	np.ones(net_out[0].shape), np.zeros(net_out[0].shape))
			net_out_hard = np.where(np.equal(np.matlib.repmat(np.maximum.reduce(net_out[0], axis = 1), 2, 1).T, \
				net_out[0]), np.ones(net_out[0].shape), np.zeros(net_out[0].shape))
			for k in range(batch_size):
				print(net_out_hard[k*batch_size:(k+1)*batch_size].T)
				print(tmp_out[k*batch_size:(k+1)*batch_size].T)
			# print(net_out_hard[10:20], net_out_hard.shape)
			# print(tmp_out[10:20], tmp_out.shape)
			print('Acc: ', np.mean(np.abs(np.ones(net_out_hard.shape) - np.abs(net_out_hard - tmp_out))))
			# break

		iter += 1
