# Tutorial from: https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn



# Seed RNG
rng_seed = 12345
np.random.seed(rng_seed)
tf.set_random_seed(rng_seed)

# Declare constants

# Dataset generation params
num_tokens = 10
dataset_size = 400000

# LSTM is unrolled through 10 time steps
seq_len = 10
# Number of hidden LSTM units
num_units = 10
# Size of each input
n_input = 1
# Learning rate
learning_rate = 0.01
# Beta (for ADAM)
beta = 0.9
# Momentum
momentum = 0.099
# Number of classes
n_classes = seq_len + 1
# Batch size
batch_size = 500
# Train/Test split
train_split = 0.8
# Number of epochs
num_epochs = 1000
# Flag to check if loss has been stepped down
stepFlag = False
# Class weights (for 0->no-repetition vs 1->repetition)
class_weights = tf.constant([0.1, 0.9])
# class_weights = np.ones((batch_size * seq_len,n_classes))
# class_weights[:,0] = 0.1*class_weights[:,0]
# class_weights[:,1] = 0.1*class_weights[:,1]
# class_weights_tensor = tf.constant(class_weights, dtype = tf.float32)

# More variable definitions
num_iters = int(np.floor(dataset_size / batch_size))
train_iters = int(train_split * num_iters)
test_iters = num_iters - train_iters
num_train = int(np.floor(train_split * dataset_size))
num_test = dataset_size - num_train

# Verbosity controls
print_experiment_summary = True
if print_experiment_summary:
	print('Total number of samples:', dataset_size)
	print('Train samples:', num_train)
	print('Test samples:', num_test)
	print('Batch size:', batch_size)
	print('Train batches:', train_iters)
	print('Test batches:', test_iters)
	print('Max epochs:', num_epochs)
print_train_every = 1000
print_test_every = 100


# Synthesize data

data = np.zeros((dataset_size, seq_len))
label = np.zeros((dataset_size, n_classes))
print(label.shape)
for i in range(dataset_size):
	# Generate a random permutation of all tokens. 
	# Throw in a random translation of all tokens.
	tmp = np.random.permutation(num_tokens) #+ np.random.randint(50)
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
		label[i,tmpIdx_dst+1] = 1.0
	else:
		label[i,0] = 1.0
	data[i,:] = tmp
	# print(data[i,:])
	# print(label[i,:])


# Define placeholders

# Outputs
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
# out_bias = tf.Variable(tf.random_normal([n_classes]))
out_bias = tf.Variable(tf.constant(0.0, shape = [n_classes], dtype = tf.float32))

# Inputs
x = tf.placeholder("float", [None, seq_len, n_input])
y = tf.placeholder("float", [None, n_classes])

# Reshape x from shape [batch_size, seq_len, n_input] to
# 'seq_len' number of [batch_size, n_input] tensors
input = tf.unstack(x, seq_len, axis = 1)

# Define network
lstm_layer = rnn.GRUCell(num_units)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype = tf.float32)
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

# Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
alpha = 0.0000001
regularizer = tf.nn.l2_loss(out_weights)
loss = tf.reduce_mean(loss + alpha * regularizer)
# Optimization
# opt = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta).minimize(loss)
opt = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = momentum).minimize(loss)

# Accuracy Computation
mistakes = tf.not_equal(tf.argmax(y, axis = 1), tf.argmax(prediction, axis = 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

# Init variables
init = tf.global_variables_initializer()

# Run session
with tf.Session() as sess:
	
	sess.run(init)
	epoch = 0
	
	while epoch < num_epochs:

		# if epoch == 50 or epoch == 100 or epoch == 150 or epoch == 250 or epoch == 300 or epoch == 500:
		# 	learning_rate = 0.1 * learning_rate
		# 	if epoch == 150:
		# 		beta1 = 0.7
		if epoch % 50 == 0 and epoch < 500:
			learning_rate = 0.1 * learning_rate
			if epoch == 150:
				beta1 = 0.7
			if epoch == 450:
				beta1 = 0.6

		shuffledOrder = np.random.permutation(dataset_size)

		# if epoch > 7 and not stepFlag:
		# 	learning_rate = learning_rate / 10
		# 	stepFlag = True

		iter = 0
		train_error_this_epoch = 0.0
		train_error_temp = 0.0
		while iter < train_iters:

			curIterInds = shuffledOrder[iter*batch_size:(iter+1)*batch_size]
			batch_x = data[curIterInds,:]
			batch_x = np.expand_dims(batch_x, -1)
			batch_y = label[curIterInds,:]

			net_out = sess.run([loss, opt, prediction, error], feed_dict = {x: batch_x, y: batch_y})

			train_error_this_epoch += net_out[3]
			train_error_temp += net_out[3]
			if iter % print_train_every == 0:
				print('Epoch: ', epoch, 'Iter', iter, 'Loss:', net_out[0])
				train_error_temp = 0.0
			
			iter += 1

		test_error_this_epoch = 0.0
		test_error_temp = 0.0
		tmp_counter = 0
		while iter < train_iters + test_iters:
			# batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
			# batch_x = batch_x.reshape((batch_size, seq_len, n_input))
			startIdx = iter*batch_size
			endIdx = (iter+1)*batch_size
			batch_x = data[startIdx:endIdx,:]
			batch_x = np.expand_dims(batch_x, -1)
			batch_y = label[startIdx:endIdx,:]
			net_out = sess.run([loss, prediction, error], feed_dict = {x: batch_x, y: batch_y})

			test_error_this_epoch += net_out[2]
			tmp_counter += 1
			test_error_temp += net_out[2]
			if iter % print_test_every == 0:
				print('Epoch: ', epoch, 'Test Err:', net_out[2])
				test_error_temp = 0.0
				random_disp = np.random.randint(batch_size)
				print(np.squeeze(batch_x[random_disp]))
				print('Pred:', np.argmax(net_out[1][random_disp]), \
					'GT:', np.argmax(batch_y[random_disp]))
			
			iter += 1
		print('#######################')
		print('Error:', test_error_this_epoch / float(tmp_counter))
		print('#######################')

		epoch += 1
