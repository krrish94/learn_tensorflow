# LSTM to count the number of '1's in a binary string
# Reference: https://becominghuman.ai/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow-1907a5bbb1fa


import numpy as np
from random import shuffle
import tensorflow as tf



"""
Parameters
"""

# Seed for all RNGs
rng_seed = 12345
np.random.seed(rng_seed)
tf.set_random_seed(rng_seed)

# Length of each binary string (i.e., length of each input sequence)
seq_len = 15
# Maximum range (i.e., max val of the integer reprsented by the bit string)
# Note that, max val is 2**num_range
num_range = 15
# Train split (fraction of data to be used for training)
train_split = 0.8
# Number of train samples
num_samples = 2 ** num_range
num_train = int(np.floor(train_split * num_samples))
num_test = num_samples - num_train

# Dimensions
dim_input = 1
dim_output = num_range + 1 		# Since num of bits can only be in the range [0, num_range]

# Model parameters
num_hidden = 10

# Other hyperparameters
batch_size = 50
learning_rate = 0.01
momentum = 0.09
beta1 = 0.7

num_epochs = 10
num_train_batches = int(np.floor(float(num_train) / float(batch_size)))
num_test_batches = int(np.floor(float(num_test) / float(batch_size)))

# Verbosity controls
print_experiment_summary = True
if print_experiment_summary:
	print('Total number of samples:', num_samples)
	print('Train samples:', num_train)
	print('Test samples:', num_test)
	print('Batch size:', batch_size)
	print('Train batches:', num_train_batches)
	print('Test batches:', num_test_batches)
	print('Max epochs:', num_epochs)
print_train_every = 100
print_test_every = 10


"""
Generate training data
"""

# Generate all strings of numbers in the interval [0, 2**num_range]
dataset = ['{0:^0{str_len}b}'.format(i, str_len = seq_len) for i in range(2**num_range)]
# Convert the string to a set of integers
dataset = np.array([[[int(j)] for j in list(dataset[i])] for i in range(len(dataset))])
# print(dataset)

labels_helper = np.array([[np.sum(num)] for num in dataset])
labels = np.zeros((num_samples, dim_output))
cur = 0
for ind in labels_helper:
	labels[cur][ind] = 1.0
	cur += 1
# print(labels)



"""
Build the computation graph
"""

data = tf.placeholder(tf.float32, [None, seq_len, dim_input])
target = tf.placeholder(tf.float32, [None, dim_output])

recurrent_unit = tf.contrib.rnn.LSTMCell(num_hidden)
val, _ = tf.nn.dynamic_rnn(recurrent_unit, data, dtype = tf.float32)

val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight_fc = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias_fc = tf.Variable(tf.constant(0.1, shape = [target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight_fc) + bias_fc)
cross_entropy = - tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

loss = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1).minimize(cross_entropy)

# Accuracy computation
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


"""
Execute graph
"""

init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init)
	epoch = 0

	# 'Epoch' loop
	while epoch < num_epochs:

		batch = 0

		# Shuffle train data
		train_order = np.random.permutation(num_train)

		# 'Iteration' loop
		train_error_this_epoch = 0.0
		train_error_temp = 0.0
		while batch < num_train_batches:

			startIdx = batch*batch_size
			endIdx = (batch+1)*batch_size
			inds = train_order[startIdx:endIdx]
			# input_batch, label_batch = dataset[startIdx:endIdx], labels[startIdx:endIdx]	# no shuffle
			input_batch, label_batch = dataset[inds], labels[inds]

			net_out = sess.run([loss, error], feed_dict = {data: input_batch, target: label_batch})

			train_error_temp += net_out[1]
			train_error_this_epoch += net_out[1]
			if batch % print_train_every == 0:
				print('Epoch: ', epoch, 'Error: ', train_error_temp/float(print_train_every))
				train_error_temp = 0.0

			batch += 1
		# print('Epoch:', epoch, 'Full train set:', train_error_this_epoch/float(num_train))

		# Test
		if epoch % 2 == 0:
			test_error_this_epoch = 0.0
			test_error_temp = 0.0
			while batch < num_train_batches + num_test_batches:

				startIdx = batch*batch_size
				endIdx = (batch+1)*batch_size

				input_batch, label_batch = dataset[startIdx:endIdx], labels[startIdx:endIdx]

				net_out = sess.run([error, prediction], feed_dict = {data: input_batch, target: label_batch})

				test_error_temp += net_out[0]
				test_error_this_epoch += net_out[0]
				if batch % print_test_every == 0:
					print('Epoch: ', epoch, 'Error: ', test_error_temp/float(print_test_every))
					test_error_temp = 0.0
					random_disp = np.random.randint(batch_size)
					print(np.squeeze(input_batch[random_disp]))
					print('Pred:', np.argmax(net_out[1][random_disp]), 'GT:', \
						np.argmax(label_batch[random_disp]))

				batch += 1
			print('Epoch: ', epoch, 'Full test set:', test_error_this_epoch/float(num_test))

		epoch += 1
