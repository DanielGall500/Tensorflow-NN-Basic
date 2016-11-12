import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt 

n_samples = 20000
n_classes = 2

max_steps = 100
batch_size = 180

precision = 0.05
lr = 0.1

n_features = 4
hidden_layer = 6
out_layer = 2

data_x, data_y  = make_classification(n_samples=n_samples, n_features=n_features, 
	n_classes=n_classes, n_informative=n_features/2, n_redundant=n_features/2)

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2)

def create_weights_biases(in_dim, out_dim, stddev=0.04):
	weights = tf.Variable(tf.truncated_normal(
		shape=[in_dim, out_dim], stddev=stddev))

	biases = tf.Variable(tf.fill([out_dim], 0.1))

	return weights, biases

def neural_net(x, num_features, hidden_nodes, output_nodes=2):

	w1, b1 = create_weights_biases(num_features, hidden_nodes)

	op1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2, b2 = create_weights_biases(hidden_nodes, output_nodes)

	logits = tf.nn.relu(tf.matmul(op1, w2) + b2)

	return logits

def loss(logits, labels):

	labels = tf.cast(labels, tf.int32)

	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits, labels)

	ce_mean = tf.reduce_mean(ce)

	return ce_mean

def get_batch(x, y, step, batch_size):
	start = step * batch_size
	end = start + batch_size

	features = x[start:end]
	labels = y[start:end]

	return features, labels

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

logits = neural_net(x, n_features, hidden_layer, out_layer)

total_loss = loss(logits, y)

train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)

with tf.Session() as sess:

	init = tf.initialize_all_variables()
	sess.run(init)

	loss_array = []

	for step in range(max_steps):

		batch_x, batch_y = get_batch(train_x, train_y, step, batch_size)

		_, l, logs = sess.run([train_op, total_loss, logits], 
			feed_dict={x: batch_x, y: batch_y})

		loss_array.append(l)

		if l <= precision:
			print "Precision Reached: {}".format(l)

			break

	print "----- Testing Accuracy -----"

	correct_pred = tf.equal(tf.argmax(logits,1), tf.cast(y,tf.int64))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	test_y = test_y.astype(np.float32)
	test_x = test_x.astype(np.float32)

	test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})

	print "Accuracy: {}%".format(int(np.round(test_acc * 100)))

plt.plot(loss_array)
plt.ylim(0,0.8)
plt.show()

























