import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn, rnn_cell

in_sample_size = 200

x_data = np.random.randn(in_sample_size).astype("float32")
y_data = np.zeros(in_sample_size)

y_data[1:in_sample_size] = x_data[0:in_sample_size-1]
y_data[0] = 0


naiveInError = tf.reduce_mean(tf.square(y_data))

n_hidden = 10
num_lables = 1
n_features = 1
n_steps = 2


x = tf.placeholder("float", [None, n_features])

istate = tf.placeholder("float", [None, 2*n_hidden])

#y = tf.placeholder("float", [None, 1])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_features, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, 1]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([1]))
}


def RNN(_X, _istate, _weights, _biases):

    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    #_X = tf.split(0, n_step, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, tf.split(0, 1, _X), initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = RNN(x, istate, weights, biases)

learning_rate = 0.1
num_iterations = 5000

# Define loss and optimizer
# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(pred - y_data))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#tv = tf.trainable_variables()
# Launch the graph.
sess = tf.Session()
# Initializing the variables
init = tf.initialize_all_variables()
sess.run(init)

initialLoss = sess.run(loss, feed_dict={x: np.reshape(x_data, (in_sample_size,n_features)),  istate: np.zeros((in_sample_size, 2*n_hidden))}) / sess.run(naiveInError)

print("loss before traning", initialLoss)
for step in xrange(num_iterations):
   sess.run(optimizer, feed_dict={x: np.reshape(x_data, (in_sample_size,n_features)),  istate: np.zeros((in_sample_size, 2*n_hidden))})
   #sess.run(optimizer, feed_dict={istate: np.zeros((in_sample_size, 2*n_hidden)), x: np.reshape(x_data, (in_sample_size,n_features))})

   #print("step ", step)
   if step % 20 == 0:
       currentLoss = sess.run(loss, feed_dict={x: np.reshape(x_data, (in_sample_size,n_features)),  istate: np.zeros((in_sample_size, 2*n_hidden))}) / sess.run(naiveInError)
       print(step, "loss", currentLoss)
        #print("prediction", sess.run(pred, feed_dict={x: np.reshape(x_data, (in_sample_size,n_features)),  istate: np.zeros((in_sample_size, 2*n_hidden))}), "ydata", y_data)

x_test = np.random.rand(in_sample_size).astype("float32")
zzz = sess.run(pred, feed_dict={x: np.reshape(x_test, (in_sample_size,n_features)),  istate: np.zeros((in_sample_size, 2*n_hidden))})

#print("outsample xdata", x_test, "out ", zzz)


#print(step, "loss in sample", sess.run(loss, feed_dict={x: np.reshape(x_data, (in_sample_size,n_features)),  istate: np.zeros((in_sample_size, 2*n_hidden))}))
