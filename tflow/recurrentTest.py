import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell

#x_data = np.random.randn(in_sample_size).astype("float32")

learning_rate = 0.08
num_iterations = 300
n_hidden = 50
num_lables = 1
n_features = 1
num_layers = 2

def generateTestPattern(size, start, end, step):
    ptrPart = []
    res = []
    for k in xrange(2):
        ptrPart.append(start)
    for inc in np.arange(start, end, step):
        ptrPart.append(inc)
    for dec in reversed(np.arange(start, end, step)):
        ptrPart.append(dec)

    for k in xrange(size):
        res+= ptrPart

    return res

x_data = generateTestPattern(100, 0.2, 2.0, 0.2)
in_sample_size = len(x_data)

y_data = np.zeros(in_sample_size)

y_data[0:in_sample_size-1] = x_data[1:in_sample_size]
y_data[in_sample_size - 1] = y_data[in_sample_size - 2]

y_data = np.matrix(y_data).reshape((in_sample_size, 1))

naiveInError = tf.reduce_mean(tf.square(y_data - x_data))


# tf Graph input
x = tf.placeholder("float", [None, n_features])
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
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
batch_size = 1
def RNN(_X, _istate, _weights, _biases):
    global initial_state
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.0)

    if num_layers > 1:
        lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

    k = tf.split(0, 1, _X)
    initial_state= lstm_cell.zero_state(batch_size, tf.float32)
    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, k , initial_state=_istate)
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out'], states[-1]

pred, lastStates = RNN(x, istate, weights, biases)

# Define loss and optimizer
# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(pred - y_data))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

#tv = tf.trainable_variables()
# Launch the graph.
sess = tf.Session()
# Initializing the variables
init = tf.initialize_all_variables()
sess.run(init)

naiveerr = sess.run(naiveInError)

init_state = sess.run(initial_state)

#initialLoss = sess.run(loss, feed_dict={x: np.reshape(x_data, (in_sample_size,n_features)),  istate: init_state}) / naiveerr
batch_size = 1
state = init_state
resulted_prediction = []
for step in xrange(num_iterations):
    for ind in range(0, len(x_data)):
        nextarr = x_data[ind:ind + batch_size]
        #prediction, state = sess.run([pred, lastStates], feed_dict={x: np.reshape(nextarr, (batch_size, n_features)),  istate: state})
        #resulted_prediction.append(prediction[0][0])
        _, state = sess.run([optimizer, lastStates], feed_dict={x: np.reshape(nextarr, (batch_size,n_features)),  istate: state})
        if step % 20 == 0:
            currentLoss = sess.run(loss, feed_dict={x: np.reshape(nextarr, (batch_size,n_features)),  istate: state}) / naiveerr
            print(step, "loss", currentLoss)



valid_test = generateTestPattern(100, 0.2, 2.0, 0.2)
batch_size = len(valid_test)
state = np.zeros((batch_size, 2*n_hidden))


out_test = valid_test



batch_size = 1
state = np.zeros((batch_size, 2*n_hidden))
for ind in range(0, len(out_test)):
    nextarr = out_test[ind:ind + batch_size]
    sampleSize = len(nextarr)
    prediction, state = sess.run([pred, lastStates], feed_dict={x: np.reshape(nextarr, (sampleSize, n_features)),  istate: state})
    resulted_prediction.append(prediction[0][0])
    #print("input", nextarr, "prediction", prediction)


print("valie pred", valid_pred, "outsample prediction", resulted_prediction)
print("end")
#print(step, "loss in sample", sess.run(loss, feed_dict={x: np.reshape(x_data, (in_sample_size,n_features)),  istate: np.zeros((in_sample_size, 2*n_hidden))}))