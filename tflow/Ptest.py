import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
data_size = 2000
x_data = np.random.rand(data_size).astype("float32") * 6 - 3
xmat = np.matrix(x_data).reshape(data_size,1)
y_data =  xmat * 0.2 - 0.3

y_data = y_data + np.random.randn(data_size) * 1.0

naiveGuess = -0.3
naive_y_err = tf.reduce_mean(tf.square(y_data - naiveGuess))

num_hidden = 2
num_lables = 1
num_features = 1


#weights_hidden = tf.Variable(tf.random_uniform([num_features, num_hidden]))

weights_hidden = tf.Variable(tf.ones([num_features, num_hidden]))
b_hidden = tf.Variable(tf.zeros([1, num_hidden]))
weights_out = tf.Variable(tf.zeros([num_hidden, num_lables]))
#weights_out = tf.Variable(tf.constant([[1.0]]))
#weights_out = tf.Variable(tf.constant([[0.0], [0.0], [1.0]]))

#weights_out.assign(tf.ones_like(weights_out))

b_out = tf.Variable(tf.zeros([1, num_lables]))

hidden = tf.nn.tanh(tf.matmul(xmat, weights_hidden) + b_hidden)
y = tf.matmul(hidden, weights_out) + b_out

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.4)
train = optimizer.minimize(loss)


# Launch the graph.
sess = tf.Session()

init = tf.initialize_all_variables()
sess.run(init)
#sess.run(init, feed_dict={weights_out : tf.constant([[0,0,1]])})

#print("wout", sess.run(weights_out), "y", sess.run(tf.matmul(hidden, weights_out) + b_out))

n = sess.run(naive_y_err);

print("before trainging loss", sess.run(loss/n), "naive err", n)

for step in xrange(2000):
   sess.run(train)
   if step % 20 == 0:
        print(step, "loss", sess.run(loss/n))

print("wout", sess.run(weights_out), "bout", sess.run(b_out))
#print ("eval", acc.eval(feed_dict={z:5}, session=sess), "k:", sess.run(k), "z:", sess.run(z))
