import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell
import reader as reader

class RNNModel():
    def __init__(self, config):
        lstm_cell = rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=0.0)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
        self._train_op = tf.no_op()
        self._input_data = tf.placeholder(tf.int32, [config.batch_size])
        _X = tf.matmul(self._input_data, tf.get_variable("weights_out", [config.n_hidden, 1])) + tf.get_variable("bias_hidden", [config.n_hidden])
        self._targets = tf.placeholder(tf.int32, [config.batch_size])
        self._initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = self._initial_state

        outputs, states = rnn.rnn(cell, self.input_data,tf.split(0, 1, _X), initial_state=state)
        pred = tf.matmul(outputs[-1], tf.get_variable("weights_hidden", [config.n_features, config.n_hidden])) + tf.get_variable("weights_out", [1])

        self._final_state = states[-1]
        self._cost = cost  = tf.reduce_mean(tf.square(pred - self.targets))
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        if not config.is_training:
            return

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = config.learning_rate).minimize(cost)
        self._train_op = optimizer



    @property
    def train_op(self):
        return self._train_op

    @property
    def input_data(self):
        return self._input_data

    @property
    def initial_state(self):
        return self._initial_state


    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

class BaseConfig(object):
    batch_size = 1
    max_epoch = 100
    init_scale = 0.1
    n_features = 1
    n_hidden = 10
    learning_rate = 0.1


def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += m.num_steps

  return np.exp(costs / iters)


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

def main(unused_args):

  with tf.Graph().as_default(), tf.Session() as session:
    config = BaseConfig()
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      model = RNNModel(is_training=True)

    tf.initialize_all_variables().run()
    x_data = generateTestPattern(100, 0.2, 2.0, 0.2)

    for i in range(config.max_epoch):
        run_epoch(session, model, x_data)

if __name__ == "__main__":
  tf.app.run()