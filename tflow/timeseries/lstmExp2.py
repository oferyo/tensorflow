import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell
import reader as reader

#x = np.matrix([[4,5,6,1], [7,8,9,10]], int)

#z = np.reshape(x, (8, 1))


class RNNModel(object):
  """The RNN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.n_hidden
    self._input_data = tf.placeholder(tf.float32, [batch_size, None])
    self._targets = tf.placeholder(tf.float32, [batch_size, None])
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=1.0)

    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
    self._initial_state = cell.zero_state(batch_size, tf.float32)

    inputs = []
    #weights_hidden = tf.get_variable("weights_hidden", [config.num_features, config.n_hidden])

    weights_hidden = tf.constant(1.0, shape= [config.num_features, config.n_hidden])

    #bias_hidden =  tf.get_variable("bias_hidden", [1,config.n_hidden])
    for n in range(num_steps):
        #nextitem = tf.matmul(tf.reshape(self._input_data[:, n], [config.batch_size, config.num_features]) , weights_hidden) + bias_hidden
        nextitem = tf.matmul(tf.reshape(self._input_data[:, n], [config.batch_size, config.num_features]) , weights_hidden)
        inputs.append(nextitem)

    outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    pred = tf.matmul(output, tf.get_variable("weights_out", [config.n_hidden,1])) + tf.get_variable("bias_out", [1])
    self._pred = pred

    self._final_state = states[-1]
    self._cost = cost  = tf.reduce_mean(tf.square(pred - tf.reshape(self.targets, (config.num_steps * config.batch_size, 1))))

    if not is_training:
        return

    # optimizer = tf.train.AdamOptimizer(learning_rate = config.learning_rate).minimize(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = config.learning_rate).minimize(cost)
    self._train_op = optimizer

    # self._lr = tf.Variable(0.0, trainable=False)
    # tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
    #                                   config.max_grad_norm)
    # optimizer = tf.train.GradientDescentOptimizer(self.lr)
    # self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

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

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def pred(self):
    return self._pred

class BaseConfig(object):
    batch_size = 1
    max_epoch = 30000
    init_scale = 0.1
    n_features = 1
    n_hidden = 11
    learning_rate = 0.08
    num_layers = 2
    num_steps = 1
    num_features = 1
    max_grad_norm = 5
    lr_decay = 0.5

class TestConfig(BaseConfig):
    batch_size = 1
    num_steps = 1

def run_epoch(session, m, x_data, eval_op, verbose=False):
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  #because of the varied length sequence size no batching is done.
  input_size = len(x_data)
  y_data = np.zeros(input_size - 1)
  y_data[0:input_size-1] = x_data[1:input_size]
  x_data = x_data[0:input_size-1]
  batch_size = 1

  input_data = np.reshape(x_data, (batch_size, len(x_data)))
  out_data = np.reshape(y_data, (batch_size, len(x_data)))

  cost, state, pred, _ = session.run([m.cost, m.final_state, m.pred, eval_op],
                                  {m.input_data: input_data,
                                   m.targets: out_data,
                                   m.initial_state: state})
  success = np.abs(pred[0, len(y_data) -1] - y_data[input_size - 2]) < 0.2
  return success


def generateTestPattern(T, N, mean, stddev):
    startItem = 1.0
    endItem = 1.0
    if np.random.random_integers(0, 1) == 1:
        startItem = -1.0
        endItem = 0.0

    seq_len = np.random.random_integers(T, T+T/10)
    res = np.zeros(seq_len)
    res[0:N] = startItem
    res[N:seq_len - 1] = np.random.normal(mean, stddev, seq_len - N - 1)

    res[seq_len - 1] = endItem

    return res

def main(unused_args):

  with tf.Graph().as_default(), tf.Session() as session:
    config = BaseConfig()
    testConfig = TestConfig()
    initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      model = RNNModel(True, config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        testModel = RNNModel(False, testConfig)

    tf.initialize_all_variables().run()
    num_succes = 0
    T = 100
    N = 3
    mean = 0.0
    std = 0.2
    for i in range(config.max_epoch):
        x_data = generateTestPattern(T, N, mean, std)

        if (run_epoch(session, model, x_data, model.train_op)):
            num_succes+=1
        else:
            num_succes = 0

        if num_succes > 256:
            print ("END AT ", i)
            break

        if i % 20 == 0:
            print ("i", i, "num_sucess", num_succes)

if __name__ == "__main__":
  tf.app.run()