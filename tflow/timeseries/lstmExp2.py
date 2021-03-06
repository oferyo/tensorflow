import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell

class RNNModel(object):
  """The RNN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    size = config.n_hidden
    num_steps = config.num_steps
    self._input_data = tf.placeholder(tf.float32, (batch_size, config.num_steps))
    self._targets = tf.placeholder(tf.float32, [batch_size, 1])
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=2.8)
    # lstm_cell = rnn_cell.LSTMCell(size, 1)
    # cell = lstm_cell
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)
    self._train_op = tf.no_op()
    self._result = -1

    weights_hidden = tf.constant(1.0, shape= [config.num_features, config.n_hidden])
    weights_hidden = tf.get_variable("weights_hidden", [config.num_features, config.n_hidden])
    inputs = []
    for k in range(num_steps):
        nextitem = tf.matmul(tf.reshape(self._input_data[:, k], [config.batch_size, config.num_features]) , weights_hidden)
        inputs.append(nextitem)

    outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    #output = tf.reshape(tf.concat(1, outputs), [-1, config.n_hidden])

    #pred = tf.matmul(outputs[-1], tf.get_variable("weights_out", [config.n_hidden,1])) + tf.get_variable("bias_out", [1])


    output = tf.reshape(tf.concat(1, outputs[-1]), [-1, size])
    #pred = tf.matmul(output, tf.get_variable("weights_out", [config.n_hidden,1])) + tf.get_variable("bias_out", [1])
    pred = tf.sigmoid(tf.matmul(outputs[-1], tf.get_variable("weights_out", [config.n_hidden,1])) + tf.get_variable("bias_out", [1]))
    self._pred = pred

    self._final_state = states[-1]
    self._cost = cost = tf.square((pred[:,0] - self.targets[:,0]))
    self._result = tf.abs(pred[0, 0] - self.targets[0,0])

    # self._cost = cost = tf.abs(pred[0, 0] - self.targets[0,0])

    if not config.is_training:
        return

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = config.learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    self._train_op = optimizer
    print("top ", self._train_op)


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

  @property
  def rnncell(self):
    return self._rnncell

  @property
  def result(self):
    return self._result

class BaseConfig(object):
    batch_size = 1
    max_epoch = 300000
    init_scale = 0.1
    n_features = 1
    n_hidden = 5
    learning_rate = 0.09
    num_layers = 1
    num_features = 1
    max_grad_norm = 5
    lr_decay = 0.5
    num_steps = 15
    is_training = True

class TestConfig(BaseConfig):
    batch_size = 1
    num_steps = 1
    is_training = False

def run_epoch(session, m, x_data, eval_op, config, state, verbose=False):
  costs = 0.0
  iters = 0
  # state = m.initial_state.eval()
  #because of the varied length sequence size no batching is done.
  input_size = len(x_data)
  y_data = np.zeros(input_size - 1)
  y_data[0:input_size-1] = x_data[1:input_size]
  x_data = x_data[0:input_size-1]
  batch_size = 1

  input_len = len(x_data)

  success = True
  target = np.reshape(y_data[input_len-1], (config.batch_size, 1))
  op = tf.no_op()
  for i in range(0, input_len, config.num_steps):
      if i == input_len - config.num_steps:
          op = eval_op

      input_data = np.reshape(x_data[i:i + config.num_steps], (batch_size, config.num_steps))
      cost, state, pred,result, _ = session.run([m.cost, m.final_state, m.pred, m.result, op],
                                   {m.targets: target,
                                    m.input_data : input_data,
                                    m.initial_state: state})
  #compare the last item

  success = result < 0.2
  return success, cost, state


def generateTestPattern(T, N, mean, stddev):
    startItem = 1.0
    endItem = 1.0
    if np.random.random_integers(0, 1) == 1:
        startItem = -1.0
        endItem = 0.0

    # seq_len = np.random.random_integers(T, T+T/10)
    seq_len = T + 1
    res = np.zeros(seq_len)
    res[0:N] = startItem
    res[N:seq_len - 1] = np.random.normal(mean, stddev, seq_len - N - 1)
    res[seq_len - 1] = endItem

    return res

def main(unused_args):

  with tf.Graph().as_default(), tf.Session() as session:
    config = BaseConfig()
    T = 100
    config.num_steps = 50
    # testConfig = TestConfig()
    initializer = tf.random_normal_initializer(0.0, 1.0, None)
    #initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      model = RNNModel(True, config)
    # with tf.variable_scope("model", reuse=True, initializer=initializer):
    #     testModel = RNNModel(False, testConfig)

    tf.initialize_all_variables().run()
    num_succes = 0
    N = 3
    mean = 0.0
    std = np.sqrt(0.2)
    max = 0
    state = model.initial_state.eval()
    for i in range(config.max_epoch):
        x_data = generateTestPattern(T, N, mean, std)
        success, cost, state = run_epoch(session, model, x_data, model.train_op, config, model.initial_state.eval())

        if (success):
            num_succes+=1
        else:
            num_succes = 0

        if num_succes > max:
            max = num_succes
            print ("NewMax ", max, "i", i)

        if num_succes > 256:
            print ("END AT ", i)
            break

        if i % 50 == 0:
            if x_data[0] == 1.0:
                print ("            cost 1 ", cost, "i", i, "max", max)
            else:
                print ("cost -1 ", cost, "i", i, "max", max)



if __name__ == "__main__":
  tf.app.run()