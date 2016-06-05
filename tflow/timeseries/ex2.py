import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell
import reader as reader

class RNNModel(object):
  """The RNN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    size = config.n_hidden
    num_steps = config.num_steps
    self._input_data = tf.placeholder(tf.float32, (batch_size, config.num_steps))
    self._targets = tf.placeholder(tf.float32, [batch_size, 1])
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=2.7)
    # lstm_cell = rnn_cell.LSTMCell(size, 1)
    # cell = lstm_cell
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    vocab_size = 10

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
    self._cost = cost = tf.reduce_mean(tf.square((pred[:,0] - self.targets[:,0])))
    self._result = pred[:,0] - self.targets[:,0]

    # self._cost = cost = tf.abs(pred[0, 0] - self.targets[0,0])

    if not config.is_training:
        return

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate = config.learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    self._train_op = optimizer

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
    batch_size = 2
    max_epoch = 100
    init_scale = 0.1
    n_features = 1
    n_hidden = 4
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

def run_epoch(session, m, data, eval_op, config, epoch_num, verbose=False):
  success_inarow = 0
  max = 0
  state = m.initial_state.eval()
  final_step = 0
  for step, (x, y) in enumerate(reader.ts_iterator(data, m.batch_size,
                                                    config.num_steps + 1)):

      targets = np.reshape(x[:, -1], (config.batch_size, 1))
      in_data = x[:, 0:-1]
      final_step = step
      cost, pred, result, _ = session.run([m.cost, m.pred, m.result, eval_op],
                                   {m.targets: targets,
                                    m.input_data : in_data,
                                    m.initial_state: state})
      success = result < 0.2
      if np.sum(success) >= config.batch_size:
          success_inarow += config.batch_size
      else:
          success_inarow = 0

      if step % 50 == 0:
          print ("cost ", cost, "step", step, "epoch_num", epoch_num)

      if success_inarow >= 256:
          print ("SUCESS ", step, "num_seq", step * config.batch_size)
          break


  return (final_step + 1) * config.batch_size, success_inarow

def generateTestPattern(T, N, mean, stddev):
    startItem = 1.0
    endItem = 0.2
    if np.random.random_integers(0, 1) == 1:
        startItem = -1.0
        endItem = 0.8

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
    T = 50
    config.num_steps = T
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
    total_seq = 0
    success_inarow = 0
    max = 0
    finish = False
    for i in range(config.max_epoch):
        if finish:
            break
        num_seq = 5000
        raw_data = np.zeros(num_seq*(T+1))
        for s in range(num_seq):
            raw_data[s*(T+1): (s+1)*(T+1)] = generateTestPattern(T, N, mean, std)
        state = model.initial_state.eval()

        for step, (x, y) in enumerate(reader.ts_iterator(raw_data, config.batch_size,
                                                            config.num_steps + 1)):

              targets = np.reshape(x[:, -1], (config.batch_size, 1))
              noisytargets= targets + np.random.normal(0.0, np.square(0.1), (config.batch_size, 1))
              in_data = x[:, 0:-1]
              cost, pred, result, _ = session.run([model.cost, model.pred, model.result, model.train_op],
                                           {model.targets: noisytargets,
                                            model.input_data : in_data,
                                            model.initial_state: state})
              total_seq+= config.batch_size
              res = pred[:,0] - targets[:,0]
              success = np.abs(res) < 0.1
              if np.sum(success) >= config.batch_size:
                  success_inarow += config.batch_size
              else:
                  success_inarow = 0

              if step % 50 == 0:
                  print ("cost ", cost, "num_seq", total_seq, "max", max, "success", success_inarow)

              if success_inarow > max:
                  print ("newmax", max)
                  max = success_inarow

              if success_inarow >= 256:
                  print ("SUCESS ", step, "num_seq", total_seq)
                  finish = True
                  break

if __name__ == "__main__":
  tf.app.run()