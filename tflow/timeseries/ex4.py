import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.models.rnn import linear


class RNNModel(object):
  """The RNN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    size = config.n_hidden
    num_steps = config.num_steps
    self._input_data = tf.placeholder(tf.float32, (batch_size, config.num_features, config.num_steps))
    self._targets = tf.placeholder(tf.float32, [batch_size, 1])
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=config.forget_bias)
    #lstm_cell = rnn_cell.LSTMCell(size , size, True)
    # cell = lstm_cell
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)


    # weights_hidden = tf.constant(1.0, shape= [config.num_features, config.n_hidden])
    weights_hidden = tf.get_variable("weights_hidden", [config.num_features, size])

    # in_lstm = [tf.squeeze(input_, [2])
    #              for input_ in tf.split(2, config.num_steps, self._input_data)]
    #
    # in_lstm = linear.linear(in_lstm, size, True)
    inputs = []
    for k in range(num_steps):
        # litem = linear.linear(self._input_data[:, :, k], size, True)
        # nextitem = tf.matmul(tf.reshape(self._input_data[:, :, k], [config.batch_size, config.num_features]) , weights_hidden)
        nextitem = tf.matmul(self._input_data[:, :, k] , weights_hidden)
        inputs.append(nextitem)

    outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    #output = tf.reshape(tf.concat(1, outputs), [-1, config.n_hidden])

    #pred = tf.matmul(outputs[-1], tf.get_variable("weights_out", [config.n_hidden,1])) + tf.get_variable("bias_out", [1])


    output = tf.reshape(tf.concat(1, outputs[-1]), [-1, size])

    # pred = tf.sigmoid(tf.matmul(outputs[-1], tf.get_variable("weights_out", [config.n_hidden,1])) + tf.get_variable("bias_out", [1]))
    pred = tf.sigmoid(tf.matmul(outputs[-1], tf.get_variable("weights_out", [config.n_hidden,1])))
    self._pred = pred

    self._final_state = states[-1]
    self._cost = cost = tf.reduce_mean(tf.square((pred[:,0] - self.targets[:,0])))
    self._result = pred[:,0] - self.targets[:,0]

    # self._cost = cost = tf.abs(pred[0, 0] - self.targets[0,0])

    if not config.is_training:
        return

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = config.learning_rate).minimize(cost)
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
    batch_size = 1
    max_epoch = 300000
    init_scale = 0.1
    n_features = 2
    n_hidden = 4
    learning_rate = 0.09
    num_layers = 1
    num_features = 2
    max_grad_norm = 5
    lr_decay = 0.5
    num_steps = 15
    is_training = True

class TestConfig(BaseConfig):
    batch_size = 1
    num_steps = 1
    is_training = False


def generateTestPattern(T):
    res = np.zeros((2, T) , float)

    res[0, :] = np.random.uniform(-1.0, 1.0, (T))
    res[1, 0] = -1.0
    res[1, T-2] = -1.0

    x1locatoin = np.random.random_integers(2, T/2)
    #x2location = np.random.random_integers(2,np.ceil(T / 2))
    x2location = x1locatoin
    while x2location == x1locatoin:
          x2location = np.random.random_integers(1,11)
        # x2location = np.random.random_integers(1,T/2)

    x1 = res[0, x1locatoin]
    x2 = res[0, x2location]

    res[1, x1locatoin] = 1.0
    res[1, x2location] = 1.0

    target = 0.5 + (x1+x2)/4.0
    # print("traget", target)
    res[0, T-1] = target

    return res

def get_batch_data(batchSize, T):
    res = np.zeros((batchSize, 2, T), float)
    for i in range(batchSize):
        res[i] = generateTestPattern(T)

    return res

def run_configuration(std_for_init, forget_bias, num_hidden, b_size, max_sequences, verbose):

  with tf.Graph().as_default(), tf.Session() as session:
    config = BaseConfig()
    config.forget_bias = forget_bias
    config.n_hidden = num_hidden
    config.batch_size = b_size

    T = 100
    config.num_steps = T
    T = T+1

    # testConfig = TestConfig()
    initializer = tf.random_normal_initializer(0.0, std_for_init, None)
    # initializer = tf.random_uniform_initializer(-std_for_init, std_for_init)
    config.initializer = initializer

    with tf.variable_scope("model", reuse=None, initializer=initializer):
      model = RNNModel(True, config)
    # with tf.variable_scope("model", reuse=True, initializer=initializer):
    #     testModel = RNNModel(False, testConfig)

    tf.initialize_all_variables().run()
    # tv = tf.trainable_variables()
    total_seq = 0
    success_inarow = 0
    max = 0
    state = model.initial_state.eval()
    avg_cost = 0.0
    avg_anive = 0.0
    for i in range(config.max_epoch):
        x = get_batch_data(config.batch_size, T)

        targets = np.reshape(x[:, 0, -1], (config.batch_size, 1))
        in_data = x[:, :, 0:-1]
        # cost, state, pred, result, _ = session.run([model.cost, model.final_state, model.pred, model.result, model.train_op],
        cost, state, pred, result, _ = session.run([model.cost, model.final_state, model.pred, model.result, model.train_op],
                                   {model.targets: targets,
                                    model.input_data : in_data,
                                    model.initial_state: state})
        avg_cost+=cost
        total_seq+= config.batch_size
        res = pred[:,0] - targets[:,0]
        avg_anive+= np.mean(np.square(res - 0.5))
        success = np.abs(res) <= 0.04
        if np.sum(success) >= config.batch_size:
            success_inarow += config.batch_size
        else:
            success_inarow = 0

        if total_seq >= max_sequences:
            break

        if success_inarow > max:
            max = success_inarow

        if success_inarow >= 2000:
           print ("Success num_seq", total_seq)
           break

        if verbose and (i % 50 == 0):
            print ("cost", cost, "total seq", total_seq, "max", max, "bs", config.batch_size)
    return max, (avg_cost/i), avg_anive/i

def main(unused_args):
    n_hidden = 2
    max_sequence = 30000
    print ("start ", n_hidden)
    # for f in  np.arange(2.0, 0.1, -0.2, float):
    #     for s in np.arange(1.5, 0.4, -0.5, dfloat):
    #         max, cost = run_configuration(s, f, n_hidden, max_sequence)
    #         print ("std:", s, "fb", f, "cost", cost, "max", max)
    # for j in range(3):
    #     for i in range(2, 5, 1):
    #         for b in range(2,11,5):
    #             max, cost, naive = run_configuration(1.0, 1.2, i,b, max_sequence)
    #             print ("hidden:", i, "b", b, "cost", cost, "naive", naive, "max", max)

    for i in range(6):
        for f in  np.arange(2.8, 2.9, 0.3, float):
            max, cost, ncost = run_configuration(0.3, f, 4, 10, 800000, False)
            print ("i", i, "fb", f, "cost", cost, "max", max, "ncost", ncost, "bs", 10)
            # max, cost, ncost = run_configuration(0.3, f, 4, 20, 400000, False)
            # print ("fb", f, "cost", cost, "max", max, "ncost", ncost, "bs", 20)

if __name__ == "__main__":
  tf.app.run()