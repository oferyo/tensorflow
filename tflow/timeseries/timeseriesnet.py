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
    self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=1.0)
    # if is_training and config.keep_prob < 1:
    #   lstm_cell = rnn_cell.DropoutWrapper(
    #       lstm_cell, output_keep_prob=config.keep_prob)

    # test_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    # with tf.device("/cpu:0"):
    #   embedding = tf.get_variable("embedding", [3, size])
    #   inputs = tf.nn.embedding_lookup(embedding, test_data)
    #   inputs = [tf.squeeze(input_, [1])
    #             for input_ in tf.split(1, num_steps, inputs)]


    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    x_debug_test = [2.0,2.0,2.0]

    inputs_debug = []
    for j in range(size):
        inputs_debug.append(x_debug_test)

    tensor_debug = tf.constant(inputs_debug, shape=[batch_size,3, size])
    outputs, states = rnn.rnn(cell, [tensor_debug], initial_state=self._initial_state)

    inputs = []
    #weights_hidden = tf.get_variable("weights_hidden", [config.num_features, config.n_hidden])

    weights_hidden = tf.constant(1.0, shape= [config.num_features, config.n_hidden])

    #bias_hidden =  tf.get_variable("bias_hidden", [1,config.n_hidden])
    for n in range(num_steps):
        #nextitem = tf.matmul(tf.reshape(self._input_data[:, n], [config.batch_size, config.num_features]) , weights_hidden) + bias_hidden
        nextitem = tf.matmul(tf.reshape(self._input_data[:, n], [config.batch_size, config.num_features]) , weights_hidden)
        inputs.append(nextitem)

    # if is_training and config.keep_prob < 1:
    #   inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # from tensorflow.models.rnn import rnn
    outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)

    # outputs = []
    # states = []
    # state = self._initial_state
    # with tf.variable_scope("RNN"):
    #   for time_step in range(num_steps):
    #     if time_step > 0: tf.get_variable_scope().reuse_variables()
    #     currentInput = tf.matmul(tf.reshape(self._input_data[:, time_step:time_step+1], [config.batch_size, config.num_features]) , tf.get_variable("weights_hidden", [config.num_features, config.n_hidden])) + tf.get_variable("bias_hidden", [1,config.n_hidden])
    #
    #     (cell_output, state) = cell(currentInput, state)
    #     outputs.append(cell_output)
    #     states.append(state)

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
    max_epoch = 300
    init_scale = 0.1
    n_features = 1
    n_hidden = 11
    learning_rate = 0.08
    num_layers = 2
    num_steps = 5
    num_features = 1
    max_grad_norm = 5
    lr_decay = 0.5

class TestConfig(BaseConfig):
    batch_size = 1
    num_steps = 1

def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for step, (x, y) in enumerate(reader.ts_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, pred, _ = session.run([m.cost, m.final_state, m.pred, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})

    if verbose:
        print("x", x[0,0], "y", y[0, 0], "pred", pred[0,0])
    costs += cost
    iters += 1.0

  return costs / iters


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

def getNaiveError(x_data):
    in_sample_size = len(x_data)

    y_data = np.zeros(in_sample_size)
    y_data[0:in_sample_size-1] = x_data[1:in_sample_size]
    y_data[in_sample_size - 1] = y_data[in_sample_size - 2]

    return np.mean(np.square(y_data - x_data))

def main(unused_args):

  with tf.Graph().as_default(), tf.Session() as session:
    config = BaseConfig()
    testConfig = TestConfig()
    init_scale = 1.0 // np.sqrt(config.n_hidden)
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    initializer = tf.random_normal_initializer(0.0, 1.0, None)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      model = RNNModel(True, config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        testModel = RNNModel(False, testConfig)

    tf.initialize_all_variables().run()
    x_data = generateTestPattern(200, 0.2, 2.0, 0.2)
    naiveInError = getNaiveError(x_data)

    test_data = generateTestPattern(200, 0.2, 2.0, 0.2)
    test_data = test_data[20:len(test_data)]
    naiveTestError = getNaiveError(test_data)

    for i in range(config.max_epoch):
        # lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        # model.assign_lr(session, config.learning_rate * lr_decay)

        cost = run_epoch(session, model, x_data, model.train_op) / naiveInError
        if i % 20 == 0:
            print ("cost", cost)

    cost = run_epoch(session, model, x_data, model.train_op) / naiveInError
    test_cost = run_epoch(session, testModel, test_data, tf.no_op(), True) / naiveTestError

    print("final cost", cost, "test_cost", test_cost)

if __name__ == "__main__":
  tf.app.run()


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