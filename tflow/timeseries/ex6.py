import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell

class RNNModel(object):
  """The RNN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    hidden_size = config.n_hidden
    num_steps = config.num_steps
    self._input_data = tf.placeholder(tf.float32, (batch_size, config.num_steps, config.num_features))
    self._targets = tf.placeholder(tf.float32, [batch_size, config.num_labels])
    lstm_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=config.forget_bias)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    weights_hidden = tf.get_variable("weights_hidden", [config.num_features, hidden_size])

    # test_input = tf.placeholder(tf.int32, (batch_size, 2))
    # with tf.device("/cpu:0"):
    #   embedding = tf.get_variable("embedding", [10, 11])
    #    inputs_test = tf.nn.embedding_lookup(embedding, test_input)


    b_hidden = tf.get_variable("b_hidden", [1, hidden_size])
    inputs = []
    for k in range(num_steps):
        nextitem = tf.matmul(self._input_data[:, k, :] , weights_hidden)
        inputs.append(nextitem)

    outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    # pred = tf.sigmoid(tf.matmul(outputs[-1], tf.get_variable("weights_out", [config.n_hidden,config.num_labels]))) + tf.get_variable("bias_out", [config.num_labels])
    pred = tf.sigmoid(tf.matmul(outputs[-1], tf.get_variable("weights_out", [config.n_hidden,config.num_labels])))
    # pred = tf.sigmoid(tf.matmul(outputs[-1], tf.get_variable("weights_out", [config.n_hidden,config.num_labels]))  + tf.get_variable("bias_out", [config.num_labels]))
    self._pred = pred

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred, self._targets, name='xentropy')
    self._cost = cost = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    self._final_state = states[-1]

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
    batch_size = 5
    max_epoch = 3000000
    init_scale = 0.1
    n_hidden = 4
    learning_rate = 0.5
    num_layers = 1
    num_features = 8
    num_labels = 4
    max_grad_norm = 5
    lr_decay = 0.5
    num_steps = 15
    forget_bias = 1.3

    is_training = True

class TestConfig(BaseConfig):
    batch_size = 1
    num_steps = 1
    is_training = False

#input is of form B, b, c, d, e, E, X, Y

def generateTestPattern(T):
    res = np.zeros((T, 8), float)
    res[0, 0] = 1
    res[T-1, 5] = 1

    t1locatoin = np.random.random_integers(10, 20)
    # t2locatoin = np.random.random_integers(50, 60)
    t2locatoin = np.random.random_integers(30, 40)

    x1 = np.random.random_integers(6, 7, 2)

    res[t1locatoin, x1[0]] = 1
    res[t2locatoin, x1[1]] = 1

    for j in range(1, T-1, 1):
        if (j != t1locatoin and j != t2locatoin):
            res[j, np.random.random_integers(1, 4)] = 1

    #set the target
    target = np.zeros(4)
    if x1[0] == 6 and x1[1] == 6:
        target[0] = 1
    elif x1[0] == 6 and x1[1] == 7:
        target[1] = 1
    elif x1[0] == 7 and x1[1] == 6:
        target[2] = 1
    elif x1[0] == 7 and x1[1] == 7:
        target[3] = 1

    return res, target

def get_batch_data(batchSize, T):
    res = np.zeros((batchSize, T, 8), float)
    target = np.zeros((batchSize, 4), float)
    for i in range(batchSize):
        x, y = generateTestPattern(T)
        res[i] = x
        target[i] = y

    return res, target

def run_configuration(std_for_init, forget_bias, num_hidden, b_size, max_sequences, verbose):

  with tf.Graph().as_default(), tf.Session() as session:
    config = BaseConfig()
    config.forget_bias = forget_bias
    config.n_hidden = num_hidden
    config.batch_size = b_size

    T = 50
    config.num_steps = T

    # testConfig = TestConfig()
    # initializer = tf.random_normal_initializer(0.0, std_for_init, None)

    initializer = tf.truncated_normal_initializer(0.0, std_for_init, None)
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
    success_inarow_2 = 0
    i_state = state = model.initial_state.eval()

    # k = tf.trainable_variables()
    # avg_anive = 0.0
    total_success = 1.0
    total_fails = 1.0

    total_success_2 = 1.0
    total_fails_2 = 1.0

    max = 0
    for i in range(config.max_epoch):
        x, targets = get_batch_data(config.batch_size, T)
        cost, state, pred, _ = session.run([model.cost, model.final_state, model.pred, model.train_op],
                                   {model.targets: targets,
                                    model.input_data : x,
                                    # model.initial_state: state})
                                     model.initial_state: i_state})
        total_seq+= config.batch_size
        res = pred[:,0] - targets[:,0]
        # avg_anive+= np.mean(np.square(res - 0.5))
        res = np.zeros(config.batch_size)
        ind = np.argmax(pred, 1)
        for k in range(config.batch_size):
            res[k] = (targets[k, ind[k]] == 1)

        d1 = np.abs(pred - targets) <= 0.3
        ds = np.sum(d1, 1)

        diff = (ds >= 4.0)
        success_2 = np.sum(diff)

        success_count = np.sum(res)
        total_success+= success_count
        total_fails+= (config.batch_size - success_count)

        total_success_2+= success_2
        total_fails_2+= (config.batch_size - success_2)

        if np.sum(success_count) >= config.batch_size:
            success_inarow += config.batch_size
        else:
            success_inarow = 0

        if np.sum(success_2) >= config.batch_size:
            success_inarow_2 += config.batch_size
        else:
            success_inarow_2 = 0


        if total_seq >= max_sequences:
            break

        if success_inarow > max:
            max = success_inarow

        if success_inarow >= 2000:
           print ("Success num_seq", total_seq)
           break

        if verbose and (total_seq % 1000 == 0):
            print ("cost", cost,  "ts", total_seq, "s_pcg", total_success / total_fails, "pcg_2", total_success_2 / total_fails_2, "s_in_row", success_inarow, "s_in_row_2", success_inarow_2, "max", max)
            total_fails = 1.0
            total_success = 1.0
            total_fails_2 = 1.0
            total_success_2 = 1.0

    return total_success / total_fails

def main(unused_args):

    n_hidden = 3
    forget_bias = 0.6
    max_sequence = 3000000
    batch_size = 5
    std = 0.5
    print ("start ", n_hidden)
    run_configuration(std , forget_bias, n_hidden, batch_size, max_sequence, True)
    # for f in  np.arange(2.0, 0.1, -0.2, float):
    #     for s in np.arange(1.5, 0.4, -0.5, dfloat):
    #         max, cost = run_configuration(s, f, n_hidden, max_sequence)
    #         print ("std:", s, "fb", f, "cost", cost, "max", max)
    # for j in range(3):
    #     for i in range(2, 5, 1):
    #         for b in range(2,11,5):
    #             max, cost, naive = run_configuration(1.0, 1.2, i,b, max_sequence)
    #             print ("hidden:", i, "b", b, "cost", cost, "naive", naive, "max", max)

    # for i in range(6):
    #     for f in  np.arange(2.8, 2.9, 0.3, float):
    #         max, cost, ncost = run_configuration(0.3, f, 4, 10, 800000, False)
    #         print ("i", i, "fb", f, "cost", cost, "max", max, "ncost", ncost, "bs", 10)
            # max, cost, ncost = run_configuration(0.3, f, 4, 20, 400000, False)
            # print ("fb", f, "cost", cost, "max", max, "ncost", ncost, "bs", 20)

if __name__ == "__main__":
  tf.app.run()