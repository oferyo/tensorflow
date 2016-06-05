import tensorflow as tf
import numpy as np
import ts_skew_data_reader as skew_data_reader
from tensorflow.models.rnn import rnn, rnn_cell

import pymssql
from datetime import timedelta, datetime

class RNNModel(object):
  """The RNN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    size = config.n_hidden
    num_steps = config.num_steps
    self._input_data = tf.placeholder(tf.float32, (batch_size, config.num_steps, config.num_features))
    self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    # weights_hidden = tf.contant(1.0, shape= [config.num_features, config.n_hidden])
    weights_hidden = tf.get_variable("weights_hidden", [config.num_features, config.n_hidden])

    #inputs of shape [bs, num_hidden] - input length is num_steps
    inputs = []
    for k in range(num_steps):
        nextitem = tf.matmul(self._input_data[:, k, :] , weights_hidden)
        inputs.append(nextitem)

    outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)

    #tf.concat(1, outputs) is shape of [bs, ns*h_size]
    # output = tf.reshape(tf.concat(1, outputs), [-1, size]) is shape of [bs * ns, hs]

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    pred = tf.matmul(output, tf.get_variable("weights_out", [config.n_hidden,1])) + tf.get_variable("bias_out", [1])
    self._pred = pred

    self._final_state = states[-1]
    self._cost = cost  = tf.reduce_mean(tf.square(pred - tf.reshape(self.targets, (config.num_steps * config.batch_size, 1))))

    if not is_training:
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
    batch_size = 4
    max_epoch = 100000
    init_scale = 0.1
    n_hidden = 6
    learning_rate = 0.09
    num_layers = 1
    num_features = 4
    max_grad_norm = 5
    lr_decay = 0.5
    num_steps = 500
    is_training = True

class TestConfig(BaseConfig):
    batch_size = 1
    num_steps = 1
    is_training = False

def run_epoch(session, data_reader, model, config, train_op, naive_guess):
    avg_cost = 0
    prod_cost = 0

    for step, (x, y) in enumerate(data_reader.ts_iterator(config.batch_size, config.num_steps)):

        # sqrtt = x[:,:, 0]
        naive_pred = naive_guess
        cost, pred, _ = session.run([model.cost, model.pred, train_op],
                                       {model.targets: y,
                                        model.input_data : x})
        avg_cost+=cost
        prod_cost+=data_reader.get_avg_prod_cost(naive_guess)

    return avg_cost, prod_cost


def run_configuration(from_date, to_date, out_from_date, out_to_date):

  with tf.Graph().as_default(), tf.Session() as session:
    config = BaseConfig()
    std = 0.5 / np.sqrt(config.n_hidden)
    # std = 0.01
    initializer = tf.random_normal_initializer(0.0, std, None)
    max_epoch = 800
    limit = -1
    data_reader_in = skew_data_reader.DataReader(from_date, to_date)
    data_reader_out = skew_data_reader.DataReader(out_from_date, out_to_date, limit)

    min_rows = config.num_steps * config.batch_size
    if (data_reader_in.get_total_rows() < min_rows or data_reader_out.get_total_rows() < min_rows):
        return False, 0,0
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = RNNModel(True, config)

    tf.initialize_all_variables().run()

    naive_guess = data_reader_in.naive_avg
    for i in range(max_epoch):
        in_avg_rss, naive_rss = run_epoch(session, data_reader_in, model, config, model.train_op, naive_guess)
        out_avg_rss, naive__out_rss  = run_epoch(session, data_reader_out, model, config, tf.no_op(), naive_guess)

        if i % 10 == 0:
            print ("in", (in_avg_rss/naive_rss) , "out", (out_avg_rss/naive__out_rss) , "i", i)

    out_avg_rss, naive_rss = run_epoch(session, data_reader_out, model, config, tf.no_op(), naive_guess)

    print ("run_epoch ", (out_avg_rss / naive_rss))
    return True, out_avg_rss, naive_rss

def get_min_date():
    conn = pymssql.connect("192.168.122.200", "sa","a:123456a:123456" , "Goldfish")
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT top 1 ExeDate FROM ExePairs WHERE BidAskWidthMultOpen is not null order by ExeDate')
    for row in cursor:
        return row[u'ExeDate']

    return datetime.min

def main(unused_args):
    days_in = 15
    days_out = 10
    from_date = get_min_date()

    while from_date < datetime.now():
        to_date = from_date + timedelta(days=days_in)
        from_out = to_date + timedelta(hours=1)
        to_out = from_out + timedelta(days=days_out)
        success, rss, n_rss = run_configuration(from_date, to_date, from_out, to_out)
        if success:
            print ("rss_ratio  ", rss/n_rss, from_out, to_out)
        from_date+= timedelta(days=days_out)

if __name__ == "__main__":
  tf.app.run()