import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell
import ta_reader as tareader
import batch_reader as reader


import pymssql
from datetime import timedelta, datetime


class RNNModel(object):
  """The RNN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    size = config.n_hidden
    num_steps = config.num_steps
    self._input_data = tf.placeholder(tf.float32, (batch_size, config.num_features, config.num_steps))
    self._targets = tf.placeholder(tf.float32, [batch_size, num_steps])
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=1.0)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    # weights_hidden = tf.contant(1.0, shape= [config.num_features, config.n_hidden])
    weights_hidden = tf.get_variable("weights_hidden", [config.num_features, config.n_hidden])

    #inputs of shape [bs, num_hidden] - input length is num_steps
    inputs = []
    for k in range(num_steps):
        nextitem = tf.matmul(self._input_data[:, :, k] , weights_hidden)
        inputs.append(nextitem)

    outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)


    #tf.concat(1, outputs) is shape of [bs, ns*h_size]
    # output = tf.reshape(tf.concat(1, outputs), [-1, size]) is shape of [bs * ns, hs]

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    pred = tf.matmul(output, tf.get_variable("weights_out", [config.n_hidden,1])) + tf.get_variable("bias_out", [1])
    self._pred = pred

    self._final_state = states[-1]
    self._cost = cost  = tf.reduce_mean(tf.square(pred - tf.reshape(self.targets, (config.num_steps * config.batch_size, 1))))

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
    batch_size = 4
    max_epoch = 100000
    init_scale = 0.1
    n_hidden = 8
    learning_rate = 0.09
    num_layers = 1
    num_features = 1
    max_grad_norm = 5
    lr_decay = 0.5
    num_steps = 150
    is_training = True

class TestConfig(BaseConfig):
    batch_size = 1
    num_steps = 1
    is_training = False


def get_min_date():
    conn = pymssql.connect("192.168.122.200", "sa","a:123456a:123456" , "Goldfish")
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT top 1 Time FROM [Goldfish].[dbo].[Index] order by Time')
    for row in cursor:
        return row[u'Time']

    return datetime.min


def run_training_session(from_date, to_date):
    with tf.Graph().as_default(), tf.Session() as session:
        config = BaseConfig()
        # testConfig = TestConfig()
        initializer = tf.random_normal_initializer(0.0, 0.2, None)
        #initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        forecast_step = 60 * 5 # five minutes
        with tf.variable_scope("model", reuse=None, initializer=initializer):
          model = RNNModel(True, config)
        ta_reader = tareader.TAReader(from_date, to_date, forecast_step)

        tf.initialize_all_variables().run()

        raw_returns = ta_reader.get_raw_returns()
        future_returns = ta_reader.get_future_returns()

        print ("staring ", " from_date ", from_date, " to_date ", to_date)
        for i in range(config.max_epoch):
            t_cost = 0
            n_cost = 0
            state = model.initial_state.eval()
            total_patterns = 0
            for step, (x, y) in enumerate(reader.ts_iterator(raw_returns, future_returns, config.batch_size,
                                                                config.num_steps)):
                  cost, pred, state, _ = session.run([model.cost, model.pred, model.final_state, model.train_op],
                                               {model.targets: y,
                                                model.input_data : np.reshape(x,(config.batch_size, config.num_features, config.num_steps)),
                                                model.initial_state: state})
                  num_seq = config.batch_size * config.num_steps
                  total_patterns+= num_seq
                  n_cost+= np.sum(np.square(np.reshape(y, (config.batch_size * config.num_steps, 1))))

                  t_cost+=(cost*num_seq)
            if i % 5 == 0:
                print ("cost ", t_cost/n_cost , "i", i, "total_patterns", total_patterns)


def main(unused_args):
    days_in = 20
    days_out = 20
    from_date = get_min_date()


    while from_date < datetime.now():
        to_date = from_date + timedelta(days=days_in)
        from_out = to_date + timedelta(hours=1)
        to_out = from_out + timedelta(days=days_out)
        run_training_session(from_date, to_date)

        from_date+= timedelta(days=days_out)

if __name__ == "__main__":
  tf.app.run()