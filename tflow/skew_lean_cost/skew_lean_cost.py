import tensorflow as tf
import numpy as np
import reader as reader
from datetime import date, timedelta, datetime
class NNModel(object):

  def __init__(self, is_training, config):
      num_features = config.num_features

      batch_size = config.batch_size
      num_hidden = config.num_hidden
      self._input_data = tf.placeholder(tf.float32, (batch_size, num_features))
      self._targets = tf.placeholder(tf.float32, [batch_size, 1])

      weights_hidden = tf.Variable(tf.ones([num_features, num_hidden]))
      b_hidden = tf.Variable(tf.zeros([1, num_hidden]))
      weights_out = tf.Variable(tf.zeros([num_hidden, 1]))
      b_out = tf.Variable(tf.zeros([1]))
      self.b_out = b_out

      hidden_out = tf.nn.tanh(tf.matmul(self._input_data, weights_hidden) + b_hidden)

      pred = b_out + tf.matmul(hidden_out, weights_out)
      self._pred = pred

      self._result = pred - self._targets
      self._cost = cost = tf.reduce_mean(tf.square(pred[:,0] - self._targets[:,0]))
      self._train_op = tf.no_op()

      if is_training:
          # optimizer = tf.train.GradientDescentOptimizer(learning_rate = config.learning_rate).minimize(cost)
          optimizer = tf.train.AdamOptimizer().minimize(cost)
          self._train_op = optimizer


  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def train_op(self):
    return self._train_op

  @property
  def result(self):
    return self._result

  @property
  def cost(self):
    return self._cost

  @property
  def pred(self):
    return self._pred

def run_epoch(session, data_reader, normalizer, model, config, train_op):
    naive_cost = 0
    total_seq = 0
    dynamic_cost = 0
    total_batches = 0
    for step, (x, y) in enumerate(data_reader.date_iterator()):

          nx = normalizer.normalize(x)
          ny = normalizer.normalize_out(y)
          cost, pred, result, _ = session.run([model.cost, model.pred, model.result, train_op],
                                       {model.targets: ny,
                                        model.input_data : nx})
          total_batches+=1
          total_seq+=config.batch_size
          uny = normalizer.de_normalize_out(pred)
          dynamic_cost+= np.sum(np.square(uny - y)) #undo the sqrtt mult
          naive_cost+= np.sum(np.square(y))
    return dynamic_cost / total_seq, naive_cost / total_seq


def main(unused_args):

  with tf.Graph().as_default(), tf.Session() as session:
    config = BaseConfig()
    std = 1.0 / np.sqrt(config.num_hidden)
    initializer = tf.random_normal_initializer(0.0, std, None)
    max_epoch = 10000

    from_date = datetime.now() - timedelta(days=30)
    to_date = datetime.now() - timedelta(days=1)

    days = 10
    out_from_date = datetime.now() - timedelta(days=39)
    out_to_date = datetime.now() - timedelta(days=31)
    data_reader = reader.DataReader(config.batch_size, from_date, to_date)
    data_reader_out = reader.DataReader(config.batch_size, out_from_date, out_to_date)

    with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = NNModel(True, config)

    tf.initialize_all_variables().run()
    for i in range(max_epoch):
        cost, naive_cost = run_epoch(session, data_reader, data_reader, model, config, model.train_op)
        if i % 50 == 0:
            print ("cost", cost, "naivecost", naive_cost, "relative_cost", (cost/naive_cost), "i", i)

        if i % 50 ==0:
            cost, naive_cost = run_epoch(session, data_reader_out, data_reader, model, config, tf.no_op())
            if (cost < 1.0):
                print ("test model ", cost, "ncost ", naive_cost, "relative", cost / naive_cost, "i", i)


    cost, naive_cost = run_epoch(session, data_reader_out, data_reader, model, config, tf.no_op())
    print ("end test model ", cost, "ncost ", naive_cost, "relative", cost / naive_cost)


class BaseConfig(object):
    batch_size = 500
    num_features = 11
    num_hidden = 3
    num_layers = 1
    learning_rate = 0.2

if __name__ == "__main__":
  tf.app.run()