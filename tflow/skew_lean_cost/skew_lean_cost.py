import tensorflow as tf
import numpy as np
import reader as reader
import pymssql
from datetime import date, timedelta, datetime
class NNModel(object):

  def __init__(self, is_training, config):
      num_features = config.num_features

      batch_size = config.batch_size
      num_hidden = config.num_hidden
      self._cost_coeff = tf.placeholder(tf.float32, (batch_size, 1))
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
      self._cost  = tf.reduce_mean(tf.square(pred[:,0] - self._targets[:,0]) * self._cost_coeff)

      regularization_param = 0.1

      l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
      self._cost = self._cost +  regularization_param * l2_loss
      cost = self._cost

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

  @property
  def cost_coeff(self):
    return self._cost_coeff

def run_epoch(session, data_reader, normalizer, model, config, train_op, mean_out):
    naive_cost = 0
    total_seq = 0
    dynamic_cost = 0
    total_batches = 0
    sum_gaps = 0
    for step, (x, y) in enumerate(data_reader.date_iterator()):

          nx = normalizer.normalize(x)
          ny = normalizer.normalize_out(y)
          cost_coeff = np.reshape(x[:,12], (config.batch_size,1))
          cost, pred, result, _ = session.run([model.cost, model.pred, model.result, train_op],
                                       {model.targets: ny,
                                        model.cost_coeff : cost_coeff,
                                        model.input_data : nx})
          total_batches+=1
          total_seq+=config.batch_size
          uny = normalizer.de_normalize_out(pred)
          # sqrtt = np.reshape(x[:,2], (config.batch_size,1))

          un_norm =   np.reshape(x[:,11], (config.batch_size,1))
          gap = np.abs((uny - y) * un_norm)
          # gap = np.abs((uny - y))

          sum_gaps+= np.sum(gap)
          dynamic_cost+= np.sum(np.square(gap)) #undo the sqrtt mult
          naive_cost+= np.sum(np.square((y - mean_out)))
    return dynamic_cost / naive_cost , sum_gaps, total_seq

def run_configuration(from_date, to_date, out_from_date, out_to_date):

  with tf.Graph().as_default(), tf.Session() as session:
    config = BaseConfig()
    std = 1.0 / np.sqrt(config.num_hidden)
    # std = 0.01
    initializer = tf.random_normal_initializer(0.0, std, None)
    max_epoch = 200
    limit = 500
    data_reader_in = reader.DataReader(config.batch_size, from_date, to_date)
    data_reader_out = reader.DataReader(config.batch_size, out_from_date, out_to_date, limit)

    with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = NNModel(True, config)

    tf.initialize_all_variables().run()
    best_out = 2.0
    best_sum_out = 1e+6
    in_rss = 2.0
    best_iteration = 0

    # cost, naive_cost, total_seq = run_epoch(session, data_reader_in, data_reader_in, model, config, tf.no_op())
    # t_cost, t_naive_cost, t_total_seq = run_epoch(session, data_reader_out, data_reader_in, model, config, tf.no_op())
    # print ("first is ", (cost/naive_cost), "first test", (t_cost/t_naive_cost))

    for i in range(max_epoch):
        in_rss, sum_in, total_seq = run_epoch(session, data_reader_in, data_reader_in, model, config, model.train_op, data_reader_in.mean_out())
        out_rss, sum_out, total_seq_out = run_epoch(session, data_reader_out, data_reader_in, model, config, tf.no_op(), data_reader_in.mean_out())

        # if i % 10 == 0:
        #     print ("in", (cost/naive_cost), "out", (t_cost/t_naive_cost), "i", i)

        if (sum_out < best_sum_out):
            best_out = out_rss
            best_sum_out = sum_out
            best_iteration = i
                # print ("test model ", cost, "ncost ", naive_cost, "relative", cost / naive_cost, "i", i)
    last_out, last_sum_out, total_seq_out = run_epoch(session, data_reader_out, data_reader_in, model, config, tf.no_op(), data_reader_in.mean_out())

    print ("best_iteration ", best_iteration, "total_sql", total_seq_out)
    return best_out, best_sum_out, last_out,last_sum_out, in_rss, total_seq_out

def get_min_date():
    conn = pymssql.connect("192.168.122.200", "sa","a:123456a:123456" , "Goldfish")
    cursor = conn.cursor(as_dict=True)
    cursor.execute('SELECT top 1 ExeDate FROM ExePairs WHERE BidAskWidthMultOpen is not null order by ExeDate')
    for row in cursor:
        return row[u'ExeDate']

    return datetime.min

def main(unused_args):

    days_in = 30
    days_out = 20
    from_date = get_min_date()

    while from_date < datetime.now():
        to_date = from_date + timedelta(days=days_in)
        from_out = to_date + timedelta(hours=1)
        to_out = from_out + timedelta(days=days_out)
        best_out, best_sum_out, last_out, last_sum_out, last_in, count_out_samples = run_configuration(from_date, to_date, from_out, to_out)
        print ("best_sum  ", best_sum_out, "  last_sum_out  ", last_sum_out, " last_out ", last_out, "  best_out  ", best_out, "in_rss", last_in, "count_out", count_out_samples, from_out, to_out)
        from_date+= timedelta(days=days_out)

class BaseConfig(object):
    batch_size = 500
    num_features = 13
    num_hidden = 3
    num_layers = 1
    learning_rate = 0.2

if __name__ == "__main__":
  tf.app.run()