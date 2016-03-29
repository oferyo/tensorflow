import numpy as np

def ts_iterator(raw_data, target_data, batch_size, num_steps):
  """Iterate on the raw data.

  Args:
    raw_data: np array of type float32
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.float32)
  y_data = np.zeros([batch_size, batch_len], dtype=np.float32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
    y_data[i] = target_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = batch_len // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = y_data[:, i*num_steps:(i+1)*num_steps]
    yield (x, y)
