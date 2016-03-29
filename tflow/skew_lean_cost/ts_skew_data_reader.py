import pymssql
import numpy as np


class DataReader(object):

    def __init__(self, from_date, to_date, limit = -1):
        self.raw_data = []
        conn = pymssql.connect("192.168.122.200", "sa","a:123456a:123456" , "Goldfish")
        cursor = conn.cursor(as_dict=True)

        if limit < 0:
            sql = 'SELECT  * FROM ExePairs WHERE ExeDate>%s  and ExeDate < %s and Pnl is not null and ExpiryMonthOpen = ExpiryMonthClose and IvPnlClose is not null  and BuyOpen != BuyClose and BidAskWidthMultOpen is not null order by ExeDate'
        else:
            sql = 'SELECT  top ' +  str(2*limit) + ' * FROM ExePairs WHERE ExeDate>%s  and ExeDate < %s and Pnl is not null and ExpiryMonthOpen = ExpiryMonthClose and IvPnlClose is not null  and BuyOpen != BuyClose and BidAskWidthMultOpen is not null order by ExeDate'
        cursor.execute(sql, (from_date, to_date))
        total_rows = 0
        raw_data = []
        naive_avg = 0.0
        raw_ind = 0
        for row in cursor:
            if total_rows == limit:
                break
            total_rows+=1
            sqrtt = np.sqrt(row[u'ExpiryOpen'] / 365.0)
            raw_data.append((sqrtt, row[u'IvPnlOpen'] * 100.0))
            naive_avg+= (sqrtt * row[u'IvPnlOpen'] * 100.0)

        naive_avg/= max(total_rows, 1)
        raw_data = np.asarray(raw_data, float)
        self.raw_data = raw_data
        std = np.std(raw_data, axis=0, dtype=float)
        mean = np.mean(raw_data, axis=0, dtype=float)

        self._std_in = std
        self._mean_in = mean
        self.naive_avg = naive_avg

        conn.close()

        print ("finished totalRows", total_rows)

    def get_raw_data(self):
        return self.raw_data

    def date_iterator(self):
        len_data = len(self.raw_data)
        for i in range(len_data):
            yield self.raw_data[i]

    def diff_in_iv_point_details(self, strike, index, sh, expiry):
        return (strike - index) / (sh * np.sqrt(expiry / 365.0) * index)

    def diff_in_iv_point(self, row, open):
        if open:
            return self.diff_in_iv_point_details(row[u'StrikeOpen'], row[u'IndexOpen'], row[u'SkewHeightOpen'], row[u'ExpiryOpen'])
        return self.diff_in_iv_point_details(row[u'StrikeClose'], row[u'IndexClose'], row[u'SkewHeightClose'], row[u'ExpiryClose'])

    def distance(self, row):
        return self.diff_in_iv_point(row, True) - self.diff_in_iv_point(row, False)

    def normalize(self, batch_in):
        return (batch_in - self._mean_in) / self._std_in

    def mean_out(self):
        return self._mean_out


    def ts_iterator(self, batch_size, num_steps):
        raw_data = self.get_raw_data();
        data_len = len(raw_data)
        batch_len = data_len // batch_size
        num_features = 2
        data = np.zeros([batch_size, batch_len, num_features], dtype=np.float32)
        for i in range(batch_size):
            data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

        epoch_size = batch_len // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i*num_steps:(i+1)*num_steps]
            y = data[:, (i*num_steps + 1) :((i+1)*num_steps + 1), 1]
            yield (x, y)