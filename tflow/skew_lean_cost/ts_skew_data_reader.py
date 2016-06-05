import pymssql
import numpy as np
from datetime import timedelta, datetime

class DataReader(object):

    def __init__(self, from_date, to_date, limit = -1):
        self.raw_data = []
        conn = pymssql.connect("192.168.122.200", "sa","a:123456a:123456" , "Goldfish")
        cursor = conn.cursor(as_dict=True)

        if limit < 0:
            sql = 'SELECT  * FROM ExePairs WHERE ExeDate>%s  and ExeDate < %s and IvPnlOpen is not null and ExpiryMonthOpen = ExpiryMonthClose and IvPnlClose is not null  and BuyOpen != BuyClose and BidAskWidthMultOpen is not null order by ExeDate'
        else:
            sql = 'SELECT  top ' +  str(2*limit) + ' * FROM ExePairs WHERE ExeDate>%s  and ExeDate < %s and Pnl is not null and ExpiryMonthOpen = ExpiryMonthClose and IvPnlClose is not null  and BuyOpen != BuyClose and BidAskWidthMultOpen is not null order by ExeDate'
        cursor.execute(sql, (from_date, to_date))
        total_rows = 0
        raw_data = []
        raw_targets = []
        our_pnl = []
        naive_avg = 0.0
        current_date = None
        dt = 0

        for row in cursor:
            if total_rows == limit:
                break
            if current_date != None:
                dt = (row[u'ExeDate'] - current_date).microseconds / (1000.0 * 60.0 * 60.0 * 9.0 * 365.0)

            current_date = row[u'ExeDate']
            total_rows+=1
            sqrtt = np.sqrt(row[u'ExpiryOpen'] / 365.0)

            # b_s_mult = 1.0 if row[u'BuyOpen'] else -1.0
            b_s_mult = 1.0
            executed_vega = row[u'VegaOpen'] * (row[u'ExecutedDelta'] / np.abs(row[u'DeltaOpen'])) * b_s_mult

            raw_data.append((dt, executed_vega, row[u'BuyOpen'], row[u'IsQuote']))

            raw_targets.append(sqrtt * row[u'IvPnlOpen'] * 100.0 * b_s_mult)
            our_pnl.append(sqrtt * row[u'IvPnlOpen'] * 100.0)

            naive_avg+= (sqrtt * row[u'IvPnlOpen'] * 100.0)

        if total_rows > 0:
            naive_avg/= max(total_rows, 1)
            self._total_rows = total_rows
            self._raw_data = np.asarray(raw_data, float)
            self._raw_targets = np.asarray(raw_targets, float)
            self._our_pnl = np.asarray(our_pnl, float)
            std = np.std(raw_data, axis=0, dtype=float)
            mean = np.mean(raw_data, axis=0, dtype=float)

            self._std_in = std
            self._mean_in = mean
            self.naive_avg = naive_avg

        conn.close()

        print ("finished totalRows", total_rows)


    def get_avg_prod_cost(self, naive_avg):
        return np.mean(np.square(self._our_pnl - naive_avg))

    def get_total_rows(self):
        return self._total_rows

    def get_raw_data(self):
        return self._raw_data

    def get_raw_targets(self):
        return self._raw_targets

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
        raw_data = self.get_raw_data()
        raw_targets = self.get_raw_targets()
        data_len = len(raw_data)
        batch_len = data_len // batch_size
        num_features = 4
        data = np.zeros([batch_size, batch_len, num_features], dtype=np.float32)
        y_data = np.zeros([batch_size, batch_len], dtype=np.float32)
        for i in range(batch_size):
            data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
            y_data[i] = raw_targets[batch_len * i:batch_len * (i + 1)]

        epoch_size = batch_len // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i*num_steps:(i+1)*num_steps]
            y = y_data[:, (i*num_steps + 1) :((i+1)*num_steps + 1)]
            yield (x, y)