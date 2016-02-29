import pymssql
import numpy as np

class DataReader(object):

    def __init__(self, batch_size, from_date, to_date):
        self.raw_data = []
        conn = pymssql.connect("192.168.122.200", "sa","a:123456a:123456" , "Goldfish")
        cursor = conn.cursor(as_dict=True)
        num_features = 11
        max_rows = 85000
        cursor.execute('SELECT top 85000 * FROM ExePairs WHERE ExeDate>%s  and ExeDate < %s and IvPnlOpen is not null and IvPnlClose is not null  and BuyOpen != BuyClose order by ExeDate', (from_date, to_date))
        next_batch = np.zeros((batch_size, num_features), float)
        next_targets = np.zeros((batch_size, 1), float)
        ind = 0
        totalRows = 0

        raw_data = np.zeros((max_rows, num_features), float)
        raw_output = np.zeros((max_rows, 1), float)
        raw_ind = 0
        for row in cursor:
            totalRows+=1
            next_batch[ind, 0] = row[u'Kasasab']
            next_batch[ind, 1] = row[u'BidAskWidthMultOpen']
            sqrtt = np.sqrt(row[u'ExpiryOpen'] / 365.0)
            next_batch[ind, 2] = sqrtt
            next_batch[ind, 3] = self.diff_in_iv_point(row, True)
            next_batch[ind, 4] = self.diff_in_iv_point(row, False)
            next_batch[ind, 5] = self.distance(row)
            next_batch[ind, 6] = np.abs(row[u'VegaOpen'])
            next_batch[ind, 7] = np.abs(row[u'VegaClose'])
            next_batch[ind, 8] = np.abs(row[u'ExecutedDelta'])
            next_batch[ind, 9] = np.abs(row[u'VegaOpen']/row[u'DeltaOpen'])
            next_batch[ind, 10] = np.abs(row[u'VegaClose']/row[u'DeltaClose'])

            # next_targets[ind ,0] = 100.0 * (row[u'IvPnlOpen'] + row[u'IvPnlClose'])
            next_targets[ind ,0] = row[u'Pnl']
            raw_data[raw_ind] = next_batch[ind]
            raw_output[raw_ind] = next_targets[ind ,0]
            ind+=1
            raw_ind+=1
            if ind == batch_size:
                self.raw_data.append((next_batch, next_targets))
                next_batch = np.zeros((batch_size, num_features), float)
                next_targets = np.zeros((batch_size, 1), float)
                ind = 0

        raw_data = raw_data[0:totalRows, :]
        raw_output = raw_output[0:totalRows, :]
        std = np.std(raw_data, axis=0, dtype=float)
        mean = np.mean(raw_data, axis=0, dtype=float)

        std_out = np.std(raw_output)
        mean_out = np.mean(raw_output)

        self._std_in = std
        self._mean_in = mean
        self._std_out = std_out
        self._mean_out = mean_out

        conn.close()

        print ("finished totalRows", totalRows, "std_out", std_out, "mean_out", mean_out, "std_in", std, "mean_in", mean)


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
        return (batch_in - self._mean_in) / np.max(self._std_in, 1e-12)

    def normalize_out(self, out):
        return (out - self._mean_out) / np.max(self._std_out, 1e-8)

    def de_normalize_out(self, pred):
        return (pred * np.max(self._std_out, 1e-8)) + self._mean_out