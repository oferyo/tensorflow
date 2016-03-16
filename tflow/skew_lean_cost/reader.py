import pymssql
import numpy as np

class DataReader(object):

    def __init__(self, batch_size, from_date, to_date, limit = -1):
        self.raw_data = []
        conn = pymssql.connect("192.168.122.200", "sa","a:123456a:123456" , "Goldfish")
        cursor = conn.cursor(as_dict=True)
        num_features = 13
        max_rows = 300000
        if limit < 0:
            sql = 'SELECT  * FROM ExePairs WHERE ExeDate>%s  and ExeDate < %s and Pnl is not null and ExpiryMonthOpen = ExpiryMonthClose and IvPnlClose is not null  and BuyOpen != BuyClose and BidAskWidthMultOpen is not null order by ExeDate'
        else:
            sql = 'SELECT  top ' +  str(2*limit) + ' * FROM ExePairs WHERE ExeDate>%s  and ExeDate < %s and Pnl is not null and ExpiryMonthOpen = ExpiryMonthClose and IvPnlClose is not null  and BuyOpen != BuyClose and BidAskWidthMultOpen is not null order by ExeDate'
        cursor.execute(sql, (from_date, to_date))
        next_batch = np.zeros((batch_size, num_features), float)
        next_targets = np.zeros((batch_size, 1), float)
        ind = 0
        totalRows = 0

        raw_data = np.zeros((max_rows, num_features), float)
        raw_output = np.zeros((max_rows, 1), float)
        raw_ind = 0
        for row in cursor:
            if totalRows == limit:
                break
            next_batch[ind, 0] = row[u'Kasasab']
            next_batch[ind, 1] = row[u'BidAskWidthMultOpen']
            sqrtt = np.sqrt(row[u'ExpiryOpen'] / 365.0)
            distance = self.distance(row)
            next_batch[ind, 2] = sqrtt
            next_batch[ind, 3] = self.diff_in_iv_point(row, True)
            next_batch[ind, 4] = self.diff_in_iv_point(row, False)
            next_batch[ind, 5] = distance
            next_batch[ind, 6] = np.abs(row[u'VegaOpen'])
            next_batch[ind, 7] = np.abs(row[u'VegaClose'])
            #next_batch[ind, 8] = np.abs(row[u'ExecutedDelta'])
            # next_batch[ind, 8] = np.abs(row[u'VegaOpen']/)
            vodo = np.abs(row[u'VegaOpen']) / np.abs(row[u'DeltaOpen'])
            vcdc = np.abs(row[u'VegaClose']) / np.abs(row[u'DeltaClose'])
            next_batch[ind, 8] = vcdc
            next_batch[ind, 9] = vodo
            # next_batch[ind, 11] = np.abs(row[u'DeltaClose'])

            next_batch[ind, 10] = row[u'BidAskWidthMultClose']
            # next_batch[ind, 11] = row[u'ExecutedDelta']
            hx = np.sqrt((np.abs(vodo-vcdc) ** 2) + (distance**2)*(vcdc**2))
            un_norm = max(hx * np.abs(row[u'ExecutedDelta']), 1e-8)
            if un_norm < 1e-6:
                continue
            totalRows+=1
            next_batch[ind, 11] = un_norm
            next_batch[ind, 12] = (hx**2) * (row[u'ExecutedDelta'] **2)
            next_targets[ind ,0] = row[u'Pnl'] / un_norm
            # next_targets[ind ,0] = row[u'Pnl']
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

        print ("finished totalRows", totalRows, "std_out", std_out, "mean_out", mean_out)


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

    def normalize_out(self, out):
        return (out - self._mean_out) / self._std_out
        # return out

    def de_normalize_out(self, pred):
        return (pred * self._std_out) + self._mean_out
        # return pred

    def mean_out(self):
        return self._mean_out;