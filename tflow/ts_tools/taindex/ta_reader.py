import pymssql
from datetime import datetime

class TAReader(object):

    def __init__(self, from_date, to_date, forecast_step):
        self.raw_data = []
        conn = pymssql.connect("192.168.122.200", "sa","a:123456a:123456" , "Goldfish")
        cursor = conn.cursor(as_dict=True)

        sql = 'SELECT  * FROM [Goldfish].[dbo].[Index] WHERE Time>%s  and Time < %s order by Time'
        cursor.execute(sql, (from_date, to_date))

        total_rows = 0
        raw_data = []
        index_return = []
        future_return = []
        first_month = -1
        for row in cursor:
            if self.in_trading_hours(row[u'Time']):
                if first_month == -1:
                    first_month = row[u'Month']
                raw_data.append(row[u'Index'])
                total_rows+=1

                # if first_month != -1 and row[u'Month'] != first_month: #not sure if we want to mix months
                #     break
                #should we check cross days?


        for j in range(total_rows - forecast_step - 1):
            index_return.append(raw_data[j+1]/ raw_data[j] - 1.0)
            future_return.append(raw_data[j+forecast_step]/ raw_data[j] - 1.0)

        self.index_returns = index_return
        self.future_returns = future_return
        conn.close()

        print ("TAReader finished totalRows", total_rows)


    def get_raw_returns(self):
        return self.index_returns

    def get_future_returns(self):
        return self.future_returns

    def in_trading_hours(self, time):
        start = datetime(time.year, time.month, time.day, 9, 31, 0, 0)
        end_hour = 17
        if time.weekday() ==  6: #Sunday
            end_hour = 16

        end = datetime(time.year, time.month, time.day, end_hour, 30, 0, 0)

        return time >= start and time <= end

