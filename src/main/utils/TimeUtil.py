import calendar
import time
import datetime
from datetime import timezone
from datetime import timedelta

class Timeutil(object):
    def getTimestamp(self, x: str) -> int:
        print("input time is " + x)
        return int(time.mktime(time.strptime(str(x), '%Y-%m-%d %H:%M:%S')))
        # return int(datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').timestamp())
        # return calendar.timegm(datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')..utctimetuple())
        # return datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        # return int(datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone(timedelta(hours=8))).timestamp())
        # return int(datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp())


    def getDateRange(self,dateStart:str,dateEnd:str) -> list:
        date_list = []
        begin_date = datetime.datetime.strptime(dateStart, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(dateEnd, "%Y-%m-%d")
        while begin_date <= end_date:
            date_str = begin_date.strftime("%Y-%m-%d")
            date_list.append(date_str)
            begin_date += datetime.timedelta(days=1)
        return date_list

    def getYesterday(self) -> str:
        yesterday = datetime.datetime.today() + timedelta(-1)
        return str(yesterday.strftime('%Y-%m-%d'))


if __name__ == '__main__':
    timeUtil = Timeutil()
    print(timeUtil.getDateRange('2019-07-28','2019-07-30'))
