import time
class LogUtil(object):
    def setPrintInfo(self,info):
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(str(now) + ' ' + str(info))