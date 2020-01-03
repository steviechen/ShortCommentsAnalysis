from sqlalchemy.engine import create_engine
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta

class NPSOracleEngine(object):
    def __init__(self):
        self.DIALECT = 'oracle'
        self.SQL_DRIVER = 'cx_oracle'
        self.USERNAME = 'stapuser' #enter your username
        self.PASSWORD = 'se#0stpdb' #enter your password
        self.HOST = '10.252.8.134' #enter the oracle db host url
        self.PORT = 1909 # enter the oracle port number
        self.SID='stapdb1'
        # self.SERVICE = 'your_oracle_service_name' # enter the oracle db service name
        self.ENGINE_PATH_WIN_AUTH = self.DIALECT + '+' + self.SQL_DRIVER + '://' + self.USERNAME + ':' + self.PASSWORD +'@' + self.HOST + ':' + str(self.PORT) + '/' + self.SID

        self.engine = create_engine(self.ENGINE_PATH_WIN_AUTH)
        # self.currdate = dt.datetime.now().strftime('%F')

        self.oracle_sql = '''
        SELECT 
            nps.DATETIME as datetime,
            nps.SITEID as siteid,
            CONCAT(nps.sitename ,'.webex.com') as siteurl,
            CASE WHEN nps.sitename LIKE '%.my'
                    THEN 'MC-ONline Site'
                    ELSE 'Enterprice Site'
                END as sitetype,
            nps.relversion as relversion,
            version_info.CLIENTVER as cliversion,
            nps.domainname as domainname,
            CASE WHEN min(cast(nps.nps_score as INT))>10
                    THEN NULL
                 WHEN min(cast(nps.nps_score as INT))<0
                    THEN NULL
                    ELSE min(cast(nps.nps_score as INT))
                END as nps_score,
            max(nps.stars_score) as stars_score,
            max(nvl(nps.comments, '')) as comments,
            max(nvl(nps.join_issue, '')) as join_issue,
            max(nvl(nps.audio_issue, '')) as audio_issue,
            max(nvl(nps.video_issue, '')) as video_issue,
            max(nvl(nps.sharing_issue, '')) as sharing_issue
        FROM STAPUSER.STAP_NPS_SCORE_RAW nps left join 
             STAPUSER.Site_pageclient_version version_info on (nps.siteid = to_char(version_info.siteid))
        group by 
             nps.DATETIME, nps.siteid, nps.sitename, nps.relversion, 
             version_info.CLIENTVER, nps.domainname
        '''
        self.oracle_sql_period = '''
                SELECT 
                    nps.DATETIME as datetime,
                    nps.SITEID as siteid,
                    CONCAT(nps.sitename ,'.webex.com') as siteurl,
                    CASE WHEN nps.sitename LIKE '%.my'
                            THEN 'MC-ONline Site'
                            ELSE 'Enterprice Site'
                        END as sitetype,
                    nps.relversion as relversion,
                    version_info.CLIENTVER as cliversion,
                    nps.domainname as domainname,
                    CASE WHEN min(cast(nps.nps_score as INT))>10
                            THEN NULL
                         WHEN min(cast(nps.nps_score as INT))<0
                            THEN NULL
                            ELSE min(cast(nps.nps_score as INT))
                        END as nps_score,
                    max(nps.stars_score) as stars_score,
                    max(nvl(nps.comments, '')) as comments,
                    max(nvl(nps.join_issue, '')) as join_issue,
                    max(nvl(nps.audio_issue, '')) as audio_issue,
                    max(nvl(nps.video_issue, '')) as video_issue,
                    max(nvl(nps.sharing_issue, '')) as sharing_issue
                FROM STAPUSER.STAP_NPS_SCORE_RAW nps left join 
                     STAPUSER.Site_pageclient_version version_info on (nps.siteid = to_char(version_info.siteid))
                WHERE
                     (nps.DATETIME>= '{0}' ) and (nps.DATETIME< '{1}' )
                group by 
                     nps.DATETIME, nps.siteid, nps.sitename, nps.relversion, 
                     version_info.CLIENTVER, nps.domainname
                '''

    def queryFromOracle(self, startDate, endDate):
        return pd.read_sql_query(self.oracle_sql_period.format(startDate, endDate), self.engine)

    def queryAll(self):
        return pd.read_sql_query(self.oracle_sql, self.engine)

    def queryNMonth(self, month):
        curr_date = dt.datetime.now().strftime('%F')
        n_month_ago = (dt.datetime.now() - relativedelta(months=month)).strftime('%F')
        return self.queryFromOracle(n_month_ago, curr_date)