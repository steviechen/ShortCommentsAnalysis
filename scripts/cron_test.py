import pandas as pd

import datetime as dt
from dateutil.relativedelta import relativedelta
from sqlalchemy.engine import create_engine
from sqlalchemy import types

DIALECT = 'oracle'
SQL_DRIVER = 'cx_oracle'
USERNAME = 'stapuser' #enter your username
PASSWORD = 'se#0stpdb' #enter your password
HOST = '10.252.8.134' #enter the oracle db host url
PORT = 1909 # enter the oracle port number
SID='stapdb1'
# SERVICE = 'your_oracle_service_name' # enter the oracle db service name
ENGINE_PATH_WIN_AUTH = DIALECT + '+' + SQL_DRIVER + '://' + USERNAME + ':' + PASSWORD +'@' + HOST + ':' + str(PORT) + '/' + SID

engine = create_engine(ENGINE_PATH_WIN_AUTH)

pd.DataFrame([['001', 'rdxt', 0.342]], columns=['NPS_TOPIC_ID', 'COMM_ID', 'TOPIC_PERC_CONTRIB']).\
to_sql('STAP_NPS_TOPICS_COMMENTS_H', engine, if_exists='append', chunksize = 100, index=False)
