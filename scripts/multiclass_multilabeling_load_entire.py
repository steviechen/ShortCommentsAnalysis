import datetime as dt
from dateutil.relativedelta import relativedelta
from sqlalchemy.engine import create_engine
from sqlalchemy import types

import pandas as pd
import numpy as np

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

# proj_dir = '/data/part1/jupyter_project/StevieChen/OPS_Team/Logex/'
proj_dir = '/home/wbxbuilds/nps_analysis/data/'

final_df = pd.read_csv(proj_dir + 'output/df/final_clean.csv', encoding='latin-1').iloc[:,[i for i in range(1,17)]]

final_df["comments"] = final_df["comments"].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii'))
final_df["join_issue"] = final_df["join_issue"].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii'))
final_df["audio_issue"] = final_df["audio_issue"].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii'))
final_df["video_issue"] = final_df["video_issue"].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii'))
final_df["sharing_issue"] = final_df["sharing_issue"].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii'))
final_df["other_issue"] = final_df["other_issue"].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii'))
final_df["comm_wl"] = final_df["comm_wl"].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii'))
print(final_df[100:110])
final_df.to_sql('STAP_NPS_COMMENTS_ALL', engine, if_exists='append', chunksize = 100, index=False)

