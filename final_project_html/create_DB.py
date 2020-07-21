import pymysql
import pandas as pd
from re import sub
import numpy as np

config = {
    'host': '127.0.0.1',
    'user': 'scott',
    'password': 'tiger',
    'database': 'work',
    'port': 3306,
    'charset': 'utf8',
    'use_unicode': True}

df = pd.read_csv("C:/ITWILL/3_Python/workspace/final_project_html/static/fs_all_df.csv", encoding='euc-kr')
df.info()
df.drop('Unnamed: 0', axis=1, inplace=True)
df.info()

try:
    from sqlalchemy import create_engine

    # MySQL Connector using pymysql
    pymysql.install_as_MySQLdb()
    import MySQLdb

    engine = create_engine("mysql+mysqldb://scott:" + "tiger" + "@127.0.0.1:3306/work?charset=utf8", encoding='utf-8')
    conn = engine.connect()

    df.to_sql(name='fs', con=engine, if_exists='fail')
    print("df table created")

except Exception as e:
    print('db error :', e)
finally:
    conn.close()


