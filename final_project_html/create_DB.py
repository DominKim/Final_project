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

# 데이터 가져오기. final
df = pd.read_csv(
    "C:/ITWILL/project_df1.csv",
    thousands=',', encoding='euc-kr')
df.info()
df.columns

df_cluster = pd.read_csv(
    "C:/ITWILL/project_df2.csv",
    thousands=',', encoding='euc-kr')
df_cluster.info()

df['cluster'] = df_cluster['cluster']
df.info()

# 결측치 확인
df.isnull().sum().sort_values().tail(100)

cols = list(df.columns)
len(cols)  # 179

# 총계[abstract] 삭제
abst = []
for col in cols:
    if '[abstract]' in col:
        print(col)
        abst.append(col)
'''
재무상태표 [abstract]
자산 [abstract]
부채 [abstract]
자본 [abstract]
'''

df.drop(abst, axis=1, inplace=True)
df.info()  # columns : 179 > 174

# 결측치 채우기
df.isnull().sum()
df_fillna = df.fillna(999999)

df_fillna.iloc[0]
df_fillna.info()
df_fillna.isnull().sum().sort_values()

try:
    from sqlalchemy import create_engine

    # MySQL Connector using pymysql
    pymysql.install_as_MySQLdb()
    import MySQLdb

    engine = create_engine("mysql+mysqldb://scott:" + "tiger" + "@127.0.0.1:3306/work?charset=utf8", encoding='utf-8')
    conn = engine.connect()

    df_fillna.to_sql(name='fs_simple', con=engine, if_exists='fail')
    print("df table created")

except Exception as e:
    print('db error :', e)
finally:
    conn.close()
