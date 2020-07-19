# -*- coding: utf-8 -*-
from scipy import stats
import pandas as pd

data = pd.read_excel("C:\\Users\\user\\Documents\\통합 문서1.xlsx", index=True)
# 데이터는 업종코드만 추출하여 사용하였음, 전체 산업코드와 각 집단 산업코드를 분리 추출
data.info()

dis_all = data['전체'].value_counts()
dis_0 = data['0번'].value_counts()
dis_1 = data['1번'].value_counts()
dis_2 = data['2번'].value_counts()
dis_4 = data['4번'].value_counts()
dis_5 = data['5번'].value_counts()
dis_7 = data['7번'].value_counts()

dis_all.shape
dis_0.shape
dis_1.shape
dis_2.shape
dis_4.shape
dis_5.shape
dis_7.shape

dis_all.index
dis_4.index

chis1 = stats.chisquare(dis_0, dis_all)
# 오류 : 길이 불일치

a = []
for i in dis_all.index:
    if i in dis_0.index:
        a.append(dis_0[i])
    else:
        a.append(0)

        
dis_all1 = list(dis_all)

chis0 = stats.chisquare(dis_all1, a)
print('statistic = %.3f, pvalue = %.3f'%(chis0))


b = []
for i in dis_all.index:
    if i in dis_1.index:
        b.append(dis_1[i])
    else:
        b.append(0)

chis1 = stats.chisquare(dis_all1, b)
print('statistic = %.3f, pvalue = %.3f'%(chis1))


c = []
for i in dis_all.index:
    if i in dis_2.index:
        c.append(dis_2[i])
    else:
        c.append(0)

chis2 = stats.chisquare(dis_all1, c)
print('statistic = %.3f, pvalue = %.3f'%(chis2))


d = []
for i in dis_all.index:
    if i in dis_4.index:
        d.append(dis_4[i])
    else:
        d.append(0)

chis4 = stats.chisquare(dis_all1, d)
print('statistic = %.3f, pvalue = %.3f'%(chis4))


e = []
for i in dis_all.index:
    if i in dis_5.index:
        e.append(dis_5[i])
    else:
        e.append(0)

chis5 = stats.chisquare(dis_all1, e)
print('statistic = %.3f, pvalue = %.3f'%(chis5))


f = []
for i in dis_all.index:
    if i in dis_7.index:
        f.append(dis_7[i])
    else:
        f.append(0)

chis7 = stats.chisquare(dis_all1, f)
print('statistic = %.3f, pvalue = %.3f'%(chis7))


