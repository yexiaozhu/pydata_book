#!/usr/bin/env python 2.7.12
#coding=utf-8
#author=yexiaozhu
import pandas
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

# names = ['date', 'AA', 'AAPL', 'GE', 'IBM', 'JNJ', 'MSFT', 'PEP', 'SPX', 'XOM']
# prices_data = pd.read_csv('ch11/stock_px.csv', skiprows=1, header=None, names=names)
# # print prices_data[:5]
# prices = prices_data[['date', 'AAPL', 'JNJ', 'SPX', 'XOM']]
# # print prices[prices['date'] > '2011-09-06']
# # print prices[prices['date'] < '2011-09-15']
# prices = prices[(prices['date'] > '2011-09-06')&(prices['date'] < '2011-09-15')]
# print prices
# # print type(prices)
# volumes_data = pd.read_csv('ch11/volume.csv',  skiprows=1, header=None, names=names)
# # print volumes_data[:5]
# volumes = volumes_data[['date', 'AAPL', 'JNJ', 'XOM']]
# # print volumes[volumes['date'] > '2011-09-06']
# # print volumes[volumes['date'] < '2011-09-13']
# volume = volumes[(volumes['date'] > '2011-09-06')&(volumes['date'] < '2011-09-13')]
# print volume
# print volume[0:1]
#
# # print type(volume)
# print prices * volume
prices_data = pd.read_csv('ch11/stock_px.csv', index_col=0)
prices = prices_data.ix[[5443,5444,5445,5446,5447,5448,5449],['AAPL', 'JNJ', 'SPX', 'XOM']]
# print prices
volume_data = pd.read_csv('ch11/volume.csv', index_col=0)
volume = volume_data.ix[[5443,5444,5445,5446,5447],['AAPL', 'JNJ', 'XOM']]
# print volume
# print prices * volume
vwap = (prices * volume).sum() / volume.sum()
# print vwap
# print vwap.dropna()
# print prices.align(volume, join='inner')
s1 = Series(range(3), index=['a', 'b', 'c'])
s2 = Series(range(4), index=['d', 'b', 'c', 'e'])
s3 = Series(range(3), index=['f', 'a', 'c'])
# print DataFrame({'one': s1, 'two': s2, 'three': s3})
# print DataFrame({'one': s1, 'two': s2, 'three': s3}, index=list('face'))
ts1 = Series(np.random.randn(3), index=pd.date_range('2012-6-13', periods=3, freq='W-WED'))
# print ts1
# print ts1.resample('B')
# print ts1.resample('B').ffill()
dates = pd.DatetimeIndex(['2012-6-12', '2012-6-17', '2012-6-18',
                          '2012-6-21', '2012-6-22', '2012-6-29'])
ts2 = Series(np.random.randn(6), index=dates)
# print ts2
# print ts1.reindex(ts2.index).ffill()
# print ts2 + ts1.reindex(ts2.index).ffill()
gdp = Series([1.78, 1.94, 2.08, 2.01, 2.05, 2.31, 2.46],
             index=pd.period_range('1984Q2', periods=7, freq='Q-SEP'))
infl = Series([0.025, 0.045, 0.037, 0.04],
              index=pd.period_range('1982', periods=4, freq='A-DEC'))
# print gdp
# print infl
infl_q = infl.asfreq('Q-SEP', how='end')
# print infl_q
# print infl_q.reindex(gdp.index).ffill()
rng = pd.date_range('2012-06-01 09:30', '2012-06-01 15:59', freq='T') # 生成一个交易日内的日期范围和时间序列
rng = rng.append([rng + pd.offsets.BDay(i) for i in range(1, 4)]) # 生成5天时间点(9:30-15:59的值)
ts = Series(np.arange(len(rng), dtype=float), index=rng)
# print ts
# print type(ts)
# ts.to_csv('ts.csv')
from datetime import time
# print ts[time(10, 0)]
# print ts.at_time(time(10, 0))
# print ts.between_time(time(10, 0), time(10, 1))
# 将该时间序列的大部分内容随机设置为NA
indexer = np.sort(np.random.permutation(len(ts))[700:])
irr_ts = ts.copy()
irr_ts[indexer] = np.nan
# print irr_ts['2012-06-01 09:50':'2012-06-01 10:00']
selection = pd.date_range('2012-06-01 10:00', periods=4, freq='B')
# print irr_ts.asof(selection)
data1 = DataFrame(np.ones((6, 3), dtype=float),
                  columns=['a', 'b', 'c'],
                  index=pd.date_range('6/12/2012', periods=6))
data2 = DataFrame(np.ones((6, 3), dtype=float) * 2,
                  columns=['a', 'b', 'c'],
                  index=pd.date_range('6/13/2012', periods=6))
spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]])
# print spliced
data2 = DataFrame(np.ones((6, 4), dtype=float) * 2,
                  columns=['a', 'b', 'c', 'd'],
                  index=pd.date_range('6/13/2012', periods=6))
spliced = pd.concat([data1.ix[:'2012-06-14'], data2.ix['2012-06-15':]])
# print spliced
spliced_filled = spliced.combine_first(data2)
# print spliced_filled
spliced.update(data2, overwrite=False)
# print spliced
cp_spliced = spliced.copy()
cp_spliced[['a', 'c']] = data1[['a', 'c']]
# print cp_spliced
import random; random.seed(0)
import string
N = 1000
def rands(n):
    choices = string.ascii_uppercase
    return ''.join([random.choice(choices) for _ in xrange(n)])
tickers = np.array([rands(5) for _ in xrange(N)])
M = 500
df = DataFrame({'Momentum' : np.random.randn(M) / 200 + 0.03,
                'Value' : np.random.randn(M) / 200 + 0.08,
                'ShortIntrtest' : np.random.randn(M) / 200 - 0.02},
               index=tickers[:M])
ind_names = np.array(['FINANCIAL', 'TECH'])
sampler = np.random.randint(0, len(ind_names), N)
industries = Series(ind_names[sampler], index=tickers, name='industry')
by_industry = df.groupby(industries)
# print by_industry.mean()
# print by_industry.describe()
# 行业内标准化处理
def zsocre(group):
    return (group - group.mean()) / group.std()
df_stand = by_industry.apply(zsocre)
# print df_stand.groupby(industries).agg(['mean', 'std'])
# 行业内降序排名
ind_rank = by_industry.rank(ascending=False)
# print ind_rank.groupby(industries).agg(['min', 'max'])
# 行业内排名和标准化
# print by_industry.apply(lambda x: zsocre(x.rank()))
from numpy.random import rand
fac1, fac2, fac3 = np.random.rand(3, 1000)
ticker_subset = tickers.take(np.random.permutation(N)[:1000])
# 因子加权和以及噪声
port = Series(0.7 * fac1 - 1.2 * fac2 + 0.3 * fac3 + rand(1000), index=ticker_subset)
factors = DataFrame({'f1': fac1, 'f2': fac2, 'f3': fac3}, index=ticker_subset)
# print factors.corrwith(port)
# print port
# print factors
results = pd.stats.ols.MovingOLS(y=port, x=factors).fit()
print results
def beta_exposure(chunk, factors=None):
    return pd.sm.ols(y=chunk, x=factors).beta
by_ind = port.groupby(industries)
exposures = by_ind.apply(beta_exposure, factors=factors)
print exposures.unstack()