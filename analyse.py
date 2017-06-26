#!/usr/bin/env python 2.7.12
#coding=utf-8
#author=yexiaozhu

from datetime import datetime, timedelta

import pandas as pd

now = datetime.now()
# print now
# print now.year, now.month, now.day
delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
# print delta
# print delta.days
# print delta.seconds
start = datetime(2011, 1, 7)
# print start + timedelta(12)
# print start - 2 * timedelta(12)
stamp = datetime(2011, 1, 3)
# print str(stamp)
# print stamp.strftime('%Y-%m-%d')
value = '2011-01-03'
# print datetime.strptime(value, '%Y-%m-%d')
datestrs = ['7/6/2011', '8/6/2011']
# print [datetime.strptime(x, '%m/%d/%Y') for x in datestrs]
from dateutil.parser import parse
# print parse('2011-01-03')
# print parse('Jan 31, 1997 10:45 PM')
# print parse('6/12/2011', dayfirst=True)
# print datestrs
# print pd.to_datetime(datestrs)
idx = pd.to_datetime(datestrs + [None])
# print idx
# print idx[2]
# print pd.isnull(idx)
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7), datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12), ]
ts = pd.Series(pd.np.random.randn(6), index=dates)
# print ts
# print type(ts)
# print ts.index
# print ts + ts[::2]
# print ts.index.dtype
stamp = ts.index[0]
# print stamp
stamp = ts.index[2]
# print ts[stamp]
# print ts['1/10/2011']
# print ts['20110110']
from pandas import Series, np, DataFrame

longer_ts = Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
# print longer_ts
# print longer_ts['2001']
# print longer_ts['2001-05']
# print ts[datetime(2011, 1, 7):]
# print ts
# print ts['1/6/2011':'1/11/2011']
# print ts.truncate(after='1/9/2011')
dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = DataFrame(np.random.randn(100, 4),
                    index=dates,
                    columns=['Colorado', 'Texas', 'New York', 'Ohio'])
# print long_df.ix['5-2001']
dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000', '1/3/2000'])
dup_ts = Series(np.arange(5), index=dates)
# print dup_ts
# print dup_ts.index.is_unique
# print dup_ts['1/3/2000']
# print dup_ts['1/2/2000']
grouped = dup_ts.groupby(level=0)
# print grouped.mean()
# print grouped.count()
# print ts
# print ts.resample('D')
index = pd.date_range('4/1/2012', '6/1/2012')
# print index
# print pd.date_range(start='4/1/2012', periods=20)
# print pd.date_range(end='4/1/2012', periods=20)
# print pd.date_range('1/1/2000', '12/1/2000', freq='BM')
# print pd.date_range('5/2/2012 12:56:31', periods=5)
# print pd.date_range('5/2/2012 12:56:31', periods=5, normalize=True)
from pandas.tseries.offsets import Hour, Minute
hour = Hour()
# print hour
four_hours = Hour(4)
# print four_hours
# print pd.date_range('1/1/2000', '1/3/2000 23:59', freq='4h')
# print Hour(2) + Minute(30)
# print pd.date_range('1/1/2000', periods=10, freq='1h30min')
rng = pd.date_range('1/1/2012', '9/1/2012', freq='WOM-3FRI')
# print list(rng)
ts = Series(np.random.randn(4), index=pd.date_range('1/1/2000', periods=4, freq='M'))
# print ts
# print ts.shift(2)
# print ts.shift(-2)
# print ts.shift(2, freq='M')
# print ts.shift(3, freq='D')
# print ts.shift(1, freq='3D')
# print ts.shift(1, freq='90T')
from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2011, 11, 17)
# print now + 3 * Day()
# print now + MonthEnd()
# print now + MonthEnd(2)
offset = MonthEnd()
# print offset.rollforward(now)
# print offset.rollback(now)
ts = Series(np.random.randn(20), index=pd.date_range('1/15/2000', periods=20, freq='4d'))
# print ts.groupby(offset.rollforward).mean()
# print ts.resample('M').mean()
import pytz
# print pytz.common_timezones[-5:]
tz = pytz.timezone('US/Eastern')
# print tz
rng = pd.date_range('3/9/2012 9:30', periods=6, freq='D')
ts = Series(np.random.randn(len(rng)), index=rng)
# print ts.index.tz
# print pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC')
# print pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='US/Eastern')
ts_utc = ts.tz_localize('UTC')
# print ts_utc
# print ts_utc.index
# print ts_utc.tz_convert('US/Eastern')
ts_eastern = ts.tz_localize('US/Eastern')
# print ts_eastern.tz_convert('UTC')
# print ts_eastern.tz_convert('Europe/Berlin')
# print ts.index.tz_localize('Asia/Shanghai')
stamp = pd.Timestamp('2011-03-12 04:00')
stamp_utc = stamp.tz_localize('utc')
# print stamp_utc.tz_convert('US/Eastern')
stamp = pd.Timestamp('2011-03-12 04:00')
stamp_utc = stamp.tz_localize('utc')
# print stamp_utc.tz_convert('US/Eastern')
stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')
# print stamp_moscow
# print stamp_utc.value
# print stamp_utc.tz_convert('US/Eastern').value
stamp = pd.Timestamp('2012-03-12 01:30', tz='US/Eastern')
# print stamp
# print stamp + Hour()
stamp = pd.Timestamp('2012-11-04 00:30', tz='US/Eastern')
# print stamp
# print stamp + 2 * Hour()
rng = pd.date_range('3/7/2012 9:30', periods=10, freq='B')
ts = Series(np.random.randn(len(rng)), index=rng)
# print ts
ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts1[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2
# print result.index
p = pd.Period(2007, freq='A-DEC')
# print p
# print p + 5
# print p - 2
# print pd.Period('2014', freq='A-DEC') - p
rng = pd.period_range('1/1/2000', '6/30/2000', freq='M')
# print rng
# print Series(np.random.randn(6), index=rng)
values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values, freq='Q-DEC')
# print index
p = pd.Period('2007', freq='A-DEC')
# print p.asfreq('M', how='start')
# print p.asfreq('M', how='end')
p = pd.Period('2007', freq='A-JUN')
# print p.asfreq('M', 'start')
# print p.asfreq('M', 'end')
p = pd.Period('2007-08', 'M')
# print p.asfreq('A-JUN')
rng = pd.period_range('2006', '2009', freq='A-DEC')
ts = Series(np.random.randn(len(rng)), index=rng)
# print ts
# print ts.asfreq('M', how='start')
# print ts.asfreq('B', how='end')
p = pd.Period('2012Q4', freq='Q-JAN')
# print p
# print p.asfreq('D', 'start')
# print p.asfreq('D', 'end')
p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
# print p4pm
# print p4pm.to_timestamp()
rng = pd.period_range('2011Q3', '2012Q4', freq='Q-JAN')
ts = Series(np.arange(len(rng)), index=rng)
# print ts
new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
ts.index = new_rng.to_timestamp()
# print ts
rng = pd.date_range('1/1/2000', periods=3, freq='M')
ts = Series(np.random.randn(3), index=rng)
pts = ts.to_period()
# print ts
# print pts
rng = pd.date_range('1/29/2000', periods=6, freq='D')
ts2 = Series(np.random.randn(6), index=rng)
# print ts2.to_period('M')
pts = ts.to_period()
# print pts
# print pts.to_timestamp(how='end')
data = pd.read_csv('ch08/macrodata.csv')
# print data.year
# print data.quarter
index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
# print index
data.index = index
# print data.infl
rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(np.random.randn(len(rng)), index=rng)
# print ts.resample('M', how='mean')
# print ts.resample('M', how='mean', kind='period')
rng = pd.date_range('1/1/2000', periods=12, freq='T')
ts = Series(np.arange(12), index=rng)
# print ts
# print ts.resample('5min', how='sum')
# print ts.resample('5min', closed='right').sum()
# print ts.resample('5min', closed='left', label='left').sum()
# print ts.resample('5min', loffset='-1s', closed='left', label='left').sum()
# print ts.resample('5min', closed='left').ohlc()
rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(np.arange(100), index=rng)
# print ts.groupby(lambda x: x.month).mean()
# print ts.groupby(lambda x: x.weekday).mean()
frame = DataFrame(np.random.randn(2, 4),
                  index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
                  columns=['Colorado', 'Texas', 'New York', 'Ohio'])
# print frame[:5]
df_daily = frame.resample('D')
# print df_daily
# print frame.resample('D').ffill()
# print frame.resample('D', limit=2).ffill()
frame = DataFrame(np.random.randn(23, 4),
                  index=pd.date_range('1-2000', '12-2001', freq='M'),
                  columns=['Colorado', 'Texas', 'New York', 'Ohio'])
# print frame
# print frame[:5]
# annual_frame = frame.resample('A-DEC', how='mean')
annual_frame = frame.resample('A-DEC').mean()
# print annual_frame
# print annual_frame.resample('Q-DEC').ffill()
# print annual_frame.resample('Q-MAR').ffill()
# print annual_frame.resample('Q-DEC', convention='start').ffill()
close_px_all = pd.read_csv('ch09/stock_px.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']]
close_px = close_px.resample('B').ffill()
# print close_px
# close_px['AAPL'].plot()
# close_px.ix['2009'].plot()
# close_px['AAPL'].ix['01-2011': '03-2011'].plot()
appl_q = close_px['AAPL'].resample('Q-DEC').ffill()
# appl_q.ix['2009':].plot()
# close_px.AAPL.plot()
# pd.rolling_mean(close_px.AAPL, 250).plot()
import matplotlib.pyplot as plt
# plt.show()
appl_std250 = close_px.AAPL.rolling(min_periods=10,window=250,center=False).std()
# print appl_std250[5:12]
# appl_std250.plot()
# plt.show()
expanding_mean = lambda x: pd.rolling_mean(x, len(x), min_periods=1)
# pd.rolling_mean(close_px, 60).plot(logy=True)
# close_px.rolling(window=60, center=False).mean().plot(logy=True)
# plt.show()
# fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(12, 7))
aapl_px = close_px.AAPL['2005':'2009']
# ma60 = pd.rolling_mean(aapl_px, 60, min_periods=50)
ma60 = aapl_px.rolling(min_periods=50,window=60,center=False).mean()
# ewma60 = pd.ewma(aapl_px, span=60)
ewma60 = aapl_px.ewm(ignore_na=False,span=60,min_periods=0,adjust=True).mean()
# aapl_px.plot(style='k-', ax=axes[0])
# ma60.plot(style='k--', ax=axes[0])
# aapl_px.plot(style='k-', ax=axes[1])
# ewma60.plot(style='k--', ax=axes[1])
# axes[0].set_title('Simple MA')
# axes[1].set_title('Exponentially-weighted MA')
# plt.show()
spx_px = close_px_all['SPX']
spx_rets = spx_px / spx_px.shift(1) - 1
returns = close_px.pct_change()
# corr = pd.rolling_corr(returns.AAPL, spx_rets, 125, min_periods=100)
# corr = pd.rolling_corr(returns, spx_rets, 125, min_periods=100)
# corr.plot()
# plt.show()
from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x, 0.02)
result = pd.rolling_apply(returns.AAPL, 250, score_at_2percent)
# result.plot()
# plt.show()
rng = pd.date_range('1/1/2000', periods=10000000, freq='10ms')
ts = Series(np.random.randn(len(rng)), index=rng)
# print ts
# print ts.resample('15min').ohlc()
%timeit ts.resample('15min').ohlc()
rng = pd.date_range('1/1/2000', periods=10000000, freq='1s')
ts = Series(np.random.randn(len(rng)), index=rng)
print ts
%timeit ts.resample('15s').ohlc()

