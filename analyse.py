#!/usr/bin/env python 2.7.12
#coding=utf-8
#author=yexiaozhu
import pandas as pd
from pandas import DataFrame, np, Series

df = DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
                'key2' : ['one', 'two', 'one', 'two', 'one'],
                'data1' : np.random.randn(5),
                'data2' : np.random.randn(5)})
# print df
grouped = df['data1'].groupby(df['key1'])
# print grouped
# print grouped.mean()
means = df['data1'].groupby([df['key1'], df['key2']]).mean()
# print means
# print means.unstack()
states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
# print df['data1'].groupby([states, years]).mean()
# print df.groupby('key1').mean()
# print df.groupby(['key1', 'key2']).mean()
# print df.groupby(['key1', 'key2']).size()
# for name, group in df.groupby('key1'):
#     print name
#     print group
# for (k1, k2), group in df.groupby(['key1', 'key2']):
#     print k1, k2
#     print group
pieces = dict(list(df.groupby('key1')))
# print pieces['b']
# print df.dtypes
grouped = df.groupby(df.dtypes, axis=1)
# print dict(list(grouped))
# print df.groupby(['key1', 'key2'])[['data2']].mean()
s_grouped = df.groupby(['key1', 'key2'])['data2']
# print s_grouped
# print s_grouped.mean()
people = DataFrame(np.random.randn(5, 5),
                   columns=['a', 'b', 'c', 'd', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.ix[2:3, ['b', 'c']] = np.nan
# print people
mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd': 'blue', 'e': 'red', 'f': 'orange'}
by_column = people.groupby(mapping, axis=1)
# print by_column.sum()
map_series = Series(mapping)
# print map_series
# print people.groupby(map_series, axis=1).count()
# print people.groupby(len).sum()
key_list = ['one', 'one', 'one', 'two', 'two']
# print people.groupby([len, key_list]).min()
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'],
                                     [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
hier_df = DataFrame(np.random.randn(4, 5), columns=columns)
# print hier_df
# print hier_df.groupby(level='cty', axis=1).count()
# print df
grouped = df.groupby('key1')
# print grouped['data1'].quantile(0.9)
def peak_to_peak(arr):
    return arr.max() - arr.min()
# print grouped.agg(peak_to_peak)
# print grouped.describe()
tips = pd.read_csv('ch08/tips.csv')
tips['tip_pct'] = tips['tip'] / tips['total_bill']
# print tips[:6]
grouped = tips.groupby(['sex', 'smoker'])
grouped_pct = grouped['tip_pct']
# print grouped_pct.agg('mean')
# print grouped_pct.agg(['mean', 'std', peak_to_peak])
# print grouped_pct.agg([('foo', 'mean'),( 'bar', np.std)])
functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
# print result
# print result['tip_pct']
ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
# print grouped['tip_pct', 'total_bill'].agg(ftuples)
# print grouped.agg({'tip': np.max, 'size' : 'sum'})
# print grouped.agg({'tip': ['min', 'max', 'mean', 'std'], 'size' : 'sum'})
# print tips.groupby(['sex', 'smoker'], as_index=False).mean()
# print df
k1_means = df.groupby('key1').mean().add_prefix('mean_')
# print k1_means
# print pd.merge(df, k1_means, left_on='key1', right_index=True)
key = ['one', 'two', 'one', 'two', 'one']
# print people.groupby(key).mean()
# print people.groupby(key).transform(np.mean)
def demean(arr):
    return arr - arr.mean()
demeaned = people.groupby(key).transform(demean)
# print demeaned
# print demeaned.groupby(key).mean()
def top(df, n=5, column='tip_pct'):
    return df.sort_values(by=column)[-n:]
# print top(tips, n=6)
# print tips.groupby('smoker').apply(top)
# print tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')
result = tips.groupby('smoker')['tip_pct'].describe()
# print result
# print result.unstack('smoker')
# print tips.groupby('smoker', group_keys=False).apply(top)
frame = DataFrame({'data1': np.random.randn(1000),
                   'data2': np.random.randn(1000)})
factor = pd.cut(frame.data1, 4)
# print factor[:10]
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count, 'mean': group.mean()}
grouped = frame.data2.groupby(factor)
# print grouped.apply(get_stats).unstack()
grouping = pd.qcut(frame.data1, 10, labels=False)
grouped = frame.data2.groupby(grouping)
# print grouped.apply(get_stats).unstack()
states = ['Ohio', 'New York', 'Vermont', 'Florida',
          'Oregon', 'Nevada', 'California', 'Idaho']
group_key = ['East'] * 4 + ['West'] * 4
data = Series(np.random.randn(8), index=states)
data[['Vermont', 'Nevada', 'Idaho']] = np.nan
# print data
# print data.groupby(group_key).mean()
fill_mean = lambda g: g.fillna(g.mean())
# print data.groupby(group_key).apply(fill_mean)
fill_values = {'East': 0.5, 'West': -1}
fill_func = lambda g: g.fillna(fill_values[g.name])
# print data.groupby(group_key).apply(fill_func)
# 红桃(Hearts), 黑桃(Spades), 梅花(Clubs), 方片(DIamonds)
suits = ['H', 'S', 'C', 'D']
card_val = (range(1, 11) + [10]* 3) * 4
base_names = ['A'] + range(2, 11) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)
deck = Series(card_val, index=cards)
# print deck[:13]
def draw(deck, n=5):
    return deck.take(np.random.permutation(len(deck))[:n])
# print draw(deck)
get_suit = lambda card: card[-1]
# print deck.groupby(get_suit).apply(draw, n=2)
# print deck.groupby(get_suit, group_keys=False).apply(draw, n=2)
df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
                'data': np.random.randn(8),
                'weights': np.random.rand(8)})
# print df
grouped = df.groupby('category')
get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
# print grouped.apply(get_wavg)
close_px = pd.read_csv('ch09/stock_px.csv', parse_dates=True, index_col=0)
# print close_px
# print close_px[-4:]
rets = close_px.pct_change().dropna()
spx_corr = lambda x: x.corrwith(x['SPX'])
by_year = rets.groupby(lambda x: x.year)
# print by_year.apply(spx_corr)
# print by_year.apply(lambda g: g['AAPL'].corr(g['MSFT']))
# import statsmodels.api as sm
# def regress(data, yvar, xvars):
#     Y = data[yvar]
#     X = data[xvars]
#     X['intercept'] = 1.
#     result = sm.OLS(Y, X).fit()
#     return result.params
# print by_year.apply(regress, 'AAPL', ['SPX'])
# print tips.pivot_table(['tip_pct', 'size'], index=['sex', 'day'],
#                        columns='smoker', margins=True)
# print tips.pivot_table('tip_pct', index=['sex', 'smoker'], columns='day', aggfunc=len, margins=True)
# print tips.pivot_table('size', index=['time', 'sex', 'smoker'], columns='day', aggfunc='sum', fill_value=0)
fec = pd.read_csv('ch09/P00000001-ALL.csv', low_memory=False)
# print fec
# print fec.ix[123456]
unique_cands = fec.cand_nm.unique()
# print unique_cands
parties = {'Bachmann, Michelle': 'Republican',
           'Romney, Mitt': 'Republican',
           'Obama, Barack': 'Democrat',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'Paul, Ron': 'Republican',
           'Santorum, Rick': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Perry, Rick': 'Republican'}
# print fec.cand_nm[123456:123461]
# print fec.cand_nm[123456:123461].map(parties)
fec['party'] = fec.cand_nm.map(parties)
# print fec['party'].value_counts()
# print (fec.contb_receipt_amt > 0).value_counts()
fec = fec[fec.contb_receipt_amt > 0]
fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]
# print fec.contbr_occupation.value_counts()[:10]
occ_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
   'C.E.O.': 'CEO'
}
# If no mapping provided, return x
f = lambda x: occ_mapping.get(x, x)
fec.contbr_occupation = fec.contbr_occupation.map(f)
emp_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'SELF' : 'SELF-EMPLOYED',
   'SELF EMPLOYED' : 'SELF-EMPLOYED',
}

# If no mapping provided, return x
f = lambda x: emp_mapping.get(x, x)
fec.contbr_employer = fec.contbr_employer.map(f)
by_occupation = fec.pivot_table('contb_receipt_amt',
                                index='contbr_occupation',
                                columns='party', aggfunc='sum')
over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
# print over_2mm
# over_2mm.plot(kind='barh')
import matplotlib.pyplot as plt
# plt.show()
def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    return totals.sort_values(ascending=False)[n:]
grouped = fec_mrbo.groupby('cand_nm')
# print grouped.apply(get_top_amounts, 'contbr_occupation', n=7)
# print grouped.apply(get_top_amounts, 'contbr_employer', n=10)
bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
# print labels
grouped = fec_mrbo.groupby(['cand_nm', labels])
# print grouped.size().unstack(0)
bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
# print bucket_sums
normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
# print normed_sums
# normed_sums[:-2].plot(kind='barh', stacked=True)
# plt.show()
grouped = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals = totals[totals.sum(1) > 100000]
# print totals[:10]
percent = totals.div(totals.sum(1), axis=0)
# print percent[:10]
