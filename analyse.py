#!/usr/bin/env python 2.7.12
#coding=utf-8
#author=yexiaozhu
import pandas as pd

df = pd.read_csv('ch06/ex1.csv')
# print df
# print pd.read_csv('ch06/ex1.csv', sep=',')
# print pd.read_csv('ch06/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])
names = ['a', 'b', 'c', 'd', 'message']
# print pd.read_csv('ch06/ex2.csv', names=names, index_col='message')
parsed = pd.read_csv('ch06/csv_mindex.csv', index_col=['key1', 'key2'])
# print parsed
# print list(open('ch06/ex3.txt'))
result = pd.read_table('ch06/ex3.txt', sep='\s+')
# print result
# print pd.read_csv('ch06/ex4.csv', skiprows=[0, 2, 3])
result =  pd.read_csv('ch06/ex5.csv')
# print result
# print pd.isnull(result)
result = pd.read_csv('ch06/ex5.csv', na_values=['NULL'])
# print result
sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
# print pd.read_csv('ch06/ex5.csv', na_values=sentinels)
result = pd.read_csv('ch06/ex6.csv')
# print result
# print pd.read_csv('ch06/ex6.csv', nrows=5)
chunker = pd.read_csv('ch06/ex6.csv', chunksize=1000)
# print chunker
tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)
tot = tot.sort_values(ascending=False) # order替换为sort_value
# print tot[:10]
data = pd.read_csv('ch06/ex5.csv')
# print data
# data.to_csv('ch06/out.csv')
datas = pd.date_range('1/1/2000', periods=7)
from pandas import Series, np
ts = Series(np.arange(7), index=datas)
# ts.to_csv('ch06/tseries.csv')
# print Series.from_csv('ch06/tseries.csv', parse_dates=True)
import csv
f = open('ch06/ex7.csv')
reader = csv.reader(f)
# for line in reader:
    # print line
lines = list(csv.reader(open('ch06/ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
# print data_dict
import json
obj = """
{"name": "Wes",
 "palce_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
 {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""
result = json.loads(obj)
# print result
asjson = json.dumps(result)
# print asjson
from pandas import DataFrame
siblings = DataFrame(result['siblings'], columns=['name', 'age'])
# print siblings
from lxml.html import parse
from urllib2 import urlopen
parsed = parse(urlopen('https://finance.yahoo.com/q/op?s=AAPL+Options'))
doc = parsed.getroot()
links = doc.findall('.//a')
# print links[15:20]
link = links[28]
# print link
# print link.get('href')
# print link.text_content()
urls = [link.get('href') for link in doc.findall('.//a')]
# print urls
# print urls[-10:]
tables = doc.findall('.//table')
# print tables
calls = tables[2]
puts = tables[2]
rows = calls.findall('.//tr')
def _unpack(row, kind='td'):
    elts = row.findall('.//%s' %kind)
    return [val.text_content() for val in elts]
# print _unpack(rows[0], kind='th')
# print _unpack(rows[1], kind='td')
from pandas.io.parsers import TextParser
def parse_options_data(table):
    rows = table.findall('.//tr')
    header = _unpack(rows[0], kind='th')
    data = [_unpack(r) for r in rows[1:]]
    return TextParser(data, names=header).get_chunk()
call_data = parse_options_data(calls)
put_data = parse_options_data(puts)
# print call_data[:10]
from StringIO import StringIO
from lxml import objectify
tag = '<a href="http://www.google.com">Google</a>'
root = objectify.parse(StringIO(tag)).getroot()
# print root
# print root.get('href')
# print root.text
frame = pd.read_csv('ch06/ex1.csv')
# print frame
# print type(frame)
# frame.save('ch06/frame_prckle')  # 没有save这个方法了
# print pd.load('ch06/frame_pickle')
# store = pd.HDFStore('mydata.h5')
# print frame
# store['obj1'] = frame
# store['obj1_col'] = frame['a']
# print store # 需安装table库
# print store['obj1']
import requests, json
url = 'https://api.twitter.com/1.1/search/tweets.json?q=python'
# 需要twitter账号验证 暂未申请
# resp = requests.get(url)
# print resp
# data = json.loads(resp.text)
# print data.keys()
# tweet_fields = ['created_at', 'from_user', 'id', 'text']
# tweets = DataFrame(data['results'], columns=tweet_fields)
# print tweets
# print tweets.ix[7]

import sqlite3
query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
c REAL, d INTEGER
);"""
con = sqlite3.connect(':memory:')
con.execute(query)
con.commit()
data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
con.executemany(stmt, data)
con.commit()
cursor = con.execute('select * from test')
rows = cursor.fetchall()
# print rows
# print cursor.description
# print DataFrame(rows, columns=zip(*cursor.description)[0])
from pandas import read_sql
# print read_sql('select * from test', con) # pandas.io.sql.read_frame已被read_sql替换