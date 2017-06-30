#!/usr/bin/env python 2.7.12
#coding=utf-8
#author=yexiaozhu
import numpy as np
from pandas import Series

ints = np.ones(10, dtype=np.uint8)
floats = np.ones(10, dtype=np.float32)
# print np.issubdtype(ints.dtype, np.integer)
# print np.issubdtype(floats.dtype, np.floating)
# print np.float64.mro()
arr = np.arange(8)
# print arr
# print arr.reshape((4, 2))
# print arr.reshape((4, 2)).reshape((2, 4))
arr = np.arange(15)
# print arr.reshape((5, -1))
other_arr = np.ones((3, 5))
# print other_arr.shape
# print arr.reshape(other_arr.shape)
arr = np.arange(15).reshape((5, 3))
# print arr
# print arr.ravel()
# print arr.flatten()
arr = np.arange(12).reshape((3, 4))
# print arr
# print arr.ravel()
# print arr.ravel('F')
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
# print np.concatenate([arr1, arr2], axis=0)
# print np.concatenate([arr1, arr2], axis=1)
# print np.vstack((arr1, arr2))
# print np.hstack((arr1, arr2))
from numpy.random import randn
arr = randn(5, 2)
# print arr
first, second, third = np.split(arr, [1, 3])
# print first
# print second
# print third
arr = np.arange(6)
arr1 = arr.reshape((3, 2))
arr2 = randn(3, 2)
# print np.r_[arr1, arr2]
# print np.c_[np.r_[arr1, arr2], arr]
# print np.c_[1:6, -10:-5]
arr =np.arange(3)
# print arr.repeat(3)
# print arr.repeat([2, 3, 4])
arr = randn(2, 2)
# print arr
# print arr.repeat(2, axis=0)
# print arr.repeat([2, 3], axis=0)
# print arr.repeat([2, 3], axis=1)
# print arr
# print np.tile(arr, 2)
# print np.tile(arr, (2, 1))
# print np.tile(arr, (3, 2))
arr = np.arange(10) * 100
inds = [7, 1, 2, 6]
# print arr[inds]
# print arr.take(inds)
arr.put(inds, 42)
# print arr
arr.put(inds, [40, 41, 42, 43])
# print arr
inds = [2, 0, 2, 1]
arr = randn(2, 4)
# print arr
# print arr.take(inds, axis=1)
arr = randn(1000, 50)
# 500行随机样本
inds = np.random.permutation(1000)[:500]
# print arr[inds]
# print arr.take(inds, axis=0)
arr = np.arange(5)
# print arr
# print arr * 4
arr = randn(4, 3)
# print arr.mean(0)
demeaned = arr - arr.mean(0)
# print demeaned
# print demeaned.mean(0)
arr = np.arange(16).reshape(4, 4)
# print arr
row_means = arr.mean(1)
# print row_means
# print row_means.reshape((4, 1))
demeaned = arr - row_means.reshape((4, 1))
# print demeaned
# print demeaned.mean(1)
arr = np.zeros((4, 4))
# print arr
arr_3d = arr[:, np.newaxis, :]
# print arr_3d
# print arr_3d.shape
# arr_1d = np.random.normal(size=3)
# print arr_1d
arr_1d = np.arange(3)
# print arr_1d[:, np.newaxis]
# print arr_1d[np.newaxis, :]
arr = randn(3, 4, 5)
# print arr
depth_means = arr.mean(2)
# print depth_means
demeaned = arr - depth_means[:, :, np.newaxis]
# print demeaned.mean(2)
arr = np.zeros((4, 3))
arr[:] = 5
# print arr
col = np.array([1.28, -0.42, 0.44, 1.6])
arr[:] = col[:, np.newaxis]
# print arr
arr[:2] = [[-1.37], [0.509]]
# print arr
arr = np.arange(10)
# print np.add.reduce(arr)
# print arr.sum()
arr = randn(5, 5)
arr[::2].sort(1)
# print arr
# print arr[:, :-1] < arr[:, 1:]
# print np.logical_and.reduce(arr[:, :-1] < arr[:, 1:], axis=1)
arr = np.arange(15).reshape((3, 5))
# print np.add.accumulate(arr, axis=1)
arr = np.arange(3).repeat([1, 2, 3])
# print arr
# print np.multiply.outer(arr, np.arange(5))
result = np.subtract.outer(randn(3, 4), randn(5))
# print result.shape
arr = np.arange(10)
# print np.add.reduceat(arr, [0, 5, 8])
arr = np.multiply.outer(np.arange(4), np.arange(5))
# print arr
# print np.add.reduceat(arr, [0, 2, 4], axis=1)
def add_elements(x, y):
    return x + y
add_them = np.frompyfunc(add_elements, 2, 1)
# print add_them(np.arange(8), np.arange(8))
add_them = np.vectorize(add_elements, otypes=[np.float64])
# print add_them(np.arange(8), np.arange(8))
# arr = randn(10000)
dtype = [('x', np.float64), ('y', np.int32)]
sarr = np.array([(1.5, 6), (np.pi, -2)], dtype=dtype)
# print sarr
# print sarr[0]
# print sarr[0]['y']
# print sarr['x']
dtype = [('x', np.int64, 3), ('y', np.int32)]
arr = np.zeros(4, dtype=dtype)
# print arr
# print arr[0]['x']
# print arr['x']
dtype = [('x', [('a', 'f8'), ('b', 'f4')]), ('y', np.int32)]
data = np.array([((1, 2), 5), ((3, 4), 6)], dtype=dtype)
# print data['x']
# print data['y']
# print data['x']['a']
arr = randn(6)
arr.sort()
# print arr
arr = randn(3, 5)
# print arr
arr[:, 0].sort()
# print arr
arr = randn(5)
# print arr
# print np.sort(arr)
# print arr
arr = randn(3, 5)
# print arr
arr.sort(axis=1)
# print arr
# print arr[:, ::-1]
values = np.array([5, 0, 1, 3, 2])
indexer = values.argsort()
# print indexer
# print values[indexer]
arr = randn(3, 5)
arr[0] = values
# print arr
# print arr[:, arr[0].argsort()]
first_name = np.array(['Bob', 'Jane', 'Steve', 'Bill', 'Barbara'])
last_name = np.array(['Jones', 'Arnold', 'Arnold', 'Jones', 'Walters'])
sorter = np.lexsort((first_name, last_name))
# print zip(last_name[sorter], first_name[sorter])
values = np.array(['2:first', '2:second', '1:first', '1:second', '1:third'])
key = np.array([2, 2, 1, 1, 1])
indexer = key.argsort(kind='mergesort')
# print indexer
# print values.take(indexer)
arr = np.array([0, 1, 7, 12, 15])
# print arr
# print arr.searchsorted(9)
# print arr.searchsorted(4)
# print arr.searchsorted([0, 8, 11, 16])
arr = np.array([0, 0, 0, 1, 1, 1, 1])
# print arr.searchsorted([0, 1])
# print arr.searchsorted([0, 1], side='right')
data = np.floor(np.random.uniform(0, 10000, size=50))
bins = np.array([0, 100, 1000, 5000, 10000])
# print data
labels = bins.searchsorted(data)
# print labels
# print Series(data).groupby(labels).mean()
# print np.digitize(data, bins)
X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])
# print X[:, 0] # 一维的
y = X[:, :1] # 切片操作可产生二维结果
# print X
# print y
# print np.dot(y.T, np.dot(X, y))
Xm = np.matrix(X)
ym = Xm[:, 0]
# print Xm
# print ym
# print ym.T * Xm * ym
# print Xm.I * X #报错numpy.linalg.linalg.LinAlgError: Singular matrix
mmap = np.memmap('mymmap', dtype='float64', mode='w+', shape=(10000, 10000))
# print mmap
section = mmap[:5]
section[:] = np.random.randn(5, 10000)
mmap.flush()
# print mmap
del mmap
mmap = np.memmap('mymmap', dtype='float64', shape=(10000, 10000))
# print mmap
arr_c = np.ones((1000, 1000), order='C')
arr_f = np.ones((1000, 1000), order='F')
# print arr_c.flags
# print arr_f.flags
# print arr_f.flags.f_contiguous
# print arr_f.copy('C').flags
# print arr_c[:50].flags.contiguous
# print arr_c[:, :50].flags
