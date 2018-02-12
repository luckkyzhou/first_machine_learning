# -*- coding: utf-8 -*-
# udacity深度学习课程
# written by Luckky_Zhou
# 2018/2/6

from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display,Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

image_size = 28
def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes #整除
    tsize_per_class = train_size // num_classes #获取每个pickle的size

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files): #用于循环中的对象，可用index和item返回索引值和元素
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set) #将数据顺序随机打乱，多维中只对行进行随机排序
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

train_datasets = ['C:/Users/dong/Desktop/hanjia/notMNIST_large/A.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_large/B.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_large/C.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_large/D.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_large/E.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_large/F.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_large/G.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_large/H.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_large/I.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_large/J.pickle']
test_datasets = ['C:/Users/dong/Desktop/hanjia/notMNIST_small/A.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_small/B.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_small/C.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_small/D.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_small/E.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_small/F.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_small/G.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_small/H.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_small/I.pickle','C:/Users/dong/Desktop/hanjia/notMNIST_small/J.pickle']
valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0]) #permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

data_root = '.'
pickle_file = os.path.join(data_root, 'notMNIST.pickle')

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file) #在给定的路径上执行一个系统 stat 的调用
print('Compressed pickle size:', statinfo.st_size)