#!/usr/bin/env python
#-*- coding:utf-8 -*-

import time
import cPickle #tool for python object serialization
import tensorflow as tf

def timeit(f):
  """time for a function.
  this is a decorator, working for count the time of a function run.
  """
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
    return result
  return timed

def get_time():
  return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

@timeit
def save_pkl(obj, path):
  """save a object as file.
  To make this data can work at different system,it use Pickle as serialization tool.
  """
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print("  [*] save %s" % path)

@timeit
def load_pkl(path):
  """load a object from a file.
  The counterpart of @save_pkl.
  """
  with open(path) as f:
    obj = cPickle.load(f)
    print("  [*] load %s" % path)
    return obj

@timeit
def save_npy(obj, path):
  """save a numpy object as file.
  """
  np.save(path, obj)
  print("  [*] save %s" % path)

@timeit
def load_npy(path):
  obj = np.load(path)
  print("  [*] load %s" % path)
  return obj
