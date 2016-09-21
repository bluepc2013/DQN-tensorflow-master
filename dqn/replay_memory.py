#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import logging
import numpy as np

from utils import save_npy, load_npy

class ReplayMemory:
  """
  """
  def __init__(self, config, model_dir):
    self.model_dir = model_dir

    self.cnn_format = config.cnn_format # 'NHWC' or 'NCHW'
    self.memory_size = config.memory_size # 存储的帧数
	# 根据Bellman等式，一条完整的经验应该包含 (s,a,r,s').其中s和s'包含在
    self.actions = np.empty(self.memory_size, dtype = np.uint8) #a
    self.rewards = np.empty(self.memory_size, dtype = np.integer) #r
	# 这里的存储单位是帧，不是state，根据定义一个state等于连续的4帧
    self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype = np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool) # game over 标记
    self.history_length = config.history_length #4帧，s长度
    self.dims = (config.screen_height, config.screen_width)
    self.batch_size = config.batch_size # 每次训练提取的经验的条数
    self.count = 0 # 循环队列中的经验条数
    self.current = 0 # 循环队列的尾指针

    # pre-allocate prestates and poststates for minibatch
    # 为了节省内存，一次实验的所有帧按照序列的方式保存在screens中，其中每四个构成一个state
    # 例如： [1,2,3,4]和[2,3,4,5]分别是s和s'
    self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)

  def add(self, screen, reward, action, terminal):
    """add a new record to circular queue.
    """
    assert screen.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size #这里在控制循环队列

  def getState(self, index):
    """build a experiment from screens.
    get sreens from sreens[index-history_length+1] to screens[index+1].

    Returns:
        return 4 frame picture( state).
    """
    assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # because this is a circular queue, so it is 
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
      # 所有的4帧图像都在中间，直接取就可以了
      # use faster slicing
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # 4帧图像包含了首尾两端，所以需要一点技巧
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]

  def sample(self):
    """sample a batch of experiment randomly.
    Be careful, every 4 frames form a experience.

    Return:
        (s,a,r,s',t), @t for terminal/game over.

    """
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        # 索引值不能包含current位置，因为current左右的screen序列是断裂的，实际中不存在这种state
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
 		# 多次实验的帧序列是顺序记录在@screens中，不同实验间的帧是断裂的，不能作为经验使用
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
	  # 同一次实验过程中的所有帧序列，相邻的4个帧算是一个状态，例如：[1,2,3,4]与[2,3,4,5]分别时s和s'
      self.prestates[len(indexes), ...] = self.getState(index - 1)
      self.poststates[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    if self.cnn_format == 'NHWC':
      return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
    else:
      return self.prestates, actions, rewards, self.poststates, terminals

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      array = load_npy(os.path.join(self.model_dir, name))
