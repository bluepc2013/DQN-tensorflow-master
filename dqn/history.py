#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

class History:
  def __init__(self, config):
    self.cnn_format = config.cnn_format #"NHWC" or "NCHW"

    batch_size, history_length, screen_height, screen_width = \
        config.batch_size, config.history_length, config.screen_height, config.screen_width

    self.history = np.zeros(
        [history_length, screen_height, screen_width], dtype=np.float32)

  def add(self, screen):
    """record a screen.
    the length is 4. new screen will be insert at end.
    """
    self.history[:-1] = self.history[1:] #列表元素向前移动一位, [1,2,3] -> [2,3,3]
    self.history[-1] = screen

  def reset(self):
    self.history *= 0 #clear @self.history. A funny way.

  def get(self):
    if self.cnn_format == 'NHWC':
      return np.transpose(self.history, (1, 2, 0)) #?it must be NCHW,but NCHW and CPU?
    else:
      return self.history
