#!/usr/bin/env python
#-*- coding:utf-8 -*-

import cv2
import gym
import random
import numpy as np

class Environment(object):
  def __init__(self, config):
    #create a game.
    self.env = gym.make(config.env_name)

    screen_width, screen_height, self.action_repeat, self.random_start = \
        config.screen_width, config.screen_height, config.action_repeat, config.random_start

    self.display = config.display
    self.dims = (screen_width, screen_height)

    self._screen = None    #a frame picture.
    self.reward = 0
    self.terminal = True     #whether game over.

  def new_game(self, from_random_game=False):
    """在游戏中执行一步操作，并返回新的游戏状态。
    """

    if self.lives == 0: #Be careful, @self.lives is a "property function(pf)".
      self._screen = self.env.reset()
    self._step(0)
    self.render()
    return self.screen, 0, 0, self.terminal #@self.screen is a pf.

  def new_random_game(self):
    """在游戏中执行随机前进一段时间，最大步数是30.
    """
    self.new_game(True)
    for _ in xrange(random.randint(0, self.random_start - 1)):
      self._step(0)
    self.render()
    return self.screen, 0, 0, self.terminal

  def _step(self, action):
    """按照指定的动作前进一步，并更新环境数据。
    """
    self._screen, self.reward, self.terminal, _ = self.env.step(action)

  def _random_step(self):
    """前进一步，动作任选.
    perform random action at one step.
    """

    action = self.env.action_space.sample() #get a action from action space.
    self._step(action)

  @property #define a method @getter for variable @screen.
  def screen(self):
    """图像预处理。
    将RGB图片转换成gray图，再将像素值归一化到[0,1]区间，再调整尺寸。

    args:
       use @self._screen as input.
    return:
       screen matrix, may be a numpy object.
    """
    return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY)/255., self.dims)
    #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

  @property
  def action_size(self):
    """当前游戏的行动个数。
    """
    return self.env.action_space.n

  @property
  def lives(self):
    """生命数？
    """
    return self.env.ale.lives()

  @property
  def state(self):
    """返回当前游戏的状态？
    """
    return self.screen, self.reward, self.terminal

  def render(self):
    """绘制当前的帧。
    """
    if self.display:
      self.env.render()

  def after_act(self, action):
    """显示当前帧。
    """
    self.render()

class GymEnvironment(Environment):
  def __init__(self, config):
    super(GymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    cumulated_reward = 0 # 累计奖励
    start_lives = self.lives # 起始生命数，注意 @self.lives is pf.

    for _ in xrange(self.action_repeat):
      self._step(action)
      cumulated_reward = cumulated_reward + self.reward

      #在训练状态下，只允许有一条生命
      if is_training and start_lives > self.lives:
        cumulated_reward -= 1
        self.terminal = True

      if self.terminal:
        break

    self.reward = cumulated_reward #??

    self.after_act(action)
    return self.state

class SimpleGymEnvironment(Environment):
  def __init__(self, config):
    super(SimpleGymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    self._step(action)

    self.after_act(action)
    return self.state
