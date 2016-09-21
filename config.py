#!/usr/bin/env python
#-*- coding:utf-8 -*-

class AgentConfig(object):
  scale = 1000
  display = False # whether show a window.

  max_step = 5000 * scale # 训练的最大的次数
  memory_size = 100 * scale # 记录的episode个数，episode的信息为(screen,action,reward,terminal)，不是经验个数

  batch_size = 32 # 每次训练使用的经验的条数
  random_start = 30 # 当运行一次随机步数操作时，对最大步数的限制。
  cnn_format = 'NCHW' # ?!!
  discount = 0.99
  target_q_update_step = 1 * scale # 每执行scale次，更新一次网络参数，延迟更新
  learning_rate = 0.00025
  learning_rate_minimum = 0.00025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 5 * scale

  # 当没有指定随机选择的概率时，随机选择的概率是从ep_start逐渐下降到ep_end
  ep_end = 0.1   
  ep_start = 1.  
  ep_end_t = memory_size # 随机选择概率的下降速度参数

  history_length = 4 # 4帧
  train_frequency = 4 # 当step是4的倍数时才训练网络
  learn_start = 5. * scale  # 当ReplayMemory中的信息超过learn_start时，开始训练网络.

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False

  _test_step = 5 * scale
  _save_step = _test_step * 10

class EnvironmentConfig(object):
  env_name = 'Breakout-v0'

  screen_width  = 84
  screen_height = 84
  max_reward = 1.  #限定能够收到的最大奖励
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'  #运行的环境类型，目前没有多大差别
  action_repeat = 1    #一个动作的重复执行次数

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2

  for k, v in FLAGS.__dict__['__flags'].items():
    if k == 'gpu':
      if v == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v)

  return config
