# -*- coding: utf-8 -*-


class Config:
    ENV_NAME = "CartPole-v1"
    GAMMA = 0.99  # discount factor for target Q
    INITIAL_EPSILON = 1.0  # starting value of epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    EPSILON_DECAY = 0.9985
    START_TRAINING = 1000  # experience replay buffer size
    BATCH_SIZE = 64  # size of minibatch
    UPDATE_TARGET_NET = 200  # update eval_network params every 200 steps
    LEARNING_RATE = 0.001
    DEMO_RATIO = 0.1
    LAMBDA_1 = 1.0
    LAMBDA_2 = 10e-5
    PRETRAIN_STEPS = 1000
    MODEL_PATH = './model/DQfDDDQN_model'
    DEMO_DATA_PATH_1 = '/Users/vikramanksingh/Desktop/DQfD/demo/DDQN_demo.p'
    DEMO_DATA_PATH_2 = '/Users/vikramanksingh/Desktop/DQfD/demo/AC_demo.p'
    DEMO_DATA_PATH_3 = '/Users/vikramanksingh/Desktop/DQfD/demo/PG_demo.p'

    replay_buffer_size = 2000
    demo_buffer_size = 500 * 50
    iteration = 5
    episode = 300  # 300 games per iteration


class DDQNConfig(Config):
    demo_mode = 'get_demo'


class DQfDConfig(Config):
    demo_mode = 'use_demo'
    demo_num = int(Config.BATCH_SIZE * Config.DEMO_RATIO)


