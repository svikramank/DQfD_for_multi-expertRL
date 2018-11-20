# -*- coding: utf-8 -*
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
from Config import Config, DDQNConfig, DQfDConfig
from DQfDDDQN import DQfDDDQN


def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def run_DQfD(index, env):
    with tf.variable_scope('DQfD_' + str(index)):
        agent = DQfDDDQN(env, DQfDConfig())
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        agent.demo_buffer = pickle.load(f)
    agent.pre_train()  # use the demo data to pre-train network
    scores = []
    for e in range(Config.episode):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive(state, action, reward, next_state, done, 0.0)
            agent.train_Q_network(pre_train=False, update=False)
            state = next_state
        if done:
            scores.append(score)
            agent.sess.run(agent.update_target_net)
            print("episode:", e, "  score:", score, "  memory length:", len(agent.replay_buffer), "  epsilon:",
                  agent.epsilon)
            # if np.mean(scores[-min(10, len(scores)):]) > 495:
            #     break
    return scores


if __name__ == '__main__':
    env = gym.make(Config.ENV_NAME)
    # ----------------------------- get DQfD scores --------------------------------
    dqfd_sum_scores = np.zeros(Config.episode)
    for i in range(Config.iteration):
        scores = run_DQfD(i, env)
        dqfd_sum_scores = [a + b for a, b in zip(scores, dqfd_sum_scores)]
    dqfd_mean_scores = dqfd_sum_scores / Config.iteration
    with open('/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores.p', 'wb') as f:
        pickle.dump(dqfd_mean_scores, f, protocol=2)

    map_scores(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores,
    xlabel='Red: dqfd', ylabel='Scores')





















