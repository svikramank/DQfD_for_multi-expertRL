# -*- coding: utf-8 -*
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
import random
from collections import deque
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


# def run_DDQN(index, env):
#     with tf.variable_scope('DDQN_' + str(index)):
#         agent = DQfDDDQN(env, DDQNConfig())
#     scores = []
#     for e in range(Config.episode):
#         done = False
#         score = 0  # sum of reward in one episode
#         state = env.reset()
#         while done is False:
#             action = agent.egreedy_action(state)  # e-greedy action for train
#             next_state, reward, done, _ = env.step(action)
#             score += reward
#             reward = reward if not done or score == 499 else -100
#             agent.perceive(state, action, reward, next_state, done, 0.0)  # 0. means it is not a demo data
#             agent.train_Q_network(pre_train=False, update=False)
#             state = next_state
#         if done:
#             scores.append(score)
#             agent.sess.run(agent.update_target_net)
#             print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
#                   "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
#             # if np.mean(scores[-min(10, len(scores)):]) > 490:
#             #     break
#     return scores


def run_DQfD(index, env):
    with tf.variable_scope('DQfD_' + str(index)):
        agent = DQfDDDQN(env, DQfDConfig())
    # load the expert data for demonstration
    dq1 = deque()
    dq2 = deque()
    dq3 = deque()
    fav1 = pickle.load(open(Config.DEMO_DATA_PATH_1, "rb" ))
    fav2 = pickle.load(open(Config.DEMO_DATA_PATH_2, "rb" ))
    fav3 = pickle.load(open(Config.DEMO_DATA_PATH_3, "rb" ))
    dq1 = fav1

    # converting AC dict to a deque class
    state = []
    next_state = []
    action = []
    reward = []
    done = []
    expert = []
    demo = []
    for i in range(len(fav2['observations'])):
        state.append(fav2['observations'][i])
        action.append(fav2['actions'][i])
        reward.append(fav2['rewards'][i])
        next_state.append(fav2['next_observations'][i])
        done.append(fav2['done'][i])
        expert.append(fav2['expert'][i])
    for i in range(len(state)):
        demo.append((state[i], action[i], reward[i], next_state[i], done[i], expert[i]))
    dq2.extend(demo)

    # converting PG dict to a deque class
    state1 = []
    next_state1 = []
    action1 = []
    reward1 = []
    done1 = []
    expert1 = []
    demo1 = []
    for i in range(len(fav3['observations'])):
        state1.append(fav3['observations'][i])
        action1.append(fav3['actions'][i])
        reward1.append(fav3['rewards'][i])
        next_state1.append(fav3['next_observations'][i])
        done1.append(fav3['done'][i])
        expert1.append(fav3['expert'][i])
    for i in range(len(state1)):
        demo1.append((state1[i], action1[i], reward1[i], next_state1[i], done1[i], expert1[i]))
    dq3.extend(demo1)

    # append all the experts data to demo_buffer of the agent 
    agent.demo_buffer.extend(dq1)
    agent.demo_buffer.extend(dq2)
    agent.demo_buffer.extend(dq3)

    # shuffle the demo_buffer before pre-training
    random.shuffle(agent.demo_buffer)

    # use the demo data to pre-train network
    agent.pre_train()  

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


# get expert demo data
# def get_demo_data(env):
#     env = wrappers.Monitor(env, '/tmp/CartPole-v1', force=True)
#     #agent.restore_model()
#     with tf.variable_scope('get_demo_data'):
#         agent = DQfDDDQN(env, DDQNConfig())
#     for e in range(Config.episode):
#         done = False
#         score = 0  # sum of reward in one episode
#         state = env.reset()
#         demo = []
#         while done is False:
#             action = agent.egreedy_action(state)  # e-greedy action for train
#             next_state, reward, done, _ = env.step(action)
#             score += reward
#             reward = reward if not done or score == 499 else -100
#             agent.perceive(state, action, reward, next_state, done, 0.0)  # 0. means it is not a demo data
#             demo.append((state, action, reward, next_state, done, 1.0))  # record the data that could be expert-data
#             agent.train_Q_network(pre_train=False, update=False)
#             state = next_state
#         if done:
#             if score > 275 and score < 325:  # expert demo data
#                 agent.demo_buffer.extend(demo)
#             agent.sess.run(agent.update_target_net)
#             print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
#                   "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
#             if len(agent.demo_buffer) >= Config.demo_buffer_size:
#                 agent.demo_buffer = agent.demo_buffer[:Config.demo_buffer_size]
#                 break
#     # write the demo data to a file
#     with open(Config.DEMO_DATA_PATH_1, 'wb') as f:
#         pickle.dump(agent.demo_buffer, f, protocol=2)



if __name__ == '__main__':
    env = gym.make(Config.ENV_NAME)
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # ------------------------ get demo scores by DDQN -----------------------------
    # get_demo_data(env)
    # --------------------------  get DDQN scores ----------------------------------
    # ddqn_sum_scores = np.zeros(Config.episode)
    # for i in range(Config.iteration):
    #     print("Running DDQN")
    #     scores = run_DDQN(i, env)
    #     ddqn_sum_scores = [a + b for a, b in zip(scores, ddqn_sum_scores)]
    # ddqn_mean_scores = [x/Config.iteration for x in ddqn_sum_scores]
    # with open('/Users/vikramanksingh/Desktop/DQfD/ddqn_mean_scores.p', 'wb') as f:
    #     pickle.dump(ddqn_mean_scores, f, protocol=2)
    # ----------------------------- get DQfD scores for PG expert --------------------------------
    dqfd_sum_scores = np.zeros(Config.episode)
    
    # dqfd_mean_scores_with_PG_expert_large_decay = []
    # dqfd_mean_scores_with_AC_expert_large_decay = []
    # dqfd_mean_scores_with_DDQN_expert_large_decay = []
    dqfd_mean_scores = []

    for i in range(Config.iteration):
        scores = run_DQfD(i, env)
        dqfd_sum_scores = [a + b for a, b in zip(scores, dqfd_sum_scores)]
    dqfd_mean_scores = [x/Config.iteration for x in dqfd_sum_scores]
    with open('/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores.p', 'wb') as f:
        pickle.dump(dqfd_mean_scores, f, protocol=2)


    # dqfd_mean_scores_with_all_expert = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores.p", "rb"))
    ddqn_mean_scores = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/ddqn_mean_scores.p", "rb"))


    map_scores(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores,
    xlabel='Red:dqfd_mean_scores_with_all_expert       Blue: ddqn_without_demo', ylabel='Scores')
    # env.close()
    # gym.upload('/tmp/carpole_DDQN-1', api_key='sk_VcAt0Hh4RBiG2yRePmeaLA')


