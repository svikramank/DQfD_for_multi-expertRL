import pickle
from collections import deque	
import random 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dq1 = deque()
dq2 = deque()
dq3 = deque()
dq4 = deque()

fav0 = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/ddqn_mean_scores.p", "rb" ))
# fav1 = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores_with_AC_expert.p", "rb" ))
# fav2 = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores_with_PG_expert.p", "rb" ))
# fav3 = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores_with_DDQN_expert.p", "rb" ))

# fav0_ld = 
fav1_ld = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores_with_AC_expert_large_decay.p", "rb" ))
fav2_ld = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores_with_PG_expert_large_decay.p", "rb" ))
#fav3_ld = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores_with_DDQN_expert_large_decay.p", "rb" ))

model = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores.p", "rb" ))
#model1 = pickle.load(open("/Users/vikramanksingh/Desktop/DQfD/dqfd_mean_scores_large_decay.p", "rb" ))

#dq1 = fav1

# print(type(fav1))
# print(type(fav2))
# # print(type(fav3))

# print(" ")
# print(" ")
# print(" ")


# # converting AC dict to a deque class
# state = []
# next_state = []
# action = []
# reward = []
# done = []
# expert = []
# demo = []
# for i in range(len(fav2['observations'])):
# 	state.append(fav2['observations'][i])
# 	action.append(fav2['actions'][i])
# 	reward.append(fav2['rewards'][i])
# 	next_state.append(fav2['next_observations'][i])
# 	done.append(fav2['done'][i])
# 	expert.append(fav2['expert'][i])
# for i in range(len(state)):
# 	demo.append((state[i], action[i], reward[i], next_state[i], done[i], expert[i]))
# dq2.extend(demo)



# # converting PG dict to a deque class
# state1 = []
# next_state1 = []
# action1 = []
# reward1 = []
# done1 = []
# expert1 = []
# demo1 = []
# for i in range(len(fav3['observations'])):
# 	state1.append(fav3['observations'][i])
# 	action1.append(fav3['actions'][i])
# 	reward1.append(fav3['rewards'][i])
# 	next_state1.append(fav3['next_observations'][i])
# 	done1.append(fav3['done'][i])
# 	expert1.append(fav3['expert'][i])
# for i in range(len(state1)):
# 	demo1.append((state1[i], action1[i], reward1[i], next_state1[i], done1[i], expert1[i]))
# dq3.extend(demo1)

# dq4.extend(dq1)
# dq4.extend(dq2)
# dq4.extend(dq3)


# random.shuffle(dq4)



# scores = []
# scores1 = []
# b = 0
# t = 0
# # batch the rewards for AC
# while b!= len(done):
# 	score = 0
# 	while done[b] is np.logical_not(True):
# 		score = score + reward[b]
# 		b = b+1
# 	if done[b]:
# 		scores.append(score)
# 		b = b+1

# # batch the rewards for PG
# while t!= len(done1):
# 	score1 = 0
# 	while done1[t] is np.logical_not(True):
# 		score1 = score1 + reward1[b]
# 		t = t+1
# 	if done1[t]:
# 		scores1.append(score1)
# 		t = t+1



# lst1 = list(dq1)
# lst4 = list(dq4)

# scores2 = []
# x = 0

# # batch the rewards for DDQN
# while x!= len(lst1):
# 	score2 = 0
# 	while lst1[x][4] is False:
# 		score2 = score2 + lst1[x][2]
# 		x = x+1
# 	if lst1[x][4]:
# 		scores2.append(score2)
# 		x = x+1



# print("scores for AC", scores)
# print("")
# print("")
# print("")
# print("scores for PG", scores1)
# print("")
# print("")
# print("")
# print("scores for DDQN", scores2)


# print(len(scores))
# print(len(scores1))
# print(len(scores2))
# print(len(model))

#print(type(fav0))


# taking running maximum over the data
fav0_new = np.maximum.accumulate(fav0)
# fav1_new = np.maximum.accumulate(fav1)
# fav2_new = np.maximum.accumulate(fav2)
#fav3_new = np.maximum.accumulate(fav3)
model_new = np.maximum.accumulate(model)
#model1_new = np.maximum.accumulate(model1)


fav1_ld_new = np.maximum.accumulate(fav1_ld)
fav2_ld_new = np.maximum.accumulate(fav2_ld)
#fav3_ld_new = np.maximum.accumulate(fav3_ld)



plt.plot(fav0_new, label= 'Vanilla DDQN')
plt.plot(fav1_ld_new, label='Trained only on Actor critic with high decay')
plt.plot(fav2_ld_new, label='Trained only on Policy Gradient with high decay')
#plt.plot(fav3_ld_new, label='Trained only on DDQN with high decay')
#plt.plot(model_new, label='Trained on all 3 experts_with_low_decay')
plt.plot(model_new, label= 'Trained on all 3 experts_with_high_decay')
plt.legend()
plt.show()
























































