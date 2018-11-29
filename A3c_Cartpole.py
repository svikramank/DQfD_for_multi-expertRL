import numpy as numpy
import tensorflow as tf 
import gym
from keras.layers import Dense, Input
from keras.models import Sequential, Model 
from keras.optimizers import Adam
from keras import backend as K 
import pylab 
import time
import threading 

class A2CAgent:
	def __init__(self, state_size, action_size):
		self.render = False
		self.load_model = False
		self.state_size = state_size
		self.action_size = action_size
		self.value_size = 1

		self.discount_factor = 0.99
		self.actor_lr = 0.001
		self.critic_lr = 0.005

		self.actor = self.build_actor()
		self.critic = self.build_critic()

		if self.load_model:
			self.actor.load_weights("./save_model/cartpole_actor.h5")
			self.critic.load_weights("./save_model/cartpole_critic.h5")

	def build_actor(self):
		actor = Sequential()
		actor.add(Dense(24, input_dim= self.state_size, activation= 'relu', kernel_initialization= 'he_uniform'))
		actor.add(Dense(self.action_size,activation='softmax', kernel_initialization= 'he_uniform'))
		actor.compile(loss= 'categorical_crossentropy', )