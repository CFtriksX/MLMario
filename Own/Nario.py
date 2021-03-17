import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from Model import createModel
from wrappers import wrapper


env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper(env)

states = (84, 84, 4)
actions = env.action_space.n
