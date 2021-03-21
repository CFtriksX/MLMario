import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from Model import createModel
from wrappers import wrapper
from random import randint
import imageio

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper(env)

modelFilePath = './TrainedModel.HDF5'

model = createModel()

identity = np.identity(env.action_space.n) # for quickly get a hot vector, like 0001000000000000
totalReward = 0

for i in range(10000):
    lastState = env.reset()
    step = 0
    done = False
    while step < (10 * i):
        if randint(0, (int)(i / 100)) == 0:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.expand_dims(lastState, 0)))
        lastState, reward, done, info = env.step(action)
        totalReward += reward
        if reward > 0:
            model.train_on_batch(x=np.expand_dims(lastState, axis=0), y=identity[action: action+1])
        step += 1
        if done or info['life'] != 2 or info['flag_get']:
            break

    if i % 100 == 0:
        print("Medium reward per episode is :", totalReward / 100)
        video_filename = './video/Right/Epoch' + str(i) + '.mp4'
        with imageio.get_writer(video_filename, fps=60) as video:
            for _ in range(2):
                lastState = env.reset()
                video.append_data(env.render(mode='rgb_array'))
                while True:
                    action = np.argmax(model.predict(np.expand_dims(lastState, 0)))
                    state, _, done, info = env.step(action)
                    video.append_data(env.render(mode='rgb_array'))
                    if done or info['flag_get']:
                        break
        totalReward = 0


model.save(modelFilePath)
env.close()
