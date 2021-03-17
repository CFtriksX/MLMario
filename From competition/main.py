import time
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from agent import DQNAgent
from wrappers import wrapper
import imageio


# Build env (first level, right only)
env = gym_super_mario_bros.make('SuperMarioBros-1-2-v2')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = wrapper(env)
# Parameters
states = (84, 84, 4)
actions = env.action_space.n

# Agent
agent = DQNAgent(states=states, actions=actions, max_memory=100000, double_q=True)

# Episodes
episodes = 10000
rewards = []

# Timing
start = time.time()
step = 0

# Main loop
for e in range(episodes):

    # Reset env
    state = env.reset()

    # Reward
    total_reward = 0
    iter = 0

    # Play
    while True:

        # # Show env
        # if e % 100 == 0:
        #     env.render()

        # Run agent
        action = agent.run(state=state)

        # Perform action
        next_state, reward, done, info = env.step(action=action)

        # Remember
        agent.add(experience=(state, next_state, action, reward, done))

        # Replay
        agent.learn()

        # Total reward
        total_reward += reward

        # Update state
        state = next_state

        # Increment
        iter += 1

        # If done break loop
        if done or info['flag_get']:
            break

    # Rewards
    rewards.append(total_reward / iter)

    # Print
    if e % 100 == 0:
        print('Episode {e} - '
              'Frame {f} - '
              'Frames/sec {fs} - '
              'Epsilon {eps} - '
              'Mean Reward {r}'.format(e=e,
                                       f=agent.step,
                                       fs=np.round((agent.step - step) / (time.time() - start)),
                                       eps=np.round(agent.eps, 4),
                                       r=np.mean(rewards[-100:])))
        video_filename = './video/Complex/Epoch' + str(e) + '.mp4'
        with imageio.get_writer(video_filename, fps=60) as video:
            for _ in range(2):
                state = env.reset()
                video.append_data(env.render(mode='rgb_array'))
                while True:
                    action = agent.run(state=state)
                    state, _, done, info = env.step(action=action)
                    video.append_data(env.render(mode='rgb_array'))
                    if done or info['flag_get']:
                        break

        start = time.time()
        step = agent.step

# Save rewards
np.save('rewards.npy', rewards)
